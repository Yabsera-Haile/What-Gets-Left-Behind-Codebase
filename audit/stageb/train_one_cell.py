from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from audit.common import ensure_chat_template

GEMMA_LANG_TOWER_REGEX = (
    r"(?!.*vision).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")


def load_cfg(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fold_system_into_user(messages: list[dict]) -> list[dict]:
    if not messages or messages[0].get("role") != "system":
        return messages
    sys_text = (messages[0].get("content") or "").strip()
    rest = [dict(m) for m in messages[1:]]
    for m in rest:
        if m.get("role") == "user":
            m["content"] = (sys_text + "\n\n" + (m.get("content") or "")).strip()
            return rest
    return rest


def encode_for_sft(messages: list[dict], tok, max_len: int) -> dict | None:
    messages = fold_system_into_user(messages)
    input_ids: list[int] = []
    labels: list[int] = []
    for i, msg in enumerate(messages):
        prev = (tok.apply_chat_template(messages[:i], tokenize=True,
                                        add_generation_prompt=False) if i else [])
        cur = tok.apply_chat_template(messages[:i + 1], tokenize=True,
                                      add_generation_prompt=False)
        turn = cur[len(prev):]
        input_ids.extend(turn)
        labels.extend(turn if msg.get("role") == "assistant" else [-100] * len(turn))
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one Stage-B LoRA cell.")
    ap.add_argument("--train", required=True, help="Subset jsonl with `messages`.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="audit/configs/stageb_train.yaml")
    ap.add_argument("--model", default=None, help="Override model.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Training seed (init/data order). Vary it to measure run variance "
                         "on an identical subset.")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    model_name = args.model or cfg["model"]

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, DataCollatorForSeq2Seq)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ensure_chat_template(tok, model_name)
    if tok.chat_template is None:
        raise SystemExit(f"No chat template available for {model_name}; cannot format SFT data.")

    raw = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    encoded = [e for e in (encode_for_sft(r["messages"], tok, cfg["max_seq_len"])
                           for r in raw) if e is not None]
    ds = Dataset.from_list(encoded)
    print(f"[{os.path.basename(args.output_dir)}] examples: {len(raw)} -> {len(ds)} encoded")

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if os.getenv("HF_TOKEN"):
        load_kwargs["token"] = os.getenv("HF_TOKEN")
    if cfg.get("load_4bit"):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    gc = cfg.get("gradient_checkpointing", True)
    device_map = {"": 0} if cfg.get("load_4bit") else None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, **load_kwargs)
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_name, device_map=device_map, **load_kwargs)
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    if cfg.get("load_4bit"):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gc)
    elif gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    has_vision = any("vision" in n.lower() for n, _ in model.named_modules())
    target = GEMMA_LANG_TOWER_REGEX if has_vision else cfg["target_modules"]
    lora = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=target, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    if gc:
        model.enable_input_require_grads()

    adapted = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    n_vision = sum(1 for n in adapted if "vision" in n.lower())
    print(f"[{os.path.basename(args.output_dir)}] LoRA modules: {len(adapted)} "
          f"(language={len(adapted) - n_vision}, vision={n_vision}) | vision_tower={has_vision} "
          f"| grad_checkpointing={gc}")
    if n_vision:
        raise SystemExit(f"LoRA attached to {n_vision} vision-tower modules -- aborting.")
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg.get("weight_decay", 0.0),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=False,
        optim=cfg.get("optim", "adamw_torch"),
        seed=args.seed, data_seed=args.seed,
        logging_steps=5, save_strategy="no", report_to=[],
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    trainer.train()
    runtime = round(time.time() - t0, 1)
    peak_vram = round(torch.cuda.max_memory_allocated(0) / 1024 ** 2)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    metrics = {
        "condition": os.path.basename(args.output_dir),
        "model": model_name, "device": "cuda",
        "n_examples": len(ds), "steps": trainer.state.global_step,
        "epochs": cfg["num_train_epochs"], "effective_batch": cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
        "runtime_s": runtime, "peak_vram_mib": peak_vram, "seed": args.seed,
        "load_4bit": bool(cfg.get("load_4bit")),
    }
    print("STAGEB_METRICS " + json.dumps(metrics))


if __name__ == "__main__":
    main()
