from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import re
import statistics

import pandas as pd

from audit.common import (QUALITY_DEV_MODEL, QUALITY_REAL_MODEL, VramSampler,
                          announce_dev, results_base, run_meta)

logger = logging.getLogger("audit.run_quality")

BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50]
SCORE_LO, SCORE_HI = 1, 5
PROMPT_PATH = "audit/configs/quality_judge_prompt.txt"
TEXT_CLIP = 1500


def _first_text(messages, role):
    for m in messages or []:
        if m.get("role") == role:
            return (m.get("content") or "").strip()
    return ""


def parse_score(text: str):
    for tok in re.findall(r"\d+", text):
        v = int(tok)
        if SCORE_LO <= v <= SCORE_HI:
            return float(v)
    return None


def score_quality(pool, work_dir, model_name, prompt_template, dtype, batch_size,
                  max_new_tokens, force, limit):
    os.makedirs(work_dir, exist_ok=True)
    out = os.path.join(work_dir, "quality_scores.pkl")
    partial = out + ".partial"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sampler = VramSampler()
    if os.path.exists(out) and not force:
        logger.info("Reusing existing quality scores: %s", out)
        with open(out, "rb") as f:
            return out, sampler, pickle.load(f)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    kwargs = {"torch_dtype": dtypes.get(dtype, torch.bfloat16)}
    if os.getenv("HF_TOKEN"):
        kwargs["token"] = os.getenv("HF_TOKEN")
    logger.info("Loading quality judge %s (%s) ...", model_name, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.generation_config.do_sample = False
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)
    device = next(model.parameters()).device

    examples = [json.loads(l) for l in open(pool, encoding="utf-8") if l.strip()]
    if limit:
        examples = examples[:limit]

    def build_prompt(ex):
        msgs = ex.get("messages", [])
        instr = _first_text(msgs, "user")[:TEXT_CLIP]
        resp = _first_text(msgs, "assistant")[:TEXT_CLIP]
        user = prompt_template.format(instruction=instr, response=resp)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

    def _generate(batch):
        prompts = [build_prompt(ex) for ex in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=4096).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        cont = gen[:, enc["input_ids"].shape[1]:]
        return tokenizer.batch_decode(cont, skip_special_tokens=True)

    raw_scores: dict[int, float | None] = {}
    if os.path.exists(partial) and not force:
        with open(partial, "rb") as f:
            raw_scores = {int(k): v for k, v in pickle.load(f).items()}
        logger.info("Resuming quality: %d / %d already scored", len(raw_scores), len(examples))
    failures = 0
    with sampler:
        for bi, start in enumerate(range(0, len(examples), batch_size)):
            end = min(start + batch_size, len(examples))
            if all(i in raw_scores for i in range(start, end)):
                continue
            batch = examples[start:end]
            try:
                texts = _generate(batch)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("OOM at batch %d (rows %d-%d); retrying one-by-one", bi, start, end)
                texts = []
                for ex in batch:
                    texts.extend(_generate([ex]))
                    torch.cuda.empty_cache()
            for j, txt in enumerate(texts):
                s = parse_score(txt)
                if s is None:
                    failures += 1
                raw_scores[start + j] = s
            if bi % 40 == 0:
                with open(partial, "wb") as f:
                    pickle.dump(raw_scores, f)
                torch.cuda.empty_cache()
                logger.info("  ...quality scored %d / %d (checkpointed)",
                            len(raw_scores), len(examples))

    valid = [s for s in raw_scores.values() if s is not None]
    median = statistics.median(valid) if valid else (SCORE_LO + SCORE_HI) / 2
    scores = {i: (s if s is not None else median) for i, s in raw_scores.items()}
    logger.info("Quality scoring done in %ss, peak VRAM %s MiB. Parse failures: %d/%d "
                "(assigned median=%.1f).", sampler.runtime_s, sampler.peak_mib,
                failures, len(scores), median)
    with open(out, "wb") as f:
        pickle.dump(scores, f)
    if os.path.exists(partial):
        os.remove(partial)
    return out, sampler, scores


def print_bucket_means(scores: dict, meta_df: pd.DataFrame) -> None:
    sc = pd.DataFrame({"pool_row_idx": list(scores.keys()), "score": list(scores.values())})
    merged = sc.merge(meta_df[["pool_row_idx", "resource_bucket"]], on="pool_row_idx", how="left")
    logger.info("Mean quality score per resource_bucket (0=low ... 5=high):")
    for b in sorted(merged["resource_bucket"].dropna().unique()):
        sub = merged[merged["resource_bucket"] == b]["score"]
        logger.info("  bucket %d: mean=%.3f  n=%d", int(b), sub.mean(), len(sub))
    logger.info("Overall mean=%.3f", merged["score"].mean())


def generate(pool, metadata, out_dir, work_dir, model, dev, dtype="bf16", batch_size=16,
             max_new_tokens=8, budgets=BUDGETS, force_score=False, limit=None):
    meta_df = pd.read_parquet(metadata).sort_values("pool_row_idx").reset_index(drop=True)
    n_pool = len(meta_df)
    idx2id = {int(r): str(i) for r, i in zip(meta_df["pool_row_idx"], meta_df["id"])}
    os.makedirs(out_dir, exist_ok=True)

    with open(PROMPT_PATH, encoding="utf-8") as f:
        prompt_template = f.read()

    _, sampler, scores = score_quality(pool, work_dir, model, prompt_template, dtype,
                                       batch_size, max_new_tokens, force_score, limit)
    print_bucket_means(scores, meta_df)

    base_meta = run_meta(model, dev, sampler, extra={
        "selector_kind": "llm_quality", "judge_prompt": PROMPT_PATH, "scale": "1-5",
        "dtype": dtype, "n_scored": len(scores)})

    ranked = sorted(scores.keys(), key=lambda i: -scores[i])
    written = []
    for budget in budgets:
        k = round(budget * n_pool)
        chosen = ranked[:k]
        ids = [idx2id[int(i)] for i in chosen if int(i) in idx2id]
        obj = {"selector": "quality", "budget": budget, "seed": 0,
               "selected_ids": ids, "meta": base_meta}
        path = os.path.join(out_dir, f"quality__b{budget}__s0.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        written.append(path)
        logger.info("quality b=%.2f -> %d ids -> %s", budget, len(ids), os.path.basename(path))
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="LLM-quality filter selector.")
    ap.add_argument("--pool", default="audit/results/pilot_pool.jsonl")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--model", default=None, help="Override the judge model.")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--force_score", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Score only first N (testing).")
    args = ap.parse_args()

    announce_dev(args.dev, logger)
    base = results_base(args.dev)
    out_dir = args.out_dir or os.path.join(base, "selections")
    work_dir = args.work_dir or os.path.join(base, "quality_work")
    model = args.model or (QUALITY_DEV_MODEL if args.dev else QUALITY_REAL_MODEL)

    paths = generate(args.pool, args.metadata, out_dir, work_dir, model, args.dev,
                     dtype=args.dtype, batch_size=args.batch_size,
                     max_new_tokens=args.max_new_tokens, force_score=args.force_score,
                     limit=args.limit)
    logger.info("Wrote %d quality selection files to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()
