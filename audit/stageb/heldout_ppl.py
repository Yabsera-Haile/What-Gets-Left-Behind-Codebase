from __future__ import annotations

import argparse
import json
import math
import os

from audit.common import ensure_chat_template

ISO_TO_FLORES = {
    "eng": "eng_Latn", "ceb": "ceb_Latn", "plt": "plt_Latn", "bod": "bod_Tibt",
    "yor": "yor_Latn", "tsn": "tsn_Latn", "som": "som_Latn", "kir": "kir_Cyrl",
    "fuv": "fuv_Latn", "hau": "hau_Latn", "wol": "wol_Latn", "mlt": "mlt_Latn",
    "zul": "zul_Latn", "spa": "spa_Latn", "cmn": "zho_Hans", "arb": "arb_Arab",
}


def build_labels(tok, prompt: str, text: str, max_len: int):
    prefix_text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          tokenize=False, add_generation_prompt=True)
    prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
    full_ids = tok(prefix_text + text, add_special_tokens=False).input_ids
    start = 0
    while (start < len(prefix_ids) and start < len(full_ids)
           and prefix_ids[start] == full_ids[start]):
        start += 1
    labels = [-100] * start + full_ids[start:]
    return full_ids[:max_len], labels[:max_len]


def score_sentences(model, tok, sentences, prompt="", max_len=1024, device=None):
    import torch
    device = device or (next(model.parameters()).device)
    model.eval()
    total_nll, total_tok, n_scored = 0.0, 0, 0
    for text in sentences:
        if not text or not text.strip():
            continue
        ids, labels = build_labels(tok, prompt, text, max_len)
        if all(l == -100 for l in labels):
            continue
        input_ids = torch.tensor([ids], device=device)
        lab = torch.tensor([labels], device=device)
        with torch.no_grad():
            logits = model(input_ids).logits.float()
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = lab[:, 1:].reshape(-1)
        nll = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, ignore_index=-100, reduction="sum")
        n = int((shift_labels != -100).sum().item())
        total_nll += float(nll.item())
        total_tok += n
        n_scored += 1
    mean_nll = (total_nll / total_tok) if total_tok else float("nan")
    return {"nll": mean_nll, "ppl": (math.exp(mean_nll) if total_tok else float("nan")),
            "n_tokens": total_tok, "n_sentences": n_scored}


def load_flores(flores_code: str, split="devtest", limit=None,
                dataset="Muennighoff/flores200"):
    from datasets import load_dataset
    ds = load_dataset(dataset, flores_code, split=split, trust_remote_code=True)
    field = "sentence" if "sentence" in ds.column_names else ds.column_names[-1]
    sents = [s for s in ds[field] if s and str(s).strip()]
    return sents[:limit] if limit else sents


def load_model_for_scoring(base_model: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_src = adapter or base_model
    tok = AutoTokenizer.from_pretrained(tok_src)
    ensure_chat_template(tok, base_model)
    kw = {"torch_dtype": torch.bfloat16, "attn_implementation": "eager", "device_map": {"": 0}}
    if os.getenv("HF_TOKEN"):
        kw["token"] = os.getenv("HF_TOKEN")
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **kw)
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(base_model, **kw)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok


def selftest(model_id="sshleifer/tiny-gpt2") -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from audit.common import GEMMA_CHAT_TEMPLATE
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = GEMMA_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu").eval()

    toy = ["Hello world, this is a test.", "The quick brown fox jumps.", "Language modeling."]
    prompt = ""

    ids, labels = build_labels(tok, prompt, toy[0], 1024)
    content = [i for i, l in zip(ids, labels) if l != -100]
    masked = sum(1 for l in labels if l == -100)
    decoded = tok.decode(content)
    print(f"[selftest] seq_len={len(ids)} masked(prompt/template)={masked} content_tok={len(content)}")
    print(f"[selftest] content decodes to: {decoded!r}")
    assert masked > 0, "no prompt/template tokens were masked"
    assert len(content) > 0, "no content tokens"
    assert toy[0].strip()[:8] in decoded, "content tokens do not match the sentence text"

    res = score_sentences(model, tok, toy, prompt=prompt, device="cpu")
    print(f"[selftest] result: {res}")
    assert math.isfinite(res["nll"]) and math.isfinite(res["ppl"]), "non-finite NLL/ppl"
    assert res["ppl"] > 0 and res["n_tokens"] > 0 and res["n_sentences"] == 3

    ncontent = sum(len([i for i, l in zip(*build_labels(tok, prompt, t, 1024)) if l != -100])
                   for t in toy)
    nfull = sum(len(tok(tok.apply_chat_template([{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True) + t,
                        add_special_tokens=False).input_ids) for t in toy)
    print(f"[selftest] content tokens={ncontent} < full(prefix+content) tokens={nfull}")
    assert ncontent == res["n_tokens"] and ncontent < nfull, "masking/token accounting off"

    assert set(res) == {"nll", "ppl", "n_tokens", "n_sentences"}
    print("[selftest] PASS — finite ppl, prompt/template excluded, schema OK.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out per-language perplexity (FLORES-200).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest_model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--base", default="google/gemma-3-4b-pt")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (None = base).")
    ap.add_argument("--condition", default=None, help="Output tag (default: adapter basename or 'base').")
    ap.add_argument("--langs", nargs="*", default=["hau"], help="Target ISO codes.")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--limit", type=int, default=None, help="Sentences/language (None = all ~1012).")
    ap.add_argument("--prompt", default="", help="Fixed neutral user prompt (held constant).")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--flores_dataset", default="Muennighoff/flores200",
                    help="Non-gated FLORES-200 mirror; per-language configs (e.g. plt_Latn).")
    ap.add_argument("--out_dir", default="audit/results/stageb/gemma-3-4b-pt/round4/metrics/heldout_ppl")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.selftest_model)
        return

    model, tok = load_model_for_scoring(args.base, args.adapter)
    cond = args.condition or (os.path.basename(args.adapter) if args.adapter else "base")
    out = {"condition": cond, "base_model": args.base, "adapter": args.adapter,
           "prompt": args.prompt, "split": args.split, "languages": {}}
    for iso in args.langs:
        fcode = ISO_TO_FLORES.get(iso, iso)
        sents = load_flores(fcode, args.split, args.limit, args.flores_dataset)
        res = score_sentences(model, tok, sents, args.prompt, args.max_len)
        out["languages"][iso] = res
        print(f"[{cond}] {iso} ({fcode}): ppl={res['ppl']:.3f} nll={res['nll']:.4f} "
              f"n_tok={res['n_tokens']} n_sent={res['n_sentences']}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{cond}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
