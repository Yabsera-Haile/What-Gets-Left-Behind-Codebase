from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd

MODEL_PARAMS = {
    "qwen2.5-7b": 7.6e9, "qwen2.5-1.5b": 1.5e9, "qwen2.5-0.5b": 0.5e9,
    "llama-2-7b": 6.7e9, "llama-7b": 6.7e9, "pythia-1.4b": 1.4e9,
    "pythia-160m": 1.6e8, "all-minilm-l6-v2": 2.2e7,
}
KIND_PASSES = {
    "perplexity": 1, "ifd": 2, "llm_quality": 1, "rds+_multitask": 1,
    "semdedup": 0, "random": 0,
}


def params_for(model: str):
    if not model:
        return None
    m = model.lower()
    for key, p in MODEL_PARAMS.items():
        if key in m:
            return p
    mm = re.search(r"(\d+\.?\d*)\s*b\b", m)
    if mm:
        return float(mm.group(1)) * 1e9
    mm = re.search(r"(\d+)\s*m\b", m)
    if mm:
        return float(mm.group(1)) * 1e6
    return None


def load_metas(selections_dir: str) -> dict:
    meta = {}
    for path in sorted(glob.glob(os.path.join(selections_dir, "*.json"))):
        try:
            obj = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        meta.setdefault(obj.get("selector", "?"), obj.get("meta", {}) or {})
    return meta


def total_pool_tokens(metadata: str) -> tuple[float, bool]:
    df = pd.read_parquet(metadata)
    approx = bool(df["token_len_approx"].any()) if "token_len_approx" in df else True
    total = float(df["token_len"].sum())
    return (total * 1.3 if approx else total), approx


def _fmt_flops(x):
    if x is None or x == 0:
        return "-"
    for unit, scale in (("E", 1e18), ("P", 1e15), ("T", 1e12), ("G", 1e9)):
        if x >= scale:
            return f"{x/scale:.2f} {unit}FLOPs"
    return f"{x:.0f}"


def build_table(metas: dict, tokens: float) -> tuple[str, dict]:
    lines = ["| Selector | Model | Params | Passes | ~Tokens | ~FLOPs | Runtime (s) | "
             "Peak VRAM (MiB) |", "|---|---|---|---|---|---|---|---|"]
    totals = {"flops": 0.0, "runtime": 0.0}
    for sel in sorted(metas):
        m = metas[sel]
        model = m.get("model", "-")
        kind = m.get("selector_kind", "")
        passes = KIND_PASSES.get(kind, 1 if model not in ("-", "none(random)") else 0)
        params = params_for(model)
        sel_tokens = tokens * passes if passes else 0
        flops = 2 * params * sel_tokens if (params and sel_tokens) else 0
        rt = m.get("runtime_s"); vr = m.get("vram_peak_mib")
        totals["flops"] += flops or 0
        totals["runtime"] += rt or 0
        lines.append(
            f"| {sel} | {model} | {('%.1fB' % (params/1e9)) if params else '-'} | {passes} | "
            f"{('%.2e' % sel_tokens) if sel_tokens else '-'} | {_fmt_flops(flops)} | "
            f"{'-' if rt is None else f'{rt:.0f}'} | {'-' if vr is None else vr} |")
    lines.append(f"| **TOTAL** | | | | | {_fmt_flops(totals['flops'])} | "
                 f"{totals['runtime']:.0f} | |")
    return "\n".join(lines), totals


def main() -> None:
    ap = argparse.ArgumentParser(description="Budget-aware compute accounting.")
    ap.add_argument("--selections_dir", default="audit/results/selections")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    metas = load_metas(args.selections_dir)
    tokens, approx = total_pool_tokens(args.metadata)
    table, totals = build_table(metas, tokens)
    header = (f"# Selector compute accounting\n\n"
             f"Pool forward-pass tokens ~= {tokens:.3e} "
             f"({'whitespace-word proxy x1.3' if approx else 'tokenizer'}). "
             f"FLOPs ~= 2 * params * tokens * passes (forward-only estimate).\n\n")
    md = header + table + (f"\n\nTotal estimated FLOPs: {_fmt_flops(totals['flops'])}; "
                          f"total measured runtime: {totals['runtime']:.0f}s.\n")
    print(md)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
