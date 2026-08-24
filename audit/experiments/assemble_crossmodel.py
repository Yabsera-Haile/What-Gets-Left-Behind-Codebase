from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics

DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
BASES = [("aya-expanse-8b", "Aya-Expanse-8B"), ("llama-3.1-8b", "Llama-3.1-8B")]
GEMMA = {
    "lang": {("perplexity-low", "none"): 36.30, ("perplexity-low", "proportional"): 29.21,
             ("perplexity-low", "absolute"): 27.09, ("quality", "none"): 35.52,
             ("quality", "proportional"): 31.14, ("quality", "absolute"): 27.04, "baseline": 29.1},
    "ifeval": {("perplexity-low", "none"): 0.158, ("perplexity-low", "proportional"): 0.218,
               ("perplexity-low", "absolute"): 0.357, ("perplexity-high", "none"): 0.113,
               ("perplexity-high", "proportional"): 0.303, ("perplexity-high", "absolute"): 0.368,
               ("quality", "none"): 0.302, ("quality", "proportional"): 0.245,
               ("quality", "absolute"): 0.367},
    "gsm8k": {("perplexity-high", "none"): 0.268, ("perplexity-high", "absolute"): 0.407},
}


def parse_cell(cell):
    if cell == "base":
        return ("base", "none", "none", 0)
    m = re.match(r"^(.+?)__(none|lang|ifeval|math)(?:__(proportional|absolute))?__s(\d+)$", cell)
    if not m:
        return None
    sel, purpose, floor, seed = m.groups()
    return (sel, purpose, floor or "none", int(seed))


def ppl_macro(path):
    langs = json.load(open(path, encoding="utf-8")).get("languages", {})
    v = [langs[l]["ppl"] for l in DECISIVE if langs.get(l, {}).get("ppl") is not None]
    return sum(v) / len(v) if v else None


def collect(root):
    rows = []
    for slug, name in BASES:
        for p in glob.glob(f"{root}/{slug}/eval/metrics/heldout_ppl/*.json"):
            pc = parse_cell(os.path.splitext(os.path.basename(p))[0])
            if pc is None:
                continue
            m = ppl_macro(p)
            if m is not None:
                rows.append([name, *pc, "decisive_ppl", m])
        sp = f"{root}/{slug}/eval_skill/skill_cells_results.parquet"
        if os.path.exists(sp):
            import pandas as pd
            for _, r in pd.read_parquet(sp).iterrows():
                pc = parse_cell(r["cell"])
                if pc is not None and r["benchmark"] in ("ifeval", "gsm8k"):
                    rows.append([name, *pc, r["benchmark"], float(r["value"])])
    return rows


def agg(df, base, selector, purpose, floor, metric):
    s = df[(df.base == base) & (df.selector == selector) & (df.purpose == purpose)
           & (df.floor == floor) & (df.metric == metric)]["value"]
    if s.empty:
        return None, None
    return round(s.mean(), 3), (round(statistics.pstdev(list(s)), 3) if len(s) > 1 else 0.0)


def cell(v, sd):
    return "  --  " if v is None else (f"{v:.2f}±{sd:.2f}" if v >= 1 else f"{v:.3f}±{sd:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Assemble the cross-model replication.")
    ap.add_argument("--root", default="audit/results/crossmodel")
    ap.add_argument("--out", default="audit/results/crossmodel/results.parquet")
    args = ap.parse_args()
    import pandas as pd
    df = pd.DataFrame(collect(args.root),
                      columns=["base", "selector", "purpose", "floor", "seed", "metric", "value"])
    if df.empty:
        raise SystemExit("no eval outputs found under " + args.root)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Assembled {len(df)} rows -> {args.out}\n")
    names = [n for _, n in BASES]

    print("=" * 92)
    print("LANGUAGE NECESSITY — decisive macro perplexity (lower=better). "
          "Q: does quality FAIL fair-share where ppl-low CURES?")
    print("=" * 92)
    hdr = f"{'selector':16} {'floor':13}" + "".join(f"{n:>18}" for n in names) + f"{'Gemma-4B':>12}"
    print(hdr)
    for selector in ("perplexity-low", "quality"):
        for floor, purpose in (("none", "none"), ("proportional", "lang"), ("absolute", "lang")):
            line = f"{selector:16} {floor:13}"
            for n in names:
                line += f"{cell(*agg(df, n, selector, purpose, floor, 'decisive_ppl')):>18}"
            g = GEMMA["lang"].get((selector, floor))
            line += f"{('%.2f' % g if g else '--'):>12}"
            print(line)
    print(f"  (baselines: Gemma random/full ~29.1; per-base 'random' ref not trained here — "
          f"the test is proportional vs absolute WITHIN each selector.)")

    print("\n" + "=" * 92)
    print("IFEval NECESSITY (Llama) — prompt-strict (higher=better). "
          "Q: is the absolute floor (500 IF) required vs fair-share (96)?")
    print("=" * 92)
    print(f"{'selector':16} {'floor':13}{'Llama-3.1-8B':>18}{'Gemma-4B':>12}")
    for selector in ("perplexity-low", "perplexity-high", "quality"):
        for floor, purpose in (("none", "none"), ("proportional", "ifeval"), ("absolute", "ifeval")):
            v, sd = agg(df, "Llama-3.1-8B", selector, purpose, floor, "ifeval")
            g = GEMMA["ifeval"].get((selector, floor))
            print(f"{selector:16} {floor:13}{cell(v, sd):>18}{('%.3f' % g if g else '--'):>12}")

    print("\n" + "=" * 92)
    print("GSM8K MATH-FLIP anchor (Llama) — strict (higher=better). "
          "Q: does ppl-high erode math below base, floor recover?")
    print("=" * 92)
    print(f"{'selector':16} {'floor':13}{'Llama-3.1-8B':>18}{'Gemma-4B':>12}")
    for floor, purpose in (("none", "none"), ("absolute", "math")):
        v, sd = agg(df, "Llama-3.1-8B", "perplexity-high", purpose, floor, "gsm8k")
        g = GEMMA["gsm8k"].get(("perplexity-high", floor))
        print(f"{'perplexity-high':16} {floor:13}{cell(v, sd):>18}{('%.3f' % g if g else '--'):>12}")

    print("\nBASE anchors (decisive macro ppl):",
          {n: agg(df, n, "base", "none", "none", "decisive_ppl")[0] for n in names})


if __name__ == "__main__":
    main()
