from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pickle
import random
import statistics

from audit.stagec.rarity_aware import (rarity_aware_select, qualifying_groups, load_nabs,
                                       default_quality_gate)
from audit.stagec.materialize_stagec_variants import read_jsonl

DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
MOVABLE_SKILLS = ["math", "instruction_following", "multilingual"]
SPEC = {"perplexity-low": ("ppl_work/nlls.pkl", "low"),
        "perplexity-high": ("ppl_work/nlls.pkl", "high"),
        "perplexity-mid": ("ppl_work/nlls.pkl", "mid"),
        "quality": ("quality_work/quality_scores.pkl", "high"),
        "ifd": ("ifd_work/ifd_scores.pkl", "high_finite"),
        "semdedup": ("semdedup_work/semdedup_scores.pkl", "high")}

WAVE1 = [
    ("quality", "language", "none", 0.10), ("quality", "language", "proportional", 0.10),
    ("quality", "language", "absolute", 0.10),
    ("quality", "language", "none", 0.02), ("quality", "language", "proportional", 0.02),
    ("quality", "language", "absolute", 0.02),
    ("quality", "skill", "proportional", 0.10), ("quality", "skill", "absolute", 0.10),
    ("perplexity-high", "skill", "none", 0.10), ("perplexity-high", "skill", "proportional", 0.10),
    ("perplexity-high", "skill", "absolute", 0.10),
    ("random", "skill", "proportional", 0.10),
]

WAVE2 = [
    ("semdedup", "skill", "none", 0.10),
    ("semdedup", "skill", "proportional", 0.10),
    ("semdedup", "skill", "absolute", 0.10),
]
SEEDS = [0, 1, 2]


def load_scores(selector, stage_a_dir, pool, seed, ppl_nlls):
    if selector == "random":
        rng = random.Random(seed)
        return {str(r["id"]): rng.random() for r in pool}, True
    rel, mode = SPEC[selector]
    path = ppl_nlls if selector.startswith("perplexity") else os.path.join(stage_a_dir, rel)
    raw = pickle.load(open(path, "rb"))
    idx2id = {r.get("pool_row_idx"): str(r["id"]) for r in pool}
    sc = {idx2id[k]: float(v) for k, v in raw.items() if k in idx2id}
    if mode == "low":
        return sc, False
    if mode == "high":
        return sc, True
    if mode == "high_finite":
        lo = min(v for v in sc.values() if math.isfinite(v))
        return {i: (v if math.isfinite(v) else lo - 1.0) for i, v in sc.items()}, True
    if mode == "mid":
        med = statistics.median([v for v in sc.values() if math.isfinite(v)])
        return {i: -abs(v - med) for i, v in sc.items()}, True
    raise ValueError(mode)


def cell_name(selector, axis, floor, budget):
    bp = f"{int(round(budget * 100)):02d}"
    return (f"{selector}__none__b{bp}" if floor == "none"
            else f"{selector}__{axis}__{floor}__b{bp}")


def materialize(stage_a_dir, pool, out_dir, cells, ppl_nlls):
    os.makedirs(out_dir, exist_ok=True)
    id2row = {str(r["id"]): r for r in pool}
    focus_of = {"language": DECISIVE, "skill": MOVABLE_SKILLS}
    report = {}
    for selector, axis, floor, budget in cells:
        n_abs = load_nabs("skill") if axis == "skill" else 500
        k = round(budget * len(pool))
        for seed in SEEDS:
            scores, hib = load_scores(selector, stage_a_dir, pool, seed, ppl_nlls)
            protected = None
            if floor in ("absolute", "hybrid"):
                qual, _ = qualifying_groups(pool, scores, hib, k, n_abs=n_abs,
                                            group_key=axis, quality_gate=True)
                protected = [g for g in qual if g in focus_of[axis]]
            r = rarity_aware_select(pool, scores, hib, k, floor_mode=floor, n_abs=n_abs,
                                    group_key=(axis if floor != "none" else "language"),
                                    quality_gate=True, protected_groups=protected)
            name = cell_name(selector, axis, floor, budget)
            with open(os.path.join(out_dir, f"{name}__s{seed}.jsonl"), "w", encoding="utf-8") as f:
                for i in r["selected_ids"]:
                    f.write(json.dumps(id2row[i], ensure_ascii=False) + "\n")
            if seed == 0:
                fc = collections.Counter(
                    (r_["skill_label"] if axis == "skill" else r_["language"])
                    for r_ in (id2row[i] for i in r["selected_ids"]))
                foc = {g: fc.get(g, 0) for g in focus_of[axis]}
                report[name] = {"total": r["total"], "overflow": r["overflow"],
                                "protected": protected, "focus_kept": foc}
                tag = f"OVERFLOW +{r['overflow']}" if r["overflow"] else ""
                print(f"  {name:44} total={r['total']:6} {tag:14} focus={foc}")
    json.dump(report, open(os.path.join(out_dir, "materialize_report.json"), "w"), indent=2)
    print(f"\nmaterialized {len(cells)} cells x {len(SEEDS)} seeds -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Materialize the Wave-1 necessity-map cells.")
    ap.add_argument("--pool", default="audit/results/stagec/phase1/pools/stagec_pool.jsonl")
    ap.add_argument("--stage_a", default="audit/results/stagec/phase2/stage_a")
    ap.add_argument("--ppl_nlls",
                    default="audit/results/stagec/phase1/stage_a_work/ppl_work/nlls.pkl",
                    help="perplexity nlls.pkl (scored in Phase-1 Wave 0; shared by low/high/mid).")
    ap.add_argument("--wave", type=int, default=1, choices=[1, 2],
                    help="1 = necessity map (perplexity/quality/random); 2 = SemDeDup skill(math).")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    pool = read_jsonl(args.pool)
    cells = WAVE1 if args.wave == 1 else WAVE2
    out_dir = args.out_dir or (f"audit/results/stagec/phase2/matrix/subsets_wave{args.wave}")
    materialize(args.stage_a, pool, out_dir, cells, args.ppl_nlls)
    if args.wave == 1:
        print("NOTE: perplexity-low necessity cells are REUSED (Phase-1 language@10%, arm A "
              "language@2%, arm C skill-IF@10%); random/full language refs reused from Phase 1.")
    else:
        print("NOTE: Wave 2 = SemDeDup skill(math) only; run_semdedup --dump_scores must have "
              "produced semdedup_work/semdedup_scores.pkl first (validate it before materializing).")


if __name__ == "__main__":
    main()
