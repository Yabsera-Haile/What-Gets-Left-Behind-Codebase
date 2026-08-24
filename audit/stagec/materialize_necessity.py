from __future__ import annotations

import argparse
import collections
import json
import os

from audit.stagec.rarity_aware import rarity_aware_select, load_nabs
from audit.stagec.materialize_stagec_variants import read_jsonl, load_scores

DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
ISOLATE = "kir"
SKILL_TARGET = "instruction_following"
MODES = ["none", "proportional", "absolute"]
SEEDS = [0, 1, 2]

ARMS = {
    "armA_langtight":    dict(group="language", protected=DECISIVE,      budget=0.02, pool="full"),
    "armB_kir_isolated": dict(group="language", protected=[ISOLATE],     budget=0.10, pool="isolate_kir"),
    "armC_if_skill":     dict(group="skill",    protected=[SKILL_TARGET], budget=0.10, pool="full"),
}


def filter_pool(pool, how):
    if how == "isolate_kir":
        drop = set(DECISIVE) - {ISOLATE}
        return [r for r in pool if r.get("language") not in drop]
    return pool


def counts_in(sel_ids, id2grp, groups):
    c = collections.Counter(id2grp.get(str(i)) for i in sel_ids)
    return {g: c.get(g, 0) for g in groups}


def materialize_arm(name, cfg, pool_full, scores, out_root):
    group = cfg["group"]
    pool = filter_pool(pool_full, cfg["pool"])
    n_abs = load_nabs("skill") if group == "skill" else 500
    id2grp = {str(r["id"]): (r.get("skill_label") if group == "skill" else r.get("language"))
              for r in pool}
    id2row = {str(r["id"]): r for r in pool}
    k = round(cfg["budget"] * len(pool))
    focus = cfg["protected"] if group == "language" else [SKILL_TARGET]
    out_dir = os.path.join(out_root, name, "subsets")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {name} | axis={group} pool={len(pool)} budget={cfg['budget']:.0%} k={k} "
          f"protect={cfg['protected'] if len(cfg['protected'])<=3 else str(len(cfg['protected']))+' langs'} ===")
    report = {}
    for mode in MODES:
        r = rarity_aware_select(pool, scores, False, k, floor_mode=mode, n_abs=n_abs,
                                group_key=group, quality_gate=True,
                                protected_groups=(cfg["protected"] if mode in ("absolute",) else None))
        fc = counts_in(r["selected_ids"], id2grp, focus)
        report[mode] = {"total": r["total"], "budget": k, "overflow": r["overflow"],
                        "focus_kept": fc}
        for s in SEEDS:
            with open(os.path.join(out_dir, f"perplexity-low__{mode}__s{s}.jsonl"),
                      "w", encoding="utf-8") as f:
                for i in r["selected_ids"]:
                    f.write(json.dumps(id2row[i], ensure_ascii=False) + "\n")
        tag = ("OVERFLOW +%d" % r["overflow"]) if r["overflow"] else ""
        print(f"  {mode:13} total={r['total']:6} (budget {k}) {tag:14} focus_kept={fc}")
    json.dump(report, open(os.path.join(out_root, name, "materialize_report.json"), "w"), indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description="Materialize the C2-Step 9 necessity arms.")
    ap.add_argument("--pool", default="audit/results/stagec/phase1/pools/stagec_pool.jsonl")
    ap.add_argument("--nlls", required=True, help="Phase-1 perplexity-low NLL pickle (pool_row_idx).")
    ap.add_argument("--out_root", default="audit/results/stagec/phase2/necessity")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), help="Subset of arms to build.")
    args = ap.parse_args()
    pool = read_jsonl(args.pool)
    scores = load_scores(pool, args.nlls)
    for name in args.arms:
        materialize_arm(name, ARMS[name], pool, scores, args.out_root)
    print("\nNOTE: armA 'absolute' exceeds the 2% budget by design (500/lang guarantee); its "
          "effective size is the necessity claim's cost. armB isolates kir (6 cross-helpers "
          "removed). armC tests IF fair-share (~96) vs floor (500) on the movable IFEval.")


if __name__ == "__main__":
    main()
