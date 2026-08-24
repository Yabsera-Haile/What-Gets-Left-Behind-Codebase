from __future__ import annotations

import argparse
import collections
import json
import os

from audit.stagec.rarity_aware import rarity_aware_select, qualifying_groups, load_nabs
from audit.stagec.materialize_stagec_variants import read_jsonl
from audit.stagec.materialize_matrix import load_scores

SEEDS = [0, 1, 2]
BUDGET = 0.10
PURPOSE = {"code": ("skill", ["code"]), "safety": ("skill", ["safety"])}
CODE_SELECTORS = ["semdedup", "perplexity-low", "perplexity-high"]
SAFETY_CANDIDATES = ["quality", "perplexity-low", "perplexity-high"]
FLOORS = ["proportional", "absolute"]
DELETE_THRESHOLD = 0.80


def counts(sel_ids, id2grp, groups):
    c = collections.Counter(id2grp.get(str(i)) for i in sel_ids)
    return {g: c.get(g, 0) for g in groups}


def materialize(stage_a, pool, ppl_nlls, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    id2row = {str(r["id"]): r for r in pool}
    id2skill = {str(r["id"]): r.get("skill_label") for r in pool}
    n_skill = collections.Counter(v for v in id2skill.values())
    nabs = load_nabs("skill")
    k = round(BUDGET * len(pool))
    prop = {s: BUDGET * n_skill[s] for s in ("code", "safety")}
    report = {}

    def write(name, ids):
        for s in SEEDS:
            with open(os.path.join(out_dir, f"{name}__s{s}.jsonl"), "w", encoding="utf-8") as f:
                for i in ids:
                    f.write(json.dumps(id2row[i], ensure_ascii=False) + "\n")

    print(f"pool={len(pool)} k={k}  proportional share @10%: "
          f"code={prop['code']:.0f} safety={prop['safety']:.0f}\n")
    print(f"{'selector':16} {'code_kept':>9} {'code/prop':>9} {'safety_kept':>11} {'safety/prop':>11}")
    none_counts = {}
    for sel in sorted(set(CODE_SELECTORS + SAFETY_CANDIDATES)):
        scores, hib = load_scores(sel, stage_a, pool, 0, ppl_nlls)
        r = rarity_aware_select(pool, scores, hib, k, floor_mode="none",
                                group_key="language", quality_gate=True)
        c = counts(r["selected_ids"], id2skill, ["code", "safety"])
        none_counts[sel] = {"ids": r["selected_ids"], "code": c["code"], "safety": c["safety"],
                            "total": r["total"]}
        print(f"{sel:16} {c['code']:>9} {c['code']/prop['code']:>9.2f} "
              f"{c['safety']:>11} {c['safety']/prop['safety']:>11.2f}")

    code_decisive = min(CODE_SELECTORS, key=lambda s: none_counts[s]["code"])
    safety_selectors = [s for s in SAFETY_CANDIDATES
                        if none_counts[s]["safety"] < DELETE_THRESHOLD * prop["safety"]]
    if not safety_selectors:
        safety_selectors = [min(SAFETY_CANDIDATES, key=lambda s: none_counts[s]["safety"])]
    safety_decisive = min(safety_selectors, key=lambda s: none_counts[s]["safety"])
    print(f"\nCODE decisive (deletes most): {code_decisive}")
    print(f"SAFETY selectors (delete safety, <{DELETE_THRESHOLD:.0%} of fair share): {safety_selectors}"
          f"  | decisive: {safety_decisive}")

    used = sorted(set(CODE_SELECTORS + safety_selectors))
    for sel in used:
        write(f"{sel}__none", none_counts[sel]["ids"])
        report[f"{sel}__none"] = {"total": none_counts[sel]["total"],
                                  "code": none_counts[sel]["code"],
                                  "safety": none_counts[sel]["safety"]}

    plan = [(s, "code") for s in CODE_SELECTORS] + [(s, "safety") for s in safety_selectors]
    print(f"\n{'cell':42} {'total':>6}  focus_kept")
    for sel, purpose in plan:
        axis, focus = PURPOSE[purpose]
        for floor in FLOORS:
            scores, hib = load_scores(sel, stage_a, pool, 0, ppl_nlls)
            qual, _ = qualifying_groups(pool, scores, hib, k, n_abs=nabs,
                                        group_key=axis, quality_gate=True)
            protected = [g for g in qual if g in focus]
            r = rarity_aware_select(pool, scores, hib, k, floor_mode=floor, n_abs=nabs,
                                    group_key=axis, quality_gate=True, protected_groups=protected)
            name = f"{sel}__{purpose}__{floor}"
            write(name, r["selected_ids"])
            fc = counts(r["selected_ids"], id2skill, focus)
            report[name] = {"total": r["total"], "overflow": r["overflow"],
                            "protected": protected, "focus_kept": fc}
            print(f"{name:42} {r['total']:>6}  {fc}"
                  + (f"  OVERFLOW+{r['overflow']}" if r["overflow"] else ""))

    json.dump({"code_decisive": code_decisive, "safety_selectors": safety_selectors,
               "safety_decisive": safety_decisive, "proportional_share": prop,
               "delete_threshold": DELETE_THRESHOLD, "cells": report},
              open(os.path.join(out_dir, "materialize_report.json"), "w"), indent=2)
    n_cells = len(used) + len(plan) * len(FLOORS)
    print(f"\nmaterialized {n_cells} cells x {len(SEEDS)} seeds = {n_cells * len(SEEDS)} trainings "
          f"-> {out_dir}")
    print("  code recovery = HumanEval (movable) + MBPP (saturated, ratio-only reference)")
    print("  safety = XSTest dual sub-score (calibration shift, NOT capability recovery)")


def main():
    ap = argparse.ArgumentParser(description="Materialize the Part-B code + safety cells.")
    ap.add_argument("--pool", default="audit/results/stagec/phase1/pools/stagec_pool.jsonl")
    ap.add_argument("--stage_a", default="audit/results/stagec/phase2/stage_a")
    ap.add_argument("--ppl_nlls",
                    default="audit/results/stagec/phase1/stage_a_work/ppl_work/nlls.pkl")
    ap.add_argument("--out_dir", default="audit/results/stagec/partb/subsets")
    args = ap.parse_args()
    pool = read_jsonl(args.pool)
    materialize(args.stage_a, pool, args.ppl_nlls, args.out_dir)


if __name__ == "__main__":
    main()
