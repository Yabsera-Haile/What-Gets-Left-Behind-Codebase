from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pickle
import random

from audit.stagec.rarity_aware import rarity_aware_select, default_quality_gate

DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
BUDGET_B = 0.10
N_ABS = 500
PILOT_THRESHOLD = 250
SEEDS = [0, 1, 2]
CONDITIONS = [
    ("perplexity-low__none", "none", True, False),
    ("perplexity-low__proportional", "proportional", True, True),
    ("perplexity-low__absolute", "absolute", True, True),
    ("perplexity-low__hybrid", "hybrid", True, True),
    ("perplexity-low__proportional-nogate", "proportional", False, True),
]

NOISED_CONDITIONS = [
    ("perplexity-low__absolute-gateon", "absolute", True, True),
    ("perplexity-low__absolute-gateoff", "absolute", False, True),
]


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_pool(pool_path):
    pool = read_jsonl(pool_path)
    n_pool = len(pool)
    k = round(BUDGET_B * n_pool)
    return pool, n_pool, k


def decisive_counts(sel_ids, id2lang):
    c = collections.Counter(id2lang.get(str(i)) for i in sel_ids)
    return {l: c.get(l, 0) for l in DECISIVE}


def allocation_report(pool, id2lang):
    scores = {str(r["id"]): 0.0 for r in pool}
    k = round(BUDGET_B * len(pool))
    print(f"pool={len(pool)} budget b={BUDGET_B:.0%} -> k={k} | N_abs={N_ABS} thr={PILOT_THRESHOLD}\n")
    print(f"{'condition':38} {'per-decisive-lang kept':>24}  {'total':>6}  {'overflow':>8}")
    rows = {}
    for name, mode, gate, _ in CONDITIONS:
        r = rarity_aware_select(pool, scores, False, k, floor_mode=mode, n_abs=N_ABS,
                                group_key="language", quality_gate=gate,
                                protected_groups=DECISIVE)
        dc = {l: r["allocation"].get(l, {}).get("filled", 0) for l in DECISIVE}
        per = dc[DECISIVE[0]]
        uniform = all(v == per for v in dc.values())
        rows[name] = dc
        print(f"{name:38} {(str(per)+'/lang' if uniform else str(dc)):>24}  "
              f"{r['total']:>6}  {r['overflow']:>8}")
    prop = rows["perplexity-low__proportional"][DECISIVE[0]]
    absl = rows["perplexity-low__absolute"][DECISIVE[0]]
    hybr = rows["perplexity-low__hybrid"][DECISIVE[0]]
    ok = prop < N_ABS == absl == hybr and prop < PILOT_THRESHOLD <= absl
    print(f"\nACCEPTANCE (kept-counts): proportional({prop}) < N_abs({N_ABS}) = absolute({absl}) "
          f"= hybrid({hybr}); proportional < threshold({PILOT_THRESHOLD}) <= absolute : {ok}")
    return ok


def load_scores(pool, nlls_path):
    with open(nlls_path, "rb") as f:
        nll = pickle.load(f)
    idx2id = {r.get("pool_row_idx"): str(r["id"]) for r in pool}
    scores = {}
    for key, v in nll.items():
        sid = idx2id.get(key, str(key))
        scores[sid] = float(v)
    missing = [str(r["id"]) for r in pool if str(r["id"]) not in scores]
    if missing:
        raise SystemExit(f"{len(missing)} pool rows lack an NLL score (e.g. {missing[:3]}).")
    return scores


def materialize(pool, id2lang, nlls_path, out_dir):
    scores = load_scores(pool, nlls_path)
    k = round(BUDGET_B * len(pool))
    os.makedirs(out_dir, exist_ok=True)
    id2row = {str(r["id"]): r for r in pool}
    report = {}
    for name, mode, gate, multi in CONDITIONS:
        for seed in (SEEDS if multi else [0]):
            r = rarity_aware_select(pool, scores, False, k, floor_mode=mode, n_abs=N_ABS,
                                    group_key="language", quality_gate=gate,
                                    protected_groups=DECISIVE)
            tag = f"{name}__s{seed}"
            with open(os.path.join(out_dir, f"{tag}.jsonl"), "w", encoding="utf-8") as f:
                for i in r["selected_ids"]:
                    f.write(json.dumps(id2row[i], ensure_ascii=False) + "\n")
            report[tag] = {"total": r["total"], "gate_drops": sum(r["gate_drops"].values()),
                           "decisive": decisive_counts(r["selected_ids"], id2lang)}
    for seed in SEEDS:
        rng = random.Random(seed)
        ids = [str(r["id"]) for r in pool if default_quality_gate(r)]
        sel = rng.sample(ids, min(k, len(ids)))
        with open(os.path.join(out_dir, f"random__s{seed}.jsonl"), "w", encoding="utf-8") as f:
            for i in sel:
                f.write(json.dumps(id2row[i], ensure_ascii=False) + "\n")
        report[f"random__s{seed}"] = {"total": len(sel),
                                      "decisive": decisive_counts(sel, id2lang)}
    with open(os.path.join(out_dir, "full__s0.jsonl"), "w", encoding="utf-8") as f:
        for r in pool:
            if default_quality_gate(r):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(report, open(os.path.join(out_dir, "variant_report.json"), "w"), indent=2)
    print(f"materialized {len(report)+1} subsets -> {out_dir}")
    for tag, r in report.items():
        print(f"  {tag:44} total={r['total']:6} decisive={r['decisive']}")


def noised_ablation(pool, id2lang, nlls_path, out_dir):
    scores = load_scores(pool, nlls_path)
    k = round(BUDGET_B * len(pool))
    os.makedirs(out_dir, exist_ok=True)
    id2row = {str(r["id"]): r for r in pool}
    id2noised = {str(r["id"]): bool(r.get("is_noised")) for r in pool}
    print(f"pool={len(pool)} (noised rows={sum(id2noised.values())}) budget k={k} N_abs={N_ABS}\n")
    print(f"{'condition':40} {'kept':>5} {'decisive_kept':>13} {'noised_kept(decisive)':>21}")
    report = {}
    for name, mode, gate, _ in NOISED_CONDITIONS:
        for seed in SEEDS:
            r = rarity_aware_select(pool, scores, False, k, floor_mode=mode, n_abs=N_ABS,
                                    group_key="language", quality_gate=gate,
                                    protected_groups=DECISIVE)
            tag = f"{name}__s{seed}"
            with open(os.path.join(out_dir, f"{tag}.jsonl"), "w", encoding="utf-8") as f:
                for i in r["selected_ids"]:
                    f.write(json.dumps(id2row[i], ensure_ascii=False) + "\n")
            dc = decisive_counts(r["selected_ids"], id2lang)
            noised_dec = collections.Counter(
                id2lang[i] for i in r["selected_ids"]
                if id2noised.get(i) and id2lang.get(i) in DECISIVE)
            report[tag] = {"total": r["total"], "decisive": dc,
                           "noised_kept": {l: noised_dec.get(l, 0) for l in DECISIVE},
                           "noised_kept_total": sum(noised_dec.values())}
            if seed == 0:
                per = dc[DECISIVE[0]]
                uni = all(v == per for v in dc.values())
                print(f"{name:40} {r['total']:>5} "
                      f"{(str(per)+'/lang' if uni else 'mixed'):>13} "
                      f"{sum(noised_dec.values()):>21}")
    json.dump(report, open(os.path.join(out_dir, "noised_ablation_report.json"), "w"), indent=2)
    on = report["perplexity-low__absolute-gateon__s0"]["noised_kept_total"]
    off = report["perplexity-low__absolute-gateoff__s0"]["noised_kept_total"]
    print(f"\nGATE EFFECT: gate-on admits {on} corrupted decisive rows; "
          f"gate-off admits {off}. Expect on==0 and off>0 (perplexity-low prefers the "
          f"low-ppl truncated rows).")
    if on != 0:
        raise SystemExit(f"gate-on admitted {on} noised rows — the gate is not filtering.")
    print(f"materialized {len(report)} subsets -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Materialize the Stage-C perplexity-low variants.")
    ap.add_argument("--pool", default=None,
                    help="Pool jsonl (default: clean pool, or noised pool with --noised_ablation).")
    ap.add_argument("--nlls", default=None, help="perplexity-low NLL pickle over THIS pool.")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--allocation_only", action="store_true")
    ap.add_argument("--noised_ablation", action="store_true",
                    help="C1-Step 5: absolute floor gate-on vs gate-off on the noised pool.")
    args = ap.parse_args()
    pool_default = ("audit/results/stagec/phase1/pools/stagec_pool_noised.jsonl"
                    if args.noised_ablation
                    else "audit/results/stagec/phase1/pools/stagec_pool.jsonl")
    out_default = ("audit/results/stagec/phase1/subsets_noised" if args.noised_ablation
                   else "audit/results/stagec/phase1/subsets")
    pool_path = args.pool or pool_default
    out_dir = args.out_dir or out_default
    pool = read_jsonl(pool_path)
    id2lang = {str(r["id"]): r.get("language") for r in pool}
    if args.noised_ablation:
        if not args.nlls:
            raise SystemExit("--noised_ablation needs --nlls (perplexity NLLs over the NOISED pool).")
        noised_ablation(pool, id2lang, args.nlls, out_dir)
    elif args.allocation_only or not args.nlls:
        ok = allocation_report(pool, id2lang)
        if not args.allocation_only:
            print("\n(no --nlls given: printed the score-independent allocation only; pass the "
                  "perplexity-low NLLs from the server scoring pass to materialize subsets.)")
        if not ok:
            raise SystemExit(1)
    else:
        materialize(pool, id2lang, args.nlls, out_dir)


if __name__ == "__main__":
    main()
