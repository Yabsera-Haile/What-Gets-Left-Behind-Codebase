from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd


def discover(meta_n: int):
    found = []
    for p in glob.glob("audit/**/quality_scores.pkl", recursive=True):
        try:
            with open(p, "rb") as f:
                found.append((p, len(pickle.load(f))))
        except Exception as e:
            found.append((p, f"ERR:{e}"))
    exact = [p for p, n in found if n == meta_n]
    return exact, found


def main():
    ap = argparse.ArgumentParser(description="Stage-A 4b: mean judge score by resource tier.")
    ap.add_argument("--scores", default=None, help="quality_scores.pkl (auto-discovered if omitted).")
    ap.add_argument("--metadata", default="audit/stageb/data/metadata_pilot.parquet")
    ap.add_argument("--out", default="audit/stageb/data/stage_a_4b_quality_by_tier.csv")
    args = ap.parse_args()

    m = pd.read_parquet(args.metadata)
    n = len(m)
    print(f"pilot metadata: {n} rows ({args.metadata})")
    if args.scores is None:
        exact, found = discover(n)
        print("quality_scores.pkl found under audit/:")
        for p, c in found:
            print(f"    n={c}  {p}")
        if not exact:
            raise SystemExit(f"No quality_scores.pkl with {n} entries (the pilot). Pass --scores.")
        if len(exact) > 1:
            print(f"WARNING: multiple {n}-entry pickles; using the first: {exact[0]}")
        args.scores = exact[0]
    print(f"using judge scores: {args.scores}")
    with open(args.scores, "rb") as f:
        scores = pickle.load(f)

    m["judge"] = m["pool_row_idx"].map(scores)
    matched = int(m["judge"].notna().sum())
    print(f"matched {matched}/{n} rows by pool_row_idx"
          + ("" if matched >= 0.99 * n else "  <-- LOW MATCH: key mismatch, investigate"))
    vals = sorted(set(v for v in scores.values() if isinstance(v, (int, float))))
    print(f"score scale observed: {vals[:8]}{' ...' if len(vals) > 8 else ''} "
          f"(min {min(vals)}, max {max(vals)})")

    g = m.groupby("resource_bucket")["judge"].agg(["mean", "std", "count"]).round(3)
    print("\n=== 4b: mean judge score (1-5) by resource tier "
          "(0 = lowest-resource ... 5 = highest) ===")
    print(g.to_string())

    tiers = g.index.astype(float).values
    means = g["mean"].values
    slope = float(np.polyfit(tiers, means, 1)[0])
    low = float(m[m.resource_bucket.isin([0, 1, 2])]["judge"].mean())
    high = float(m[m.resource_bucket == 5]["judge"].mean())
    ml = float(m[m.skill_label == "multilingual"]["judge"].mean())
    rest = float(m[m.skill_label != "multilingual"]["judge"].mean())
    print(f"\nTIER GRADIENT: slope = {slope:+.4f} judge-points per +1 tier")
    print(f"  low-resource (tiers 0-2) mean = {low:.3f}   high-resource (tier 5) mean = {high:.3f}"
          f"   contrast = {high - low:+.3f}")
    print(f"  multilingual mean = {ml:.3f}   non-multilingual mean = {rest:.3f}"
          f"   contrast = {ml - rest:+.3f}")
    print("Reading: quality selector keeps HIGH scores, so a lower judge mean on low-resource / "
          "multilingual is the bias at the SCORING stage that drives the 4a deletion.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = g.reset_index()
    out.to_csv(args.out, index=False)
    with open(os.path.splitext(args.out)[0] + "_gradient.txt", "w", encoding="utf-8") as f:
        f.write(f"slope_per_tier={slope:.4f}\nlow_0_2_mean={low:.4f}\nhigh_5_mean={high:.4f}\n"
                f"low_high_contrast={high - low:.4f}\nmultilingual_mean={ml:.4f}\n"
                f"nonmultilingual_mean={rest:.4f}\nml_contrast={ml - rest:.4f}\nn_scored={n}\n"
                f"scores_pickle={args.scores}\n")
    print(f"\nwrote {args.out} (+ _gradient.txt) — commit these back")


if __name__ == "__main__":
    main()
