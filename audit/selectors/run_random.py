from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from audit.common import announce_dev, results_base, run_meta

logger = logging.getLogger("audit.run_random")

BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50]
SEEDS = [0, 1, 2]


def generate(metadata: str, out_dir: str, dev: bool = False,
             budgets=BUDGETS, seeds=SEEDS) -> list[str]:
    df = pd.read_parquet(metadata)
    ids = df["id"].astype(str).to_numpy()
    n = len(ids)
    os.makedirs(out_dir, exist_ok=True)
    meta = run_meta("none(random)", dev, extra={"selector_kind": "random"})

    written = []
    for budget in budgets:
        k = round(budget * n)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            picked = rng.choice(ids, size=k, replace=False)
            obj = {
                "selector": "random",
                "budget": budget,
                "seed": seed,
                "selected_ids": [str(x) for x in picked],
                "meta": meta,
            }
            path = os.path.join(out_dir, f"random__b{budget}__s{seed}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f)
            written.append(path)
            logger.info("random b=%.2f s=%d -> %d ids -> %s",
                        budget, seed, k, os.path.basename(path))
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Random selection adapter.")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--out_dir", default=None,
                    help="Override output dir (default: <results_base>/selections).")
    ap.add_argument("--dev", action="store_true", help="Dev run -> audit/results/dev/.")
    args = ap.parse_args()
    announce_dev(args.dev, logger)
    out_dir = args.out_dir or os.path.join(results_base(args.dev), "selections")
    paths = generate(args.metadata, out_dir, dev=args.dev)
    logger.info("Wrote %d random selection files to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()
