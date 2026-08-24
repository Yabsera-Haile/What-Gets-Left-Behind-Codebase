from __future__ import annotations

import argparse
import json
import logging
import os

os.environ.setdefault("USE_TORCH", "0")

from audit.metadata import pool_io

logger = logging.getLogger("audit.export_pilot_jsonl")

METADATA_COLUMNS = [
    "pool_row_idx", "id", "source", "skill_label", "language",
    "resource_bucket", "quality_score", "is_clean", "is_noised",
]


def load_yaml(path: str) -> dict:
    if path and os.path.exists(path):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "fsspec", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(description="Export pilot pool jsonl for selectors.")
    ap.add_argument("--config", default="audit/configs/build_metadata.yaml")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--output", default="audit/results/pilot_pool.jsonl")
    ap.add_argument("--materialized-pool", dest="materialized_pool", default=None)
    args = ap.parse_args()

    import pandas as pd

    cfg = load_yaml(args.config)
    materialized_pool = args.materialized_pool or cfg.get(
        "materialized_pool", "audit/results/pool_pilot.jsonl")

    pool_path = pool_io.materialize_pilot_pool(
        source=cfg.get("pool", "allenai/tulu-3-sft-mixture"),
        out_path=materialized_pool,
        sample_size=int(cfg.get("sample_size", 10000)),
        seed=int(cfg.get("seed", 42)),
        buffer_size=int(cfg.get("buffer_size", 50000)),
        force=False,
        strategy=cfg.get("sampling_strategy", "reservoir"),
    )
    originals = pool_io.read_pool(pool_path)
    logger.info("Recovered %d original examples from %s", len(originals), pool_path)

    meta = pd.read_parquet(args.metadata).sort_values("pool_row_idx").reset_index(drop=True)
    logger.info("Loaded %d metadata rows from %s", len(meta), args.metadata)

    if len(meta) != len(originals):
        raise ValueError(
            f"Row count mismatch: metadata={len(meta)} vs pool={len(originals)}. "
            "The pool file and metadata parquet must come from the same seed/sample.")

    meta_records = meta.to_dict("records")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, (orig, mrow) in enumerate(zip(originals, meta_records)):
            if int(mrow["pool_row_idx"]) != i:
                raise ValueError(f"pool_row_idx out of order at position {i}: {mrow['pool_row_idx']}")
            merged = {"messages": orig.get("messages")}
            for col in METADATA_COLUMNS:
                val = mrow[col]
                merged[col] = val.item() if hasattr(val, "item") else val
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            n += 1
    logger.info("Wrote %d rows -> %s", n, args.output)

    with open(args.output, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == n, f"Expected {n} rows, got {len(lines)}"
    idxs = [r["pool_row_idx"] for r in lines]
    assert idxs == list(range(len(lines))), "pool_row_idx not 0..N-1 in order"
    has_messages = all("messages" in r for r in lines)
    has_meta = all(all(c in r for c in METADATA_COLUMNS) for r in lines)
    logger.info("VERIFIED: %d rows, pool_row_idx 0..%d in order, "
                "messages present=%s, metadata cols present=%s",
                len(lines), len(lines) - 1, has_messages, has_meta)
    logger.info("Sample row 0 keys: %s", sorted(lines[0].keys()))


if __name__ == "__main__":
    main()
