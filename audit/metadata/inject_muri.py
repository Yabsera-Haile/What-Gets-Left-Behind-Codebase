from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os

os.environ.setdefault("USE_TORCH", "0")

from audit.metadata import pool_io
from audit.metadata.build_metadata import ResourceMapper
from audit.metadata.langid import LanguageIdentifier

logger = logging.getLogger("audit.inject_muri")

LOW_BUCKETS = [0, 1, 2]
NON_LANGUAGE_CONFIGS = {"code"}
SKIP_CONFIGS = {"eng"}


def mint_muri_id(inp: str, out: str) -> str:
    h = hashlib.sha256((inp + "\n\n" + out).encode("utf-8")).hexdigest()[:24]
    return f"muri:{h}"


def existing_bucket_counts(metadata_path: str) -> dict[int, int]:
    if not os.path.exists(metadata_path):
        logger.warning("No existing metadata at %s; assuming 0 per bucket.", metadata_path)
        return {b: 0 for b in LOW_BUCKETS}
    import pandas as pd
    vc = pd.read_parquet(metadata_path)["resource_bucket"].value_counts().to_dict()
    return {b: int(vc.get(b, 0)) for b in LOW_BUCKETS}


def candidate_configs(muri_dataset: str, resource_mapper: ResourceMapper) -> list[str]:
    from datasets import get_dataset_config_names
    configs = [c for c in get_dataset_config_names(muri_dataset)
               if c not in NON_LANGUAGE_CONFIGS and c not in SKIP_CONFIGS]
    by_bucket = collections.defaultdict(list)
    for c in configs:
        by_bucket[resource_mapper.bucket(c)].append(c)
    ordered = by_bucket.get(1, []) + by_bucket.get(2, []) + by_bucket.get(0, [])
    logger.info("MURI candidate configs: %d (nominal bucket1=%d, bucket2=%d, bucket0=%d)",
                len(ordered), len(by_bucket.get(1, [])), len(by_bucket.get(2, [])),
                len(by_bucket.get(0, [])))
    return ordered


def sample_muri(muri_dataset, configs, quotas, langid, resource_mapper,
                per_config_cap, muri_n, max_scan):
    from datasets import load_dataset
    rows = []
    achieved = collections.Counter()
    seen_ids: set[str] = set()
    scanned = 0
    for cfg in configs:
        if sum(quotas.values()) <= 0 or len(rows) >= muri_n or scanned >= max_scan:
            break
        try:
            ds = load_dataset(muri_dataset, cfg, split="train", streaming=True)
        except Exception as exc:
            logger.warning("Skipping config %s (%s)", cfg, exc)
            continue
        taken_here = 0
        for ex in ds:
            if taken_here >= per_config_cap or len(rows) >= muri_n or scanned >= max_scan:
                break
            scanned += 1
            inp = (ex.get("input") or "").strip()
            out = (ex.get("output") or "").strip()
            if not inp or not out:
                continue
            mid = mint_muri_id(inp, out)
            if mid in seen_ids:
                continue
            iso3, _, _, _ = langid.predict(inp)
            bucket = resource_mapper.bucket(iso3)
            if bucket in quotas and quotas[bucket] > 0:
                seen_ids.add(mid)
                rows.append({
                    "id": mid,
                    "source": "muri",
                    "messages": [{"role": "user", "content": inp},
                                 {"role": "assistant", "content": out}],
                })
                quotas[bucket] -= 1
                achieved[bucket] += 1
                taken_here += 1
        if taken_here:
            logger.info("  config %-5s -> +%d (remaining deficits: %s)", cfg, taken_here,
                        {b: quotas[b] for b in LOW_BUCKETS})
    logger.info("Scanned %d MURI examples; injected %d.", scanned, len(rows))
    return rows, achieved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for n in ("httpx", "urllib3", "filelock", "fsspec", "huggingface_hub", "datasets"):
        logging.getLogger(n).setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(description="Inject MURI-IT low-resource examples.")
    ap.add_argument("--config", default="audit/configs/build_metadata.yaml")
    ap.add_argument("--muri_dataset", default=None)
    ap.add_argument("--base_pool", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--muri_n", type=int, default=None)
    ap.add_argument("--target_per_bucket", type=int, default=None)
    ap.add_argument("--per_config_cap", type=int, default=None)
    ap.add_argument("--max_scan", type=int, default=None)
    ap.add_argument("--joshi_resource", default=None)
    ap.add_argument("--langid_backend", default=None)
    cli = ap.parse_args()

    cfg = {}
    if cli.config and os.path.exists(cli.config):
        import yaml
        cfg = yaml.safe_load(open(cli.config, encoding="utf-8")) or {}

    def pick(name, yaml_key, default):
        v = getattr(cli, name)
        return v if v is not None else cfg.get(yaml_key, default)

    args = argparse.Namespace(
        muri_dataset=pick("muri_dataset", "muri_dataset", "akoksal/muri"),
        base_pool=pick("base_pool", "materialized_pool", "audit/results/pool_pilot.jsonl"),
        metadata=pick("metadata", "output", "audit/results/metadata_pilot.parquet"),
        output=pick("output", "enriched_pool", "audit/results/pool_pilot_enriched.jsonl"),
        muri_n=int(pick("muri_n", "muri_n", 2000)),
        target_per_bucket=int(pick("target_per_bucket", "muri_target_per_bucket", 500)),
        per_config_cap=int(pick("per_config_cap", "muri_per_config_cap", 150)),
        max_scan=int(pick("max_scan", "muri_max_scan", 200000)),
        joshi_resource=pick("joshi_resource", "joshi_resource", "audit/configs/joshi_resource.json"),
        langid_backend=pick("langid_backend", "langid_backend", "glotlid"),
    )

    resource_mapper = ResourceMapper(args.joshi_resource)
    langid = LanguageIdentifier(backend=args.langid_backend)

    existing = existing_bucket_counts(args.metadata)
    quotas = {b: max(0, args.target_per_bucket - existing[b]) for b in LOW_BUCKETS}
    logger.info("Existing low-bucket counts: %s", existing)
    logger.info("Target per bucket=%d -> deficits/quotas: %s (muri_n cap=%d)",
                args.target_per_bucket, quotas, args.muri_n)

    configs = candidate_configs(args.muri_dataset, resource_mapper)
    muri_rows, achieved = sample_muri(args.muri_dataset, configs, dict(quotas), langid,
                                      resource_mapper, args.per_config_cap, args.muri_n,
                                      args.max_scan)

    base = pool_io.read_pool(args.base_pool)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in base:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for ex in muri_rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Wrote enriched pool: %d base + %d MURI = %d rows -> %s",
                len(base), len(muri_rows), len(base) + len(muri_rows), args.output)

    logger.info("Projected low-resource bucket counts after enrichment:")
    short = []
    for b in LOW_BUCKETS:
        total = existing[b] + achieved.get(b, 0)
        flag = "  <-- still under 300!" if total < 300 else ""
        logger.info("  bucket %d: existing %d + injected %d = %d%s",
                    b, existing[b], achieved.get(b, 0), total, flag)
        if total < 300:
            short.append(b)
    if short:
        logger.warning("Buckets %s still under 300 — increase --muri_n/--target_per_bucket "
                       "or MURI lacks enough of those Joshi classes.", short)
    logger.info("NEXT: build_metadata --pool %s  then  export_pilot_jsonl "
                "--materialized-pool %s", args.output, args.output)
    logger.info("NOTE: the pool changed, so ALL pool-specific caches are stale (they are "
                "keyed by pool_row_idx over the old 10k and would silently MISS the MURI "
                "rows). Before re-running selectors, delete audit/results/*_work and the "
                "RDS+ .pt (or pass each scorer's --force) so perplexity, ifd, quality, the "
                "RDS+ index and semdedup all recompute over the enriched pool.")


if __name__ == "__main__":
    main()
