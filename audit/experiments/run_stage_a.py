from __future__ import annotations

import argparse
import logging
import os

from audit.common import announce_dev, ensure_utf8, results_base

logger = logging.getLogger("audit.run_stage_a")


def main() -> None:
    ensure_utf8("audit.experiments.run_stage_a")
    from audit.selectors import (run_random, run_perplexity, run_rdsplus, run_ifd,
                                 run_semdedup, run_quality)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for n in ("sentence_transformers", "httpx", "urllib3", "datasets", "filelock",
              "huggingface_hub", "fsspec"):
        logging.getLogger(n).setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(description="Stage A / M1 selector driver.")
    ap.add_argument("--pool", default="audit/results/pilot_pool.jsonl")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--selections_dir", default=None,
                    help="Where to write selection JSONs (default <base>/selections). "
                         "Set this for a scale run so it doesn't clobber the pilot.")
    ap.add_argument("--work_root", default=None,
                    help="Root for the per-selector *_work caches (default <base>).")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Selectors to skip: random perplexity ifd rdsplus semdedup quality.")
    args = ap.parse_args()

    announce_dev(args.dev, logger)
    base = results_base(args.dev)
    sel_dir = args.selections_dir or os.path.join(base, "selections")
    work_root = args.work_root or base
    os.makedirs(sel_dir, exist_ok=True)

    if "random" not in args.skip:
        logger.info("=== [1/6] random ===")
        run_random.generate(args.metadata, sel_dir, dev=args.dev)
    if "perplexity" not in args.skip:
        logger.info("=== [2/6] perplexity ===")
        ppl_model = run_perplexity.PPL_DEV_MODEL if args.dev else run_perplexity.PPL_REAL_MODEL
        run_perplexity.generate(
            args.pool, args.metadata, sel_dir, os.path.join(work_root, "ppl_work"),
            model=ppl_model, dev=args.dev)
    if "ifd" not in args.skip:
        logger.info("=== [3/6] ifd ===")
        ifd_model = run_ifd.PPL_DEV_MODEL if args.dev else run_ifd.PPL_REAL_MODEL
        run_ifd.generate(
            args.pool, args.metadata, sel_dir, os.path.join(work_root, "ifd_work"),
            model=ifd_model, dev=args.dev)
    rds_work = os.path.join(work_root, "rds_work")
    if "rdsplus" not in args.skip:
        logger.info("=== [4/6] rdsplus ===")
        run_rdsplus.generate(
            args.pool, args.metadata, sel_dir, rds_work,
            model=(run_rdsplus.RDS_DEV_MODEL if args.dev else run_rdsplus.RDS_REAL_MODEL),
            dev=args.dev)
    if "semdedup" not in args.skip:
        logger.info("=== [5/6] semdedup (reuses RDS+ index) ===")
        index_path = os.path.join(rds_work, "pool_index.pt" if args.dev
                                  else "cosine_train_reps.pt")
        if os.path.exists(index_path):
            run_semdedup.generate(args.metadata, index_path, sel_dir, args.dev)
        else:
            logger.warning("Skipping semdedup: RDS+ index not found at %s "
                           "(run rdsplus first).", index_path)
    if "quality" not in args.skip:
        logger.info("=== [6/6] quality (LLM judge) ===")
        q_model = run_quality.QUALITY_DEV_MODEL if args.dev else run_quality.QUALITY_REAL_MODEL
        run_quality.generate(
            args.pool, args.metadata, sel_dir, os.path.join(work_root, "quality_work"),
            model=q_model, dev=args.dev)

    logger.info("Stage A selections written to %s", sel_dir)
    logger.info("Next: python -m audit.metrics.run_audit %s", "--dev" if args.dev else "")


if __name__ == "__main__":
    main()
