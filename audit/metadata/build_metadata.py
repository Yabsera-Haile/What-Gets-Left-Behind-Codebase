from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import random
import sys

os.environ.setdefault("USE_TORCH", "0")

from audit.metadata import pool_io
from audit.metadata.langid import LanguageIdentifier

logger = logging.getLogger("audit.build_metadata")

REQUIRED_COLUMNS = [
    "pool_row_idx", "id", "source", "skill_label",
    "language", "resource_bucket", "quality_score",
]


DEFAULTS = {
    "pool": "allenai/tulu-3-sft-mixture",
    "materialized_pool": "audit/results/pool_pilot.jsonl",
    "sample_size": 10000,
    "seed": 42,
    "sampling_strategy": "reservoir",
    "buffer_size": 50000,
    "output": "audit/results/metadata_pilot.parquet",
    "dataset_to_skill": "audit/configs/dataset_to_skill.json",
    "joshi_resource": "audit/configs/joshi_resource.json",
    "langid_backend": "glotlid",
    "tokenizer": "",
    "log_file": "audit/results/build_metadata.log",
}


def load_config() -> dict:
    parser = argparse.ArgumentParser(description="Build pilot audit metadata parquet.")
    parser.add_argument("--config", default="audit/configs/build_metadata.yaml")
    parser.add_argument("--pool")
    parser.add_argument("--materialized-pool", dest="materialized_pool")
    parser.add_argument("--sample-size", dest="sample_size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sampling-strategy", dest="sampling_strategy",
                        choices=["reservoir", "head_buffer"])
    parser.add_argument("--buffer-size", dest="buffer_size", type=int)
    parser.add_argument("--output")
    parser.add_argument("--dataset-to-skill", dest="dataset_to_skill")
    parser.add_argument("--joshi-resource", dest="joshi_resource")
    parser.add_argument("--langid-backend", dest="langid_backend",
                        choices=["glotlid", "py3langid"])
    parser.add_argument("--tokenizer")
    parser.add_argument("--log-file", dest="log_file")
    parser.add_argument("--force", action="store_true",
                        help="Re-materialize the pool even if it exists.")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update({k: v for k, v in (yaml.safe_load(f) or {}).items()})
    for key in DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    cfg["force"] = args.force
    cfg["config_path"] = args.config
    return cfg


def setup_logging(log_file: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)
    for noisy in ("httpx", "urllib3", "filelock", "fsspec", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def first_message_by_role(messages: list[dict], role: str) -> str:
    for m in messages or []:
        if m.get("role") == role:
            return (m.get("content") or "").strip()
    return ""


def instruction_text(example: dict) -> str:
    msgs = example.get("messages")
    if msgs:
        txt = first_message_by_role(msgs, "user")
        if txt:
            return txt
        for m in msgs:
            if m.get("role") != "system" and m.get("content"):
                return m["content"].strip()
    return (example.get("instruction") or example.get("prompt") or "").strip()


def response_text(example: dict) -> str:
    msgs = example.get("messages")
    if msgs:
        return first_message_by_role(msgs, "assistant")
    return (example.get("output") or example.get("response") or "").strip()


def full_text(example: dict) -> str:
    msgs = example.get("messages")
    if msgs:
        return "\n".join((m.get("content") or "") for m in msgs)
    return " ".join(filter(None, [instruction_text(example), response_text(example)]))


def mint_id(example: dict) -> str:
    basis = instruction_text(example) + "\n\n" + response_text(example)
    return "sha256:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()


class SkillMapper:
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.exact = cfg.get("exact", {})
        self.rules = cfg.get("keyword_rules", [])
        self.default = cfg.get("default", "other")
        self.fallback_hits = collections.Counter()

    def map(self, source: str) -> str:
        if source in self.exact:
            return self.exact[source]
        low = (source or "").lower()
        for rule in self.rules:
            if rule["contains"].lower() in low:
                self.fallback_hits[source] += 1
                return rule["skill"]
        self.fallback_hits[source] += 1
        return self.default


class ResourceMapper:
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.classes = cfg.get("classes", {})
        self.default = cfg.get("default", 0)
        self.unmapped = collections.Counter()

    def bucket(self, iso3: str) -> int:
        if iso3 in self.classes:
            return self.classes[iso3]
        if iso3 not in ("und", ""):
            self.unmapped[iso3] += 1
        return self.default


def quality_score(example: dict, token_len: int) -> tuple[float, bool]:
    resp = response_text(example)
    c_nonempty = 1.0 if resp else 0.0
    c_length = 1.0 if 5 <= token_len <= 2000 else 0.0
    c_fences = 1.0 if full_text(example).count("```") % 2 == 0 else 0.0
    score = (c_nonempty + c_length + c_fences) / 3.0
    return round(score, 4), bool(resp)


def build(cfg: dict) -> "pandas.DataFrame":
    import pandas as pd

    random.seed(cfg["seed"])

    pool_path = pool_io.materialize_pilot_pool(
        source=cfg["pool"],
        out_path=cfg["materialized_pool"],
        sample_size=cfg["sample_size"],
        seed=cfg["seed"],
        buffer_size=cfg["buffer_size"],
        force=cfg["force"],
        strategy=cfg["sampling_strategy"],
    )
    pool = pool_io.read_pool(pool_path)
    logger.info("Loaded %d pool examples from %s", len(pool), pool_path)

    src_key = pool_io.detect_source_key(pool[0]) if pool else None
    logger.info("Detected provenance key in first example: %r", src_key)
    prov_counts = collections.Counter(ex.get(src_key, "<MISSING>") for ex in pool)
    logger.info("Distinct provenance values in pilot (%d):", len(prov_counts))
    for val, c in prov_counts.most_common():
        logger.info("    %6d  %s", c, val)

    skill_mapper = SkillMapper(cfg["dataset_to_skill"])
    resource_mapper = ResourceMapper(cfg["joshi_resource"])
    langid = LanguageIdentifier(backend=cfg["langid_backend"])

    use_tokenizer = bool(cfg["tokenizer"])
    tokenizer = None
    if use_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
        logger.info("Using tokenizer %s for token_len.", cfg["tokenizer"])
    else:
        logger.info("No tokenizer configured: token_len is a whitespace-word proxy "
                    "(token_len_approx=True).")

    id_branch = collections.Counter()
    rows = []
    for idx, ex in enumerate(pool):
        instr = instruction_text(ex)

        raw_id = ex.get("id")
        if raw_id not in (None, "", []):
            ex_id = str(raw_id)
            id_branch["pool_id"] += 1
        else:
            ex_id = mint_id(ex)
            id_branch["hashed"] += 1

        source = ex.get(src_key, "<MISSING>") if src_key else "<MISSING>"
        skill = skill_mapper.map(source)

        iso3, script, prob, backend = langid.predict(instr)
        bucket = resource_mapper.bucket(iso3)

        if use_tokenizer:
            token_len = len(tokenizer.encode(full_text(ex)))
            approx = False
        else:
            token_len = len(full_text(ex).split())
            approx = True

        qscore, resp_nonempty = quality_score(ex, token_len)

        rows.append({
            "pool_row_idx": idx,
            "id": ex_id,
            "source": source,
            "skill_label": skill,
            "language": iso3,
            "language_script": script,
            "language_prob": round(prob, 4),
            "langid_backend": backend,
            "resource_bucket": bucket,
            "token_len": token_len,
            "token_len_approx": approx,
            "quality_score": qscore,
            "llm_judge_score": None,
            "is_clean": True,
            "is_noised": False,
            "n_messages": len(ex.get("messages", []) or []),
            "response_nonempty": resp_nonempty,
        })
        if (idx + 1) % 2000 == 0:
            logger.info("  ...processed %d / %d", idx + 1, len(pool))

    df = pd.DataFrame(rows)

    logger.info("id source: %s", dict(id_branch))
    if skill_mapper.fallback_hits:
        logger.info("skill mapping fell back to keyword/default for %d distinct sources.",
                    len(skill_mapper.fallback_hits))
    if resource_mapper.unmapped:
        logger.warning("Languages with no Joshi mapping (defaulted to class 0): %s",
                       dict(resource_mapper.unmapped.most_common()))

    if df["pool_row_idx"].duplicated().any():
        logger.error("pool_row_idx is not unique!")
    dup_ids = int(df["id"].duplicated().sum())
    if dup_ids:
        logger.warning("%d duplicate id values (content-identical examples).", dup_ids)

    return df


def print_summaries(df: "pandas.DataFrame") -> None:
    logger.info("=" * 60)
    logger.info("SUMMARY: counts per skill_label")
    for skill, c in df["skill_label"].value_counts().items():
        logger.info("    %-14s %6d", skill, c)
    n_skills_nontrivial = int((df["skill_label"].value_counts() >= 50).sum())

    logger.info("SUMMARY: counts per resource_bucket (0=low ... 5=high)")
    for bucket in sorted(df["resource_bucket"].unique()):
        c = int((df["resource_bucket"] == bucket).sum())
        flag = "   <-- under 50 (Phase 5 Aya injection target)" if c < 50 else ""
        logger.info("    bucket %s: %6d%s", bucket, c, flag)

    logger.info("SUMMARY: top languages")
    for lang, c in df["language"].value_counts().head(15).items():
        logger.info("    %-8s %6d", lang, c)

    low = int(df["resource_bucket"].isin([0, 1, 2]).sum())
    high = int(df["resource_bucket"].isin([3, 4, 5]).sum())
    logger.info("Low-resource (0-2): %d | High-resource (3-5): %d", low, high)
    logger.info("Non-trivial skill labels (>=50): %d", n_skills_nontrivial)
    logger.info("=" * 60)


def main() -> None:
    cfg = load_config()
    setup_logging(cfg["log_file"])
    logger.info("Config: %s", {k: cfg[k] for k in DEFAULTS})

    df = build(cfg)

    out = cfg["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Wrote %d rows x %d cols -> %s", len(df), df.shape[1], out)

    print_summaries(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("ACCEPTANCE FAIL: missing required columns: %s", missing)
        sys.exit(1)
    logger.info("Acceptance columns present: %s", REQUIRED_COLUMNS)


if __name__ == "__main__":
    main()
