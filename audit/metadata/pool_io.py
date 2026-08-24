from __future__ import annotations

import json
import logging
import os
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

SOURCE_KEY_CANDIDATES = ("dataset", "source")


def looks_like_local_path(spec: str) -> bool:
    if os.path.exists(spec):
        return True
    return spec.endswith(".jsonl") or spec.endswith(".json")


def detect_source_key(example: dict) -> Optional[str]:
    for key in SOURCE_KEY_CANDIDATES:
        if key in example:
            return key
    return None


def materialize_pilot_pool(
    source: str,
    out_path: str,
    sample_size: int,
    seed: int,
    buffer_size: int = 50_000,
    force: bool = False,
    strategy: str = "reservoir",
) -> str:
    if strategy not in ("reservoir", "head_buffer"):
        raise ValueError(f"Unknown sampling strategy: {strategy!r}")

    if looks_like_local_path(source):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Local pool not found: {source}")
        logger.info("Using local pool directly as pilot pool: %s", source)
        return source

    if os.path.exists(out_path) and not force:
        logger.info("Reusing existing materialized pilot pool: %s", out_path)
        return out_path

    from datasets import load_dataset

    stream = load_dataset(source, split="train", streaming=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if strategy == "head_buffer":
        logger.info(
            "Sampling HF %r via head_buffer (seed=%d, buffer=%d, take=%d)",
            source, seed, buffer_size, sample_size,
        )
        sampled = list(stream.shuffle(seed=seed, buffer_size=buffer_size).take(sample_size))
    else:
        import random
        logger.info(
            "Sampling HF %r via full-stream reservoir (seed=%d, target=%d)",
            source, seed, sample_size,
        )
        rng = random.Random(seed)
        reservoir: list[dict] = []
        seen = 0
        for ex in stream:
            if seen < sample_size:
                reservoir.append(ex)
            else:
                j = rng.randint(0, seen)
                if j < sample_size:
                    reservoir[j] = ex
            seen += 1
            if seen % 50_000 == 0:
                logger.info("  ...scanned %d rows (reservoir full=%s)",
                            seen, len(reservoir) >= sample_size)
        logger.info("Reservoir scan complete: saw %d rows total.", seen)
        rng.shuffle(reservoir)
        sampled = reservoir

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for example in sampled:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)
    logger.info("Materialized %d examples to %s", len(sampled), out_path)
    if len(sampled) < sample_size:
        logger.warning("Requested %d but only %d examples available.",
                       sample_size, len(sampled))
    return out_path


def iter_pool(path: str) -> Iterator[tuple[int, dict]]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def read_pool(path: str) -> list[dict]:
    return [ex for _, ex in iter_pool(path)]
