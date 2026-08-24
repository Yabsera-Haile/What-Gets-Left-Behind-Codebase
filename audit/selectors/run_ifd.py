from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle

import pandas as pd

from audit.common import (PPL_DEV_MODEL, PPL_REAL_MODEL, VramSampler,
                          announce_dev, results_base, run_meta)

logger = logging.getLogger("audit.run_ifd")

BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50]
DISCARD_THRESHOLD = 1.0


def _split_messages(messages: list[dict]) -> tuple[str, str, str]:
    sys_txt = user_txt = resp_txt = ""
    for m in messages or []:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "system" and not sys_txt:
            sys_txt = content
        elif role == "user" and not user_txt:
            user_txt = content
        elif role == "assistant" and user_txt and not resp_txt:
            resp_txt = content
            break
    return sys_txt, user_txt, resp_txt


def _build_inputs(messages, tokenizer, max_len):
    sys_txt, user_txt, resp_txt = _split_messages(messages)
    if not user_txt or not resp_txt:
        return None
    prompt = ""
    if sys_txt:
        prompt += "<|system|>\n" + sys_txt + "\n"
    prompt += "<|user|>\n" + user_txt + "\n<|assistant|>\n"
    response = resp_txt + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    resp_ids = tokenizer(response, add_special_tokens=False).input_ids
    if len(resp_ids) == 0:
        return None
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    cond_ids = (prompt_ids + resp_ids)[:max_len]
    n_prompt = min(len(prompt_ids), max_len)
    cond_labels = ([-100] * n_prompt + resp_ids)[:max_len]
    uncond_ids = (bos + resp_ids)[:max_len]
    uncond_labels = ([-100] * len(bos) + resp_ids)[:max_len]
    if all(l == -100 for l in cond_labels):
        return None
    return cond_ids, cond_labels, uncond_ids, uncond_labels


def score_ifd(pool: str, work_dir: str, model_name: str, dtype: str, max_len: int,
              force: bool, limit: int | None = None) -> tuple[str, VramSampler]:
    os.makedirs(work_dir, exist_ok=True)
    out = os.path.join(work_dir, "ifd_scores.pkl")
    sampler = VramSampler()
    if os.path.exists(out) and not force:
        logger.info("Reusing existing IFD scores: %s", out)
        return out, sampler

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    kwargs = {"torch_dtype": dtypes.get(dtype, torch.float32)}
    if os.getenv("HF_TOKEN"):
        kwargs["token"] = os.getenv("HF_TOKEN")
    logger.info("Loading IFD model %s (%s) ...", model_name, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = next(model.parameters()).device

    @torch.no_grad()
    def _loss(ids, labels):
        t = torch.tensor([ids], device=device)
        lab = torch.tensor([labels], device=device)
        attn = torch.ones_like(t)
        return model(input_ids=t, attention_mask=attn, labels=lab).loss.item()

    examples = [json.loads(l) for l in open(pool, encoding="utf-8") if l.strip()]
    if limit:
        examples = examples[:limit]
    scores: dict[int, float] = {}
    with sampler:
        for idx, ex in enumerate(examples):
            built = _build_inputs(ex.get("messages", []), tokenizer, max_len)
            if built is None:
                scores[idx] = float("inf")
                continue
            cond_ids, cond_labels, uncond_ids, uncond_labels = built
            loss_cond = _loss(cond_ids, cond_labels)
            loss_uncond = _loss(uncond_ids, uncond_labels)
            if not math.isfinite(loss_cond) or not math.isfinite(loss_uncond):
                scores[idx] = float("inf")
            else:
                scores[idx] = math.exp(loss_cond - loss_uncond)
            if (idx + 1) % 500 == 0:
                logger.info("  ...IFD scored %d / %d", idx + 1, len(examples))
    with open(out, "wb") as f:
        pickle.dump(scores, f)
    logger.info("IFD scoring done in %ss, peak VRAM %s MiB", sampler.runtime_s, sampler.peak_mib)
    return out, sampler


def print_ifd_histogram(scores: dict) -> int:
    import numpy as np
    finite = np.array([s for s in scores.values() if math.isfinite(s)])
    n_inf = sum(1 for s in scores.values() if not math.isfinite(s))
    discarded = int((finite >= DISCARD_THRESHOLD).sum()) + n_inf
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, float("inf")]
    logger.info("IFD score histogram (n=%d, +%d non-finite):", len(finite), n_inf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = int(((finite >= lo) & (finite < hi)).sum())
        bar = "#" * min(60, c * 60 // max(1, len(finite)))
        logger.info("  [%4.1f, %4s)  %6d  %s", lo, ("inf" if hi == float("inf") else f"{hi:.1f}"), c, bar)
    logger.info("Discarded (IFD >= %.1f, incl. non-finite): %d / %d (%.1f%%)",
                DISCARD_THRESHOLD, discarded, len(scores), 100.0 * discarded / max(1, len(scores)))
    return discarded


def select_ifd(scores: dict, k: int, idx2id: dict) -> tuple[list[str], int]:
    valid = [(idx, s) for idx, s in scores.items()
             if math.isfinite(s) and s < DISCARD_THRESHOLD]
    valid.sort(key=lambda kv: kv[1], reverse=True)
    chosen = valid[:k]
    ids = [idx2id[int(idx)] for idx, _ in chosen if int(idx) in idx2id]
    return ids, len(valid)


def generate(pool: str, metadata: str, out_dir: str, work_dir: str, model: str,
             dev: bool, dtype: str = "fp32", max_len: int = 2048, budgets=BUDGETS,
             force_score: bool = False, limit: int | None = None) -> list[str]:
    meta_df = pd.read_parquet(metadata)
    n_pool = len(meta_df)
    idx2id = {int(r): str(i) for r, i in zip(meta_df["pool_row_idx"], meta_df["id"])}
    os.makedirs(out_dir, exist_ok=True)

    scores_path, sampler = score_ifd(pool, work_dir, model, dtype, max_len, force_score, limit)
    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    discarded = print_ifd_histogram(scores)
    base_meta = run_meta(model, dev, sampler, extra={
        "selector_kind": "ifd", "discard_threshold": DISCARD_THRESHOLD,
        "n_discarded_ge_1": discarded, "n_scored": len(scores), "dtype": dtype})

    written = []
    for budget in budgets:
        k = round(budget * n_pool)
        ids, n_valid = select_ifd(scores, k, idx2id)
        if len(ids) < k:
            logger.warning("ifd b=%.2f: only %d examples have IFD<1 (< budget %d); "
                           "keeping all survivors.", budget, len(ids), k)
        obj = {"selector": "ifd", "budget": budget, "seed": 0,
               "selected_ids": ids, "meta": {**base_meta, "n_valid_below_1": n_valid}}
        path = os.path.join(out_dir, f"ifd__b{budget}__s0.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        written.append(path)
        logger.info("ifd b=%.2f -> %d ids -> %s", budget, len(ids), os.path.basename(path))
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="IFD (Instruction-Following Difficulty) selector.")
    ap.add_argument("--pool", default="audit/results/pilot_pool.jsonl")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--model", default=None, help="Override the IFD model.")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--dev", action="store_true", help="Dev run with proxy model -> dev/.")
    ap.add_argument("--force_score", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Score only first N (testing).")
    args = ap.parse_args()

    announce_dev(args.dev, logger)
    base = results_base(args.dev)
    out_dir = args.out_dir or os.path.join(base, "selections")
    work_dir = args.work_dir or os.path.join(base, "ifd_work")
    model = args.model or (PPL_DEV_MODEL if args.dev else PPL_REAL_MODEL)

    paths = generate(args.pool, args.metadata, out_dir, work_dir, model, args.dev,
                     dtype=args.dtype, max_len=args.max_len,
                     force_score=args.force_score, limit=args.limit)
    logger.info("Wrote %d IFD selection files to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()
