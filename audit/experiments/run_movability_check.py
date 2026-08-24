from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

from audit.stageb.run_eval import run_lm_eval, pick_metric, get_metric

LOGLIK = {"mmlu", "mmlu_stem"}

BENCHMARKS = [
    ("gsm8k",     ["gsm8k"],     8,    "max_gen_toks=512", False, "math",
     "exact_match,strict-match"),
    ("mbpp",      ["mbpp"],      None, "max_gen_toks=256", True,  "code", None),
    ("humaneval", ["humaneval"], None, "max_gen_toks=512", True,  "code", None),
    ("ifeval",    ["ifeval"],    None, "max_gen_toks=512", False, "instruction_following", None),
    ("mmlu",      ["mmlu"],      5,    None,               False, "general", None),
    ("mmlu_stem", ["mmlu_stem"], 5,    None,               False, "science/stem", None),
]


def active_benchmarks(args):
    if not getattr(args, "benchmarks", None):
        return BENCHMARKS
    want = set(args.benchmarks)
    return [b for b in BENCHMARKS if b[0] in want]


def reference_models(ckpt_dir: str):
    refs = [("base", None)]
    for tag in ["full__s0", "random__s0", "random__s1", "random__s2"]:
        d = os.path.join(ckpt_dir, tag)
        if os.path.exists(os.path.join(d, "adapter_config.json")):
            refs.append((tag, d))
    return refs


def bench_metric(results: dict, name: str) -> float | None:
    if name == "gsm8k":
        return get_metric(results, "gsm8k", "exact_match,strict-match")
    if name == "mmlu_stem":
        return pick_metric(results, "mmlu_stem", "mmlu")
    return pick_metric(results, name, name)


def eval_model(tag, adapter, args):
    margs = (f"pretrained={args.base_model},dtype=bfloat16,"
             f"add_bos_token=True,attn_implementation=eager")
    if adapter is not None:
        margs += f",peft={adapter},tokenizer={adapter}"
    apply_ct = adapter is not None
    out_path = os.path.join(args.out_root, "metrics", f"{tag}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for name, tasks, nfs, genkw, codex, _skill, _mk in active_benchmarks(args):
        if bench_metric(results, name) is not None:
            print(f"[{tag}] {name}: present, skip", flush=True)
            continue
        bs = args.loglik_batch_size if name in LOGLIK else args.batch_size
        r = run_lm_eval(margs, tasks, nfs, genkw, codex,
                        os.path.join(args.out_root, "work", tag, name),
                        bs, args.limit, apply_ct)
        results.update(r)
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"[{tag}] {name}: saved", flush=True)
    print(f"[{tag}] done", flush=True)


def assemble(args):
    refs = reference_models(args.ckpt_dir)
    scores = {}
    for tag, _ in refs:
        p = os.path.join(args.out_root, "metrics", f"{tag}.json")
        if not os.path.exists(p):
            continue
        results = json.load(open(p, encoding="utf-8"))
        for name, *_ in active_benchmarks(args):
            v = bench_metric(results, name)
            if v is not None:
                scores.setdefault(name, {})[tag] = v

    rows, verdict = [], {}
    print(f"\n{'benchmark':11} {'skill':22} {'base':>7} {'full':>7} "
          f"{'rand_mean':>9} {'rand_sd':>8} {'spread':>7} {'noise':>7}  label")
    for name, tasks, nfs, genkw, codex, skill, _mk in active_benchmarks(args):
        s = scores.get(name, {})
        base = s.get("base")
        full = s.get("full__s0")
        rand = [s[t] for t in ("random__s0", "random__s1", "random__s2") if t in s]
        if base is None or full is None or not rand:
            print(f"{name:11} {skill:22} MISSING (base/full/random incomplete)")
            verdict[name] = {"skill": skill, "label": "INCOMPLETE"}
            continue
        rmean = statistics.mean(rand)
        rsd = statistics.pstdev(rand) if len(rand) > 1 else 0.0
        refs_vals = [base, full, rmean]
        spread = max(refs_vals) - min(refs_vals)
        noise = max(3 * rsd, 0.02)
        label = "movable" if spread > noise else "saturated"
        print(f"{name:11} {skill:22} {base:>7.3f} {full:>7.3f} {rmean:>9.3f} {rsd:>8.4f} "
              f"{spread:>7.3f} {noise:>7.3f}  {label.upper()}"
              + ("  (ratio-only)" if label == "saturated" else ""))
        verdict[name] = {"skill": skill, "base": base, "full": full,
                         "random_mean": rmean, "random_sd": rsd, "spread": spread,
                         "noise_band": noise, "label": label,
                         "reporting": "native-benchmark" if label == "movable"
                                      else "stage-a-representation-ratio-only"}
        rows.append([name, skill, base, full, rmean, rsd, spread, label])

    os.makedirs(args.out_root, exist_ok=True)
    json.dump({"base_model": args.base_model, "reference_models": [t for t, _ in refs],
               "limit": args.limit, "criterion": "spread > max(3*random_sd, 0.02)",
               "benchmarks": verdict},
              open(os.path.join(args.out_root, "movability.json"), "w"), indent=2)
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=["benchmark", "skill", "base", "full", "random_mean",
                                    "random_sd", "spread", "label"]).to_parquet(
            os.path.join(args.out_root, "movability.parquet"), index=False)
    except Exception as e:
        print(f"(parquet skipped: {e})")
    movable = [v["skill"] for v in verdict.values() if v.get("label") == "movable"]
    saturated = [v["skill"] for v in verdict.values() if v.get("label") == "saturated"]
    print(f"\nMOVABLE (necessity claims allowed):     {movable}")
    print(f"SATURATED (ratio-only reporting):       {saturated}")
    print("Multilingual: measured by held-out perplexity (movable, established elsewhere).")
    print(f"-> {os.path.join(args.out_root, 'movability.json')}")


def main():
    ap = argparse.ArgumentParser(description="Skill-benchmark movability pre-check (C2-Step 8).")
    ap.add_argument("--base_model", default="google/gemma-3-4b-pt")
    ap.add_argument("--ckpt_dir", default="audit/results/stagec/phase1/checkpoints")
    ap.add_argument("--out_root", default="audit/results/stagec/phase2/movability")
    ap.add_argument("--limit", type=int, default=200, help="Examples per task (0=full).")
    ap.add_argument("--batch_size", type=int, default=8, help="Generation tasks (gsm8k/mbpp/ifeval).")
    ap.add_argument("--loglik_batch_size", type=int, default=2,
                    help="MMLU/MMLU-STEM loglik batch (small: 262k-vocab log_softmax OOMs at 8).")
    ap.add_argument("--benchmarks", nargs="*", default=None,
                    help=f"Subset of {[b[0] for b in BENCHMARKS]} (default: all). "
                         "Part B code check: --benchmarks mbpp humaneval.")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--assemble_only", action="store_true")
    ap.add_argument("--worker_tag", default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.assemble_only:
        assemble(args)
        return
    refs = reference_models(args.ckpt_dir)
    if args.worker_tag is not None:
        for tag, adapter in refs:
            if tag == args.worker_tag:
                eval_model(tag, adapter, args)
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    log_dir = os.path.join(args.out_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    active = active_benchmarks(args)
    print(f"Movability check: {len(refs)} reference models "
          f"{[t for t, _ in refs]} x {[b[0] for b in active]} on {gpus}")
    pending, running = [t for t, _ in refs], []
    free = list(gpus)
    import time
    while pending or running:
        while free and pending:
            tag, gpu = pending.pop(0), free.pop(0)
            lf = open(os.path.join(log_dir, f"{tag}.log"), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            cmd = [sys.executable, "-m", "audit.experiments.run_movability_check",
                   "--worker_tag", tag, "--base_model", args.base_model,
                   "--ckpt_dir", args.ckpt_dir, "--out_root", args.out_root,
                   "--limit", str(args.limit), "--batch_size", str(args.batch_size),
                   "--loglik_batch_size", str(args.loglik_batch_size)]
            if args.benchmarks:
                cmd += ["--benchmarks", *args.benchmarks]
            running.append({"tag": tag, "gpu": gpu, "lf": lf,
                            "p": subprocess.Popen(cmd, env=env, stdout=lf,
                                                  stderr=subprocess.STDOUT)})
            print(f"launch {tag:14} on GPU{gpu}")
        prog, still = False, []
        for r in running:
            if r["p"].poll() is None:
                still.append(r)
                continue
            r["lf"].close()
            free.append(r["gpu"])
            prog = True
            print(f"done   {r['tag']:14} {'OK' if r['p'].returncode == 0 else 'FAILED'}")
        running = still
        if running and not prog:
            time.sleep(5)
    assemble(args)


if __name__ == "__main__":
    main()
