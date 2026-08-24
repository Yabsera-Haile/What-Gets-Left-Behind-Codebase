from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time

from audit.experiments.run_movability_check import BENCHMARKS, LOGLIK, bench_metric
from audit.stageb.run_eval import run_lm_eval

BENCH_BY_NAME = {b[0]: b for b in BENCHMARKS}


def cells(ckpt_dir):
    out = [("base", None)]
    for d in sorted(glob.glob(os.path.join(ckpt_dir, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "adapter_config.json")):
            out.append((os.path.basename(d), d))
    return out


def parse_cell(tag):
    if tag == "base":
        return "base", 0
    body, _, seed = tag.rpartition("__s")
    mode = body.split("__", 1)[1] if "__" in body else body
    return mode, (int(seed) if seed.isdigit() else 0)


def eval_cell(tag, adapter, args):
    extra = args.model_args_extra
    margs = f"pretrained={args.base_model},dtype=bfloat16" + (f",{extra}" if extra else "")
    if adapter is not None:
        margs += f",peft={adapter},tokenizer={adapter}"
    apply_ct = adapter is not None
    out_path = os.path.join(args.out_root, "metrics", f"{tag}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for name in args.benchmarks:
        _, tasks, nfs, genkw, codex, _skill, _mk = BENCH_BY_NAME[name]
        if bench_metric(results, name) is not None:
            print(f"[{tag}] {name}: present, skip", flush=True)
            continue
        bs = args.loglik_batch_size if name in LOGLIK else args.batch_size
        r = run_lm_eval(margs, tasks, nfs, genkw, codex,
                        os.path.join(args.out_root, "work", tag, name), bs, args.limit, apply_ct)
        results.update(r)
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"[{tag}] {name}: saved", flush=True)
    print(f"[{tag}] done", flush=True)


def assemble(args):
    import pandas as pd
    rows = []
    for tag, _ in cells(args.ckpt_dir):
        p = os.path.join(args.out_root, "metrics", f"{tag}.json")
        if not os.path.exists(p):
            continue
        results = json.load(open(p, encoding="utf-8"))
        mode, seed = parse_cell(tag)
        for name in args.benchmarks:
            v = bench_metric(results, name)
            if v is not None:
                rows.append([tag, mode, seed, name, v])
    df = pd.DataFrame(rows, columns=["cell", "mode", "seed", "benchmark", "value"])
    os.makedirs(args.out_root, exist_ok=True)
    df.to_parquet(os.path.join(args.out_root, "skill_cells_results.parquet"), index=False)
    print(f"\nAssembled {len(df)} rows -> {args.out_root}/skill_cells_results.parquet")
    order = ["base", "none", "proportional", "absolute", "hybrid", "random", "full"]
    for name in args.benchmarks:
        sub = df[df.benchmark == name]
        if sub.empty:
            continue
        print(f"\n=== {name} (by floor mode; mean +- SD across seeds) ===")
        g = sub.groupby("mode")["value"]
        stat = {m: (round(g.get_group(m).mean(), 4),
                    round(statistics.pstdev(list(g.get_group(m))), 4) if len(g.get_group(m)) > 1 else 0.0)
                for m in g.groups}
        for m in [x for x in order if x in stat] + [x for x in stat if x not in order]:
            print(f"  {m:14} {stat[m][0]:.4f} ± {stat[m][1]:.4f}  (n={len(sub[sub['mode']==m])})")


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained cells on skill benchmarks.")
    ap.add_argument("--base_model", default="google/gemma-3-4b-pt")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=["ifeval", "mmlu"],
                    help="Subset of: gsm8k mbpp ifeval mmlu mmlu_stem")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--loglik_batch_size", type=int, default=2)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--model_args_extra", default=None,
                    help="Extra key=val,... for lm-eval --model_args. Default: Gemma gets "
                         "'add_bos_token=True,attn_implementation=eager', other families get ''.")
    ap.add_argument("--assemble_only", action="store_true")
    ap.add_argument("--worker_tag", default=None)
    args = ap.parse_args()
    if args.model_args_extra is None:
        args.model_args_extra = ("add_bos_token=True,attn_implementation=eager"
                                 if "gemma" in args.base_model.lower() else "")
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    bad = [b for b in args.benchmarks if b not in BENCH_BY_NAME]
    if bad:
        raise SystemExit(f"unknown benchmarks {bad}; choose from {list(BENCH_BY_NAME)}")

    if args.assemble_only:
        assemble(args)
        return
    todo = cells(args.ckpt_dir)
    if args.worker_tag is not None:
        for tag, adapter in todo:
            if tag == args.worker_tag:
                eval_cell(tag, adapter, args)
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    log_dir = os.path.join(args.out_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Skill-cell eval: {len(todo)} cells x {args.benchmarks} on {gpus}")
    pending, running, free = [t for t, _ in todo], [], list(gpus)
    while pending or running:
        while free and pending:
            tag, gpu = pending.pop(0), free.pop(0)
            lf = open(os.path.join(log_dir, f"{tag}.log"), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            cmd = [sys.executable, "-m", "audit.experiments.run_skill_cells_eval",
                   "--worker_tag", tag, "--base_model", args.base_model,
                   "--ckpt_dir", args.ckpt_dir, "--out_root", args.out_root,
                   "--benchmarks", *args.benchmarks, "--limit", str(args.limit),
                   "--batch_size", str(args.batch_size),
                   "--loglik_batch_size", str(args.loglik_batch_size),
                   "--model_args_extra", args.model_args_extra]
            running.append({"tag": tag, "gpu": gpu, "lf": lf,
                            "p": subprocess.Popen(cmd, env=env, stdout=lf,
                                                  stderr=subprocess.STDOUT)})
            print(f"launch {tag:34} on GPU{gpu}")
        prog, still = False, []
        for r in running:
            if r["p"].poll() is None:
                still.append(r); continue
            r["lf"].close(); free.append(r["gpu"]); prog = True
            print(f"done   {r['tag']:34} {'OK' if r['p'].returncode == 0 else 'FAILED'}")
        running = still
        if running and not prog:
            time.sleep(5)
    assemble(args)


if __name__ == "__main__":
    main()
