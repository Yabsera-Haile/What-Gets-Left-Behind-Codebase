from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger("audit.run_round4_eval")

LANGS = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
FLORES = {"ceb": "ceb_Latn", "hau": "hau_Latn", "kir": "kir_Cyrl", "mlt": "mlt_Latn",
          "plt": "plt_Latn", "som": "som_Latn", "zul": "zul_Latn"}


def conditions(ckpt_dir: str):
    out = [("base", None)]
    for d in sorted(glob.glob(os.path.join(ckpt_dir, "*__b*__s*"))):
        if os.path.exists(os.path.join(d, "adapter_config.json")):
            out.append((os.path.basename(d), d))
    return out


def parse_cond(cond: str):
    if cond == "base":
        return "base", 0.0, 0
    sel, rest = cond.split("__b", 1)
    b, s = rest.split("__s")
    return sel, float(b), int(s)


def run_lm_eval(margs, tasks, workdir, batch_size, limit, apply_ct, include_path=None,
                num_fewshot=None):
    os.makedirs(workdir, exist_ok=True)
    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", margs,
           "--tasks", ",".join(tasks), "--batch_size", str(batch_size),
           "--output_path", workdir]
    if include_path:
        cmd += ["--include_path", include_path]
    if apply_ct:
        cmd.append("--apply_chat_template")
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def eval_condition(cond, adapter, args):
    margs = (f"pretrained={args.base_model},dtype=bfloat16,"
             f"add_bos_token=True,attn_implementation=eager")
    if adapter is not None:
        margs += f",peft={adapter},tokenizer={adapter}"
    apply_ct = adapter is not None

    ppl_dir = os.path.join(args.out_root, "metrics", "heldout_ppl")
    if not os.path.exists(os.path.join(ppl_dir, f"{cond}.json")):
        cmd = [sys.executable, "-m", "audit.stageb.heldout_ppl", "--base", args.base_model,
               "--condition", cond, "--langs", *LANGS, "--out_dir", ppl_dir]
        if adapter is not None:
            cmd += ["--adapter", adapter]
        print("  $ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    work = os.path.join(args.out_root, "metrics", "lm_eval", cond)
    if not glob.glob(os.path.join(work, "belebele", "**", "results*.json"), recursive=True):
        run_lm_eval(margs, [f"belebele_{FLORES[l]}" for l in LANGS],
                    os.path.join(work, "belebele"), args.batch_size, 0, apply_ct)
    if not glob.glob(os.path.join(work, "flores", "**", "results*.json"), recursive=True):
        tasks = [f"flores_eng_{l}" for l in LANGS] + [f"flores_{l}_eng" for l in LANGS]
        run_lm_eval(margs, tasks, os.path.join(work, "flores"), args.batch_size,
                    args.flores_limit, apply_ct, include_path=args.flores_tasks)
    if cond == args.mmlu_on and not glob.glob(
            os.path.join(work, "mmlu", "**", "results*.json"), recursive=True):
        run_lm_eval(margs, ["mmlu"], os.path.join(work, "mmlu"), args.batch_size, 200,
                    apply_ct, num_fewshot=5)
    print(f"[{cond}] metrics done", flush=True)


def _latest(pattern_dir):
    files = sorted(glob.glob(os.path.join(pattern_dir, "**", "results*.json"), recursive=True),
                   key=os.path.getmtime)
    return json.load(open(files[-1], encoding="utf-8")).get("results", {}) if files else {}


def _num(d, *keys):
    for k in keys:
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    for k, v in d.items():
        if isinstance(v, (int, float)) and "stderr" not in k:
            return float(v)
    return None


def assemble(args):
    import pandas as pd
    rows = []
    for cond, adapter in conditions(args.ckpt_dir):
        sel, bud, seed = parse_cond(cond)
        ppl_path = os.path.join(args.out_root, "metrics", "heldout_ppl", f"{cond}.json")
        if os.path.exists(ppl_path):
            langs = json.load(open(ppl_path, encoding="utf-8"))["languages"]
            for l in LANGS:
                if langs.get(l, {}).get("ppl") is not None:
                    rows.append([cond, sel, bud, seed, l, "heldout_ppl", langs[l]["ppl"]])
                    rows.append([cond, sel, bud, seed, l, "heldout_nll", langs[l]["nll"]])
        work = os.path.join(args.out_root, "metrics", "lm_eval", cond)
        bel = _latest(os.path.join(work, "belebele"))
        for l in LANGS:
            v = _num(bel.get(f"belebele_{FLORES[l]}", {}), "acc,none", "acc")
            if v is not None:
                rows.append([cond, sel, bud, seed, l, "belebele", v])
        flo = _latest(os.path.join(work, "flores"))
        for l in LANGS:
            for task, metric in ((f"flores_eng_{l}", "chrf_eng_to_xx"),
                                 (f"flores_{l}_eng", "chrf_xx_to_eng")):
                v = _num(flo.get(task, {}), "chrf,none", "chrf")
                if v is not None:
                    rows.append([cond, sel, bud, seed, l, metric, v])
        mm = _num(_latest(os.path.join(work, "mmlu")).get("mmlu", {}), "acc,none", "acc")
        if mm is not None:
            rows.append([cond, sel, bud, seed, "mmlu", "mmlu", mm])
    df = pd.DataFrame(rows, columns=["condition", "selector", "budget", "seed", "group",
                                     "metric", "value"])
    out = os.path.join(args.out_root, "round4_results.parquet")
    os.makedirs(args.out_root, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"\nAssembled {len(df)} rows -> {out}")
    for metric, better in [("heldout_ppl", "lower"), ("chrf_eng_to_xx", "higher"),
                           ("belebele", "higher")]:
        sub = df[(df.metric == metric) & (df.group.isin(LANGS))]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="condition", columns="group", values="value")
        piv["MACRO"] = piv[LANGS].mean(axis=1)
        piv = piv.reindex(index=[c for c, _ in conditions(args.ckpt_dir)])
        print(f"\n=== {metric} ({better} is better) — decisive languages ===")
        print(piv.round(3).to_string())
    mm = df[df.metric == "mmlu"]
    if not mm.empty:
        print("\nMMLU (saturation reference):",
              {r.condition: round(r.value, 3) for r in mm.itertuples()})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Evaluate the Round-4 selection matrix.")
    ap.add_argument("--base_model", default="google/gemma-3-4b-pt")
    ap.add_argument("--out_root", default="audit/results/stageb/gemma-3-4b-pt/round4/eval")
    ap.add_argument("--ckpt_dir",
                    default="audit/results/stageb/gemma-3-4b-pt/round4/checkpoints")
    ap.add_argument("--flores_tasks", default="audit/configs/flores_tasks")
    ap.add_argument("--flores_limit", type=int, default=200)
    ap.add_argument("--mmlu_on", default="base", help="Condition to run MMLU on (once).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--assemble_only", action="store_true")
    ap.add_argument("--worker_cond", default=None)
    args = ap.parse_args()

    if args.assemble_only:
        assemble(args)
        return
    conds = conditions(args.ckpt_dir)
    if args.worker_cond is not None:
        for cond, adapter in conds:
            if cond == args.worker_cond:
                eval_condition(cond, adapter, args)
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    pending = [c[0] for c in conds]
    free, running = list(gpus), []
    log_dir = os.path.join(args.out_root, "eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.info("Round-4 eval: %d conditions x %d langs across GPUs %s (flores_limit=%d)",
                len(conds), len(LANGS), gpus, args.flores_limit)
    while pending or running:
        while free and pending:
            cond, gpu = pending.pop(0), free.pop(0)
            lf = open(os.path.join(log_dir, f"{cond}.log"), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            cmd = [sys.executable, "-m", "audit.experiments.run_round4_eval",
                   "--worker_cond", cond, "--base_model", args.base_model,
                   "--out_root", args.out_root, "--ckpt_dir", args.ckpt_dir,
                   "--flores_tasks", args.flores_tasks, "--flores_limit", str(args.flores_limit),
                   "--mmlu_on", args.mmlu_on, "--batch_size", str(args.batch_size)]
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running.append({"cond": cond, "gpu": gpu, "p": p, "lf": lf})
            logger.info("launch %-28s on GPU%s", cond, gpu)
        progressed, still = False, []
        for r in running:
            if r["p"].poll() is None:
                still.append(r)
                continue
            r["lf"].close()
            free.append(r["gpu"])
            progressed = True
            logger.info("done   %-28s %s", r["cond"],
                        "OK" if r["p"].returncode == 0 else f"FAILED({r['p'].returncode})")
        running = still
        if running and not progressed:
            time.sleep(5)
    assemble(args)


if __name__ == "__main__":
    main()
