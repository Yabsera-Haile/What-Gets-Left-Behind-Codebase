from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger("audit.run_stagec_eval")

DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
CONTROL = ["eng", "spa", "cmn", "arb"]
FLORES = {"ceb": "ceb_Latn", "hau": "hau_Latn", "kir": "kir_Cyrl", "mlt": "mlt_Latn",
          "plt": "plt_Latn", "som": "som_Latn", "zul": "zul_Latn",
          "eng": "eng_Latn", "spa": "spa_Latn", "cmn": "zho_Hans", "arb": "arb_Arab"}
AXIS = {**{l: "help" for l in DECISIVE}, **{l: "cost" for l in CONTROL}}


def conditions(ckpt_dir: str):
    out = [("base", None)]
    for d in sorted(glob.glob(os.path.join(ckpt_dir, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "adapter_config.json")):
            out.append((os.path.basename(d), d))
    return out


def parse_cond(cond: str):
    if cond == "base":
        return "base", "base", 0
    body, seed = (cond.rsplit("__s", 1) + ["0"])[:2]
    seed = int(seed) if seed.isdigit() else 0
    if "__" in body:
        base_sel, floor = body.split("__", 1)
    else:
        base_sel = floor = body
    return base_sel, floor, seed


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
    all_langs = DECISIVE + CONTROL

    ppl_dir = os.path.join(args.out_root, "metrics", "heldout_ppl")
    if not os.path.exists(os.path.join(ppl_dir, f"{cond}.json")):
        cmd = [sys.executable, "-m", "audit.stageb.heldout_ppl", "--base", args.base_model,
               "--condition", cond, "--langs", *all_langs, "--out_dir", ppl_dir]
        if adapter is not None:
            cmd += ["--adapter", adapter]
        print("  $ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    if getattr(args, "ppl_only", False):
        print(f"[{cond}] ppl-only: skipped lm-eval (belebele/flores/mmlu)", flush=True)
        return
    work = os.path.join(args.out_root, "metrics", "lm_eval", cond)
    if not glob.glob(os.path.join(work, "belebele", "**", "results*.json"), recursive=True):
        run_lm_eval(margs, [f"belebele_{FLORES[l]}" for l in all_langs],
                    os.path.join(work, "belebele"), args.batch_size, 0, apply_ct)
    if not glob.glob(os.path.join(work, "flores", "**", "results*.json"), recursive=True):
        tasks = [f"flores_eng_{l}" for l in DECISIVE] + [f"flores_{l}_eng" for l in DECISIVE]
        run_lm_eval(margs, tasks, os.path.join(work, "flores"), args.batch_size,
                    args.flores_limit, apply_ct, include_path=args.flores_tasks)
    if not glob.glob(os.path.join(work, "mmlu", "**", "results*.json"), recursive=True):
        run_lm_eval(margs, ["mmlu"], os.path.join(work, "mmlu"), args.batch_size,
                    args.mmlu_limit, apply_ct, num_fewshot=5)
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
    all_langs = DECISIVE + CONTROL
    rows = []
    for cond, adapter in conditions(args.ckpt_dir):
        sel, floor, seed = parse_cond(cond)

        def add(group, metric, value):
            if value is not None:
                axis = AXIS.get(group, "cost")
                rows.append([cond, sel, floor, seed, axis, group, metric, value])

        ppl_path = os.path.join(args.out_root, "metrics", "heldout_ppl", f"{cond}.json")
        if os.path.exists(ppl_path):
            langs = json.load(open(ppl_path, encoding="utf-8"))["languages"]
            for l in all_langs:
                if langs.get(l, {}).get("ppl") is not None:
                    add(l, "heldout_ppl", langs[l]["ppl"])
                    add(l, "heldout_nll", langs[l]["nll"])
        work = os.path.join(args.out_root, "metrics", "lm_eval", cond)
        bel = _latest(os.path.join(work, "belebele"))
        for l in all_langs:
            add(l, "belebele", _num(bel.get(f"belebele_{FLORES[l]}", {}), "acc,none", "acc"))
        flo = _latest(os.path.join(work, "flores"))
        for l in DECISIVE:
            for task, metric in ((f"flores_eng_{l}", "chrf_eng_to_xx"),
                                 (f"flores_{l}_eng", "chrf_xx_to_eng")):
                add(l, metric, _num(flo.get(task, {}), "chrf,none", "chrf"))
        mm = _num(_latest(os.path.join(work, "mmlu")).get("mmlu", {}), "acc,none", "acc")
        if mm is not None:
            rows.append([cond, sel, floor, seed, "cost", "mmlu", "mmlu", mm])

    df = pd.DataFrame(rows, columns=["condition", "base_selector", "floor", "seed", "axis",
                                     "group", "metric", "value"])
    out = os.path.join(args.out_root, "results.parquet")
    os.makedirs(args.out_root, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nAssembled {len(df)} rows -> {out}")

    order = [c for c, _ in conditions(args.ckpt_dir)]

    def macro(metric, langs):
        sub = df[(df.metric == metric) & (df.group.isin(langs))]
        if sub.empty:
            return None
        piv = sub.pivot_table(index="condition", columns="group", values="value")
        piv["MACRO"] = piv[[l for l in langs if l in piv.columns]].mean(axis=1)
        return piv.reindex(index=order)

    print("\n" + "=" * 96)
    print("AXIS 1 — HELP (decisive languages): held-out FLORES perplexity (PRIMARY, lower=better)")
    print("=" * 96)
    p1 = macro("heldout_ppl", DECISIVE)
    if p1 is not None:
        print(p1.round(2).to_string())
    for metric, name in (("chrf_eng_to_xx", "chrF++ eng->xx (higher=better)"),
                         ("belebele", "Belebele (higher=better, expected flat)")):
        m = macro(metric, DECISIVE)
        if m is not None:
            print(f"\n--- Axis-1 secondary: {name} — decisive MACRO ---")
            print(m[["MACRO"]].round(3).to_string())

    print("\n" + "=" * 96)
    print("AXIS 2 — COST (preserved capability): MMLU + control langs (eng spa cmn arb)")
    print("=" * 96)
    mm = df[df.metric == "mmlu"].set_index("condition")["value"].reindex(order)
    cp = macro("heldout_ppl", CONTROL)
    cb = macro("belebele", CONTROL)
    cost = pd.DataFrame(index=order)
    cost["MMLU(5s)"] = mm.round(3)
    if cp is not None:
        cost["control_ppl_MACRO"] = cp["MACRO"].round(2)
    if cb is not None:
        cost["control_belebele_MACRO"] = cb["MACRO"].round(3)
    print(cost.to_string())

    print("\n" + "=" * 96)
    have_help = set(df[(df.axis == "help") & (df.metric == "heldout_ppl")].condition)
    have_cost = set(df[df.metric == "mmlu"].condition) | set(
        df[(df.axis == "cost") & (df.metric == "heldout_ppl")].condition)
    missing = [c for c in order if c not in (have_help & have_cost)]
    print(f"Coverage: {len(have_help & have_cost)}/{len(order)} cells have BOTH axes"
          + ("" if not missing else f" | MISSING: {missing}"))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Two-axis eval of the Stage-C variant matrix.")
    ap.add_argument("--base_model", default="google/gemma-3-4b-pt")
    ap.add_argument("--out_root", default="audit/results/stagec/phase1/eval")
    ap.add_argument("--ckpt_dir", default="audit/results/stagec/phase1/checkpoints")
    ap.add_argument("--flores_tasks", default="audit/configs/flores_tasks")
    ap.add_argument("--flores_limit", type=int, default=200)
    ap.add_argument("--mmlu_limit", type=int, default=100,
                    help="MMLU examples per subject (5-shot). 100 keeps 18 cells tractable.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--ppl_only", action="store_true",
                    help="Only held-out perplexity (decisive + control); skip belebele/flores/mmlu.")
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
    logger.info("Stage-C eval: %d conditions | axis1=%d decisive, axis2=%d control + MMLU | "
                "GPUs %s (flores_limit=%d, mmlu_limit=%d)", len(conds), len(DECISIVE),
                len(CONTROL), gpus, args.flores_limit, args.mmlu_limit)
    while pending or running:
        while free and pending:
            cond, gpu = pending.pop(0), free.pop(0)
            lf = open(os.path.join(log_dir, f"{cond}.log"), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            cmd = [sys.executable, "-m", "audit.experiments.run_stagec_eval",
                   "--worker_cond", cond, "--base_model", args.base_model,
                   "--out_root", args.out_root, "--ckpt_dir", args.ckpt_dir,
                   "--flores_tasks", args.flores_tasks, "--flores_limit", str(args.flores_limit),
                   "--mmlu_limit", str(args.mmlu_limit), "--batch_size", str(args.batch_size)]
            if args.ppl_only:
                cmd.append("--ppl_only")
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running.append({"cond": cond, "gpu": gpu, "p": p, "lf": lf})
            logger.info("launch %-40s on GPU%s", cond, gpu)
        progressed, still = False, []
        for r in running:
            if r["p"].poll() is None:
                still.append(r)
                continue
            r["lf"].close()
            free.append(r["gpu"])
            progressed = True
            logger.info("done   %-40s %s", r["cond"],
                        "OK" if r["p"].returncode == 0 else f"FAILED({r['p'].returncode})")
        running = still
        if running and not progressed:
            time.sleep(5)
    assemble(args)


if __name__ == "__main__":
    main()
