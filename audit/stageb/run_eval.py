from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_MODEL = "Qwen/Qwen2.5-7B"

ISO_TO_FLORES = {
    "eng": "eng_Latn", "ceb": "ceb_Latn", "plt": "plt_Latn", "bod": "bod_Tibt",
    "yor": "yor_Latn", "tsn": "tsn_Latn", "som": "som_Latn", "kir": "kir_Cyrl",
    "fuv": "fuv_Latn", "hau": "hau_Latn", "wol": "wol_Latn", "mlt": "mlt_Latn",
    "zul": "zul_Latn", "spa": "spa_Latn", "cmn": "zho_Hans", "arb": "arb_Arab",
}
FLORES_TO_ISO = {v: k for k, v in ISO_TO_FLORES.items()}
SKILL_OF = {"gsm8k": "math", "mmlu": "general", "mbpp": "code",
            "ifeval": "instruction_following"}
METRIC_CANDIDATES = {
    "belebele": ["acc,none", "acc_norm,none", "acc"],
    "mmlu": ["acc,none", "acc"],
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract",
              "exact_match,none", "exact_match"],
    "mbpp": ["pass_at_1,none", "pass@1,none", "pass_at_1", "acc,none"],
    "humaneval": ["pass_at_1,none", "pass@1,none", "pass@1,create_test", "pass_at_1", "pass@1"],
    "ifeval": ["prompt_level_strict_acc,none", "inst_level_strict_acc,none",
               "prompt_level_loose_acc,none"],
}

CHANCE = 0.25
ABOVE_CHANCE_Z = 1.645


def group_calls(group: str, belebele_tasks: list[str], full_langs: set, default_limit: int):
    if group == "fast":
        full = [t for t in belebele_tasks if FLORES_TO_ISO.get(t[len("belebele_"):]) in full_langs]
        rest = [t for t in belebele_tasks if t not in full]
        calls = []
        if full:
            calls.append(("belebele_full", full, None, None, False, 0))
        if rest:
            calls.append(("belebele", rest, None, None, False, default_limit))
        calls.append(("mmlu", ["mmlu"], 5, None, False, default_limit))
        return calls
    return [
        ("ifeval", ["ifeval"], None, "max_gen_toks=512", False, default_limit),
        ("mbpp", ["mbpp"], None, "max_gen_toks=256", True, default_limit),
        ("gsm8k", ["gsm8k"], 8, "max_gen_toks=512", False, default_limit),
    ]


def get_metric(results: dict, task: str, key: str):
    v = results.get(task, {}).get(key)
    return float(v) if isinstance(v, (int, float)) else None


def pick_metric(results: dict, task_key: str, kind: str):
    r = results.get(task_key, {})
    for c in METRIC_CANDIDATES.get(kind, []):
        if c in r and isinstance(r[c], (int, float)):
            return float(r[c])
    for k, v in r.items():
        if isinstance(v, (int, float)) and "stderr" not in k:
            return float(v)
    return None


def run_lm_eval(model_args, tasks, num_fewshot, gen_kwargs, code_exec, workdir,
                batch_size, limit, apply_ct, model_type="hf") -> dict:
    os.makedirs(workdir, exist_ok=True)
    cmd = [sys.executable, "-m", "lm_eval", "--model", model_type, "--model_args", model_args,
           "--tasks", ",".join(tasks), "--batch_size", str(batch_size),
           "--output_path", workdir]
    if apply_ct:
        cmd.append("--apply_chat_template")
    if code_exec:
        cmd.append("--confirm_run_unsafe_code")
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if gen_kwargs:
        cmd += ["--gen_kwargs", gen_kwargs]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    files = sorted(glob.glob(os.path.join(workdir, "**", "results*.json"), recursive=True),
                   key=os.path.getmtime)
    if not files:
        raise RuntimeError(f"No results json under {workdir}")
    return json.load(open(files[-1], encoding="utf-8")).get("results", {})


def have_all(results: dict, tasks: list[str]) -> bool:
    for t in tasks:
        kind = "belebele" if t.startswith("belebele") else t
        if pick_metric(results, t, kind) is None:
            return False
    return True


def enumerate_models(ckpt_dir: str):
    models = [("base", 0.0, None)]
    for d in sorted(glob.glob(os.path.join(ckpt_dir, "*__b*"))):
        if os.path.isdir(d):
            cond, _, budget = os.path.basename(d).rpartition("__b")
            models.append((cond, float(budget), d))
    return models


def tag_of(cond, budget, adapter):
    return "base" if adapter is None else f"{cond}__b{budget}"


def eval_one(cond, budget, adapter, group, out_dir, work_dir, base_model,
             batch_size, limit, apply_ct, model_type="hf", extra_args="", full_langs=()):
    tag = tag_of(cond, budget, adapter)
    margs = f"pretrained={base_model},dtype=bfloat16"
    if adapter is not None:
        margs += f",peft={adapter},tokenizer={adapter}"
    if extra_args:
        margs += f",{extra_args}"
    langs = json.load(open("audit/configs/eval_languages.json", encoding="utf-8"))["languages"]
    belebele_tasks = [f"belebele_{ISO_TO_FLORES[e['iso']]}" for e in langs
                      if e["iso"] in ISO_TO_FLORES]
    path = os.path.join(out_dir, f"{tag}.json")
    os.makedirs(out_dir, exist_ok=True)
    results = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for name, tasks, nfs, genkw, codex, call_limit in group_calls(
            group, belebele_tasks, set(full_langs), limit):
        if have_all(results, tasks):
            print(f"[{tag}] {name}: already present, skipping", flush=True)
            continue
        r = run_lm_eval(margs, tasks, nfs, genkw, codex,
                        os.path.join(work_dir, tag, name), batch_size, call_limit, apply_ct,
                        model_type)
        results.update(r)
        json.dump(results, open(path, "w"), indent=2)
        print(f"[{tag}] {name}: saved -> {path}", flush=True)
    print(f"[{tag}] {group} group complete", flush=True)


def belebele_acc_stderr(results: dict, task_key: str):
    r = results.get(task_key, {})
    for base_key in ("acc", "acc_norm"):
        if isinstance(r.get(f"{base_key},none"), (int, float)):
            se = r.get(f"{base_key}_stderr,none")
            return float(r[f"{base_key},none"]), (float(se) if isinstance(se, (int, float)) else None)
    v = pick_metric(results, task_key, "belebele")
    return (v, None) if v is not None else (None, None)


def above_chance_set(out_dir: str, langs: list[dict], role: dict) -> set:
    base_path = os.path.join(out_dir, "base.json")
    low_res, control, keep = [], [], set()
    if os.path.exists(base_path):
        results = json.load(open(base_path, encoding="utf-8"))
        for e in langs:
            iso = e["iso"]
            acc, se = belebele_acc_stderr(results, f"belebele_{ISO_TO_FLORES.get(iso, '')}")
            if acc is None:
                continue
            lb = acc - ABOVE_CHANCE_Z * se if se is not None else acc
            passed = lb > CHANCE
            if passed:
                keep.add(iso)
            d = {"iso": iso, "base_acc": round(acc, 4),
                 "base_stderr": (round(se, 4) if se is not None else None),
                 "lower_bound": round(lb, 4), "above_chance": passed}
            (low_res if role.get(iso) == "low_resource" else control).append(d)
    lr_pass = sorted(d["iso"] for d in low_res if d["above_chance"])
    ctrl_pass = sorted(d["iso"] for d in control if d["above_chance"])
    slug = os.path.basename(os.path.dirname(out_dir)) or "model"
    rec = {"model_slug": slug,
           "rule": f"base Belebele acc - {ABOVE_CHANCE_Z}*stderr > {CHANCE} (one-sided 95%)",
           "chance": CHANCE, "z": ABOVE_CHANCE_Z,
           "low_resource": {"n_above_chance": len(lr_pass), "above_chance_isos": lr_pass,
                            "languages": low_res},
           "control": {"n_above_chance": len(ctrl_pass), "above_chance_isos": ctrl_pass,
                       "languages": control}}
    with open(os.path.join(os.path.dirname(out_dir), f"abovechance_{slug}.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"\n[GATE {slug}] above-chance LOW-RESOURCE ({len(lr_pass)}): {lr_pass or 'NONE'}")
    print(f"[GATE {slug}] above-chance CONTROL     ({len(ctrl_pass)}): {ctrl_pass or 'NONE'}")
    return keep


def assemble_parquet(out_dir: str, eval_langs: str):
    import pandas as pd
    langs = json.load(open(eval_langs, encoding="utf-8"))["languages"]
    role = {e["iso"]: e["role"] for e in langs}
    keep = above_chance_set(out_dir, langs, role)
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        tag = os.path.splitext(os.path.basename(path))[0]
        cond, _, budget = tag.rpartition("__b")
        if tag == "base":
            cond, budget = "base", 0.0
        budget = float(budget)
        results = json.load(open(path, encoding="utf-8"))
        lowres, controls, abovechance, lowres_ac = [], [], [], []
        for e in langs:
            iso = e["iso"]
            score = pick_metric(results, f"belebele_{ISO_TO_FLORES.get(iso, '')}", "belebele")
            if score is None:
                continue
            rows.append([cond, budget, "belebele", "language", iso, score])
            r = role.get(iso)
            (lowres if r == "low_resource" else controls).append(score)
            if iso in keep:
                abovechance.append(score)
                if r == "low_resource":
                    lowres_ac.append(score)
        skill_scores = []
        for task, skill in SKILL_OF.items():
            if task == "gsm8k":
                strict = get_metric(results, "gsm8k", "exact_match,strict-match")
                flex = get_metric(results, "gsm8k", "exact_match,flexible-extract")
                if strict is not None:
                    rows.append([cond, budget, "gsm8k", "skill", "math_strict", strict])
                    skill_scores.append(strict)
                if flex is not None:
                    rows.append([cond, budget, "gsm8k", "skill", "math_flexible", flex])
                continue
            score = pick_metric(results, task, task)
            if score is not None:
                rows.append([cond, budget, task, "skill", skill, score])
                skill_scores.append(score)
        mmlu = pick_metric(results, "mmlu", "mmlu")
        if mmlu is not None:
            rows.append([cond, budget, "mmlu", "average", "mmlu_overall", mmlu])
        if skill_scores:
            rows.append([cond, budget, "skill_macro", "average", "skill_macro",
                         sum(skill_scores) / len(skill_scores)])
        if lowres:
            rows.append([cond, budget, "belebele", "average", "lowres_macro",
                         sum(lowres) / len(lowres)])
        if controls:
            rows.append([cond, budget, "belebele", "average", "control_macro",
                         sum(controls) / len(controls)])
        if abovechance:
            rows.append([cond, budget, "belebele", "average", "abovechance_macro",
                         sum(abovechance) / len(abovechance)])
        if lowres_ac:
            rows.append([cond, budget, "belebele", "average", "lowres_abovechance_macro",
                         sum(lowres_ac) / len(lowres_ac)])
    df = pd.DataFrame(rows, columns=["condition", "budget", "task", "group_type",
                                     "group", "score"])
    out = os.path.join(os.path.dirname(out_dir), "stageb_results.parquet")
    df.to_parquet(out, index=False)
    ac_low = sorted(iso for iso in keep if role.get(iso) == "low_resource")
    print(f"\nAbove-chance LOW-RESOURCE languages (decisive set): {ac_low or 'NONE'}")
    print(f"Assembled {len(df)} rows from {len(glob.glob(os.path.join(out_dir,'*.json')))} "
          f"models -> {out}")
    if ac_low:
        sub = df[(df.group_type == "language") & (df.group.isin(ac_low))]
        piv = sub.pivot_table(index=["condition", "budget"], columns="group", values="score")
        print("\n=== DECISIVE: per-language Belebele on base-above-chance LOW-RESOURCE langs ===")
        print(piv.round(3).to_string())
    print("\n=== Macros (lowres_abovechance_macro = decisive) ===")
    print(df[df.group_type == "average"].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-B two-wave evaluation (lm-eval).")
    ap.add_argument("--group", choices=["fast", "slow"], required=True)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Condition names to eval (e.g. perplexity-low quality random).")
    ap.add_argument("--ckpt_dir", default="audit/results/stageb/checkpoints")
    ap.add_argument("--eval_langs", default="audit/configs/eval_languages.json")
    ap.add_argument("--out_dir", default="audit/results/stageb/eval")
    ap.add_argument("--work_dir", default="audit/results/stageb/eval_work")
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--model_type", default="hf",
                    help="lm-eval --model backend: 'hf' (CausalLM) or 'hf-multimodal' "
                         "(vision-text Gemma-3 4B/12B).")
    ap.add_argument("--model_args_extra", default=None,
                    help="Extra key=val,... appended to lm-eval --model_args. Default: auto "
                         "'add_bos_token=True,attn_implementation=eager' for Gemma (Gemma is "
                         "at chance without BOS), '' otherwise. Pass '' to disable.")
    ap.add_argument("--limit", type=int, default=200, help="Examples per task (0=full).")
    ap.add_argument("--full_langs", nargs="*", default=[],
                    help="ISO codes evaluated on FULL Belebele (no limit); others use --limit. "
                         "Round 3: the gate's above-chance low-resource languages.")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Default 16 (fast) / 8 (slow).")
    ap.add_argument("--gpus", default="0,1,2", help="GPUs to distribute models across.")
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--assemble_only", action="store_true")
    ap.add_argument("--worker_tags", nargs="*", default=None)
    args = ap.parse_args()

    batch = args.batch_size or (16 if args.group == "fast" else 8)
    apply_ct = not args.no_chat_template
    extra = args.model_args_extra
    if extra is None:
        extra = ("add_bos_token=True,attn_implementation=eager"
                 if "gemma" in args.base_model.lower() else "")

    if args.assemble_only:
        assemble_parquet(args.out_dir, args.eval_langs)
        return

    all_models = enumerate_models(args.ckpt_dir)
    if args.worker_tags is not None:
        want = set(args.worker_tags)
        for cond, budget, adapter in all_models:
            if tag_of(cond, budget, adapter) in want:
                eval_one(cond, budget, adapter, args.group, args.out_dir, args.work_dir,
                         args.base_model, batch, args.limit, apply_ct, args.model_type, extra,
                         args.full_langs)
        return

    models = [m for m in all_models if args.conditions is None or m[0] in args.conditions]
    tags = [tag_of(*m) for m in models]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    buckets = {g: [] for g in gpus}
    for i, t in enumerate(tags):
        buckets[gpus[i % len(gpus)]].append(t)
    print(f"Group={args.group} batch={batch} limit={args.limit or 'full'} chat={apply_ct}")
    print("GPU assignment:", {g: b for g, b in buckets.items() if b})

    os.makedirs(args.work_dir, exist_ok=True)
    procs = []
    for g, btags in buckets.items():
        if not btags:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        cmd = [sys.executable, "-m", "audit.stageb.run_eval", "--group", args.group,
               "--worker_tags", *btags, "--limit", str(args.limit),
               "--batch_size", str(batch), "--base_model", args.base_model,
               "--model_type", args.model_type, "--model_args_extra", extra,
               "--ckpt_dir", args.ckpt_dir, "--out_dir", args.out_dir,
               "--work_dir", args.work_dir]
        if args.full_langs:
            cmd += ["--full_langs", *args.full_langs]
        if args.no_chat_template:
            cmd.append("--no_chat_template")
        logf = open(os.path.join(args.work_dir, f"worker_gpu{g}.log"), "w")
        print(f"  launching GPU{g}: {btags}  (log: {logf.name})")
        procs.append((g, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)))
    failed = [g for g, p in procs if p.wait() != 0]
    if failed:
        print(f"!! workers on GPUs {failed} exited non-zero — check their logs in {args.work_dir}")
    assemble_parquet(args.out_dir, args.eval_langs)


if __name__ == "__main__":
    main()
