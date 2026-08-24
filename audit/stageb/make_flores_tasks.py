from __future__ import annotations

import argparse
import glob
import json
import os

LANG = {
    "eng": ("eng_Latn", "English"), "spa": ("spa_Latn", "Spanish"),
    "cmn": ("zho_Hans", "Chinese"), "arb": ("arb_Arab", "Arabic"),
    "ceb": ("ceb_Latn", "Cebuano"), "plt": ("plt_Latn", "Malagasy"),
    "bod": ("bod_Tibt", "Tibetan"), "yor": ("yor_Latn", "Yoruba"),
    "tsn": ("tsn_Latn", "Tswana"), "som": ("som_Latn", "Somali"),
    "kir": ("kir_Cyrl", "Kyrgyz"), "fuv": ("fuv_Latn", "Fula"),
    "hau": ("hau_Latn", "Hausa"), "wol": ("wol_Latn", "Wolof"),
    "mlt": ("mlt_Latn", "Maltese"), "zul": ("zul_Latn", "Zulu"),
}
TASKS_DIR = "audit/configs/flores_tasks"


def task_name(src_iso: str, tgt_iso: str) -> str:
    return f"flores_{src_iso}_{tgt_iso}"


def yaml_for(src_iso: str, tgt_iso: str) -> str:
    src_code, src_name = LANG[src_iso]
    tgt_code, tgt_name = LANG[tgt_iso]
    doc_to_text = (f"Translate the following sentence from {src_name} to {tgt_name}.\\n"
                   f"{src_name}: {{{{sentence_{src_code}}}}}\\n{tgt_name}:")
    doc_to_target = f"{{{{sentence_{tgt_code}}}}}"
    return (
        f"task: {task_name(src_iso, tgt_iso)}\n"
        f"dataset_path: Muennighoff/flores200\n"
        f"dataset_name: all\n"
        f"dataset_kwargs:\n"
        f"  trust_remote_code: true    # Muennighoff/flores200 uses a loader script\n"
        f"test_split: devtest\n"
        f"output_type: generate_until\n"
        f'doc_to_text: "{doc_to_text}"\n'
        f'doc_to_target: "{doc_to_target}"\n'
        f"generation_kwargs:\n"
        f"  until:\n"
        f'    - "\\n"\n'
        f"  max_gen_toks: 256\n"
        f"  do_sample: false\n"
        f"metric_list:\n"
        f"  - metric: chrf                 # chrF++ via the custom aggregation below\n"
        f"    aggregation: !function utils.chrf_pp_agg\n"
        f"    higher_is_better: true\n"
        f"metadata:\n"
        f"  version: 1.0\n"
    )


def generate(target_isos: list[str]) -> list[str]:
    os.makedirs(TASKS_DIR, exist_ok=True)
    written = []
    for iso in target_isos:
        if iso == "eng" or iso not in LANG:
            continue
        for src, tgt in [("eng", iso), (iso, "eng")]:
            path = os.path.join(TASKS_DIR, f"{task_name(src, tgt)}.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_for(src, tgt))
            written.append(task_name(src, tgt))
    return written


def assemble(results_dir: str, condition: str, out_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(results_dir, "**", "results*.json"), recursive=True),
                   key=os.path.getmtime)
    if not files:
        raise SystemExit(f"No results*.json under {results_dir}")
    results = json.load(open(files[-1], encoding="utf-8")).get("results", {})
    langs: dict = {}
    for task, metrics in results.items():
        if not task.startswith("flores_"):
            continue
        _, src, tgt = task.split("_", 2)
        chrf = next((v for k, v in metrics.items()
                     if k.startswith("chrf") and isinstance(v, (int, float))), None)
        if chrf is None:
            continue
        if src == "eng":
            langs.setdefault(tgt, {})["eng_to_xx"] = chrf
        elif tgt == "eng":
            langs.setdefault(src, {})["xx_to_eng"] = chrf
    os.makedirs(out_dir, exist_ok=True)
    out = {"condition": condition, "metric": "chrf++", "languages": langs}
    path = os.path.join(out_dir, f"{condition}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(f"wrote {path}: {json.dumps(langs, indent=2)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate/assemble FLORES-200 chrF++ tasks.")
    ap.add_argument("--langs", nargs="*", default=["hau", "ceb", "zul"],
                    help="Target ISO codes; both eng<->iso directions are generated.")
    ap.add_argument("--assemble", default=None, help="lm-eval results dir to assemble from.")
    ap.add_argument("--condition", default="base")
    ap.add_argument("--out_dir",
                    default="audit/results/stageb/gemma-3-4b-pt/round4/metrics/flores")
    args = ap.parse_args()
    if args.assemble:
        assemble(args.assemble, args.condition, args.out_dir)
    else:
        names = generate(args.langs)
        print(f"wrote {len(names)} task YAMLs to {TASKS_DIR}/:")
        for n in names:
            print("  ", n)


if __name__ == "__main__":
    main()
