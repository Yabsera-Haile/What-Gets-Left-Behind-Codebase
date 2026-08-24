from __future__ import annotations

import argparse
import json

import pandas as pd

LOW_BUCKETS = [0, 1, 2]
CONTROL_BUCKET = 5
EXCLUDE_LOW = {"ary", "arz", "acm", "apc", "ars", "aeb", "ajp", "und", "zxx"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Define the Stage-B eval language set.")
    ap.add_argument("--metadata", default="audit/stageb/data/metadata_pilot.parquet")
    ap.add_argument("--belebele", default="audit/configs/belebele_languages.json")
    ap.add_argument("--output", default="audit/configs/eval_languages.json")
    ap.add_argument("--n_low", type=int, default=14, help="Max low-resource eval langs.")
    ap.add_argument("--min_count", type=int, default=10, help="Min pool count to include.")
    ap.add_argument("--n_controls", type=int, default=3, help="Bucket-5 controls beyond English.")
    args = ap.parse_args()

    df = pd.read_parquet(args.metadata)
    belebele = set(json.load(open(args.belebele, encoding="utf-8"))["iso3"])

    lang_bucket = (df.groupby("language")["resource_bucket"]
                   .agg(lambda s: int(s.mode().iloc[0])).to_dict())
    lang_count = df["language"].value_counts().to_dict()

    low = df[df["resource_bucket"].isin(LOW_BUCKETS)]
    low_counts = low["language"].value_counts()
    print("=== low-resource languages in buckets 0-2 (top 30; * = Belebele) ===")
    for lang, c in low_counts.head(30).items():
        star = "*" if lang in belebele else " "
        print(f"  {star} {lang:6} bucket{lang_bucket.get(lang,'?')}  n={c}")

    chosen_low = [lang for lang, c in low_counts.items()
                  if lang in belebele and lang not in EXCLUDE_LOW
                  and c >= args.min_count][: args.n_low]

    hi = df[df["resource_bucket"] == CONTROL_BUCKET]["language"].value_counts()
    controls = ["eng"]
    for lang, _ in hi.items():
        if len(controls) >= 1 + args.n_controls:
            break
        if lang != "eng" and lang in belebele:
            controls.append(lang)

    eval_langs = []
    for lang in chosen_low:
        eval_langs.append({"iso": lang, "resource_bucket": lang_bucket[lang],
                           "role": "low_resource", "pool_count": int(lang_count[lang])})
    for lang in controls:
        eval_langs.append({"iso": lang, "resource_bucket": int(lang_bucket.get(lang, 5)),
                           "role": "control", "pool_count": int(lang_count.get(lang, 0))})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"languages": eval_langs}, f, indent=2)

    print(f"\n=== chosen eval set ({len(eval_langs)} languages) -> {args.output} ===")
    for e in eval_langs:
        print(f"  {e['role']:12} {e['iso']:5} bucket{e['resource_bucket']}  pool_n={e['pool_count']}")

    n_low = sum(1 for e in eval_langs if e["role"] == "low_resource")
    n_ctrl = sum(1 for e in eval_langs if e["role"] == "control")
    all_bel = all(e["iso"] in belebele for e in eval_langs)
    low_in_023 = all(e["resource_bucket"] in LOW_BUCKETS
                     for e in eval_langs if e["role"] == "low_resource")
    ok = (12 <= len(eval_langs) <= 20 and n_low >= 8 and n_ctrl >= 3
          and all_bel and low_in_023 and "eng" in {e["iso"] for e in eval_langs})
    print(f"\nACCEPTANCE: {'PASS' if ok else 'FAIL'} — {n_low} low-resource (buckets 0-2) "
          f"+ {n_ctrl} controls; all Belebele-supported={all_bel}; English present.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
