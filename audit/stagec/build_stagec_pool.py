from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random

os.environ.setdefault("USE_TORCH", "0")

DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
MURI_CONFIG = {"ceb": "ceb", "hau": "hau", "kir": "kir", "mlt": "mlt",
               "plt": "mlg", "som": "som", "zul": "zul"}
N_ABS = 500
PILOT_THRESHOLD = 250
N_INJECT = 1100
BUDGET_B = 0.10
NOISE_FRAC = 0.30


def mint_muri_id(inp, out):
    return "muri:" + hashlib.sha256((inp + "\n\n" + out).encode("utf-8")).hexdigest()[:24]


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_base(sample_size, seed, out_dir):
    from audit.metadata import pool_io
    from audit.metadata.build_metadata import ResourceMapper, SkillMapper, instruction_text
    from audit.metadata.langid import LanguageIdentifier
    rm, sm = ResourceMapper("audit/configs/joshi_resource.json"), SkillMapper("audit/configs/dataset_to_skill.json")
    lid = LanguageIdentifier(backend="glotlid")
    raw_path = os.path.join(out_dir, f"tulu_base_{sample_size}.jsonl")
    pool_io.materialize_pilot_pool("allenai/tulu-3-sft-mixture", raw_path, sample_size,
                                   seed, strategy="reservoir")
    raw = pool_io.read_pool(raw_path)
    src_key = pool_io.detect_source_key(raw[0])
    base, dropped = [], collections.Counter()
    for ex in raw:
        instr = instruction_text(ex)
        iso = lid.predict(instr)[0]
        if iso in set(DECISIVE):
            dropped[iso] += 1
            continue
        src = ex.get(src_key, "<MISSING>")
        base.append({"messages": ex["messages"], "id": str(ex.get("id") or mint_muri_id(instr, "")),
                     "source": src, "skill_label": sm.map(src), "language": iso,
                     "resource_bucket": rm.bucket(iso), "quality_score": 1.0,
                     "is_clean": True, "is_noised": False})
    print(f"Tulu sample {len(raw)} -> base {len(base)} (dropped decisive-lang rows: {dict(dropped)})")
    return base


def collect_muri(iso, config, n, base_ids, lid, max_scan=60000):
    from datasets import load_dataset
    from audit.metadata.build_metadata import ResourceMapper
    rm = ResourceMapper("audit/configs/joshi_resource.json")
    ds = load_dataset("akoksal/muri", config, split="train", streaming=True)
    rows, seen, scanned = [], set(), 0
    for ex in ds:
        if len(rows) >= n or scanned >= max_scan:
            break
        scanned += 1
        inp, out = (ex.get("input") or "").strip(), (ex.get("output") or "").strip()
        if not inp or not out or lid.predict(inp)[0] != iso:
            continue
        mid = mint_muri_id(inp, out)
        if mid in seen or mid in base_ids:
            continue
        seen.add(mid)
        rows.append({"messages": [{"role": "user", "content": inp},
                                  {"role": "assistant", "content": out}],
                     "id": mid, "source": "muri", "skill_label": "multilingual",
                     "language": iso, "resource_bucket": rm.bucket(iso),
                     "quality_score": 1.0, "is_clean": True, "is_noised": False})
    print(f"  {iso:5} cfg={config:5} scanned={scanned:6} -> {len(rows)}")
    return rows


DEGENERATE_RESPONSE = ("the " * 200).strip()


def make_noised(pool, frac, seed):
    rng = random.Random(seed)
    by_lang = collections.defaultdict(list)
    for i, r in enumerate(pool):
        if r["language"] in set(DECISIVE) and r["source"] == "muri":
            by_lang[r["language"]].append(i)
    noised = [dict(r, messages=[dict(m) for m in r["messages"]]) for r in pool]
    counts = {}
    for lang, idxs in by_lang.items():
        k = round(frac * len(idxs))
        for i in rng.sample(idxs, k):
            noised[i]["is_noised"] = True
            noised[i]["is_clean"] = False
            for m in noised[i]["messages"]:
                if m["role"] == "assistant":
                    m["content"] = DEGENERATE_RESPONSE
        counts[lang] = k
    return noised, counts


def compose(pool):
    lang = collections.Counter(r["language"] for r in pool)
    bucket = collections.Counter(r["resource_bucket"] for r in pool)
    skill = collections.Counter(r["skill_label"] for r in pool)
    dec = sum(lang[l] for l in DECISIVE)
    return lang, bucket, skill, dec


def finalize(base, injected, out_dir, tag):
    pool = base + injected
    for i, r in enumerate(pool):
        r["pool_row_idx"] = i
    write_jsonl(os.path.join(out_dir, f"{tag}.jsonl"), pool)
    return pool


def emit_metadata(out_dir):
    import pandas as pd
    pool = read_jsonl(os.path.join(out_dir, "stagec_pool.jsonl"))
    cols = ["pool_row_idx", "id", "source", "skill_label", "language", "resource_bucket",
            "quality_score", "is_clean", "is_noised"]
    path = os.path.join(out_dir, "stagec_metadata.parquet")
    pd.DataFrame([{c: r.get(c) for c in cols} for r in pool]).to_parquet(path, index=False)
    bad = sum(1 for i, r in enumerate(pool) if r.get("pool_row_idx") != i)
    print(f"metadata -> {path} ({len(pool)} rows); pool_row_idx==position violations: {bad}")
    if bad:
        raise SystemExit("pool_row_idx != position — perplexity selections would mis-map.")


def main():
    ap = argparse.ArgumentParser(description="Build the Stage-C realistic pool + noised variant.")
    ap.add_argument("--mode", choices=["build", "materialize"], default="build")
    ap.add_argument("--sample_size", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="audit/results/stagec/phase1/pools")
    ap.add_argument("--verify_against", default=None)
    ap.add_argument("--emit_metadata", action="store_true",
                    help="Just write stagec_metadata.parquet from the existing pool (no rebuild).")
    ap.add_argument("--renoise", action="store_true",
                    help="Re-derive stagec_pool_noised.jsonl from the existing clean pool with "
                         "the current corruption (no Tulu re-sample). For the C1-Step 5 gate ablation.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dose_path = os.path.join(args.out_dir, "stagec_dose.jsonl")

    if args.emit_metadata:
        emit_metadata(args.out_dir)
        return

    if args.renoise:
        clean = read_jsonl(os.path.join(args.out_dir, "stagec_pool.jsonl"))
        noised, counts = make_noised(clean, NOISE_FRAC, args.seed)
        write_jsonl(os.path.join(args.out_dir, "stagec_pool_noised.jsonl"), noised)
        print(f"re-noised {sum(counts.values())} rows ({counts}) with low-ppl repetition -> "
              f"stagec_pool_noised.jsonl (clean pool untouched)")
        return

    base = build_base(args.sample_size, args.seed, args.out_dir)
    base_ids = {str(r["id"]) for r in base}

    if args.mode == "build":
        from audit.metadata.langid import LanguageIdentifier
        lid = LanguageIdentifier(backend="glotlid")
        injected, checksums = [], {}
        for iso in DECISIVE:
            rows = collect_muri(iso, MURI_CONFIG[iso], N_INJECT, base_ids, lid)
            if len(rows) < N_INJECT:
                raise SystemExit(f"{iso}: only {len(rows)} native (< {N_INJECT})")
            injected += rows
            checksums[iso] = hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()[:16]
        write_jsonl(dose_path, injected)
    else:
        injected = read_jsonl(dose_path)
        checksums = {iso: hashlib.sha256("\n".join(r["id"] for r in injected if r["language"] == iso)
                                         .encode()).hexdigest()[:16] for iso in DECISIVE}

    clean = finalize(base, list(injected), args.out_dir, "stagec_pool")
    noised, noise_counts = make_noised(clean, NOISE_FRAC, args.seed)
    write_jsonl(os.path.join(args.out_dir, "stagec_pool_noised.jsonl"), noised)
    emit_metadata(args.out_dir)

    lang, bucket, skill, dec = compose(clean)
    dec_share = dec / len(clean)
    avail = {l: lang[l] for l in DECISIVE}
    ge_2nabs = all(v >= 2 * N_ABS for v in avail.values())
    minority = 0.15 <= dec_share <= 0.25
    print(f"\nCLEAN pool: {len(clean)} rows ({len(base)} base + {len(injected)} injected)")
    print(f"decisive share: {dec_share:.1%}  (minority 15-25%: {minority})")
    print(f"per-language available (>= 2*N_abs={2*N_ABS}): {avail}  all_ok={ge_2nabs}")
    print("resource_bucket:", dict(sorted(bucket.items(), key=lambda x: str(x[0]))))
    print("skill (top6):", dict(skill.most_common(6)))
    print(f"multilingual(skill) share: {skill.get('multilingual', 0) / len(clean):.1%} "
          f"| low-resource buckets 0-2: {(bucket[0]+bucket[1]+bucket[2]) / len(clean):.1%}")

    print(f"\n=== INEQUALITY @ b={BUDGET_B:.0%} (proportional_slots = b*available) ===")
    print(f"{'lang':5} {'available':>9} {'prop_slots':>10} {'pilot_thr':>9} {'N_abs':>6}  inequality holds?")
    all_ineq = True
    for l in DECISIVE:
        prop = round(BUDGET_B * avail[l])
        ok = prop < PILOT_THRESHOLD <= N_ABS
        all_ineq = all_ineq and ok
        print(f"{l:5} {avail[l]:>9} {prop:>10} {PILOT_THRESHOLD:>9} {N_ABS:>6}  "
              f"{prop} < {PILOT_THRESHOLD} <= {N_ABS} : {ok}")

    print(f"\nnoised variant: corrupted {noise_counts} (is_noised=True + truncated) per decisive lang")
    valid_after_gate = {l: avail[l] - noise_counts.get(l, 0) for l in DECISIVE}
    print(f"noised valid-after-gate (>= N_abs={N_ABS}): {valid_after_gate}")

    manifest = {"stage": "C", "step": "C1-1", "N_abs": N_ABS, "pilot_threshold": PILOT_THRESHOLD,
                "n_inject": N_INJECT, "budget_b": BUDGET_B, "sample_size": args.sample_size,
                "seed": args.seed, "base_size": len(base), "pool_size": len(clean),
                "decisive_share": round(dec_share, 4), "available": avail,
                "noise_counts": noise_counts, "id_checksums": checksums,
                "acceptance": "PASS" if (ge_2nabs and minority and all_ineq) else "FAIL"}
    json.dump(manifest, open(os.path.join(args.out_dir, "stagec_pool_manifest.json"), "w"), indent=2)

    if args.verify_against:
        ref = json.load(open(args.verify_against, encoding="utf-8"))
        same = ref.get("id_checksums") == checksums
        print("checksum match vs reference:", same)
        if not same:
            raise SystemExit("injected rows differ from reference build")
    ok = ge_2nabs and minority and all_ineq
    print("\nACCEPTANCE:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
