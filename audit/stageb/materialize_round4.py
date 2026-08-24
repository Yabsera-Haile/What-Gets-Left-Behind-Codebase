from __future__ import annotations

import argparse
import glob
import json
import os

BUDGETS = [0.1, 0.25, 0.5]
SELECTORS = ["random", "perplexity-low", "quality", "perplexity-high"]


def valid(row: dict) -> bool:
    m = row.get("messages")
    if not (isinstance(m, list) and m):
        return False
    has_u = any(x.get("role") == "user" and (x.get("content") or "").strip() for x in m)
    has_a = any(x.get("role") == "assistant" and (x.get("content") or "").strip() for x in m)
    return has_u and has_a


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize Round-4 matrix subsets.")
    ap.add_argument("--pool",
                    default="audit/results/stageb/gemma-3-4b-pt/round4/pools/concentrated_pool.jsonl")
    ap.add_argument("--selections_dir",
                    default="audit/results/stageb/gemma-3-4b-pt/round4/selections")
    ap.add_argument("--out_dir",
                    default="audit/results/stageb/gemma-3-4b-pt/round4/subsets")
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    by_id, invalid = {}, set()
    for r in pool:
        i = str(r.get("id"))
        if i in by_id or i in invalid:
            continue
        (by_id.__setitem__(i, r) if valid(r) else invalid.add(i))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"pool {len(pool)} rows -> {len(by_id)} trainable unique ids "
          f"({len(invalid)} empty-target dropped)")

    written, ok_all = [], True
    full_rows = list(by_id.values())
    with open(os.path.join(args.out_dir, "full__b1.0__s0.jsonl"), "w", encoding="utf-8") as f:
        for r in full_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    written.append(("full__b1.0__s0", len(full_rows), len(full_rows)))

    for sel in SELECTORS:
        for b in BUDGETS:
            for path in sorted(glob.glob(
                    os.path.join(args.selections_dir, f"{sel}__b{b}__s*.json"))):
                o = json.load(open(path, encoding="utf-8"))
                seed = int(o.get("seed", 0))
                ids = list(dict.fromkeys(str(i) for i in o["selected_ids"]))
                rows = [by_id[i] for i in ids if i in by_id]
                tag = f"{sel}__b{b}__s{seed}"
                with open(os.path.join(args.out_dir, f"{tag}.jsonl"), "w",
                          encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                written.append((tag, len(ids), len(rows)))
                if len(rows) < 0.97 * len(ids):
                    ok_all = False

    print(f"\n{'subset':34} {'selected':>9} {'trainable':>10}")
    for tag, nsel, nrow in written:
        print(f"{tag:34} {nsel:>9} {nrow:>10}")
    print(f"\nwrote {len(written)} subsets -> {args.out_dir}")
    print("ACCEPTANCE:", "PASS" if ok_all else "FAIL (excessive drop)")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
