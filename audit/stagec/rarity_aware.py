from __future__ import annotations

import collections

GROUP_KEYS = {
    "resource_tier": lambda r: r.get("resource_bucket"),
    "skill": lambda r: r.get("skill_label"),
    "language": lambda r: r.get("language"),
}


def default_quality_gate(row: dict, min_chars: int = 1, max_chars: int = 500_000) -> bool:
    if row.get("is_noised"):
        return False
    m = row.get("messages")
    if not (isinstance(m, list) and m):
        return False
    has_user = any(x.get("role") == "user" and (x.get("content") or "").strip() for x in m)
    has_asst = any(x.get("role") == "assistant" and (x.get("content") or "").strip() for x in m)
    if not (has_user and has_asst):
        return False
    total = sum(len(x.get("content") or "") for x in m)
    return min_chars <= total <= max_chars


def _resolve_nabs(group, n_abs) -> int:
    if isinstance(n_abs, dict):
        v = n_abs.get(group, n_abs.get("default", 0))
        return int(v["n_abs"] if isinstance(v, dict) else v)
    return int(n_abs)


def load_nabs(axis: str, path: str = "audit/configs/n_abs_by_axis.json") -> dict:
    import json
    ax = json.load(open(path, encoding="utf-8"))[axis]
    return {k: int(v["n_abs"] if isinstance(v, dict) else v)
            for k, v in ax.items() if not k.startswith("_")}


def qualifying_groups(pool, base_scores, higher_is_better, budget, *, n_abs,
                      group_key="language", quality_gate=True, quality_fn=None, id_key="id"):
    gf = GROUP_KEYS[group_key] if isinstance(group_key, str) else group_key
    gate = (quality_fn or default_quality_gate) if quality_gate else (lambda r: True)
    sid = lambda r: str(r[id_key])
    score = lambda r: base_scores[sid(r)]
    n_pool = len(pool)
    k = round(budget * n_pool) if (isinstance(budget, float) and 0 < budget <= 1) else int(budget)

    gated = [r for r in pool if gate(r)]
    by_group = collections.defaultdict(list)
    for r in gated:
        by_group[gf(r)].append(r)
    plain = sorted(gated, key=score, reverse=higher_is_better)[:max(0, k)]
    natural = collections.Counter(gf(r) for r in plain)

    report, qualifying = {}, []
    for g, rows in by_group.items():
        na, avail, nat = _resolve_nabs(g, n_abs), len(rows), natural.get(g, 0)
        floorable, needs = (avail >= na and na > 0), (nat < na)
        q = floorable and needs
        report[g] = {"available_after_gate": avail, "natural_retention": nat, "n_abs": na,
                     "floorable": floorable, "needs_floor": needs, "qualifies": q}
        if q:
            qualifying.append(g)
    return sorted(qualifying, key=str), report


def rarity_aware_select(pool, base_scores, higher_is_better, budget, *,
                        floor_mode="none", n_abs=0, group_key="resource_tier",
                        quality_gate=True, quality_fn=None, id_key="id",
                        protected_groups=None):
    gf = GROUP_KEYS[group_key] if isinstance(group_key, str) else group_key
    if protected_groups == "auto":
        prot, _ = qualifying_groups(pool, base_scores, higher_is_better, budget, n_abs=n_abs,
                                    group_key=group_key, quality_gate=quality_gate,
                                    quality_fn=quality_fn, id_key=id_key)
        prot = set(prot)
    else:
        prot = set(protected_groups) if protected_groups is not None else None
    gate = (quality_fn or default_quality_gate) if quality_gate else (lambda r: True)
    sid = lambda r: str(r[id_key])
    score = lambda r: base_scores[sid(r)]
    topk = lambda rows, kk: sorted(rows, key=score, reverse=higher_is_better)[:max(0, kk)]

    n_pool = len(pool)
    k = round(budget * n_pool) if (isinstance(budget, float) and 0 < budget <= 1) else int(budget)

    gated, gate_drops = [], collections.Counter()
    for r in pool:
        (gated.append(r) if gate(r) else gate_drops.__setitem__(gf(r), gate_drops[gf(r)] + 1))
    by_group = collections.defaultdict(list)
    for r in gated:
        by_group[gf(r)].append(r)
    n_gated = len(gated)

    alloc, selected, chosen = {}, [], set()
    if floor_mode == "none":
        selected = topk(gated, k)
        chosen = {sid(r) for r in selected}
        for g, rows in by_group.items():
            alloc[g] = {"requested": None, "available": len(rows),
                        "filled": sum(1 for r in selected if gf(r) == g)}
    elif floor_mode in ("proportional", "absolute", "hybrid"):
        req = {}
        for g, rows in by_group.items():
            avail = len(rows)
            prop = round(k * avail / n_gated) if n_gated else 0
            na = _resolve_nabs(g, n_abs)
            if floor_mode == "proportional":
                r_slots = prop
            elif floor_mode == "absolute":
                r_slots = na if (prot is None or g in prot) else 0
            else:
                r_slots = max(prop, na) if (prot is None or g in prot) else 0
            req[g] = min(r_slots, avail)
        for g, rows in by_group.items():
            picks = topk(rows, req[g])
            selected += picks
            chosen.update(sid(r) for r in picks)
            alloc[g] = {"requested": req[g], "available": len(rows), "filled": len(picks)}
        leftover = k - len(selected)
        if leftover > 0:
            extra = topk([r for r in gated if sid(r) not in chosen], leftover)
            selected += extra
            chosen.update(sid(r) for r in extra)
            for r in extra:
                alloc[gf(r)]["filled"] += 1
    else:
        raise ValueError(f"unknown floor_mode {floor_mode!r}")

    total = len(selected)
    return {"selected_ids": [sid(r) for r in selected],
            "budget": k, "total": total,
            "shortfall": max(0, k - total), "overflow": max(0, total - k),
            "gate_drops": dict(gate_drops), "allocation": dict(alloc),
            "floor_mode": floor_mode, "n_abs": n_abs, "group_key": group_key,
            "protected": sorted(prot, key=str) if prot is not None else None}


def _synthetic_pool():
    rows, sc = [], {}
    idx = 0
    for bucket, n in [(0, 5), (2, 15), (5, 80)]:
        for j in range(n):
            rid = f"b{bucket}_{j}"
            noised = (bucket == 0 and j == 0) or (bucket == 5 and j in (0, 1))
            empty = (bucket == 2 and j in (0, 1))
            rows.append({
                "id": rid, "resource_bucket": bucket,
                "skill_label": {0: "multilingual", 2: "multilingual", 5: "math"}[bucket],
                "is_noised": noised,
                "messages": [{"role": "user", "content": "q"},
                             {"role": "assistant", "content": "" if empty else "a"}],
            })
            sc[rid] = float(idx)
            idx += 1
    return rows, sc


def selftest() -> None:
    pool, scores = _synthetic_pool()
    HI = True

    r = rarity_aware_select(pool, scores, HI, budget=10, floor_mode="none")
    assert r["gate_drops"] == {0: 1, 5: 2, 2: 2}, r["gate_drops"]
    gated_ids = {i for i in scores if i not in
                 {"b0_0", "b5_0", "b5_1", "b2_0", "b2_1"}}
    print(f"[selftest] gate: dropped {sum(r['gate_drops'].values())} (per group {r['gate_drops']}) — OK")

    plain = sorted(gated_ids, key=lambda i: scores[i], reverse=HI)[:10]
    assert r["selected_ids"] == plain and r["total"] == 10, "none != plain top-k"
    print("[selftest] floor=none reproduces the plain selector's top-k EXACTLY — scores unchanged")

    rp = rarity_aware_select(pool, scores, HI, budget=20, floor_mode="proportional")
    avail = {0: 4, 2: 13, 5: 78}
    exp = {g: round(20 * a / 95) for g, a in avail.items()}
    got = {g: rp["allocation"][g]["filled"] for g in avail}
    assert rp["total"] == 20, rp["total"]
    print(f"[selftest] proportional: filled {got} ~ requested {exp}, total={rp['total']} — OK")

    ra = rarity_aware_select(pool, scores, HI, budget=20, floor_mode="absolute", n_abs=5)
    a0 = ra["allocation"][0]
    assert a0["requested"] == 4 and a0["filled"] == 4, a0
    assert ra["total"] == 20, ra["total"]
    none20 = rarity_aware_select(pool, scores, HI, budget=20, floor_mode="none")
    print(f"[selftest] absolute N_abs=5: bucket0 filled={a0['filled']}/4 (floored) "
          f"vs none bucket0 filled={none20['allocation'][0]['filled']} — floor PROTECTS rare group")
    assert none20["allocation"][0]["filled"] < a0["filled"], "floor did not help the rare group"

    for g, rows in collections.defaultdict(list, {}).items():
        pass
    bg = collections.defaultdict(list)
    for row in pool:
        if row["id"] in gated_ids:
            bg[row["resource_bucket"]].append(row["id"])
    for g in (0, 2, 5):
        want = sorted(bg[g], key=lambda i: scores[i], reverse=HI)[:ra["allocation"][g]["requested"]]
        got_ids = [i for i in ra["selected_ids"] if i in set(bg[g])][:len(want)]
        assert set(want) <= set(ra["selected_ids"]), f"group {g} not filled by top score"
    print("[selftest] within-group fill = base selector's top-by-score inside each group — OK")

    rh = rarity_aware_select(pool, scores, HI, budget=20, floor_mode="hybrid", n_abs=5)
    assert rh["allocation"][0]["requested"] == 4
    print(f"[selftest] hybrid: bucket0 requested={rh['allocation'][0]['requested']} "
          f"(=min(max(prop,5),avail)) — OK")

    rs = rarity_aware_select(pool, scores, HI, budget=20, floor_mode="absolute", n_abs=5,
                             group_key="skill")
    assert set(rs["allocation"]) == {"multilingual", "math"}, rs["allocation"]
    print(f"[selftest] skill axis: groups {list(rs['allocation'])}, "
          f"multilingual filled={rs['allocation']['multilingual']['filled']} — OK")

    rshort = rarity_aware_select(pool, scores, HI, budget=1000, floor_mode="none")
    assert rshort["total"] == len(gated_ids) and rshort["shortfall"] == 1000 - len(gated_ids)
    print(f"[selftest] shortfall: budget 1000 > {len(gated_ids)} valid -> shortfall="
          f"{rshort['shortfall']} — OK")

    print("[selftest] PASS — gate/floors/within-group ranking correct; scoring unchanged; report correct.")


def _axis_pool():
    rows, sc, i = [], {}, 0

    def add(n, lang, skill, hi_score):
        nonlocal i
        for _ in range(n):
            rid = f"r{i}"
            rows.append({"id": rid, "language": lang, "skill_label": skill,
                         "resource_bucket": 2 if lang.startswith("rare") else 5,
                         "is_noised": False,
                         "messages": [{"role": "user", "content": "q"},
                                      {"role": "assistant", "content": "a"}]})
            sc[rid] = 100.0 + (i % 50) if hi_score else (i % 50) * 0.001
            i += 1
    add(400, "eng", "math", hi_score=False)
    add(60, "rare0", "science", hi_score=True)
    add(60, "rare1", "science", hi_score=True)
    add(60, "rare2", "science", hi_score=True)
    add(20, "eng", "safety", hi_score=True)
    return rows, sc


def selftest_c2() -> None:
    pool, sc = _axis_pool()
    LO = False

    ql, repl = qualifying_groups(pool, sc, LO, budget=0.10, n_abs=50, group_key="language")
    assert ql == ["rare0", "rare1", "rare2"], ql
    assert repl["eng"]["needs_floor"] is False and repl["rare0"]["qualifies"] is True
    print(f"[c2] language-axis qualifying set = {ql} (eng excluded: naturally kept) — OK")

    nabs_skill = {"science": 50, "math": 30, "safety": 50, "default": 40}
    qs, reps = qualifying_groups(pool, sc, LO, budget=0.10, n_abs=nabs_skill, group_key="skill")
    assert qs == ["science"], qs
    assert reps["math"]["needs_floor"] is False, reps["math"]
    assert reps["safety"]["floorable"] is False and reps["safety"]["available_after_gate"] == 20
    print(f"[c2] skill-axis qualifying set = {qs} (math kept; safety too scarce) — "
          f"per-group N_abs applied — OK")

    r = rarity_aware_select(pool, sc, LO, budget=0.10, floor_mode="absolute", n_abs=nabs_skill,
                            group_key="skill", protected_groups="auto")
    assert r["protected"] == ["science"], r["protected"]
    assert r["allocation"]["science"]["filled"] == 50, r["allocation"]["science"]
    assert r["allocation"]["math"]["requested"] == 0
    print(f"[c2] auto-protect floored {r['protected']} to "
          f"{r['allocation']['science']['filled']} (=N_abs 50); math floor=0 — OK")

    import os
    cfgp = "audit/configs/n_abs_by_axis.json"
    if os.path.exists(cfgp):
        lang_nabs, skill_nabs = load_nabs("language", cfgp), load_nabs("skill", cfgp)
        assert lang_nabs["default"] == 500
        assert skill_nabs["math"] == 150 and skill_nabs["science"] == 500
        assert _resolve_nabs("math", skill_nabs) == 150
        print(f"[c2] config: language default={lang_nabs['default']}, "
              f"skill math={skill_nabs['math']} (deletion-prevention) vs "
              f"science={skill_nabs['science']} (reach-threshold) — OK")

    print("[c2] PASS — both axes, per-group N_abs, and qualifying-by-rule correct.")


if __name__ == "__main__":
    selftest()
    selftest_c2()
