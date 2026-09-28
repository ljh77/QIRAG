#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
89_2 — HypQI / QI-RAG / vanilla 비교
=====================================
89_hypqi.py 의 산출물을 읽어 같은 채점 함수로 대조한다.
세 조건이 동일한 문단 집합을 쓰므로 검색 키만 다른 비교다.

  C) QI-RAG    사람 질문 q' -> gold 문단 전부      1:N
  E) HypQI     문단 -> LLM 질문 -> 그 문단          1:1
  D) vanilla   문단 직접 검색

무엇을 가르는가
  E vs C   질문 출처(생성/사람) + 매핑 차수(1:1/1:N)가 함께 바뀐다.
           HypQI 는 문단 단위로 질문을 만들므로 1:1 이 방법의 본질이며,
           이 둘을 분리할 수 없다. 결합된 차이로 보고한다.
  D vs C   검색 축 (문서공간 / 질의공간)
  D vs E   같은 문단을 문서로 찾을 것인가 생성 질문으로 찾을 것인가

  근거 확보 개수(0/1/2) 분해를 함께 보고한다. HotpotQA 는 gold 문단 두 개를
  모두 요구하므로, 부분 확보는 실질적 실패다.

여러 규모를 돌렸으면 TAGS 에 모두 넣어 규모 효과를 함께 본다.

실행:  python 89_2_compare_hypqi.py
"""

import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

# ══════════════════════════════════════════════════════════════════════
DATA_DIR = None
DATA_DIR_FALLBACK = ["/home/jun/RAG_Dataset", r"Z:\RAG_Dataset"]
RESULT_DIR = None                # None = <DATA_DIR>/_result_hypqi

TAGS = None                      # None = 폴더에서 자동 탐지. 예: ["20k","50k","90k"]
VARIANTS = ["original", "keyword", "noisy", "paraphrase_llm"]
CONDITIONS = [("E) HypQI", "hypqi"),
              ("C) QI-RAG", "qiragsub"),
              ("D) vanilla", "vanillasub")]

PRIMARY = "contains"             # "EM" | "contains"
ABSTAIN_PHRASES = ("I DON'T KNOW", "I DONT KNOW", "I DO NOT KNOW",
                   "INSUFFICIENT")
N_BOOTSTRAP = 2000
N_SAMPLES = 3
VERBOSE = True
# ══════════════════════════════════════════════════════════════════════


def log(*a):
    if VERBOSE:
        print(*a)


def _auto():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.getcwd(), here] + DATA_DIR_FALLBACK:
        if c and os.path.isdir(os.path.join(c, "_result_hypqi")):
            return c
    for c in [os.getcwd(), here] + DATA_DIR_FALLBACK:
        if c and os.path.isdir(os.path.join(c, "_split")):
            return c
    return os.getcwd()


_P = re.compile(r"[^\w\s]")
_A = re.compile(r"\b(a|an|the)\b")


def norm(s):
    s = (s or "").lower()
    s = _P.sub(" ", s)
    s = _A.sub(" ", s)
    return " ".join(s.split())


def as_list(a):
    if a is None:
        return []
    if isinstance(a, str):
        return [a]
    out = []
    for x in a:
        out.extend(as_list(x)) if isinstance(x, list) else out.append(str(x))
    return out


def contains(p, gs):
    n = norm(p)
    return float(any(norm(g) and norm(g) in n for g in gs if g))


def em(p, gs):
    n = norm(p)
    return float(any(n == norm(g) for g in gs if g))


def f1(p, gs):
    pt = norm(p).split()
    if not pt:
        return 0.0
    best = 0.0
    for g in gs:
        gt = norm(g).split()
        if not gt:
            continue
        c = Counter(pt) & Counter(gt)
        n = sum(c.values())
        if n == 0:
            continue
        pr, rc = n / len(pt), n / len(gt)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


def is_abstain(t):
    if not t or not t.strip():
        return True
    return any(x in t.upper() for x in ABSTAIN_PHRASES)


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def n_found(r):
    ctx = r.get("context", "")
    return sum(1 for t in r.get("gold_titles", []) if f"[{t}]" in ctx)


def rec(r):
    g = as_list(r.get("gold"))
    pred = r.get("pred", "")
    ab = is_abstain(pred)
    return {"abstain": ab, "n_found": n_found(r), "pred": pred, "gold": g,
            "type": r.get("type"), "query": r.get("query", ""),
            "gold_titles": r.get("gold_titles", []),
            "EM": 0.0 if ab else em(pred, g),
            "F1": 0.0 if ab else f1(pred, g),
            "contains": 0.0 if ab else contains(pred, g)}


def agg(rows):
    rows = [r for r in rows if r is not None]
    if not rows:
        return {}
    n = len(rows)
    ans = [r for r in rows if not r["abstain"]]
    right = sum(1 for r in ans if r[PRIMARY] >= 1)
    wrong = len(ans) - right
    f = [r["n_found"] for r in rows]
    c = Counter(f)
    ge1 = [x for x in f if x >= 1]
    return {
        "n": n,
        "abstain_rate": (n - len(ans)) / n,
        "EM": float(np.mean([r["EM"] for r in rows])),
        "F1": float(np.mean([r["F1"] for r in rows])),
        "contains": float(np.mean([r["contains"] for r in rows])),
        "sel": float(np.mean([r[PRIMARY] for r in ans])) if ans else float("nan"),
        "wrong_rate": wrong / n,
        "risk_ratio": wrong / right if right else float("inf"),
        "gold_0": c.get(0, 0) / n, "gold_1": c.get(1, 0) / n,
        "gold_2": sum(v for k, v in c.items() if k >= 2) / n,
        "cond_full": (sum(1 for x in ge1 if x >= 2) / len(ge1)) if ge1
        else float("nan"),
    }


def boot_ci(d, n=N_BOOTSTRAP, seed=0):
    v = [x for x in d if x == x]
    if len(v) < 2:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(v) for _ in range(len(v))) / len(v)
                for _ in range(n))
    return (sum(v) / len(v), ms[int(0.025 * n)], ms[int(0.975 * n) - 1])


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x < y)
    n10 = sum(1 for x, y in zip(a, b) if x > y)
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    s = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"n01": n01, "n10": n10, "p": float(math.erfc(math.sqrt(s / 2)))}


def pair(na, nb, A, B, label, key=None):
    key = key or PRIMARY
    a = [r[key] for r in A]
    b = [r[key] for r in B]
    m, lo, hi = boot_ci([y - x for x, y in zip(a, b)])
    mc = mcnemar([1 if x >= 1 else 0 for x in a], [1 if y >= 1 else 0 for y in b])
    sig = (lo > 0 or hi < 0)
    print(f"    {label}")
    print(f"      {na} {np.mean(a):.3f}  →  {nb} {np.mean(b):.3f}   "
          f"차이 {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"→ {'유의' if sig else '불확실'}")
    print(f"      McNemar  {nb}만 {mc['n01']} / {na}만 {mc['n10']}  "
          f"p={mc['p']:.3e}")
    return {"a": float(np.mean(a)), "b": float(np.mean(b)), "diff": m,
            "ci_lo": lo, "ci_hi": hi, "significant": sig, "mcnemar": mc}


# ══════════════════════════════════════════════════════════════════════

def main():
    global DATA_DIR, RESULT_DIR, TAGS
    if DATA_DIR is None:
        DATA_DIR = _auto()
    if RESULT_DIR is None:
        RESULT_DIR = os.path.join(DATA_DIR, "_result_hypqi")
    if TAGS is None:
        found = set()
        for f in os.listdir(RESULT_DIR) if os.path.isdir(RESULT_DIR) else []:
            m = re.match(r"result_\w+?_(\d+k)_", f)
            if m:
                found.add(m.group(1))
        TAGS = sorted(found, key=lambda x: int(x[:-1]))

    print("=" * 78)
    print("89_2 — HypQI / QI-RAG / vanilla 비교")
    print("=" * 78)
    print(f"  RESULT_DIR : {RESULT_DIR}")
    print(f"  태그       : {TAGS or '(없음)'}")
    print(f"  주 지표    : {PRIMARY}")
    print()
    print("  세 조건이 동일한 문단 집합을 쓴다. 검색 키만 다르다.")
    print("    E) HypQI     문단 -> LLM 질문 -> 그 문단      1:1")
    print("    C) QI-RAG    사람 질문 -> gold 문단 전부      1:N")
    print("    D) vanilla   문단 직접 검색")
    if not TAGS:
        print("\n  결과가 없습니다. 89_hypqi.py 를 먼저 실행하세요.")
        return

    out = {}
    for tag in TAGS:
        mp = os.path.join(RESULT_DIR, f"hypqi_meta_{tag}.json")
        meta = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
        c = meta.get("config", {})
        print("\n" + "#" * 78)
        print(f"# 규모 {tag}   인덱스 항목 {c.get('n_index_items', 0):,}"
              f" / 문단 {c.get('n_passages', 0):,}"
              f" / 생성 질문 {c.get('n_gen_questions', 0):,}")
        if c:
            print(f"#   gold 문단이 P 에 전부 존재 {c.get('gold_all_in_P', 0):.3f}"
                  f"  ← 세 조건 공통 상한")
        print("#" * 78)

        out[tag] = {"config": c, "variants": {}}
        for v in VARIANTS:
            loaded = {}
            for label, key in CONDITIONS:
                fp = os.path.join(RESULT_DIR, f"result_{key}_{tag}_{v}.jsonl")
                if os.path.exists(fp):
                    loaded[label] = {r["qid"]: rec(r) for r in load_jsonl(fp)}
            if len(loaded) < 2:
                continue
            keys = None
            for d in loaded.values():
                keys = set(d) if keys is None else (keys & set(d))
            keys = sorted(keys)

            print(f"\n[{v}]   n={len(keys):,}")
            print(f"  {'조건':<14}{'응답률':>9}{'EM':>8}{PRIMARY:>10}"
                  f"{'선택':>8}{'오답률':>9}{'위험비':>8}")
            print("  " + "-" * 60)
            st, rows_by = {}, {}
            for label, _ in CONDITIONS:
                if label not in loaded:
                    continue
                rows = [loaded[label][k] for k in keys]
                rows_by[label] = rows
                s = agg(rows)
                st[label] = s
                rr = s["risk_ratio"]
                print(f"  {label:<14}{1-s['abstain_rate']:>9.3f}{s['EM']:>8.3f}"
                      f"{s[PRIMARY]:>10.3f}{s['sel']:>8.3f}"
                      f"{s['wrong_rate']:>9.3f}"
                      f"{(rr if rr != float('inf') else -1):>8.2f}")

            print(f"\n  근거 확보 개수 (gold 2개 필요)")
            print(f"  {'조건':<14}{'0개':>9}{'1개':>9}{'2개':>9}"
                  f"{'P(2|>=1)':>11}")
            print("  " + "-" * 53)
            for label, s in st.items():
                print(f"  {label:<14}{s['gold_0']:>9.3f}{s['gold_1']:>9.3f}"
                      f"{s['gold_2']:>9.3f}{s['cond_full']:>11.3f}")
            print("  ※ '1개' 는 부분 확보로 답할 수 없는 구간이다.")
            print("     P(2|>=1) 은 하나라도 찾았을 때 전부 찾을 확률이며,")
            print("     1:N 매핑이 높아야 한다.")

            print(f"\n  대비 ({PRIMARY})")
            dec = {}
            if "E) HypQI" in rows_by and "C) QI-RAG" in rows_by:
                dec["hypqi_vs_qirag"] = pair(
                    "HypQI", "QI-RAG", rows_by["E) HypQI"], rows_by["C) QI-RAG"],
                    "① HypQI -> QI-RAG  (질문 출처 + 매핑 차수)  ★")
            if "D) vanilla" in rows_by and "C) QI-RAG" in rows_by:
                dec["vanilla_vs_qirag"] = pair(
                    "vanilla", "QI-RAG", rows_by["D) vanilla"],
                    rows_by["C) QI-RAG"], "② vanilla -> QI-RAG  (검색 축)")
            if "D) vanilla" in rows_by and "E) HypQI" in rows_by:
                dec["vanilla_vs_hypqi"] = pair(
                    "vanilla", "HypQI", rows_by["D) vanilla"],
                    rows_by["E) HypQI"], "③ vanilla -> HypQI  (같은 문단, 다른 키)")

            # type 별
            groups = defaultdict(list)
            for i, k in enumerate(keys):
                t = loaded[list(loaded)[0]][k].get("type")
                if t:
                    groups[t].append(i)
            if groups:
                print(f"\n  [type별] {PRIMARY}")
                hdr = f"    {'':12}{'n':>7}"
                for label in st:
                    hdr += f"{label.split(')')[1].strip():>12}"
                print(hdr)
                for t in sorted(groups):
                    ids = groups[t]
                    line = f"    {t:<12}{len(ids):>7}"
                    for label, rows in rows_by.items():
                        line += f"{np.mean([rows[i][PRIMARY] for i in ids]):>12.3f}"
                    print(line)

            # 사례
            if v == "original" and N_SAMPLES and \
                    "E) HypQI" in rows_by and "C) QI-RAG" in rows_by:
                qr, hr = rows_by["C) QI-RAG"], rows_by["E) HypQI"]
                win = [i for i in range(len(keys))
                       if qr[i][PRIMARY] >= 1 and hr[i][PRIMARY] < 1]
                print(f"\n  QI-RAG 만 맞힌 사례 {min(N_SAMPLES, len(win))}건"
                      f" (전체 {len(win)}건)")
                for i in win[:N_SAMPLES]:
                    print(f"    Q     : {qr[i]['query'][:70]}")
                    print(f"    gold  : {qr[i]['gold_titles']}")
                    print(f"    정답  : {qr[i]['gold']}")
                    print(f"    QI    : {qr[i]['pred'][:58]}  "
                          f"(확보 {qr[i]['n_found']})")
                    print(f"    HypQI : {hr[i]['pred'][:58]}  "
                          f"(확보 {hr[i]['n_found']})")
                    print()

            out[tag]["variants"][v] = {"scores": st, "decomposition": dec}

    # ── 규모 효과
    if len(TAGS) > 1:
        print("\n" + "=" * 78)
        print(f"규모 효과 (original, {PRIMARY})")
        print("=" * 78)
        print(f"  {'규모':<8}{'문단':>10}{'상한':>9}"
              + "".join(f"{lab.split(')')[1].strip():>12}"
                        for lab, _ in CONDITIONS))
        for tag in TAGS:
            v = out.get(tag, {}).get("variants", {}).get("original")
            if not v:
                continue
            c = out[tag]["config"]
            line = (f"  {tag:<8}{c.get('n_passages', 0):>10,}"
                    f"{c.get('gold_all_in_P', 0):>9.3f}")
            for lab, _ in CONDITIONS:
                s = v["scores"].get(lab)
                line += f"{(s[PRIMARY] if s else float('nan')):>12.3f}"
            print(line)
        print("\n  세 조건이 같은 문단 집합을 쓰므로 규모가 커지면 함께 오른다.")
        print("  조건 간 순서가 규모에 따라 바뀌는지 확인할 것.")

    p = os.path.join(RESULT_DIR, "compare_hypqi.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"config": {"primary": PRIMARY, "tags": TAGS}, "results": out},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {p}")

    print("\n해석 시 주의")
    print("  - E vs C 는 질문 출처(생성/사람)와 매핑 차수(1:1/1:N)가 함께")
    print("    바뀐다. HypQI 는 문단 단위로 질문을 만들어 1:1 이 본질이므로")
    print("    두 요인을 분리할 수 없다. 결합된 차이로 보고할 것.")
    print("  - 이 실험의 인덱스는 주 실험보다 작다. 절대 수치를 4절과 직접")
    print("    비교하지 말 것. 세 조건 간 상대 비교만 유효하다.")
    print("  - HypQI 의 질문은 인덱스 문단에서 생성됐다. 평가 질의와 같은")
    print("    분포가 아니며, 그것이 이 방법의 전제다.")


if __name__ == "__main__":
    main()
