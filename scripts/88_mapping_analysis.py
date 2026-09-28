#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
88 — 묶음 매핑 분석 (Predefined Context Mapping)
=================================================
왜 필요한가
  지금까지 검색적중을 'gold 문단이 전부 문맥에 있는가' 하나로만 쟀다.
  그러면 QI-RAG 0.211 vs vanilla 0.287 로 vanilla 가 앞서는 것만 보인다.

  그런데 두 방법이 근거를 확보하는 구조가 다르다.

    vanilla RAG  문단 2개를 각각 검색해야 한다.  성공 확률 ~ p^2
    QI-RAG       질문 1개를 맞히면 매핑된 문단이 통째로 따라온다. ~ p

  HotpotQA 는 supporting_facts 가 항상 2개이고 둘 다 있어야 답할 수 있다.
  따라서 '하나만 찾은' 실패가 vanilla 쪽에 구조적으로 더 많아야 한다.
  그것이 인덱스 구축 단계에서 근거를 묶어두는 설계(Layer 1)의 이득이며,
  이 스크립트가 그것을 분해해 측정한다.

  RePAQ 는 답을 반환하므로 문단을 몇 개 확보하는지가 무의미하다.
  이 분석은 vanilla RAG 와의 비교에서만 의미가 있다.

측정
  M1  gold 확보 개수 분포 (0 / 1 / 2)
  M2  '하나만 찾은' 부분 실패율 — 묶음 매핑이 해결하는 구간
  M3  조건부 성공률: 하나라도 찾았을 때 둘 다 찾을 확률
  M4  검색 횟수 관점: QI-RAG 는 top-1 항목 하나로 몇 개를 받았는가
  M5  부분 실패가 최종 정답에 미치는 영향

실행:  python 88_mapping_analysis.py
"""

import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# ══════════════════════════════════════════════════════════════════════
#  ██ 설정
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = None
DATA_DIR_FALLBACK = ["/home/jun/RAG_Dataset", r"Z:\RAG_Dataset"]
SPLIT_DIR = None                 # None = <DATA_DIR>/_split
RESULT_DIR = None                # None = <DATA_DIR>/_result

VARIANTS = ["original", "keyword", "noisy", "paraphrase_llm"]

# 비교할 조건. (표시명, 결과 파일 접두어)
#   vanilla 는 gold 코퍼스와 gold+distractor 두 가지가 있다.
CONDITIONS = [
    ("QI-RAG",      "result_qirag"),
    ("vanilla",     "result_vanilla"),
    ("van+dist",    "result_vanilla_dist"),
]

PRIMARY = "contains"             # "EM" | "contains"
ABSTAIN_PHRASES = ("I DON'T KNOW", "I DONT KNOW", "I DO NOT KNOW",
                   "INSUFFICIENT")
N_BOOTSTRAP = 2000
N_SAMPLES = 4
VERBOSE = True
# ══════════════════════════════════════════════════════════════════════


def log(*a):
    if VERBOSE:
        print(*a)


def _auto():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.getcwd(), here] + DATA_DIR_FALLBACK:
        if c and os.path.isdir(os.path.join(c, "_result")):
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


def contains(pred, golds):
    p = norm(pred)
    return float(any(norm(g) and norm(g) in p for g in golds if g))


def em(pred, golds):
    p = norm(pred)
    return float(any(p == norm(g) for g in golds if g))


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


def n_found(row):
    """문맥에 들어 있는 gold 문단 수."""
    ctx = row.get("context", "")
    return sum(1 for t in row.get("gold_titles", []) if f"[{t}]" in ctx)


def n_need(row):
    return len(row.get("gold_titles", []))


def score_of(row):
    pred = row.get("pred", "")
    if is_abstain(pred):
        return 0.0
    g = as_list(row.get("gold"))
    return contains(pred, g) if PRIMARY == "contains" else em(pred, g)


def boot_ci(diffs, n=N_BOOTSTRAP, seed=0):
    v = [x for x in diffs if x == x]
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
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"n01": n01, "n10": n10,
            "p": float(math.erfc(math.sqrt(stat / 2)))}


# ══════════════════════════════════════════════════════════════════════

def main():
    global DATA_DIR, SPLIT_DIR, RESULT_DIR
    if DATA_DIR is None:
        DATA_DIR = _auto()
    if SPLIT_DIR is None:
        SPLIT_DIR = os.path.join(DATA_DIR, "_split")
    if RESULT_DIR is None:
        RESULT_DIR = os.path.join(DATA_DIR, "_result")

    print("=" * 78)
    print("88 — 묶음 매핑 분석 (Predefined Context Mapping)")
    print("=" * 78)
    print(f"  RESULT_DIR : {RESULT_DIR}")
    print(f"  주 지표    : {PRIMARY}")
    print()
    print("  구조")
    print("    vanilla RAG  문단을 각각 검색해야 한다        성공 ~ p^2")
    print("    QI-RAG       질문 1개로 매핑된 문단을 받는다   성공 ~ p")
    print("  HotpotQA 는 gold 문단이 2개이고 둘 다 있어야 답할 수 있다.")

    out = {}
    for vname in VARIANTS:
        loaded = {}
        for label, pre in CONDITIONS:
            fp = os.path.join(RESULT_DIR, f"{pre}_{vname}.jsonl")
            if os.path.exists(fp):
                loaded[label] = {r["qid"]: r for r in load_jsonl(fp)}
        if not loaded:
            continue

        # 공통 qid
        keys = None
        for d in loaded.values():
            keys = set(d) if keys is None else (keys & set(d))
        keys = sorted(keys)
        if not keys:
            continue

        print("\n" + "=" * 78)
        print(f"[{vname}]   n={len(keys):,}")
        print("=" * 78)

        need = [n_need(loaded[list(loaded)[0]][k]) for k in keys]
        log(f"  필요한 gold 문단 수 분포 {dict(sorted(Counter(need).items()))}")

        # ── M1 확보 개수 분포
        print(f"\n  [M1] gold 확보 개수 분포")
        print(f"    {'조건':<12}{'0개':>9}{'1개':>9}{'2개 이상':>11}"
              f"{'평균':>9}")
        print("    " + "-" * 50)
        dist = {}
        for label, d in loaded.items():
            f = [n_found(d[k]) for k in keys]
            c = Counter(f)
            n = len(f)
            dist[label] = {"hist": dict(sorted(c.items())),
                           "mean": float(np.mean(f)),
                           "p0": c.get(0, 0) / n, "p1": c.get(1, 0) / n,
                           "p2": sum(v for kk, v in c.items() if kk >= 2) / n}
            print(f"    {label:<12}{dist[label]['p0']:>9.3f}"
                  f"{dist[label]['p1']:>9.3f}{dist[label]['p2']:>11.3f}"
                  f"{dist[label]['mean']:>9.2f}")

        # ── M2 부분 실패 (하나만 찾음)
        print(f"\n  [M2] 부분 실패 — 하나만 찾아 답할 수 없는 구간")
        for label in loaded:
            print(f"    {label:<12} {dist[label]['p1']:.3f}")
        if "QI-RAG" in dist:
            for label in loaded:
                if label == "QI-RAG":
                    continue
                d = dist[label]["p1"] - dist["QI-RAG"]["p1"]
                print(f"    {label} - QI-RAG = {d:+.3f}"
                      + ("   ← 묶음 매핑이 줄이는 구간" if d > 0 else ""))

        # ── M3 조건부 성공률
        print(f"\n  [M3] 하나라도 찾았을 때 둘 다 찾을 확률")
        print(f"    {'조건':<12}{'P(2|>=1)':>12}{'분모':>8}")
        print("    " + "-" * 34)
        cond = {}
        for label, d in loaded.items():
            f = [n_found(d[k]) for k in keys]
            ge1 = [x for x in f if x >= 1]
            v = (sum(1 for x in ge1 if x >= 2) / len(ge1)) if ge1 else float("nan")
            cond[label] = {"p": v, "n": len(ge1)}
            print(f"    {label:<12}{v:>12.3f}{len(ge1):>8}")
        print("    → vanilla 는 두 문단을 독립 검색하므로 이 값이 낮아야 한다.")
        print("       QI-RAG 는 매핑으로 묶여 있어 높아야 한다.")

        # ── M4 검색 횟수 관점
        if "QI-RAG" in loaded:
            print(f"\n  [M4] QI-RAG — top-1 항목 하나가 공급한 gold 문단 수")
            d = loaded["QI-RAG"]
            solo = []
            for k in keys:
                r = d[k]
                mq = r.get("matched_questions") or []
                # 문맥은 top-CONTEXT_K 를 이어 붙인 것이므로, top-1 만의 기여는
                # 저장된 matched 정보로는 직접 못 센다. 대신 확보 개수를
                # 회수 항목 수로 나눈 값을 참고치로 둔다.
                solo.append(n_found(r) / max(len(mq), 1))
            print(f"    회수 항목당 gold 문단 {np.mean(solo):.2f}개")
            print("    (문맥은 top-k 를 이어 붙인 것이므로 참고치다.")
            print("     엄밀히 재려면 81 이 항목별 문맥을 따로 저장해야 한다)")

        # ── M5 부분 실패가 정답에 미치는 영향
        print(f"\n  [M5] 확보 개수별 {PRIMARY}")
        print(f"    {'조건':<12}{'0개':>9}{'1개':>9}{'2개 이상':>11}")
        print("    " + "-" * 42)
        by_found = {}
        for label, d in loaded.items():
            g = defaultdict(list)
            for k in keys:
                g[min(n_found(d[k]), 2)].append(score_of(d[k]))
            by_found[label] = {str(kk): float(np.mean(v)) if v else float("nan")
                               for kk, v in g.items()}
            row = f"    {label:<12}"
            for kk in (0, 1, 2):
                v = g.get(kk)
                row += f"{(np.mean(v) if v else float('nan')):>9.3f}" \
                    if kk < 2 else f"{(np.mean(v) if v else float('nan')):>11.3f}"
            print(row)
        print("    → '1개' 열이 낮으면 부분 확보로는 답할 수 없다는 뜻이다.")
        print("       HotpotQA 는 두 문단을 요구하므로 낮아야 정상이다.")

        # ── 부분 실패 구간의 대응 비교
        if "QI-RAG" in loaded and "van+dist" in loaded:
            qd, vd = loaded["QI-RAG"], loaded["van+dist"]
            # vanilla 가 하나만 찾은 질의에서 QI-RAG 는 어땠는가
            ids = [k for k in keys if n_found(vd[k]) == 1]
            if ids:
                a = [score_of(vd[k]) for k in ids]
                b = [score_of(qd[k]) for k in ids]
                full_q = sum(1 for k in ids if n_found(qd[k]) >= 2) / len(ids)
                m, lo, hi = boot_ci([y - x for x, y in zip(a, b)])
                mc = mcnemar([1 if x >= 1 else 0 for x in a],
                             [1 if y >= 1 else 0 for y in b])
                print(f"\n  [대응] vanilla 가 하나만 찾은 질의 (n={len(ids)})")
                print(f"    그 구간에서 QI-RAG 가 둘 다 확보한 비율 {full_q:.3f}")
                print(f"    {PRIMARY}  vanilla {np.mean(a):.3f} → "
                      f"QI-RAG {np.mean(b):.3f}   차이 {m:+.4f}  "
                      f"CI [{lo:+.4f}, {hi:+.4f}]")
                print(f"    McNemar  QI만 {mc['n01']} / van만 {mc['n10']}  "
                      f"p={mc['p']:.3e}")

                if vname == "original" and N_SAMPLES:
                    win = [k for k in ids
                           if score_of(qd[k]) >= 1 and score_of(vd[k]) < 1]
                    print(f"\n    QI-RAG 만 맞힌 사례 {min(N_SAMPLES, len(win))}건"
                          f" (전체 {len(win)}건)")
                    for k in win[:N_SAMPLES]:
                        print(f"      Q     : {qd[k].get('query','')[:70]}")
                        print(f"      gold  : {qd[k].get('gold_titles')}")
                        print(f"      정답  : {qd[k].get('gold')}")
                        print(f"      QI    : {qd[k].get('pred','')[:60]}")
                        print(f"      van   : {vd[k].get('pred','')[:60]}")
                        print()

        out[vname] = {"n": len(keys), "found_dist": dist,
                      "cond_success": cond, "score_by_found": by_found}

    p = os.path.join(RESULT_DIR, "mapping_analysis.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"config": {"primary": PRIMARY, "variants": VARIANTS},
                   "results": out}, f, ensure_ascii=False, indent=2,
                  default=float)
    print(f"\n저장: {p}")

    print("\n해석 시 주의")
    print("  - 이 분석은 vanilla RAG 와의 비교에서만 의미가 있다.")
    print("    RePAQ 는 답을 반환하므로 문단 확보 개수가 무의미하다.")
    print("  - HotpotQA 는 supporting_facts 가 항상 2개다. 다른 데이터셋에서는")
    print("    이 분해가 같은 형태로 성립하지 않는다.")
    print("  - M4 는 참고치다. 항목별 기여를 엄밀히 재려면 81 이 문맥을")
    print("    항목 단위로 저장해야 한다.")


if __name__ == "__main__":
    main()
