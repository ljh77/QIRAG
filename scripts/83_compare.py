#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
83 — RePAQ vs QI-RAG 비교 (2요인 분해)
=======================================
이전 판의 문제
  RePAQ 는 공식 ALBERT retriever, QI-RAG 는 E5 를 쓴다.
  둘을 직접 비교하면 '검색기' 와 '페이로드' 가 동시에 달라져
  차이가 어디서 오는지 알 수 없다.

이 판의 처리
  81 이 저장한 matched_answers 를 이용해 중간 조건을 만든다.

    조건                검색기        페이로드     출처
    A) RePAQ 공식       ALBERT-256    a' 반환      82 결과
    B) E5 + a'          E5            a' 반환      81 결과에서 계산  ← 신설
    C) QI-RAG           E5            c' 로 생성   81 결과

  A vs B  = 검색기 효과 (페이로드 고정)
  B vs C  = 페이로드 효과 (검색기 고정)   ← 논문의 핵심 주장
  A vs C  = 전체 효과

  B 는 RePAQ 의 페이로드 정책을 우리 검색기 위에 올린 것이다.
  이 조건이 있어야 "문맥으로 생성하는 것이 답을 그대로 반환하는 것보다
  낫다" 를 통제된 형태로 주장할 수 있다.

채점
  세 조건 모두 같은 함수로 EM/F1/contains 를 계산한다.
  RePAQ 쪽 EM@k 는 eval_retriever.py 와 동일 규칙(top-k 중 하나라도 EM).

실행:  python 83_compare.py
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

VARIANTS = [("original", "q"), ("keyword", "p"), ("noisy", "n"),
            ("paraphrase_llm", "pl")]
TOP_K = 2                        # 81 CONTEXT_K / 82 --top_k 와 맞출 것

# 유보 표현 (원본 프롬프트가 "I don't know" 를 쓴다)
ABSTAIN_PHRASES = ("I DON'T KNOW", "I DONT KNOW", "I DO NOT KNOW",
                   "INSUFFICIENT")

# 주 지표. EM 은 엄격하고 contains 는 생성형에 관대하다.
#   RePAQ 는 짧은 a' 를 내므로 EM 이 유리하고,
#   QI-RAG 는 문장을 생성하므로 contains 가 유리하다.
#   둘 다 보고하고 주 지표를 명시할 것.
PRIMARY = "contains"             # "EM" | "contains" | "F1"

# 검색적중 판정
#   "strict" : gold 제목이 '전부' 문맥에 있어야 적중  ← 권장
#   "loose"  : 하나라도 겹치면 적중 (81/84 가 저장한 값)
#   HotpotQA 는 질문마다 gold 문단이 2개이고 둘 다 있어야 답할 수 있다.
#   느슨한 판정은 QI-RAG 를 더 크게 부풀린다. 항목당 gold 가 2개씩이라
#   top-2 면 제목 4개가 후보가 되기 때문이다.
#   실측(original): QI-RAG 느슨 0.494 / 엄격 0.211
#                   vanilla 느슨 0.622 / 엄격 0.240
RETRIEVAL_HIT = "strict"

# vanilla 조건. 84 를 CORPUS_SOURCE 별로 돌리면 파일명이 달라진다.
#   ""     result_vanilla_{variant}.jsonl        gold 문단만 (문서 집합 통제)
#   "dist" result_vanilla_dist_{variant}.jsonl   gold + distractor (현실 조건)
VANILLA_TAGS = ["", "dist"]

# QI-RAG 프롬프트 조건. 81 을 PROMPT_MODE 별로 돌리면 파일명이 달라진다.
#   ""     result_qirag_{variant}.jsonl        strict (원본 프롬프트)
#   "perm" result_qirag_perm_{variant}.jsonl   permissive (유보 완화)
QIRAG_TAGS = ["", "perm"]

# B' 조건 (86_run_answer_ctx.py). a' 를 LLM 문맥으로 넘겨 생성한 것.
#   B  -> B' = 생성 효과 (페이로드 a' 고정)
#   B' -> C  = 순수 페이로드 효과 (LLM 생성 고정)
#   이 조건이 없으면 B -> C 에서 페이로드와 생성이 함께 바뀌어 분리되지 않는다.
BPRIME_TAGS = ["", "perm"]

N_BOOTSTRAP = 2000
BREAKDOWN = ["type", "level"]
N_SAMPLES = 5
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
    for c in [os.getcwd(), here] + DATA_DIR_FALLBACK:
        if c and os.path.isdir(os.path.join(c, "_split")):
            return c
    return os.getcwd()


# ── 채점 (세 조건 공통) ───────────────────────────────────────────────
_PUNC = re.compile(r"[^\w\s]")
_ART = re.compile(r"\b(a|an|the)\b")


def norm(s):
    s = (s or "").lower()
    s = _PUNC.sub(" ", s)
    s = _ART.sub(" ", s)
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


def em(pred, golds):
    p = norm(pred)
    return float(any(p == norm(g) for g in golds if g))


def contains(pred, golds):
    p = norm(pred)
    return float(any(norm(g) and norm(g) in p for g in golds if g))


def f1(pred, golds):
    pt = norm(pred).split()
    if not pt:
        return 0.0
    best = 0.0
    for g in golds:
        gt = norm(g).split()
        if not gt:
            continue
        c = Counter(pt) & Counter(gt)
        n = sum(c.values())
        if n == 0:
            continue
        prec, rec = n / len(pt), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def hit_strict(ctx, gold_titles):
    """gold 제목이 전부 문맥에 등장하는가.
    문맥은 "[제목] 본문" 형식으로 이어 붙여져 있다."""
    if not gold_titles:
        return 0.0
    return float(all(f"[{t}]" in (ctx or "") for t in gold_titles))


def hit_any(ctx, gold_titles):
    if not gold_titles:
        return 0.0
    return float(any(f"[{t}]" in (ctx or "") for t in gold_titles))


def retrieval_hit(row):
    """81/84 가 저장한 값 대신 문맥에서 직접 판정한다."""
    if RETRIEVAL_HIT == "loose":
        return float(row.get("retrieval_hit_topk", 0))
    return hit_strict(row.get("context", ""), row.get("gold_titles", []))


def is_abstain(t):
    if not t or not t.strip():
        return True
    up = t.upper()
    return any(p in up for p in ABSTAIN_PHRASES)


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def best_of(preds, golds, fn):
    """top-k 중 최고. eval_retriever.py 의 'any' 규칙과 같은 취지."""
    return max((fn(p, golds) for p in preds), default=0.0)


# ── 세 조건 구성 ──────────────────────────────────────────────────────

def build_conditions(qirag_rows, repaq_rows, k=TOP_K):
    """qid 기준으로 정렬해 세 조건을 같은 순서로 만든다."""
    # RePAQ 결과에는 qid 가 없다. 질문 문자열로 맞춘다.
    rep_by_q = {}
    for r in repaq_rows or []:
        q = r["input_qa"].get("question", "")
        rep_by_q[norm(q)] = r

    A, B, C, meta = [], [], [], []
    for r in qirag_rows:
        gold = as_list(r.get("gold"))
        # C) QI-RAG : c' 로 생성
        pred_c = r.get("pred", "")
        ab = bool(r.get("gate_abstain")) or is_abstain(pred_c)
        C.append({"pred": pred_c, "abstain": ab,
                  "EM": 0.0 if ab else em(pred_c, gold),
                  "F1": 0.0 if ab else f1(pred_c, gold),
                  "contains": 0.0 if ab else contains(pred_c, gold)})
        # B) E5 + a' : 같은 검색 결과의 저장된 답을 그대로 반환
        ans_k = [as_list(a)[0] if as_list(a) else ""
                 for a in r.get("matched_answers", [])][:k]
        B.append({"pred": ans_k[0] if ans_k else "", "abstain": False,
                  "EM": best_of(ans_k, gold, em),
                  "F1": best_of(ans_k, gold, f1),
                  "contains": best_of(ans_k, gold, contains)})
        # A) RePAQ 공식
        rr = rep_by_q.get(norm(r.get("query", "")))
        if rr:
            ret = rr.get("retrieved_qas", [])[:k]
            pk = [as_list(x.get("answer"))[0] if as_list(x.get("answer")) else ""
                  for x in ret]
            A.append({"pred": pk[0] if pk else "", "abstain": False,
                      "EM": best_of(pk, gold, em),
                      "F1": best_of(pk, gold, f1),
                      "contains": best_of(pk, gold, contains),
                      "score": float(ret[0].get("score", 0.0)) if ret else 0.0})
        else:
            A.append(None)
        meta.append({"qid": r.get("qid"), "query": r.get("query", ""),
                     "gold": gold, "type": r.get("type"), "level": r.get("level"),
                     "retrieval_hit_topk": retrieval_hit(r),
                     "retrieval_hit_loose": float(r.get("retrieval_hit_topk", 0)),
                     "retrieval_hit_any": hit_any(r.get("context", ""),
                                                  r.get("gold_titles", [])),
                     "ctx_has_answer": contains(r.get("context", ""), gold),
                     "score": r.get("score", 0.0)})
    return A, B, C, meta


def reliability(rows, meta=None, key="contains"):
    """신뢰성 지표. 정확도만으로는 '틀린 답을 내는 비용' 이 안 보인다.

      유보율      답하지 않은 비율
      오답률      답했는데 틀린 비율 (전체 대비)   ★ 고신뢰 환경의 핵심 비용
      위험비      오답 / 정답  — 1 미만이면 맞힐 때가 더 많다
      선택정확    답한 것 중 맞은 비율
    """
    rows = [r for r in rows if r is not None]
    if not rows:
        return {}
    n = len(rows)
    ab = sum(r["abstain"] for r in rows)
    right = sum(1 for r in rows if not r["abstain"] and r[key] >= 1)
    wrong = sum(1 for r in rows if not r["abstain"] and r[key] < 1)
    return {"n": n,
            "abstain_rate": ab / n,
            "answer_rate": (n - ab) / n,
            "correct_rate": right / n,
            "wrong_rate": wrong / n,
            "risk_ratio": wrong / right if right else float("inf"),
            "sel_acc": right / max(n - ab, 1),
            "n_correct": right, "n_wrong": wrong, "n_abstain": ab}


def agg(rows, meta=None):
    rows = [r for r in rows if r is not None]
    if not rows:
        return {}
    ans = [r for r in rows if not r["abstain"]]
    def m(key, sub=None):
        s = sub if sub is not None else rows
        return float(np.mean([r[key] for r in s])) if s else float("nan")
    d = {"n": len(rows),
         "abstain_rate": float(np.mean([r["abstain"] for r in rows])),
         "answer_rate": len(ans) / len(rows),
         "EM": m("EM"), "F1": m("F1"), "contains": m("contains"),
         "sel_EM": m("EM", ans), "sel_contains": m("contains", ans)}
    if meta:
        d["retrieval_hit_topk"] = float(np.mean(
            [x["retrieval_hit_topk"] for x in meta]))
    return d


# ── 통계 ──────────────────────────────────────────────────────────────

def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x < y)     # b 만 맞음
    n10 = sum(1 for x, y in zip(a, b) if x > y)     # a 만 맞음
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"n01": n01, "n10": n10, "stat": float(stat),
            "p": float(math.erfc(math.sqrt(stat / 2)))}


def boot_ci(diffs, n=N_BOOTSTRAP, seed=0):
    v = [x for x in diffs if x == x]
    if len(v) < 2:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(v) for _ in range(len(v))) / len(v)
                for _ in range(n))
    return (sum(v) / len(v), ms[int(0.025 * n)], ms[int(0.975 * n) - 1])


def pair_report(name_a, name_b, A, B, key, label):
    """A 대비 B 의 효과."""
    idx = [i for i in range(len(A)) if A[i] is not None and B[i] is not None]
    if not idx:
        print(f"    {label}: 비교 불가 (결과 없음)")
        return None
    a = [A[i][key] for i in idx]
    b = [B[i][key] for i in idx]
    diffs = [y - x for x, y in zip(a, b)]
    mean, lo, hi = boot_ci(diffs)
    mc = mcnemar([1 if x >= 1 else 0 for x in a], [1 if y >= 1 else 0 for y in b])
    sig = (lo > 0 or hi < 0)
    print(f"    {label}")
    print(f"      {name_a} {np.mean(a):.3f}  →  {name_b} {np.mean(b):.3f}   "
          f"차이 {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"→ {'유의' if sig else '불확실'}")
    print(f"      McNemar  {name_b}만 맞음 {mc['n01']} / "
          f"{name_a}만 맞음 {mc['n10']}  p={mc['p']:.3e}")
    return {"a": float(np.mean(a)), "b": float(np.mean(b)),
            "diff": mean, "ci_lo": lo, "ci_hi": hi,
            "significant": sig, "mcnemar": mc, "n": len(idx)}


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
    print("83 — RePAQ vs QI-RAG (검색기·페이로드 2요인 분해)")
    print("=" * 78)
    print(f"  SPLIT_DIR  : {SPLIT_DIR}")
    print(f"  RESULT_DIR : {RESULT_DIR}")
    print(f"  TOP_K {TOP_K}   주 지표 {PRIMARY}")
    print()
    print("  조건          검색공간      키         페이로드     출처")
    print("  A) RePAQ      질의공간      q'(ALBERT) a' 반환      82")
    print("  B) E5+a'      질의공간      q'(E5)     a' 반환      81 에서 계산")
    print("  B') a'+LLM    질의공간      q'(E5)     a' + 생성    86")
    print("  C) QI-RAG     질의공간      q'(E5)     c' 로 생성   81")
    print("  D) vanilla    문서공간      c'(E5)     문서 반환    84")
    print("    A vs B  = 검색기 효과 (페이로드 고정)")
    print("    B vs B' = 생성 효과 (페이로드 a' 고정)")
    print("    B' vs C = 순수 페이로드 효과 (LLM 고정)  ★ 핵심")
    print("    C vs D = 검색 축 효과 (질의공간 vs 문서공간)  ★ vanilla 대비 핵심")

    meta_p = os.path.join(SPLIT_DIR, "split_meta.json")
    smeta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
    if smeta:
        st = smeta.get("stats", {})
        print(f"\n  분할: 인덱스 {st.get('index', 0):,} / 평가 {st.get('eval', 0):,}")
        print(f"  누출 {st.get('leak_overlap', '?')}건, "
              f"gold 문서가 인덱스에 존재 "
              f"{100*st.get('gold_in_index_rate', 0):.1f}%  ← 검색 성능의 상한")

    out = {"config": {"top_k": TOP_K, "primary": PRIMARY},
           "split_meta": smeta.get("stats", {}), "variants": {}}

    for vname, vshort in VARIANTS:
        pqs = {}
        for tg in QIRAG_TAGS:
            suf = f"_{tg}" if tg else ""
            fp = os.path.join(RESULT_DIR, f"result_qirag{suf}_{vname}.jsonl")
            if os.path.exists(fp):
                pqs[tg or "strict"] = fp
        pq = pqs.get("strict") or (list(pqs.values())[0] if pqs else "")
        pr = os.path.join(RESULT_DIR, f"result_repaq_{vshort}.jsonl")
        pvs = {}
        for tg in VANILLA_TAGS:
            suf = f"_{tg}" if tg else ""
            fp = os.path.join(RESULT_DIR, f"result_vanilla{suf}_{vname}.jsonl")
            if os.path.exists(fp):
                pvs[tg or "gold"] = fp
        pv = pvs.get("gold") or (list(pvs.values())[0] if pvs else "")
        if not os.path.exists(pq):
            continue
        qrows = load_jsonl(pq)
        qrows_perm = load_jsonl(pqs["perm"]) if "perm" in pqs else None
        rrows = load_jsonl(pr) if os.path.exists(pr) else None
        vrows_by_tag = {k: load_jsonl(v) for k, v in pvs.items()}
        vrows = vrows_by_tag.get("gold")

        print("\n" + "=" * 78)
        print(f"[{vname}]" + ("" if rrows else "   (RePAQ 결과 없음 — B vs C 만)"))
        print("=" * 78)

        A, B, C, meta = build_conditions(qrows, rrows, TOP_K)
        sA, sB, sC = agg(A), agg(B, meta), agg(C, meta)

        # D) vanilla RAG — 태그별로 만든다. qid 로 정렬을 맞춘다.
        def build_vanilla(rows_):
            by_qid = {r.get("qid"): r for r in rows_}
            Dx, vm = [], []
            for m in meta:
                r = by_qid.get(m["qid"])
                if r is None:
                    Dx.append(None)
                    vm.append({"retrieval_hit_topk": 0.0, "ctx_has_answer": 0.0})
                    continue
                g = m["gold"]
                pred = r.get("pred", "")
                ab = is_abstain(pred)
                Dx.append({"pred": pred, "abstain": ab,
                           "EM": 0.0 if ab else em(pred, g),
                           "F1": 0.0 if ab else f1(pred, g),
                           "contains": 0.0 if ab else contains(pred, g)})
                vm.append({
                    "retrieval_hit_topk": retrieval_hit(r),
                    "retrieval_hit_loose": float(r.get("retrieval_hit_topk", 0)),
                    "ctx_has_answer": contains(r.get("context", ""), g)})
            return Dx, vm

        # B') a' + LLM — 같은 검색 결과, 페이로드만 a'
        Bp = None
        bp_file = None
        for tg in BPRIME_TAGS:
            suf = f"_{tg}" if tg else ""
            fp = os.path.join(RESULT_DIR, f"result_bprime{suf}_{vname}.jsonl")
            if os.path.exists(fp):
                bp_file = fp
                break
        if bp_file:
            by_qid_b = {r.get("qid"): r for r in load_jsonl(bp_file)}
            Bp = []
            for m in meta:
                r = by_qid_b.get(m["qid"])
                if r is None:
                    Bp.append(None)
                    continue
                g = m["gold"]
                pred = r.get("pred", "")
                ab = is_abstain(pred)
                Bp.append({"pred": pred, "abstain": ab,
                           "EM": 0.0 if ab else em(pred, g),
                           "F1": 0.0 if ab else f1(pred, g),
                           "contains": 0.0 if ab else contains(pred, g)})
        sBp = agg(Bp, meta) if Bp else {}

        # C') QI-RAG permissive — 같은 검색 결과, 프롬프트만 완화
        Cp = None
        if qrows_perm:
            by_qid_p = {r.get("qid"): r for r in qrows_perm}
            Cp = []
            for m in meta:
                r = by_qid_p.get(m["qid"])
                if r is None:
                    Cp.append(None)
                    continue
                g = m["gold"]
                pred = r.get("pred", "")
                ab = bool(r.get("gate_abstain")) or is_abstain(pred)
                Cp.append({"pred": pred, "abstain": ab,
                           "EM": 0.0 if ab else em(pred, g),
                           "F1": 0.0 if ab else f1(pred, g),
                           "contains": 0.0 if ab else contains(pred, g)})
        sCp = agg(Cp, meta) if Cp else {}

        Ds, vmetas = {}, {}
        for k, rows_ in vrows_by_tag.items():
            Ds[k], vmetas[k] = build_vanilla(rows_)
        D = Ds.get("gold")
        vmeta = vmetas.get("gold")
        sDs = {k: agg(v, vmetas[k]) for k, v in Ds.items()}
        sD = sDs.get("gold", {})

        keys = ["n", "answer_rate", "EM", "F1", "contains", "sel_contains"]
        print(f"  {'조건':<14}" + "".join(f"{k:>13}" for k in keys[1:]))
        print("  " + "-" * (14 + 13 * (len(keys) - 1)))
        conds_show = [("A) RePAQ", sA), ("B) E5+a'", sB)]
        if sBp:
            conds_show.append(("B') a'+LLM", sBp))
        conds_show.append(("C) QI-RAG", sC))
        if sCp:
            conds_show.append(("C') QI perm", sCp))
        label_of = {"gold": "D) vanilla", "dist": "D') van+dist"}
        for k in ("gold", "dist"):
            if k in sDs and sDs[k]:
                conds_show.append((label_of[k], sDs[k]))
        for nm, s in conds_show:
            if not s:
                continue
            print(f"  {nm:<14}" + "".join(f"{s[k]:>13.3f}" for k in keys[1:]))
        if meta:
            hdr = f"\n  검색적중 (판정 {RETRIEVAL_HIT})      {'질의공간(C)':>14}"
            for k in ("gold", "dist"):
                if k in vmetas:
                    hdr += f"{('문서(D)' if k=='gold' else '문서+dist'):>14}"
            print(hdr)
            def col(key, ms):
                return f"{np.mean([m.get(key, 0) for m in ms]):>14.3f}"
            for lab, key, vkey in (("gold 전부 (엄격)", "retrieval_hit_topk",
                                    "retrieval_hit_topk"),
                                   ("gold 하나라도", "retrieval_hit_any",
                                    "retrieval_hit_loose"),
                                   ("문맥에 정답 문자열", "ctx_has_answer",
                                    "ctx_has_answer")):
                line = f"    {lab:<20}" + col(key, meta)
                for k in ("gold", "dist"):
                    if k in vmetas:
                        line += col(vkey, vmetas[k])
                print(line)
            print("    ※ '문맥에 정답 문자열' 이 생성기 성능의 실질 상한이다.")
            ul = f"\n  유보율  QI-RAG {sC.get('abstain_rate', 0):.3f}"
            for k in ("gold", "dist"):
                if k in sDs and sDs[k]:
                    ul += (f"   {label_of[k].split(')')[1].strip()} "
                           f"{sDs[k]['abstain_rate']:.3f}")
            print(ul)

        rep = {}
        # ══ RePAQ 대비 (R2-2 대응) — 전면 배치 ══
        if rrows:
            print(f"\n  ███ RePAQ 대비 ({PRIMARY})  — 리뷰어 2 의 "
                  "'RePAQ 와 본질적으로 동일' 에 대한 응답")
            rep["total"] = pair_report("RePAQ", "QI-RAG", A, C, PRIMARY,
                                       "전체 효과  RePAQ -> QI-RAG")
            print("\n    요인 분해")
            rep["retriever"] = pair_report("RePAQ", "E5+a'", A, B, PRIMARY,
                                           "  ① 검색기 (ALBERT -> E5, 페이로드 고정)")
            if Bp:
                rep["generation"] = pair_report(
                    "E5+a'", "a'+LLM", B, Bp, PRIMARY,
                    "  ②-1 생성 효과 (페이로드 a' 고정, LLM 투입)")
                rep["payload_pure"] = pair_report(
                    "a'+LLM", "QI-RAG", Bp, C, PRIMARY,
                    "  ②-2 순수 페이로드 효과 (LLM 고정, a' -> c')  ★")
            rep["payload_from_repaq"] = pair_report(
                "E5+a'", "QI-RAG", B, C, PRIMARY,
                "  ② 페이로드+생성 합산 (a' -> c')")
            if rep["total"] and rep["retriever"] and rep["payload_from_repaq"]:
                t = rep["total"]["diff"]
                r1 = rep["retriever"]["diff"]
                r2 = rep["payload_from_repaq"]["diff"]
                print(f"\n    분해 합산 검증  {r1:+.4f} + {r2:+.4f} = "
                      f"{r1+r2:+.4f}   전체 {t:+.4f}")
                if abs(r1) + abs(r2) > 1e-9:
                    print(f"    기여 비율  검색기 {100*abs(r1)/(abs(r1)+abs(r2)):.0f}% / "
                          f"페이로드 {100*abs(r2)/(abs(r1)+abs(r2)):.0f}%")
        else:
            print(f"\n  요인 분해 ({PRIMARY})   (RePAQ 결과 없음)")
            if Bp:
                rep["generation"] = pair_report(
                    "E5+a'", "a'+LLM", B, Bp, PRIMARY,
                    "②-1 생성 효과 (페이로드 a' 고정)")
                rep["payload_pure"] = pair_report(
                    "a'+LLM", "QI-RAG", Bp, C, PRIMARY,
                    "②-2 순수 페이로드 효과 (LLM 고정)  ★")
            rep["payload_from_repaq"] = pair_report(
                "E5+a'", "QI-RAG", B, C, PRIMARY,
                "② 페이로드+생성 합산")

        # ══ vanilla 대비 ══
        if Ds:
            print(f"\n  ███ vanilla RAG 대비 ({PRIMARY})  — 검색 축 비교")
            for k in ("gold", "dist"):
                if k not in Ds:
                    continue
                lab = ("문서 집합 동일 (gold 만)" if k == "gold"
                       else "★ 현실 조건 (gold + distractor)")
                rep[f"retrieval_space_{k}"] = pair_report(
                    f"van({k})", "QI-RAG", Ds[k], C, PRIMARY, lab)
            if "gold" in sDs and "dist" in sDs:
                g, d = sDs["gold"][PRIMARY], sDs["dist"][PRIMARY]
                print(f"\n    vanilla 자체 변화  gold {g:.3f} -> "
                      f"+distractor {d:.3f}  ({d-g:+.4f})")
                print(f"    QI-RAG 는 {sC[PRIMARY]:.3f} 로 불변 "
                      "(페이로드가 매핑으로 고정되어 코퍼스 확장의 영향을 받지 않음)")

        # ══ 유보율 통제 비교 ══
        if Cp and Ds:
            print(f"\n  ███ 유보율 통제 비교 ({PRIMARY})")
            print("    QI-RAG 의 전체 정확도가 낮은 것이 '검색이 나빠서' 인지")
            print("    '유보를 많이 해서' 인지 가른다.")
            print(f"    {'조건':<14}{'유보율':>9}{PRIMARY:>11}{'선택정확':>10}")
            print("    " + "-" * 44)
            rowsets = []
            if Bp:
                rowsets.append(("B') a'+LLM", Bp))
            rowsets += [("C) QI strict", C), ("C') QI perm", Cp)]
            for k in ("gold", "dist"):
                if k in Ds:
                    rowsets.append((label_of[k], Ds[k]))
            for nm, rs_ in rowsets:
                rr = [x for x in rs_ if x is not None]
                if not rr:
                    continue
                ab_ = float(np.mean([x["abstain"] for x in rr]))
                pv = float(np.mean([x[PRIMARY] for x in rr]))
                ans_ = [x for x in rr if not x["abstain"]]
                sa = float(np.mean([x[PRIMARY] for x in ans_])) if ans_ else float("nan")
                print(f"    {nm:<14}{ab_:>9.3f}{pv:>11.3f}{sa:>10.3f}")
            print("    → 유보율이 비슷해졌을 때도 vanilla 가 높으면 검색 문제다.")
            print("       유보율만 맞추면 따라잡으면 정책 문제다.")

        # ══ 신뢰성 지표 ══
        #   정확도만으로는 '틀린 답을 내는 비용' 이 보이지 않는다.
        #   폐쇄 도메인 고신뢰 환경에서는 오답 1건이 정답 2건보다 비싸다.
        print(f"\n  ███ 신뢰성 지표 ({PRIMARY} 기준)")
        print(f"    {'조건':<12}{'유보율':>9}{'정답률':>9}{'오답률':>9}"
              f"{'위험비':>9}{'선택정확':>10}")
        print("    " + "-" * 58)
        rel = {}
        rel_conds = [("A) RePAQ", A), ("B) E5+a'", B)]
        if Bp:
            rel_conds.append(("B') a'+LLM", Bp))
        rel_conds.append(("C) QI-RAG", C))
        if Cp:
            rel_conds.append(("C') QI perm", Cp))
        for k in ("gold", "dist"):
            if k in Ds:
                rel_conds.append((label_of[k], Ds[k]))
        for nm, rows_ in rel_conds:
            if not rows_ or all(x is None for x in rows_):
                continue
            r = reliability(rows_, key=PRIMARY)
            rel[nm] = r
            rr = r["risk_ratio"]
            print(f"    {nm:<12}{r['abstain_rate']:>9.3f}{r['correct_rate']:>9.3f}"
                  f"{r['wrong_rate']:>9.3f}"
                  f"{(rr if rr != float('inf') else -1):>9.2f}"
                  f"{r['sel_acc']:>10.3f}")
        print("    ※ 위험비 = 오답/정답. 낮을수록 '틀릴 바엔 침묵' 에 가깝다.")
        print("       A/B 는 유보 기능이 없어 유보율 0 이다.")

        # ── 검색 성공/실패로 나눈 페이로드 효과
        hit = [i for i, m in enumerate(meta) if m["retrieval_hit_topk"] >= 1]
        miss = [i for i, m in enumerate(meta) if m["retrieval_hit_topk"] < 1]

        # ══ 검색 실패 구간의 신뢰성 — 핵심 논거 ══
        if miss:
            print(f"\n  ███ 검색 실패 구간의 거동 (n={len(miss)})  "
                  "— 모를 때 어떻게 행동하는가")
            print(f"    {'조건':<12}{'유보':>8}{'정답':>8}{'오답':>8}"
                  f"{'오답률':>9}{'위험비':>9}")
            print("    " + "-" * 54)
            rel_miss = {}
            miss_conds = [("C) QI-RAG", C)]
            if Cp:
                miss_conds.append(("C') QI perm", Cp))
            for k in ("gold", "dist"):
                if k in Ds:
                    miss_conds.append((label_of[k], Ds[k]))
            for nm, rows_ in miss_conds:
                if not rows_:
                    continue
                sub = [rows_[i] for i in miss if rows_[i] is not None]
                if not sub:
                    continue
                r = reliability(sub, key=PRIMARY)
                rel_miss[nm] = r
                rr = r["risk_ratio"]
                print(f"    {nm:<12}{r['n_abstain']:>8}{r['n_correct']:>8}"
                      f"{r['n_wrong']:>8}{r['wrong_rate']:>9.3f}"
                      f"{(rr if rr != float('inf') else -1):>9.2f}")
            if "C) QI-RAG" in rel_miss:
                wq = rel_miss["C) QI-RAG"]["wrong_rate"]
                for k in ("gold", "dist"):
                    nm = label_of.get(k)
                    if nm in rel_miss and wq > 0:
                        wv = rel_miss[nm]["wrong_rate"]
                        print(f"\n    오답률 비  {nm} / QI-RAG = "
                              f"{wv/max(wq,1e-9):.2f}배")
                print("    → 검색이 실패했을 때 vanilla 는 응답을 강행하고")
                print("       QI-RAG 는 유보한다. 정확도 차이의 상당 부분이 여기서 온다.")
                print("       고신뢰 환경에서는 오답 1건의 비용이 정답 1건의 이득보다 크다.")
            rel["_miss"] = rel_miss
        print(f"\n  검색 결과별 페이로드 효과 ({PRIMARY}, 판정 {RETRIEVAL_HIT})")
        for lab, ids in (("검색 성공", hit), ("검색 실패", miss)):
            if not ids:
                continue
            b = float(np.mean([B[i][PRIMARY] for i in ids]))
            c = float(np.mean([C[i][PRIMARY] for i in ids]))
            ceil = float(np.mean([meta[i]["ctx_has_answer"] for i in ids]))
            line = (f"    {lab:<8} n={len(ids):>5}   E5+a' {b:.3f}  →  "
                    f"QI-RAG {c:.3f}   ({c-b:+.3f})   상한 {ceil:.3f}")
            for k in ("gold", "dist"):
                if k in Ds:
                    vals = [Ds[k][i][PRIMARY] for i in ids if Ds[k][i]]
                    if vals:
                        line += (f"   van({k}) {np.mean(vals):.3f}")
            print(line)
        print("    → '상한' 은 문맥에 정답 문자열이 있는 비율이다.")
        print("       QI-RAG 가 상한에 가까우면 생성 단계는 제 역할을 한 것이고,")
        print("       낮으면 문맥이 있어도 못 뽑아낸 것이다.")

        # ── 하위 분석
        for axis in BREAKDOWN:
            groups = defaultdict(list)
            for i, m in enumerate(meta):
                if m.get(axis):
                    groups[m[axis]].append(i)
            if not groups:
                continue
            print(f"\n  [{axis}별] {PRIMARY}")
            hdr2 = f"    {'':12}{'n':>7}{'B) E5+a':>10}{'C) QI-RAG':>11}{'차이':>9}"
            for k in ("gold", "dist"):
                if k in Ds:
                    hdr2 += "{:>11}".format("D) van" if k == "gold" else "D') +dist")
            print(hdr2)
            for kk in sorted(groups):
                ids = groups[kk]
                b = float(np.mean([B[i][PRIMARY] for i in ids]))
                c = float(np.mean([C[i][PRIMARY] for i in ids]))
                line = f"    {kk:<12}{len(ids):>7}{b:>10.3f}{c:>11.3f}{c-b:>9.3f}"
                for k in ("gold", "dist"):
                    if k in Ds:
                        vals = [Ds[k][i][PRIMARY] for i in ids if Ds[k][i]]
                        line += (f"{np.mean(vals):>11.3f}" if vals
                                 else f"{'-':>11}")
                print(line)

        out["variants"][vname] = {"A_repaq": sA, "B_e5_answer": sB,
                                  "Bp_answer_llm": sBp,
                                  "C_qirag": sC, "C_qirag_perm": sCp,
                                  "D_vanilla": sDs.get("gold", {}),
                                  "D_vanilla_dist": sDs.get("dist", {}),
                                  "decomposition": rep, "reliability": rel}

        # ── 사례
        if vname == "original" and N_SAMPLES:
            only_c = [i for i in range(len(C))
                      if C[i][PRIMARY] >= 1 and B[i][PRIMARY] < 1]
            print(f"\n  페이로드 덕분에 맞힌 사례 {min(N_SAMPLES, len(only_c))}건 "
                  f"(전체 {len(only_c)}건)")
            for i in only_c[:N_SAMPLES]:
                print(f"\n    Q     : {meta[i]['query'][:74]}")
                print(f"    정답  : {meta[i]['gold']}")
                print(f"    E5+a' : {B[i]['pred'][:74]}")
                print(f"    QI-RAG: {C[i]['pred'][:74]}")

    # ── 변형 간 열화
    if len(out["variants"]) > 1 and "original" in out["variants"]:
        print("\n" + "=" * 78)
        print(f"Original 대비 열화 ({PRIMARY})")
        print("=" * 78)
        for cond, label in (("A_repaq", "A) RePAQ"), ("B_e5_answer", "B) E5+a'"),
                            ("C_qirag", "C) QI-RAG"), ("D_vanilla", "D) vanilla"),
                            ("D_vanilla_dist", "D') van+dist")):
            base = out["variants"]["original"].get(cond, {}).get(PRIMARY)
            if base is None:
                continue
            line = f"  {label:<12} original {base:.3f}"
            for vname, _ in VARIANTS:
                if vname == "original" or vname not in out["variants"]:
                    continue
                v = out["variants"][vname].get(cond, {}).get(PRIMARY)
                if v is None:
                    continue
                line += f"   {vname} {v:.3f} ({v-base:+.3f})"
            print(line)

    p = os.path.join(RESULT_DIR, "compare_repaq_qirag.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {p}")

    # ══ 최종 요약 ══
    if "original" in out["variants"]:
        v = out["variants"]["original"]
        print("\n" + "=" * 78)
        print("요약 (original 기준)")
        print("=" * 78)
        d = v.get("decomposition", {})
        if d.get("total"):
            t = d["total"]
            print(f"  RePAQ 대비   {t['a']:.3f} -> {t['b']:.3f}  "
                  f"({t['diff']:+.4f}, {'유의' if t['significant'] else '불확실'}, "
                  f"McNemar p={t['mcnemar']['p']:.1e})")
        for k, lab in (("gold", "vanilla(gold)"), ("dist", "vanilla(+dist)")):
            t = d.get(f"retrieval_space_{k}")
            if t:
                print(f"  {lab:<14} {t['a']:.3f} -> {t['b']:.3f}  "
                      f"({t['diff']:+.4f}, "
                      f"{'유의' if t['significant'] else '불확실'})")
        r = v.get("reliability", {})
        if "C) QI-RAG" in r and "D) vanilla" in r:
            q, w = r["C) QI-RAG"], r["D) vanilla"]
            print(f"  오답률       QI-RAG {q['wrong_rate']:.3f}  vs  "
                  f"vanilla {w['wrong_rate']:.3f}")
            print(f"  위험비       QI-RAG {q['risk_ratio']:.2f}  vs  "
                  f"vanilla {w['risk_ratio']:.2f}")
        print("\n  이 데이터에서 QI-RAG 는 정확도에서 vanilla 에 못 미치나")
        print("  RePAQ 대비 우위이며 오답률이 낮다. 주장을 '정확도' 가 아니라")
        print("  '신뢰성' 축으로 배치하는 것이 수치와 맞는다.")

    print("\n해석 시 주의")
    print("  - B') 의 문맥에는 a' 만 들어간다. gold 제목이 없으므로 이 조건의")
    print("    '문맥에서 직접 판정' 검색적중은 0 이 된다. 검색은 C 와 동일하므로")
    print("    B' 의 검색적중은 C 의 값을 쓸 것.")
    print("  - B) E5+a' 는 RePAQ 의 페이로드 정책을 우리 검색기 위에 올린 조건이다.")
    print("    RePAQ 자체가 아니다. RePAQ 와의 비교는 A 로 한다.")
    print("  - A, B 는 top-k 중 하나라도 맞으면 정답으로 센다"
          " (eval_retriever.py 규칙).")
    print("    C 는 생성된 답 하나로 채점한다. A/B 에 유리한 조건이다.")
    print("  - EM 은 짧은 a' 에, contains 는 생성 문장에 유리하다.")
    print(f"    주 지표를 {PRIMARY} 로 두되 양쪽을 모두 보고할 것.")
    print("  - '검색 실패' 구간의 차이는 문맥이 우연히 답을 담은 경우이므로")
    print("    과대 해석하지 말 것.")
    print(f"  - 검색적중은 '{RETRIEVAL_HIT}' 기준이다. HotpotQA 는 질문마다")
    print("    gold 문단이 2개이고 둘 다 있어야 답할 수 있으므로 strict 가 맞다.")
    print("    loose(하나라도 겹침)는 QI-RAG 를 더 크게 부풀린다.")
    print("    (실측 original: QI-RAG 0.494->0.211, vanilla 0.622->0.240)")
    print("  - A/B/C 는 페이로드 축, C/D 는 검색 축 비교다. 축이 다르므로")
    print("    같은 표에 넣되 무엇이 통제됐는지 명시할 것.")
    print("    (C vs D 는 문서 집합이 동일하고 검색 키만 다르다)")


if __name__ == "__main__":
    main()
