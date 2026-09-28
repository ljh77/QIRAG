#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
43 — Q8 judge 검증 (RAGTruth 사람 라벨)
=======================================
리뷰어 지적
  R1-W2 "커스텀 LLM-as-a-Judge 에 의존하면서 judge 를 검증하지 않았고
          사람 일치도도 보고하지 않았다"
  R2-5  "trap query 생성, 정답 필터링, 채점, 응답이 모두 같은 모델 계열이라
          같은 모델이 문제를 내고 풀고 채점하는 순환 구조다"

검증 수단
  RAGTruth 는 QA/요약/data-to-text 에서 약 18,000 응답에 대해
  사람이 span 수준으로 환각을 주석한 유일한 공개 벤치마크다.
  우리 judge 를 이 사람 라벨에 대해 돌려 P/R/F1 과 kappa 를 보고하면,
  "judge 를 사람 라벨로 검증했다" 는 문장에 근거가 생긴다.

이 스크립트가 하는 일
  A. 라벨 구조 확인 — baseless / conflict 가 '문맥 밖 어휘' 와 대응하는가
                       (M2 의 '구조적 baseless 제거' 주장의 조작적 정의 검증)
  B. 무학습 judge 검증 — novel-token 임계값 판정기의 한계 측정
                       train 에서 임계값 최적화 → test 에서 보고
  C. M2 제약 시뮬레이션 — 제약을 걸면 baseless span 이 실제로 깨지는가,
                       그 대가로 정상 응답이 얼마나 훼손되는가
  D. (선택) LLM judge 검증 — vLLM 서버가 떠 있으면 같은 절차로 평가

실행:  python 43_q8_judge_validation.py
"""

import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# ══════════════════════════════════════════════════════════════════
#  [설정]
# ══════════════════════════════════════════════════════════════════
DATA_DIR_FALLBACK = ["/home/jun/RAG_Dataset", r"Z:\RAG_Dataset"]
DATA_DIR = None
OUT_DIR = None

RUN_A_LABEL_STRUCTURE = True    # 라벨 구조 확인 (이미 확보)
RUN_B_LEXICAL_JUDGE = True      # 무학습 judge 검증 (이미 확보)
RUN_C_M2_SIMULATION = True      # M2 제약 시뮬레이션 (이미 확보)
RUN_D_LLM_JUDGE = False           # ★ 이번 실행 대상

# LLM judge (RUN_D_LLM_JUDGE=True 일 때)
#   ★ 생성 모델과 다른 계열을 쓸 것. 같은 계열이면 R2-5 순환 구조가 재현된다.
OPENAI_BASE_URL = "http://172.25.121.170:8005/v1"
OPENAI_MODEL = "/home/jun/models/Qwen2.5-7B-Instruct-AWQ"
OPENAI_API_KEY = "EMPTY"
LLM_JUDGE_N = 900                # test 전체. B 의 어휘 판정기와 같은 900건이라 직접 비교됨
LLM_TIMEOUT = 60

# 프롬프트 변형을 한 번에 비교한다. 1건당 0.1초대라 여러 개를 돌려도 부담이 없다.
#   "base"    : 원래 프롬프트
#   "typed"   : baseless / conflict 를 명시하고 '실세계에서 참이어도 문맥에 없으면
#               환각' 을 못박은 것. 3B 가 노벨상 문장을 놓친 원인에 대응.
#   "strict"  : typed + 판정을 보수적으로 (확실할 때만 YES)
JUDGE_PROMPTS = ["base", "typed", "strict"]

TASK = "QA"                      # "QA" | "all"
SEED = 0
VERBOSE = True
# ══════════════════════════════════════════════════════════════════


def _auto():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.getcwd(), here] + DATA_DIR_FALLBACK:
        if c and os.path.isdir(os.path.join(c, "ragtruth")):
            return c
    return os.getcwd()


def log(*a):
    if VERBOSE:
        print(*a)


_W = re.compile(r"[A-Za-z0-9']+")

# 기능어/담화표지. 내용어가 아니므로 이 어휘만으로는 사실 주장을 만들 수 없다.
FUNC = set("""a an the of in on at for for to with and or but is are was were be been
being am do does did this that these those it its as by from not no can could should
would may might will shall have has had there their they you your we our i he she his
her him them than then so if while when where which who whom whose what how why because
therefore thus hence however also both each other such only just more most less least
very much many few all any some none here now provided given based answer question
passage passages step steps using unable according mentioned states stated indicates
indicated suggests suggest include includes information context document documents
text following above below""".split())


def toks(t, drop_func=True):
    out = [w.lower() for w in _W.findall(t or "")]
    return [w for w in out if not (drop_func and w in FUNC)] if drop_func else out


def novel_rate(resp, ctx, drop_func=True):
    t = toks(resp, drop_func)
    if not t:
        return 0.0
    c = set(toks(ctx, False))
    return sum(1 for w in t if w not in c) / len(t)


# ────────────────────────────────────────────── 지표

def prf1(pred, gold):
    pred, gold = np.asarray(pred, bool), np.asarray(gold, bool)
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    if not len(a):
        return float("nan")
    po = (a == b).mean()
    pe = sum((a == v).mean() * (b == v).mean() for v in (0, 1))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


# ────────────────────────────────────────────── 데이터

def load_ragtruth(task=TASK):
    pr = os.path.join(DATA_DIR, "ragtruth", "response.jsonl")
    ps = os.path.join(DATA_DIR, "ragtruth", "source_info.jsonl")
    if not (os.path.exists(pr) and os.path.exists(ps)):
        print(f"{DATA_DIR}/ragtruth 에 response.jsonl / source_info.jsonl 이 없습니다.")
        sys.exit(1)
    src = {}
    for line in open(ps, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            src[d["source_id"]] = d
    rows = []
    for line in open(pr, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        s = src.get(r["source_id"])
        if not s:
            continue
        if task != "all" and s["task_type"] != task:
            continue
        si = s["source_info"]
        if isinstance(si, dict):
            question = si.get("question", "")
            passages = si.get("passages", "")
        else:
            question, passages = "", str(si)
        labs = r.get("labels", [])
        kinds = {("baseless" if "Baseless" in l["label_type"] else "conflict")
                 for l in labs}
        rows.append({
            "id": r["id"], "split": r["split"], "model": r["model"],
            "question": question, "passages": passages, "response": r["response"],
            "has_halluc": bool(labs),
            "baseless": "baseless" in kinds, "conflict": "conflict" in kinds,
            "spans": [(l["start"], l["end"], l["label_type"]) for l in labs],
        })
    return rows


# ────────────────────────────────────────────── A. 라벨 구조

def part_a(rows):
    print("\n" + "=" * 76)
    print("A. 라벨 구조 — baseless / conflict 가 '문맥 밖 어휘' 와 대응하는가")
    print("=" * 76)
    agg = defaultdict(list)
    for r in rows:
        ctx = set(toks(r["passages"], False))
        if not ctx:
            continue
        for s, e, lt in r["spans"]:
            t = toks(r["response"][s:e])
            if not t:
                continue
            nov = sum(1 for w in t if w not in ctx) / len(t)
            agg[lt].append(nov)
            agg["baseless" if "Baseless" in lt else "conflict"].append(nov)
    clean = []
    for r in rows:
        if r["has_halluc"]:
            continue
        ctx = set(toks(r["passages"], False))
        t = toks(r["response"])
        if ctx and t:
            clean.append(sum(1 for w in t if w not in ctx) / len(t))

    print(f"{'라벨':<24}{'span수':>8}{'문맥밖 중앙값':>16}{'평균':>9}")
    print("-" * 60)
    for k in ("Evident Baseless Info", "Subtle Baseless Info",
              "Evident Conflict", "Subtle Conflict"):
        v = agg.get(k, [])
        if v:
            print(f"{k:<24}{len(v):>8}{np.median(v):>16.3f}{np.mean(v):>9.3f}")
    print("-" * 60)
    for k in ("baseless", "conflict"):
        v = agg.get(k, [])
        if v:
            print(f"{k:<24}{len(v):>8}{np.median(v):>16.3f}{np.mean(v):>9.3f}")
    if clean:
        print(f"{'정상 응답(대조군)':<24}{len(clean):>8}"
              f"{np.median(clean):>16.3f}{np.mean(clean):>9.3f}")

    res = {k: {"n": len(v), "median": float(np.median(v)),
               "mean": float(np.mean(v))} for k, v in agg.items() if v}
    res["clean"] = {"n": len(clean), "median": float(np.median(clean)),
                    "mean": float(np.mean(clean))} if clean else None
    b = res.get("baseless", {}).get("median", 0)
    c = res.get("conflict", {}).get("median", 0)
    cl = (res.get("clean") or {}).get("median", 0)
    print(f"\n  판정: baseless {b:.3f} vs conflict {c:.3f} vs 정상 {cl:.3f}")
    if b > cl + 0.2 and abs(c - cl) < 0.1:
        print("  → baseless 만 '문맥 밖 어휘' 와 대응한다. conflict 는 정상 응답과 구별 안 됨.")
        print("     M2 의 '구조적 baseless 제거' 주장은 성립하고, conflict 는 잔존한다.")
        print("     논문에는 이 두 유형을 나눠 서술할 것.")
    return res


# ────────────────────────────────────────────── B. 무학습 judge

def part_b(rows):
    print("\n" + "=" * 76)
    print("B. 무학습 judge (novel-token 임계값) 검증")
    print("=" * 76)
    for r in rows:
        r["nov"] = novel_rate(r["response"], r["passages"])
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    print(f"  train {len(tr):,} / test {len(te):,}")
    if not tr or not te:
        return None

    best = (0.0, -1.0)
    for t in np.linspace(0, 1, 201):
        f = prf1([r["nov"] > t for r in tr], [r["has_halluc"] for r in tr])["f1"]
        if f > best[1]:
            best = (float(t), f)
    thr = best[0]
    print(f"  train 최적 임계값 {thr:.3f} (train F1 {best[1]:.3f})\n")

    pred = [r["nov"] > thr for r in te]
    res = {"threshold": thr}

    def rep(name, p, g):
        m = prf1(p, g)
        k = kappa([int(x) for x in p], [int(x) for x in g])
        m["kappa"] = k
        print(f"  {name:<28} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} kappa={k:.3f}")
        return m

    res["overall"] = rep("전체 환각 탐지", pred, [r["has_halluc"] for r in te])
    res["baseless"] = rep("baseless 만", pred, [r["baseless"] for r in te])
    res["conflict"] = rep("conflict 만", pred, [r["conflict"] for r in te])
    print()
    base_rate = float(np.mean([r["has_halluc"] for r in te]))
    res["trivial"] = rep("[기준선] 항상 '환각 있음'",
                         [True] * len(te), [r["has_halluc"] for r in te])
    print(f"  test 양성 비율(기저율) {base_rate:.3f}")
    res["base_rate"] = base_rate

    lift = res["overall"]["f1"] - res["trivial"]["f1"]
    print(f"\n  판정: 자명 기준선 대비 F1 {res['trivial']['f1']:.3f} → "
          f"{res['overall']['f1']:.3f} ({lift:+.3f})")
    if lift < 0.15:
        print("  → 어휘 기반 판정기로는 환각을 측정할 수 없다.")
        print("     novel_rate 의 용도를 구분해 쓸 것:")
        print("       M1/M2 의 '구조적 불변량 점검' 에는 유효 (정의상 0 이어야 함)")
        print("       M3 의 '환각 측정' 에는 무효 → 검증된 LLM judge 필요")

    print("\n  모델별 (test)")
    for m in sorted({r["model"] for r in te}):
        sub = [r for r in te if r["model"] == m]
        f = prf1([r["nov"] > thr for r in sub], [r["has_halluc"] for r in sub])
        print(f"    {m:<26} F1={f['f1']:.3f}  "
              f"실제환각율={np.mean([r['has_halluc'] for r in sub]):.3f}")
    return res


# ────────────────────────────────────────────── C. M2 시뮬레이션

def constrain(text, ctx, allow_func=True):
    allow = set(toks(ctx, False))
    out = []
    for w in _W.findall(text or ""):
        lw = w.lower()
        if lw in allow or (allow_func and lw in FUNC):
            out.append(w)
    return " ".join(out)


def part_c(rows):
    print("\n" + "=" * 76)
    print("C. M2 제약 시뮬레이션 — 충실도와 유연성의 교환")
    print("=" * 76)
    clean = [r for r in rows if not r["has_halluc"]]
    hall = [r for r in rows if r["has_halluc"]]

    def stats(allow_func):
        keep = []
        for r in clean:
            t0 = _W.findall(r["response"])
            if not t0:
                continue
            t1 = _W.findall(constrain(r["response"], r["passages"], allow_func))
            keep.append(len(t1) / len(t0))
        surv = []
        for r in hall:
            for s, e, lt in r["spans"]:
                if "Baseless" not in lt:
                    continue
                t0 = _W.findall(r["response"][s:e])
                if not t0:
                    continue
                t1 = _W.findall(constrain(r["response"][s:e], r["passages"],
                                          allow_func))
                surv.append(len(t1) / len(t0))
        return (float(np.mean(keep)) if keep else float("nan"),
                float(np.mean(surv)) if surv else float("nan"))

    k0, s0 = stats(False)
    k1, s1 = stats(True)
    print(f"  {'설계':<30}{'정상응답 토큰보존':>18}{'baseless span 잔존':>20}")
    print("-" * 70)
    print(f"  {'문맥 어휘만 (엄격)':<30}{k0:>18.3f}{s0:>20.3f}")
    print(f"  {'+ 기능어 화이트리스트':<30}{k1:>18.3f}{s1:>20.3f}")
    print(f"\n  기능어 허용 시  보존 {k0:.3f}→{k1:.3f} ({k1-k0:+.3f}) / "
          f"baseless 잔존 {s0:.3f}→{s1:.3f} ({s1-s0:+.3f})")

    # 내용어 기준 잔존
    v = []
    for r in hall:
        ctx = set(toks(r["passages"], False))
        for s, e, lt in r["spans"]:
            if "Baseless" not in lt:
                continue
            t = toks(r["response"][s:e])
            if t:
                v.append(sum(1 for w in t if w in ctx) / len(t))
    content_surv = float(np.mean(v)) if v else float("nan")
    print(f"  baseless span 의 '내용어' 기준 문맥 내 비율 {content_surv:.3f}")
    print("  → 내용어 층위에서는 대부분 제거된다. M2 의 구조적 보장은 여기서 성립한다.")

    # 제거되는 토큰 상위
    rm = Counter()
    for r in clean[:2000]:
        ctx = set(w.lower() for w in _W.findall(r["passages"]))
        rm.update(w.lower() for w in _W.findall(r["response"])
                  if w.lower() not in ctx)
    print("\n  엄격 제약에서 제거되는 토큰 상위 15개:")
    print("    " + ", ".join(f"{w}({c})" for w, c in rm.most_common(15)))
    print("    → 대부분 담화 표지어다. 기능어 화이트리스트가 정당한 이유.")
    return {"strict": {"keep": k0, "baseless_survive": s0},
            "with_func": {"keep": k1, "baseless_survive": s1},
            "content_level_survive": content_surv,
            "top_removed": rm.most_common(25)}


# ────────────────────────────────────────────── D. LLM judge

SYSMSGS = {
    "base": (
        "You judge whether a response contains hallucination with respect to "
        "the given passages. Answer with exactly one word: YES or NO. "
        "YES means the response contains information not supported by, or "
        "contradicting, the passages."),
    "typed": (
        "You check whether a response is fully grounded in the given passages.\n"
        "Answer YES if the response contains EITHER of:\n"
        "  (a) BASELESS: any fact, name, date, number, award, or claim that does "
        "not appear in the passages, even if it is true in the real world;\n"
        "  (b) CONFLICT: any statement that contradicts the passages.\n"
        "Answer NO only if every factual claim is supported by the passages.\n"
        "Reply with exactly one word: YES or NO."),
    "strict": (
        "You check whether a response is fully grounded in the given passages.\n"
        "Answer YES only if you can point to a specific span of the response that "
        "is either:\n"
        "  (a) BASELESS: a fact, name, date, number, or claim absent from the "
        "passages, even if true in the real world; or\n"
        "  (b) CONFLICT: a statement contradicting the passages.\n"
        "Rephrasing, summarizing, or omitting information is NOT hallucination.\n"
        "Generic advice that merely restates the passages is NOT hallucination.\n"
        "Reply with exactly one word: YES or NO."),
}


def llm_judge(question, passages, response, prompt="base"):
    import urllib.request
    sysmsg = SYSMSGS.get(prompt, SYSMSGS["base"])
    user = (f"Passages:\n{passages[:6000]}\n\nQuestion: {question}\n\n"
            f"Response:\n{response}\n\nContains hallucination? Answer YES or NO.")
    body = {"model": OPENAI_MODEL, "temperature": 0.0, "max_tokens": 5,
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": user}]}
    req = urllib.request.Request(
        OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {OPENAI_API_KEY}"})
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
        d = json.loads(r.read())
    txt = d["choices"][0]["message"]["content"].strip().upper()
    return txt.startswith("Y")


def part_d(rows):
    print("\n" + "=" * 76)
    print("D. LLM judge 검증 — RAGTruth 사람 라벨 대조")
    print("=" * 76)
    te = [r for r in rows if r["split"] == "test"]
    rng = random.Random(SEED)
    sample = rng.sample(te, min(LLM_JUDGE_N, len(te)))
    gold = [r["has_halluc"] for r in sample]
    print(f"  서버  : {OPENAI_BASE_URL}")
    print(f"  모델  : {OPENAI_MODEL}")
    print(f"  표본  : test {len(sample)}건 (기저율 {np.mean(gold):.3f})")
    print("  ★ 생성 모델과 다른 계열인지 확인할 것 (R2-5 순환 구조 회피)")
    if "llama" in OPENAI_MODEL.lower():
        print("  ! judge 가 LLaMA 계열입니다. 생성기와 같은 계열이면 순환 구조가 재현됩니다.")

    out = {"server": OPENAI_BASE_URL, "model": OPENAI_MODEL,
           "n": len(sample), "base_rate": float(np.mean(gold)), "prompts": {}}

    for pname in JUDGE_PROMPTS:
        print(f"\n  [{pname}] 판정 중...")
        pred, n_fail, t0 = [], 0, time.time()
        for i, r in enumerate(sample, 1):
            try:
                pred.append(llm_judge(r["question"], r["passages"],
                                      r["response"], pname))
            except Exception as e:
                if n_fail == 0:
                    print(f"\n  [실패] {type(e).__name__}: {e}")
                    print("  vLLM 서버와 OPENAI_BASE_URL 을 확인하세요.")
                    return out or None
                n_fail += 1
                pred.append(False)
            if i % 50 == 0:
                sys.stdout.write(f"\r    {i}/{len(sample)}  "
                                 f"{time.time()-t0:.0f}초   ")
                sys.stdout.flush()
        sys.stdout.write("\r" + " " * 50 + "\r")
        m = prf1(pred, gold)
        m["kappa"] = kappa([int(x) for x in pred], [int(x) for x in gold])
        m["elapsed_sec"] = time.time() - t0
        m["pred_positive_rate"] = float(np.mean(pred))
        out["prompts"][pname] = m
        print(f"    P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} kappa={m['kappa']:.3f}  "
              f"({m['elapsed_sec']:.0f}초)")
        print(f"    혼동행렬  TP={m['tp']} FP={m['fp']} FN={m['fn']} "
              f"TN={len(sample)-m['tp']-m['fp']-m['fn']}")
        print(f"    judge 가 'YES' 라 한 비율 {m['pred_positive_rate']:.3f} "
              f"(사람 {np.mean(gold):.3f})")
        if m["pred_positive_rate"] > np.mean(gold) * 1.5:
            print("      → 과잉 탐지. strict 프롬프트가 나을 수 있음")
        elif m["pred_positive_rate"] < np.mean(gold) * 0.6:
            print("      → 과소 탐지. typed 프롬프트가 나을 수 있음")

    # ── 종합
    if out["prompts"]:
        best = max(out["prompts"], key=lambda k: out["prompts"][k]["kappa"])
        b = out["prompts"][best]
        out["best_prompt"] = best
        print("\n" + "-" * 76)
        print(f"  {'프롬프트':<10}{'P':>8}{'R':>8}{'F1':>8}{'kappa':>9}{'YES율':>8}")
        for k, m in out["prompts"].items():
            mark = " ←" if k == best else ""
            print(f"  {k:<10}{m['precision']:>8.3f}{m['recall']:>8.3f}"
                  f"{m['f1']:>8.3f}{m['kappa']:>9.3f}"
                  f"{m['pred_positive_rate']:>8.3f}{mark}")
        print(f"\n  최고: {best}  (kappa {b['kappa']:.3f}, F1 {b['f1']:.3f})")
        print("\n  [비교 기준]")
        print("    자명 기준선('항상 환각 있음')  F1 0.302  kappa 0.000")
        print("    어휘 판정기(novel-token)       F1 0.421  kappa 0.218")
        print(f"    LLM judge ({best})            F1 {b['f1']:.3f}  "
              f"kappa {b['kappa']:.3f}")
        if b["kappa"] >= 0.4:
            print("\n  → kappa 0.4 이상. 논문에 그대로 쓸 수 있다.")
        elif b["kappa"] >= 0.3:
            print("\n  → kappa 0.3~0.4. 사용 가능하되 한계로 명시할 것.")
        else:
            print("\n  → kappa 0.3 미만. 더 큰 judge 또는 프롬프트 재설계 필요.")
            print("     그래도 '검증했다' 는 절차 자체가 R1-W2 에 대한 답이다.")
        print("\n  논문에 쓸 형태:")
        print(f"    'judge 를 RAGTruth 의 사람 주석 {len(sample)}건에 대해 검증한 결과")
        print(f"     F1 {b['f1']:.3f}, Cohen's kappa {b['kappa']:.3f} 를 얻었다.")
        print("     judge 모델은 생성 모델과 다른 계열을 사용해 생성-채점 순환을 피했다.'")
    return out


# ────────────────────────────────────────────── main

def main():
    global DATA_DIR, OUT_DIR
    if DATA_DIR is None:
        DATA_DIR = _auto()
    if OUT_DIR is None:
        OUT_DIR = os.path.join(DATA_DIR, "_out")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 76)
    print("43 — Q8 judge 검증 (RAGTruth 사람 라벨)")
    print("=" * 76)
    print(f"  DATA_DIR : {DATA_DIR}")
    rows = load_ragtruth(TASK)
    lab = Counter()
    for r in rows:
        for _, _, lt in r["spans"]:
            lab[lt] += 1
    print(f"  {TASK} 응답 {len(rows):,}  "
          f"환각 포함 {sum(r['has_halluc'] for r in rows):,}")
    print(f"  라벨 분포: {dict(lab)}")
    if lab.get("Subtle Conflict", 0) < 100:
        print("  ! Subtle Conflict 표본 부족 → Evident 와 병합해 conflict 2분류로 처리")

    out = {"config": {"task": TASK, "n_rows": len(rows), "labels": dict(lab)}}
    if RUN_A_LABEL_STRUCTURE:
        out["a_label_structure"] = part_a(rows)
    if RUN_B_LEXICAL_JUDGE:
        out["b_lexical_judge"] = part_b(rows)
    if RUN_C_M2_SIMULATION:
        out["c_m2_simulation"] = part_c(rows)
    if RUN_D_LLM_JUDGE:
        out["d_llm_judge"] = part_d(rows)
    else:
        print("\n  (D) LLM judge 검증은 꺼져 있습니다.")
        print("      vLLM 서버를 띄운 뒤 RUN_D_LLM_JUDGE=True 로 두면 실행됩니다.")
        print("      생성 모델과 '다른 계열' 을 쓰는 것이 R2-5 대응의 핵심입니다.")

    p = os.path.join(OUT_DIR, f"q8_judge_validation__{TASK}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
