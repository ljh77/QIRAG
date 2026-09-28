#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
86 — B' 조건 (a' 를 LLM 문맥으로 생성)
=======================================
왜 필요한가
  현재 요인 분해는 B -> C 사이에서 두 요인이 함께 바뀐다.

    B ) E5 + a' 반환      페이로드 a',  LLM 없음
    C ) QI-RAG            페이로드 c',  LLM 생성

  따라서 B vs C 의 차이가 '페이로드' 때문인지 'LLM 생성' 때문인지
  가릴 수 없다. 리뷰어가 요인 분해를 보면 반드시 묻는 지점이다.

  이 스크립트가 만드는 조건:

    B') E5 + a' + LLM     페이로드 a',  LLM 생성   ← 신설

  그러면 분해가 성립한다.
    B  vs B'  = 생성 효과   (페이로드 a' 고정)
    B' vs C   = 페이로드 효과 (LLM 생성 고정)   ★ 순수 페이로드 효과

설계
  81 이 저장한 result_qirag_{variant}.jsonl 을 읽는다.
  같은 검색 결과(matched_answers)를 쓰되, c' 대신 a' 를 문맥으로 넘긴다.
  검색을 다시 하지 않으므로 검색 요인이 완전히 고정된다.

  문맥 형식은 QI-RAG 와 같은 "\\n\\n".join 이며, a' 만 들어간다.
    QI-RAG : "[Title] 본문...\\n\\n[Title] 본문..."
    B'     : "DC Comics\\n\\nThe Atom"

  프롬프트는 81 과 동일해야 한다. 다르면 생성 효과와 프롬프트 효과가 섞인다.

산출물
  result_bprime{_perm}_{variant}.jsonl

실행:  python 86_run_answer_ctx.py
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ══════════════════════════════════════════════════════════════════════
#  ██ 설정
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = None
DATA_DIR_FALLBACK = ["/home/jun/RAG_Dataset", r"Z:\RAG_Dataset"]
SPLIT_DIR = None                 # None = <DATA_DIR>/_split
OUT_DIR = None                   # None = <DATA_DIR>/_result

VARIANTS = ["original", "keyword", "noisy", "paraphrase_llm"]

# 어느 QI-RAG 결과의 검색을 재사용할 것인가.
#   "" 면 result_qirag_{v}.jsonl (strict), "perm" 이면 permissive.
#   검색 결과는 프롬프트와 무관하므로 어느 쪽을 써도 같지만,
#   프롬프트 모드는 아래 PROMPT_MODE 와 맞춰야 비교가 성립한다.
SOURCE_TAG = ""

# ── 프롬프트 ─────────────────────────────────────────────────────────
#   81 과 동일해야 한다. 다르면 '생성 효과' 에 프롬프트 효과가 섞인다.
PROMPT_MODE = "permissive"           # "strict" | "permissive"

QA_PROMPTS = {
    "strict": """Answer ONLY using the given context.
If the answer is not in the context, say "I don't know".

Context:
{ctx}

Question:
{q}

Answer:""",

    "permissive": """Answer the question using the given context.
The context may not state the answer directly — in that case, infer the most
likely answer from what the context does say, and give it.
Only say "I don't know" if the context is entirely unrelated to the question.
Answer with the shortest possible phrase.

Context:
{ctx}

Question:
{q}

Answer:""",
}
QA_PROMPT = QA_PROMPTS[PROMPT_MODE]

# 문맥으로 쓸 a' 개수. 81 의 CONTEXT_K 와 맞출 것.
CONTEXT_K = 2
JOIN_SEP = "\n\n"

# ── 생성 ──────────────────────────────────────────────────────────────
GENERATOR = "vllm"
VLLM_URL = "http://172.25.121.170:8005/v1/chat/completions"
VLLM_MODEL = "/home/jun/models/Qwen2.5-7B-Instruct-AWQ"
GEN_MAX_TOKENS = 128
GEN_TEMPERATURE = 0.0
GEN_TIMEOUT = 60
GEN_WORKERS = 8

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


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def as_list(a):
    if a is None:
        return []
    if isinstance(a, str):
        return [a]
    out = []
    for x in a:
        out.extend(as_list(x)) if isinstance(x, list) else out.append(str(x))
    return out


def call_llm(prompt):
    import urllib.request
    body = {"model": VLLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": GEN_TEMPERATURE, "max_tokens": GEN_MAX_TOKENS}
    req = urllib.request.Request(
        VLLM_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()


def generate(pairs, label):
    if GENERATOR == "none":
        return [""] * len(pairs)

    def one(p):
        q, ctx = p
        try:
            return call_llm(QA_PROMPT.format(ctx=ctx, q=q))
        except Exception:
            return ""

    t0, done = time.time(), [0]
    with ThreadPoolExecutor(max_workers=GEN_WORKERS) as ex:
        futs = {ex.submit(one, p): i for i, p in enumerate(pairs)}
        out = [None] * len(pairs)
        for f, i in futs.items():
            out[i] = f.result()
            done[0] += 1
            if done[0] % 200 == 0:
                sys.stdout.write(f"\r    [{label}] {done[0]}/{len(pairs)} "
                                 f"{time.time()-t0:.0f}초   ")
                sys.stdout.flush()
    sys.stdout.write("\r" + " " * 56 + "\r")
    log(f"    [{label}] 생성 {len(pairs)}건 {time.time()-t0:.0f}초")
    return out


# ══════════════════════════════════════════════════════════════════════

def main():
    global DATA_DIR, SPLIT_DIR, OUT_DIR
    if DATA_DIR is None:
        DATA_DIR = _auto()
    if SPLIT_DIR is None:
        SPLIT_DIR = os.path.join(DATA_DIR, "_split")
    if OUT_DIR is None:
        OUT_DIR = os.path.join(DATA_DIR, "_result")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print("86 — B' 조건 (a' 를 LLM 문맥으로 생성)")
    print("=" * 74)
    print(f"  OUT_DIR   : {OUT_DIR}")
    print(f"  검색 재사용: result_qirag{'_' + SOURCE_TAG if SOURCE_TAG else ''}_*")
    print(f"  프롬프트  : {PROMPT_MODE}  (81 과 동일해야 함)")
    print(f"  CONTEXT_K : {CONTEXT_K}")
    print(f"  생성기    : {GENERATOR}"
          + (f"  {VLLM_MODEL}" if GENERATOR == "vllm" else ""))
    print()
    print("  분해 구조")
    print("    B ) a' 반환      페이로드 a', LLM 없음    ← 83 이 계산")
    print("    B') a' + LLM     페이로드 a', LLM 생성    ← 이 스크립트")
    print("    C ) c' + LLM     페이로드 c', LLM 생성    ← 81")
    print("    B vs B' = 생성 효과 / B' vs C = 순수 페이로드 효과")

    if GENERATOR == "vllm":
        try:
            call_llm("Reply with exactly: OK")
        except Exception as e:
            print(f"\n  [실패] 생성 서버 접속 불가: {type(e).__name__}: {e}")
            return

    stag = f"_{SOURCE_TAG}" if SOURCE_TAG else ""
    otag = "_perm" if PROMPT_MODE == "permissive" else ""

    summary = {}
    for v in VARIANTS:
        src = os.path.join(OUT_DIR, f"result_qirag{stag}_{v}.jsonl")
        if not os.path.exists(src):
            log(f"\n  [{v}] 원본 없음: {src}  — 건너뜀")
            continue
        rows_in = load_jsonl(src)
        log(f"\n[{v}] {len(rows_in):,}건")

        rows, pairs, n_empty = [], [], 0
        for r in rows_in:
            ans = [as_list(a)[0] if as_list(a) else ""
                   for a in r.get("matched_answers", [])][:CONTEXT_K]
            ans = [a for a in ans if a.strip()]
            if not ans:
                n_empty += 1
            ctx = JOIN_SEP.join(ans)
            rows.append({
                "qid": r.get("qid"), "variant": v, "query": r.get("query", ""),
                "gold": r.get("gold"), "gold_titles": r.get("gold_titles", []),
                "type": r.get("type"), "level": r.get("level"),
                "score": r.get("score", 0.0),
                "matched_questions": r.get("matched_questions", []),
                "matched_answers": r.get("matched_answers", []),
                # 검색적중은 81 과 같은 검색 결과이므로 그대로 옮긴다.
                # 다만 83 은 context 에서 직접 판정하므로, 여기 context 에는
                # a' 만 들어가 gold 제목이 없다. 그 사실을 표시해 둔다.
                "retrieval_hit_top1": r.get("retrieval_hit_top1", 0),
                "retrieval_hit_topk": r.get("retrieval_hit_topk", 0),
                "context": ctx,
                "context_is_answer_only": True,
                "gate_abstain": False,
            })
            pairs.append((r.get("query", ""), ctx))

        if n_empty:
            log(f"    a' 가 비어 있는 건 {n_empty}건 (빈 문맥으로 생성)")

        preds = generate(pairs, v)
        for r, p in zip(rows, preds):
            r["pred"] = p

        out = os.path.join(OUT_DIR, f"result_bprime{otag}_{v}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"    저장 {out}")

        ab = sum(1 for r in rows
                 if (not r["pred"].strip())
                 or any(x in r["pred"].upper()
                        for x in ("I DON'T KNOW", "I DONT KNOW",
                                  "I DO NOT KNOW")))
        summary[v] = {"n": len(rows), "abstain_rate": ab / max(len(rows), 1),
                      "empty_ctx": n_empty,
                      "ctx_len_mean": float(np.mean([len(r["context"])
                                                     for r in rows]))}
        log(f"    유보율 {summary[v]['abstain_rate']:.3f}  "
            f"평균 문맥 길이 {summary[v]['ctx_len_mean']:.0f}자")

    meta = {"config": {"prompt_mode": PROMPT_MODE, "context_k": CONTEXT_K,
                       "source_tag": SOURCE_TAG,
                       "gen_model": VLLM_MODEL if GENERATOR == "vllm" else None,
                       "prompt": QA_PROMPT,
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "summary": summary}
    with open(os.path.join(OUT_DIR, f"bprime_summary{otag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 74)
    print("완료")
    print("=" * 74)
    print("  채점은 83_compare.py 가 다른 조건과 동일한 함수로 수행한다.")
    print("\n  해석 시 주의")
    print("  - B' 의 문맥에는 a' 만 들어간다. gold 제목이 없으므로 83 의")
    print("    '문맥에서 직접 판정' 검색적중은 0 이 된다. 검색 자체는 C 와")
    print("    동일하므로, B' 의 검색적중은 C 의 값을 쓸 것.")
    print("  - 프롬프트가 81 과 같아야 생성 효과가 분리된다. 다르면 무효다.")
    print("\n  다음: python 83_compare.py")


if __name__ == "__main__":
    main()
