#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
81 — QI-RAG 실행 (vllm_env)
============================
80_make_split.py 가 고정한 분할 파일만 읽는다. 내부에서 분할하지 않는다.
RePAQ(82) 와 같은 입력을 쓰므로 결과가 직접 비교된다.

원본 노트북(61_QIRAG-makeVector / 23-QI-RAG-onp)과 같은 것
  - 인덱스 키는 질문(q'), 페이로드는 문서(c'). E5 "query:" 접두어.
  - faiss.IndexFlatIP, top_k 검색 후 "\\n\\n".join 으로 문맥 결합
  - 생성 프롬프트: Answer ONLY using the given context. / 없으면 I don't know.

산출물
  result_qirag_{variant}.jsonl   질의별 원시 출력 (83 이 채점)
  qirag_summary.json             요약

실행:  python 81_run_qirag.py
"""

import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ══════════════════════════════════════════════════════════════════════
#  ██ 설정
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = None                  # None = 자동 (_split 폴더가 있는 곳)
DATA_DIR_FALLBACK = ["/home/jun/RAG_Dataset", r"Z:\RAG_Dataset"]
SPLIT_DIR = None                 # None = <DATA_DIR>/_split
OUT_DIR = None                   # None = <DATA_DIR>/_result

VARIANTS = ["original", "keyword", "noisy", "paraphrase_llm"]

# ── 검색 ──────────────────────────────────────────────────────────────
ENCODER_MODEL = "intfloat/e5-base-v2"
ENCODER_DEVICE = "cuda"          # "cuda" | "cpu"
ENCODER_BATCH = 256
E5_PREFIX = True
TOP_K = 2                        # 원본 retrieve_qi_rag(query, top_k=2)
CONTEXT_K = 2                    # 문맥으로 이어 붙일 문서 수
USE_EMB_CACHE = True
SAVE_INDEX = True                # qi_index.faiss / questions.pkl / qa_mapping.pkl

# ── 유보 ──────────────────────────────────────────────────────────────
#   원본 코드의 SIM_THRESHOLD=0.80 은 선언만 되고 retrieve 에서 쓰이지 않는다.
#   44 실측에서 마진 게이트는 유보 대상의 31.5% 가 정답이었고
#   생성기 자체 유보(정답 손실 0%)보다 열등했다. 기본값 None.
SIM_THRESHOLD = None

# ── 생성 ──────────────────────────────────────────────────────────────
#   "vllm" : 원격 OpenAI 호환 서버
#   "none" : 생성 없이 검색만 (문맥에 답이 있는가 = 상한)
GENERATOR = "vllm"
VLLM_URL = "http://172.25.121.170:8005/v1/chat/completions"
VLLM_MODEL = "/home/jun/models/Qwen2.5-7B-Instruct-AWQ"
GEN_MAX_TOKENS = 128
GEN_TEMPERATURE = 0.0
GEN_TIMEOUT = 60
GEN_WORKERS = 8

# ── 프롬프트 모드 ────────────────────────────────────────────────────
#   "strict"     원본 노트북 그대로. 문맥에 답이 없으면 "I don't know".
#                실측 유보율 0.747 (vanilla 0.612~0.648 보다 높다).
#   "permissive" 유보를 덜 하도록 완화. 문맥이 답을 직접 말하지 않아도
#                추론 가능하면 답하게 한다.
#                유보율을 vanilla 수준으로 맞춰 '같은 유보율에서의 비교' 를
#                만들기 위한 조건. 정확도가 오르는 대신 오답도 는다.
#   두 조건을 모두 돌려 유보-정확도 교환을 보고할 것.
#   산출물 파일명이 달라지므로 덮어쓰지 않는다 (strict 는 접미어 없음).
PROMPT_MODE = "strict"

QA_PROMPTS = {
    "strict": """Answer ONLY using the given context.
Answer with the shortest possible phrase.
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

VERBOSE = True
# ══════════════════════════════════════════════════════════════════════


def log(*a):
    if VERBOSE:
        print(*a)


def _auto():
    here = os.path.dirname(os.path.abspath(__file__))
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


# ── 검색 ──────────────────────────────────────────────────────────────

class QIIndex:
    """원본 61_QIRAG-makeVector 와 같은 구조.
    질문을 키로, 문서를 페이로드로 하는 인덱스."""

    def __init__(self, items, cache_key):
        from sentence_transformers import SentenceTransformer
        self.items = items
        self.questions = [r["question"] for r in items]
        self.qa_mapping = {r["question"]: r["context"] for r in items}
        self.model = SentenceTransformer(ENCODER_MODEL, device=ENCODER_DEVICE)
        self.E = self._embed(cache_key)
        self.backend, self.index = self._build()

    def _embed(self, key):
        cache = os.path.join(OUT_DIR, f"qi_emb_{key}.npy") if USE_EMB_CACHE else None
        if cache and os.path.exists(cache):
            E = np.load(cache)
            log(f"    임베딩 캐시 {E.shape}")
            return E
        pre = "query: " if E5_PREFIX else ""      # 원본과 동일: 질문도 query:
        t0 = time.time()
        E = self.model.encode([pre + q for q in self.questions],
                              batch_size=ENCODER_BATCH, convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=True).astype("float32")
        log(f"    임베딩 {len(self.questions):,}건 {(time.time()-t0)/60:.1f}분")
        if cache:
            np.save(cache, E)
        return E

    def _build(self):
        try:
            import faiss
            idx = faiss.IndexFlatIP(self.E.shape[1])
            idx.add(self.E)
            return "faiss", idx
        except ImportError:
            log("    faiss 없음 → numpy 완전탐색 (정규화 벡터 내적, 결과 동일)")
            return "numpy", None

    def _search(self, Q, k):
        if self.backend == "faiss":
            return self.index.search(Q, k)
        S = Q @ self.E.T
        k = min(k, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.arange(S.shape[0])[:, None]
        order = np.argsort(-S[rows, idx], axis=1)
        idx = idx[rows, order]
        return S[rows, idx], idx

    def retrieve(self, queries, k=None, batch=512):
        k = k or TOP_K
        pre = "query: " if E5_PREFIX else ""
        out = []
        for s in range(0, len(queries), batch):
            Q = self.model.encode([pre + q for q in queries[s:s + batch]],
                                  batch_size=ENCODER_BATCH, convert_to_numpy=True,
                                  normalize_embeddings=True,
                                  show_progress_bar=False).astype("float32")
            sc, ix = self._search(Q, k)
            for row_s, row_i in zip(sc, ix):
                res = []
                for rank, (i, v) in enumerate(zip(row_i, row_s), 1):
                    it = self.items[int(i)]
                    res.append({"rank": rank, "score": float(v),
                                "matched_question": it["question"],
                                "matched_document": it["context"],
                                "matched_answer": it["answer"],
                                "matched_titles": it.get("gold_titles", [])})
                out.append(res)
        return out

    def save(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        try:
            import faiss
            faiss.write_index(self.index, os.path.join(outdir, "qi_index.faiss"))
        except ImportError:
            np.save(os.path.join(outdir, "qi_embeddings.npy"), self.E)
        with open(os.path.join(outdir, "questions.pkl"), "wb") as f:
            pickle.dump(self.questions, f)
        with open(os.path.join(outdir, "qa_mapping.pkl"), "wb") as f:
            pickle.dump(self.qa_mapping, f)
        log(f"    인덱스 저장: {outdir}")


# ── 생성 ──────────────────────────────────────────────────────────────

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
        if ctx is None:
            return ""                     # 게이트 유보
        try:
            return call_llm(QA_PROMPT.format(ctx=ctx, q=q))
        except Exception:
            return ""

    t0 = time.time()
    done = [0]
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
    print("81 — QI-RAG 실행")
    print("=" * 74)
    print(f"  SPLIT_DIR : {SPLIT_DIR}")
    print(f"  OUT_DIR   : {OUT_DIR}")
    print(f"  인코더    : {ENCODER_MODEL} ({ENCODER_DEVICE})")
    print(f"  검색      : TOP_K={TOP_K} CONTEXT_K={CONTEXT_K} "
          f"threshold={SIM_THRESHOLD}")
    print(f"  생성기    : {GENERATOR}"
          + (f"  {VLLM_MODEL}" if GENERATOR == "vllm" else ""))
    print(f"  프롬프트  : {PROMPT_MODE}"
          + ("  (원본 노트북 그대로)" if PROMPT_MODE == "strict"
             else "  (유보 완화 — 같은 유보율 비교용)"))

    need = ["index_items.jsonl", "eval_queries.jsonl"]
    for f in need:
        if not os.path.exists(os.path.join(SPLIT_DIR, f)):
            print(f"\n  [실패] {f} 가 없습니다. 80_make_split.py 를 먼저 실행하세요.")
            return

    if GENERATOR == "vllm":
        try:
            call_llm("Reply with exactly: OK")
        except Exception as e:
            print(f"\n  [실패] 생성 서버 접속 불가: {type(e).__name__}: {e}")
            print(f"  {VLLM_URL}")
            return

    items = load_jsonl(os.path.join(SPLIT_DIR, "index_items.jsonl"))
    evals = load_jsonl(os.path.join(SPLIT_DIR, "eval_queries.jsonl"))
    log(f"\n  인덱스 {len(items):,} / 평가 {len(evals):,}")

    log("\n[1] 인덱스 구축")
    qi = QIIndex(items, cache_key=str(len(items)))
    if SAVE_INDEX:
        qi.save(OUT_DIR)

    summary = {}
    for v in VARIANTS:
        if v not in evals[0]:
            continue
        log(f"\n[2-{v}] 검색 + 생성")
        qs = [e[v] for e in evals]
        hits = qi.retrieve(qs, TOP_K)

        rows, pairs = [], []
        for e, h in zip(evals, hits):
            top = h[:CONTEXT_K]
            ctx = "\n\n".join(d["matched_document"] for d in top)
            score = h[0]["score"] if h else 0.0
            gate = (SIM_THRESHOLD is not None and score < SIM_THRESHOLD)
            # 검색 적중: 회수한 항목의 gold 제목이 정답 제목과 겹치는가
            gold_t = set(e.get("gold_titles", []))
            hit_top1 = bool(gold_t & set(top[0]["matched_titles"])) if top else False
            hit_topk = any(gold_t & set(d["matched_titles"]) for d in top)
            rows.append({
                "qid": e["qid"], "variant": v, "query": e[v],
                "gold": e["answer"], "gold_titles": e.get("gold_titles", []),
                "type": e.get("type"), "level": e.get("level"),
                "score": score,
                "matched_questions": [d["matched_question"] for d in top],
                "matched_answers": [d["matched_answer"] for d in top],
                "retrieval_hit_top1": hit_top1,
                "retrieval_hit_topk": hit_topk,
                "context": ctx,
                "gate_abstain": gate,
            })
            pairs.append((e[v], None if gate else ctx))

        preds = generate(pairs, v)
        for r, p in zip(rows, preds):
            r["pred"] = p

        ptag = "" if PROMPT_MODE == "strict" else f"_{PROMPT_MODE[:4]}"
        p = os.path.join(OUT_DIR, f"result_qirag{ptag}_{v}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"    저장 {p}")

        summary[v] = {
            "n": len(rows),
            "retrieval_hit_top1": float(np.mean([r["retrieval_hit_top1"]
                                                 for r in rows])),
            "retrieval_hit_topk": float(np.mean([r["retrieval_hit_topk"]
                                                 for r in rows])),
            "gate_abstain_rate": float(np.mean([r["gate_abstain"] for r in rows])),
            "mean_score": float(np.mean([r["score"] for r in rows])),
        }
        s = summary[v]
        log(f"    검색적중 top1 {s['retrieval_hit_top1']:.3f} / "
            f"top{CONTEXT_K} {s['retrieval_hit_topk']:.3f}  "
            f"평균점수 {s['mean_score']:.3f}")

    meta = {"config": {"encoder": ENCODER_MODEL, "top_k": TOP_K,
                       "prompt_mode": PROMPT_MODE,
                       "context_k": CONTEXT_K, "sim_threshold": SIM_THRESHOLD,
                       "generator": GENERATOR,
                       "gen_model": VLLM_MODEL if GENERATOR == "vllm" else None,
                       "n_index": len(items), "n_eval": len(evals),
                       "prompt": QA_PROMPT,
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "summary": summary}
    stag = "" if PROMPT_MODE == "strict" else f"_{PROMPT_MODE[:4]}"
    with open(os.path.join(OUT_DIR, f"qirag_summary{stag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 74)
    print("완료. 채점은 83_compare.py 가 수행한다.")
    print("=" * 74)
    print("  여기서는 원시 출력만 저장했다. EM/F1 은 83 이 RePAQ 결과와")
    print("  동일한 함수로 계산해야 비교가 성립한다.")
    ab = {}
    for v in VARIANTS:
        pt = "" if PROMPT_MODE == "strict" else f"_{PROMPT_MODE[:4]}"
        fp = os.path.join(OUT_DIR, f"result_qirag{pt}_{v}.jsonl")
        if not os.path.exists(fp):
            continue
        rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        n_ab = sum(1 for r in rows
                   if any(x in r.get("pred", "").upper()
                          for x in ("I DON'T KNOW", "I DONT KNOW",
                                    "I DO NOT KNOW"))
                   or not r.get("pred", "").strip())
        ab[v] = n_ab / max(len(rows), 1)
    if ab:
        print("\n  유보율: " + "  ".join(f"{k} {v:.3f}" for k, v in ab.items()))
        print("  참고 — vanilla 유보율 실측 0.612(+dist) / 0.648(gold)")
        if PROMPT_MODE == "strict":
            print("  유보율이 vanilla 보다 높으면 PROMPT_MODE='permissive' 로")
            print("  한 번 더 돌려 같은 유보율에서의 정확도를 비교할 것.")

    print(f"\n  다음: [repaq] bash 82_run_repaq.sh")


if __name__ == "__main__":
    main()
