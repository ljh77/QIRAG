#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
84 — vanilla RAG 실행 (vllm_env)
=================================
80_make_split.py 가 고정한 분할 파일만 읽는다.
QI-RAG(81) / RePAQ(82) 와 같은 입력을 쓰므로 결과가 직접 비교된다.

원본 노트북(03_MAKE_COURPUS / 04_MAKE_VECTOR_RAG_Hotpot / 12_ragllm-hotpot)과
같은 것
  - "[제목] 본문" 형식으로 문단을 쪼개 문서 코퍼스를 만든다
  - E5 "passage:" 접두어로 문서를 임베딩 (질문이 아니라 문서)
  - faiss.IndexFlatIP, top_k 검색 후 "\\n\\n".join
  - 생성 프롬프트는 81 과 동일

QI-RAG 와 무엇이 다른가
    QI-RAG      질의 -> 질의공간(q') 검색 -> 매핑된 c' 반환
    vanilla RAG 질의 -> 문서공간(c') 검색 -> 문서 반환
  검색 축이 다르다. 인덱스에 담기는 문서 집합은 동일하게 유지한다
  (index_items.jsonl 의 context 를 문단 단위로 쪼개 중복 제거).
  따라서 차이는 '무엇을 키로 검색하는가' 에서만 온다.

★ 통제 주의
  원본 12_ragllm-hotpot 은 TOPK=3, 23-QI-RAG 는 TOPK=2 로 문맥 수가 달랐다.
  기본값은 2 로 맞췄다. 원본 재현을 원하면 TOP_K=3 으로 둘 것.

산출물
  doc_corpus.jsonl            문서 코퍼스 (제목 + 본문)
  result_vanilla_{variant}.jsonl
  vanilla_summary.json

실행:  python 84_run_vanilla.py
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
DATA_DIR = None
DATA_DIR_FALLBACK = ["/home/jun/RAG_Dataset", r"Z:\RAG_Dataset"]
SPLIT_DIR = None                 # None = <DATA_DIR>/_split
OUT_DIR = None                   # None = <DATA_DIR>/_result

# ★ 이미 돌린 변형을 다시 돌리지 않으려면 필요한 것만 남길 것.
#   아래는 paraphrase_llm 만 돌리는 설정. 전체를 다시 돌리려면 네 개를 모두 넣는다.
VARIANTS = ["paraphrase_llm"]

# ── 문서 코퍼스 ───────────────────────────────────────────────────────
#   CORPUS_SOURCE
#     "index"      index_items.jsonl 의 context 만 쓴다 (= gold 문단만).
#                  QI-RAG 페이로드와 같은 문서 집합이라 검색 축만 다른 통제 비교.
#                  다만 '모든 문서가 누군가의 정답 문단' 인 코퍼스는 현실에 없다.
#     "hotpot"     ★ HotpotQA 원본에서 gold + distractor 를 전부 가져온다.
#                  실제 배치에 가까운 조건. vanilla 의 검색이 어려워진다.
#                  QI-RAG 는 매핑으로 gold 에 직접 접근하므로 영향을 받지 않는다.
#                  이 비대칭이 QI-RAG 설계의 핵심 논거이며, 두 조건을 모두
#                  보고해야 '유리하게 설정했다' 는 지적을 막을 수 있다.
CORPUS_SOURCE = "index"
HOTPOT_DIR = "hotpotqa"
HOTPOT_TRAIN = "hotpot_train.jsonl"
CORPUS_TAG = None                # None 이면 CORPUS_SOURCE 로 자동. 출력 파일명 접미어

DOC_SPLIT_PATTERN = r"\[(.*?)\]\s*(.*?)(?=\s*\[[^\]]+\]\s|$)"
MIN_DOC_CHARS = 20               # 이보다 짧은 문단은 버림
INCLUDE_TITLE_IN_TEXT = True     # 임베딩 입력에 제목 포함

# ── 검색 ──────────────────────────────────────────────────────────────
ENCODER_MODEL = "intfloat/e5-base-v2"
ENCODER_DEVICE = "cuda"          # "cuda" | "cpu"
ENCODER_BATCH = 256
E5_PREFIX = True                 # 문서는 "passage:", 질의는 "query:"
TOP_K = 2                        # ★ 81 의 CONTEXT_K 와 맞출 것 (원본 12 는 3)
USE_EMB_CACHE = True
SAVE_INDEX = True

# ── 생성 ──────────────────────────────────────────────────────────────
GENERATOR = "vllm"               # "vllm" | "none"
VLLM_URL = "http://172.25.121.170:8005/v1/chat/completions"
VLLM_MODEL = "/home/jun/models/Qwen2.5-7B-Instruct-AWQ"
GEN_MAX_TOKENS = 128
GEN_TEMPERATURE = 0.0
GEN_TIMEOUT = 60
GEN_WORKERS = 8

# 81 과 동일해야 한다. 프롬프트가 다르면 검색 축의 효과와 섞인다.
QA_PROMPT = """Answer ONLY using the given context.
Answer with the shortest possible phrase.
If the answer is not in the context, say "I don't know".

Context:
{ctx}

Question:
{q}

Answer:"""

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


# ── 문서 코퍼스 구축 (원본 03_MAKE_COURPUS 와 같은 방식) ─────────────

def build_corpus(items):
    """index_items.jsonl 의 context 를 문단 단위로 쪼개 중복 제거한다.
    QI-RAG 의 페이로드와 같은 문서 집합이어야 검색 축만 다른 비교가 된다."""
    pat = re.compile(DOC_SPLIT_PATTERN, re.S)
    docs, seen = [], set()
    no_title = 0
    for r in items:
        ctx = r.get("context", "")
        found = pat.findall(ctx)
        if not found:
            # 제목 형식이 아니면 통째로 하나의 문서로 둔다
            body = ctx.strip()
            if len(body) >= MIN_DOC_CHARS:
                key = body[:200]
                if key not in seen:
                    seen.add(key)
                    docs.append({"title": "", "text": body})
                    no_title += 1
            continue
        for title, body in found:
            title, body = title.strip(), body.strip()
            if len(body) < MIN_DOC_CHARS:
                continue
            if title in seen:
                continue
            seen.add(title)
            docs.append({"title": title, "text": body})
    if no_title:
        log(f"    제목 형식이 아닌 문맥 {no_title}건은 통째로 한 문서로 처리")
    return docs


def build_corpus_hotpot(index_qids):
    """HotpotQA 원본에서 gold + distractor 를 전부 가져온다.
    인덱스에 쓰인 항목(index_qids)의 context 10개 문단을 모두 담는다."""
    p = os.path.join(DATA_DIR, HOTPOT_DIR, HOTPOT_TRAIN)
    if not os.path.exists(p):
        print(f"  [실패] {p} 가 없습니다. CORPUS_SOURCE='index' 로 두거나")
        print("         HotpotQA train 을 먼저 받으세요.")
        sys.exit(1)
    want = set(index_qids)
    docs, seen = [], set()
    n_rows = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r.get("id") or r.get("_id")) not in want:
                continue
            n_rows += 1
            ctx = r["context"]
            if isinstance(ctx, dict):
                pairs = zip(ctx["title"], ctx["sentences"])
            else:
                pairs = [(c[0], c[1]) for c in ctx]
            for title, sents in pairs:
                title = str(title).strip()
                body = "".join(sents).strip()
                if len(body) < MIN_DOC_CHARS or title in seen:
                    continue
                seen.add(title)
                docs.append({"title": title, "text": body})
    log(f"    HotpotQA 원본에서 {n_rows:,}개 항목의 문단을 수집")
    return docs


# ── 검색 ──────────────────────────────────────────────────────────────

class DocIndex:
    """문서를 키로 하는 인덱스. 원본 04_MAKE_VECTOR_RAG_Hotpot 과 같은 구조."""

    def __init__(self, docs, cache_key):
        from sentence_transformers import SentenceTransformer
        self.tag = cache_key.split("_")[0]
        self.docs = docs
        self.texts = [(f"{d['title']} {d['text']}" if INCLUDE_TITLE_IN_TEXT
                       and d["title"] else d["text"]) for d in docs]
        self.model = SentenceTransformer(ENCODER_MODEL, device=ENCODER_DEVICE)
        self.E = self._embed(cache_key)
        self.backend, self.index = self._build()

    def _embed(self, key):
        cache = os.path.join(OUT_DIR, f"doc_emb_{key}.npy") if USE_EMB_CACHE else None
        if cache and os.path.exists(cache):
            E = np.load(cache)
            log(f"    임베딩 캐시 {E.shape}")
            return E
        pre = "passage: " if E5_PREFIX else ""     # ★ 문서는 passage:
        t0 = time.time()
        E = self.model.encode([pre + t for t in self.texts],
                              batch_size=ENCODER_BATCH, convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=True).astype("float32")
        log(f"    임베딩 {len(self.texts):,}건 {(time.time()-t0)/60:.1f}분")
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
        pre = "query: " if E5_PREFIX else ""       # 질의는 query:
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
                    d = self.docs[int(i)]
                    res.append({"rank": rank, "score": float(v),
                                "title": d["title"], "text": d["text"]})
                out.append(res)
        return out

    def save(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        try:
            import faiss
            faiss.write_index(self.index,
                              os.path.join(outdir, f"rag_index_{self.tag}.faiss"))
        except ImportError:
            np.save(os.path.join(outdir, f"rag_embeddings_{self.tag}.npy"), self.E)
        with open(os.path.join(outdir, f"doc_titles_{self.tag}.pkl"), "wb") as f:
            pickle.dump([d["title"] for d in self.docs], f)
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
    print("84 — vanilla RAG 실행")
    print("=" * 74)
    print(f"  SPLIT_DIR : {SPLIT_DIR}")
    print(f"  OUT_DIR   : {OUT_DIR}")
    print(f"  인코더    : {ENCODER_MODEL} ({ENCODER_DEVICE})")
    print(f"  검색      : 질의 -> 문서공간,  TOP_K={TOP_K}")
    print(f"  생성기    : {GENERATOR}"
          + (f"  {VLLM_MODEL}" if GENERATOR == "vllm" else ""))
    if TOP_K != 2:
        print(f"\n  ! TOP_K={TOP_K} 입니다. 81(QI-RAG)의 CONTEXT_K 와 다르면")
        print("    문맥 수가 교락됩니다. 원본 재현 목적이 아니면 2 로 맞추세요.")

    for f in ("index_items.jsonl", "eval_queries.jsonl"):
        if not os.path.exists(os.path.join(SPLIT_DIR, f)):
            print(f"\n  [실패] {f} 가 없습니다. 80_make_split.py 를 먼저 실행하세요.")
            return

    if GENERATOR == "vllm":
        try:
            call_llm("Reply with exactly: OK")
        except Exception as e:
            print(f"\n  [실패] 생성 서버 접속 불가: {type(e).__name__}: {e}")
            return

    items = load_jsonl(os.path.join(SPLIT_DIR, "index_items.jsonl"))
    evals = load_jsonl(os.path.join(SPLIT_DIR, "eval_queries.jsonl"))
    log(f"\n  인덱스 항목 {len(items):,} / 평가 {len(evals):,}")

    # ── 1) 문서 코퍼스
    tag = CORPUS_TAG or ("dist" if CORPUS_SOURCE == "hotpot" else "gold")
    log(f"\n[1] 문서 코퍼스 구축  (source={CORPUS_SOURCE}, tag={tag})")
    if CORPUS_SOURCE == "hotpot":
        docs = build_corpus_hotpot([r["qid"] for r in items])
    else:
        docs = build_corpus(items)
    log(f"  고유 문서 {len(docs):,}")
    lens = [len(d["text"]) for d in docs]
    log(f"  문서 길이 중앙값 {int(np.median(lens))}자  범위 {min(lens)}~{max(lens)}")
    cp = os.path.join(OUT_DIR, f"doc_corpus_{tag}.jsonl")
    with open(cp, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    log(f"  저장 {cp}")

    # 평가 질의의 gold 문서가 코퍼스에 있는가 = 검색 성능의 상한
    titles = {d["title"] for d in docs if d["title"]}
    cover = sum(1 for e in evals
                if any(t in titles for t in e.get("gold_titles", [])))
    log(f"  평가 질의의 gold 문서가 코퍼스에 존재 "
        f"{cover}/{len(evals)} ({100*cover/max(len(evals),1):.1f}%)")
    if CORPUS_SOURCE == "index":
        log("  → 81 의 분할 메타와 같은 값이어야 정상 (같은 문서 집합)")
    else:
        log("  → gold+distractor 조건. 코퍼스가 커져 검색이 어려워진다.")
        log("     QI-RAG 는 매핑으로 gold 에 직접 접근하므로 영향을 받지 않는다.")

    # ── 2) 인덱스
    log("\n[2] 인덱스 구축")
    di = DocIndex(docs, cache_key=f"{tag}_{len(docs)}")
    if SAVE_INDEX:
        di.save(OUT_DIR)

    # ── 3) 변형별 검색 + 생성
    summary = {}
    for v in VARIANTS:
        if v not in evals[0]:
            continue
        log(f"\n[3-{v}] 검색 + 생성")
        qs = [e[v] for e in evals]
        hits = di.retrieve(qs, TOP_K)

        rows, pairs = [], []
        for e, h in zip(evals, hits):
            top = h[:TOP_K]
            ctx = "\n\n".join(
                (f"[{d['title']}] {d['text']}" if d["title"] else d["text"])
                for d in top)
            gold_t = set(e.get("gold_titles", []))
            hit1 = bool(top) and top[0]["title"] in gold_t
            hitk = any(d["title"] in gold_t for d in top)
            rows.append({
                "qid": e["qid"], "variant": v, "query": e[v],
                "gold": e["answer"], "gold_titles": e.get("gold_titles", []),
                "type": e.get("type"), "level": e.get("level"),
                "score": top[0]["score"] if top else 0.0,
                "matched_titles": [d["title"] for d in top],
                "retrieval_hit_top1": hit1, "retrieval_hit_topk": hitk,
                "context": ctx, "gate_abstain": False,
            })
            pairs.append((e[v], ctx))

        preds = generate(pairs, v)
        for r, p in zip(rows, preds):
            r["pred"] = p

        suffix = "" if tag == "gold" else f"_{tag}"
        p = os.path.join(OUT_DIR, f"result_vanilla{suffix}_{v}.jsonl")
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
            "mean_score": float(np.mean([r["score"] for r in rows])),
        }
        s = summary[v]
        log(f"    검색적중 top1 {s['retrieval_hit_top1']:.3f} / "
            f"top{TOP_K} {s['retrieval_hit_topk']:.3f}  "
            f"평균점수 {s['mean_score']:.3f}")

    meta = {"config": {"encoder": ENCODER_MODEL, "top_k": TOP_K,
                       "corpus_source": CORPUS_SOURCE, "tag": tag,
                       "generator": GENERATOR,
                       "gen_model": VLLM_MODEL if GENERATOR == "vllm" else None,
                       "n_docs": len(docs), "n_eval": len(evals),
                       "gold_in_corpus_rate": cover / max(len(evals), 1),
                       "prompt": QA_PROMPT,
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "summary": summary}
    with open(os.path.join(OUT_DIR, f"vanilla_summary_{tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 74)
    print("완료")
    print("=" * 74)
    print("  원시 출력만 저장했다. 채점은 83_compare.py 가 QI-RAG/RePAQ 와")
    print("  동일한 함수로 수행한다.")
    print("\n  비교의 성격")
    print("    QI-RAG vs vanilla RAG 는 '검색 축' 비교다 (질의공간 vs 문서공간).")
    print("    RePAQ vs QI-RAG 의 '페이로드' 비교와 축이 다르므로")
    print("    두 결과를 같은 표에 넣되 무엇이 다른지 명시할 것.")
    print(f"\n  산출물 접미어: {tag}")
    print("  두 조건을 모두 돌리려면 CORPUS_SOURCE 를 'index' 와 'hotpot' 으로")
    print("  각각 실행할 것. 파일명이 달라 덮어쓰지 않는다.")
    print("\n  다음: python 83_compare.py")


if __name__ == "__main__":
    main()
