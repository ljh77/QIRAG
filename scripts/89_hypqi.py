#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
89 — Hypothetical Question Indexing (HypQI) 파이프라인
=======================================================
왜 필요한가
  리뷰어 1(W1)이 지목한 문헌군 중 hypothetical question indexing
  (LlamaIndex QuestionsAnsweredExtractor 계열)은 본 연구와 가장 가깝다.
  문서 대신 질문을 인덱싱하고 페이로드로 문서를 두는 점이 같다.

  차이는 둘이다.
    질문 출처  LLM 이 문서에서 생성      vs  사람이 작성한 질의
    매핑 차수  문단 1개 -> 질문 (1:1)     vs  질문 -> 필요한 근거 전부 (1:N)

  HypQI 는 문단 단위로 질문을 만들므로 매핑이 구조적으로 1:1 이다.
  다중 근거를 요하는 질의에서 이 제약이 어떻게 작용하는지가 비교의 핵심이다.

공정성
  세 조건이 같은 문단 집합 P 를 쓴다. P 는 표집한 인덱스 항목의 gold 문단을
  중복 제거한 것이다. 문서 커버리지가 동일하므로 검색 키만 다른 비교가 된다.

    C) QI-RAG   표집 항목의 질문 q' -> 그 항목의 문맥(gold 전부)   1:N
    E) HypQI    P 의 각 문단 -> LLM 질문 생성 -> 그 문단          1:1
    D) vanilla  P 를 문서 코퍼스로 직접 검색

  평가 질의는 _split/eval_queries.jsonl 을 그대로 쓴다. 주 실험과 같은 질의다.

산출물 (OUT_DIR)
  hypqi_questions_{tag}.jsonl        생성된 질문
  result_hypqi_{tag}_{variant}.jsonl
  result_qiragsub_{tag}_{variant}.jsonl
  result_vanillasub_{tag}_{variant}.jsonl
  hypqi_meta_{tag}.json

실행:  python 89_hypqi.py
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ══════════════════════════════════════════════════════════════════════
#  ██ 설정
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = None
DATA_DIR_FALLBACK = ["/home/jun/RAG_Dataset", r"Z:\RAG_Dataset"]
SPLIT_DIR = None                 # None = <DATA_DIR>/_split
OUT_DIR = None                   # None = <DATA_DIR>/_result_hypqi

# ── 규모 ──────────────────────────────────────────────────────────────
#   표집할 인덱스 항목 수. 그 항목들의 gold 문단이 공통 문단 집합 P 가 된다.
#   실측 비율: 항목 90,447 -> 고유 문단 105,889 (약 1.17배)
#   생성 처리량 약 9.5건/초 기준 예상 시간
#     20,000 항목 -> 문단 약 23,000 -> 질문 생성 40분
#     50,000 항목 -> 문단 약 58,000 -> 질문 생성 100분
#     90,447 항목 -> 문단 약 105,000 -> 질문 생성 185분
N_INDEX_ITEMS = 90_447
K_QUESTIONS = 1                  # 문단당 생성 질문 수. 2 이상이면 인덱스가 커진다
SEED = 0

VARIANTS = ["original", "keyword", "noisy", "paraphrase_llm"]
RUN_CONDITIONS = ["hypqi", "qiragsub", "vanillasub"]

# ── 실행 모드 ─────────────────────────────────────────────────────────
#   질문 생성이 끝나면 품질을 점검하고, 검색·생성을 얼마나 돌릴지 고른다.
#   "ask"    품질 점검 후 물어본다 (기본)
#   "sample" 평가 질의 N_EVAL_SAMPLE 건만 (빠른 확인)
#   "full"   평가 질의 전부
#   "qgen"   질문 생성과 품질 점검까지만 하고 종료
RUN_MODE = "ask"
N_EVAL_SAMPLE = 200              # 샘플 모드에서 쓸 평가 질의 수
SAMPLE_VARIANTS = ["original"]   # 샘플 모드에서 돌릴 변형

# ── 검색 ──────────────────────────────────────────────────────────────
ENCODER_MODEL = "intfloat/e5-base-v2"
ENCODER_DEVICE = "cuda"
ENCODER_BATCH = 256
TOP_K = 2
CONTEXT_K = 2
USE_EMB_CACHE = True

# ── 생성 ──────────────────────────────────────────────────────────────
GENERATOR = "vllm"
VLLM_URL = "http://172.25.121.170:8005/v1/chat/completions"
VLLM_MODEL = "/home/jun/models/Qwen2.5-7B-Instruct-AWQ"
GEN_TIMEOUT = 60                 # 소켓 타임아웃
GEN_WORKERS = 16

# ── 안정성 ────────────────────────────────────────────────────────────
#   질문 생성은 7만 건 규모라 한 번에 돌리면 중간 실패 시 전부 날아간다.
#   실측: 서버가 응답을 멈춰 33분간 진행 0. 소켓 타임아웃은 연결이 살아 있으면
#   발동하지 않으므로, 배치 단위 저장과 결과 단위 타임아웃을 함께 둔다.
QGEN_BATCH = 5_000               # 이 단위로 저장하고 재시작 시 이어받는다
FUTURE_TIMEOUT = 180             # 요청 하나가 이 시간을 넘으면 포기 (초)
STALL_ABORT_MIN = 10             # 배치 안에서 이만큼 진행이 없으면 중단 (분)

# 질문 생성. LlamaIndex QuestionsAnsweredExtractor 의 취지를 따른다.
QGEN_MAX_TOKENS = 64
QGEN_PROMPT = """Here is a passage.

{ctx}

Write {k} question(s) that this passage can answer. \
Output only the question(s), one per line. Do not answer them."""

# 답 생성. 81 과 동일해야 조건 간 비교가 성립한다.
GEN_MAX_TOKENS = 128
GEN_TEMPERATURE = 0.0
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


def dump_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── LLM ───────────────────────────────────────────────────────────────

def call_llm(prompt, max_tokens):
    import urllib.request
    body = {"model": VLLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": GEN_TEMPERATURE, "max_tokens": max_tokens}
    req = urllib.request.Request(
        VLLM_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()


class Stalled(RuntimeError):
    """진행이 멈춰 중단. 서버가 응답을 붙잡고 있을 때 발생한다."""


def run_llm(prompts, max_tokens, label, offset=0, total=None):
    """결과 단위 타임아웃과 정체 감지를 둔다.
    소켓 타임아웃만으로는 '연결은 살아 있는데 응답이 안 오는' 상태를
    잡지 못한다(실측 33분 정체)."""
    if GENERATOR == "none":
        return [""] * len(prompts)
    total = total or len(prompts)

    def one(p):
        try:
            return call_llm(p, max_tokens)
        except Exception:
            return ""

    from concurrent.futures import as_completed
    t0 = time.time()
    done = 0
    last_progress = time.time()
    out = [""] * len(prompts)
    with ThreadPoolExecutor(max_workers=GEN_WORKERS) as ex:
        futs = {ex.submit(one, p): i for i, p in enumerate(prompts)}
        try:
            for f in as_completed(futs, timeout=FUTURE_TIMEOUT * len(prompts)):
                i = futs[f]
                try:
                    out[i] = f.result(timeout=FUTURE_TIMEOUT)
                except Exception:
                    out[i] = ""
                done += 1
                last_progress = time.time()
                if done % 500 == 0:
                    el = time.time() - t0
                    rate = done / max(el, 1e-9)
                    rem = (total - offset - done) / max(rate, 1e-9)
                    sys.stdout.write(
                        f"\r    [{label}] {offset+done:,}/{total:,} "
                        f"{rate:.1f}/s  남은 {rem/60:.0f}분   ")
                    sys.stdout.flush()
                if time.time() - last_progress > STALL_ABORT_MIN * 60:
                    raise Stalled(f"{STALL_ABORT_MIN}분간 진행 없음")
        except Stalled:
            sys.stdout.write("\r" + " " * 66 + "\r")
            log(f"    [{label}] ! {STALL_ABORT_MIN}분간 진행이 없어 중단합니다.")
            log("      생성 서버를 확인하세요. 저장된 배치까지는 보존됩니다.")
            for f in futs:
                f.cancel()
            raise
        except Exception as e:
            sys.stdout.write("\r" + " " * 66 + "\r")
            log(f"    [{label}] ! 중단: {type(e).__name__}: {e}")
            for f in futs:
                f.cancel()
            raise Stalled(str(e))
    sys.stdout.write("\r" + " " * 66 + "\r")
    n_empty = sum(1 for x in out if not x)
    log(f"    [{label}] {len(prompts):,}건 {(time.time()-t0)/60:.1f}분"
        + (f"  빈 응답 {n_empty}" if n_empty else ""))
    return out


def parse_questions(text, k):
    qs = [x.strip().lstrip("-•0123456789. ").strip()
          for x in (text or "").split("\n")]
    return [x for x in qs if len(x) >= 10 and "?" in x][:k]


# ── 생성 질문 품질 점검 ───────────────────────────────────────────────

_QW = re.compile(r"[A-Za-z0-9']+")
_QSTOP = set("a an the of in on for to with and or is are was were be been "
             "what which who whom how why when where does did do can could "
             "should would this that these those it its".split())


def _content(s):
    return set(w.lower() for w in _QW.findall(s) if w.lower() not in _QSTOP)


def check_questions(gen_rows, evals, n_probe=3000):
    """생성 질문이 RePAQ/QI-RAG 의 (q', c') 구조에 쓸 만한지 본다.

    핵심은 '질문이 그 문단으로 답할 수 있는가' 다. 직접 재려면 생성이 한 번
    더 필요하므로, 질문의 내용어가 문단에 얼마나 들어 있는지를 대리 지표로
    쓴다. 낮으면 LLM 이 문단을 보지 않고 질문을 만든 것이며,
    q_gen -> 문단 매핑이 무효가 되어 실험이 성립하지 않는다.
    """
    import statistics as st
    n = len(gen_rows)
    print("\n" + "-" * 70)
    print("생성 질문 품질 점검")
    print("-" * 70)
    ok = True

    noq = sum(1 for r in gen_rows if "?" not in r["question"])
    short = sum(1 for r in gen_rows if len(r["question"].split()) < 5)
    pre = sum(1 for r in gen_rows if r["question"].lower().lstrip()
              .startswith(("rewritten", "question:", "here", "sure", "1.")))
    print(f"  총 {n:,}건")
    print(f"  물음표 없음   {noq:,} ({noq/max(n,1):.1%})")
    print(f"  5단어 미만    {short:,} ({short/max(n,1):.1%})")
    print(f"  접두어 잔존   {pre:,}")
    if noq / max(n, 1) > 0.05:
        print("    ! 물음표 없는 질문이 5% 를 넘습니다. 파싱을 확인하세요.")
        ok = False

    probe = gen_rows[:n_probe]
    ov = [len(_content(r["question"]) & _content(r["text"]))
          / max(len(_content(r["question"])), 1) for r in probe]
    med = st.median(ov) if ov else 0.0
    low = sum(1 for x in ov if x < 0.5)
    print(f"\n  질문 내용어가 문단에 있는 비율 (표본 {len(probe):,})")
    print(f"    중앙값 {med:.2f}   0.5 미만 {low:,} ({low/max(len(probe),1):.1%})")
    print("    → 0.6 이상이 정상. 낮으면 LLM 이 문단을 보지 않고 만든 것이며")
    print("       q_gen -> 문단 매핑이 무효가 되어 실험이 성립하지 않는다.")
    if med < 0.6:
        print("    ! 중앙값이 0.6 미만입니다. 프롬프트나 파싱을 재검토하세요.")
        ok = False

    gl = [len(r["question"].split()) for r in gen_rows]
    el = [len(e["original"].split()) for e in evals]
    print(f"\n  길이 중앙값   생성 {st.median(gl):.0f}단어 / "
          f"평가 질의 {st.median(el):.0f}단어")
    if abs(st.median(gl) - st.median(el)) > 10:
        print("    ! 평가 질의와 길이 분포가 크게 다릅니다. 매칭이 어려울 수 있습니다.")

    c = Counter(r["question"].lower().strip() for r in gen_rows)
    dup = n - len(c)
    print(f"\n  고유 질문     {len(c):,} / {n:,}   중복 {dup:,} "
          f"({dup/max(n,1):.1%})")
    if dup / max(n, 1) > 0.05:
        print("    ! 중복이 5% 를 넘습니다. 인덱스가 낭비됩니다.")
    top = c.most_common(3)
    if top and top[0][1] > 1:
        print(f"    가장 흔한 질문: {top[0][0][:60]} ({top[0][1]}회)")

    print(f"\n  샘플 3건")
    for r in random.Random(0).sample(gen_rows, min(3, n)):
        print(f"    문단 [{r['title']}] {r['text'][:64]}")
        print(f"    질문 {r['question'][:70]}")
    print("-" * 70)
    print(f"  판정: {'사용 가능' if ok else '재검토 필요'}")
    return ok, {"n": n, "no_question_mark": noq, "too_short": short,
                "prefix_left": pre, "overlap_median": med,
                "overlap_low": low / max(len(probe), 1),
                "len_median_gen": st.median(gl), "len_median_eval": st.median(el),
                "unique": len(c), "dup_rate": dup / max(n, 1), "pass": ok}


def ask_mode(default="sample"):
    print("\n  다음 단계를 고르세요.")
    print(f"    1) 샘플 테스트   평가 질의 {N_EVAL_SAMPLE}건, "
          f"변형 {SAMPLE_VARIANTS}   (몇 분)")
    print("    2) 전체 실행     평가 질의 전부, 변형 전부")
    print("    3) 중단          질문 생성 결과만 남기고 종료")
    try:
        a = input("  번호: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  입력이 없어 샘플 테스트로 진행합니다.")
        return default
    return {"1": "sample", "2": "full", "3": "stop"}.get(a, default)


# ── 검색 ──────────────────────────────────────────────────────────────

class Index:
    def __init__(self, texts, prefix, cache_key):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(ENCODER_MODEL, device=ENCODER_DEVICE)
        self.prefix = prefix
        self.E = self._embed(texts, cache_key)
        self.backend, self.idx = self._build()

    def _embed(self, texts, key):
        cache = (os.path.join(OUT_DIR, f"emb_{key}.npy")
                 if USE_EMB_CACHE else None)
        if cache and os.path.exists(cache):
            E = np.load(cache)
            if E.shape[0] == len(texts):
                log(f"    임베딩 캐시 {E.shape}")
                return E
        t0 = time.time()
        E = self.m.encode([self.prefix + t for t in texts],
                          batch_size=ENCODER_BATCH, convert_to_numpy=True,
                          normalize_embeddings=True,
                          show_progress_bar=True).astype("float32")
        log(f"    임베딩 {len(texts):,}건 {(time.time()-t0)/60:.1f}분")
        if cache:
            np.save(cache, E)
        return E

    def _build(self):
        try:
            import faiss
            ix = faiss.IndexFlatIP(self.E.shape[1])
            ix.add(self.E)
            return "faiss", ix
        except ImportError:
            log("    faiss 없음 → numpy 완전탐색 (결과 동일)")
            return "numpy", None

    def search(self, queries, k=TOP_K, batch=512):
        out = []
        for s in range(0, len(queries), batch):
            Q = self.m.encode(["query: " + q for q in queries[s:s + batch]],
                              batch_size=ENCODER_BATCH, convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=False).astype("float32")
            if self.backend == "faiss":
                sc, ix = self.idx.search(Q, k)
            else:
                S = Q @ self.E.T
                kk = min(k, S.shape[1])
                ix = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
                rows = np.arange(S.shape[0])[:, None]
                order = np.argsort(-S[rows, ix], axis=1)
                ix = ix[rows, order]
                sc = S[rows, ix]
            for row_s, row_i in zip(sc, ix):
                out.append([(int(i), float(v)) for i, v in zip(row_i, row_s)])
        return out


# ══════════════════════════════════════════════════════════════════════

def main():
    global DATA_DIR, SPLIT_DIR, OUT_DIR
    if DATA_DIR is None:
        DATA_DIR = _auto()
    if SPLIT_DIR is None:
        SPLIT_DIR = os.path.join(DATA_DIR, "_split")
    if OUT_DIR is None:
        OUT_DIR = os.path.join(DATA_DIR, "_result_hypqi")
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{N_INDEX_ITEMS // 1000}k"

    print("=" * 76)
    print("89 — Hypothetical Question Indexing (HypQI)")
    print("=" * 76)
    print(f"  SPLIT_DIR : {SPLIT_DIR}")
    print(f"  OUT_DIR   : {OUT_DIR}   태그 {tag}")
    print(f"  표집 항목 : {N_INDEX_ITEMS:,}   문단당 질문 {K_QUESTIONS}")
    print(f"  검색      : TOP_K={TOP_K} CONTEXT_K={CONTEXT_K}")
    print(f"  조건      : {RUN_CONDITIONS}")
    print()
    print("  세 조건이 같은 문단 집합을 쓴다. 검색 키만 다르다.")
    print("    C) QI-RAG    질문 q' -> gold 문단 전부      1:N")
    print("    E) HypQI     문단 -> LLM 질문 -> 그 문단     1:1")
    print("    D) vanilla   문단 직접 검색")

    for f in ("index_items.jsonl", "eval_queries.jsonl"):
        if not os.path.exists(os.path.join(SPLIT_DIR, f)):
            print(f"\n  [실패] {f} 가 없습니다. 80_make_split.py 를 먼저 실행하세요.")
            return
    if GENERATOR == "vllm":
        try:
            call_llm("Reply with exactly: OK", 8)
        except Exception as e:
            print(f"\n  [실패] 생성 서버 접속 불가: {type(e).__name__}: {e}")
            return

    items = load_jsonl(os.path.join(SPLIT_DIR, "index_items.jsonl"))
    evals = load_jsonl(os.path.join(SPLIT_DIR, "eval_queries.jsonl"))
    rng = random.Random(SEED)
    rng.shuffle(items)
    sub = items[:N_INDEX_ITEMS] if N_INDEX_ITEMS else items
    log(f"\n[1] 표집  인덱스 항목 {len(sub):,} / 평가 {len(evals):,}")

    # ── 공통 문단 집합 P
    pat = re.compile(r"\[(.*?)\]\s*(.*?)(?=\s*\[[^\]]+\]\s|$)", re.S)
    P, seen = [], set()
    for r in sub:
        for title, body in pat.findall(r.get("context", "")):
            title, body = title.strip(), body.strip()
            if not body or title in seen:
                continue
            seen.add(title)
            P.append({"title": title, "text": body})
    log(f"  공통 문단 집합 P  {len(P):,}개  (항목 대비 {len(P)/max(len(sub),1):.2f}배)")

    titles = {d["title"] for d in P}
    cover = sum(1 for e in evals
                if all(t in titles for t in e.get("gold_titles", [])))
    cover1 = sum(1 for e in evals
                 if any(t in titles for t in e.get("gold_titles", [])))
    log(f"  평가 질의의 gold 문단이 P 에 전부 존재 "
        f"{cover}/{len(evals)} ({cover/len(evals):.3f})   "
        f"하나라도 {cover1/len(evals):.3f}")
    log("  → 이 값이 세 조건 공통의 상한이다.")

    est = len(P) * K_QUESTIONS / 9.5 / 60
    log(f"\n  질문 생성 예상 {est:.0f}분 (9.5건/초 기준)")

    # ── 2) HypQI 질문 생성 (배치 저장 + 재개)
    qpath = os.path.join(OUT_DIR, f"hypqi_questions_{tag}.jsonl")
    dpath = os.path.join(OUT_DIR, f"hypqi_done_{tag}.jsonl")   # 처리한 문단 기록
    log(f"\n[2] 질문 생성  (배치 {QGEN_BATCH:,}건 단위 저장)")

    done_titles = set()
    if os.path.exists(dpath):
        n_zero = 0
        for r in load_jsonl(dpath):
            # 질문을 하나도 못 뽑은 문단은 재시도 대상으로 둔다.
            if r.get("n", 1) > 0:
                done_titles.add(r["title"])
            else:
                n_zero += 1
        log(f"  이미 처리한 문단 {len(done_titles):,}건 — 이어서 진행합니다.")
        if n_zero:
            log(f"  질문을 못 뽑은 문단 {n_zero:,}건은 다시 시도합니다.")

    todo = [d for d in P if d["title"] not in done_titles]
    if not todo:
        log("  전부 처리됨.")
    else:
        log(f"  남은 문단 {len(todo):,}건")
        n_fail = 0
        for s0 in range(0, len(todo), QGEN_BATCH):
            chunk = todo[s0:s0 + QGEN_BATCH]
            prompts = [QGEN_PROMPT.format(ctx=f"[{d['title']}] {d['text']}",
                                          k=K_QUESTIONS) for d in chunk]
            try:
                raw = run_llm(prompts, QGEN_MAX_TOKENS, "qgen",
                              offset=len(done_titles), total=len(P))
            except Stalled:
                log(f"\n  중단됨. 저장된 질문 {len(done_titles):,}건은 보존됩니다.")
                log(f"  서버 확인 후 같은 명령을 다시 실행하면 이어서 진행합니다.")
                return
            batch_q, batch_d = [], []
            for d, t in zip(chunk, raw):
                qs = parse_questions(t, K_QUESTIONS)
                if not qs:
                    n_fail += 1
                for q in qs:
                    batch_q.append({"question": q, "title": d["title"],
                                    "text": d["text"]})
                batch_d.append({"title": d["title"], "n": len(qs)})
            # 이어쓰기 — 중간에 죽어도 여기까지는 남는다
            with open(qpath, "a", encoding="utf-8") as f:
                for r in batch_q:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(dpath, "a", encoding="utf-8") as f:
                for r in batch_d:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done_titles.update(d["title"] for d in chunk)
            log(f"    저장 {len(done_titles):,}/{len(P):,}문단  "
                f"누적 질문 {sum(1 for _ in open(qpath, encoding='utf-8')):,}")

        log(f"  질문 생성 완료  실패 문단 {n_fail:,}")

    gen_rows = load_jsonl(qpath) if os.path.exists(qpath) else []
    if not gen_rows:
        print("  [실패] 생성된 질문이 없습니다.")
        return
    log(f"  총 생성 질문 {len(gen_rows):,}  (문단 {len(P):,})")

    # ── 2-b) 품질 점검
    q_ok, q_stat = check_questions(gen_rows, evals)

    mode = RUN_MODE
    if mode == "ask":
        if not q_ok:
            print("\n  ! 품질 점검에서 문제가 발견됐습니다.")
            print("    그대로 진행하면 결과를 신뢰할 수 없습니다.")
        mode = ask_mode()
    if mode == "qgen" or mode == "stop":
        print("\n  질문 생성까지만 수행하고 종료합니다.")
        print(f"  파일: {qpath}")
        print("  이어서 돌리려면 같은 명령을 다시 실행하세요"
              " (질문 생성은 재사용됩니다).")
        return

    if mode == "sample":
        evals = evals[:N_EVAL_SAMPLE]
        variants_run = [v for v in SAMPLE_VARIANTS if v in VARIANTS]
        tag = tag + "_s"
        log(f"\n  [샘플 모드] 평가 {len(evals)}건 / 변형 {variants_run}")
        log(f"  산출물 태그 {tag} — 전체 실행 결과를 덮어쓰지 않습니다.")
    else:
        variants_run = VARIANTS
        log(f"\n  [전체 실행] 평가 {len(evals):,}건 / 변형 {variants_run}")

    # ── 3) 인덱스
    log(f"\n[3] 인덱스 구축")
    idxs = {}
    if "hypqi" in RUN_CONDITIONS:
        log("  HypQI  (생성 질문)")
        idxs["hypqi"] = Index([r["question"] for r in gen_rows], "query: ",
                              f"hypq_{tag}_{len(gen_rows)}")
    if "qiragsub" in RUN_CONDITIONS:
        log("  QI-RAG (사람 질문)")
        idxs["qiragsub"] = Index([r["question"] for r in sub], "query: ",
                                 f"qsub_{tag}_{len(sub)}")
    if "vanillasub" in RUN_CONDITIONS:
        log("  vanilla (문단)")
        idxs["vanillasub"] = Index([f"{d['title']} {d['text']}" for d in P],
                                   "passage: ", f"dsub_{tag}_{len(P)}")

    # ── 4) 변형별 검색 + 생성
    summary = {}
    for v in variants_run:
        if v not in evals[0]:
            continue
        qs = [e[v] for e in evals]
        log(f"\n[4-{v}]")
        for cond, ix in idxs.items():
            hits = ix.search(qs, TOP_K)
            rows, pairs = [], []
            for e, h in zip(evals, hits):
                top = h[:CONTEXT_K]
                if cond == "hypqi":
                    parts, mq = [], []
                    for j, _ in top:
                        g = gen_rows[j]
                        parts.append(f"[{g['title']}] {g['text']}")
                        mq.append(g["question"])
                elif cond == "qiragsub":
                    parts = [sub[j]["context"] for j, _ in top]
                    mq = [sub[j]["question"] for j, _ in top]
                else:
                    parts = [f"[{P[j]['title']}] {P[j]['text']}" for j, _ in top]
                    mq = [P[j]["title"] for j, _ in top]
                ctx = "\n\n".join(parts)
                rows.append({
                    "qid": e["qid"], "variant": v, "query": e[v],
                    "gold": e["answer"], "gold_titles": e.get("gold_titles", []),
                    "type": e.get("type"), "level": e.get("level"),
                    "score": top[0][1] if top else 0.0,
                    "matched_questions": mq,
                    "context": ctx, "gate_abstain": False,
                })
                pairs.append(QA_PROMPT.format(ctx=ctx, q=e[v]))
            try:
                preds = run_llm(pairs, GEN_MAX_TOKENS, f"{cond}/{v}")
            except Stalled:
                log(f"    [{cond}/{v}] 중단. 이 조건은 건너뜁니다.")
                continue
            for r, p in zip(rows, preds):
                r["pred"] = p
            fp = os.path.join(OUT_DIR, f"result_{cond}_{tag}_{v}.jsonl")
            dump_jsonl(fp, rows)
            full = float(np.mean([
                all(f"[{t}]" in r["context"] for t in r["gold_titles"])
                for r in rows]))
            part = float(np.mean([
                sum(1 for t in r["gold_titles"] if f"[{t}]" in r["context"]) == 1
                for r in rows]))
            summary.setdefault(v, {})[cond] = {
                "n": len(rows), "gold_all": full, "gold_one_only": part}
            log(f"    {cond:<12} gold 전부 {full:.3f}  하나만 {part:.3f}")

    meta = {"question_quality": q_stat,
            "config": {"run_mode": mode, "n_eval": len(evals),
                       "variants_run": variants_run,
                       "n_index_items": len(sub), "n_passages": len(P),
                       "k_questions": K_QUESTIONS, "n_gen_questions": len(gen_rows),
                       "encoder": ENCODER_MODEL, "top_k": TOP_K,
                       "context_k": CONTEXT_K, "seed": SEED, "tag": tag,
                       "gen_model": VLLM_MODEL, "qgen_prompt": QGEN_PROMPT,
                       "qa_prompt": QA_PROMPT,
                       "gold_all_in_P": cover / len(evals),
                       "gold_any_in_P": cover1 / len(evals),
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "summary": summary}
    with open(os.path.join(OUT_DIR, f"hypqi_meta_{tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 76)
    print("완료")
    print("=" * 76)
    print(f"  태그 {tag}   채점은 89_2_compare_hypqi.py 가 수행한다.")
    if mode == "sample":
        print("  샘플 결과입니다. 전체를 돌리려면 다시 실행해 2번을 고르세요.")
        print("  질문 생성은 재사용되므로 검색·생성만 다시 합니다.")
    print("\n  규모를 바꾸려면 N_INDEX_ITEMS 를 고쳐 다시 실행할 것.")
    print("  산출물 파일명에 태그가 붙어 덮어쓰지 않는다.")
    print(f"\n  다음: python 89_2_compare_hypqi.py")


if __name__ == "__main__":
    main()
