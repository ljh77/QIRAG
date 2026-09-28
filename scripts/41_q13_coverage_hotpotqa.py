#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
41 — Q13 항목당 커버리지 (HotpotQA)
====================================
논문 특장점 논거의 직접 측정. 80_make_split.py 산출물을 읽는다.

  주장: a' 는 단일 사실, c' 는 그 사실을 포함하는 문서다.
        따라서 인덱스 항목 1개가 덮는 질의 공간이 c' 쪽이 넓다.

설계 — 완전히 통제된 비교
  같은 문서(gold title)를 공유하는 질문들은 문맥이 같고 답이 다르다.

    문단: Jean Loring
      Q1 Who published the comic book...      A: DC Comics
      Q2 Jean Loring was the primary love...  A: The Atom
      Q3 In what comic book did...            A: "The Flash"

  각 그룹에서 질문 하나를 빼서 평가 질의로 쓰고(leave-one-out) 나머지를
  인덱스에 넣는다. 인덱스의 q' 집합은 두 조건이 동일하고 페이로드만 다르다.

    Q-A (RePAQ 등가) : (q', a') 저장, 검색된 a' 를 그대로 반환
    Q-C (QI-RAG)     : (q', c') 저장, 검색된 c' 를 문맥으로 반환

  평가 질의는 인덱스에 없다. 검색은 같은 문단의 '다른' 질문을 찾게 되고,
  그때 페이로드가 답을 담고 있는지가 갈린다.

HotpotQA 를 쓰는 이유
  PAQ 는 자동 생성 질문이라 같은 문단에서 유사 질문이 나온다.
  HotpotQA 는 사람이 만든 서로 다른 질문이며, 실측상
    2건 이상 공유 문서 36,126개, 그중 답이 다른 그룹 33,590개(93%)
  로 Q13 의 전제가 더 깨끗하게 성립한다.
  또한 type(bridge/comparison) 과 level 축으로 하위 분석이 가능하다.

측정
  M1 항목당 커버리지  인덱스 항목 하나가 그룹의 몇 개 질문에 답할 수 있는가
  M2 leave-one-out    검색+페이로드로 held-out 질의에 답할 수 있는가
  M3 분해              검색 성공률 x 페이로드 성공률
  M4 커버리지 절단     문서를 비율별로 제거했을 때의 열화
  McNemar              동일 질의에 대한 대응 비교

생성기를 쓰지 않는다. Q-C 의 '답 포함' 은 생성기 성능의 상한이므로
논문에서는 상한으로 명시해 보고할 것.

실행:  python 41_q13_coverage_hotpotqa.py
"""

import json
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
DATA_DIR = None                  # None = 자동 인식 (_split 폴더가 있는 곳)
SPLIT_DIR = None                 # None = <DATA_DIR>/_split   (80 산출물)
OUT_DIR = None                   # None = <DATA_DIR>/_result

# 그룹 필터
#   답이 모두 같은 그룹은 Q-A 도 우연히 맞으므로 Q13 의 전제가 성립하지 않는다.
#   실측: 전체 36,126 중 답이 다른 그룹 33,590 (93%)
REQUIRE_DIFFERENT_ANSWERS = True
MIN_GROUP_SIZE = 2

# ★ yes/no 답 제외
#   Q-C 판정은 '답 문자열이 문맥에 있는가' 다. 그런데 comparison 유형의 답인
#   yes/no 는 문맥에 그 문자열로 등장하지 않는다(문맥은 두 엔티티의 사실을
#   서술할 뿐 "yes" 라고 쓰여 있지 않다). 따라서 이 유형에서 Q-C 는
#   구조적으로 0 이 되고, Q-A 는 두 값 중 하나라 우연히 50% 맞는다.
#   측정 대상으로 부적합하므로 기본값은 제외다.
#   포함해 보고하려면 False 로 두고 type 별 분해를 반드시 함께 실을 것.
EXCLUDE_YESNO = True
YESNO_SET = {"yes", "no"}

# 검색기
#   "st"   : sentence-transformers (논문 실험)
#   "word" : tfidf 단어 (빠른 확인)
ANALYZER = "st"
ST_MODEL = "intfloat/e5-base-v2"
ST_DEVICE = "cuda"
ST_BATCH = 256
ST_CACHE = True

# 규모. 그룹(문서) 수 기준. None 이면 전체.
MAX_PASSAGES = 40_000
SEEDS = [0, 1, 2]

# ★ 원본 QI-RAG 노트북(23-QI-RAG-h.ipynb)과 맞춘 값
#   retrieve_qi_rag(query, top_k=2) 로 문서 2개를 검색하고
#   "\n\n".join(...) 으로 이어 붙여 LLM 에 넘긴다.
#   이전 측정은 CONTEXT_K=1 이어서 Q-C 커버리지를 과소평가했다.
TOP_K = 5              # 검색 상위 k (지표 계산용)
CONTEXT_K = 2          # 실제로 문맥으로 쓰는 상위 문서 수 (원본 TOPK=2)

# 커버리지 절단 비율 (문단 단위로 제거)
COVERAGE_LEVELS = [1.0, 0.75, 0.50, 0.25]

# 정답 매칭: 정규화 후 부분 문자열 포함 여부
# (c' 는 문단이므로 EM 이 아니라 포함으로 판정한다)
BREAKDOWN = ["type", "level"]
VERBOSE = True
# ══════════════════════════════════════════════════════════════════


def _auto():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.getcwd(), here] + DATA_DIR_FALLBACK:
        if c and os.path.isdir(os.path.join(c, "_split")):
            return c
    for c in [os.getcwd(), here] + DATA_DIR_FALLBACK:
        if c and os.path.isdir(c):
            return c
    return os.getcwd()


def log(*a):
    if VERBOSE:
        print(*a)


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
    return any(p == norm(g) for g in golds if g)


def contains(text, golds):
    t = norm(text)
    return any(norm(g) and norm(g) in t for g in golds if g)


# ────────────────────────────────────────────── 검색기

class STRetriever:
    def __init__(self, texts, cache_key=None):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(ST_MODEL, device=ST_DEVICE)
        self.e5 = "e5" in ST_MODEL.lower()
        cache = os.path.join(OUT_DIR, f"q13hp_emb_{cache_key}.npy") if (
            ST_CACHE and cache_key) else None
        if cache and os.path.exists(cache):
            self.E = np.load(cache)
            log(f"    임베딩 캐시 로드 {self.E.shape}")
            return
        t0 = time.time()
        pre = "passage: " if self.e5 else ""
        self.E = self.m.encode([pre + t for t in texts], batch_size=ST_BATCH,
                               convert_to_numpy=True, normalize_embeddings=True,
                               show_progress_bar=True).astype("float32")
        log(f"    임베딩 {len(texts):,}건 {(time.time()-t0)/60:.1f}분")
        if cache:
            np.save(cache, self.E)

    def search(self, queries, k=TOP_K, batch=1024):
        pre = "query: " if self.e5 else ""
        out = []
        for s in range(0, len(queries), batch):
            Q = self.m.encode([pre + q for q in queries[s:s + batch]],
                              batch_size=ST_BATCH, convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=False).astype("float32")
            S = Q @ self.E.T
            for row in S:
                kk = min(k, len(row))
                top = np.argpartition(-row, kk - 1)[:kk]
                top = top[np.argsort(-row[top])]
                out.append([(int(j), float(row[j])) for j in top])
        return out


class WordRetriever:
    def __init__(self, texts, cache_key=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        self.vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                   sublinear_tf=True)
        self._n = normalize
        t0 = time.time()
        self.E = normalize(self.vec.fit_transform(texts))
        log(f"    인덱스 {len(texts):,}건 {self.E.shape[1]:,}차원 "
            f"{time.time()-t0:.1f}초")

    def search(self, queries, k=TOP_K, batch=2048):
        out = []
        for s in range(0, len(queries), batch):
            Q = self._n(self.vec.transform(queries[s:s + batch]))
            S = np.asarray((Q @ self.E.T).todense())
            for row in S:
                kk = min(k, len(row))
                top = np.argpartition(-row, kk - 1)[:kk]
                top = top[np.argsort(-row[top])]
                out.append([(int(j), float(row[j])) for j in top])
        return out


def make_retriever(texts, cache_key=None):
    return (STRetriever if ANALYZER == "st" else WordRetriever)(texts, cache_key)


# ────────────────────────────────────────────── 데이터

def load_triples():
    """80_make_split.py 산출물을 읽는다.
    index_items.jsonl 의 각 항목이 (q', a', c') 삼중쌍이고,
    passage_groups.json 이 문서 제목 -> qid 목록 매핑이다."""
    p = os.path.join(SPLIT_DIR, "index_items.jsonl")
    g = os.path.join(SPLIT_DIR, "passage_groups.json")
    if not (os.path.exists(p) and os.path.exists(g)):
        print(f"{SPLIT_DIR} 에 index_items.jsonl / passage_groups.json 이 없습니다.")
        print("80_make_split.py 를 먼저 실행하세요.")
        sys.exit(1)
    tri = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        # 41 이 기대하는 키로 맞춘다. passage_id 는 그룹 키(문서 제목).
        tri[r["qid"]] = {"qid": r["qid"], "question": r["question"],
                         "answer": r["answer"], "context": r["context"],
                         "gold_titles": r.get("gold_titles", []),
                         "type": r.get("type"), "level": r.get("level")}
    if EXCLUDE_YESNO:
        before = len(tri)
        tri = {k: v for k, v in tri.items()
               if norm(as_list(v["answer"])[0] if as_list(v["answer"]) else "")
               not in YESNO_SET}
        log(f"  yes/no 답 제외: 항목 {before:,} -> {len(tri):,}")

    groups = json.load(open(g, encoding="utf-8"))
    groups = {k: [q for q in v if q in tri] for k, v in groups.items()}
    groups = {k: v for k, v in groups.items() if len(v) >= MIN_GROUP_SIZE}

    if REQUIRE_DIFFERENT_ANSWERS:
        before = len(groups)
        kept = {}
        for k, qids in groups.items():
            ans = {norm(as_list(tri[q]["answer"])[0] if as_list(tri[q]["answer"])
                        else "") for q in qids}
            if len(ans) >= 2:
                kept[k] = qids
        groups = kept
        log(f"  답이 다른 그룹만 유지: {before:,} -> {len(groups):,}")
    return tri, groups


# ────────────────────────────────────────────── M1 항목당 커버리지

def per_item_coverage(tri, groups):
    """인덱스 항목 하나가 그룹의 몇 개 질문에 답할 수 있는가.
    Q-A: 자기 질문 1개 (정의상)
    Q-C: c' 안에 답 문자열이 있는 질문 수
    """
    qa, qc, sizes = [], [], []
    for pid, qids in groups.items():
        qs = [tri[q] for q in qids if q in tri]
        if len(qs) < MIN_GROUP_SIZE:
            continue
        # ★ HotpotQA 는 항목마다 gold 문단이 2개이고 c' 가 그 둘의 결합이다.
        #   따라서 같은 그룹이라도 c' 가 완전히 같지는 않다.
        #   항목당 커버리지는 '인덱스 항목 1개' 기준이므로 첫 항목의 c' 로 잰다.
        ctx = qs[0]["context"]
        # CONTEXT_K>1 이면 실제로는 다른 문단도 함께 들어가지만,
        # 항목당 커버리지는 '인덱스 항목 1개' 기준이므로 단일 문맥으로 잰다.
        n_cov = sum(1 for r in qs if contains(ctx, as_list(r["answer"])))
        qa.append(1)
        qc.append(n_cov)
        sizes.append(len(qs))
    return {
        "n_groups": len(qa),
        "mean_group_size": float(np.mean(sizes)) if sizes else 0.0,
        "qa_mean": float(np.mean(qa)) if qa else 0.0,
        "qc_mean": float(np.mean(qc)) if qc else 0.0,
        "qc_ratio": float(np.mean([c / s for c, s in zip(qc, sizes)])) if qc else 0.0,
        "qc_hist": dict(Counter(qc)),
        # PAQ 전체 기준(단일 질문 문단 포함) 추정치.
        # MIN_Q_PER_PASSAGE 필터 때문에 위 평균은 과대평가된다.
        "paq_unconditional_est": 65_000_000 / 21_000_000,
    }


# ────────────────────────────────────────────── M2/M3 leave-one-out

def build_split(tri, groups, seed, keep_ratio=1.0):
    """문단마다 질문 하나를 평가로 빼고 나머지를 인덱스에 넣는다."""
    rng = random.Random(seed)
    pids = sorted(groups)
    rng.shuffle(pids)
    if MAX_PASSAGES:
        pids = pids[:MAX_PASSAGES]
    if keep_ratio < 1.0:
        pids_idx = set(pids[:int(len(pids) * keep_ratio)])
    else:
        pids_idx = set(pids)

    index, evals = [], []
    seen_idx = set()
    for pid in pids:
        qids = [q for q in groups[pid] if q in tri]
        if len(qids) < MIN_GROUP_SIZE:
            continue
        rng.shuffle(qids)
        held, rest = qids[0], qids[1:]
        evals.append({"pid": pid, **tri[held]})
        if pid in pids_idx:
            for q in rest:
                # 한 질문이 여러 문서 그룹에 속할 수 있다. 중복 삽입을 막는다.
                if q in seen_idx:
                    continue
                seen_idx.add(q)
                it = dict(tri[q])
                it["pid"] = pid          # 이 항목이 대표하는 문서
                index.append(it)
    return index, evals


def run_loo(index, evals, seed, tag):
    """검색은 한 번만 하고 페이로드만 바꿔 평가한다(완전 통제)."""
    if not index or not evals:
        return None
    R = make_retriever([r["question"] for r in index],
                       cache_key=f"hp_{ANALYZER}_{tag}_{len(index)}_{seed}")
    hits = R.search([e["question"] for e in evals], k=TOP_K)

    rows = []
    for e, h in zip(evals, hits):
        gold = as_list(e["answer"])
        j, score = h[0]
        item = index[j]
        # 회수한 항목이 평가 질의와 같은 문서를 공유하는가
        same_passage = e["pid"] in set(item.get("gold_titles", []))

        # ── Q-A: 저장된 답을 그대로 반환 (RePAQ 의 페이로드 정책)
        #    답은 단일 항목이므로 top-1 만 쓴다.
        qa_ok = em(str(as_list(item["answer"])[0] if as_list(item["answer"]) else ""),
                   gold)
        qa_ok_k = any(
            em(str(as_list(index[jj]["answer"])[0]
                   if as_list(index[jj]["answer"]) else ""), gold)
            for jj, _ in h[:CONTEXT_K])

        # ── Q-C: 원본과 동일하게 상위 CONTEXT_K 문서를 이어 붙인다
        #    ("\n\n".join([doc['matched_document'] for doc in r]))
        ctx_k = "\n\n".join(index[jj]["context"] for jj, _ in h[:CONTEXT_K])
        qc_ok = contains(ctx_k, gold)
        qc_ok_top1 = contains(item["context"], gold)     # 이전 측정과의 대조용
        same_passage_k = any(e["pid"] in set(index[jj].get("gold_titles", []))
                             for jj, _ in h[:CONTEXT_K])

        k_same = any(e["pid"] in set(index[jj].get("gold_titles", []))
                     for jj, _ in h)
        rows.append({"margin": score, "same_passage": same_passage,
                     "type": e.get("type"), "level": e.get("level"),
                     "same_passage_k": same_passage_k,
                     "k_same": k_same,
                     "qa": qa_ok, "qa_k": qa_ok_k,
                     "qc": qc_ok, "qc_top1": qc_ok_top1,
                     "qc_spurious": bool(qc_ok and not same_passage_k),
                     "qa_diff_answer": bool(same_passage and not qa_ok)})
    return rows


def summarize(rows):
    n = len(rows)
    sp = sum(r["same_passage"] for r in rows) / n
    spk = sum(r["same_passage_k"] for r in rows) / n
    ks = sum(r["k_same"] for r in rows) / n
    qa = sum(r["qa"] for r in rows) / n
    qa_k = sum(r["qa_k"] for r in rows) / n
    qc = sum(r["qc"] for r in rows) / n
    qc1 = sum(r["qc_top1"] for r in rows) / n
    # 검색이 같은 문단을 찾은 경우에 한정한 페이로드 성공률
    sub = [r for r in rows if r["same_passage"]]
    subk = [r for r in rows if r["same_passage_k"]]
    qa_c = sum(r["qa"] for r in sub) / len(sub) if sub else float("nan")
    qc_c = sum(r["qc"] for r in subk) / len(subk) if subk else float("nan")
    spur = sum(r["qc_spurious"] for r in rows) / n
    diff = (sum(r["qa_diff_answer"] for r in rows) / len(sub)) if sub else float("nan")
    return {"n": n,
            "retrieval_same_passage": sp,          # top-1
            "retrieval_same_passage_k": spk,       # top-CONTEXT_K
            "retrieval_topk_same": ks,
            "qa_acc": qa, "qa_acc_k": qa_k,
            "qc_acc": qc, "qc_acc_top1": qc1,
            "qa_given_retrieval": qa_c, "qc_given_retrieval": qc_c,
            "qc_spurious_rate": spur, "qa_diff_answer_rate": diff,
            "qc_over_qa": qc / qa if qa else float("nan"),
            "qc_gain_from_k": qc - qc1}


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    n10 = sum(1 for x, y in zip(a, b) if x and (not y))
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "stat": 0.0, "p": 1.0}
    import math
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"n01": n01, "n10": n10, "stat": float(stat),
            "p": float(math.erfc(math.sqrt(stat / 2)))}


def bootstrap_ci(v, n=2000, seed=0):
    v = [x for x in v if x == x]
    if len(v) < 2:
        return (v[0] if v else float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(v) for _ in range(len(v))) / len(v) for _ in range(n))
    return (sum(v) / len(v), ms[int(0.025 * n)], ms[int(0.975 * n) - 1])


# ────────────────────────────────────────────── main

def main():
    global DATA_DIR, SPLIT_DIR, OUT_DIR
    if DATA_DIR is None:
        DATA_DIR = _auto()
    if SPLIT_DIR is None:
        SPLIT_DIR = os.path.join(DATA_DIR, "_split")
    if OUT_DIR is None:
        OUT_DIR = os.path.join(DATA_DIR, "_result")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 76)
    print("41 — Q13 항목당 커버리지 (HotpotQA, Q-A vs Q-C)")
    print("=" * 76)
    print(f"  DATA_DIR : {DATA_DIR}")
    print(f"  검색기   : {ANALYZER}" + (f" ({ST_MODEL})" if ANALYZER == "st" else ""))
    print(f"  문서 상한: {MAX_PASSAGES or '전체'}   시드 {SEEDS}")
    print(f"  그룹 조건: 크기>={MIN_GROUP_SIZE}"
          + ("  답이 서로 다른 그룹만" if REQUIRE_DIFFERENT_ANSWERS else "")
          + ("  yes/no 제외" if EXCLUDE_YESNO else "  yes/no 포함"))
    if not EXCLUDE_YESNO:
        print("  ! yes/no 답은 문맥에 그 문자열로 등장하지 않아 Q-C 가")
        print("    구조적으로 0 이 된다. type 별 분해를 반드시 함께 보고할 것.")
    print(f"  CONTEXT_K: {CONTEXT_K}  (원본 노트북 TOPK=2 와 동일)")

    tri, groups = load_triples()
    gs = Counter(len(v) for v in groups.values())
    print(f"  인덱스 항목 {len(tri):,} / 그룹 {len(groups):,}"
          f"  평균 {sum(k*n for k,n in gs.items())/max(len(groups),1):.2f}개")

    results = {"config": {"analyzer": ANALYZER, "max_passages": MAX_PASSAGES,
                          "seeds": SEEDS, "top_k": TOP_K,
                          "exclude_yesno": EXCLUDE_YESNO,
                          "require_different_answers": REQUIRE_DIFFERENT_ANSWERS,
                          "st_model": ST_MODEL if ANALYZER == "st" else None}}

    # ── M1
    print("\n[M1] 항목당 커버리지 — 인덱스 항목 1개가 덮는 질문 수")
    sub = dict(list(groups.items())[:MAX_PASSAGES]) if MAX_PASSAGES else groups
    m1 = per_item_coverage(tri, sub)
    print(f"  그룹 {m1['n_groups']:,}개, 평균 크기 {m1['mean_group_size']:.2f}")
    print(f"  Q-A (a' 저장) : {m1['qa_mean']:.2f}개  (정의상 자기 질문 1개)")
    print(f"  Q-C (c' 저장) : {m1['qc_mean']:.2f}개  "
          f"(그룹의 {100*m1['qc_ratio']:.1f}%)")
    print(f"  → 항목당 커버리지 비 {m1['qc_mean']/max(m1['qa_mean'],1e-9):.2f}배")
    print(f"  분포: {dict(sorted(m1['qc_hist'].items())[:8])}")
    if m1["qc_ratio"] > 0.99:
        print("  ! 그룹의 거의 100% 가 덮인다면 확인이 필요하다.")
        print("    HotpotQA 는 질문마다 gold 문단이 2개이고 c' 가 그 결합이므로,")
        print("    같은 그룹이라도 c' 가 완전히 같지는 않다.")
    print("  ! 이 평균은 2건 이상 공유 문서만 대상으로 한 값이다.")
    print("    HotpotQA train 전체 기준 gold 문서 105,570개 중")
    print("    2건 이상 공유가 36,126개(34.2%)이며, 나머지는 단일 질문 문서다.")
    print("    두 값을 함께 보고할 것.")
    results["m1_per_item_coverage"] = m1

    # ── M2/M3
    print("\n[M2/M3] leave-one-out — 평가 질의는 인덱스에 없음")
    per_seed = []
    for seed in SEEDS:
        index, evals = build_split(tri, groups, seed)
        rows = run_loo(index, evals, seed, "full")
        if not rows:
            continue
        s = summarize(rows)
        mc = mcnemar([r["qa"] for r in rows], [r["qc"] for r in rows])
        s["mcnemar"] = mc
        per_seed.append(s)
        print(f"  seed {seed}: |idx|={len(index):,} |eval|={len(evals):,}")
        print(f"    검색 top-1 같은 문단 {s['retrieval_same_passage']:.3f} / "
              f"top-{CONTEXT_K} {s['retrieval_same_passage_k']:.3f} / "
              f"top-{TOP_K} {s['retrieval_topk_same']:.3f}")
        print(f"    Q-A 정확도 {s['qa_acc']:.3f} (top-{CONTEXT_K} 중 하나라도: "
              f"{s['qa_acc_k']:.3f})")
        print(f"    Q-C 정확도 {s['qc_acc']:.3f} "
              f"(문맥 1개일 때 {s['qc_acc_top1']:.3f}, "
              f"문맥 {CONTEXT_K}개로 +{s['qc_gain_from_k']:.3f})")
        print(f"    검색 성공 조건부  Q-A {s['qa_given_retrieval']:.3f}  "
              f"Q-C {s['qc_given_retrieval']:.3f}")
        print(f"    McNemar n01={mc['n01']} n10={mc['n10']} p={mc['p']:.2e}")
        print(f"    [진단] 같은 문단인데 답이 달라 Q-A 실패 "
              f"{s['qa_diff_answer_rate']:.3f}")
        print(f"    [진단] 다른 문단인데 답 문자열 우연 일치(Q-C 과대) "
              f"{s['qc_spurious_rate']:.3f}")
    if per_seed:
        for k in ("qa_acc", "qa_acc_k", "qc_acc", "qc_acc_top1", "qc_gain_from_k",
                  "qc_over_qa", "retrieval_same_passage",
                  "retrieval_same_passage_k",
                  "qa_given_retrieval", "qc_given_retrieval",
                  "qa_diff_answer_rate", "qc_spurious_rate"):
            m, lo, hi = bootstrap_ci([s[k] for s in per_seed])
            print(f"  {k:<24} {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        results["m2_loo"] = per_seed

    # ── M4
    print("\n[M4] 커버리지 절단 — 문단을 비율별로 제거")
    print("  ※ 절대 낙폭은 시작값이 높은 쪽이 커지므로 비교 지표가 못 된다.")
    print("    비율(Q-C/Q-A)과 상대 보존율로 판단할 것.")
    print(f"    {'lv':>6}{'Q-A':>9}{'Q-C':>9}{'Q-C/Q-A':>10}{'검색성공':>10}"
          f"{'Q-A보존':>10}{'Q-C보존':>10}")
    m4 = []
    for lv in COVERAGE_LEVELS:
        accs = []
        for seed in SEEDS[:1]:          # 절단은 시드 1개로 (시간 절약)
            index, evals = build_split(tri, groups, seed, keep_ratio=lv)
            rows = run_loo(index, evals, seed, f"cov{lv:.2f}")
            if rows:
                accs.append(summarize(rows))
        if not accs:
            continue
        a = {k: float(np.mean([x[k] for x in accs]))
             for k in ("qa_acc", "qc_acc", "retrieval_same_passage")}
        a["level"] = lv
        a["qc_over_qa"] = a["qc_acc"] / a["qa_acc"] if a["qa_acc"] else float("nan")
        base = m4[0] if m4 else a
        a["qa_retention"] = a["qa_acc"] / base["qa_acc"] if base["qa_acc"] else float("nan")
        a["qc_retention"] = a["qc_acc"] / base["qc_acc"] if base["qc_acc"] else float("nan")
        m4.append(a)
        print(f"    {lv:>6.2f}{a['qa_acc']:>9.3f}{a['qc_acc']:>9.3f}"
              f"{a['qc_over_qa']:>10.2f}{a['retrieval_same_passage']:>10.3f}"
              f"{a['qa_retention']:>10.3f}{a['qc_retention']:>10.3f}")
    results["m4_coverage"] = m4
    if len(m4) >= 2:
        r0, r1 = m4[0]["qc_over_qa"], m4[-1]["qc_over_qa"]
        print(f"\n  Q-C/Q-A 비: {r0:.2f} → {r1:.2f}  ({r1-r0:+.2f})")
        if r1 > r0:
            print("  → 커버리지가 줄수록 Q-C 의 상대 우위가 커진다.")
            print("     항목당 커버리지 논거가 예측하는 방향이며,")
            print("     R4-T1('커버리지 의존은 QI-RAG 도 마찬가지')에 대한 답이 된다.")
        else:
            print("  → 상대 우위가 커지지 않았다. 항목당 커버리지 논거를 재검토할 것.")
        print(f"  상대 보존율  Q-A {m4[-1]['qa_retention']:.3f}  "
              f"Q-C {m4[-1]['qc_retention']:.3f}")

    out = os.path.join(OUT_DIR, f"q13_coverage_hotpotqa__{ANALYZER}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {out}")

    print("\n원본 구현과의 대조:")
    print(f"  원본 23-QI-RAG-h.ipynb 는 retrieve_qi_rag(query, top_k=2) 로")
    print(f"  문서 2개를 '\\n\\n'.join 으로 이어 붙여 LLM 에 넘긴다.")
    print(f"  이 실험은 CONTEXT_K={CONTEXT_K} 로 그 구성을 따랐다.")
    print("  이전 측정(CONTEXT_K=1)과의 차이는 qc_gain_from_k 로 보고한다.")

    print("\n해석 시 주의:")
    print("  Q-C 의 정확도는 '문맥에 답 문자열이 있는가' 로 판정한 값이며,")
    print("  생성기가 그 문맥에서 답을 정확히 뽑아낸다는 가정이 들어간다.")
    print("  따라서 생성기 성능의 상한이다. 논문에는 상한임을 명시하고,")
    print("  실제 생성기를 붙인 값을 함께 보고할 것.")
    print("  Q-A 는 저장된 답을 그대로 반환하므로 상한이 아니라 실제값이다.")
    print("\n  HotpotQA 특유의 주의")
    print("  - 답이 yes/no 인 항목은 기본적으로 제외했다"
          f" (EXCLUDE_YESNO={EXCLUDE_YESNO}).")
    print("    Q-C 판정이 '답 문자열이 문맥에 있는가' 인데 yes/no 는")
    print("    문맥에 그 문자열로 등장하지 않아 구조적으로 0 이 되기 때문이다.")
    print("    제외 사실과 그 이유를 논문에 명시할 것.")
    print("  - 한 질문이 gold 문서 2개에 속하므로 여러 그룹에 나타난다.")
    print("    인덱스 중복 삽입은 차단했다.")
    print("\n  추가로 명시할 것:")
    print("  - M1 의 '그룹 100% 커버' 는 PAQ 생성 방식(답이 문단 내 span)의 귀결이다.")
    print("  - 항목당 커버리지 평균은 MIN_Q_PER_PASSAGE 필터에 의존한다.")
    print("    PAQ 전체 기준 추정치(약 3.1개)를 함께 보고할 것.")
    print("  - Q-C 정확도에는 다른 문단의 우연한 문자열 일치가 소량 섞여 있다.")
    print("    qc_spurious_rate 로 그 크기를 보고할 것.")


if __name__ == "__main__":
    main()
