#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
85 — vanilla RAG 파이프라인 검증
=================================
84 가 '올바르게 구축된 vanilla RAG' 인지 확인한다.
돌아가는 것과 올바른 것은 다르다. 45 에서 QI-RAG 조건을 검증했던 것과 같은 취지.

검증 항목
  V1 접두어     E5 는 문서에 "passage:", 질의에 "query:" 를 쓴다.
                잘못 쓰면 성능이 떨어지는데 에러는 나지 않는다.
  V2 문서 집합  코퍼스가 QI-RAG 페이로드와 같은 문서 집합인가.
                다르면 '검색 축' 비교가 아니라 '데이터' 비교가 된다.
  V3 문단 분리  "[제목] 본문" 정규식이 실제로 쪼갰는가.
                실패하면 문맥 통째가 한 문서가 된다(길이 분포로 확인).
  V4 자기검색   코퍼스 문서를 그대로 질의로 넣으면 자기를 찾는가 (1.000 이어야).
  V5 제목 일치  검색적중 판정에 쓰는 title 문자열이 gold_titles 와 맞는가.
                공백/대소문자가 틀어지면 적중이 0 에 가깝게 나온다.
  V6 프롬프트   81 과 동일한가. 다르면 검색 축 효과와 프롬프트 효과가 섞인다.

실행:  python 85_validate_vanilla.py
"""

import json
import os
import re
import sys
from collections import Counter

import numpy as np

# ══════════════════════════════════════════════════════════════════════
DATA_DIR = None
DATA_DIR_FALLBACK = ["/home/jun/RAG_Dataset", r"Z:\RAG_Dataset"]
SPLIT_DIR = None
RESULT_DIR = None

ENCODER_MODEL = "intfloat/e5-base-v2"
ENCODER_DEVICE = "cuda"
N_PROBE = 300                    # V1/V4 표본 수
TOP_K = 2
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


def load_jsonl(p, limit=None):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if limit and len(out) >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class Check:
    def __init__(self):
        self.rows = []

    def add(self, code, ok, msg, note=""):
        self.rows.append({"code": code, "ok": bool(ok), "msg": msg})
        print(f"  {'[통과]' if ok else '[실패]'} {code}  {msg}")
        if note:
            print(f"          {note}")
        return ok

    @property
    def n_fail(self):
        return sum(1 for r in self.rows if not r["ok"])


# ══════════════════════════════════════════════════════════════════════

def main():
    global DATA_DIR, SPLIT_DIR, RESULT_DIR
    if DATA_DIR is None:
        DATA_DIR = _auto()
    if SPLIT_DIR is None:
        SPLIT_DIR = os.path.join(DATA_DIR, "_split")
    if RESULT_DIR is None:
        RESULT_DIR = os.path.join(DATA_DIR, "_result")

    print("=" * 76)
    print("85 — vanilla RAG 파이프라인 검증")
    print("=" * 76)
    print(f"  SPLIT_DIR  : {SPLIT_DIR}")
    print(f"  RESULT_DIR : {RESULT_DIR}")

    cp = os.path.join(RESULT_DIR, "doc_corpus.jsonl")
    if not os.path.exists(cp):
        print("\n  doc_corpus.jsonl 이 없습니다. 84_run_vanilla.py 를 먼저 실행하세요.")
        return
    docs = load_jsonl(cp)
    items = load_jsonl(os.path.join(SPLIT_DIR, "index_items.jsonl"))
    evals = load_jsonl(os.path.join(SPLIT_DIR, "eval_queries.jsonl"))
    log(f"  코퍼스 {len(docs):,} / 인덱스 항목 {len(items):,} / 평가 {len(evals):,}")

    c = Check()

    # ── V3 문단 분리 ──────────────────────────────────────────────────
    print("\n[V3] 문단 분리")
    lens = [len(d["text"]) for d in docs]
    n_title = sum(1 for d in docs if d["title"])
    log(f"  제목이 있는 문서 {n_title:,}/{len(docs):,} "
        f"({100*n_title/max(len(docs),1):.1f}%)")
    log(f"  문서 길이  중앙값 {int(np.median(lens))}자  "
        f"평균 {np.mean(lens):.0f}  범위 {min(lens)}~{max(lens)}")
    c.add("V3-1", n_title / max(len(docs), 1) > 0.95,
          f"제목 파싱 성공률 {100*n_title/max(len(docs),1):.1f}%",
          "낮으면 DOC_SPLIT_PATTERN 이 안 맞는 것. 84 의 정규식 확인.")
    # 문단 분리 여부는 '문서 수' 가 아니라 '길이' 로 판정한다.
    #   문서 수는 제목 공유율에 좌우되므로 기준이 될 수 없다.
    #   분리에 실패하면 문서 길이가 원본 문맥 길이와 같아진다.
    ctx_lens = [len(r.get("context", "")) for r in items[:3000]]
    med_ctx = float(np.median(ctx_lens)) if ctx_lens else 0.0
    med_doc = float(np.median(lens))
    log(f"  원본 문맥 길이 중앙값 {int(med_ctx)}자 / 문서 {int(med_doc)}자  "
        f"(비율 {med_doc/max(med_ctx,1):.2f})")
    c.add("V3-2", med_doc < med_ctx * 0.9,
          f"문서가 문맥보다 짧다 ({int(med_doc)} < {int(med_ctx)}자)",
          "비율이 1.0 에 가까우면 문단이 안 쪼개진 것. 정규식을 확인.")

    # 항목당 평균 문서 수 (gold 모드면 supporting_facts 개수와 비슷해야 함)
    pat = re.compile(r"\[(.*?)\]\s*(.*?)(?=\s*\[[^\]]+\]\s|$)", re.S)
    per_item = [len(pat.findall(r.get("context", ""))) for r in items[:3000]]
    if per_item:
        log(f"  항목당 문단 수 중앙값 {int(np.median(per_item))}  "
            f"(gold 모드면 supporting_facts 개수와 같아야 정상)")
        c.add("V3-3", np.median(per_item) >= 1,
              f"항목당 문단이 1개 이상 분리됨 "
              f"(중앙값 {int(np.median(per_item))})")
    # 고유 제목 대비 코퍼스 (공유율 참고용, 판정 아님)
    log(f"  코퍼스 {len(docs):,} / 인덱스 항목 {len(items):,}  "
        f"(제목 공유가 많으면 코퍼스가 더 작다 — 정상)")

    # ── V2 문서 집합 일치 ─────────────────────────────────────────────
    print("\n[V2] 문서 집합")
    titles = {d["title"] for d in docs if d["title"]}
    cover = sum(1 for e in evals
                if any(t in titles for t in e.get("gold_titles", [])))
    rate = cover / max(len(evals), 1)
    meta_p = os.path.join(SPLIT_DIR, "split_meta.json")
    ref = None
    if os.path.exists(meta_p):
        ref = json.load(open(meta_p, encoding="utf-8"))\
            .get("stats", {}).get("gold_in_index_rate")
    log(f"  평가 질의의 gold 문서가 코퍼스에 존재 {cover}/{len(evals)} "
        f"({100*rate:.1f}%)")
    if ref is not None:
        log(f"  80 의 분할 메타 기준값 {100*ref:.1f}%")
        c.add("V2-1", abs(rate - ref) < 0.02,
              f"코퍼스 커버리지 {100*rate:.1f}% vs 분할 메타 {100*ref:.1f}%",
              "다르면 문서가 누락됐거나 제목이 달라진 것. "
              "검색 축 비교가 성립하지 않는다.")
    else:
        c.add("V2-1", rate > 0.5, f"코퍼스 커버리지 {100*rate:.1f}%")

    # 인덱스 항목의 gold_titles 가 코퍼스에 다 있는가
    miss = set()
    for r in items[:5000]:
        for t in r.get("gold_titles", []):
            if t not in titles:
                miss.add(t)
    c.add("V2-2", len(miss) == 0,
          f"인덱스 항목의 gold 제목 중 코퍼스에 없는 것 {len(miss)}개",
          (f"예: {list(miss)[:3]}" if miss else ""))

    # ── V5 제목 일치 ──────────────────────────────────────────────────
    print("\n[V5] 제목 문자열 일치")
    gold_all = Counter()
    for e in evals:
        for t in e.get("gold_titles", []):
            gold_all[t] += 1
    exact = sum(v for t, v in gold_all.items() if t in titles)
    lowered = {t.lower().strip() for t in titles}
    loose = sum(v for t, v in gold_all.items()
                if t.lower().strip() in lowered and t not in titles)
    log(f"  gold 제목 {sum(gold_all.values()):,}회 중 정확 일치 {exact:,}회")
    c.add("V5-1", loose == 0,
          f"대소문자/공백만 다른 불일치 {loose}회",
          "0 이 아니면 정규화가 필요하다. 검색적중이 과소평가된다.")

    # ── V6 프롬프트 ───────────────────────────────────────────────────
    print("\n[V6] 프롬프트 동일성")
    ps, pv = None, None
    for f, key in ((os.path.join(RESULT_DIR, "qirag_summary.json"), "ps"),
                   (os.path.join(RESULT_DIR, "vanilla_summary.json"), "pv")):
        if os.path.exists(f):
            v = json.load(open(f, encoding="utf-8"))\
                .get("config", {}).get("prompt")
            if key == "ps":
                ps = v
            else:
                pv = v
    if ps and pv:
        c.add("V6-1", ps == pv, "81 과 84 의 QA 프롬프트가 동일",
              "다르면 검색 축 효과와 프롬프트 효과가 섞인다.")
    else:
        log("  요약 파일이 없어 건너뜀 (81/84 를 먼저 실행)")

    # top_k 일치
    tk_q = tk_v = None
    f = os.path.join(RESULT_DIR, "qirag_summary.json")
    if os.path.exists(f):
        tk_q = json.load(open(f, encoding="utf-8"))\
            .get("config", {}).get("context_k")
    f = os.path.join(RESULT_DIR, "vanilla_summary.json")
    if os.path.exists(f):
        tk_v = json.load(open(f, encoding="utf-8"))\
            .get("config", {}).get("top_k")
    if tk_q and tk_v:
        c.add("V6-2", tk_q == tk_v,
              f"문맥 수 일치  QI-RAG CONTEXT_K={tk_q} / vanilla TOP_K={tk_v}",
              "원본은 QI-RAG 2 / vanilla 3 으로 달랐다. 맞추지 않으면 교락.")

    # ── V1/V4 임베딩 검사 ─────────────────────────────────────────────
    print("\n[V1/V4] 임베딩과 검색")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers 가 없어 건너뜁니다 (vllm_env 에서 실행하세요).")
        SentenceTransformer = None

    if SentenceTransformer is not None:
        m = SentenceTransformer(ENCODER_MODEL, device=ENCODER_DEVICE)
        sub = docs[:N_PROBE]
        texts = [(f"{d['title']} {d['text']}" if d["title"] else d["text"])
                 for d in sub]

        def enc(ts, pre):
            return m.encode([pre + t for t in ts], batch_size=128,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False).astype("float32")

        # V4 자기검색: 문서를 질의로 넣으면 자기를 찾아야 한다
        Ed = enc(texts, "passage: ")
        Eq = enc(texts, "query: ")
        S = Eq @ Ed.T
        self_acc = float(np.mean(np.argmax(S, axis=1) == np.arange(len(sub))))
        c.add("V4-1", self_acc > 0.95,
              f"자기검색 정확도 {self_acc:.3f} (1.000 에 가까워야 정상)",
              "낮으면 임베딩이나 인덱스 정렬에 문제가 있다.")

        # V1 접두어: 올바른 조합이 잘못된 조합보다 나아야 한다
        ev = evals[:N_PROBE]
        gold_t = [set(e.get("gold_titles", [])) for e in ev]
        Q_right = enc([e["original"] for e in ev], "query: ")
        Q_wrong = enc([e["original"] for e in ev], "passage: ")
        D_wrong = enc(texts, "query: ")

        def hit_rate(Q, D):
            S = Q @ D.T
            k = min(TOP_K, S.shape[1])
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            out = []
            for r, ids in enumerate(idx):
                ts = {sub[int(i)]["title"] for i in ids}
                out.append(bool(ts & gold_t[r]))
            return float(np.mean(out))

        h_ok = hit_rate(Q_right, Ed)          # query: / passage:  ← 올바름
        h_qq = hit_rate(Q_right, D_wrong)     # query: / query:
        h_sw = hit_rate(Q_wrong, Ed)          # passage: / passage:
        log(f"  접두어 조합별 검색적중 (표본 {len(ev)}, 코퍼스 {len(sub)})")
        log(f"    query:/passage:  {h_ok:.3f}   ← 84 가 쓰는 조합")
        log(f"    query:/query:    {h_qq:.3f}")
        log(f"    passage:/passage:{h_sw:.3f}")
        c.add("V1-1", h_ok >= max(h_qq, h_sw) - 0.01,
              "올바른 접두어 조합이 가장 높다",
              "아니면 E5 접두어 사용을 재검토할 것.")

    # ── 결과 파일 점검 ────────────────────────────────────────────────
    print("\n[결과 파일]")
    for v in ("original", "paraphrase", "noisy"):
        p = os.path.join(RESULT_DIR, f"result_vanilla_{v}.jsonl")
        if not os.path.exists(p):
            log(f"  {v:<12} 없음")
            continue
        rows = load_jsonl(p)
        hit = float(np.mean([r["retrieval_hit_topk"] for r in rows]))
        empty = sum(1 for r in rows if not r.get("pred", "").strip())
        log(f"  {v:<12} n={len(rows):,}  검색적중 {hit:.3f}  "
            f"빈 응답 {empty}")
        if empty > len(rows) * 0.1:
            log("             ! 빈 응답이 많습니다. 생성 서버를 확인하세요.")

    # ── 종합 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("종합")
    print("=" * 76)
    print(f"  {len(c.rows)}항목 중 실패 {c.n_fail}")
    if c.n_fail:
        print("\n  실패 항목을 고치기 전에는 vanilla 결과를 비교에 쓰지 말 것.")
        for r in c.rows:
            if not r["ok"]:
                print(f"    [{r['code']}] {r['msg']}")
    else:
        print("  통과. 84 결과를 83_compare.py 의 D 조건으로 쓸 수 있다.")

    p = os.path.join(RESULT_DIR, "validate_vanilla.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"checks": c.rows, "n_docs": len(docs),
                   "corpus_coverage": rate}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
