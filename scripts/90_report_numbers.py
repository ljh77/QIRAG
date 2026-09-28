#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
90 — 논문용 수치 추출 및 대조
==============================
화면 출력을 눈으로 읽어 옮기면 줄을 잘못 짚는다(실제로 Q13 의 Q-C/Q-A 를
2.96 으로 잘못 읽은 적이 있다. 저장값은 1.63 이었다).
이 스크립트는 저장된 JSON 에서 직접 꺼내 한 곳에 모은다.

읽는 파일 (없으면 건너뛴다)
  _split/split_meta.json                  분할 조건
  _result/q13_coverage_hotpotqa__st.json  Q13 항목당 커버리지
  _result/compare_repaq_qirag.json        RePAQ / vanilla 비교
  _result/qirag_summary*.json             QI-RAG 실행 설정
  _result/vanilla_summary_*.json          vanilla 실행 설정
  _out/q8_judge_validation__QA.json       judge 검증 (RAGTruth)

실행:  python 90_report_numbers.py
"""

import json
import os
import sys

import numpy as np

# ══════════════════════════════════════════════════════════════════════
DATA_DIR = None
DATA_DIR_FALLBACK = ["/home/jun/RAG_Dataset", r"Z:\RAG_Dataset"]
PRIMARY = "contains"             # 83 의 PRIMARY 와 맞출 것
# ══════════════════════════════════════════════════════════════════════


def _auto():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.getcwd(), here] + DATA_DIR_FALLBACK:
        if c and os.path.isdir(os.path.join(c, "_split")):
            return c
    return os.getcwd()


def load(*parts):
    p = os.path.join(DATA_DIR, *parts)
    if not os.path.exists(p):
        return None, p
    try:
        return json.load(open(p, encoding="utf-8")), p
    except Exception as e:
        print(f"  [읽기 실패] {p}: {e}")
        return None, p


def sect(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def fmt(v, n=3):
    if v is None:
        return "-"
    if isinstance(v, float) and v != v:
        return "-"
    return f"{v:.{n}f}" if isinstance(v, (int, float)) else str(v)


def main():
    global DATA_DIR
    if DATA_DIR is None:
        DATA_DIR = _auto()
    print("=" * 76)
    print("90 — 논문용 수치 추출")
    print("=" * 76)
    print(f"  DATA_DIR : {DATA_DIR}")
    print(f"  주 지표  : {PRIMARY}")

    out = {}

    # ── 분할 조건 ─────────────────────────────────────────────────────
    sect("1. 실험 설정")
    m, p = load("_split", "split_meta.json")
    if m:
        c, s = m.get("config", {}), m.get("stats", {})
        print(f"  분할 모드        {c.get('split_mode')}")
        print(f"  인덱스           {s.get('index', 0):,}"
              f"  (n_index={c.get('n_index')})")
        print(f"  평가 질의        {s.get('eval', 0):,}")
        print(f"  문맥 구성        {c.get('context_mode')}")
        print(f"  누출             {s.get('leak_overlap')}건")
        print(f"  gold 커버리지    {fmt(s.get('gold_in_index_rate'))}"
              "   ← 검색 성능의 상한")
        print(f"  문단 그룹        {s.get('groups', 0):,}")
        print(f"  변형             {c.get('variants')}")
        print(f"  LLM 재작성 실패  {s.get('llm_paraphrase_fallback')}건")
        print(f"  평가 type        {s.get('eval_type')}")
        print(f"  평가 level       {s.get('eval_level')}")
        out["split"] = {"config": c, "stats": s}
    else:
        print(f"  없음: {p}")

    for tag in ("", "_perm"):
        q, _ = load("_result", f"qirag_summary{tag}.json")
        if q:
            c = q.get("config", {})
            print(f"\n  QI-RAG{tag or ' (strict)'}  인코더 {c.get('encoder')}"
                  f"  TOP_K={c.get('top_k')} CONTEXT_K={c.get('context_k')}"
                  f"  프롬프트={c.get('prompt_mode')}")
    for tag in ("gold", "dist"):
        v, _ = load("_result", f"vanilla_summary_{tag}.json")
        if v:
            c = v.get("config", {})
            print(f"  vanilla({tag})  문서 {c.get('n_docs', 0):,}"
                  f"  TOP_K={c.get('top_k')}"
                  f"  코퍼스={c.get('corpus_source')}")

    # ── Q13 ───────────────────────────────────────────────────────────
    sect("2. Q13 항목당 커버리지  (R4-T1 / R4-T2)")
    d, p = load("_result", "q13_coverage_hotpotqa__st.json")
    if d:
        m1 = d.get("m1_per_item_coverage", {})
        print(f"  그룹 {m1.get('n_groups', 0):,}  "
              f"평균 크기 {fmt(m1.get('mean_group_size'), 2)}")
        print(f"  항목당 커버리지   Q-A {fmt(m1.get('qa_mean'), 2)}개"
              f"  vs  Q-C {fmt(m1.get('qc_mean'), 2)}개"
              f"   (그룹의 {100*(m1.get('qc_ratio') or 0):.1f}%)")
        ratio = (m1.get('qc_mean') or 0) / max(m1.get('qa_mean') or 1, 1e-9)
        print(f"  커버리지 비       {fmt(ratio, 2)}배")

        loo = d.get("m2_loo", [])
        if loo:
            print(f"\n  leave-one-out (시드 {len(loo)}회)")
            for k, lab in (("qa_acc", "Q-A 정확도"), ("qc_acc", "Q-C 정확도"),
                           ("qc_over_qa", "Q-C/Q-A"),
                           ("retrieval_same_passage_k", "검색적중"),
                           ("qa_given_retrieval", "Q-A 조건부"),
                           ("qc_given_retrieval", "Q-C 조건부"),
                           ("qa_diff_answer_rate", "Q-A 실패(답 다름)"),
                           ("qc_spurious_rate", "Q-C 우연일치")):
                vals = [x.get(k) for x in loo if isinstance(x.get(k), (int, float))]
                if not vals:
                    continue
                print(f"    {lab:<20} {np.mean(vals):.3f}"
                      f"   범위 {min(vals):.3f}~{max(vals):.3f}")
            mc = [x.get("mcnemar", {}) for x in loo]
            n01 = [x.get("n01") for x in mc if x.get("n01") is not None]
            n10 = [x.get("n10") for x in mc if x.get("n10") is not None]
            ps = [x.get("p") for x in mc if x.get("p") is not None]
            if n01:
                print(f"    McNemar  n01 {int(np.mean(n01))} / "
                      f"n10 {int(np.mean(n10))}  p={max(ps):.2e}")
        m4 = d.get("m4_coverage", [])
        if m4:
            print(f"\n  커버리지 절단")
            print(f"    {'lv':>6}{'Q-A':>9}{'Q-C':>9}{'Q-C/Q-A':>10}{'검색':>9}")
            for r in m4:
                print(f"    {r['level']:>6.2f}{r['qa_acc']:>9.3f}"
                      f"{r['qc_acc']:>9.3f}{r.get('qc_over_qa', 0):>10.2f}"
                      f"{r.get('retrieval_same_passage', 0):>9.3f}")
        out["q13"] = {"m1": m1, "m2": loo, "m4": m4,
                      "config": d.get("config", {})}
    else:
        print(f"  없음: {p}")

    # ── RePAQ / vanilla 비교 ──────────────────────────────────────────
    sect("3. RePAQ / vanilla 비교  (R2-2, 검색 축)")
    d, p = load("_result", "compare_repaq_qirag.json")
    if d:
        for vname, v in d.get("variants", {}).items():
            print(f"\n  [{vname}]")
            rows = [("A) RePAQ", "A_repaq"), ("B) E5+a'", "B_e5_answer"),
                    ("C) QI-RAG", "C_qirag"), ("C') QI perm", "C_qirag_perm"),
                    ("D) vanilla", "D_vanilla"), ("D') van+dist",
                                                  "D_vanilla_dist")]
            print(f"    {'조건':<14}{'응답률':>9}{'EM':>9}{PRIMARY:>11}"
                  f"{'선택'+PRIMARY[:4]:>11}")
            for lab, key in rows:
                s = v.get(key) or {}
                if not s:
                    continue
                print(f"    {lab:<14}{fmt(s.get('answer_rate')):>9}"
                      f"{fmt(s.get('EM')):>9}{fmt(s.get(PRIMARY)):>11}"
                      f"{fmt(s.get('sel_' + PRIMARY)):>11}")
            dec = v.get("decomposition", {})
            if dec:
                print(f"\n    요인 분해 ({PRIMARY})")
                for key, lab in (("total", "RePAQ -> QI-RAG (전체)"),
                                 ("retriever", "  검색기"),
                                 ("payload_from_repaq", "  페이로드"),
                                 ("retrieval_space_gold", "vanilla(gold) 대비"),
                                 ("retrieval_space_dist", "vanilla(+dist) 대비")):
                    t = dec.get(key)
                    if not t:
                        continue
                    sig = "유의" if t.get("significant") else "불확실"
                    print(f"      {lab:<24} {fmt(t.get('a'))} -> {fmt(t.get('b'))}"
                          f"  차이 {t.get('diff', 0):+.4f}"
                          f"  CI [{t.get('ci_lo', 0):+.4f}, {t.get('ci_hi', 0):+.4f}]"
                          f"  {sig}"
                          f"  p={t.get('mcnemar', {}).get('p', float('nan')):.2e}")
            rel = v.get("reliability", {})
            if rel:
                print(f"\n    신뢰성")
                print(f"      {'조건':<14}{'유보율':>9}{'정답률':>9}"
                      f"{'오답률':>9}{'위험비':>9}")
                for lab, r in rel.items():
                    if lab.startswith("_") or not isinstance(r, dict):
                        continue
                    rr = r.get("risk_ratio")
                    rr = -1 if (rr is None or rr == float("inf")) else rr
                    print(f"      {lab:<14}{fmt(r.get('abstain_rate')):>9}"
                          f"{fmt(r.get('correct_rate')):>9}"
                          f"{fmt(r.get('wrong_rate')):>9}{rr:>9.2f}")
                miss = rel.get("_miss", {})
                if miss:
                    print(f"\n    검색 실패 구간")
                    print(f"      {'조건':<14}{'유보':>8}{'정답':>8}{'오답':>8}"
                          f"{'오답률':>9}")
                    for lab, r in miss.items():
                        print(f"      {lab:<14}{r.get('n_abstain', 0):>8}"
                              f"{r.get('n_correct', 0):>8}{r.get('n_wrong', 0):>8}"
                              f"{fmt(r.get('wrong_rate')):>9}")
        out["compare"] = d.get("variants", {})
    else:
        print(f"  없음: {p}")

    # ── judge 검증 ────────────────────────────────────────────────────
    sect("4. judge 검증  (R1-W2 / R2-5)")
    d, p = load("_out", "q8_judge_validation__QA.json")
    if d:
        a = d.get("a_label_structure", {})
        if a:
            print("  사람 span 라벨의 '문맥 밖 토큰' 중앙값")
            for k in ("baseless", "conflict", "clean"):
                if a.get(k):
                    print(f"    {k:<10} {fmt(a[k].get('median'))}"
                          f"   (n={a[k].get('n')})")
        b = d.get("b_lexical_judge", {})
        if b:
            o = b.get("overall", {})
            t = b.get("trivial", {})
            print(f"\n  어휘 판정기   F1 {fmt(o.get('f1'))}"
                  f"  kappa {fmt(o.get('kappa'))}")
            print(f"  자명 기준선   F1 {fmt(t.get('f1'))}"
                  f"  kappa {fmt(t.get('kappa'))}")
        dd = d.get("d_llm_judge", {})
        if dd and dd.get("prompts"):
            print(f"\n  LLM judge ({dd.get('model', '').split('/')[-1]}, "
                  f"n={dd.get('n')})")
            for k, r in dd["prompts"].items():
                print(f"    {k:<12} P {fmt(r.get('precision'))}"
                      f"  R {fmt(r.get('recall'))}  F1 {fmt(r.get('f1'))}"
                      f"  kappa {fmt(r.get('kappa'))}")
        out["judge"] = d
    else:
        print(f"  없음: {p}")

    # ── 저장 ──────────────────────────────────────────────────────────
    od = os.path.join(DATA_DIR, "_result")
    os.makedirs(od, exist_ok=True)
    pth = os.path.join(od, "paper_numbers.json")
    with open(pth, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {pth}")
    print("\n  이 출력의 값만 논문에 쓸 것. 화면 로그를 눈으로 옮기지 말 것.")


if __name__ == "__main__":
    main()
