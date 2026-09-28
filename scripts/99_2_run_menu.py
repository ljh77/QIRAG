#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
99 — 실행 메뉴
===============
옵션을 바꿔가며 여러 번 돌려야 하는 스크립트들을 번호로 고른다.
파일을 열어 상수를 고치고 다시 저장하는 과정에서 값을 안 바꾸고 돌리는
사고가 반복됐다(PROMPT_MODE 를 안 바꿔 11분 헛돌린 적, N_INDEX 가
20,000 으로 되돌아가 모든 결과가 무효가 된 적).
이 스크립트는 대상 모듈의 상수를 실행 직전에 주입하므로 파일을 고치지 않는다.

실행:  python 99_run_menu.py
       python 99_run_menu.py 3        번호를 인자로 줘도 된다
       python 99_run_menu.py 3 4 5    여러 개를 순서대로
       python 99_run_menu.py all      전부
"""

import importlib.util
import os
import sys
import time
import traceback

# ══════════════════════════════════════════════════════════════════════
#  ██ 메뉴 정의
#     각 항목: (번호, 설명, 파일, 주입할 설정, 예상시간)
#     설정의 키는 대상 모듈의 전역 변수명이다.
# ══════════════════════════════════════════════════════════════════════
VARIANTS_ALL = ["original", "keyword", "noisy", "paraphrase_llm"]

MENU = [
    ("1", "분할 고정  (train/dev 공식 분할, LLM 재작성 포함)",
     "80_make_split.py",
     {"SPLIT_MODE": "split", "N_INDEX": None, "N_EVAL": 2000,
      "CONTEXT_MODE": "gold", "MAKE_VARIANTS": VARIANTS_ALL},
     "5분"),

    ("1b", "분할 고정  (원본 재현 — 인덱스 = 평가, 자기검색 발생)",
     "80_make_split.py",
     {"SPLIT_MODE": "leak", "N_INDEX": None, "N_EVAL": 2000,
      "CONTEXT_MODE": "gold", "MAKE_VARIANTS": VARIANTS_ALL},
     "5분"),

    ("1c", "분할 고정  (1:1 매핑 절제 — gold 문단 1개만)",
     "80_make_split.py",
     {"SPLIT_MODE": "split", "N_INDEX": None, "N_EVAL": 2000,
      "CONTEXT_MODE": "gold_first", "MAKE_VARIANTS": VARIANTS_ALL},
     "5분"),

    ("2", "QI-RAG  strict      (1:N 매핑, 원본 프롬프트)",
     "81_run_qirag.py",
     {"PROMPT_MODE": "strict", "MAP_TAG": None, "VARIANTS": VARIANTS_ALL},
     "18분"),

    ("2b", "QI-RAG  1:1 매핑 절제  ★ Layer 1 의 기여 측정",
     "81_run_qirag.py",
     {"PROMPT_MODE": "strict", "MAP_TAG": "1to1", "VARIANTS": VARIANTS_ALL},
     "18분"),

    ("3", "QI-RAG  permissive  (유보 완화 — 같은 유보율 비교용)",
     "81_run_qirag.py",
     {"PROMPT_MODE": "permissive", "MAP_TAG": None, "VARIANTS": VARIANTS_ALL},
     "14분"),

    ("4", "vanilla  gold        (문서 집합 통제)",
     "84_run_vanilla.py",
     {"CORPUS_SOURCE": "index", "VARIANTS": VARIANTS_ALL},
     "12분"),

    ("5", "vanilla  gold+dist   (현실 조건)",
     "84_run_vanilla.py",
     {"CORPUS_SOURCE": "hotpot", "VARIANTS": VARIANTS_ALL},
     "25분"),

    ("6", "B'  a'+LLM  strict      (생성 효과 분리)",
     "86_run_answer_ctx.py",
     {"PROMPT_MODE": "strict", "SOURCE_TAG": "", "VARIANTS": VARIANTS_ALL},
     "12분"),

    ("7", "B'  a'+LLM  permissive",
     "86_run_answer_ctx.py",
     {"PROMPT_MODE": "permissive", "SOURCE_TAG": "", "VARIANTS": VARIANTS_ALL},
     "12분"),

    ("8", "Q13 항목당 커버리지  (HotpotQA)",
     "41_q13_coverage_hotpotqa.py",
     {"ANALYZER": "st", "SEEDS": [0, 1, 2]},
     "25분"),

    ("h2", "HypQI  (문단 20,000)",
     "89_hypqi.py",
     {"N_INDEX_ITEMS": 20_000, "K_QUESTIONS": 1, "VARIANTS": VARIANTS_ALL},
     "45분"),

    ("h5", "HypQI  (문단 50,000)  ★ 권장 시작점",
     "89_hypqi.py",
     {"N_INDEX_ITEMS": 50_000, "K_QUESTIONS": 1, "VARIANTS": VARIANTS_ALL},
     "2시간"),

    ("h9", "HypQI  (전체 90,447)",
     "89_hypqi.py",
     {"N_INDEX_ITEMS": None, "K_QUESTIONS": 1, "VARIANTS": VARIANTS_ALL},
     "3.5시간"),

    ("hc", "HypQI 비교  (89_2)",
     "89_2_compare_hypqi.py", {}, "즉시"),

    ("9", "비교 집계  (83)",
     "83_compare.py", {}, "즉시"),

    ("10", "논문용 수치 추출  (90)",
     "90_report_numbers.py", {}, "즉시"),

    ("v", "vanilla 파이프라인 검증  (85)",
     "85_validate_vanilla.py", {}, "2분"),
]

# 묶음 실행
GROUPS = {
    "all": ["1", "2", "3", "4", "5", "6", "7", "9", "10"],
    "map": ["1c", "2b", "9", "10"],              # 매핑 차수 절제
    "hyp": ["h5", "hc"],                         # HypQI 50k + 비교만
    "gen": ["2", "3", "4", "5", "6", "7"],       # 생성이 필요한 것 전부
    "agg": ["9", "10"],                          # 집계만
}

# RePAQ 는 별도 conda 환경이라 여기서 실행하지 않는다. 명령만 안내한다.
REPAQ_CMD = """\
  PY=~/miniconda3/envs/repaq/bin/python
  SPLIT=~/RAG_Dataset/_split
  OUT=~/RAG_Dataset/_result
  MODEL=./data/models/retrievers/retriever_multi_base_256
  cd ~/PAQ
  for V in q p n pl; do
    PYTHONPATH= $PY -m paq.retrievers.retrieve \\
      --model_name_or_path $MODEL \\
      --qas_to_answer $SPLIT/eval_qa_$V.jsonl \\
      --qas_to_retrieve_from $SPLIT/index_qa.jsonl \\
      --faiss_index_path $OUT/repaq_index.faiss \\
      --top_k 2 --output_file $OUT/result_repaq_$V.jsonl \\
      --memory_friendly_parsing --verbose
  done"""
# ══════════════════════════════════════════════════════════════════════


def find(fname):
    here = os.path.dirname(os.path.abspath(__file__))
    for d in dict.fromkeys((os.getcwd(), here)):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def run(path, overrides, label):
    """모듈을 불러와 전역 상수를 주입한 뒤 main() 을 부른다.
    파일을 수정하지 않으므로 다음 실행에 영향이 없다."""
    name = "m_" + os.path.basename(path).replace(".", "_") + "_" + str(time.time_ns())
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)

    applied, missing = [], []
    for k, v in overrides.items():
        if hasattr(m, k):
            setattr(m, k, v)
            applied.append(f"{k}={v!r}")
        else:
            missing.append(k)
    # 프롬프트 모드를 바꿨으면 프롬프트 본문도 따라가야 한다
    if "PROMPT_MODE" in overrides and hasattr(m, "QA_PROMPTS"):
        m.QA_PROMPT = m.QA_PROMPTS[overrides["PROMPT_MODE"]]

    print("\n" + "#" * 74)
    print(f"# {label}")
    print(f"# {os.path.basename(path)}")
    if applied:
        print("# 주입: " + ", ".join(applied))
    if missing:
        print("# ! 대상에 없는 설정: " + ", ".join(missing))
        print("#   파일 버전이 다를 수 있습니다. 확인하세요.")
    print("#" * 74)

    t0 = time.time()
    m.main()
    print(f"\n[{label}] 완료  {(time.time()-t0)/60:.1f}분")
    sys.modules.pop(name, None)


def show_menu():
    print("=" * 74)
    print("99 — 실험 실행 메뉴")
    print("=" * 74)
    for num, desc, fname, ov, est in MENU:
        mark = " " if find(fname) else "!"
        print(f" {mark}{num:>3}) {desc:<44} {est:>6}")
    print()
    print("  묶음:  " + "   ".join(f"{k} = {' '.join(v)}"
                                   for k, v in GROUPS.items()))
    print("   r )  RePAQ 검색 명령 보기 (별도 conda 환경)")
    print("   q )  종료")
    print()
    print("  ! 표시는 파일을 찾지 못한 항목입니다.")


def main():
    args = sys.argv[1:]
    if not args:
        show_menu()
        try:
            raw = input("번호 (공백으로 여러 개): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        args = raw.split()
    if not args or args[0] in ("q", "quit", "exit"):
        return

    # 묶음 전개
    seq = []
    for a in args:
        if a in GROUPS:
            seq += GROUPS[a]
        else:
            seq.append(a)

    if "r" in seq:
        print("\nRePAQ 검색 (repaq 환경에서 실행)")
        print(REPAQ_CMD)
        seq = [x for x in seq if x != "r"]
        if not seq:
            return

    table = {num: (desc, fname, ov, est) for num, desc, fname, ov, est in MENU}
    plan = []
    for a in seq:
        if a not in table:
            print(f"  [무시] 알 수 없는 번호: {a}")
            continue
        desc, fname, ov, est = table[a]
        path = find(fname)
        if not path:
            print(f"  [건너뜀] {a}) {fname} 를 찾지 못했습니다.")
            continue
        plan.append((a, desc, path, ov, est))

    if not plan:
        print("  실행할 항목이 없습니다.")
        return

    print("\n실행 계획")
    for a, desc, path, ov, est in plan:
        print(f"  {a:>3}) {desc}  ({est})")
    print()

    fails = []
    t_all = time.time()
    for a, desc, path, ov, est in plan:
        try:
            run(path, ov, f"{a}) {desc}")
        except KeyboardInterrupt:
            print("\n중단됨.")
            return
        except Exception:
            fails.append(a)
            print(f"\n[{a}] 실패")
            traceback.print_exc()

    print("\n" + "=" * 74)
    print(f"전체 {(time.time()-t_all)/60:.1f}분")
    if fails:
        print(f"실패: {', '.join(fails)}")
    else:
        print("모두 완료")


if __name__ == "__main__":
    main()
