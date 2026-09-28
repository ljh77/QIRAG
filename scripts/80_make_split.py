#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
80 — 분할 고정 (HotpotQA)
==========================
RePAQ 와 QI-RAG 가 같은 무대에서 비교되려면 입력이 동일해야 한다.
두 실험은 환경이 달라(transformers 4.1.0 vs sentence-transformers) 한 프로세스에서
못 돌린다. 그래서 분할을 먼저 파일로 고정하고, 양쪽이 그 파일만 읽게 한다.

산출물 (WORK_DIR)
  index_items.jsonl     인덱스 항목 (q', a', c', title)   ← 양쪽 공통
  index_qa.jsonl        RePAQ 용 {question, answer} 만    ← 공식 코드 입력 형식
  eval_queries.jsonl    평가 질의 + Q/P/N 변형 + 정답
  eval_qa_{q,p,n}.jsonl RePAQ 용 변형별 {question, answer}
  passage_groups.json   Q13 용 문단 그룹 (제목 -> qid 목록)
  split_meta.json       설정과 통계

왜 train/dev 를 쓰는가
  원본 QI-RAG 는 ragllm_outputs_572.csv 의 질문으로 인덱스를 만들고
  같은 질문의 변형으로 평가했다. Original 조건에서 질의가 인덱스에 문자
  그대로 존재해 자기검색이 된다 (리뷰어 2 W4).
  HotpotQA 의 공식 train/dev 분할을 쓰면 자기검색이 구조적으로 불가능하다.

실측 (2026-09-15)
  train 90,447건 / dev 7,405건
  gold 문서 105,570개 중 2건 이상 공유 36,126개 (34.2%)
  -> Q13 항목당 커버리지가 HotpotQA 로 가능하다
  type  bridge 72,991 / comparison 17,456
  level medium 56,814 / easy 17,972 / hard 15,661

실행:  python 80_make_split.py
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

# ══════════════════════════════════════════════════════════════════════
#  ██ 설정
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = None                      # None = 자동 (hotpotqa 폴더가 있는 곳)
DATA_DIR_FALLBACK = ["/home/jun/RAG_Dataset", r"Z:\RAG_Dataset"]
HOTPOT_DIR = "hotpotqa"
TRAIN_FILE = "hotpot_train.jsonl"
DEV_FILE = "hotpot_dev_distractor.jsonl"
#   None 이면 SPLIT_MODE 에 따라 _split / _split_leak 로 자동 분기
WORK_DIR = None

# ── 분할 모드 ────────────────────────────────────────────────────────
#   "split" : train 으로 인덱스, dev 로 평가. 자기검색이 구조적으로 불가능.
#             논문의 주 조건.
#   "leak"  : ★ 원본 재현. dev 에서 뽑은 같은 집합을 인덱스와 평가에 모두 쓴다.
#             원본 QI-RAG 는 ragllm_outputs_572.csv 의 질문으로 인덱스를 만들고
#             같은 질문의 변형으로 평가했다. Original 조건에서 질의가 인덱스에
#             문자 그대로 존재해 자기검색이 된다 (리뷰어 2 W4).
#             이 조건의 수치와 "split" 을 대조하면 누출이 성능을 얼마나
#             부풀렸는지 정량화된다.
SPLIT_MODE = "split"

# ── 규모 ──────────────────────────────────────────────────────────────
#   RePAQ 공식 retriever 는 CPU 에서 약 58 QA/s.
#   인덱스 20,000 -> 약 6분, 90,447 -> 약 26분.
N_INDEX = None                       # None = train 전체 (90,447)
#   ★ 20,000 으로 줄이면 gold 문서 커버리지가 80.2% -> 44.2% 로 떨어져
#     검색 성능의 상한이 반토막 난다. 실측 확인된 값이므로 전체를 쓸 것.
N_EVAL = 2_000                       # None = dev 전체

# ── 인덱스 문맥 구성 ──────────────────────────────────────────────────
#   "gold"     : supporting_facts 에 해당하는 문단만 (원본 QI-RAG 방식)
#   "gold+dist": gold + distractor 전부 (vanilla RAG 와 공정)
#   원본은 gold 만 담아 정답 문서가 보장됐다. vanilla RAG 는 distractor 가
#   섞인 코퍼스에서 찾으므로 비대칭이었다. 기본값은 공정 조건.
CONTEXT_MODE = "gold"
JOIN_SEP = "\n\n"                    # 문단 연결 구분자 (원본과 동일)
TITLE_PREFIX = True                  # "[제목] 본문" 형식 (원본 03 파싱 형식)

# ── Q13 문단 그룹 ─────────────────────────────────────────────────────
MIN_Q_PER_PASSAGE = 2                # 이 수 미만인 문서는 그룹에서 제외
MAX_Q_PER_PASSAGE = 8

# ── 질의 변형 ────────────────────────────────────────────────────────
#   original    원본 질문
#   keyword     ★ 기능어 제거 + 어순 재배열. 이전 이름은 "paraphrase" 였으나
#               어휘가 그대로이고 물음표와 어순이 사라져 질문 형태가 파괴된다.
#                 원본: Between Harvey Pekar and Denise Levertov who was born earlier?
#                 변형: Levertov Pekar Harvey Between born Denise earlier
#               패러프레이즈가 아니라 키워드 나열이므로 이름을 정정했다.
#               질의 인덱싱에 구조적으로 불리하고 문서 검색에는 유리하다.
#   noisy       DL-typo 실측 통계로 보정한 오타. 문장 구조는 유지된다.
#   paraphrase_llm  ★ LLM 재작성. 어휘를 바꾸되 의미와 질문 형태를 보존한다.
#               R4-T4("LLM 질의 재작성이 더 효과적일 것")에 답하려면 필요하다.
MAKE_VARIANTS = ["original", "keyword", "noisy", "paraphrase_llm"]
NOISE_WORD_FRAC = 0.25               # DL-typo 실측 24.9%
NOISE_ROUNDS = 1

# ── LLM 재작성 (paraphrase_llm 이 MAKE_VARIANTS 에 있을 때만) ─────────
LLM_URL = "http://172.25.121.170:8005/v1/chat/completions"
LLM_MODEL = "/home/jun/models/Qwen2.5-7B-Instruct-AWQ"
LLM_WORKERS = 8
LLM_TIMEOUT = 60
LLM_MAX_TOKENS = 64
PARAPHRASE_PROMPT = """Rewrite the question using different words while keeping \
the exact same meaning. Keep it a question. Do not answer it. \
Do not add or remove any information.

Question: {q}

Rewritten question:"""

# ── 필터 ──────────────────────────────────────────────────────────────
LEVEL_FILTER = None                  # None | {"hard"} | {"easy","medium"}
TYPE_FILTER = None                   # None | {"bridge"} | {"comparison"}
YESNO_KEEP = True                    # answer 가 yes/no 인 항목 유지 여부

SEED = 0
VERBOSE = True
# ══════════════════════════════════════════════════════════════════════


def log(*a):
    if VERBOSE:
        print(*a)


def _auto():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.getcwd(), here] + DATA_DIR_FALLBACK:
        if c and os.path.isdir(os.path.join(c, HOTPOT_DIR)):
            return c
    return os.getcwd()


_PUNC = re.compile(r"[^\w\s]")
_ART = re.compile(r"\b(a|an|the)\b")
_W = re.compile(r"[A-Za-z0-9']+")
STOP = set("a an the of in on for to with and or is are was were do does did "
           "what which who whom how why when where can could should would "
           "be been being have has had".split())


def norm(s):
    s = (s or "").lower()
    s = _PUNC.sub(" ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


# ── 질의 변형 ─────────────────────────────────────────────────────────

def _typo(w, rng):
    if len(w) < 4:
        return w
    kind = rng.choice(("del", "sub", "trans", "ins"))
    i = rng.randrange(1, len(w) - 1)
    if kind == "del":
        return w[:i] + w[i + 1:]
    if kind == "ins":
        return w[:i] + w[i] + w[i:]
    if kind == "trans":
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    near = {"a": "s", "s": "a", "e": "r", "r": "e", "i": "o", "o": "i",
            "n": "m", "m": "n", "t": "y", "y": "t", "c": "v", "v": "c",
            "l": "k", "k": "l", "u": "y", "d": "f", "f": "d"}
    return w[:i] + near.get(w[i].lower(), w[i]) + w[i + 1:]


def make_noisy(q, rng):
    for _ in range(NOISE_ROUNDS):
        ws = q.split()
        if not ws:
            return q
        n = max(1, int(round(len(ws) * NOISE_WORD_FRAC)))
        for i in rng.sample(range(len(ws)), min(n, len(ws))):
            ws[i] = _typo(ws[i], rng)
        q = " ".join(ws)
    return q


def make_paraphrase(q, rng):
    """기능어 제거 + 어순 재배열 = 'keyword' 변형.
    ★ 패러프레이즈가 아니다. 어휘가 그대로이고 물음표와 어순이 사라져
      질문 형태 자체가 파괴된다. 질의 인덱싱에는 구조적으로 불리하고
      문서 검색(키워드 매칭)에는 유리하다.
      실측 유보율: original 0.748 / keyword 0.827 / noisy 0.792
      진짜 패러프레이즈 강건성은 paraphrase_llm 조건으로 검증한다."""
    ws = _W.findall(q)
    keep = [w for w in ws if w.lower() not in STOP] or ws
    rng.shuffle(keep)
    return " ".join(keep)


# ── LLM 재작성 ────────────────────────────────────────────────────────

def call_llm(prompt):
    import urllib.request
    body = {"model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": LLM_MAX_TOKENS}
    req = urllib.request.Request(
        LLM_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()


def make_paraphrase_llm(questions):
    """LLM 으로 의미를 보존하며 어휘를 바꾼다.
    실패하거나 빈 응답이면 원문을 그대로 쓴다(그 건수를 보고할 것)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(q):
        try:
            t = call_llm(PARAPHRASE_PROMPT.format(q=q)).strip()
            t = t.strip('"').strip()
            # 모델이 설명을 덧붙이면 첫 줄만 쓴다
            t = t.split("\n")[0].strip()
            return t if len(t) >= 5 else q
        except Exception:
            return q

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        out = list(ex.map(one, questions))
    n_fail = sum(1 for a, b in zip(questions, out) if a == b)
    log(f"  LLM 재작성 {len(questions)}건 {time.time()-t0:.0f}초  "
        f"(원문 유지 {n_fail}건)")
    return out, n_fail


# ── HotpotQA 파싱 ─────────────────────────────────────────────────────

def parse_row(r):
    """HF 형식(dict) 과 원본 JSON 형식(list) 을 모두 처리한다."""
    ctx = r["context"]
    if isinstance(ctx, dict):                       # HF: {"title":[], "sentences":[[]]}
        titles = ctx["title"]
        sents = ctx["sentences"]
    else:                                           # 원본: [[title, [sents]], ...]
        titles = [c[0] for c in ctx]
        sents = [c[1] for c in ctx]

    sf = r["supporting_facts"]
    if isinstance(sf, dict):
        gold_titles = list(dict.fromkeys(sf["title"]))
    else:
        gold_titles = list(dict.fromkeys(t for t, _ in sf))

    paras = {}
    for t, ss in zip(titles, sents):
        paras[t] = "".join(ss).strip()

    return {"qid": r.get("id") or r.get("_id"),
            "question": r["question"],
            "answer": r["answer"],
            "gold_titles": [t for t in gold_titles if t in paras],
            "paras": paras,
            "type": r.get("type"), "level": r.get("level")}


def build_context(item, mode):
    if mode == "gold":
        titles = item["gold_titles"]
    else:
        titles = list(item["paras"])                # gold + distractor
    parts = []
    for t in titles:
        body = item["paras"].get(t, "")
        parts.append(f"[{t}] {body}" if TITLE_PREFIX else body)
    return JOIN_SEP.join(parts)


def load_jsonl(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and len(out) >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def passes_filter(it):
    if LEVEL_FILTER and it["level"] not in LEVEL_FILTER:
        return False
    if TYPE_FILTER and it["type"] not in TYPE_FILTER:
        return False
    if not YESNO_KEEP and norm(it["answer"]) in ("yes", "no"):
        return False
    if not it["gold_titles"]:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════

def main():
    global DATA_DIR, WORK_DIR
    if DATA_DIR is None:
        DATA_DIR = _auto()
    if WORK_DIR is None:
        WORK_DIR = os.path.join(
            DATA_DIR, "_split" if SPLIT_MODE == "split" else "_split_leak")
    os.makedirs(WORK_DIR, exist_ok=True)
    hp = os.path.join(DATA_DIR, HOTPOT_DIR)

    print("=" * 74)
    print("80 — 분할 고정 (HotpotQA)")
    print("=" * 74)
    print(f"  DATA_DIR : {DATA_DIR}")
    print(f"  WORK_DIR : {WORK_DIR}")
    print(f"  분할     : {SPLIT_MODE}"
          + ("  (train 인덱스 / dev 평가)" if SPLIT_MODE == "split"
             else "  ★ 원본 재현 — 인덱스 = 평가, 자기검색 발생"))
    if SPLIT_MODE == "split":
        print(f"  인덱스   : train {N_INDEX or '전체'}   문맥 {CONTEXT_MODE}")
        print(f"  평가     : dev {N_EVAL or '전체'}   변형 {MAKE_VARIANTS}")
    else:
        print(f"  인덱스=평가: dev {N_EVAL or '전체'}   문맥 {CONTEXT_MODE}")
        print(f"  변형     : {MAKE_VARIANTS}")
    print(f"  필터     : level={LEVEL_FILTER} type={TYPE_FILTER} "
          f"yes/no={'유지' if YESNO_KEEP else '제외'}")

    for f in (TRAIN_FILE, DEV_FILE):
        p = os.path.join(hp, f)
        if not os.path.exists(p):
            print(f"\n  [실패] {p} 가 없습니다.")
            print("  HuggingFace 에서 먼저 받으세요:")
            print("    from datasets import load_dataset")
            print("    ds = load_dataset('hotpotqa/hotpot_qa', 'distractor', "
                  "split='train')")
            return

    rng = random.Random(SEED)

    # ── 인덱스
    if SPLIT_MODE == "split":
        log("\n[1] 인덱스 (train)")
        raw = load_jsonl(os.path.join(hp, TRAIN_FILE))
        log(f"  train 원본 {len(raw):,}")
        items = [parse_row(r) for r in raw]
        items = [it for it in items if passes_filter(it)]
        log(f"  필터 통과 {len(items):,}")
        rng.shuffle(items)
        if N_INDEX:
            items = items[:N_INDEX]
    else:
        # 원본 재현: dev 에서 뽑아 인덱스와 평가에 모두 쓴다
        log("\n[1] 인덱스 (dev — 평가와 동일 집합)")
        raw = load_jsonl(os.path.join(hp, DEV_FILE))
        log(f"  dev 원본 {len(raw):,}")
        items = [parse_row(r) for r in raw]
        items = [it for it in items if passes_filter(it)]
        log(f"  필터 통과 {len(items):,}")
        rng.shuffle(items)
        if N_EVAL:
            items = items[:N_EVAL]

    index_items = []
    for it in items:
        index_items.append({
            "qid": it["qid"], "question": it["question"],
            "answer": [it["answer"]],
            "context": build_context(it, CONTEXT_MODE),
            "gold_titles": it["gold_titles"],
            "type": it["type"], "level": it["level"],
        })
    log(f"  인덱스 항목 {len(index_items):,}")

    # ── 평가
    if SPLIT_MODE == "split":
        log("\n[2] 평가 질의 (dev)")
        raw_d = load_jsonl(os.path.join(hp, DEV_FILE))
        log(f"  dev 원본 {len(raw_d):,}")
        devs = [parse_row(r) for r in raw_d]
        devs = [it for it in devs if passes_filter(it)]
        rng2 = random.Random(SEED + 1)
        rng2.shuffle(devs)
        if N_EVAL:
            devs = devs[:N_EVAL]
    else:
        log("\n[2] 평가 질의 (인덱스와 동일 집합)")
        devs = list(items)          # ★ 인덱스와 같은 항목
    log(f"  평가 질의 {len(devs):,}")

    eval_rows = []
    for i, it in enumerate(devs):
        q = it["question"]
        row = {"qid": it["qid"], "answer": [it["answer"]],
               "gold_titles": it["gold_titles"],
               "type": it["type"], "level": it["level"],
               "original": q}
        if "keyword" in MAKE_VARIANTS:
            row["keyword"] = make_paraphrase(q, random.Random(SEED * 100 + i))
        if "noisy" in MAKE_VARIANTS:
            row["noisy"] = make_noisy(q, random.Random(SEED * 200 + i))
        eval_rows.append(row)

    n_llm_fail = 0
    if "paraphrase_llm" in MAKE_VARIANTS:
        log("\n[2-b] LLM 재작성")
        try:
            call_llm("Reply with exactly: OK")
        except Exception as e:
            print(f"  [실패] LLM 서버 접속 불가: {type(e).__name__}: {e}")
            print(f"  {LLM_URL}")
            print("  MAKE_VARIANTS 에서 'paraphrase_llm' 을 빼거나 서버를 켜세요.")
            return
        rewritten, n_llm_fail = make_paraphrase_llm([r["original"]
                                                    for r in eval_rows])
        for r, t in zip(eval_rows, rewritten):
            r["paraphrase_llm"] = t
        # 샘플 확인
        for r in eval_rows[:2]:
            log(f"    원본: {r['original'][:70]}")
            log(f"    재작성: {r['paraphrase_llm'][:70]}")

    # ── 누출 감사
    log("\n[3] 누출 감사")
    idx_norm = {norm(r["question"]) for r in index_items}
    hit = sum(1 for e in eval_rows if norm(e["original"]) in idx_norm)
    rate = 100 * hit / max(len(eval_rows), 1)
    log(f"  평가 질의 중 인덱스에 동일 표면형 {hit}/{len(eval_rows)} "
        f"({rate:.2f}%)")
    if SPLIT_MODE == "split":
        log("  (train/dev 공식 분할이므로 0 에 가까워야 정상)")
        if rate > 1.0:
            log("  ! 누출이 있습니다. 분할을 확인하세요.")
    else:
        log("  (원본 재현 조건이므로 100% 여야 정상 — Original 질의가")
        log("   인덱스에 문자 그대로 존재한다)")
        if rate < 99.0:
            log("  ! 100% 가 아닙니다. 재현 조건이 성립하지 않습니다.")
    # gold 문서가 인덱스에 존재하는 비율 = 성능 상한
    idx_titles = set()
    for r in index_items:
        idx_titles.update(r["gold_titles"])
    cover = sum(1 for e in eval_rows
                if any(t in idx_titles for t in e["gold_titles"]))
    log(f"  평가 질의의 gold 문서가 인덱스에 존재 "
        f"{cover}/{len(eval_rows)} ({100*cover/max(len(eval_rows),1):.1f}%)")
    log("  → 이 값이 검색 성능의 상한이다. 낮으면 N_INDEX 를 키울 것.")

    if SPLIT_MODE == "leak":
        log("\n  ※ 이 조건에서는 검색 시 자기 자신을 제외해야 의미 있는 비교가")
        log("    된다. 81/84 에서 EXCLUDE_SELF 를 켜거나, 이 조건은 '원본이")
        log("    어떤 수치를 냈는가' 를 보여주는 용도로만 쓸 것.")

    # ── Q13 문단 그룹
    log("\n[4] 문단 그룹 (Q13 용)")
    groups = defaultdict(list)
    for r in index_items:
        for t in r["gold_titles"]:
            if len(groups[t]) < MAX_Q_PER_PASSAGE:
                groups[t].append(r["qid"])
    groups = {k: v for k, v in groups.items() if len(v) >= MIN_Q_PER_PASSAGE}
    gsz = Counter(len(v) for v in groups.values())
    log(f"  {MIN_Q_PER_PASSAGE}건 이상 공유 문서 {len(groups):,}")
    log(f"  분포 {dict(sorted(gsz.items())[:8])}")
    if len(groups) < 500:
        log("  ! 그룹이 적습니다. N_INDEX 를 키우면 늘어납니다.")
        log("    (train 90,447 전체 기준 2건 이상 공유 문서 36,126개)")

    # ── 저장
    log("\n[5] 저장")
    def dump(name, rows):
        p = os.path.join(WORK_DIR, name)
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"  {name:<28}{len(rows):>8,}")
        return p

    dump("index_items.jsonl", index_items)
    # RePAQ 공식 코드 입력 형식: {"question", "answer"}
    dump("index_qa.jsonl",
         [{"question": r["question"], "answer": r["answer"]} for r in index_items])
    dump("eval_queries.jsonl", eval_rows)
    for v in MAKE_VARIANTS:
        short = {"original": "q", "keyword": "p", "noisy": "n",
                 "paraphrase_llm": "pl"}[v]
        dump(f"eval_qa_{short}.jsonl",
             [{"question": r[v], "answer": r["answer"]} for r in eval_rows])

    with open(os.path.join(WORK_DIR, "passage_groups.json"), "w",
              encoding="utf-8") as f:
        json.dump(groups, f)
    log(f"  {'passage_groups.json':<28}{len(groups):>8,}")

    meta = {
        "config": {"split_mode": SPLIT_MODE,
                   "n_index": N_INDEX, "n_eval": N_EVAL,
                   "context_mode": CONTEXT_MODE, "seed": SEED,
                   "variants": MAKE_VARIANTS,
                   "noise_word_frac": NOISE_WORD_FRAC,
                   "min_q_per_passage": MIN_Q_PER_PASSAGE,
                   "max_q_per_passage": MAX_Q_PER_PASSAGE,
                   "level_filter": list(LEVEL_FILTER or []),
                   "type_filter": list(TYPE_FILTER or []),
                   "yesno_keep": YESNO_KEEP},
        "stats": {"index": len(index_items), "eval": len(eval_rows),
                  "leak_overlap": hit, "gold_in_index": cover,
                  "gold_in_index_rate": cover / max(len(eval_rows), 1),
                  "groups": len(groups), "group_size_hist": dict(gsz),
                  "index_type": dict(Counter(r["type"] for r in index_items)),
                  "index_level": dict(Counter(r["level"] for r in index_items)),
                  "llm_paraphrase_fallback": n_llm_fail,
                  "eval_type": dict(Counter(r["type"] for r in eval_rows)),
                  "eval_level": dict(Counter(r["level"] for r in eval_rows))},
    }
    with open(os.path.join(WORK_DIR, "split_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ── 샘플
    log("\n[6] 샘플")
    r = index_items[0]
    log(f"  [인덱스] Q: {r['question'][:70]}")
    log(f"           A: {r['answer']}")
    log(f"           C: {r['context'][:120]}...")
    log(f"           gold_titles: {r['gold_titles']}  {r['type']}/{r['level']}")
    e = eval_rows[0]
    log(f"  [평가]   O : {e['original'][:70]}")
    if "keyword" in e:
        log(f"           K : {e['keyword'][:70]}")
    if "noisy" in e:
        log(f"           N : {e['noisy'][:70]}")
    if "paraphrase_llm" in e:
        log(f"           PL: {e['paraphrase_llm'][:70]}")
    log(f"           A: {e['answer']}")

    print("\n" + "=" * 74)
    print("다음")
    print("=" * 74)
    print("  [vllm_env]  python 81_run_qirag.py")
    print("  [repaq]     bash   82_run_repaq.sh")
    print("  [아무데나]   python 83_compare.py")
    print("\n  두 실험은 이 폴더의 파일만 읽는다. 재실행해도 분할이 바뀌지 않는다.")
    print(f"  {WORK_DIR}")
    if SPLIT_MODE == "split":
        print("\n  원본 조건과 대조하려면 SPLIT_MODE='leak' 으로 한 번 더 실행할 것.")
        print("  산출물이 _split_leak 에 따로 저장되므로 덮어쓰지 않는다.")
        print("  81/84 실행 시 SPLIT_DIR 을 그쪽으로 바꿔 돌리면 된다.")
    else:
        print("\n  이 조건의 수치는 '누출이 있는 상태' 다. 주 결과가 아니라")
        print("  _split(공식 분할) 과의 대조용으로만 보고할 것.")
        print("  대조 문장 예: '원 조건에서 0.xxx 였으나 누출 제거 후 0.183 이다.'")


if __name__ == "__main__":
    main()
