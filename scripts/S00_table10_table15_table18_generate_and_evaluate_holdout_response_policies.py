"""
llm_experiments_v2.py  (LLM 실험 v2: 검증·재시도·보고 수정, 기존 동결과 호환)
==========================================================================
실행 위치: ~/2610-qi-rag

  python llm_experiments_v2.py --freeze-prompts   (이미 동결돼 있으면 중단. 기존 동결 유지)
  python llm_experiments_v2.py --run              생성 (오류·빈 답은 재시도, 완료된 문항은 건너뜀)
  python llm_experiments_v2.py --status           진행 상황
  python llm_experiments_v2.py --validate         생성 파일 완전성·무효 답변 검사 (채점 전 필수)
  python llm_experiments_v2.py --tokens           실제 토큰 수 재측정 (토크나이저, 생성 없음)
  python llm_experiments_v2.py --score            채점 (--validate 통과 후)

v1 과의 호환
  - 프롬프트, 설정, 조건 이름, 모델, 온도, 토큰 한도, P 는 v1 과 동일 -> 동결 해시 그대로.
  - v1 로 생성한 파일을 그대로 읽는다. 같은 문항에 기록이 여러 개면 마지막 기록을 쓴다.

보고 규칙 (수정)
  - H2 는 '정책 비교'다. QI_STRICT = 상위 고유 문서 2개 제공, QA_STRICT = 상위 질문 정답 2개 제공.
    두 조건은 제공 정보의 형식과 양이 다르므로 효과를 원문 근거의 가치로 인과 해석하지 않는다.
  - 프롬프트 길이는 '공백 단위 길이'로만 표기한다. 실제 토큰 수는 --tokens 로 따로 잰다.
  - H3-A 는 모든 문항에 유효한 후보 답이 있을 때만 주 결과로 보고한다. 무효 후보가 있으면 미완료로 표기하고,
    무효 답을 오답으로 센 민감도만 함께 보고한다.
  - AURC 차이에도 paired bootstrap CI 를 붙인다. 재표본마다 두 방법의 신뢰도 순위를 다시 매긴다.
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from S00_qi_common import DEV, INDEX, LOCKED, build_corpus, load_jsonl, rid, sha256_file, sha256_text
from S00_retrieval_policy import unique_take
ROOT = str(_PACKAGE_ROOT)
VLLM_URL = 'http://172.25.121.198:7777/v1/chat/completions'
VLLM_MODEL = 'Qwen2.5-7B-Instruct-AWQ'
LOCAL_TOKENIZER = os.path.expanduser('~/models/Qwen2.5-7B-Instruct-AWQ')
TEMPERATURE = 0.0
MAX_TOKENS = 128
TIMEOUT = 90
WORKERS = 8
RETRIES = 2
CHUNK = 200
P = 2
N_BOOT = 10000
BOOT_SEED = 0
COVERAGES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
STRICT = 'Answer ONLY using the given context.\nAnswer with the shortest possible phrase.\nIf the answer is not in the context, say "I don\'t know".\n\nContext:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:'
CAND = 'Answer the question using the given context.\nAlways give your single best answer as a short phrase, even if uncertain. Do not say "I don\'t know".\n\nContext:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:'
CONDS = ['QI_STRICT', 'VAN_STRICT', 'QA_STRICT', 'QI_CAND', 'VAN_CAND']
PRED = _pkg_resolve(os.path.join(LOCKED, 'retrieval_predictions.jsonl'))
SPLIT = _pkg_resolve(os.path.join(LOCKED, 'split_ids.json'))
FREEZE = _pkg_resolve(os.path.join(LOCKED, 'FREEZE.json'))
H1_STATS = _pkg_resolve(os.path.join(ROOT, 'out', 'holdout_final', 'e1a_statistics.json'))
LLM_FREEZE = _pkg_resolve(os.path.join(LOCKED, 'llm_prompts_FREEZE.json'))
OUT_GEN = _pkg_resolve(os.path.join(ROOT, 'out', 'holdout_llm'))
OUT_SCORE = _pkg_resolve(os.path.join(ROOT, 'out', 'holdout_llm_final'))
OUT_VAL = _pkg_resolve(os.path.join(ROOT, 'out', 'holdout_llm_validation'))

def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')

def cfg_hash():
    cfg = {'model': VLLM_MODEL, 'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS, 'P': P, 'conds': CONDS, 'strict_sha': sha256_text(STRICT), 'cand_sha': sha256_text(CAND)}
    return (sha256_text(json.dumps(cfg, sort_keys=True)), cfg)

def is_abstain(t):
    t = (t or '').lower()
    return not t.strip() or "don't know" in t or 'do not know' in t

def norm(s):
    s = (s or '').lower()
    s = re.sub('[^\\w\\s]', ' ', s)
    s = re.sub('\\b(a|an|the)\\b', ' ', s)
    return ' '.join(s.split())

def em(pred, golds):
    if is_abstain(pred):
        return False
    p = norm(pred)
    return any((norm(g) and p == norm(g) for g in golds))

def gate(need_server=False):
    problems = []
    if not os.path.exists(FREEZE):
        problems.append('FREEZE.json 없음')
    else:
        fz = json.load(open(_pkg_open_path(_pkg_resolve(FREEZE), 'r'), encoding='utf-8'))
        if not os.path.exists(SPLIT) or sha256_file(SPLIT) != fz['split_ids_sha256']:
            problems.append('분할 해시가 동결 기록과 다름')
    if not os.path.exists(H1_STATS):
        problems.append('H1 확증 결과가 없음')
    if not os.path.exists(LLM_FREEZE):
        problems.append('llm_prompts_FREEZE.json 없음')
    else:
        lf = json.load(open(_pkg_open_path(_pkg_resolve(LLM_FREEZE), 'r'), encoding='utf-8'))
        if lf.get('config_sha256') != cfg_hash()[0]:
            problems.append('프롬프트/설정이 동결 기록과 다름')
        if lf.get('retrieval_predictions_sha256') != sha256_file(PRED):
            problems.append('검색 결과 파일이 동결 기록과 다름')
    if need_server:
        try:
            call_llm('Reply with exactly: OK', 8)
        except Exception as e:
            problems.append(f'생성 서버 응답 없음: {type(e).__name__}')
    return problems

def call_llm(prompt, max_tokens=MAX_TOKENS):
    body = {'model': VLLM_MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': TEMPERATURE, 'max_tokens': max_tokens}
    req = urllib.request.Request(VLLM_URL, data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read())
    return d['choices'][0]['message']['content'].strip()

def gen_one(prompt):
    last = ''
    for attempt in range(RETRIES + 1):
        try:
            txt = call_llm(prompt)
            if not txt.strip():
                return ('', 'empty')
            return (txt, 'ok')
        except Exception as e:
            last = f'{type(e).__name__}: {e}'
            time.sleep(2.0 * (attempt + 1))
    return ('', 'error: ' + last[:200])

class Inputs:

    def __init__(self):
        index = load_jsonl(INDEX)
        self.index_answer = [str(r.get('answer', '')) for r in index]
        docs = build_corpus(index)
        self.doc_by_title = {d['title']: d for d in docs if d['title']}
        self.rows = load_jsonl(PRED)

    def fmt(self, title):
        d = self.doc_by_title.get(title)
        return f"[{title}] {d['text']}" if d else ''

    def ctx_passages(self, titles):
        return '\n\n'.join((self.fmt(t) for t in titles if t))

    def build(self, cond, r):
        if cond == 'QI_STRICT':
            return STRICT.format(ctx=self.ctx_passages(r['qi_titles']), q=r['_q'])
        if cond == 'VAN_STRICT':
            return STRICT.format(ctx=self.ctx_passages(r['va_titles']), q=r['_q'])
        if cond == 'QA_STRICT':
            items = [x['item'] for x in r['qi_top_items']]
            answers = [self.index_answer[j] for j in items if self.index_answer[j]]
            answers = list(dict.fromkeys(answers))[:P]
            ctx = '\n'.join((f'Stored answer: {a}' for a in answers))
            return STRICT.format(ctx=ctx, q=r['_q'])
        if cond == 'QI_CAND':
            return CAND.format(ctx=self.ctx_passages(r['qi_titles']), q=r['_q'])
        if cond == 'VAN_CAND':
            return CAND.format(ctx=self.ctx_passages(r['va_titles']), q=r['_q'])
        raise ValueError(cond)

    def top1_score(self, cond, r):
        if cond.startswith('VAN'):
            return float(r['va_top_docs'][0]['score'])
        return float(r['qi_top_items'][0]['score'])

def load_questions():
    want = set((str(r['qid']) for r in load_jsonl(PRED)))
    q = {}
    with open(_pkg_open_path(_pkg_resolve(DEV), 'r'), encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            i = rid(d)
            if i in want:
                q[i] = {'q': d['question'], 'gold': [d['answer']] if d.get('answer') else []}
    return q

def latest_records(cond):
    """같은 문항에 기록이 여러 개면 마지막 기록을 쓴다 (재시도 결과가 우선)."""
    path = _pkg_resolve(os.path.join(OUT_GEN, f'{cond}.jsonl'))
    recs = {}
    if os.path.exists(path):
        with open(_pkg_open_path(_pkg_resolve(path), 'r'), encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[str(r['qid'])] = r
    return recs

def cmd_freeze_prompts():
    if os.path.exists(LLM_FREEZE):
        print('이미 동결됨. 덮어쓰지 않습니다. 기존 동결 해시를 유지합니다.')
        sys.exit(1)
    if not os.path.exists(FREEZE):
        print('FREEZE.json 없음.')
        sys.exit(1)
    (h, cfg) = cfg_hash()
    rec = {'frozen_at': now(), 'config_sha256': h, 'config': cfg, 'retrieval_predictions_sha256': sha256_file(PRED)}
    with open(_pkg_open_path(_pkg_resolve(LLM_FREEZE), 'w'), 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f'동결 완료: {LLM_FREEZE}  설정 해시 {h[:16]}...')

def cmd_run():
    probs = gate(need_server=True)
    if probs:
        print('[실행 안 함] 게이트 실패:')
        for p in probs:
            print('  -', p)
        sys.exit(1)
    os.makedirs(OUT_GEN, exist_ok=True)
    inp = Inputs()
    qmap = load_questions()
    for r in inp.rows:
        r['_q'] = qmap[str(r['qid'])]['q']
    print(f'Holdout {len(inp.rows):,}문항, 조건 {len(CONDS)}개')
    t_all = time.time()
    for cond in CONDS:
        path = _pkg_resolve(os.path.join(OUT_GEN, f'{cond}.jsonl'))
        recs = latest_records(cond)
        todo = [r for r in inp.rows if recs.get(str(r['qid']), {}).get('status') != 'ok']
        print(f'\n[{cond}] 정상 완료 {len(inp.rows) - len(todo):,} / 생성·재시도 대상 {len(todo):,}')
        if not todo:
            continue
        t0 = time.time()
        n_done = 0
        with open(_pkg_open_path(_pkg_resolve(path), 'a'), 'a', encoding='utf-8') as f, ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for s in range(0, len(todo), CHUNK):
                chunk = todo[s:s + CHUNK]
                prompts = [inp.build(cond, r) for r in chunk]
                results = list(ex.map(gen_one, prompts))
                for (r, prompt, (txt, st)) in zip(chunk, prompts, results):
                    rec = {'qid': str(r['qid']), 'cond': cond, 'pred': txt, 'status': st, 'err': '' if st == 'ok' else st, 'prompt_words': len(prompt.split()), 'top1_score': inp.top1_score(cond, r), 'at': now()}
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
                n_done += len(chunk)
                rate = _pkg_resolve(n_done / max(time.time() - t0, 1e-06))
                print(f'  {n_done:,}/{len(todo):,}  {rate:.2f}건/초  남은 약 {_pkg_resolve(_pkg_resolve((len(todo) - n_done) / rate) / 60):.0f}분')
    print(f'\n생성 종료. 총 {_pkg_resolve((time.time() - t_all) / 60):.1f}분. 다음: --validate')

def cmd_status():
    qids = [str(r['qid']) for r in load_jsonl(PRED)]
    for cond in CONDS:
        recs = latest_records(cond)
        ok = sum((1 for q in qids if recs.get(q, {}).get('status') == 'ok'))
        print(f'  {cond:<11} 정상 {ok:>5,}/{len(qids):,}')

def validate_all():
    qids = [str(r['qid']) for r in load_jsonl(PRED)]
    (out, fatal) = ({}, False)
    for cond in CONDS:
        recs = latest_records(cond)
        missing = [q for q in qids if q not in recs]
        extra = [q for q in recs if q not in set(qids)]
        errors = sum((1 for q in qids if q in recs and recs[q]['status'] not in ('ok', 'empty')))
        empty = sum((1 for q in qids if q in recs and recs[q]['status'] != 'ok'))
        abst = sum((1 for q in qids if q in recs and recs[q]['status'] == 'ok' and is_abstain(recs[q]['pred'])))
        complete = not missing and (not extra) and (errors == 0) and (empty == 0) and (len(set(qids)) == len(qids))
        if not complete:
            fatal = True
        info = {'n_expected': len(qids), 'missing': len(missing), 'extra': len(extra), 'errors': errors, 'empty_or_failed': empty, 'abstain': abst, 'complete': complete}
        if cond.endswith('CAND'):
            info['valid_candidates'] = len(qids) - abst - empty - len(missing)
            info['h3a_primary_eligible'] = complete and abst == 0
        out[cond] = info
    os.makedirs(OUT_VAL, exist_ok=True)
    with open(_pkg_open_path(_pkg_resolve(_pkg_resolve(os.path.join(OUT_VAL, 'validation.json'))), 'w'), 'w', encoding='utf-8') as f:
        json.dump({'created_at': now(), 'conditions': out, 'all_complete': not fatal}, f, ensure_ascii=False, indent=2)
    return (out, fatal)

def cmd_validate():
    (out, fatal) = validate_all()
    for (cond, info) in out.items():
        line = f"  {cond:<11} 완료 {('예' if info['complete'] else '아니오')}  누락 {info['missing']}  오류 {info['errors']}  빈답 {info['empty_or_failed']}  유보 {info['abstain']}"
        if cond.endswith('CAND'):
            line += f"  H3-A 주 분석 {('가능' if info['h3a_primary_eligible'] else '불가 (무효 후보 있음)')}"
        print(line)
    print('\n전체 완료' if not fatal else '\n[주의] 미완료 조건이 있어 --score 가 중단됩니다.')
    print(f"저장: {_pkg_resolve(os.path.join(OUT_VAL, 'validation.json'))}")

def cmd_tokens():
    from transformers import AutoTokenizer
    src = LOCAL_TOKENIZER if _pkg_isdir(LOCAL_TOKENIZER) else 'Qwen/Qwen2.5-7B-Instruct'
    print(f'토크나이저 원본: {src}')
    tok = AutoTokenizer.from_pretrained(src)
    inp = Inputs()
    qmap = load_questions()
    for r in inp.rows:
        r['_q'] = qmap[str(r['qid'])]['q']
    res = {}
    for cond in CONDS:
        counts = []
        for r in inp.rows:
            ids = tok.apply_chat_template([{'role': 'user', 'content': inp.build(cond, r)}], tokenize=True, add_generation_prompt=True)
            if isinstance(ids, dict):
                ids = ids['input_ids']
            counts.append(len(ids))
        a = np.array(counts)
        res[cond] = {'mean': float(a.mean()), 'median': float(np.median(a)), 'max': int(a.max()), 'min': int(a.min()), 'n': len(counts)}
        print(f'  {cond:<11} 토큰 평균 {a.mean():.1f}  중앙값 {np.median(a):.0f}  최대 {a.max()}')
    os.makedirs(OUT_SCORE, exist_ok=True)
    path = _pkg_resolve(os.path.join(OUT_SCORE, 'token_counts.json'))
    with open(_pkg_open_path(_pkg_resolve(path), 'w'), 'w', encoding='utf-8') as f:
        json.dump({'created_at': now(), 'tokenizer': src, 'conditions': res}, f, ensure_ascii=False, indent=2)
    print(f'저장: {path}')

def paired_ci(d):
    n = len(d)
    rng = np.random.Generator(np.random.PCG64(BOOT_SEED))
    boots = np.empty(N_BOOT)
    for s in range(0, N_BOOT, 1000):
        idx = rng.integers(0, n, (1000, n))
        boots[s:s + 1000] = d[idx].mean(axis=1)
    (lo, hi) = np.percentile(boots, [2.5, 97.5])
    return (float(d.mean()), float(lo), float(hi))

def ranking(score, tie):
    return np.lexsort((tie, -score))

def risk_at(ok, score, tie, c):
    n = len(ok)
    k = int(np.floor(c * n))
    if k == 0:
        return float('nan')
    order = ranking(score, tie)
    return float(1.0 - ok[order][:k].mean())

def aurc(ok, score, tie):
    order = ranking(score, tie)
    cum = np.cumsum(ok[order])
    risks = 1.0 - _pkg_resolve(cum / np.arange(1, len(ok) + 1))
    return float(risks.mean())

def cmd_score():
    probs = gate(need_server=False)
    if probs:
        print('[채점 안 함] 게이트 실패:')
        for p in probs:
            print('  -', p)
        sys.exit(1)
    (out, fatal) = validate_all()
    if fatal:
        print('[채점 안 함] 생성 파일이 완전하지 않습니다. --validate 결과를 확인하십시오.')
        sys.exit(1)
    if os.path.exists(_pkg_resolve(os.path.join(OUT_SCORE, 'llm_report.txt'))):
        print('[채점 안 함] 기존 결과가 있음 (덮어쓰기 금지)')
        sys.exit(1)
    inp = Inputs()
    qids = [str(r['qid']) for r in inp.rows]
    qmap = load_questions()
    gen = {c: latest_records(c) for c in CONDS}
    n = len(qids)

    def ok_arr(c):
        return np.array([em(gen[c][q]['pred'], qmap[q]['gold']) for q in qids], dtype=float)

    def ans_arr(c):
        return np.array([not is_abstain(gen[c][q]['pred']) for q in qids], dtype=float)
    ok = {c: ok_arr(c) for c in CONDS}
    ans = {c: ans_arr(c) for c in CONDS}
    words = {c: np.array([gen[c][q].get('prompt_words', gen[c][q].get('ctx_words', 0)) for q in qids]) for c in CONDS}
    tok_path = _pkg_resolve(os.path.join(OUT_SCORE, 'token_counts.json'))
    if os.path.exists(tok_path):
        tk_all = json.load(open(_pkg_open_path(_pkg_resolve(tok_path), 'r'), encoding='utf-8'))
        tk = tk_all['conditions']
        tok_line = f"  실제 토큰 평균  QI {tk['QI_STRICT']['mean']:.1f}  QA {tk['QA_STRICT']['mean']:.1f}  (토크나이저: {tk_all['tokenizer']})"
    else:
        tok_line = '  실제 토큰 수: 미측정 (먼저 python llm_experiments_v2.py --tokens 실행)'
    lines = [f'LLM 실험 채점 v2 (Holdout {n:,}문항, 생성 완전성 통과)', '=' * 74, '']
    (d_h2, lo_h2, hi_h2) = paired_ci(ok['QI_STRICT'] - ok['QA_STRICT'])
    lines += ['[H2] 정책 비교: QI_STRICT (상위 고유 문서 2개 제공) vs QA_STRICT (상위 질문의 저장 정답 2개 제공)', f"  EM  QI_STRICT {ok['QI_STRICT'].mean():.4f}   QA_STRICT {ok['QA_STRICT'].mean():.4f}", f'  차이 {d_h2:+.4f}  95% CI [{lo_h2:+.4f}, {hi_h2:+.4f}]  (사전 지정 2차, 확증 주장 없음)', tok_line, f"  (참고) 공백 단위 길이 평균  QI {words['QI_STRICT'].mean():.1f}  QA {words['QA_STRICT'].mean():.1f}  — 토큰 수 아님", '  해석 제한: 두 조건은 제공 정보의 형식과 양이 함께 다르다. 원문 근거의 가치로 인과 해석하지 않는다.', '']
    tie = np.array([int(hashlib.sha256(f'0|{q}'.encode('utf-8')).hexdigest()[:16], 16) for q in qids], dtype=np.uint64)
    score = {'QI': np.array([gen['QI_CAND'][q]['top1_score'] for q in qids]), 'VAN': np.array([gen['VAN_CAND'][q]['top1_score'] for q in qids])}
    ok_c = {'QI': ok['QI_CAND'], 'VAN': ok['VAN_CAND']}
    invalid_qi = int(n - ans['QI_CAND'].sum())
    invalid_van = int(n - ans['VAN_CAND'].sum())
    primary = invalid_qi == 0 and invalid_van == 0
    head = '[H3-A] 동일 Coverage 선택적 위험 (후보 답변, 신뢰도 = 검색 Top-1 유사도)'
    if primary:
        lines.append(head + '  — 주 결과')
    else:
        lines.append(head + '  — 미완료 (무효 후보 있음: QI ' + str(invalid_qi) + ', VAN ' + str(invalid_van) + '). 아래 수치는 무효 답을 오답으로 센 민감도 분석')
    lines.append(f"  {'Coverage':<10}{'k':>6}{'QI 위험':>10}{'VAN 위험':>10}{'차이':>10}{'CI':>26}")
    idx_all = np.random.Generator(np.random.PCG64(BOOT_SEED)).integers(0, n, (N_BOOT, n))
    for c in COVERAGES:
        rq = risk_at(ok_c['QI'], score['QI'], tie, c)
        rv = risk_at(ok_c['VAN'], score['VAN'], tie, c)
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = idx_all[i]
            diffs[i] = risk_at(ok_c['QI'][idx], score['QI'][idx], tie[idx], c) - risk_at(ok_c['VAN'][idx], score['VAN'][idx], tie[idx], c)
        diffs = diffs[~np.isnan(diffs)]
        (lo, hi) = np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (np.nan, np.nan)
        k = int(np.floor(c * n))
        lines.append(f'  {int(c * 100):>3}%      {k:>6}{rq:>10.4f}{rv:>10.4f}{rq - rv:>+10.4f}   [{lo:+.4f}, {hi:+.4f}]')
    (aq, av) = (aurc(ok_c['QI'], score['QI'], tie), aurc(ok_c['VAN'], score['VAN'], tie))
    aurc_diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = idx_all[i]
        aurc_diffs[i] = aurc(ok_c['QI'][idx], score['QI'][idx], tie[idx]) - aurc(ok_c['VAN'][idx], score['VAN'][idx], tie[idx])
    (alo, ahi) = np.percentile(aurc_diffs, [2.5, 97.5])
    lines.append(f'  AURC  QI {aq:.4f}   VAN {av:.4f}   차이 {aq - av:+.4f}   CI [{alo:+.4f}, {ahi:+.4f}]')
    lines.append('  해석: 검색 유사도 기반 선택 정책의 결과. 정답 확률로 해석하지 않는다.')
    lines.append('')
    lines.append('[H3-B] Strict Prompt 운영 성능 (보조, 응답률이 방법마다 다름)')
    lines.append(f"  {'방법':<12}{'응답률':>9}{'전체 정답':>11}{'응답 중 오답':>14}{'전체 오답':>11}{'유보율':>9}")
    for c in ['QI_STRICT', 'VAN_STRICT']:
        (a, o) = (ans[c], ok[c])
        cov = a.mean()
        wrong_ans = _pkg_resolve(((a == 1) & (o == 0)).sum() / max(a.sum(), 1))
        wrong_all = ((a == 1) & (o == 0)).mean()
        lines.append(f'  {c:<12}{cov:>9.4f}{o.mean():>11.4f}{wrong_ans:>14.4f}{wrong_all:>11.4f}{1 - cov:>9.4f}')
    lines.append('  응답률이 다르면 답변 중 오답률로 방법 간 우열을 판정하지 않는다 (명세).')
    lines += ['', '결과는 생성기와 설정에 대한 것이며, 명세에 없는 설정 변경의 근거로 쓰지 않는다.']
    os.makedirs(OUT_SCORE, exist_ok=True)
    with open(_pkg_open_path(_pkg_resolve(_pkg_resolve(os.path.join(OUT_SCORE, 'llm_report.txt'))), 'w'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(_pkg_open_path(_pkg_resolve(_pkg_resolve(os.path.join(OUT_SCORE, 'llm_statistics.json'))), 'w'), 'w', encoding='utf-8') as f:
        json.dump({'created_at': now(), 'n': n, 'h3a_primary': primary, 'invalid_candidates': {'QI': invalid_qi, 'VAN': invalid_van}, 'H2': {'diff': d_h2, 'ci': [lo_h2, hi_h2]}, 'AURC': {'qi': aq, 'van': av, 'diff': aq - av, 'ci': [float(alo), float(ahi)]}}, f, ensure_ascii=False, indent=2)
    print('\n'.join(lines))
    print(f'\n저장: {OUT_SCORE}')

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for flag in ['freeze_prompts', 'run', 'status', 'validate', 'tokens', 'score']:
        g.add_argument(f"--{flag.replace('_', '-')}", action='store_true')
    args = ap.parse_args()
    if args.freeze_prompts:
        cmd_freeze_prompts()
    elif args.run:
        cmd_run()
    elif args.status:
        cmd_status()
    elif args.validate:
        cmd_validate()
    elif args.tokens:
        cmd_tokens()
    else:
        cmd_score()
if __name__ == '__main__':
    main()
