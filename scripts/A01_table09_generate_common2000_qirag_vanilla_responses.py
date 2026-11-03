"""
01_run_generate_v2.py  (qi-rag-test)
====================================
QI-RAG 와 vanilla RAG 를 '문단 예산 P' 를 맞춰 비교하는 검색·생성 스크립트.

조건
  REPAQ_TOP1     RePAQ 공식 결과의 top-1 답 (생성 없음, 참조용)
  E5_TOP1        E5 로 질문 인덱스 검색 후 top-1 답 (생성 없음). 검색기만 E5 로 맞춘 대조군
  QIRAG_P{P}     질의 -> 질문 인덱스 상위 항목 -> 매핑된 문단을 앞에서부터 P 개로 자름 -> 생성
  VANILLA_P{P}   질의 -> 문단 인덱스 상위 P 개 -> 생성
  RANDOM_P{P}    같은 문단 수 P 를 코퍼스에서 무작위로 뽑아 생성 (정답 문단 제외). 길이 통제용
  ORACLE         평가 질문의 정답 문단 전부 (예산 무관, 코퍼스에 있는 것만). 근거가 주어졌을 때의 상한
  QAGEN_K{K}     질의 -> 질문 인덱스 상위 K 개의 저장 답을 문맥으로 넘겨 생성 (페이로드 통제용)

ORACLE 행의 gold_full 은 정답 문단이 전부 코퍼스에 있는지 여부다.
  True 인 문항만 모은 부분집합에서 조건을 비교하면 상한과 비교가 공정해진다 (02 에서 사용).

생성 조건은 모두 같은 프롬프트(strict)와 같은 생성기를 쓴다.

입력 (DATA_DIR 기준)
  _split/index_items.jsonl     qid, question, answer, context, gold_titles
  _split/eval_queries.jsonl    qid, original, keyword, noisy, paraphrase_llm, answer, gold_titles, type, level
  _result/result_repaq_{q,p,n,pl}.jsonl   (있을 때만)
  hotpotqa/hotpot_train.jsonl             (CORPUS_SOURCE = "hotpot" 일 때만)

출력 (OUT_ROOT/<모드>/)
  <조건>_<변형>.jsonl   문항별 원시 출력 (02_score_report_v2.py 가 읽는다)
  run_log.txt           실행 로그 (누적)

실행 (qi-rag-test 폴더에서)
  python 01_run_generate_v2.py
  메뉴: 1) 샘플 (먼저 실행해 점검)  2) 전체  3) 종료
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
DATA_DIR = 'data'
OUT_ROOT = 'out'
SEED = 0
N_SAMPLE = 200
SAMPLE_VARIANTS = ['original']
FULL_VARIANTS = ['original', 'keyword', 'noisy', 'paraphrase_llm']
BUDGETS = [2, 4]
TOP_ITEMS = 8
RUN_VANILLA = True
RUN_QAGEN = True
RUN_E5_TOP1 = True
RUN_RANDOM = True
RUN_ORACLE = True
RANDOM_MAX_TRIES = 100
QAGEN_K = 2
CORPUS_SOURCE = 'index'
MIN_PARA_CHARS = 20
ENCODER_MODEL = 'intfloat/e5-base-v2'
ENCODER_BATCH = 256
SEARCH_BATCH = 256
CACHE_DIR = _pkg_resolve(os.path.join(OUT_ROOT, 'cache'))
VLLM_URL = 'http://172.25.121.198:7777/v1/chat/completions'
VLLM_MODEL = 'Qwen2.5-7B-Instruct-AWQ'
GEN_MAX_TOKENS = 128
GEN_TEMPERATURE = 0.0
GEN_TIMEOUT = 60
GEN_WORKERS = 8
GEN_RETRIES = 2
REPAQ_FILES = {'original': 'result_repaq_q.jsonl', 'keyword': 'result_repaq_p.jsonl', 'noisy': 'result_repaq_n.jsonl', 'paraphrase_llm': 'result_repaq_pl.jsonl'}
QA_PROMPT = 'Answer ONLY using the given context.\nAnswer with the shortest possible phrase.\nIf the answer is not in the context, say "I don\'t know".\n\nContext:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:'
NO_GEN = {'REPAQ_TOP1', 'E5_TOP1'}
ABSTAIN_PHRASES = ("I DON'T KNOW", 'I DONT KNOW', 'I DO NOT KNOW', 'INSUFFICIENT')
PARA_RX = re.compile('\\[(.*?)\\]\\s*(.*?)(?=\\s*\\[[^\\]]+\\]\\s|$)', re.S)

def load_jsonl(path):
    out = []
    with open(_pkg_open_path(_pkg_resolve(path), 'r'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def pick_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'

def as_list(a):
    if a is None:
        return []
    if isinstance(a, list):
        return [str(x) for x in a]
    return [str(a)]

def first_answer(a):
    """저장 답이 리스트여도 첫 문자열만 남긴다. 채점에서 ['x'] 형식 때문에 틀리는 일을 막는다."""
    lst = as_list(a)
    return lst[0] if lst else ''

def is_abstain(t):
    if not t or not str(t).strip():
        return True
    up = str(t).upper()
    return any((p in up for p in ABSTAIN_PHRASES))

def split_paragraphs(ctx):
    """'[제목] 본문 [제목] 본문' 형식을 문단 목록으로 나눈다."""
    ctx = (ctx or '').strip()
    out = []
    for (title, body) in PARA_RX.findall(ctx):
        (title, body) = (title.strip(), body.strip())
        if len(body) >= MIN_PARA_CHARS:
            out.append({'title': title, 'text': body})
    if not out and len(ctx) >= MIN_PARA_CHARS:
        out.append({'title': '', 'text': ctx})
    return out

def fmt_para(p):
    return f"[{p['title']}] {p['text']}" if p['title'] else p['text']

def para_key(p):
    return p['title'] or p['text'][:200]

def build_corpus_index(pars):
    (docs, seen) = ([], set())
    for plist in pars:
        for p in plist:
            k = para_key(p)
            if k not in seen:
                seen.add(k)
                docs.append(p)
    return docs

def build_corpus_hotpot(qids):
    want = set(qids)
    (docs, seen) = ([], set())
    path = _pkg_resolve(os.path.join(DATA_DIR, 'hotpotqa', 'hotpot_train.jsonl'))
    with open(_pkg_open_path(_pkg_resolve(path), 'r'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r.get('id') or r.get('_id'))
            if rid not in want:
                continue
            ctx = r['context']
            if isinstance(ctx, dict):
                pairs = list(zip(ctx['title'], ctx['sentences']))
            else:
                pairs = [(c[0], c[1]) for c in ctx]
            for (title, sents) in pairs:
                title = str(title).strip()
                body = ''.join(sents).strip()
                if len(body) < MIN_PARA_CHARS or title in seen:
                    continue
                seen.add(title)
                docs.append({'title': title, 'text': body})
    return docs

def enc(model, texts, show=False):
    return model.encode(texts, batch_size=ENCODER_BATCH, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=show).astype('float32')

def cached_encode(model, texts, name, log):
    h = hashlib.sha1(ENCODER_MODEL.encode('utf-8'))
    for t in texts:
        h.update(t.encode('utf-8'))
        h.update(b'\x00')
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _pkg_resolve(os.path.join(CACHE_DIR, f'{name}_{len(texts)}_{h.hexdigest()[:16]}.npy'))
    if os.path.exists(path):
        log(f'  임베딩 캐시 사용: {os.path.basename(path)}')
        return np.load(path)
    t0 = time.time()
    E = enc(model, texts, show=True)
    np.save(path, E)
    log(f'  임베딩 {len(texts):,}건 {_pkg_resolve((time.time() - t0) / 60):.1f}분 -> {os.path.basename(path)}')
    return E

def topk(E, Q, k):
    (idx_all, sc_all) = ([], [])
    for s in range(0, len(Q), SEARCH_BATCH):
        S = Q[s:s + SEARCH_BATCH] @ E.T
        kk = min(k, S.shape[1])
        part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        rows = np.arange(S.shape[0])[:, None]
        order = np.argsort(-S[rows, part], axis=1)
        idx = part[rows, order]
        idx_all.append(idx)
        sc_all.append(S[rows, idx])
    return (np.vstack(idx_all), np.vstack(sc_all))

def call_llm(prompt):
    body = {'model': VLLM_MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': GEN_TEMPERATURE, 'max_tokens': GEN_MAX_TOKENS}
    req = urllib.request.Request(VLLM_URL, data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT) as r:
        d = json.loads(r.read())
    return d['choices'][0]['message']['content'].strip()

def gen_one(prompt):
    """성공 시 (텍스트, 'ok'). 모든 재시도가 실패하면 ('', 'error: ...').
    오류를 빈 답으로 바꾸지 않는다. 빈 답은 유보로 세므로 오류와 구분해야 한다."""
    last = ''
    for attempt in range(GEN_RETRIES + 1):
        try:
            return (call_llm(prompt), 'ok')
        except Exception as e:
            last = f'{type(e).__name__}: {e}'
            time.sleep(1.5 * (attempt + 1))
    return ('', 'error: ' + last[:200])

def generate_all(prompts):
    with ThreadPoolExecutor(max_workers=GEN_WORKERS) as ex:
        return list(ex.map(gen_one, prompts))

def make_row(e, v, cond, ctx, titles, n_paras, src_qids):
    gold_t = [str(t) for t in e.get('gold_titles', [])]
    hit = bool(gold_t) and all((t in titles for t in gold_t))
    return {'qid': str(e['qid']), 'cond': cond, 'variant': v, 'query': e[v], 'gold': as_list(e.get('answer')), 'gold_titles': gold_t, 'type': e.get('type'), 'level': e.get('level'), 'ctx': ctx, 'retrieved_titles': titles, 'n_paras': n_paras, 'src_qids': src_qids, 'words': len(ctx.split()), 'hit_all': hit, 'pred': '', 'status': 'pending', 'err': ''}

def qi_context(pars, item_ids, P):
    """질문 인덱스 상위 항목에서 매핑된 문단을 앞에서부터 모아 P 개에서 자른다."""
    (paras, seen, src) = ([], set(), [])
    for j in item_ids:
        j = int(j)
        used = False
        for p in pars[j]:
            if len(paras) >= P:
                break
            k = para_key(p)
            if k in seen:
                continue
            seen.add(k)
            paras.append(p)
            used = True
        if used:
            src.append(j)
        if len(paras) >= P:
            break
    return (paras, src)

def random_context(docs, gold_titles, P, qid, v):
    """정답 문단을 제외하고 코퍼스에서 P 개를 무작위로 뽑는다.
    시드를 질문·변형·예산별로 고정해서 다시 실행해도 같은 문맥이 나온다."""
    rng = random.Random(f'{SEED}|{qid}|{v}|{P}')
    gold = set(gold_titles)
    (sel, seen_idx, tries) = ([], set(), 0)
    while len(sel) < P and tries < RANDOM_MAX_TRIES * P:
        tries += 1
        k = rng.randrange(len(docs))
        d = docs[k]
        if k in seen_idx or d['title'] in gold:
            continue
        seen_idx.add(k)
        sel.append(d)
    return sel

def load_repaq(v):
    fp = _pkg_resolve(os.path.join(DATA_DIR, '_result', REPAQ_FILES[v]))
    if not os.path.exists(fp):
        return ({}, 0, False)
    (m, dup) = ({}, 0)
    for r in load_jsonl(fp):
        q = str((r.get('input_qa') or {}).get('question', '')).strip().lower()
        ret = r.get('retrieved_qas') or []
        ans = ret[0].get('answer', '') if ret else ''
        if isinstance(ans, list):
            ans = ans[0] if ans else ''
        if q in m:
            dup += 1
            continue
        m[q] = str(ans)
    return (m, dup, True)

def stat_block(rows):
    ok = [r for r in rows if r['status'] == 'ok']
    ab = [is_abstain(r['pred']) for r in ok]
    n = len(rows)
    return {'n': n, 'err': n - len(ok), 'abstain': float(np.mean(ab)) if ab else float('nan'), 'words': float(np.mean([r['words'] for r in rows])) if n else float('nan'), 'paras': float(np.mean([r['n_paras'] for r in rows])) if n else float('nan'), 'hit': float(np.mean([r['hit_all'] for r in rows])) if n else float('nan')}

def print_stats(summ, log):
    log(f"  {'조건':<16}{'n':>6}{'오류':>6}{'유보율':>8}{'단어':>8}{'문단':>6}{'검색적중':>9}")
    for (cond, s) in summ.items():
        log(f"  {cond:<16}{s['n']:>6}{s['err']:>6}{s['abstain']:>8.3f}{s['words']:>8.1f}{s['paras']:>6.2f}{s['hit']:>9.3f}")

def sample_checks(conds, log):
    log('\n  [샘플 점검]')
    for P in BUDGETS:
        for name in (f'QIRAG_P{P}', f'VANILLA_P{P}', f'RANDOM_P{P}'):
            if name in conds:
                rows = conds[name]
                share = float(np.mean([r['n_paras'] == P for r in rows]))
                tag = '통과' if share >= 0.95 else '주의 (예산이 P 에 못 미침)'
                log(f'    {name}: 문단 수가 P={P} 인 비율 {share:.3f} -> {tag}')
    if 'ORACLE' in conds:
        rows = conds['ORACLE']
        full = float(np.mean([bool(r.get('gold_full')) for r in rows]))
        log(f'    ORACLE: 정답 문단 전부 코퍼스에 있는 문항 비율 {full:.3f} (02 에서 이 부분집합으로 따로 판정한다)')
    if 'QIRAG_P2' in conds and 'VANILLA_P2' in conds:
        w_q = float(np.mean([r['words'] for r in conds['QIRAG_P2']]))
        w_v = float(np.mean([r['words'] for r in conds['VANILLA_P2']]))
        ratio = _pkg_resolve(w_q / w_v) if w_v else float('nan')
        tag = '통과' if 0.8 <= ratio <= 1.25 else '주의 (문맥 길이가 다름)'
        log(f'    P=2 단어 수 비 QI/vanilla = {ratio:.2f} -> {tag}')
    if 'RANDOM_P2' in conds and 'QIRAG_P2' in conds:
        w_r = float(np.mean([r['words'] for r in conds['RANDOM_P2']]))
        w_q = float(np.mean([r['words'] for r in conds['QIRAG_P2']]))
        ratio = _pkg_resolve(w_r / w_q) if w_q else float('nan')
        log(f'    P=2 단어 수 비 RANDOM/QI = {ratio:.2f} (참고: 1 에 가까울수록 대조가 깔끔)')
    for (name, rows) in conds.items():
        err = sum((1 for r in rows if r['status'] != 'ok'))
        if err > 0.05 * max(len(rows), 1):
            log(f'    {name}: 오류 {err}건 (5% 초과) -> 서버 상태를 확인하세요')
    first = next(iter(conds.values()))
    for i in range(min(2, len(first))):
        log(f"    예) 질의: {first[i]['query']}   정답: {first[i]['gold']}")
        for (name, rows) in conds.items():
            r = rows[i]
            log(f"       {name:<16} 제목 {r['retrieved_titles']}  답: {str(r['pred'])[:80]}")

def _run(mode, out_dir, log):
    from sentence_transformers import SentenceTransformer
    variants = SAMPLE_VARIANTS if mode == 'sample' else FULL_VARIANTS
    index_path = _pkg_resolve(os.path.join(DATA_DIR, '_split', 'index_items.jsonl'))
    eval_path = _pkg_resolve(os.path.join(DATA_DIR, '_split', 'eval_queries.jsonl'))
    for p in (index_path, eval_path):
        if not os.path.exists(p):
            log(f'[실패] 파일 없음: {p}')
            return
    try:
        call_llm('Reply with exactly: OK')
    except Exception as e:
        log(f'[실패] 생성 서버 접속 불가: {type(e).__name__}: {e}')
        return
    items = load_jsonl(index_path)
    evals_all = load_jsonl(eval_path)
    if mode == 'sample':
        rng = random.Random(SEED)
        pick = sorted(rng.sample(range(len(evals_all)), min(N_SAMPLE, len(evals_all))))
        evals = [evals_all[i] for i in pick]
    else:
        evals = evals_all
    log(f'\n==== {mode} 실행 ====')
    log(f'인덱스 {len(items):,} / 평가 {len(evals):,} / 변형 {variants}')
    log(f'예산 P={BUDGETS}  TOP_ITEMS={TOP_ITEMS}  코퍼스={CORPUS_SOURCE}  QAGEN_K={QAGEN_K}')
    log(f'대조군: E5_TOP1={RUN_E5_TOP1}  RANDOM={RUN_RANDOM}  ORACLE={RUN_ORACLE}  VANILLA={RUN_VANILLA}  QAGEN={RUN_QAGEN}')
    pars = [split_paragraphs(r.get('context', '')) for r in items]
    n_zero = sum((1 for p in pars if not p))
    mean_p = float(np.mean([len(p) for p in pars])) if pars else 0.0
    log(f'[파싱] 문단 0개 항목 {n_zero}/{len(items)}, 항목당 평균 문단 {mean_p:.2f}')
    if n_zero > 0.01 * max(len(items), 1):
        log('  주의: 문단 파싱 실패가 많습니다. 문맥 형식을 확인하세요.')
    device = pick_device()
    log(f'[인코더] {ENCODER_MODEL} ({device})')
    model = SentenceTransformer(ENCODER_MODEL, device=device)
    Eq = cached_encode(model, ['query: ' + r['question'] for r in items], 'qidx', log)
    (docs, Ed, title_map) = (None, None, None)
    if RUN_VANILLA or RUN_RANDOM or RUN_ORACLE:
        if CORPUS_SOURCE == 'hotpot':
            docs = build_corpus_hotpot([str(r['qid']) for r in items])
        else:
            docs = build_corpus_index(pars)
        log(f'[코퍼스] {CORPUS_SOURCE} 고유 문단 {len(docs):,}')
        title_map = {}
        for d in docs:
            if d['title'] and d['title'] not in title_map:
                title_map[d['title']] = d
    if RUN_VANILLA:
        texts = ['passage: ' + (f"{d['title']} {d['text']}" if d['title'] else d['text']) for d in docs]
        Ed = cached_encode(model, texts, f'docs_{CORPUS_SOURCE}', log)
    summary = {}
    for v in variants:
        if v not in evals[0]:
            log(f'[건너뜀] {v} 필드 없음')
            continue
        log(f'\n[{v}] 검색')
        Q = enc(model, ['query: ' + e[v] for e in evals])
        (qi_idx, _) = topk(Eq, Q, TOP_ITEMS)
        conds = {}
        (rmap, dup, found) = load_repaq(v)
        if found:
            rows = []
            for e in evals:
                key = str(e[v]).strip().lower()
                row = make_row(e, v, 'REPAQ_TOP1', '', [], 0, [])
                if key in rmap:
                    row['pred'] = rmap[key]
                    row['status'] = 'ok'
                else:
                    row['status'] = 'missing'
                rows.append(row)
            conds['REPAQ_TOP1'] = rows
            matched = sum((1 for r in rows if r['status'] == 'ok'))
            log(f'  RePAQ 매칭 {matched}/{len(evals)} (중복 질문 키 {dup}건 제외)')
        else:
            log('  RePAQ 결과 없음 -> REPAQ_TOP1 건너뜀')
        if RUN_E5_TOP1:
            rows = []
            for (i, e) in enumerate(evals):
                j = int(qi_idx[i][0])
                titles = [str(t) for t in items[j].get('gold_titles', [])]
                row = make_row(e, v, 'E5_TOP1', '', titles, 0, [str(items[j]['qid'])])
                row['pred'] = first_answer(items[j]['answer'])
                row['status'] = 'ok'
                rows.append(row)
            conds['E5_TOP1'] = rows
        for P in BUDGETS:
            rows = []
            for (i, e) in enumerate(evals):
                (paras, src) = qi_context(pars, qi_idx[i], P)
                ctx = '\n\n'.join((fmt_para(p) for p in paras))
                titles = [p['title'] for p in paras]
                src_qids = [str(items[j]['qid']) for j in src]
                rows.append(make_row(e, v, f'QIRAG_P{P}', ctx, titles, len(paras), src_qids))
            conds[f'QIRAG_P{P}'] = rows
        if RUN_VANILLA:
            (v_idx, _) = topk(Ed, Q, max(BUDGETS))
            for P in BUDGETS:
                rows = []
                for (i, e) in enumerate(evals):
                    sel = [docs[int(j)] for j in v_idx[i][:P]]
                    ctx = '\n\n'.join((fmt_para(p) for p in sel))
                    titles = [p['title'] for p in sel]
                    rows.append(make_row(e, v, f'VANILLA_P{P}', ctx, titles, len(sel), []))
                conds[f'VANILLA_P{P}'] = rows
        if RUN_RANDOM and docs:
            for P in BUDGETS:
                rows = []
                for e in evals:
                    gold_t = [str(t) for t in e.get('gold_titles', [])]
                    sel = random_context(docs, gold_t, P, e['qid'], v)
                    ctx = '\n\n'.join((fmt_para(p) for p in sel))
                    titles = [p['title'] for p in sel]
                    rows.append(make_row(e, v, f'RANDOM_P{P}', ctx, titles, len(sel), []))
                conds[f'RANDOM_P{P}'] = rows
        if RUN_ORACLE and title_map:
            rows = []
            for e in evals:
                gold_t = [str(t) for t in e.get('gold_titles', [])]
                sel = [title_map[t] for t in gold_t if t in title_map]
                ctx = '\n\n'.join((fmt_para(p) for p in sel))
                titles = [p['title'] for p in sel]
                row = make_row(e, v, 'ORACLE', ctx, titles, len(sel), [])
                row['gold_full'] = bool(gold_t) and len(sel) == len(gold_t)
                rows.append(row)
            conds['ORACLE'] = rows
        if RUN_QAGEN:
            rows = []
            for (i, e) in enumerate(evals):
                sel = [int(j) for j in qi_idx[i][:QAGEN_K]]
                ctx = '\n'.join((f"- {first_answer(items[j]['answer'])}" for j in sel))
                titles = [t for j in sel for t in items[j].get('gold_titles', [])]
                src_qids = [str(items[j]['qid']) for j in sel]
                rows.append(make_row(e, v, f'QAGEN_K{QAGEN_K}', ctx, titles, len(sel), src_qids))
            conds[f'QAGEN_K{QAGEN_K}'] = rows
        for (cond, rows) in conds.items():
            if cond in NO_GEN:
                continue
            log(f'  생성 {cond} ({len(rows)}건)')
            res = generate_all([QA_PROMPT.format(ctx=r['ctx'], q=r['query']) for r in rows])
            for (r, (txt, st)) in zip(rows, res):
                r['pred'] = txt
                r['status'] = 'ok' if st == 'ok' else 'error'
                r['err'] = '' if st == 'ok' else st
            log(f"    오류 {sum((1 for r in rows if r['status'] == 'error'))}/{len(rows)}")
        for (cond, rows) in conds.items():
            fp = _pkg_resolve(os.path.join(out_dir, f'{cond}_{v}.jsonl'))
            with open(_pkg_open_path(_pkg_resolve(fp), 'w'), 'w', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            log(f'  저장 {fp}')
        summary[v] = {c: stat_block(rows) for (c, rows) in conds.items()}
        print_stats(summary[v], log)
        if mode == 'sample':
            sample_checks(conds, log)
    log('\n완료. 다음: python 02_score_report_v2.py')

def run(mode):
    out_dir = _pkg_resolve(os.path.join(OUT_ROOT, mode))
    os.makedirs(out_dir, exist_ok=True)
    logf = open(_pkg_open_path(_pkg_resolve(_pkg_resolve(os.path.join(out_dir, 'run_log.txt'))), 'a'), 'a', encoding='utf-8')

    def log(*a):
        msg = ' '.join((str(x) for x in a))
        print(msg)
        logf.write(msg + '\n')
        logf.flush()
    t0 = time.time()
    try:
        _run(mode, out_dir, log)
    finally:
        log(f'소요 {_pkg_resolve((time.time() - t0) / 60):.1f}분')
        logf.close()

def main():
    print('qi-rag-test 검색·생성 실행 (v2)')
    while True:
        print('\n  1) 샘플 실행 (N_SAMPLE 문항, 먼저 실행해 점검)')
        print('  2) 전체 실행')
        print('  3) 종료')
        c = input('  선택: ').strip()
        if c == '1':
            run('sample')
        elif c == '2':
            ok = input('  전체 실행은 오래 걸립니다. 계속할까요? (y/n): ').strip().lower()
            if ok == 'y':
                run('full')
        elif c == '3':
            break
if __name__ == '__main__':
    main()
