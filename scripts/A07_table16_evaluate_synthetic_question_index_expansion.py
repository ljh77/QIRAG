"""E1-B2 post-hoc exploratory QI question-index expansion.

Run in ~/2610-qi-rag with the existing E1-B v2 outputs:
  python e1b2_qi_expansion.py prepare --dry-run
  python e1b2_qi_expansion.py prepare            # 기본: 198, 170 두 서버에 번갈아 요청
  python e1b2_qi_expansion.py prepare --urls http://172.25.121.198:7777/v1/chat/completions   # 한 대만
  python e1b2_qi_expansion.py retrieve
  python e1b2_qi_expansion.py score
  python e1b2_qi_expansion.py verify

No frozen inputs are edited. Output: out/e1b2_qi/.
Exactly the same 50k ordered extra documents as E1-B v2, nested first 25k.
Generate one question per document, match original E1-A query-index policy:
TOP_ITEMS=8, P=2, first-ranked indexed question's linked documents before next.
Each added synthetic question links to its ONE source document. Baseline QI
queries link to original 1:N docs. Thus mapping policy is unchanged, but
question density/linked-document count changes (must acknowledge).

This is a post-hoc exploratory analysis, NOT replacement of frozen H1.
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT = _PACKAGE_ROOT
OUT = _pkg_resolve(ROOT / 'out/e1b2_qi')
EXTRA = _pkg_resolve(ROOT / 'out/e1b/extra_docs.jsonl')
BUILD = _pkg_resolve(ROOT / 'out/e1b/build_meta.json')
E1B = _pkg_resolve(ROOT / 'out/holdout_e1b/e1b_retrieval.jsonl')
E1B_META = _pkg_resolve(ROOT / 'out/holdout_e1b/e1b_meta.json')
BASEPRED = _pkg_resolve(ROOT / 'out/holdout_locked/retrieval_predictions.jsonl')
BASEMAN = _pkg_resolve(ROOT / 'out/holdout_locked/retrieval_manifest.json')
SPLIT = _pkg_resolve(ROOT / 'out/holdout_locked/split_ids.json')
FREEZE = _pkg_resolve(ROOT / 'out/holdout_locked/FREEZE.json')
INDEX = _pkg_resolve(ROOT / 'data/_split/index_items.jsonl')
DEV = _pkg_resolve(ROOT / 'data/hotpotqa/hotpot_dev_distractor.jsonl')
N = 50000
N25 = 25000
P = 2
K = 8
QGEN = 'Here is a passage.\n\n{ctx}\n\nWrite exactly ONE self-contained English question answerable from this passage. Use a single interrogative sentence ending with a question mark. Do not add explanations or answers.'

def sha(p):
    h = hashlib.sha256()
    with open(_pkg_open_path(_pkg_resolve(p), 'rb'), 'rb') as f:
        for chunk in iter(lambda : f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def readl(p):
    with open(_pkg_open_path(_pkg_resolve(p), 'r'), encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def atomic_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    _pkg_output(temp).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, p)

def atomic_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    with open(_pkg_open_path(_pkg_resolve(temp), 'w'), 'w', encoding='utf-8') as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    os.replace(temp, p)

def need(test, message):
    if not test:
        raise RuntimeError(message)

def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')

def load_inputs():
    for p in (EXTRA, BUILD, E1B, E1B_META, BASEPRED, BASEMAN, SPLIT, FREEZE, INDEX, DEV):
        need(p.is_file(), f'필수 파일 없음: {p}')
    bm = json.loads(BUILD.read_text(encoding='utf-8'))
    need(sha(EXTRA) == bm['extra_docs_sha256'], 'E1-B 문서 해시 불일치')
    need(int(bm['n_extra']) == N and int(bm['n_b1']) == N25 and bm['nested'], 'E1-B 확장 조건 불일치')
    need(sha(SPLIT) == json.loads(FREEZE.read_text(encoding='utf-8'))['split_ids_sha256'], 'FREEZE 분할 불일치')
    mainman = json.loads(BASEMAN.read_text(encoding='utf-8'))
    need(sha(BASEPRED) == mainman['predictions_sha256'], '기존 H1 예측 파일 해시 불일치')
    e1bm = json.loads(E1B_META.read_text(encoding='utf-8'))
    need(sha(E1B) == e1bm['pred_sha256'], 'E1-B 예측 파일 해시 불일치')
    extra = list(readl(EXTRA))
    need(len(extra) == N, f'추가 문서 수 불일치: {len(extra)}')
    titles = [d['title'] for d in extra]
    need(len(set(titles)) == len(titles), '추가 문서 제목 중복')
    want = set(map(str, json.loads(SPLIT.read_text(encoding='utf-8'))['holdout']))
    need(len(want) == 4405, 'Holdout 수가 4,405가 아님')
    main = {str(r['qid']): r for r in readl(BASEPRED)}
    vb = {str(r['qid']): r for r in readl(E1B)}
    need(set(main) == set(vb) == want, 'Holdout / H1 / E1-B 문항 집합 불일치')
    need(all((set(main[q]['va_titles']) == set(vb[q]['base_titles']) for q in want)), 'E1-B Vanilla 기준선 재현 실패')
    return (extra, main, vb, want)

def conf(args, extra):
    import importlib.util
    path = _pkg_resolve(ROOT / '01_run_generate_v2.py')
    need(path.is_file(), f'기준 구현 없음: {path}')
    spec = importlib.util.spec_from_file_location('e1b2_reference', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    need(mod.CORPUS_SOURCE == 'index', '기준 CORPUS_SOURCE 설정 불일치')
    from S00_qi_common import ENCODER
    return {'source_files': {str(p.relative_to(ROOT)): sha(p) for p in (EXTRA, BUILD, E1B, E1B_META, BASEPRED, BASEMAN, SPLIT, FREEZE, INDEX, DEV, path)}, 'n_extra': len(extra), 'n25': N25, 'qgen_sha256': hashlib.sha256(QGEN.encode()).hexdigest(), 'llm_model': mod.VLLM_MODEL, 'encoder': ENCODER, 'P': P, 'top_items': K, 'workers_setting': 'execution-only', 'generation_temperature': 0.0}

def fmt(d):
    return f"[{d['title']}] {d['text']}"

def parse_question(t):
    for line in str(t or '').splitlines():
        line = re.sub('^\\s*(?:[-•*]|\\d+[.)])\\s*', '', line).strip()
        line = line.replace('？', '?')
        if line.endswith('?') and len(line) >= 12 and (not re.search('[\\u3400-\\u9fff]', line)):
            return line
    return ''

def call_llm(url, model, prompt, timeout, max_tokens=96):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.0, 'max_tokens': max_tokens}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        d = json.loads(f.read())
    return str(d['choices'][0]['message'].get('content') or '').strip()
PROBES = ['[Deckers Creek] Deckers Creek is a tributary of the Monongahela River in West Virginia. It begins west of Arthurdale in Preston County and flows to Morgantown.', '[Anna Sewell] Anna Sewell was an English novelist, best known as the author of the 1877 novel Black Beauty.', '[Lake Taupo] Lake Taupo is a lake in the North Island of New Zealand. It is the largest lake by surface area in the country, with a surface area of 616 square kilometres.', '[Mazda MX-5] The Mazda MX-5 is a lightweight two-passenger roadster manufactured and marketed by Mazda with a front mid-engine, rear-wheel-drive layout. It was introduced in 1989.', '[Cheddar cheese] Cheddar cheese is a relatively hard, off-white, sometimes sharp-tasting natural cheese. It originates from the English village of Cheddar in Somerset.']

def preflight(urls, model, timeout, out_dir):
    """모든 서버가 같은 모델 ID 를 서빙하고, 5개 프롬프트에 같은 답을 내는지 점검. 메타를 servers.json 에 기록.
    주의: vLLM API 는 가중치 해시를 노출하지 않는다. 모델 파일 해시(config.json/safetensors sha)는
    각 서버 호스트에서 따로 구해 servers.json 의 host_attestation 에 적어야 한다."""
    (records, fingerprints) = ([], [])
    for u in urls:
        base = u.split('/v1/')[0]
        with urllib.request.urlopen(base + '/v1/models', timeout=timeout) as f:
            data = json.loads(f.read())['data']
        meta = next((m for m in data if m.get('id') == model), None)
        need(meta is not None, f"서버 {u} 가 모델 {model} 을 서빙하지 않음: {[m.get('id') for m in data]}")
        answers = [call_llm(u, model, QGEN.format(ctx=p), timeout) for p in PROBES]
        fp = hashlib.sha256(json.dumps(answers).encode()).hexdigest()
        fingerprints.append(fp)
        records.append({'url': u, 'model_id': meta.get('id'), 'root': meta.get('root'), 'max_model_len': meta.get('max_model_len'), 'created': meta.get('created'), 'probe_fingerprint_sha256': fp, 'probe_answers': answers, 'host_attestation': {'config_json_sha1': None, 'weights_sha256': None, 'vllm_version': None, 'note': '서버 호스트에서 채워 넣을 것'}})
    need(len(set(fingerprints)) == 1, '서버 간 5개 프롬프트 응답 지문 불일치: ' + '; '.join((f"{r['url']} {r['probe_fingerprint_sha256'][:12]}" for r in records)))
    need(len(set((r['root'] for r in records))) == 1, f"서버 간 모델 root 경로 불일치: {[r['root'] for r in records]}")
    need(len(set((r['max_model_len'] for r in records))) == 1, '서버 간 max_model_len 불일치')
    sp = _pkg_resolve(out_dir / 'servers.json')
    if sp.exists():
        old = json.loads(sp.read_text(encoding='utf-8'))
        need(old['probe_fingerprint_sha256'] == fingerprints[0], '이전 실행과 서버 응답 지문이 다름 (설정 변경 의심)')
    else:
        atomic_json(sp, {'checked_at': now(), 'model': model, 'probe_prompts_sha256': hashlib.sha256(json.dumps(PROBES).encode()).hexdigest(), 'probe_fingerprint_sha256': fingerprints[0], 'servers': records, 'limitation': 'API 로는 가중치 동일성을 증명할 수 없음. host_attestation 을 각 서버에서 채울 것'})
    print(f'[PASS] 서버 {len(urls)}대 모델 ID·root·max_model_len·5개 응답 지문 일치 ({fingerprints[0][:12]})')

def gen(args, model, d, i):
    """문서 i 는 urls[i % n] 서버에 먼저 보내고, 재시도까지 실패하면 다음 서버로 넘어간다."""
    prompt = QGEN.format(ctx=fmt(d))
    n = len(args.urls)
    reason = ''
    for hop in range(n):
        url = args.urls[(i + hop) % n]
        for attempt in range(args.retries + 1):
            try:
                raw = call_llm(url, model, prompt, args.timeout)
                q = parse_question(raw)
                if q:
                    return {'question': q, 'status': 'ok', 'server': url, 'failover': hop}
                reason = 'invalid/empty generated question: ' + repr(raw[:180])
            except Exception as exc:
                reason = f'{type(exc).__name__}: {exc}'
            time.sleep(0.25 * (attempt + 1))
    return {'status': 'error', 'reason': reason[:220]}

def prepare(args):
    (extra, _, _, _) = load_inputs()
    config = conf(args, extra)
    path = _pkg_resolve(OUT / 'sources.json')
    if args.dry_run:
        print('[DRY RUN] 문서', len(extra), '전반 25k', len(extra[:N25]), '후반 25k', len(extra[N25:]))
        if path.exists():
            need(json.loads(path.read_text(encoding='utf-8')) == config, '저널과 입력 설정 불일치')
        return
    if path.exists():
        need(json.loads(path.read_text(encoding='utf-8')) == config, '이미 사용된 입력과 불일치. 별도 --out 사용')
    else:
        atomic_json(path, config)
    journal = _pkg_resolve(OUT / 'question_journal.jsonl')
    done = {}
    if journal.exists():
        for r in readl(journal):
            if r.get('status') == 'ok':
                i = int(r['i'])
                need(0 <= i < N and r['title'] == extra[i]['title'], '저널 인덱스/제목 불일치')
                need(i not in done or done[i] == r['question'], '저널 내 질문 충돌')
                done[i] = r['question']
    pending = [i for i in range(N) if i not in done]
    print(f'추가 문서={N}, 이전 생성 성공={len(done)}, 남은 생성={len(pending)}')
    if _pkg_resolve(OUT / 'questions.jsonl').exists():
        need(not pending, '저장된 질문 인덱스가 불완전함')
        need(sha(_pkg_resolve(OUT / 'questions.jsonl')) == json.loads(_pkg_resolve(OUT / 'prepare_manifest.json').read_text())['questions_sha256'], '질문 인덱스 해시 불일치')
        print('[PASS] 준비 이미 완료')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    preflight(args.urls, config['llm_model'], args.timeout, OUT)
    import queue, threading
    q = queue.Queue()
    for i in pending:
        q.put((i, 0))
    lock = threading.Lock()
    per_server = {u: 0 for u in args.urls}
    stats = {'failover': 0, 'failed': []}
    model = config['llm_model']
    t0 = time.time()
    f = open(_pkg_open_path(_pkg_resolve(journal), 'a'), 'a', encoding='utf-8')

    def worker(url):
        while True:
            try:
                (i, tries) = q.get_nowait()
            except queue.Empty:
                return
            prompt = QGEN.format(ctx=fmt(extra[i]))
            (result, reason) = (None, '')
            for attempt in range(args.retries + 1):
                try:
                    qtext = parse_question(call_llm(url, model, prompt, args.timeout))
                    if qtext:
                        result = qtext
                        break
                    reason = 'invalid/empty generated question'
                except Exception as exc:
                    reason = f'{type(exc).__name__}: {exc}'[:160]
                time.sleep(0.25 * (attempt + 1))
            with lock:
                if result:
                    f.write(json.dumps({'i': i, 'title': extra[i]['title'], 'question': result, 'status': 'ok', 'server': url, 'failover': tries}, ensure_ascii=False) + '\n')
                    f.flush()
                    done[i] = result
                    per_server[url] += 1
                    stats['failover'] += tries > 0
                elif tries + 1 < len(args.urls):
                    q.put((i, tries + 1))
                else:
                    stats['failed'].append((i, extra[i]['title'], reason))
            q.task_done()
    threads = [threading.Thread(target=worker, args=(u,), daemon=True) for u in args.urls for _ in range(args.workers)]
    for t in threads:
        t.start()
    n0 = len(done)
    while any((t.is_alive() for t in threads)):
        time.sleep(15)
        with lock:
            cur = len(done)
            rate = _pkg_resolve((cur - n0) / max(time.time() - t0, 1e-06))
            eta = _pkg_resolve(_pkg_resolve((N - cur) / max(rate, 1e-06)) / 60)
            srv = ' / '.join((f"{u.split('//')[1].split(':')[0]}: {c}" for (u, c) in per_server.items()))
            print(f"{cur}/{N} 완료  [{srv}]  failover {stats['failover']}  실패 {len(stats['failed'])}  {rate:.1f}건/초  남은 약 {eta:.0f}분", flush=True)
    for t in threads:
        t.join()
    f.close()
    if stats['failed']:
        print(f"실패 {len(stats['failed'])}건 (예: {stats['failed'][:5]}). 저널 보존됨, 재실행하면 이어서 시도")
    if len(done) != N:
        missing = [i for i in range(N) if i not in done]
        raise RuntimeError(f"{len(missing)} 질문 생성 실패; 저널 보존 및 재시도 가능. 첫 사례: {[(i, extra[i]['title']) for i in missing[:12]]}")
    result = [{'i': i, 'title': extra[i]['title'], 'question': done[i]} for i in range(N)]
    atomic_jsonl(_pkg_resolve(OUT / 'questions.jsonl'), result)
    atomic_json(_pkg_resolve(OUT / 'prepare_manifest.json'), {'created_at': now(), 'sources_sha256': sha(path), 'questions_sha256': sha(_pkg_resolve(OUT / 'questions.jsonl')), 'n_questions': N, 'journal_sha256': sha(journal)})
    print('[PASS] 새 질문 50,000개 1:1 순서 일치')
STOP = set('a an the of in on at to for from by with and or is are was were be been this that these those it its what which who whom whose when where why how does do did has have had can could would should will into as than then there their there about'.split())
GENERIC = re.compile('\\b(this (passage|text|article|document|topic|context)|the (passage|text|article|document) (above|below|given)|mentioned (above|here)|according to the (passage|text))\\b', re.I)

def toks(s):
    return [w for w in re.findall('[a-z0-9]+', str(s).lower()) if w not in STOP]

def quality(args):
    """생성 질문 품질 통계 (생성 결과는 바꾸지 않음). 표본 200개를 눈가림 수동 점검용으로 저장."""
    (extra, _, _, _) = load_inputs()
    qp = _pkg_resolve(OUT / 'questions.jsonl')
    need(qp.exists(), 'questions.jsonl 없음 (prepare 먼저)')
    newq = list(readl(qp))
    need(len(newq) == N and all((newq[i]['i'] == i and newq[i]['title'] == extra[i]['title'] for i in range(N))), '질문-문서 1:1 순서 불일치')
    qs = [x['question'] for x in newq]
    words = np.array([len(q.split()) for q in qs])
    chars = np.array([len(q) for q in qs])
    norm = [' '.join(toks(q)) for q in qs]
    exact_dup = N - len(set(qs))
    norm_dup = N - len(set(norm))
    non_ascii = sum((1 for q in qs if any((ord(c) > 127 for c in q))))
    generic = sum((1 for q in qs if GENERIC.search(q)))
    (overlap, title_hit) = ([], 0)
    for (q, d) in zip(qs, extra):
        qt = set(toks(q))
        dt = set(toks(d['title'] + ' ' + d['text']))
        overlap.append(_pkg_resolve(len(qt & dt) / len(qt)) if qt else 0.0)
        title_hit += bool(set(toks(d['title'])) & qt)
    overlap = np.array(overlap)
    low_overlap = int((overlap < 0.3).sum())
    journal = _pkg_resolve(OUT / 'question_journal.jsonl')
    j_ok = j_conflict = j_bad = 0
    seen = {}
    if journal.exists():
        for r in readl(journal):
            if r.get('status') != 'ok':
                j_bad += 1
                continue
            j_ok += 1
            i = int(r['i'])
            if i in seen and seen[i] != r['question']:
                j_conflict += 1
            seen[i] = r['question']
    stats = {'n': N, 'words': {'mean': float(words.mean()), 'median': float(np.median(words)), 'min': int(words.min()), 'max': int(words.max())}, 'chars_mean': float(chars.mean()), 'exact_duplicates': exact_dup, 'normalized_duplicates': norm_dup, 'non_ascii_questions': non_ascii, 'generic_pattern_questions': generic, 'lexical_overlap_with_doc': {'mean': float(overlap.mean()), 'median': float(np.median(overlap)), 'below_0.3': low_overlap}, 'title_token_in_question': title_hit, 'journal': {'ok_rows': j_ok, 'conflicting_rows': j_conflict, 'non_ok_rows': j_bad}, 'questions_sha256': sha(qp), 'note': '어휘 중복은 연관성의 대리 지표. 의미 품질은 표본 수동 점검으로 확인'}
    rng = np.random.Generator(np.random.PCG64(0))
    pick = sorted((int(x) for x in rng.choice(N, 200, replace=False)))
    sample = [{'i': i, 'title': extra[i]['title'], 'text': extra[i]['text'][:700], 'question': qs[i], 'flags': {'generic': bool(GENERIC.search(qs[i])), 'low_overlap': bool(overlap[i] < 0.3), 'non_ascii': any((ord(c) > 127 for c in qs[i]))}, 'label_answerable': '', 'label_relevant': ''} for i in pick]
    for p in (_pkg_resolve(OUT / 'quality_stats.json'), _pkg_resolve(OUT / 'quality_sample.jsonl'), _pkg_resolve(OUT / 'quality_report.txt')):
        need(not p.exists(), f'기존 품질 파일 덮어쓰기 금지: {p.name}')
    atomic_json(_pkg_resolve(OUT / 'quality_stats.json'), stats)
    atomic_jsonl(_pkg_resolve(OUT / 'quality_sample.jsonl'), sample)
    lines = ['E1-B2 합성 질문 품질 통계 (생성 결과 불변, 참고용)', f"  질문 수 {N:,}  단어 수 평균 {stats['words']['mean']:.1f} (중앙값 {stats['words']['median']:.0f}, 범위 {stats['words']['min']}~{stats['words']['max']})", f'  완전 중복 {exact_dup:,}  정규화 중복 {norm_dup:,}  비ASCII 포함 {non_ascii:,}', f'  일반 질문 패턴(this passage 등) {generic:,}', f'  문서 어휘 중복률 평균 {overlap.mean():.3f}  0.3 미만 {low_overlap:,}  제목 토큰 포함 {title_hit:,}', f'  저널: ok {j_ok:,} / 충돌 {j_conflict} / 비정상 {j_bad}', '  표본 200개(시드 0) -> quality_sample.jsonl 의 label_answerable / label_relevant 를 눈가림으로 채울 것', '  해석: 어휘 중복률은 연관성의 대리 지표일 뿐이며 의미적 품질을 보장하지 않는다.']
    _pkg_output(_pkg_resolve(OUT / 'quality_report.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))

def top8(row, k=8):
    ids = np.argpartition(-row, k - 1)[:k]
    return sorted((int(x) for x in ids), key=lambda j: (-float(row[j]), j))

def retrieve(args):
    (extra, main, vanilla, hold) = load_inputs()
    config = conf(args, extra)
    prepare_meta = _pkg_resolve(OUT / 'prepare_manifest.json')
    need(prepare_meta.exists(), 'prepare 먼저 실행')
    need(json.loads(_pkg_resolve(OUT / 'sources.json').read_text()) == config, '입력 설정이 prepare와 다름')
    pm = json.loads(prepare_meta.read_text())
    qp = _pkg_resolve(OUT / 'questions.jsonl')
    need(sha(qp) == pm['questions_sha256'], '질문 파일 변조/불일치')
    from S00_qi_common import ENCODER, find_cache, load_jsonl, build_corpus, split_paragraphs, para_key, INDEX as QINDEX
    from S00_retrieval_policy import qi_policy
    old_items = load_jsonl(QINDEX)
    old_docs = build_corpus(old_items)
    old_q = np.load(find_cache('qidx'), mmap_mode='r')
    need(old_q.shape[0] == len(old_items), '기준 QI 임베딩과 인덱스 행 수 불일치')
    base_doc_keys = {para_key(d): d['title'] for d in old_docs}
    item_doc_keys = [[para_key(p) for p in split_paragraphs(r.get('context', ''))] for r in old_items]
    newq = list(readl(qp))
    need(len(newq) == N and all((newq[i]['title'] == extra[i]['title'] and newq[i]['i'] == i for i in range(N))), '새 질문 순서 불일치')
    existing_titles = set(base_doc_keys.values())
    need(not existing_titles & {d['title'] for d in extra}, '신규 문서가 기존 제목과 겹침')
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(ENCODER, device=args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    needed = ['query: ' + x['question'] for x in newq]
    cache = _pkg_resolve(OUT / 'new_questions_e5.npy')
    if cache.exists():
        Enew = np.load(cache, mmap_mode='r')
        need(Enew.shape == (N, old_q.shape[1]), '새 임베딩 캐시 Shape 불일치')
        need(_pkg_resolve(OUT / 'new_questions_e5_meta.json').exists(), '임베딩 메타 없음')
        emeta = json.loads(_pkg_resolve(OUT / 'new_questions_e5_meta.json').read_text())
        need(emeta['questions_sha256'] == sha(qp) and emeta['embedding_sha256'] == sha(cache), '임베딩 캐시 입력 SHA 불일치')
        print('새 질문 임베딩 재사용')
    else:
        Enew = model.encode(needed, batch_size=args.batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).astype('float32')
        need(Enew.shape == (N, old_q.shape[1]), '임베딩 크기 불일치')
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, Enew)
        atomic_json(_pkg_resolve(OUT / 'new_questions_e5_meta.json'), {'questions_sha256': sha(qp), 'embedding_sha256': sha(cache), 'encoder': ENCODER, 'shape': list(Enew.shape)})
    devq = {}
    for x in readl(DEV):
        qid = str(x.get('_id', x.get('id')))
        if qid in hold:
            devq[qid] = str(x['question'])
    need(set(devq) == hold, 'Holdout 질문 텍스트 불일치')
    pred = _pkg_resolve(OUT / 'retrieval.jsonl')
    need(not pred.exists(), '검색 결과가 이미 존재함. 덮어쓰기 금지')
    rows = []
    qids = sorted(hold)
    for s in range(0, len(qids), args.query_batch):
        batchids = qids[s:s + args.query_batch]
        Eq = model.encode(['query: ' + devq[x] for x in batchids], batch_size=args.batch, normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        base_s = Eq @ np.asarray(old_q).T
        new_s = Eq @ np.asarray(Enew).T
        for (n, qid) in enumerate(batchids):
            base8 = top8(base_s[n], K)
            (old_keys, short) = qi_policy(base8, item_doc_keys, P)
            base_titles = [base_doc_keys[k] for k in old_keys]
            need(base_titles == main[qid]['qi_titles'], f'H1 QI 기준선 불일치: {qid}')
            budget_results = {}
            for (count, label) in ((N25, 'b1'), (N, 'b2')):
                scores = np.concatenate([base_s[n], new_s[n, :count]])
                picked = top8(scores, K)
                (used, titles) = (set(), [])
                for ix in picked:
                    if ix < len(old_items):
                        options = [base_doc_keys[k] for k in item_doc_keys[ix]]
                    else:
                        options = [extra[ix - len(old_items)]['title']]
                    for title in options:
                        if title not in used:
                            titles.append(title)
                            used.add(title)
                            if len(titles) == P:
                                break
                    if len(titles) == P:
                        break
                budget_results[label + '_titles'] = titles
                budget_results[label + '_top_item_ids'] = picked
            rows.append({'qid': qid, 'base_titles': base_titles, **budget_results})
        if s % (args.query_batch * 5) == 0:
            print(f'검색 {min(s + len(batchids), len(qids))}/{len(qids)}', flush=True)
    need(len(rows) == 4405, '검색 결과 수 불일치')
    atomic_jsonl(pred, rows)
    atomic_json(_pkg_resolve(OUT / 'retrieval_manifest.json'), {'created_at': now(), 'n': len(rows), 'pred_sha256': sha(pred), 'prepare_manifest_sha256': sha(prepare_meta), 'emb_sha256': sha(cache), 'E1B_retrieval_sha256': sha(E1B), 'gold_used_in_retrieval': False, 'P': P, 'TOP_ITEMS': K})
    print('[PASS] E1-B2 검색 완료; Gold 미사용, H1 QI 기준선 완전 일치')

def ci(x, seed=0, nboot=10000):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    boots = np.empty(nboot)
    for s in range(0, nboot, 500):
        draws = min(500, nboot - s)
        idx = rng.integers(0, len(x), (draws, len(x)))
        boots[s:s + draws] = x[idx].mean(axis=1)
    return [float(x.mean()), *map(float, np.percentile(boots, [2.5, 97.5]))]

def score():
    (extra, main, vanilla, hold) = load_inputs()
    pre = json.loads(_pkg_resolve(OUT / 'prepare_manifest.json').read_text())
    manifest = json.loads(_pkg_resolve(OUT / 'retrieval_manifest.json').read_text())
    rp = _pkg_resolve(OUT / 'retrieval.jsonl')
    need(sha(rp) == manifest['pred_sha256'], '검색 결과 해시 불일치')
    need(sha(_pkg_resolve(OUT / 'prepare_manifest.json')) == manifest['prepare_manifest_sha256'], 'prepare 매니페스트 해시 불일치')
    need(sha(_pkg_resolve(OUT / 'questions.jsonl')) == pre['questions_sha256'], '질문 인덱스 변경됨')
    rows = {str(x['qid']): x for x in readl(rp)}
    need(set(rows) == hold, '재검색 평가 ID 집합 불일치')
    need(all((rows[q]['base_titles'] == main[q]['qi_titles'] for q in hold)), '기준선 검색 결과 불일치')
    gold = {}
    for r in readl(DEV):
        qid = str(r.get('_id', r.get('id')))
        if qid in hold:
            gold[qid] = set((str(t) for t in r['supporting_facts']['title']))
    need(set(gold) == hold and all(gold.values()), 'Gold 문항 집합 불일치')
    from S00_retrieval_policy import recall_at_p
    qids = sorted(hold)
    methods = {'QI_base': lambda q: rows[q]['base_titles'], 'QI_25k': lambda q: rows[q]['b1_titles'], 'QI_50k': lambda q: rows[q]['b2_titles'], 'VAN_base': lambda q: vanilla[q]['base_titles'], 'VAN_25k': lambda q: vanilla[q]['b1_titles'], 'VAN_50k': lambda q: vanilla[q]['b2_titles']}
    arr = {name: np.asarray([recall_at_p(fn(q), gold[q]) for q in qids], dtype=float) for (name, fn) in methods.items()}
    pairs = [('QI_25k', 'QI_base'), ('QI_50k', 'QI_base'), ('VAN_25k', 'VAN_base'), ('VAN_50k', 'VAN_base'), ('QI_25k', 'VAN_25k'), ('QI_50k', 'VAN_50k'), ('QI_base', 'VAN_base')]
    stats = {'n': len(qids), 'results': {k: float(v.mean()) for (k, v) in arr.items()}, 'paired': {}, 'generated_questions': N, 'analysis': 'post-hoc exploratory; existing holdout was previously analyzed; no new confirmatory claim', 'input_sha256': {'e1b': sha(E1B), 'h1': sha(BASEPRED), 'qgen': sha(_pkg_resolve(OUT / 'questions.jsonl')), 'retrieval': sha(rp)}}
    extra_titles = {d['title'] for d in extra}
    lines = ['E1-B2 post-hoc exploratory QI question index expansion', 'N=4,405, P=2, Top-8 indexed questions; +25k and +50k same E1-B v2 docs', 'IMPORTANT: no new confirmatory claim; seen holdout; gold ONLY read by score.', '구조 차이: 합성 질문은 문서당 1개·1:1 연결, 기존 색인 질문은 1:N 연결. 따라서 이 실험이 보는 것은', '  "동일 문서 코퍼스에서 1:1 연결 방식으로 질문 색인을 확장한 효과"이며 색인 규모만의 효과가 아니다.', '']
    for (budget, key) in (('25k', 'b1'), ('50k', 'b2')):
        n_docs = sum((len(rows[q][key + '_titles']) for q in qids))
        n_extra_docs = sum((sum((t in extra_titles for t in rows[q][key + '_titles'])) for q in qids))
        n_q_touched = sum((any((t in extra_titles for t in rows[q][key + '_titles'])) for q in qids))
        stats[f'qi_{budget}_final_docs_from_extra'] = n_extra_docs
        stats[f'qi_{budget}_questions_with_extra_doc'] = n_q_touched
        lines.append(f'QI {budget}: 최종 문서 {n_docs:,}개 중 추가 문서(합성 질문 경유) {n_extra_docs:,}개, 추가 문서가 들어간 질문 {n_q_touched:,}/{len(qids):,}')
    lines.append('')
    for (name, v) in arr.items():
        lines.append(f'  {name:12} Recall@2={v.mean():.6f}')
    lines.append('')
    for (a, b) in pairs:
        triple = ci(arr[a] - arr[b])
        stats['paired'][f'{a}_minus_{b}'] = {'diff': triple[0], 'ci95': triple[1:]}
        lines.append(f'{a} − {b}: {triple[0]:+.6f} CI [{triple[1]:+.6f}, {triple[2]:+.6f}]')
    for budget in ('25k', '50k'):
        count_gain = count_loss = 0
        for q in qids:
            old = set(rows[q]['base_titles'])
            new = set(rows[q]['b1_titles' if budget == '25k' else 'b2_titles'])
            count_gain += len(new - old & gold[q])
            count_loss += len(old - new & gold[q])
        stats[f'qi_{budget}_gold_gained'] = count_gain
        stats[f'qi_{budget}_gold_lost'] = count_loss
        lines.append(f'QI {budget} Gold 문서 회수 증가/손실: {count_gain}/{count_loss}')
    need(not _pkg_resolve(OUT / 'score.json').exists() and (not _pkg_resolve(OUT / 'score.txt').exists()), '기존 채점 결과 덮어쓰기 금지')
    atomic_json(_pkg_resolve(OUT / 'score.json'), stats)
    _pkg_output(_pkg_resolve(OUT / 'score.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('[PASS] score.json / score.txt 저장')

def verify(args):
    (extra, main, van, hold) = load_inputs()
    confdata = conf(args, extra)
    sources = _pkg_resolve(OUT / 'sources.json')
    need(sources.exists() and json.loads(sources.read_text()) == confdata, '입력 동결 검증 실패')
    pman = json.loads(_pkg_resolve(OUT / 'prepare_manifest.json').read_text())
    qp = _pkg_resolve(OUT / 'questions.jsonl')
    need(sha(qp) == pman['questions_sha256'], '질문 인덱스 불일치')
    newq = list(readl(qp))
    need(len(newq) == N, f'질문 수 불일치: {len(newq)}')
    need(all((newq[i]['i'] == i and newq[i]['title'] == extra[i]['title'] and newq[i]['question'].strip() for i in range(N))), '질문-문서 1:1 연결 또는 빈 질문')
    journal = _pkg_resolve(OUT / 'question_journal.jsonl')
    need(journal.exists() and sha(journal) == pman.get('journal_sha256'), '저널 해시 불일치')
    seen = {}
    for r in readl(journal):
        if r.get('status') == 'ok':
            i = int(r['i'])
            need(i not in seen or seen[i] == r['question'], f'저널 충돌: i={i}')
            seen[i] = r['question']
    need(set(seen) == set(range(N)) and all((seen[i] == newq[i]['question'] for i in range(N))), '저널의 유효 질문과 questions.jsonl 불일치')
    print(f'[PASS] 질문 {N:,} 개 1:1 연결, 저널 유효·충돌 없음')
    cache = _pkg_resolve(OUT / 'new_questions_e5.npy')
    if cache.exists():
        em = json.loads(_pkg_resolve(OUT / 'new_questions_e5_meta.json').read_text())
        need(em['questions_sha256'] == sha(qp) and em['embedding_sha256'] == sha(cache), '임베딩 SHA 불일치')
        need(tuple(np.load(cache, mmap_mode='r').shape) == tuple(em['shape']) and em['shape'][0] == N, '임베딩 shape 불일치')
        print('[PASS] 신규 질문 임베딩 SHA·shape 일치')
    if _pkg_resolve(OUT / 'retrieval_manifest.json').exists():
        rm = json.loads(_pkg_resolve(OUT / 'retrieval_manifest.json').read_text())
        rp = _pkg_resolve(OUT / 'retrieval.jsonl')
        need(sha(rp) == rm['pred_sha256'], '검색 결과 불일치')
        need(sha(_pkg_resolve(OUT / 'prepare_manifest.json')) == rm['prepare_manifest_sha256'], '준비 버전이 검색 버전과 다름')
        need(cache.exists() and sha(cache) == rm['emb_sha256'], '검색에 쓴 임베딩 해시 불일치')
        rows = list(readl(rp))
        need(len(rows) == 4405 and {str(x['qid']) for x in rows} == hold, '검색 수/ID 불일치')
        need(all((x['base_titles'] == main[x['qid']]['qi_titles'] for x in rows)), 'QI 기준선 불일치')
        from S00_qi_common import INDEX as QINDEX, load_jsonl, build_corpus
        allowed = {d['title'] for d in build_corpus(load_jsonl(QINDEX))} | {d['title'] for d in extra}
        allowed25 = {d['title'] for d in build_corpus(load_jsonl(QINDEX))} | {d['title'] for d in extra[:N25]}
        for x in rows:
            for (key, ok_set) in (('b1_titles', allowed25), ('b2_titles', allowed)):
                t = x[key]
                need(len(t) <= P and len(set(t)) == len(t), f"{key} 개수/중복 위반: {x['qid']}")
                need(set(t) <= ok_set, f"{key} 에 허용되지 않은 문서: {x['qid']}")
        print('[PASS] 4,405 문항 재검색, 원본 QI 기준선 일치, b1/b2 문서 수·중복·출처 정상')
        if _pkg_resolve(OUT / 'score.json').exists():
            from S00_retrieval_policy import recall_at_p
            gold = {}
            for r in readl(DEV):
                qid = str(r.get('_id', r.get('id')))
                if qid in hold:
                    gold[qid] = set((str(t) for t in r['supporting_facts']['title']))
            byq = {str(x['qid']): x for x in rows}
            sc = json.loads(_pkg_resolve(OUT / 'score.json').read_text())
            qids = sorted(hold)
            recomputed = {'QI_base': np.mean([recall_at_p(byq[q]['base_titles'], gold[q]) for q in qids]), 'QI_25k': np.mean([recall_at_p(byq[q]['b1_titles'], gold[q]) for q in qids]), 'QI_50k': np.mean([recall_at_p(byq[q]['b2_titles'], gold[q]) for q in qids]), 'VAN_base': np.mean([recall_at_p(van[q]['base_titles'], gold[q]) for q in qids]), 'VAN_50k': np.mean([recall_at_p(van[q]['b2_titles'], gold[q]) for q in qids])}
            for (k, v) in recomputed.items():
                need(abs(float(v) - float(sc['results'][k])) <= 1e-09, f'score.json {k} 재계산 불일치')
            need(sc['input_sha256']['retrieval'] == sha(rp), 'score.json 이 가리키는 검색 결과 해시 불일치')
            print('[PASS] score.json 재계산 일치')
    print('[PASS] 준비 및 입력 파일 해시 일치')

def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('prepare', 'quality', 'retrieve', 'score', 'verify'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--urls', nargs='+', default=['http://172.25.121.198:7777/v1/chat/completions', 'http://172.25.121.170:7777/v1/chat/completions'], help='생성 서버 목록. 문서 i 는 urls[i %% n] 에 먼저 보내고 실패 시 다음 서버로')
    ap.add_argument('--workers', type=int, default=4, help='서버당 동시 요청 수')
    ap.add_argument('--timeout', type=int, default=120)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--query-batch', type=int, default=32)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    OUT = args.out.expanduser().resolve()
    need(args.workers >= 1 and args.batch >= 1 and (args.query_batch >= 1), 'batch/workers >=1')
    {'prepare': lambda : prepare(args), 'quality': lambda : quality(args), 'retrieve': lambda : retrieve(args), 'score': score, 'verify': lambda : verify(args)}[args.command]()
if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError, IndexError) as exc:
        raise SystemExit(f'[FAIL] {exc}')
