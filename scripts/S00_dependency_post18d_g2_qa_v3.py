"""POST-18D follow-up B: G0 vs G2 LLM QA with frozen prompts and full audit.

Run in ~/2610-qi-rag AFTER the H1 G0/G2 holdout score has passed.
To recover existing v2 responses with a conflicting successful retry:
  python post18d_g2_qa_v3.py self-test
  python post18d_g2_qa_v3.py audit     # read-only retry audit
  python post18d_g2_qa_v3.py migrate --conflict-policy first-success
  python post18d_g2_qa_v3.py status
  python post18d_g2_qa_v3.py generate --workers 8   # only missing jobs
  python post18d_g2_qa_v3.py score --bootstrap 10000

Original v2 jobs, manifest, and response history are NEVER modified.
A separate v3 output is frozen with original provenance SHA256. The
first successful append is the explicit canonical response; every distinct
later success is retained in original history and alternative-outcome audit.

Same strict prompt, model Qwen2.5-7B-Instruct-AWQ, temperature=0,
max_tokens=128 as prior E1-A LLM. NO token-length normalization. All QIDs
receive G0; G2 is run separately only if its prompt byte string differs.
When G2 prompt is byte-identical, G0 response is reused for that QID to avoid
spurious differences and unnecessary calls; this means shared-prompt rows
are ONE shared observation, NOT independent model repetitions.
Errors are logged and retried; scoring refuses incomplete/errors.
EM/F1/abstention are automatic proxies, not human semantic validation.
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse, collections, csv, hashlib, json, os, re, sys, tempfile, time, urllib.request, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
STRICT = 'Answer ONLY using the given context.\nAnswer with the shortest possible phrase.\nIf the answer is not in the context, say "I don\'t know".\n\nContext:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:'
DEFAULT_MODEL = 'Qwen2.5-7B-Instruct-AWQ'
DEFAULT_URL = 'http://172.25.121.198:7777/v1/chat/completions'

def need(cond, msg):
    if not cond:
        raise RuntimeError(msg)

def sha(p):
    h = hashlib.sha256()
    with open(_pkg_open_path(_pkg_resolve(p), 'rb'), 'rb') as f:
        for block in iter(lambda : f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def string_sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def readjl(p):
    with open(_pkg_open_path(_pkg_resolve(p), 'r'), encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def keyed(p, key):
    rows = {}
    for d in readjl(p):
        k = str(d[key])
        need(k not in rows, f'duplicate {key}: {p}: {k}')
        rows[k] = d
    return rows

def dump(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')

def dumpjl(p, items):
    with p.open('x', encoding='utf-8') as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n')

def data_paths(root, a):
    base = _pkg_resolve(root / a.stage1)
    return (base, _pkg_resolve(base / 'plan.jsonl'), _pkg_resolve(base / 'manifest.json'), _pkg_resolve(base / 'score/report.json'))

def source(root, a):
    from S00_qi_common import build_corpus, para_key
    (base, planpath, manifestpath, scorepath) = data_paths(root, a)
    for p in (planpath, manifestpath, scorepath):
        need(p.is_file(), f'missing prerequisite: {p}')
    source_manifest = json.loads(manifestpath.read_text(encoding='utf-8'))
    need(sha(planpath) == source_manifest['plan_sha256'], 'G2 policy plan hash mismatch')
    source_score = json.loads(scorepath.read_text(encoding='utf-8'))
    need(source_score['plan_sha256'] == source_manifest['plan_sha256'], 'G2 score/plan mismatch')
    index = _pkg_resolve(root / 'data/_split/index_items.jsonl')
    need(index.is_file(), 'index missing')
    need(sha(index) == source_manifest['inputs_sha256']['index'], 'source frozen index modified')
    paths = {'eval': _pkg_resolve(root / 'data/_split/eval_queries.jsonl'), 'dev': _pkg_resolve(root / 'data/hotpotqa/hotpot_dev_distractor.jsonl'), 'pred': _pkg_resolve(root / 'out/holdout_locked/retrieval_predictions.jsonl'), 'split': _pkg_resolve(root / 'out/holdout_locked/split_ids.json'), 'freeze': _pkg_resolve(root / 'out/holdout_locked/FREEZE.json'), 'retman': _pkg_resolve(root / 'out/holdout_locked/retrieval_manifest.json'), 'explore': _pkg_resolve(root / 'out/p18_evidence_qa_v1/dataset.jsonl'), 'v11': _pkg_resolve(root / 'p18d_goldblind_ablation_v11.py')}
    for (k, p) in paths.items():
        need(sha(p) == source_manifest['inputs_sha256'][k], f'stage1 input modified: {k}')
    docs = build_corpus(list(readjl(index)))
    docmap = {para_key(d): d for d in docs}
    need(len(docs) == len(docmap), 'corpus document-key collision')
    plans = keyed(planpath, 'qid')
    need(len(plans) == 4405, 'stage1 plan count is not 4405')
    return (plans, docmap, source_manifest, source_score, {'stage1_plan_sha': sha(planpath), 'stage1_score_sha': sha(scorepath)})

def prompt(question, keys, docmap):
    need(isinstance(question, str) and bool(question.strip()), 'invalid question')
    need(len(keys) == 2 and len(set(keys)) == 2, 'P2 unique-document budget failed')
    pieces = []
    for k in keys:
        need(k in docmap, f'unknown corpus key: {k}')
        d = docmap[k]
        need(d['title'] != '' and bool(d['text']), 'blank passage/title')
        pieces.append(f"[{d['title']}] {d['text']}")
    return STRICT.format(ctx='\n\n'.join(pieces), q=question)

def build_jobs(plans, docmap):
    jobs = []
    g2_same = 0
    for (qid, r) in sorted(plans.items()):
        a = prompt(r['question'], r['G0_keys'], docmap)
        b = prompt(r['question'], r['G2_keys'], docmap)
        jobs.append({'qid': qid, 'cond': 'G0', 'prompt': a, 'prompt_sha256': string_sha(a)})
        if a != b:
            jobs.append({'qid': qid, 'cond': 'G2', 'prompt': b, 'prompt_sha256': string_sha(b)})
        else:
            g2_same += 1
    return (jobs, g2_same)

def output(root, a):
    return _pkg_resolve(root / a.output).resolve()

def prepare(a):
    root = Path(_pkg_resolve(a.root)).resolve()
    (plans, docmap, sm, ss, refs) = source(root, a)
    out = output(root, a)
    need(not out.exists(), f'output already exists: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    (jobs, same) = build_jobs(plans, docmap)
    cfg = {'model': a.model, 'url': a.api_url, 'temperature': 0.0, 'max_tokens': a.max_tokens, 'prompt_template_sha256': string_sha(STRICT), 'source': 'post18d_g2_qa_v3'}
    need(a.max_tokens == 128, 'keep max_tokens=128 for original E1-A comparability')
    need(a.api_url.startswith('http://') or a.api_url.startswith('https://'), 'invalid API URL')
    need(a.api_url.rstrip('/').endswith('/chat/completions'), 'URL must end with /chat/completions')
    tmp = Path(_pkg_resolve(tempfile.mkdtemp(prefix='.p18d_qa_stage_', dir=out.parent)))
    try:
        dumpjl(_pkg_resolve(tmp / 'jobs.jsonl'), jobs)
        dump(_pkg_resolve(tmp / 'manifest.json'), {'n_qids': len(plans), 'n_jobs': len(jobs), 'g2_identical_prompt_qids': same, 'scope': sm['scope'], 'stage1_refs': refs, 'config': cfg, 'script_sha256': sha(Path(_pkg_resolve(__file__)).resolve()), 'jobs_sha256': sha(_pkg_resolve(tmp / 'jobs.jsonl')), 'protocol': 'G0 for all; G2 only if exact prompt bytes differ; shared output copied, never independently scored'})
        tmp.rename(out)
    finally:
        if tmp.exists():
            __import__('shutil').rmtree(tmp)
    print('PREPARE PASS:', out, 'QIDs', len(plans), 'jobs', len(jobs), 'identical G0/G2 prompts', same)

def verify(a):
    root = Path(_pkg_resolve(a.root)).resolve()
    out = output(root, a)
    m = json.loads(_pkg_resolve(out / 'manifest.json').read_text(encoding='utf-8'))
    need(sha(Path(_pkg_resolve(__file__)).resolve()) == m['script_sha256'], 'QA code changed since prepare')
    need(sha(_pkg_resolve(out / 'jobs.jsonl')) == m['jobs_sha256'], 'QA frozen jobs altered')
    if 'migration' in m:
        mg = m['migration']
        orig = Path(_pkg_resolve(mg['source_output'])).resolve()
        need(sha(_pkg_resolve(orig / 'jobs.jsonl')) == mg['source_jobs_sha256'], 'original v2 jobs modified since migration')
        need(sha(_pkg_resolve(orig / 'responses.jsonl')) == mg['source_responses_sha256'], 'original v2 responses modified since migration; migration must be re-audited')
        need(sha(_pkg_resolve(out / 'migration_audit.json')) == mg['migration_audit_sha256'], 'migration audit modified')
        need(json.loads(_pkg_resolve(out / 'migration_audit.json').read_text(encoding='utf-8'))['source_responses_sha256'] == mg['source_responses_sha256'], 'migration provenance mismatch')
    (plans, docs, sm, ss, refs) = source(root, a)
    need(refs == m['stage1_refs'], 'stage1 changed since QA freeze')
    (j, same) = build_jobs(plans, docs)
    need(same == m['g2_identical_prompt_qids'] and len(j) == m['n_jobs'], 'job count changed')
    jobmap = {(x['qid'], x['cond']): x for x in j}
    need(len(jobmap) == len(j), 'duplicate generated work key')
    for r in readjl(_pkg_resolve(out / 'jobs.jsonl')):
        key = (r['qid'], r['cond'])
        need(key in jobmap and r == jobmap[key], f'frozen QA job diverged: {key}')
    return (root, out, m, plans, j, sm)

def load_attempts(path, jobmap):
    done = {}
    attempts = collections.Counter()
    if not path.exists():
        return (done, attempts)
    for rec in readjl(path):
        k = (rec.get('qid'), rec.get('cond'))
        need(k in jobmap, f'unknown answer key: {k}')
        need(rec.get('prompt_sha256') == jobmap[k]['prompt_sha256'], 'mismatched response prompt hash')
        need(rec.get('status') in ('ok', 'error'), 'invalid job status')
        need(isinstance(rec.get('pred'), str), 'invalid response pred')
        attempts[k] += 1
        if rec['status'] == 'ok':
            if k in done:
                need(rec['pred'] == done[k]['pred'], f'conflicting successful retry: {k}')
            done[k] = rec
    return (done, attempts)

def inspect_old_attempts(path, jobmap, cfg):
    """Read only. Validate every line, retain first successful response per job.
    Original history remains untouched, including conflicting successes.
    """
    need(path.is_file(), f'old response history not found: {path}')
    chosen = {}
    first_line = {}
    variants = collections.defaultdict(list)
    attempts = collections.Counter()
    successful = collections.Counter()
    errors = 0
    same = 0
    with path.open('r', encoding='utf-8-sig') as f:
        for (lineno, line) in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f'invalid original response JSON at line {lineno}: {e}')
            k = (rec.get('qid'), rec.get('cond'))
            need(k in jobmap, f'unknown response key line {lineno}: {k}')
            need(rec.get('prompt_sha256') == jobmap[k]['prompt_sha256'], f'prompt SHA mismatch at line {lineno}: {k}')
            need(rec.get('status') in ('ok', 'error'), f'invalid status line {lineno}: {k}')
            need(isinstance(rec.get('pred'), str), f'invalid pred line {lineno}: {k}')
            served = rec.get('served_model', '') or ''
            need(isinstance(served, str), f'invalid served_model line {lineno}')
            if served:
                requested = cfg['model']
                need(served == requested or served.rsplit('/', 1)[-1] == requested.rsplit('/', 1)[-1], f'served model mismatch line {lineno}: {served} vs {requested}')
            attempts[k] += 1
            if rec['status'] == 'error':
                errors += 1
                continue
            successful[k] += 1
            if k not in chosen:
                chosen[k] = rec
                first_line[k] = lineno
            elif rec['pred'] == chosen[k]['pred']:
                same += 1
            else:
                variants[k].append(dict(line=lineno, pred=rec['pred'], pred_sha256=string_sha(rec['pred']), length=len(rec['pred']), served_model=served))
    conflicts = []
    for (k, alt) in sorted(variants.items()):
        c = chosen[k]
        conflicts.append({'qid': k[0], 'cond': k[1], 'chosen_rule': 'first_successful_append', 'chosen_line': first_line[k], 'chosen_pred': c['pred'], 'chosen_pred_sha256': string_sha(c['pred']), 'chosen_length': len(c['pred']), 'alternatives': alt})
    return (chosen, {'jobs': len(jobmap), 'response_rows': sum(attempts.values()), 'successful_keys': len(chosen), 'success_records': sum(successful.values()), 'error_records': errors, 'identical_success_duplicates': same, 'conflicting_keys': len(conflicts), 'conflicting_extra_success_records': sum((len(v) for v in variants.values())), 'conflicts': conflicts})

def migrate(a, dry=False):
    """Copy v2 frozen jobs into a new v3 experiment, selecting the first successful
    append for a conflicting key ONLY if the caller explicitly opts in.
    """
    root = Path(_pkg_resolve(a.root)).resolve()
    src = _pkg_resolve(root / a.from_output).resolve()
    out = output(root, a)
    need(src != out, 'source and destination must differ')
    need(_pkg_resolve(src / 'manifest.json').is_file() and _pkg_resolve(src / 'jobs.jsonl').is_file(), 'original v2 manifest/jobs missing')
    need(not out.exists(), f'v3 output already exists (do not overwrite): {out}')
    m = json.loads(_pkg_resolve(src / 'manifest.json').read_text(encoding='utf-8'))
    need(sha(_pkg_resolve(src / 'jobs.jsonl')) == m['jobs_sha256'], 'original jobs SHA does not match v2 freeze')
    old_code = _pkg_resolve(root / 'post18d_g2_qa_v2.py')
    need(old_code.is_file() and sha(old_code) == m['script_sha256'], 'original v2 QA code hash mismatch; no safe migration until v2 code is restored')
    (plans, docs, sm, ss, refs) = source(root, a)
    need(m['stage1_refs'] == refs, 'original stage1 references changed')
    (expected, g2_same) = build_jobs(plans, docs)
    need(len(expected) == m['n_jobs'] and g2_same == m['g2_identical_prompt_qids'], 'recomputed jobs count mismatch')
    jobmap = {(x['qid'], x['cond']): x for x in expected}
    original = list(readjl(_pkg_resolve(src / 'jobs.jsonl')))
    need(len(original) == len(expected) and original == expected, 'original jobs contents or order differ from frozen G0/G2 prompts')
    (chosen, audit) = inspect_old_attempts(_pkg_resolve(src / 'responses.jsonl'), jobmap, m['config'])
    audit.update({'source_output': str(src), 'source_jobs_sha256': sha(_pkg_resolve(src / 'jobs.jsonl')), 'source_responses_sha256': sha(_pkg_resolve(src / 'responses.jsonl')), 'original_manifest_sha256': sha(_pkg_resolve(src / 'manifest.json')), 'scope': m['scope'], 'resolution_policy': a.conflict_policy or 'strict', 'note': 'Raw historical attempts retained unchanged in source directory. Only chosen first-successful responses copied. Later success conflicts retained verbatim in source file.'})
    print('ORIGINAL RETRY AUDIT')
    for k in ('jobs', 'response_rows', 'successful_keys', 'success_records', 'error_records', 'identical_success_duplicates', 'conflicting_keys', 'conflicting_extra_success_records'):
        print(' ', k, '=', audit[k])
    for rec in audit['conflicts']:
        print('CONFLICT', rec['qid'], rec['cond'], 'original line', rec['chosen_line'], 'sha', rec['chosen_pred_sha256'], 'alternatives', [(x['line'], x['pred_sha256']) for x in rec['alternatives']])
    if dry:
        print('AUDIT PASS (read-only); original history preserved')
        return
    need(a.conflict_policy == 'first-success' or not audit['conflicting_keys'], 'Conflicting successes present: use audit first, then explicitly pass --conflict-policy first-success to migrate; never silently choose')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(_pkg_resolve(tempfile.mkdtemp(prefix='.p18d_h2_migration_', dir=out.parent)))
    try:
        shutil.copy2(_pkg_resolve(src / 'jobs.jsonl'), _pkg_resolve(tmp / 'jobs.jsonl'))
        dumpjl(_pkg_resolve(tmp / 'responses.jsonl'), [chosen[k] for k in sorted(chosen)])
        dump(_pkg_resolve(tmp / 'migration_audit.json'), audit)
        nm = dict(m)
        nm['source_version'] = 'v2'
        nm['script_sha256'] = sha(Path(_pkg_resolve(__file__)).resolve())
        nm['config'] = dict(m['config'], source='post18d_g2_qa_v3_migrated')
        nm['migration'] = {'source_output': str(src), 'source_jobs_sha256': sha(_pkg_resolve(src / 'jobs.jsonl')), 'source_responses_sha256': sha(_pkg_resolve(src / 'responses.jsonl')), 'migration_audit_sha256': sha(_pkg_resolve(tmp / 'migration_audit.json')), 'selected_rule': 'first_successful_append', 'conflicting_keys': audit['conflicting_keys']}
        dump(_pkg_resolve(tmp / 'manifest.json'), nm)
        tmp.rename(out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print('MIGRATE PASS:', out, 'canonical answers', len(chosen), 'of', len(expected))
    print('ORIGINAL RESPONSES UNCHANGED:', _pkg_resolve(src / 'responses.jsonl'))

def api_call(job, cfg, timeout):
    body = {'model': cfg['model'], 'messages': [{'role': 'user', 'content': job['prompt']}], 'temperature': cfg['temperature'], 'max_tokens': cfg['max_tokens']}
    req = urllib.request.Request(cfg['url'], data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        ret = json.load(f)
    served = str(ret.get('model', '')).strip()
    if served and served != cfg['model'] and (served.rsplit('/', 1)[-1] != cfg['model'].rsplit('/', 1)[-1]):
        raise RuntimeError(f"Served model mismatch: requested={cfg['model']}, got={served}")
    content = ret['choices'][0]['message']['content']
    if content is None:
        content = ''
    need(isinstance(content, str), 'non-text LLM response content')
    return (content.strip(), served)

def generate(a):
    (root, out, m, plans, jobs, sm) = verify(a)
    need(1 <= a.workers <= 32, 'workers out of range')
    need(1 <= a.timeout <= 1200, 'timeout out of range')
    path = _pkg_resolve(out / 'responses.jsonl')
    jobmap = {(x['qid'], x['cond']): x for x in jobs}
    (done, _) = load_attempts(path, jobmap)
    todo = [j for j in jobs if (j['qid'], j['cond']) not in done]
    print('GENERATE pending', len(todo), 'already complete', len(done), 'of', len(jobs), flush=True)
    if not todo:
        return

    def task(j):
        for attempt in range(a.retries + 1):
            try:
                (p, served_model) = api_call(j, m['config'], a.timeout)
                return dict(qid=j['qid'], cond=j['cond'], prompt_sha256=j['prompt_sha256'], pred=p, status='ok', served_model=served_model, err='')
            except Exception as e:
                error = f'{type(e).__name__}: {str(e)[:160]}'
                if attempt < a.retries:
                    time.sleep(min(2 * (attempt + 1), 10))
        return dict(qid=j['qid'], cond=j['cond'], prompt_sha256=j['prompt_sha256'], pred='', status='error', served_model='', err=error)
    with path.open('a', encoding='utf-8') as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
        for (i, res) in enumerate(pool.map(task, todo), 1):
            f.write(json.dumps(res, ensure_ascii=False, sort_keys=True) + '\n')
            f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"GEN {i}/{len(todo)} status={res['status']}", flush=True)
    print('GENERATE FINISHED; run status then score')

def status(a):
    (root, out, m, plans, jobs, sm) = verify(a)
    (done, attempts) = load_attempts(_pkg_resolve(out / 'responses.jsonl'), {(x['qid'], x['cond']): x for x in jobs})
    print('STATUS', len(done), '/', len(jobs), 'missing_or_errors', len(jobs) - len(done), 'identical_prompt_qids', m['g2_identical_prompt_qids'])

def normalize(s):
    s = re.sub('[^\\w\\s]', ' ', (s or '').lower())
    s = re.sub('\\b(a|an|the)\\b', ' ', s)
    return ' '.join(s.split())

def abstain(s):
    return not s.strip() or "don't know" in s.lower() or 'do not know' in s.lower()

def em(pred, answers):
    if abstain(pred):
        return 0.0
    p = normalize(pred)
    return float(any((p and p == normalize(a) for a in answers)))

def f1(pred, answers):
    if abstain(pred):
        return 0.0
    a = normalize(pred).split()
    if not a:
        return 0.0

    def one(g):
        b = normalize(g).split()
        com = collections.Counter(a) & collections.Counter(b)
        common = sum(com.values())
        if not common:
            return 0.0
        return _pkg_resolve(2 * common / (len(a) + len(b)))
    return max(map(one, answers), default=0.0)

def score(a):
    from S00_dependency_post18d_holdout_g2_v1 import boot
    (root, out, m, plans, jobs, sm) = verify(a)
    need(not _pkg_resolve(out / 'score').exists(), 'score output exists, refuse overwrite')
    jobmap = {(j['qid'], j['cond']): j for j in jobs}
    (done, _) = load_attempts(_pkg_resolve(out / 'responses.jsonl'), jobmap)
    need(len(done) == len(jobmap), f'incomplete/failed QA results: {len(done)}/{len(jobmap)}')
    want = set(plans)
    gold = {}
    types = {}
    for r in readjl(_pkg_resolve(root / 'data/hotpotqa/hotpot_dev_distractor.jsonl')):
        qid = str(r.get('_id', r.get('id')))
        if qid in want:
            need(qid not in gold, f'duplicate gold source qid: {qid}')
            value = r.get('answer')
            need(isinstance(value, str) and bool(value.strip()), f'missing gold answer: {qid}')
            gold[qid] = [value]
            types[qid] = r.get('type')
    need(set(gold) == want, 'DEV gold answers missing')
    need(set(types) == want and set(types.values()) == {'bridge', 'comparison'}, 'official DEV question types missing or invalid')
    with _pkg_pathlib.Path(_pkg_open_path(_pkg_resolve(_pkg_resolve(root / a.stage1) / 'score/items.csv'), 'r')).open(encoding='utf-8', newline='') as f:
        old = {r['qid']: r for r in csv.DictReader(f)}
    need(set(old) == want, 'stage1 scored QIDs mismatch')
    records = []
    for qid in sorted(plans):
        b = done[qid, 'G0']['pred']
        other = done.get((qid, 'G2'))
        g = b if other is None else other['pred']
        r = {'qid': qid, 'type': types[qid], 'cluster': old[qid]['cluster'], 'prompt_identical': int(other is None), 'G0_pred': b, 'G2_pred': g}
        for (cond, txt) in (('G0', b), ('G2', g)):
            correct = em(txt, gold[qid])
            answered = int(not abstain(txt))
            r[cond + '_em'] = correct
            r[cond + '_f1'] = f1(txt, gold[qid])
            r[cond + '_answered'] = answered
            r[cond + '_wrong'] = int(answered and (not correct))
        records.append(r)

    def group(rows):
        return {'n': len(rows), 'identical_prompts': sum((r['prompt_identical'] for r in rows)), 'conditions': {c: {k: _pkg_resolve(sum((r[c + '_' + k] for r in rows)) / len(rows)) for k in ('em', 'f1', 'answered', 'wrong')} for c in ('G0', 'G2')}, 'pair_delta': {k: boot([r['G2_' + k] for r in rows], [r['G0_' + k] for r in rows], [r['cluster'] for r in rows], a.bootstrap, a.seed) for k in ('em', 'f1', 'answered', 'wrong')}}
    result = {'n': len(records), 'scope': m['scope'], 'model': m['config']['model'], 'prompt_sha': m['config']['prompt_template_sha256'], 'limitations': ['automatic exact/normalized EM and token F1, not semantic judgments', 'no token-length equalization; only same prompt template and P=2', 'identical G0/G2 prompts reuse the same response; not independent repeat inference', 'previously inspected E1-A holdout; post-hoc validation, not prospective pristine confirmation'], 'overall': group(records), 'by_type': {t: group([r for r in records if r['type'] == t]) for t in ('bridge', 'comparison')}, 'responses_sha256': sha(_pkg_resolve(out / 'responses.jsonl')), 'qa_jobs_sha256': m['jobs_sha256']}
    if 'migration' in m:
        audit = json.loads(_pkg_resolve(out / 'migration_audit.json').read_text(encoding='utf-8'))
        sensitivity = []
        rowmap = {r['qid']: r for r in records}
        for conflict in audit['conflicts']:
            qid = conflict['qid']
            cond = conflict['cond']
            base = rowmap[qid]
            for alt in conflict['alternatives']:
                alternative = alt['pred']
                em_alt = em(alternative, gold[qid])
                f1_alt = f1(alternative, gold[qid])
                answered_alt = int(not abstain(alternative))
                wrong_alt = int(answered_alt and (not em_alt))
                affected = ('G0', 'G2') if cond == 'G0' and base['prompt_identical'] else (cond,)
                scores = {metric: {'G0': base['G0_' + metric], 'G2': base['G2_' + metric]} for metric in ('em', 'f1', 'answered', 'wrong')}
                newvals = {'em': em_alt, 'f1': f1_alt, 'answered': answered_alt, 'wrong': wrong_alt}
                change = {}
                for (metric, original) in scores.items():
                    revised = dict(original)
                    for c in affected:
                        revised[c] = newvals[metric]
                    change[metric] = {'original_G0': original['G0'], 'original_G2': original['G2'], 'alternate_G0': revised['G0'], 'alternate_G2': revised['G2'], 'original_delta_G2_minus_G0': original['G2'] - original['G0'], 'alternate_delta_G2_minus_G0': revised['G2'] - revised['G0']}
                sensitivity.append({'qid': qid, 'cond': cond, 'line': alt['line'], 'prompt_identical': bool(base['prompt_identical']), 'affected_conditions': list(affected), 'metrics': change, 'alternative_pred_sha256': alt['pred_sha256']})
        result['migration_sensitivity'] = {'conflicting_keys': len(audit['conflicts']), 'alternative_rows': sensitivity, 'max_aggregate_rate_impact_per_conflict': _pkg_resolve(1.0 / len(records)), 'note': 'Each conflicting answer alters each affected aggregate condition rate by at most 1/N; an identical prompt changes both arms together, preserving the paired difference. Sensitivity is computed against official gold, without rerunning the model.'}
    tmp = Path(_pkg_resolve(tempfile.mkdtemp(prefix='.qa_score_stage_', dir=out)))
    try:
        with _pkg_pathlib.Path(_pkg_open_path(_pkg_resolve(tmp / 'items.csv'), 'x')).open('x', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        dump(_pkg_resolve(tmp / 'report.json'), result)
        lines = ['# POST-18D G0 vs G2 QA', f"Scope: {m['scope']}; Model: {m['config']['model']}", '|Group|n|G0 EM|G2 EM|Delta EM [95% CI]|G0 F1|G2 F1|Delta F1 [95% CI]|', '|---|---:|---:|---:|---|---:|---:|---|']
        for (name, z) in [('all', result['overall'])] + list(result['by_type'].items()):
            e = z['pair_delta']['em']
            f = z['pair_delta']['f1']
            lines.append(f"|{name}|{z['n']}|{z['conditions']['G0']['em']:.4f}|{z['conditions']['G2']['em']:.4f}|{e['delta']:+.4f} [{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]|{z['conditions']['G0']['f1']:.4f}|{z['conditions']['G2']['f1']:.4f}|{f['delta']:+.4f} [{f['ci95'][0]:+.4f},{f['ci95'][1]:+.4f}]|")
        if 'migration_sensitivity' in result:
            sens = result['migration_sensitivity']
            lines += ['', '## Successful response retry conflict sensitivity', f"Conflicting keys: {sens['conflicting_keys']}; alternative responses: {len(sens['alternative_rows'])}", 'Source response history was not modified. First successful append is the canonical response; alternatives are audited below.', '|QID|Condition (affected)|Original ΔEM|Alternative ΔEM|Original ΔF1|Alternative ΔF1|Original ΔWrong|Alternative ΔWrong|', '|---|---|---:|---:|---:|---:|---:|---:|']
            for x in sens['alternative_rows']:
                e = x['metrics']['em']
                f = x['metrics']['f1']
                w = x['metrics']['wrong']
                lines.append(f"|{x['qid']}|{x['cond']} ({','.join(x['affected_conditions'])})|{e['original_delta_G2_minus_G0']:+.3f}|{e['alternate_delta_G2_minus_G0']:+.3f}|{f['original_delta_G2_minus_G0']:+.3f}|{f['alternate_delta_G2_minus_G0']:+.3f}|{w['original_delta_G2_minus_G0']:+.3f}|{w['alternate_delta_G2_minus_G0']:+.3f}|")
        _pkg_output(_pkg_resolve(tmp / 'report.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        tmp.rename(_pkg_resolve(out / 'score'))
    finally:
        if tmp.exists():
            __import__('shutil').rmtree(tmp)
    print(_pkg_resolve(out / 'score/report.md').read_text(encoding='utf-8'))
    print('QA SCORE PASS', _pkg_resolve(out / 'score'))

def selftest():
    assert normalize('The Foo!') == 'foo'
    assert em('The Foo!', ['foo']) == 1
    assert em("I don't know", ['foo']) == 0
    assert abs(f1('Red Blue', ['Blue Green']) - 0.5) < 1e-10
    assert STRICT.format(ctx='[X] A', q='Q').endswith('Question:\nQ\n\nAnswer:')
    assert string_sha('x') == string_sha('x')
    with tempfile.TemporaryDirectory() as td:
        folder = Path(_pkg_resolve(td))
        path = _pkg_resolve(folder / 'r.jsonl')
        job = {'qid': 'test', 'cond': 'G0', 'prompt_sha256': string_sha('P')}
        records = [dict(job, pred='A', status='ok', served_model=DEFAULT_MODEL), dict(job, pred='', status='error', served_model=''), dict(job, pred='A', status='ok', served_model=DEFAULT_MODEL), dict(job, pred='B', status='ok', served_model=DEFAULT_MODEL)]
        dumpjl(path, records)
        (d, audit) = inspect_old_attempts(path, {('test', 'G0'): job}, {'model': DEFAULT_MODEL})
        assert d['test', 'G0']['pred'] == 'A'
        assert audit['conflicting_keys'] == 1 and audit['identical_success_duplicates'] == 1 and (audit['error_records'] == 1)
        assert audit['conflicts'][0]['alternatives'][0]['pred'] == 'B'
        assert audit['conflicts'][0]['chosen_line'] == 1
        bad = _pkg_resolve(folder / 'bad.jsonl')
        dumpjl(bad, [dict(records[0], prompt_sha256='BAD')])
        try:
            inspect_old_attempts(bad, {('test', 'G0'): job}, {'model': DEFAULT_MODEL})
        except RuntimeError:
            pass
        else:
            raise AssertionError('prompt mismatch should be fatal')
    print('SELF-TEST PASS: QA and audited conflicting successful retries')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['self-test', 'prepare', 'generate', 'status', 'score', 'audit', 'migrate'])
    ap.add_argument('--root', default='.')
    ap.add_argument('--stage1', default='out/p18d_g2_holdout_v1')
    ap.add_argument('--output', default='out/p18d_g2_qa_v3')
    ap.add_argument('--from-output', default='out/p18d_g2_qa_v2')
    ap.add_argument('--conflict-policy', choices=['first-success'], default=None)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--api-url', default=DEFAULT_URL)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=90)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    need(a.retries >= 0 and a.retries <= 10, 'retries out of range')
    need(a.bootstrap >= 1000, 'bootstrap <1000')
    if a.command == 'self-test':
        selftest()
    elif a.command == 'audit':
        migrate(a, dry=True)
    elif a.command == 'migrate':
        migrate(a, dry=False)
    elif a.command == 'prepare':
        prepare(a)
    elif a.command == 'generate':
        generate(a)
    elif a.command == 'status':
        status(a)
    else:
        score(a)
if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyError, AssertionError) as e:
        print('FAIL:', e, file=sys.stderr)
        sys.exit(2)
