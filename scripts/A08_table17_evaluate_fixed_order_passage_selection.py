"""POST-18D-H3 v2: fresh paired G0/G2 generation with fixed Gold-Blind ordering.

READ-ONLY with respect to all frozen H1/H2 data; produces a distinct v2 folder.
Only the 183 questions with distinct selected DOCUMENT SETS receive fresh QA:
2 new responses per question = 366 calls by default, NO H2 answer reuse.
For the 4,222 same-document-set QIDs, fixed-order prompts are byte identical,
so their G2-G0 paired contrasts are exactly zero (no absolute QA scores claimed).

Run in ~/2610-qi-rag:
 python3 post18d_h3_fixed_order_v2.py self-test
 python3 post18d_h3_fixed_order_v2.py prepare
 python3 post18d_h3_fixed_order_v2.py status
 python3 post18d_h3_fixed_order_v2.py generate --workers 8
 python3 post18d_h3_fixed_order_v2.py status
 python3 post18d_h3_fixed_order_v2.py score --bootstrap 10000
 python3 post18d_h3_fixed_order_v2.py verify-score

Selection/order does not CONSULT Gold. Preparation validates existing Gold-based
H1 provenance; scoring uses official DEV labels and independently checks cluster.
Estimates post-hoc policy contrasts under deterministic lexical ordering, and a
paired change of policy contrasts (H3 minus H2). This is NOT a pure causal
order-effect estimate nor a pristine prospective confirmatory study.
"""
from __future__ import annotations
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse
import collections
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import S00_dependency_post18d_g2_qa_v3 as qa
N = 4405
POLICIES = ('G0', 'G2')
METRICS = ('em', 'f1', 'answered', 'wrong')

def need(test, message):
    if not test:
        raise RuntimeError(message)

def read_json(p):
    return json.loads(Path(_pkg_resolve(p)).read_text(encoding='utf-8'))

def dump_json(p, x):
    with _pkg_pathlib.Path(_pkg_open_path(Path(_pkg_resolve(p)), 'x')).open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')

def dump_jsonl(p, rows):
    with _pkg_pathlib.Path(_pkg_open_path(Path(_pkg_resolve(p)), 'x')).open('x', encoding='utf-8') as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n')

def stable_keys(keys, docs):
    need(isinstance(keys, list) and len(keys) == 2 and (len(set(keys)) == 2), 'expected exactly two different document keys')
    need(all((k in docs for k in keys)), 'document key missing in original corpus')
    return sorted(keys, key=lambda k: (docs[k]['title'].casefold(), docs[k]['title'], k))

def partition(plan):
    a = plan['G0_keys']
    b = plan['G2_keys']
    need(len(a) == len(b) == 2 and len(set(a)) == len(set(b)) == 2, f"invalid P2 plan: {plan['qid']}")
    if a == b:
        return 'identical'
    if set(a) == set(b):
        return 'order_only'
    return 'document_set_changed'

def qa_args(root, a):
    return argparse.Namespace(root=str(root), stage1=a.stage1, output=a.h2, bootstrap=1000, seed=0, workers=1, timeout=90, retries=0)

def prereq(a, with_docs=False):
    root = Path(_pkg_resolve(a.root)).resolve()
    qargs = qa_args(root, a)
    (root2, h2out, hm, plans, h2jobs, sm) = qa.verify(qargs)
    need(root2 == root and len(plans) == N and (len(h2jobs) >= N), 'H2 prerequisite mismatch')
    scorefile = _pkg_resolve(h2out / 'score/report.json')
    need(scorefile.is_file(), 'H2 score not complete')
    h2score = read_json(scorefile)
    need(h2score.get('n') == N and h2score.get('qa_jobs_sha256') == hm['jobs_sha256'], 'H2 score/job SHA mismatch')
    h2response = _pkg_resolve(h2out / 'responses.jsonl')
    need(h2response.is_file() and h2score.get('responses_sha256') == qa.sha(h2response), 'H2 response history differs from scored SHA; refuse unpinned H2')
    h2items = _pkg_resolve(h2out / 'score/items.csv')
    need(h2items.is_file(), 'H2 question-level scores missing')
    h2map = {(j['qid'], j['cond']): j for j in h2jobs}
    need(len(h2map) == len(h2jobs), 'H2 job key duplicate')
    (h2done, _) = qa.load_attempts(h2response, h2map)
    need(len(h2done) == len(h2map), 'H2 responses incomplete or conflicting')
    stage1 = _pkg_resolve(root / a.stage1)
    h1path = _pkg_resolve(stage1 / 'score/items.csv')
    need(h1path.is_file(), 'H1 question-level scores missing')
    with h1path.open(encoding='utf-8-sig', newline='') as f:
        h1rows = list(csv.DictReader(f))
    need(len(h1rows) == N and len({r['qid'] for r in h1rows}) == N, 'H1 scored QIDs missing or duplicated')
    h1 = {r['qid']: r for r in h1rows}
    need(set(h1) == set(plans), 'H1/H2 QID set mismatch')
    docs = None
    if with_docs:
        from S00_qi_common import build_corpus, para_key, corpus_fingerprint
        idx = _pkg_resolve(root / 'data/_split/index_items.jsonl')
        h1manifest = read_json(_pkg_resolve(stage1 / 'manifest.json'))
        need(qa.sha(idx) == h1manifest['inputs_sha256']['index'], 'frozen index SHA changed')
        corpus = build_corpus(list(qa.readjl(idx)))
        docs = {para_key(d): d for d in corpus}
        need(len(corpus) == len(docs) == 105889, 'document corpus/key cardinality mismatch')
        need(corpus_fingerprint(corpus) == '4976962eacc1d7a1e8398657939729795a252e4f27e44f2693b3cc13709e5872', 'independent frozen corpus fingerprint mismatch')
    refs = {'stage1_plan': qa.sha(_pkg_resolve(stage1 / 'plan.jsonl')), 'stage1_manifest': qa.sha(_pkg_resolve(stage1 / 'manifest.json')), 'stage1_score_items': qa.sha(h1path), 'stage1_score_report': qa.sha(_pkg_resolve(stage1 / 'score/report.json')), 'h2_jobs': qa.sha(_pkg_resolve(h2out / 'jobs.jsonl')), 'h2_manifest': qa.sha(_pkg_resolve(h2out / 'manifest.json')), 'h2_responses': qa.sha(h2response), 'h2_score_items': qa.sha(h2items), 'h2_score_report': qa.sha(scorefile), 'h2_script': qa.sha(Path(_pkg_resolve(qa.__file__)).resolve()), 'qa_source_code': qa.sha(Path(_pkg_resolve(qa.__file__)).resolve())}
    return (root, h2out, hm, plans, h2map, h2done, h1, docs, refs, h2score)

def build_fixed_jobs(plans, docs, h2map=None, h2done=None):
    (changed, jobs) = ([], [])
    count = collections.Counter()
    order_counts = collections.Counter()
    for (qid, p) in sorted(plans.items()):
        cat = partition(p)
        count[cat] += 1
        if cat != 'document_set_changed':
            need(stable_keys(p['G0_keys'], docs) == stable_keys(p['G2_keys'], docs), f'unchanged document set has differing fixed order: {qid}')
            continue
        fixed = {}
        pair = {}
        for pol in POLICIES:
            keys = stable_keys(p[f'{pol}_keys'], docs)
            sentence = qa.prompt(p['question'], keys, docs)
            ps = qa.string_sha(sentence)
            fixed[pol] = {'keys': keys, 'prompt_sha256': ps}
            pair[pol] = {'qid': qid, 'cond': pol, 'prompt': sentence, 'prompt_sha256': ps}
        need(fixed['G0']['prompt_sha256'] != fixed['G2']['prompt_sha256'], f'different documents unexpectedly yield same prompt: {qid}')
        first = 'G0' if int(qa.string_sha(qid), 16) % 2 == 0 else 'G2'
        second = 'G2' if first == 'G0' else 'G0'
        jobs.extend((pair[first], pair[second]))
        order_counts[first] += 1
        changed.append({'qid': qid, 'fixed': fixed, 'original_G0_keys': p['G0_keys'], 'original_G2_keys': p['G2_keys'], 'scheduled_first': first})
    need(sum(count.values()) == N, 'expected 4405 policy plans')
    need(len(jobs) == 2 * len(changed), 'fresh-pair count mismatch')
    need(abs(order_counts['G0'] - order_counts['G2']) <= 50, f'imbalanced deterministic job dispatch: {dict(order_counts)}')
    return (changed, jobs, dict(count), dict(order_counts))

def check_output_path(root, out, inputs):
    need(out != root and root in out.parents, 'output must be inside project root')
    need(out not in inputs and all((out not in p.parents for p in inputs)), 'output cannot contain frozen sources')

def prepare(a):
    (root, h2out, hm, plans, h2map, h2done, h1, docs, refs, h2score) = prereq(a, with_docs=True)
    need(hm['config']['temperature'] == 0.0 and hm['config']['max_tokens'] == 128, 'H2 inference config mismatch')
    need(hm['config']['prompt_template_sha256'] == qa.string_sha(qa.STRICT), 'H2 legacy strict prompt template modified')
    (changed, jobs, count, order_counts) = build_fixed_jobs(plans, docs)
    if a.expect_changed is not None:
        need(count['document_set_changed'] == a.expect_changed, 'changed document count mismatch')
    if a.expect_order_only is not None:
        need(count['order_only'] == a.expect_order_only, 'order-only count mismatch')
    urls = [s.strip() for s in (a.api_url or hm['config']['url']).split(',') if s.strip()]
    need(len(urls) == 1 and urls[0].startswith(('http://', 'https://')) and urls[0].rstrip('/').endswith('/chat/completions'), 'v2 requires ONE explicitly trusted OpenAI-compatible LLM endpoint')
    out = _pkg_resolve(root / a.output).resolve()
    check_output_path(root, out, [h2out, _pkg_resolve(root / a.stage1)])
    need(not out.exists(), f'output already exists: {out}; use status/generate/score')
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(_pkg_resolve(tempfile.mkdtemp(prefix='.p18d_h3v2_', dir=out.parent)))
    try:
        dump_jsonl(_pkg_resolve(temp / 'changed_qids.jsonl'), changed)
        dump_jsonl(_pkg_resolve(temp / 'jobs.jsonl'), jobs)
        m = {'protocol': 'H3-v2: lexical Gold-Blind fixed-order, fresh paired generation for all changed sets', 'n_qids': N, 'changed_sets': len(changed), 'category_counts': count, 'n_jobs': len(jobs), 'n_fresh_required': len(jobs), 'n_reused': 0, 'scheduled_first_counts': order_counts, 'fixed_order': 'title.casefold(), title, document_key', 'scope': hm['scope'], 'source_hashes': refs, 'script_sha256': qa.sha(Path(_pkg_resolve(__file__)).resolve()), 'files_sha256': {name: qa.sha(_pkg_resolve(temp / name)) for name in ('changed_qids.jsonl', 'jobs.jsonl')}, 'config': {k: hm['config'][k] for k in ('model', 'temperature', 'max_tokens', 'prompt_template_sha256')}, 'url': urls[0], 'reference_h2_output': str(h2out), 'limitations': ['Post-hoc mechanistic control on previously inspected Holdout', 'Same-set 4222 QIDs have paired delta=0; absolute H3 outcomes not generated', 'All changed-set G0/G2 generations are fresh and not H2-reused', 'Deterministic order does not eliminate all positional effects or stochasticity', 'Cluster CI descriptive, no preregistration', 'Wrong is answered and EM=0, NOT semantic hallucination']}
        dump_json(_pkg_resolve(temp / 'manifest.json'), m)
        temp.rename(out)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    print('PREPARE PASS:', out)
    print('N', N, 'changed', len(changed), 'order-only', count['order_only'], 'identical', count['identical'], 'fresh jobs', len(jobs), 'reused H2 jobs', 0)
    print('Scheduled first:', order_counts)
    print('Gold labels were NOT used for document selection or order; prior H1 audit was read.')

def verify(a):
    root = Path(_pkg_resolve(a.root)).resolve()
    out = _pkg_resolve(root / a.output).resolve()
    need(out.is_dir(), f'no prepared H3-v2 folder: {out}')
    m = read_json(_pkg_resolve(out / 'manifest.json'))
    need(m['script_sha256'] == qa.sha(Path(_pkg_resolve(__file__)).resolve()), 'H3-v2 code changed since preparation: refuse unpinned run')
    (root, h2out, hm, plans, h2map, h2done, h1, docs, refs, _) = prereq(a, with_docs=True)
    need(m['source_hashes'] == refs, 'frozen H1/H2 sources changed')
    need(m['reference_h2_output'] == str(h2out), 'H2 source path drift')
    need(m['config'] == {k: hm['config'][k] for k in ('model', 'temperature', 'max_tokens', 'prompt_template_sha256')}, 'H2 inference parameters changed')
    for (name, digest) in m['files_sha256'].items():
        need(qa.sha(_pkg_resolve(out / name)) == digest, f'H3 prepared file modified: {name}')
    changed = qa.keyed(_pkg_resolve(out / 'changed_qids.jsonl'), 'qid')
    jobs = list(qa.readjl(_pkg_resolve(out / 'jobs.jsonl')))
    need(len(changed) == m['changed_sets'] and len(jobs) == m['n_jobs'] == 2 * len(changed) and (m['n_reused'] == 0) and (m['n_qids'] == N), 'H3-v2 prepared counts inconsistent')
    (expected_changed, expected_jobs, count, first) = build_fixed_jobs(plans, docs)
    need(changed == {x['qid']: x for x in expected_changed} and jobs == expected_jobs, 'H3 fixed-order jobs differ from Gold-Blind policy reconstruction')
    need(count == m['category_counts'] and first == m['scheduled_first_counts'], 'H3 category or generation order mismatch')
    need(m['fixed_order'] == 'title.casefold(), title, document_key', 'fixed-order protocol mismatch')
    return (root, out, m, plans, h1, changed, jobs)

def attempts(out, jobs):
    return qa.load_attempts(_pkg_resolve(out / 'responses.jsonl'), {(j['qid'], j['cond']): j for j in jobs})

def status(a):
    (_, out, m, plans, h1, changed, jobs) = verify(a)
    (done, _) = attempts(out, jobs)
    print('STATUS H3-v2 fresh completed', len(done), '/', len(jobs), 'remaining', len(jobs) - len(done))
    print('Changed-set QIDs', len(changed), ' H2 reused responses 0')
    if _pkg_resolve(out / 'score').exists():
        verify_score(a)

def generate(a):
    (_, out, m, plans, h1, changed, jobs) = verify(a)
    need(not _pkg_resolve(out / 'score').exists(), 'H3 already scored; forbid further generation')
    need(1 <= a.workers <= 32 and 1 <= a.timeout <= 1200 and (0 <= a.retries <= 10), 'workers/timeout/retries invalid')
    (done, _) = attempts(out, jobs)
    todo = [j for j in jobs if (j['qid'], j['cond']) not in done]
    print('GENERATE remaining', len(todo), '/', len(jobs), flush=True)
    if not todo:
        print('GENERATE PASS (366 fresh jobs, or requested job count, already complete)')
        return

    def infer(job):
        cfg = {**m['config'], 'url': m['url']}
        error = ''
        for attempt in range(a.retries + 1):
            try:
                (pred, served) = qa.api_call(job, cfg, a.timeout)
                return {'qid': job['qid'], 'cond': job['cond'], 'prompt_sha256': job['prompt_sha256'], 'pred': pred, 'status': 'ok', 'served_model': served, 'err': ''}
            except Exception as exc:
                error = f'{type(exc).__name__}: {str(exc)[:220]}'
                if attempt < a.retries:
                    time.sleep(min(2 * (attempt + 1), 10))
        return {'qid': job['qid'], 'cond': job['cond'], 'prompt_sha256': job['prompt_sha256'], 'pred': '', 'status': 'error', 'served_model': '', 'err': error}
    with _pkg_pathlib.Path(_pkg_open_path(_pkg_resolve(out / 'responses.jsonl'), 'a')).open('a', encoding='utf-8') as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
        for (i, rec) in enumerate(pool.map(infer, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + '\n')
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print('GENERATE progress', i, '/', len(todo), rec['status'], flush=True)
    (done, _) = attempts(out, jobs)
    print('GENERATE FINISHED', len(done), '/', len(jobs), 'remaining', len(jobs) - len(done))
    need(len(done) == len(jobs), 'incomplete answers; rerun generate to resume')

def boot(deltas, clusters, reps, seed):
    import numpy as np
    vals = np.asarray(deltas, dtype=float)
    need(len(vals) == len(clusters) and len(vals) > 0, 'empty mismatched bootstrap')
    groups = collections.defaultdict(list)
    for (i, c) in enumerate(clusters):
        groups[c].append(i)
    sums = np.array([vals[idx].sum() for idx in groups.values()])
    sizes = np.array([len(idx) for idx in groups.values()])
    rng = np.random.default_rng(seed)
    draws = []
    n_clusters = len(sizes)
    for s in range(0, reps, 128):
        batch = min(128, reps - s)
        ind = rng.integers(n_clusters, size=(batch, n_clusters))
        draws.extend(_pkg_resolve(sums[ind].sum(axis=1) / sizes[ind].sum(axis=1)).tolist())
    (lo, hi) = np.quantile(draws, [0.025, 0.975]).tolist()
    return {'delta': float(vals.mean()), 'ci95': [float(lo), float(hi)], 'n': len(vals), 'clusters': n_clusters, 'reps': reps, 'method': 'paired gold-title-pair cluster bootstrap, percentile'}

def validated_gold(root, plans, h1):
    gold = {}
    types = {}
    gold_cluster = {}
    for row in qa.readjl(_pkg_resolve(root / 'data/hotpotqa/hotpot_dev_distractor.jsonl')):
        qid = str(row.get('_id', row.get('id')))
        if qid not in plans:
            continue
        need(qid not in gold, f'official DEV QID duplicate: {qid}')
        answer = row.get('answer')
        need(isinstance(answer, str) and answer.strip(), f'invalid DEV answer: {qid}')
        typ = row.get('type')
        need(typ in ('bridge', 'comparison'), f'invalid DEV question type: {qid}')
        sf = row.get('supporting_facts')
        need(isinstance(sf, dict) and isinstance(sf.get('title'), list), f'official DEV supporting title missing: {qid}')
        titles = sf['title']
        need(titles and all((isinstance(x, str) and x for x in titles)), f'official DEV supporting titles invalid: {qid}')
        ck = json.dumps(sorted(set(titles)), ensure_ascii=False, separators=(',', ':'))
        need(h1[qid]['cluster'] == ck, f'H1 Gold-title-pair cluster drift: {qid}')
        need(h1[qid]['type'] == typ, f'H1 question type drift: {qid}')
        gold[qid] = [answer]
        types[qid] = typ
        gold_cluster[qid] = ck
    need(set(gold) == set(plans), f'official DEV coverage missing {len(set(plans) - set(gold))}')
    return (gold, types, gold_cluster)

def read_h2_items(path, plans, types, clusters, h2report):
    with path.open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    need(len(records) == len(plans) == N, 'H2 scored record count mismatch')
    h2 = {}
    for row in records:
        qid = row['qid']
        need(qid in plans and qid not in h2, f'H2 duplicate/foreign QID: {qid}')
        need(row['type'] == types[qid] and row['cluster'] == clusters[qid], f'H2 score type/cluster drift: {qid}')
        for pol in POLICIES:
            for metric in METRICS:
                z = float(row[pol + '_' + metric])
                need(math.isfinite(z) and 0 <= z <= 1, f'H2 metric invalid: {qid} {pol} {metric}')
        h2[qid] = row
    need(set(h2) == set(plans), 'H2 score QID set mismatch')
    for metric in METRICS:
        obs = _pkg_resolve(sum((float(h2[q]['G2_' + metric]) - float(h2[q]['G0_' + metric]) for q in h2)) / N)
        baseline = h2report['overall']['pair_delta'][metric]['delta']
        need(abs(obs - baseline) < 1e-09, f'H2 per-QID versus published report policy delta mismatch: {metric}')
    return h2

def joint_boot(group, bootstrap, seed):
    import numpy as np
    need(group and bootstrap >= 1000, 'invalid paired bootstrap sample')
    names = [('H3', k) for k in METRICS] + [('H2', k) for k in METRICS] + [('H3_minus_H2', k) for k in METRICS]
    X = np.asarray([[r[tag + '_' + key + '_delta'] for (tag, key) in names] for r in group], dtype=np.float64)
    need(np.isfinite(X).all(), 'nonfinite QID metrics')
    cs = collections.defaultdict(list)
    for (i, r) in enumerate(group):
        cs[r['cluster']].append(i)
    sums = np.asarray([X[ix, :].sum(axis=0) for ix in cs.values()])
    sizes = np.asarray([len(ix) for ix in cs.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty((bootstrap, len(names)), dtype=np.float64)
    for begin in range(0, bootstrap, 64):
        count = min(64, bootstrap - begin)
        ids = rng.integers(0, len(sizes), size=(count, len(sizes)))
        draws[begin:begin + count, :] = _pkg_resolve(sums[ids, :].sum(axis=1) / sizes[ids].sum(axis=1)[:, None])
    out = {k: {} for k in ('H3', 'H2', 'H3_minus_H2')}
    for (i, (tag, metric)) in enumerate(names):
        ci = np.quantile(draws[:, i], [0.025, 0.975]).tolist()
        out[tag][metric] = {'delta': float(X[:, i].mean()), 'ci95': ci, 'n': len(group), 'clusters': len(cs), 'reps': bootstrap, 'method': 'paired gold-title-pair cluster percentile bootstrap'}
    return out

def score(a):
    (root, out, m, plans, h1, changed, jobs) = verify(a)
    need(not _pkg_resolve(out / 'score').exists(), 'score output exists; refuse overwrite')
    (done, _) = attempts(out, jobs)
    need(len(done) == len(jobs), f'incomplete fresh generation: {len(done)}/{len(jobs)}')
    need(len(jobs) == 2 * len(changed), 'fresh full G0/G2 pairing not satisfied')
    response_hash = qa.sha(_pkg_resolve(out / 'responses.jsonl'))
    (gold, types, clusters) = validated_gold(root, plans, h1)
    h2report = read_json(_pkg_resolve(_pkg_resolve(root / a.h2) / 'score/report.json'))
    h2 = read_h2_items(_pkg_resolve(_pkg_resolve(root / a.h2) / 'score/items.csv'), plans, types, clusters, h2report)
    records = []
    for qid in sorted(plans):
        cat = partition(plans[qid])
        r = {'qid': qid, 'type': types[qid], 'category': cat, 'cluster': clusters[qid]}
        old = h2[qid]
        for metric in METRICS:
            h2delta = float(old['G2_' + metric]) - float(old['G0_' + metric])
            r['H2_' + metric + '_delta'] = h2delta
        if cat == 'document_set_changed':
            need((qid, 'G0') in done and (qid, 'G2') in done, f'fresh changed-set pair incomplete: {qid}')
            for pol in POLICIES:
                pred = done[qid, pol]['pred']
                e = qa.em(pred, gold[qid])
                f = qa.f1(pred, gold[qid])
                ans = int(not qa.abstain(pred))
                r.update({pol + '_em': e, pol + '_f1': f, pol + '_answered': ans, pol + '_wrong': int(ans and (not e)), pol + '_pred': pred})
        else:
            for pol in POLICIES:
                for metric in METRICS:
                    r[pol + '_' + metric] = ''
                r[pol + '_pred'] = ''
        for metric in METRICS:
            h3delta = r['G2_' + metric] - r['G0_' + metric] if cat == 'document_set_changed' else 0.0
            r['H3_' + metric + '_delta'] = h3delta
            r['H3_minus_H2_' + metric + '_delta'] = h3delta - r['H2_' + metric + '_delta']
        records.append(r)

    def summary(group):
        v = joint_boot(group, a.bootstrap, a.seed)
        ch = [r for r in group if r['category'] == 'document_set_changed']
        return {'n': len(group), 'changed_n': len(ch), 'paired_delta': v['H3'], 'H2_original_paired_delta': v['H2'], 'change_in_policy_contrast': v['H3_minus_H2'], 'changed_only_conditions': {pol: {k: _pkg_resolve(sum((float(x[pol + '_' + k]) for x in ch)) / len(ch)) for k in METRICS} for pol in POLICIES} if ch else None}
    summaries = {'overall': summary(records), 'by_type': {t: summary([r for r in records if r['type'] == t]) for t in ('bridge', 'comparison')}, 'changed_only': summary([r for r in records if r['category'] == 'document_set_changed'])}
    report = {'experiment': 'POST-18D-H3-v2 Fixed-Order Fresh-Paired QA', 'status': 'posthoc_mechanistic_control', 'n': N, 'category_counts': m['category_counts'], 'n_new_jobs': len(jobs), 'n_reused_exact_prompt': 0, 'fixed_order': m['fixed_order'], 'model': m['config']['model'], 'config': m['config'], 'H3': summaries, 'qa_responses_sha256': response_hash, 'source_hashes': m['source_hashes'], 'jobs_sha256': m['files_sha256']['jobs.jsonl'], 'limitations': m['limitations']}
    tmp = Path(_pkg_resolve(tempfile.mkdtemp(prefix='.h3v2_score_', dir=out)))
    try:
        with _pkg_pathlib.Path(_pkg_open_path(_pkg_resolve(tmp / 'items.csv'), 'x')).open('x', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        report['items_sha256'] = qa.sha(_pkg_resolve(tmp / 'items.csv'))
        dump_json(_pkg_resolve(tmp / 'report.json'), report)
        lines = ['# POST-18D-H3 v2: Fixed-order fresh-paired QA', 'Gold-Blind lexical order: title.casefold(), title, document_key', f'QIDs: {N}, changed sets: {len(changed)}, fresh responses: {len(jobs)}, reused H2 responses: 0', 'Same-set 4222 QIDs have zero paired delta; their absolute EM/F1 were NOT generated.', 'Post-hoc control; wrong = answered AND EM=0, not semantic hallucination.', '', '|Group|N|Changed|H3 ΔEM [95% CI]|H3 ΔF1 [95% CI]|H3 ΔAnswered|H3 ΔWrong|', '|---|---:|---:|---|---|---:|---:|']
        for (name, s) in [('ALL', summaries['overall']), ('bridge', summaries['by_type']['bridge']), ('comparison', summaries['by_type']['comparison']), ('changed-only', summaries['changed_only'])]:
            e = s['paired_delta']['em']
            f = s['paired_delta']['f1']
            lines.append(f"|{name}|{s['n']}|{s['changed_n']}|{e['delta']:+.5f} [{e['ci95'][0]:+.5f},{e['ci95'][1]:+.5f}]|{f['delta']:+.5f} [{f['ci95'][0]:+.5f},{f['ci95'][1]:+.5f}]|{s['paired_delta']['answered']['delta']:+.5f}|{s['paired_delta']['wrong']['delta']:+.5f}|")
        lines += ['', '## Paired change in policy contrast (H3 - H2, SAME QIDs and cluster draws)', '|Group|Metric|H2 Δ|H3 Δ|H3 - H2 [95% CI]|', '|---|---|---:|---:|---|']
        for (name, s) in [('ALL', summaries['overall']), ('bridge', summaries['by_type']['bridge']), ('comparison', summaries['by_type']['comparison']), ('changed-only', summaries['changed_only'])]:
            for k in METRICS:
                x = s['change_in_policy_contrast'][k]
                lines.append(f"|{name}|{k}|{s['H2_original_paired_delta'][k]['delta']:+.6f}|{s['paired_delta'][k]['delta']:+.6f}|{x['delta']:+.6f} [{x['ci95'][0]:+.6f},{x['ci95'][1]:+.6f}]|")
        lines += ['', 'H3-H2 is a paired change of observed policy contrasts, **not** a pure causal order effect.', 'Shared-document QIDs have no H3 absolute QA outcomes and should not be assigned condition means.', 'Gold titles and clusters used ONLY for scoring/audit, not document ordering.', 'Generated batches can still differ in model/server conditions despite temperature 0.']
        _pkg_output(_pkg_resolve(tmp / 'report.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        dump_json(_pkg_resolve(tmp / 'score_manifest.json'), {'items_sha256': report['items_sha256'], 'report_sha256': qa.sha(_pkg_resolve(tmp / 'report.json')), 'report_md_sha256': qa.sha(_pkg_resolve(tmp / 'report.md')), 'responses_sha256': response_hash, 'script_sha256': qa.sha(Path(_pkg_resolve(__file__)).resolve()), 'source_hashes': m['source_hashes']})
        tmp.rename(_pkg_resolve(out / 'score'))
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(_pkg_resolve(out / 'score/report.md').read_text(encoding='utf-8'))
    print('SCORE PASS:', _pkg_resolve(out / 'score'))

def verify_score(a):
    (root, out, m, plans, h1, changed, jobs) = verify(a)
    sp = _pkg_resolve(out / 'score')
    need(sp.is_dir(), 'H3-v2 score not present')
    rep = read_json(_pkg_resolve(sp / 'report.json'))
    ms = read_json(_pkg_resolve(sp / 'score_manifest.json'))
    need(rep['qa_responses_sha256'] == ms['responses_sha256'] == qa.sha(_pkg_resolve(out / 'responses.jsonl')), 'scored H3 response history modified after scoring')
    need(ms['script_sha256'] == qa.sha(Path(_pkg_resolve(__file__)).resolve()), 'H3 score source modified')
    need(rep['items_sha256'] == ms['items_sha256'] == qa.sha(_pkg_resolve(sp / 'items.csv')), 'scored items modified')
    need(ms['report_sha256'] == qa.sha(_pkg_resolve(sp / 'report.json')) and ms['report_md_sha256'] == qa.sha(_pkg_resolve(sp / 'report.md')), 'H3 report modified')
    need(ms['source_hashes'] == rep['source_hashes'] == m['source_hashes'], 'H3 scored source hash mismatch')
    need(rep['n'] == N and rep['n_new_jobs'] == len(jobs) and (rep['n_reused_exact_prompt'] == 0), 'score metadata mismatch')
    print('VERIFY-SCORE PASS: all frozen H1/H2 refs, fresh response sha and report hashes')

def selftest():
    docs = {'z': {'title': 'Beta', 'text': 'Statement Beta'}, 'a': {'title': 'Alpha', 'text': 'Statement Alpha'}, 'y': {'title': 'gamma', 'text': 'Statement Gamma'}}
    assert stable_keys(['z', 'a'], docs) == ['a', 'z']
    assert stable_keys(['a', 'z'], docs) == ['a', 'z']
    p = {'qid': 'X', 'G0_keys': ['a', 'z'], 'G2_keys': ['z', 'a']}
    assert partition(p) == 'order_only'
    p2 = {'qid': 'X', 'G0_keys': ['a', 'z'], 'G2_keys': ['y', 'a']}
    assert partition(p2) == 'document_set_changed'
    assert partition({'qid': 'X', 'G0_keys': ['a', 'z'], 'G2_keys': ['a', 'z']}) == 'identical'
    assert qa.prompt('What?', stable_keys(p2['G0_keys'], docs), docs) != qa.prompt('What?', stable_keys(p2['G2_keys'], docs), docs)
    import numpy as np
    rows = []
    for (qid, h2, h3) in [('a', 1, 0), ('b', 0, 1), ('c', 0, 0), ('d', 1, 1)]:
        r = {'cluster': qid}
        for k in METRICS:
            r['H2_' + k + '_delta'] = h2
            r['H3_' + k + '_delta'] = h3
            r['H3_minus_H2_' + k + '_delta'] = h3 - h2
        rows.append(r)
    out = joint_boot(rows, 1000, 0)
    assert abs(out['H3_minus_H2']['em']['delta']) < 1e-12
    assert out['H3']['em']['n'] == 4 and out['H3']['em']['clusters'] == 4
    assert qa.em('Paris', ['Paris']) == 1 and qa.abstain("I don't know")
    with tempfile.TemporaryDirectory() as temp:
        rp = Path(_pkg_resolve(temp))
        _pkg_resolve(rp / 'data/hotpotqa').mkdir(parents=True)
        _pkg_output(_pkg_resolve(rp / 'data/hotpotqa/hotpot_dev_distractor.jsonl')).write_text(json.dumps({'_id': 'a', 'type': 'comparison', 'answer': 'Paris', 'supporting_facts': {'title': ['B', 'A', 'B']}}) + '\n', encoding='utf-8')
        expected = json.dumps(['A', 'B'], ensure_ascii=False, separators=(',', ':'))
        assert validated_gold(rp, {'a': {}}, {'a': {'cluster': expected, 'type': 'comparison'}})[2]['a'] == expected
        try:
            validated_gold(rp, {'a': {}}, {'a': {'cluster': 'bad', 'type': 'comparison'}})
        except RuntimeError as e:
            assert 'cluster drift' in str(e)
        else:
            raise AssertionError('bad cluster must fail')
    print('SELF-TEST PASS: fixed order, fresh G0/G2 pairs, Gold cluster drift, paired bootstrap')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['self-test', 'prepare', 'status', 'generate', 'score', 'verify-score'])
    ap.add_argument('--root', default='.')
    ap.add_argument('--stage1', default='out/p18d_g2_holdout_v1')
    ap.add_argument('--h2', default='out/p18d_g2_qa_v3')
    ap.add_argument('--output', default='out/p18d_h3_fixed_order_v2')
    ap.add_argument('--api-url', default='', help='One trusted OpenAI-compatible chat-completions URL')
    ap.add_argument('--expect-changed', type=int, default=183)
    ap.add_argument('--expect-order-only', type=int, default=855)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=90)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    need(a.bootstrap >= 1000, 'bootstrap <1000')
    if a.command == 'self-test':
        selftest()
    elif a.command == 'prepare':
        prepare(a)
    elif a.command == 'status':
        status(a)
    elif a.command == 'generate':
        generate(a)
    elif a.command == 'score':
        score(a)
    else:
        verify_score(a)
if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyError, AssertionError, FileNotFoundError, ImportError) as ex:
        print('FAIL:', ex, file=sys.stderr)
        sys.exit(2)
