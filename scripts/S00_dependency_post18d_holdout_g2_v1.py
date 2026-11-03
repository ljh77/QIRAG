"""POST-18D follow-up A: read-only frozen 4,405-QID G0 vs G2 evaluation.

Commands: audit, prepare, score, self-test.  Run in ~/2610-qi-rag.
Requires local qi_common.py, retrieval_policy.py, p18d_goldblind_ablation_v11.py.
G2 is EXACT v11 G2 (v5 D1) reciprocal-rank vote. Do not retune.
The program aborts on overlap with the 2k exploratory QIDs by default;
--allow-overlap marks findings as NONINDEPENDENT and cannot be used to claim
independent confirmation. Pre-existing exposure to other holdout analyses also
limits confirmatory interpretation even where QIDs are disjoint.

Selection is written to plan.jsonl BEFORE any gold labels are loaded. The score
step is a separate process and reads gold only after checking the frozen plan.
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse, collections, csv, hashlib, json, os, sys, tempfile
from pathlib import Path
N = 4405
CORPUS_SHA = '4976962eacc1d7a1e8398657939729795a252e4f27e44f2693b3cc13709e5872'
PRED_SHA = '4592253a0c4c46711e3525f38a51f89cf64b56e181986b9ce2acdafc9f6cfff7'
EXP_DATA_SHA = '86558b9e2e8b94eb6702b3a84e6a18188cf92e9d62097e03c20090bd1ff7167e'

def need(cond, msg):
    if not cond:
        raise RuntimeError(msg)

def sha(p):
    h = hashlib.sha256()
    with open(_pkg_open_path(_pkg_resolve(p), 'rb'), 'rb') as f:
        for block in iter(lambda : f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def readjl(p):
    with open(_pkg_open_path(_pkg_resolve(p), 'r'), encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def dumpjl(p, rows):
    with open(_pkg_open_path(_pkg_resolve(p), 'x'), 'x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')

def dump(p, obj):
    with open(_pkg_open_path(_pkg_resolve(p), 'x'), 'x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')

def load_unique(p, key):
    result = {}
    for r in readjl(p):
        k = str(r[key])
        need(k not in result, f'duplicate {key}={k} in {p}')
        result[k] = r
    return result

def files(root):
    return {'index': _pkg_resolve(root / 'data/_split/index_items.jsonl'), 'eval': _pkg_resolve(root / 'data/_split/eval_queries.jsonl'), 'dev': _pkg_resolve(root / 'data/hotpotqa/hotpot_dev_distractor.jsonl'), 'pred': _pkg_resolve(root / 'out/holdout_locked/retrieval_predictions.jsonl'), 'split': _pkg_resolve(root / 'out/holdout_locked/split_ids.json'), 'freeze': _pkg_resolve(root / 'out/holdout_locked/FREEZE.json'), 'retman': _pkg_resolve(root / 'out/holdout_locked/retrieval_manifest.json'), 'explore': _pkg_resolve(root / 'out/p18_evidence_qa_v1/dataset.jsonl'), 'v11': _pkg_resolve(root / 'p18d_goldblind_ablation_v11.py')}

def validate_sources(root, allow_overlap=False):
    from S00_qi_common import build_corpus, corpus_fingerprint, para_key, split_paragraphs
    from S00_retrieval_policy import qi_policy
    paths = files(root)
    for (name, p) in paths.items():
        need(p.is_file(), f'missing {name}: {p}')
    freeze = json.loads(paths['freeze'].read_text(encoding='utf-8'))
    split = json.loads(paths['split'].read_text(encoding='utf-8'))
    man = json.loads(paths['retman'].read_text(encoding='utf-8'))
    need(sha(paths['split']) == freeze['split_ids_sha256'], 'split hash != FREEZE')
    need(sha(paths['pred']) == man['predictions_sha256'], 'pred hash != retrieval manifest')
    need(sha(paths['pred']) == PRED_SHA, 'pred hash != known E1-A frozen reference')
    need(man.get('split_ids_sha256') == freeze['split_ids_sha256'], 'retrieval manifest split mismatch')
    need(man.get('corpus_sha256') == CORPUS_SHA, 'retrieval manifest corpus mismatch')
    if 'corpus_sha256' in freeze:
        need(freeze['corpus_sha256'] == CORPUS_SHA, 'FREEZE corpus mismatch')
    if 'policy' in man:
        need(man['policy'].get('P') == 2 and man['policy'].get('top_items') == 8, 'retrieval manifest P/top_items not 2/8')
    want = list(map(str, split['holdout']))
    need(len(want) == N and len(set(want)) == N, 'holdout count/uniqueness mismatch')
    calib = set(map(str, split.get('calibration', [])))
    need(not calib & set(want), 'calibration/holdout overlap')
    pred = load_unique(paths['pred'], 'qid')
    need(set(pred) == set(want), 'prediction holdout QID mismatch')
    need(sha(paths['explore']) == EXP_DATA_SHA, '2k exploratory dataset freeze SHA mismatch')
    explored = load_unique(paths['explore'], 'qid')
    need(len(explored) == 2000, '2k exploratory dataset size mismatch')
    overlap = sorted(set(pred) & set(explored))
    if overlap and (not allow_overlap):
        raise RuntimeError(f'EXPLORATORY/HOLDOUT QID OVERLAP: {len(overlap)}; first={overlap[:5]}. Do not claim independent confirmation. Use --allow-overlap ONLY for labeled nonindependent analysis.')
    idx = list(readjl(paths['index']))
    docs = build_corpus(idx)
    need(corpus_fingerprint(docs) == CORPUS_SHA, 'recomputed corpus SHA mismatch')
    docs_by_key = {para_key(d): d for d in docs}
    need(len(docs_by_key) == len(docs), 'document-key collision in frozen corpus')
    itemkeys = [[para_key(d) for d in split_paragraphs(r.get('context', ''))] for r in idx]
    need(len(idx) > max((r['qi_top_items'][0]['item'] for r in pred.values())), 'index too small')
    for (qid, r) in pred.items():
        ranking = r['qi_top_items']
        need(len(ranking) == 8, f'Top8 length mismatch: {qid}')
        seen = set()
        prev = float('inf')
        for x in ranking:
            j = x['item']
            score = x['score']
            need(isinstance(j, int) and (not isinstance(j, bool)) and (0 <= j < len(itemkeys)), f'item ID invalid: {qid}')
            need(j not in seen, f'item duplicate: {qid}')
            seen.add(j)
            need(isinstance(score, (int, float)) and (not isinstance(score, bool)) and __import__('math').isfinite(score), f'score invalid: {qid}')
            need(prev + 1e-08 >= score, f'score order mismatch: {qid}')
            prev = score
            need(x['titles'] == [docs_by_key[k]['title'] for k in itemkeys[j]], f'candidate titles mismatch: {qid}')
        (g0, short) = qi_policy([x['item'] for x in ranking], itemkeys, 2)
        need([docs_by_key[k]['title'] for k in g0] == r['qi_titles'] and bool(short) == bool(r['qi_short']), f'G0 E1-A frozen mismatch: {qid}')
    scope = 'nonindependent_overlap' if overlap else 'qid_disjoint_posthoc_validation'
    return {'paths': paths, 'pred': pred, 'want': want, 'idx': idx, 'itemkeys': itemkeys, 'docs_by_key': docs_by_key, 'overlap': overlap, 'scope': scope}

def choose(r, itemkeys, docmap):
    from S00_dependency_p18d_goldblind_ablation_v11 import choose_policies
    ranked = [(x['item'], float(x['score'])) for x in r['qi_top_items']]
    p = choose_policies(ranked, itemkeys, p=2, score_drop=0.05)
    (g0, g2) = (p['G0'], p['G2'])
    need([docmap[k]['title'] for k in g0] == r['qi_titles'], 'v11 G0 mismatch')
    from collections import defaultdict
    pool = []
    seen = set()
    vote = defaultdict(float)
    for (rank, (j, _)) in enumerate(ranked, 1):
        for k in dict.fromkeys(itemkeys[j]):
            if k not in seen:
                pool.append(k)
                seen.add(k)
            vote[k] += _pkg_resolve(1.0 / rank)
    direct = sorted(pool, key=lambda k: (-vote[k], pool.index(k)))[:2]
    need(direct == g2, 'G2 independent-vote implementation mismatch')
    return (g0, g2)

def commands_prepare(args):
    root = Path(_pkg_resolve(args.root)).resolve()
    v = validate_sources(root, args.allow_overlap)
    output = _pkg_resolve(root / args.output).resolve()
    need(not output.exists(), 'output exists; refuse overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    question = {}
    for d in readjl(v['paths']['dev']):
        qid = str(d.get('_id', d.get('id')))
        if qid in v['pred']:
            need(qid not in question, f'duplicate DEV question: {qid}')
            question[qid] = d['question']
    need(set(question) == set(v['pred']), 'DEV question QID set mismatch')
    rows = []
    for qid in sorted(v['pred']):
        (a, b) = choose(v['pred'][qid], v['itemkeys'], v['docs_by_key'])
        rows.append({'qid': qid, 'question': question[qid], 'G0_keys': a, 'G2_keys': b, 'G0_titles': [v['docs_by_key'][k]['title'] for k in a], 'G2_titles': [v['docs_by_key'][k]['title'] for k in b], 'changed_set': int(set(a) != set(b)), 'changed_order': int(a != b)})
    tmp = Path(_pkg_resolve(tempfile.mkdtemp(prefix='.g2_plan_stage_', dir=output.parent)))
    try:
        dumpjl(_pkg_resolve(tmp / 'plan.jsonl'), rows)
        m = {'n': N, 'P': 2, 'top_items': 8, 'policies': {'G0': 'frozen QI top-question-first', 'G2': 'v11 reciprocal-rank vote'}, 'scope': v['scope'], 'overlap_n': len(v['overlap']), 'overlap_examples': v['overlap'][:20], 'caveat': 'Previously analyzed 4,405-QID holdout: post-hoc validation, not new pristine preregistered holdout.', 'inputs_sha256': {k: sha(p) for (k, p) in v['paths'].items() if k in ('index', 'eval', 'dev', 'pred', 'split', 'freeze', 'retman', 'explore', 'v11')}, 'script_sha256': sha(Path(_pkg_resolve(__file__)).resolve()), 'plan_sha256': sha(_pkg_resolve(tmp / 'plan.jsonl')), 'changed_sets': sum((r['changed_set'] for r in rows)), 'changed_order': sum((r['changed_order'] for r in rows))}
        dump(_pkg_resolve(tmp / 'manifest.json'), m)
        tmp.rename(output)
    finally:
        if tmp.exists():
            __import__('shutil').rmtree(tmp)
    print('PREPARE PASS', output, 'changed_sets', m['changed_sets'], 'exploratory_overlap', m['overlap_n'], 'scope', m['scope'])

def boot(a, b, clusters, reps, seed):
    import numpy as np
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    need(len(a) == len(b) == len(clusters) and len(a) > 0, 'invalid boot lengths')
    groups = collections.defaultdict(list)
    for (i, c) in enumerate(clusters):
        groups[c].append(i)
    delta = a - b
    sums = np.array([delta[ii].sum() for ii in groups.values()])
    sizes = np.array([len(ii) for ii in groups.values()])
    rng = np.random.default_rng(seed)
    samples = np.empty(reps)
    for s in range(0, reps, 128):
        count = min(128, reps - s)
        d = rng.integers(len(sizes), size=(count, len(sizes)))
        samples[s:s + count] = _pkg_resolve(sums[d].sum(axis=1) / sizes[d].sum(axis=1))
    return {'delta': float(delta.mean()), 'ci95': np.quantile(samples, [0.025, 0.975]).tolist(), 'clusters': len(sizes), 'n': len(delta), 'reps': reps}

def verify_plan(args):
    root = Path(_pkg_resolve(args.root)).resolve()
    out = _pkg_resolve(root / args.output)
    m = json.loads(_pkg_resolve(out / 'manifest.json').read_text(encoding='utf-8'))
    need(m['script_sha256'] == sha(Path(_pkg_resolve(__file__)).resolve()), 'script changed since prepare')
    need(m['plan_sha256'] == sha(_pkg_resolve(out / 'plan.jsonl')), 'plan changed')
    for (k, h) in m['inputs_sha256'].items():
        need(sha(files(root)[k]) == h, f'frozen input modified: {k}')
    v = validate_sources(root, args.allow_overlap)
    need(v['scope'] == m['scope'] and len(v['overlap']) == m['overlap_n'], 'overlap audit changed')
    plans = load_unique(_pkg_resolve(out / 'plan.jsonl'), 'qid')
    need(set(plans) == set(v['pred']), 'plan QID set mismatch')
    for (qid, r) in plans.items():
        (a, b) = choose(v['pred'][qid], v['itemkeys'], v['docs_by_key'])
        need(a == r['G0_keys'] and b == r['G2_keys'], f'policy selection drift: {qid}')
    return (v, plans, m, out)

def commands_score(args):
    (v, plans, m, out) = verify_plan(args)
    dest = _pkg_resolve(out / 'score')
    need(not dest.exists(), 'score directory already exists; refuse overwrite')
    golds = load_unique(v['paths']['eval'], 'qid')
    need(set(plans).issubset(golds), 'missing gold eval QIDs')
    rows = []
    for (qid, p) in sorted(plans.items()):
        lab = golds[qid]
        g = set(lab.get('gold_titles', []))
        need(len(g) == 2, f'gold titles must contain 2 distinct items: {qid}')
        typ = lab.get('type')
        need(typ in ('bridge', 'comparison'), f'unknown question type: {qid}: {typ}')
        r = {'qid': qid, 'type': typ, 'cluster': json.dumps(sorted(g), ensure_ascii=False, separators=(',', ':')), 'changed_set': p['changed_set'], 'changed_order': p['changed_order']}
        for name in ('G0', 'G2'):
            hit = len(set(p[name + '_titles']) & g)
            r[name + '_hits'] = hit
            r[name + '_recall'] = _pkg_resolve(hit / 2)
            r[name + '_both'] = int(hit == 2)
        rows.append(r)
    need(len(rows) == N, 'wrong score n')
    needs = {'bridge': 3518, 'comparison': 887}
    got = collections.Counter((r['type'] for r in rows))
    need(dict(got) == needs, f'Holdout type counts differ from original E1-A expected {needs}: {got}')

    def report(group):
        return {'n': len(group), 'G0_recall': _pkg_resolve(sum((x['G0_recall'] for x in group)) / len(group)), 'G2_recall': _pkg_resolve(sum((x['G2_recall'] for x in group)) / len(group)), 'G0_both': _pkg_resolve(sum((x['G0_both'] for x in group)) / len(group)), 'G2_both': _pkg_resolve(sum((x['G2_both'] for x in group)) / len(group)), 'changed_n': sum((x['changed_set'] for x in group)), 'recall_pair_cluster': boot([x['G2_recall'] for x in group], [x['G0_recall'] for x in group], [x['cluster'] for x in group], args.bootstrap, args.seed), 'both_pair_cluster': boot([x['G2_both'] for x in group], [x['G0_both'] for x in group], [x['cluster'] for x in group], args.bootstrap, args.seed)}
    result = {'scope': m['scope'], 'overlap_n': m['overlap_n'], 'n': N, 'overall': report(rows), 'by_type': {k: report([r for r in rows if r['type'] == k]) for k in ('bridge', 'comparison')}, 'multiple_tests': 'Exploratory post-hoc estimates; CIs descriptive and unadjusted; preregistration not claimed.', 'retrieval_source_sha256': sha(v['paths']['pred']), 'plan_sha256': m['plan_sha256']}
    tmp = Path(_pkg_resolve(tempfile.mkdtemp(prefix='.g2_score_stage_', dir=out)))
    try:
        with _pkg_pathlib.Path(_pkg_open_path(_pkg_resolve(tmp / 'items.csv'), 'x')).open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        dump(_pkg_resolve(tmp / 'report.json'), result)
        lines = ['# POST-18D frozen G2 holdout validation', '', f"Scope: {m['scope']}; overlap={m['overlap_n']}", '|Group|N|G0 Recall|G2 Recall|G2-G0|95% cluster CI|Both-Gold G0/G2|', '|---|---:|---:|---:|---:|---|---|']
        for (label, z) in [('ALL', result['overall'])] + list(result['by_type'].items()):
            b = z['recall_pair_cluster']
            (lo, hi) = b['ci95']
            lines.append(f"|{label}|{z['n']}|{z['G0_recall']:.4f}|{z['G2_recall']:.4f}|{b['delta']:+.4f}|[{lo:+.4f},{hi:+.4f}]|{z['G0_both']:.4f}/{z['G2_both']:.4f}|")
        _pkg_output(_pkg_resolve(tmp / 'report.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        tmp.rename(dest)
    finally:
        if tmp.exists():
            __import__('shutil').rmtree(tmp)
    print(_pkg_resolve(dest / 'report.md').read_text(encoding='utf-8'))
    print('SCORE PASS', dest)

def test():
    import numpy as np
    from S00_dependency_p18d_goldblind_ablation_v11 import choose_policies
    links = [['A', 'B'], ['A', 'C'], ['C', 'D'], ['C', 'D']]
    rank = [(0, 0.99), (1, 0.98), (2, 0.97), (3, 0.96)]
    x = choose_policies(rank, links, 2)['G2']
    assert x == ['A', 'C'], x
    b = boot([1, 0, 1], [0, 0, 1], ['x', 'y', 'y'], 1000, 0)
    assert abs(b['delta'] - _pkg_resolve(1 / 3)) < 1e-09 and b['n'] == 3
    assert len(json.dumps(sorted(['A | B', 'C']))) > 0
    print('SELF-TEST PASS: voting, paired cluster stats, serialization')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['audit', 'prepare', 'score', 'self-test'])
    ap.add_argument('--root', default='.')
    ap.add_argument('--output', default='out/p18d_g2_holdout_v1')
    ap.add_argument('--allow-overlap', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    need(a.bootstrap >= 1000, 'bootstrap must be at least 1000')
    if a.command == 'self-test':
        test()
        return
    if a.command == 'audit':
        v = validate_sources(Path(_pkg_resolve(a.root)).resolve(), a.allow_overlap)
        print('AUDIT PASS n=', len(v['want']), 'overlap=', len(v['overlap']), 'scope=', v['scope'])
    elif a.command == 'prepare':
        commands_prepare(a)
    else:
        commands_score(a)
if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyError, AssertionError) as e:
        print('FAIL:', e, file=sys.stderr)
        sys.exit(2)
