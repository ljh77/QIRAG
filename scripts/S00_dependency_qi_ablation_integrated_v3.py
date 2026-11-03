"""QI-RAG frozen holdout ablation v3. Read-only inputs, no LLM calls.
ABL-01: score-ranked selective risk EXPLORATORY (not matched actual coverage).
ABL-02: paired 2x2 contrasts, interactions, transitions, cluster-aware CI.
ABL-03: retrieval evidence only with explicitly mapped document titles and gold supporting facts.
No confirmation of superiority without independent calibration and matching actual coverage.
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse, csv, hashlib, json, math, re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
CONDS = ('QI_STRICT', 'VAN_STRICT', 'QI_CAND', 'VAN_CAND', 'QA_STRICT')
PAIRS = (('QI_STRICT', 'VAN_STRICT'), ('QI_CAND', 'VAN_CAND'), ('QI_STRICT', 'QI_CAND'), ('VAN_STRICT', 'VAN_CAND'), ('QI_STRICT', 'QA_STRICT'))
ARTICLES = re.compile('\\b(a|an|the)\\b')
PUNCT = re.compile('[^\\w\\s]')
ABSTAIN = re.compile("\\b(?:i\\s+(?:do\\s+not|don't|dont)\\s+know|insufficient(?:\\s+information)?|cannot\\s+answer|not\\s+enough\\s+information)\\b", re.I)

def norm(s):
    return ' '.join(ARTICLES.sub(' ', PUNCT.sub(' ', str(s).lower())).split())

def abstain(s):
    s = str(s).strip()
    return not s or bool(ABSTAIN.search(s)) or norm(s) in {'unknown', 'none', 'n a', 'no answer'}

def em(p, g):
    return float(any((norm(p) == norm(x) for x in g)))

def f1(p, g):
    p = norm(p).split()
    out = []
    for x in g:
        a = norm(x).split()
        common = sum((Counter(p) & Counter(a)).values())
        out.append(_pkg_resolve(2 * common / (len(p) + len(a))) if p or a else 1.0)
    return max(out, default=0.0)

def read_rows(path):
    p = Path(_pkg_resolve(path))
    if not p.is_file():
        raise ValueError('Missing input: ' + str(p))
    if p.suffix == '.jsonl':
        with p.open(encoding='utf-8-sig') as f:
            for (i, line) in enumerate(f, 1):
                if line.strip():
                    try:
                        r = json.loads(line)
                    except Exception as e:
                        raise ValueError(f'{p}:{i}: {e}')
                    if not isinstance(r, dict):
                        raise ValueError(f'{p}:{i} not object')
                    yield r
    elif p.suffix == '.csv':
        with p.open(encoding='utf-8-sig', newline='') as f:
            yield from csv.DictReader(f)
    else:
        d = json.loads(p.read_text(encoding='utf-8-sig'))
        if isinstance(d, list):
            yield from d
        elif isinstance(d, dict):
            for k in ('data', 'questions', 'items', 'records'):
                if isinstance(d.get(k), list):
                    yield from d[k]
                    return
            for (k, v) in d.items():
                yield ({'qid': k, **v} if isinstance(v, dict) else {'qid': k, 'answer': v})
        else:
            raise ValueError('Unsupported JSON root')

def qid(r):
    for k in ('qid', '_id', 'id', 'question_id'):
        if r.get(k) is not None:
            return str(r[k])
    raise ValueError('Missing qid: ' + str(r)[:100])

def gold_answers(r):
    v = r.get('answer', r.get('answers', r.get('gold')))
    if isinstance(v, dict):
        v = v.get('text', v.get('answer'))
    if isinstance(v, list):
        v = [x.get('text', x.get('answer', '')) if isinstance(x, dict) else x for x in v]
    else:
        v = [v]
    ans = [str(x) for x in v if x is not None and str(x).strip()]
    if not ans:
        raise ValueError('Missing gold answer: ' + qid(r))
    return ans

def load_split(p):
    d = json.loads(Path(_pkg_resolve(p)).read_text(encoding='utf-8-sig'))
    ids = d['holdout'] if isinstance(d, dict) else d
    ids = [str(x) for x in ids]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate split IDs')
    return (ids, d)

def sha(p):
    h = hashlib.sha256()
    with _pkg_pathlib.Path(_pkg_open_path(Path(_pkg_resolve(p)), 'rb')).open('rb') as f:
        for chunk in iter(lambda : f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_gold(path, ids):
    wanted = set(ids)
    found = {}
    for r in read_rows(path):
        q = qid(r)
        if q not in wanted:
            continue
        if q in found:
            raise ValueError('Duplicate gold: ' + q)
        found[q] = r
    if set(found) != wanted:
        raise ValueError(f'Gold missing {len(wanted - set(found))} holdout QIDs')
    return found

def load_pred(path, ids, condition):
    wanted = set(ids)
    found = {}
    for r in read_rows(path):
        q = qid(r)
        if q not in wanted:
            raise ValueError(f'{condition}: unexpected QID {q}')
        if q in found:
            raise ValueError(f'{condition}: duplicate {q}')
        if r.get('cond', condition) != condition:
            raise ValueError(f'{condition}: condition mismatch {q}')
        if str(r.get('status', 'ok')).lower() not in ('ok', 'success', 'complete'):
            raise ValueError(f'{condition}: failed {q}')
        if not isinstance(r.get('pred'), str):
            raise ValueError(f'{condition}: invalid pred {q}')
        found[q] = r
    if set(found) != wanted:
        raise ValueError(f'{condition}: missing {len(wanted - set(found))} QIDs')
    return found

def arrays(ids, gold, pred):
    ans = np.array([not abstain(pred[q]['pred']) for q in ids], dtype=float)
    correct = np.array([em(pred[q]['pred'], gold_answers(gold[q])) if ans[i] else 0.0 for (i, q) in enumerate(ids)])
    ff = np.array([f1(pred[q]['pred'], gold_answers(gold[q])) if ans[i] else 0.0 for (i, q) in enumerate(ids)])
    return {'coverage': ans, 'em': correct, 'f1': ff, 'wrong_em_all': ans - correct}

def summarize(a, idx=None):
    b = {k: v if idx is None else v[idx] for (k, v) in a.items()}
    n = len(b['em'])
    answered = float(b['coverage'].sum())
    wrong = float(b['wrong_em_all'].sum())
    return {'n': n, 'em': float(b['em'].mean()), 'f1': float(b['f1'].mean()), 'coverage': _pkg_resolve(answered / n), 'wrong_em_all': _pkg_resolve(wrong / n), 'wrong_em_answered': _pkg_resolve(wrong / answered) if answered else None}

def supporting_titles(row):
    """HotpotQA JSONL: supporting_facts={title:[...],sent_id:[...]}; also accept paired lists."""
    sf = row.get('supporting_facts')
    if isinstance(sf, dict):
        titles = sf.get('title')
        if not isinstance(titles, list) or not titles or (not all((isinstance(t, str) and t.strip() for t in titles))):
            return None
        return sorted(set(titles))
    if isinstance(sf, list) and sf:
        titles = []
        for fact in sf:
            if not isinstance(fact, (list, tuple)) or not fact or (not isinstance(fact[0], str)):
                return None
            titles.append(fact[0])
        return sorted(set(titles))
    return None

def cluster_ids(ids, gold, mode):
    if mode == 'question':
        return (list(ids), 'question')
    labels = []
    for q in ids:
        titles = supporting_titles(gold[q])
        if not titles:
            return (list(ids), 'question_fallback_missing_supporting_facts')
        labels.append(' || '.join((norm(t) for t in titles)))
    return (labels, 'supporting_title_set')

def bootstrap(d, clusters, b, seed):
    d = np.asarray(d, dtype=float)
    groups = defaultdict(list)
    for (i, c) in enumerate(clusters):
        groups[c].append(i)
    group_idx = list(groups.values())
    g = len(group_idx)
    if g < 2:
        raise ValueError('Bootstrap needs >=2 clusters')
    sums = np.array([d[v].sum() for v in group_idx])
    counts = np.array([len(v) for v in group_idx])
    rng = np.random.default_rng(seed)
    samples = np.empty(b)
    for start in range(0, b, 200):
        stop = min(b, start + 200)
        pick = rng.integers(0, g, size=(stop - start, g))
        samples[start:stop] = _pkg_resolve(sums[pick].sum(axis=1) / counts[pick].sum(axis=1))
    return {'delta': float(d.mean()), 'ci95': [float(x) for x in np.quantile(samples, [0.025, 0.975])], 'n_clusters': g}

def label(a, i):
    return 'abstain' if not a['coverage'][i] else 'correct_EM' if a['em'][i] else 'wrong_EM'

def parse_titles(r, field):
    if not field:
        return None
    v = r.get(field)
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if not isinstance(v, list):
        return None
    titles = []
    for x in v:
        if isinstance(x, str):
            titles.append(x)
        elif isinstance(x, dict):
            title = x.get('title')
            if not isinstance(title, str):
                return None
            titles.append(title)
        else:
            return None
    return titles

def evidence_strata(ids, gold, retrieval_path, qi_field, van_field, arr):
    if not qi_field or not van_field:
        return {'status': 'not_run', 'reason': 'Require --qi-title-field and --van-title-field'}
    retr = {}
    for r in read_rows(retrieval_path):
        q = qid(r)
        if q in retr:
            raise ValueError('Duplicate retrieval qid ' + q)
        retr[q] = r
    if set(ids) - set(retr):
        raise ValueError('Retrieval missing holdout QIDs')
    grades = {}
    bad = []
    for q in ids:
        gold_titles = supporting_titles(gold[q])
        if not gold_titles:
            bad.append(q)
            continue
        target = {norm(t) for t in gold_titles}
        qi = parse_titles(retr[q], qi_field)
        van = parse_titles(retr[q], van_field)
        if qi is None or van is None:
            bad.append(q)
            continue
        grades[q] = (len(target.intersection(map(norm, qi))), len(target.intersection(map(norm, van))), len(target))
    if bad:
        return {'status': 'not_run', 'reason': f'Cannot safely reconstruct titles for {len(bad)} rows', 'examples': bad[:5]}
    out = {}
    for (arm, j) in [('QI', 0), ('VAN', 1)]:
        strata = {}
        for grade in ('0', '1', '2+'):
            ix = np.array([i for (i, q) in enumerate(ids) if ('2+' if grades[q][j] >= 2 else str(grades[q][j])) == grade], dtype=int)
            if len(ix):
                conditions = ('QI_STRICT', 'QI_CAND', 'QA_STRICT') if arm == 'QI' else ('VAN_STRICT', 'VAN_CAND')
                strata[grade] = {'n': len(ix), 'metrics': {c: summarize(arr[c], ix) for c in conditions}}
        out[arm] = strata
    cross = Counter(((grades[q][0], grades[q][1]) for q in ids))
    return {'status': 'descriptive', 'QI_conditioned': out['QI'], 'VAN_conditioned': out['VAN'], 'hit_cross_tab': [{'qi_hits': x, 'van_hits': y, 'n': n} for ((x, y), n) in sorted(cross.items())], 'target_title_counts': dict(Counter((grades[q][2] for q in ids))), 'caveat': 'System-conditioned strata are NOT paired subsets. Gold-title hit is NOT sentence-level evidence sufficiency.'}

def verify_retrieval_manifest(retrieval_path, split_path):
    manifest = _pkg_resolve(Path(_pkg_resolve(retrieval_path)).parent / 'retrieval_manifest.json')
    if not manifest.is_file():
        raise ValueError('Missing frozen retrieval_manifest.json: ' + str(manifest))
    m = json.loads(manifest.read_text(encoding='utf-8-sig'))
    expected = {'predictions_sha256': sha(retrieval_path), 'split_ids_sha256': sha(split_path)}
    for (name, actual) in expected.items():
        if m.get(name) != actual:
            raise ValueError(f'FROZEN HASH MISMATCH {name}: expected={m.get(name)} actual={actual}')
    if m.get('gold_used_in_retrieval') is not False:
        raise ValueError('Manifest does not assert gold_used_in_retrieval=False')
    return {'status': 'PASS', 'manifest': str(manifest), 'predictions_sha256': expected['predictions_sha256'], 'split_ids_sha256': expected['split_ids_sha256'], 'gold_used_in_retrieval': False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold', default='data/hotpotqa/hotpot_dev_distractor.jsonl')
    p.add_argument('--split', default='out/holdout_locked/split_ids.json')
    p.add_argument('--pred-dir', default='out/holdout_llm')
    p.add_argument('--retrieval', default='out/holdout_locked/retrieval_predictions.jsonl')
    p.add_argument('--out', default='out/qi_ablation_integrated_v2')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cluster', choices=('question', 'supporting_title_set'), default='question')
    p.add_argument('--qi-title-field', default='qi_titles')
    p.add_argument('--van-title-field', default='va_titles')
    p.add_argument('--expected-n', type=int, default=4405)
    args = p.parse_args()
    if args.bootstrap < 100:
        p.error('--bootstrap must be >=100')
    (ids, split) = load_split(args.split)
    if len(ids) != args.expected_n:
        raise ValueError(f'Holdout count {len(ids)} != {args.expected_n}')
    gold = load_gold(args.gold, ids)
    paths = {f'pred_{c}': _pkg_resolve(Path(_pkg_resolve(args.pred_dir)) / f'{c}.jsonl') for c in CONDS}
    pred = {c: load_pred(paths[f'pred_{c}'], ids, c) for c in CONDS}
    arr = {c: arrays(ids, gold, pred[c]) for c in CONDS}
    (clusters, cluster_method) = cluster_ids(ids, gold, args.cluster)
    summaries = {c: summarize(a) for (c, a) in arr.items()}
    contrasts = {}
    transitions = []
    for (c1, c2) in PAIRS:
        contrasts[f'{c1} - {c2}'] = {m: bootstrap(arr[c1][m] - arr[c2][m], clusters, args.bootstrap, args.seed) for m in ('em', 'f1', 'coverage', 'wrong_em_all')}
        ct = Counter(((label(arr[c1], i), label(arr[c2], i)) for i in range(len(ids))))
        transitions.extend(({'comparison': f'{c1} vs {c2}', 'left': l, 'right': r, 'n': n} for ((l, r), n) in sorted(ct.items())))
    interaction = {m: bootstrap(arr['QI_STRICT'][m] - arr['VAN_STRICT'][m] - arr['QI_CAND'][m] + arr['VAN_CAND'][m], clusters, args.bootstrap, args.seed) for m in ('em', 'f1', 'coverage', 'wrong_em_all')}
    risk = []
    for c in ('QI_CAND', 'VAN_CAND'):
        scores = []
        for (i, q) in enumerate(ids):
            raw = pred[c][q].get('top1_score')
            try:
                s = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f'Invalid score {c}/{q}: {raw}')
            if not math.isfinite(s):
                raise ValueError(f'Nonfinite score {c}/{q}')
            scores.append((-s, q, i))
        scores.sort()
        for target in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
            k = round(len(ids) * target)
            ix = np.array([t[2] for t in scores[:k]], dtype=int)
            answered = arr[c]['coverage'][ix].sum()
            wrong = arr[c]['wrong_em_all'][ix].sum()
            risk.append({'condition': c, 'target_selected_fraction': target, 'selected_n': k, 'actual_coverage': float(_pkg_resolve(answered / len(ids))), 'wrong_em_answered': float(_pkg_resolve(wrong / answered)) if answered else None, 'wrong_em_all': float(_pkg_resolve(wrong / len(ids))), 'em_all': float(_pkg_resolve(arr[c]['em'][ix].sum() / len(ids)))})
    freeze_audit = verify_retrieval_manifest(args.retrieval, args.split)
    ev = evidence_strata(ids, gold, args.retrieval, args.qi_title_field, args.van_title_field, arr)
    input_paths = {'gold': args.gold, 'split': args.split, **{k: str(v) for (k, v) in paths.items()}}
    if Path(_pkg_resolve(args.retrieval)).is_file():
        input_paths['retrieval'] = args.retrieval
    hashes = {k: sha(v) for (k, v) in input_paths.items()}
    out = Path(_pkg_resolve(args.out))
    out.mkdir(parents=True, exist_ok=True)
    report = {'experiment': 'QI-RAG integrated ablation v3', 'n': len(ids), 'cluster_method': cluster_method, 'n_clusters': len(set(clusters)), 'scores': summaries, 'ABL02': {'contrasts': contrasts, 'interactions': interaction}, 'ABL01_exploratory': {'risk': risk, 'status': 'NOT matched actual coverage; in-holdout score ranking, descriptive only'}, 'ABL03': ev, 'retrieval_freeze_audit': freeze_audit, 'input_sha256': hashes, 'warnings': ['Wrong_EM is non-abstained exact-match mismatch, NOT verified semantic hallucination', 'STRICT vs CAND prompts change instructions; effects are prompt-package differences', 'Supporting title set is not a unique document cluster; report cluster choice', 'Holdout top-K is descriptive, never claim independent calibration', 'Do not compare risk between unequal actual coverage as matched-coverage effect', 'Retrieval and split hashes checked against retrieval manifest; generated LLM outputs hashes recorded only, not independently frozen']}
    _pkg_output(_pkg_resolve(out / 'report.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    def write_csv(name, rows, cols):
        with _pkg_pathlib.Path(_pkg_open_path(_pkg_resolve(out / name), 'w')).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    write_csv('scores.csv', [{'condition': c, **v} for (c, v) in summaries.items()], ['condition', 'n', 'em', 'f1', 'coverage', 'wrong_em_all', 'wrong_em_answered'])
    write_csv('transitions.csv', transitions, ['comparison', 'left', 'right', 'n'])
    write_csv('risk_exploratory.csv', risk, ['condition', 'target_selected_fraction', 'selected_n', 'actual_coverage', 'wrong_em_answered', 'wrong_em_all', 'em_all'])
    print('QI-RAG ABLATION v2 (READ-ONLY) N=', len(ids), 'bootstrap=', args.bootstrap, 'cluster=', cluster_method, 'n_clusters=', len(set(clusters)))
    for (c, v) in summaries.items():
        print(c, 'EM=%.4f F1=%.4f Coverage=%.4f Wrong_EM_All=%.4f' % (v['em'], v['f1'], v['coverage'], v['wrong_em_all']))
    print('2x2 interaction EM:', interaction['em'])
    print('ABL03:', ev['status'])
    print('FREEZE:', freeze_audit['status'])
    print('ABL01: exploratory, NOT matched actual coverage')
    print('Saved:', _pkg_resolve(out / 'report.json'), _pkg_resolve(out / 'scores.csv'), _pkg_resolve(out / 'transitions.csv'), _pkg_resolve(out / 'risk_exploratory.csv'))
if __name__ == '__main__':
    main()
