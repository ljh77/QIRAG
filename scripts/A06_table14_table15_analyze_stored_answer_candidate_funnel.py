"""ABL-05 v7: frozen candidate retrieval -> reconstructed delivery -> output audit.
No LLM calls; no causal attribution. v6.1 input read-only. Disjoint funnel,
EM/abstention transitions, paired bootstrap and conservation checks.
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse, csv, json, collections, hashlib
from pathlib import Path
import numpy as np
import S00_dependency_qi_ablation_integrated_v3 as base
CONDS = ('QA_STRICT', 'QI_STRICT', 'QI_CAND')
METRICS = ('correct_em', 'wrong_em', 'abstain')

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def csv_load(path):
    with _pkg_pathlib.Path(_pkg_open_path(Path(_pkg_resolve(path)), 'r')).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, keys):
    with _pkg_pathlib.Path(_pkg_open_path(Path(_pkg_resolve(path)), 'w')).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def b(v):
    require(str(v) in ('0', '1'), 'Invalid binary field: ' + str(v))
    return int(v)

def stage(row):
    (rawlo, rawhi) = (b(row['raw8_lower']), b(row['raw8_upper']))
    (dlo, dhi) = (b(row['reconstructed_delivered_lower']), b(row['reconstructed_delivered_upper']))
    require(rawlo <= rawhi and dlo <= dhi and (dlo <= rawlo) and (dhi <= rawhi), 'Nonmonotone retrieval/delivery bound ' + row['qid'])
    if not rawhi:
        return 'raw8_definite_miss'
    if not rawlo:
        return 'raw8_uncertain'
    if dlo:
        return 'delivered_definite_hit'
    if not dhi:
        return 'raw8_hit_delivered_miss'
    return 'delivered_uncertain'

def st(pred, gold):
    if base.abstain(pred):
        return 'abstain'
    return 'correct_em' if base.em(pred, gold) > 0 else 'wrong_em'

def boot_mean(values, B, rng):
    a = np.asarray(values, dtype=float)
    n = len(a)
    if not n:
        return {'n': 0, 'mean': None, 'ci95': None}
    sims = np.empty(B)
    for start in range(0, B, 100):
        count = min(100, B - start)
        idx = rng.integers(0, n, size=(count, n))
        sims[start:start + count] = a[idx].mean(axis=1)
    return {'n': n, 'mean': float(a.mean()), 'ci95': [float(v) for v in np.quantile(sims, [0.025, 0.975])]}

def evaluate(group, B, seed):
    out = {'n': len(group), 'conditions': {}, 'paired': {}}
    if not group:
        return out
    for (j, c) in enumerate(CONDS):
        vals = {metric: [int(r[c + '_state'] == metric) for r in group] for metric in METRICS}
        out['conditions'][c] = {m: boot_mean(v, B, np.random.default_rng(seed + 300 * j + i)) for (i, (m, v)) in enumerate(vals.items())}
    for (j, (a, c)) in enumerate((('QA_STRICT', 'QI_STRICT'), ('QA_STRICT', 'QI_CAND'), ('QI_STRICT', 'QI_CAND'))):
        out['paired'][a + '_minus_' + c] = {m: boot_mean([int(r[a + '_state'] == m) - int(r[c + '_state'] == m) for r in group], B, np.random.default_rng(seed + 1000 + j * 10 + i)) for (i, m) in enumerate(METRICS)}
    return out

def test():

    def r(a, b, c, d):
        return {'qid': 'x', 'raw8_lower': str(a), 'raw8_upper': str(b), 'reconstructed_delivered_lower': str(c), 'reconstructed_delivered_upper': str(d)}
    assert stage(r(0, 0, 0, 0)) == 'raw8_definite_miss'
    assert stage(r(0, 1, 0, 0)) == 'raw8_uncertain'
    assert stage(r(1, 1, 0, 0)) == 'raw8_hit_delivered_miss'
    assert stage(r(1, 1, 1, 1)) == 'delivered_definite_hit'
    assert stage(r(1, 1, 0, 1)) == 'delivered_uncertain'
    assert st("I don't know.", ['Paris']) == 'abstain'
    assert st('Paris', ['Paris']) == 'correct_em'
    assert st('London', ['Paris']) == 'wrong_em'
    arr = [1, 0, 1, 1]
    t = boot_mean(arr, 300, np.random.default_rng(0))
    assert t['mean'] == 0.75
    print('SELFTEST PASS: funnel partition, uncertainty bounds, output states, bootstrap')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v6-dir', default='out/qi_abl05_delivery_reconstruct_v6_1')
    p.add_argument('--gold', default='data/hotpotqa/hotpot_dev_distractor.jsonl')
    p.add_argument('--split', default='out/holdout_locked/split_ids.json')
    p.add_argument('--retrieval', default='out/holdout_locked/retrieval_predictions.jsonl')
    p.add_argument('--pred-dir', default='out/holdout_llm')
    p.add_argument('--out', default='out/qi_abl05_failure_attribution_v7')
    p.add_argument('--bootstrap', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--expected-n', type=int, default=4405)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    test()
    if a.selftest:
        return
    require(a.bootstrap >= 100, 'bootstrap must be >=100')
    src = Path(_pkg_resolve(a.v6_dir))
    old = json.loads(_pkg_resolve(src / 'report.json').read_text(encoding='utf-8'))
    require(old['n_holdout'] == a.expected_n, 'v6 N mismatch')
    require(old['frozen_config'] == 'PASS' and old['frozen_retrieval']['status'] == 'PASS', 'v6 freeze not passed')
    require(base.sha(a.gold) == old['gold_sha256'], 'Gold SHA changed since v6')
    require(base.sha(a.retrieval) == old['frozen_retrieval']['predictions_sha256'], 'Retrieval SHA changed since v6')
    (ids, _) = base.load_split(a.split)
    require(len(ids) == a.expected_n, 'Unexpected Holdout N')
    gold = base.load_gold(a.gold, ids)
    v6rows = csv_load(_pkg_resolve(src / 'reconstructed_delivery.csv'))
    index = {}
    for r in v6rows:
        q = r['qid']
        require(q not in index, 'Duplicate v6 QID ' + q)
        index[q] = r
    require(set(index) == set(ids), 'v6 QID set mismatch')
    pred = {c: base.load_pred(_pkg_resolve(Path(_pkg_resolve(a.pred_dir)) / (c + '.jsonl')), ids, c) for c in CONDS}
    rows = []
    counts = collections.Counter()
    for q in ids:
        oldrow = index[q]
        r = dict(oldrow)
        r['stage'] = stage(r)
        r['qa_reconstructed_provenance'] = 'DETERMINISTIC_RECONSTRUCTION_NOT_HTTP_PROVEN'
        for c in CONDS:
            r[c + '_state'] = st(pred[c][q]['pred'], base.gold_answers(gold[q]))
        r['qa_vs_qi_transition'] = r['QA_STRICT_state'] + ' -> ' + r['QI_STRICT_state']
        r['qa_vs_cand_transition'] = r['QA_STRICT_state'] + ' -> ' + r['QI_CAND_state']
        counts[r['stage']] += 1
        rows.append(r)
    require(len(rows) == a.expected_n and sum(counts.values()) == a.expected_n, 'funnel count failure')
    require(sum((b(r['raw8_lower']) for r in rows)) == sum((b(r['reconstructed_delivered_lower']) for r in rows)) + counts['raw8_hit_delivered_miss'] + sum((int(r['stage'] == 'delivered_uncertain') for r in rows)), 'funnel conservation failure')
    groups = {'all': rows}
    for key in ('raw8_definite_miss', 'raw8_uncertain', 'raw8_hit_delivered_miss', 'delivered_definite_hit', 'delivered_uncertain'):
        groups[key] = [r for r in rows if r['stage'] == key]
    results = {k: evaluate(v, a.bootstrap, a.seed) for (k, v) in groups.items()}
    transitions = []
    for (group, subset) in groups.items():
        for pair in (('QA_STRICT', 'QI_STRICT'), ('QA_STRICT', 'QI_CAND')):
            ct = collections.Counter(((r[pair[0] + '_state'], r[pair[1] + '_state']) for r in subset))
            for x in METRICS:
                for y in METRICS:
                    transitions.append({'group': group, 'system_a': pair[0], 'system_b': pair[1], 'state_a': x, 'state_b': y, 'n': ct[x, y], 'group_n': len(subset)})
    reasons = []
    for r in rows:
        if r['stage'] == 'raw8_hit_delivered_miss':
            ranks = [int(s) for s in r['chosen_ranks'].split(',') if s]
            reasons.append({'qid': r['qid'], 'stage': r['stage'], 'selected_ranks': r['chosen_ranks'], 'selected_n': r['chosen_n'], 'selected_rank_beyond_first_two': int(any((x > 2 for x in ranks))), 'unique_answer_selection': 'first P unique nonempty stored answer strings', 'QA_STRICT_state': r['QA_STRICT_state']})
    integrity = {'v6_report_sha256': base.sha(_pkg_resolve(src / 'report.json')), 'v6_rows_sha256': base.sha(_pkg_resolve(src / 'reconstructed_delivery.csv')), 'gold_sha256': base.sha(a.gold), 'split_sha256': base.sha(a.split), 'retrieval_sha256': base.sha(a.retrieval), 'pred_sha256': {c: base.sha(_pkg_resolve(Path(_pkg_resolve(a.pred_dir)) / (c + '.jsonl'))) for c in CONDS}}
    report = {'experiment': 'ABL-05 v7 candidate funnel and output association', 'provenance': 'RECONSTRUCTED_NOT_TRANSMISSION_VERIFIED', 'n': len(rows), 'counts': dict(counts), 'funnel': results, 'integrity': integrity, 'limitations': ['Read-only post-hoc observational analysis, not causal failure attribution.', 'Candidate hit uses normalized exact match; semantic equivalence not adjudicated.', 'QA_STRICT receives stored answer candidates; QI_STRICT and QI_CAND receive passages, so comparison is not an isolated candidate-selection experiment.', 'Original HTTP payload and per-question source prompt bytes absent; reconstructed delivery is not independently network-verified.', 'Search-index reference validity supported by title diagnostics, not proven for all rows.', 'Subset conditional contrasts reflect selected question populations; no equality-of-coverage claim.', 'Wrong-EM is not a factual hallucination label.']}
    dest = Path(_pkg_resolve(a.out))
    dest.mkdir(parents=True, exist_ok=True)
    _pkg_output(_pkg_resolve(dest / 'report.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(_pkg_resolve(dest / 'failure_funnel.csv'), rows, list(rows[0]))
    write_csv(_pkg_resolve(dest / 'paired_transitions.csv'), transitions, list(transitions[0]))
    if reasons:
        write_csv(_pkg_resolve(dest / 'retrieved_but_excluded.csv'), reasons, list(reasons[0]))
    else:
        write_csv(_pkg_resolve(dest / 'retrieved_but_excluded.csv'), [], ['qid', 'stage', 'selected_ranks', 'selected_n', 'selected_rank_beyond_first_two', 'unique_answer_selection', 'QA_STRICT_state'])
    print('ABL-05 v7 READ-ONLY N=', len(rows), 'SOURCE_FROZEN=PASS', 'BOOT=', a.bootstrap)
    print('PROVENANCE: RECONSTRUCTED; HTTP payload unavailable')
    print('FUNNEL:', dict(counts))
    for (name, res) in results.items():

        def value(c):
            return res['conditions']['QA_STRICT'][c]['mean'] if res['n'] else None
        print('GROUP', name, 'N=', res['n'], 'QA_EM=', value('correct_em'), 'QA_WRONG_EM=', value('wrong_em'), 'QA_ABSTAIN=', value('abstain'))
    print('OUTPUTS:', _pkg_resolve(dest / 'report.json'), _pkg_resolve(dest / 'failure_funnel.csv'), _pkg_resolve(dest / 'paired_transitions.csv'), _pkg_resolve(dest / 'retrieved_but_excluded.csv'))
if __name__ == '__main__':
    main()
