"""ABL-05 v4 — conservative Q-A candidate recall with uncertain index references.

Uses frozen retrieval and ordered index rows, without editing any input or
calling an LLM. Reference metadata categories are diagnostics, not a proof
that row IDs are correct. Uncertain references are NEVER treated as known hits.
Outputs strict lower/upper bounds for normalized-exact-match candidate recall.

Raw@K: all retrieved Q-A items up to K, independent of prompt delivery.
Proxy@P: first P retrieved items, NOT independently verified delivered answers.
QA_STRICT output is observational and is not proof of successful candidate choice.
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse, csv, json
from pathlib import Path
from collections import Counter
import numpy as np
import S00_dependency_qi_ablation_integrated_v3 as base
KS = (1, 2, 4, 8)
CONDS = ('QA_STRICT', 'QI_STRICT', 'QI_CAND')
CERTAIN = {'exact_order', 'exact_set_reordered', 'index_titles_subsequence_with_extras', 'index_titles_present_reordered_with_extras'}

def require(cond, msg):
    if not cond:
        raise ValueError(msg)

def classify(index_titles, retrieved_titles):
    """Matches the separate read-only v3 title diagnostic categories."""
    require(isinstance(index_titles, list) and isinstance(retrieved_titles, list), 'invalid titles')
    require(all((isinstance(t, str) for t in index_titles + retrieved_titles)), 'nonstr title')
    a = [' '.join(t.casefold().split()) for t in index_titles]
    b = [' '.join(t.casefold().split()) for t in retrieved_titles]
    if a == b:
        return 'exact_order'
    if Counter(a) == Counter(b):
        return 'exact_set_reordered'
    i = 0
    for t in b:
        if i < len(a) and t == a[i]:
            i += 1
    if i == len(a):
        return 'index_titles_subsequence_with_extras'
    if set(a).issubset(set(b)):
        return 'index_titles_present_reordered_with_extras'
    if set(a) & set(b):
        return 'partial_overlap_or_substitution'
    return 'disjoint_titles'

def answer_values(v):
    if isinstance(v, dict):
        return answer_values(v.get('text', v.get('answer')))
    if isinstance(v, (list, tuple)):
        result = []
        for x in v:
            result.extend(answer_values(x))
        return result
    if v is None:
        return []
    return [str(v).strip()] if str(v).strip() else []

def candidate_hit(values, golds):
    return int(any((base.norm(x) == base.norm(g) for x in values for g in golds)))

def bounds(items, k, golds):
    subset = items[:k]
    known = [v for item in subset if item['trusted'] for v in item['answers']]
    low = candidate_hit(known, golds)
    uncertain = int(any((not item['trusted'] for item in subset)))
    high = int(bool(low or uncertain))
    return (low, high, uncertain)

def boot(values, B, seed):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {'n': 0, 'mean': None, 'ci95': None}
    mean = float(values.mean())
    if len(values) == 1:
        return {'n': 1, 'mean': mean, 'ci95': None}
    rng = np.random.default_rng(seed)
    sims = []
    for start in range(0, B, 128):
        n = min(128, B - start)
        indices = rng.integers(0, len(values), size=(n, len(values)))
        sims.extend(values[indices].mean(axis=1).tolist())
    return {'n': int(len(values)), 'mean': mean, 'ci95': [float(x) for x in np.quantile(sims, [0.025, 0.975])], 'method': 'question bootstrap'}

def write_csv(p, rows, fields):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def title_hit(goldrow, retrow):
    titles = base.supporting_titles(goldrow)
    require(titles, f'missing supporting title for {base.qid(goldrow)}')
    target = {base.norm(x) for x in titles}
    qit = retrow.get('qi_titles')
    vat = retrow.get('va_titles')
    require(isinstance(qit, list) and isinstance(vat, list), 'missing retrieval titles')
    return (min(2, len(target & {base.norm(x) for x in qit})), min(2, len(target & {base.norm(x) for x in vat})))

def selftest():
    assert classify(['A', 'B'], ['A', 'extra', 'B']) == 'index_titles_subsequence_with_extras'
    assert classify(['A', 'B'], ['A', 'X']) == 'partial_overlap_or_substitution'
    assert classify(['A', 'B'], ['B', 'A']) == 'exact_set_reordered'
    assert bounds([{'trusted': False, 'answers': ['Correct']}], 1, ['Correct']) == (0, 1, 1)
    assert bounds([{'trusted': True, 'answers': ['Correct']}, {'trusted': False, 'answers': []}], 2, ['Correct']) == (1, 1, 1)
    assert bounds([{'trusted': True, 'answers': ['Wrong']}], 1, ['Correct']) == (0, 0, 0)
    assert candidate_hit(['The United States'], ['United States']) == 1
    assert boot([0, 1, 0, 1], 200, 0)['mean'] == 0.5
    print('SELFTEST PASS: title classifications, conservative bounds, EM and bootstrap')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default='data/_split/index_items.jsonl')
    p.add_argument('--gold', default='data/hotpotqa/hotpot_dev_distractor.jsonl')
    p.add_argument('--split', default='out/holdout_locked/split_ids.json')
    p.add_argument('--retrieval', default='out/holdout_locked/retrieval_predictions.jsonl')
    p.add_argument('--pred-dir', default='out/holdout_llm')
    p.add_argument('--out', default='out/qi_abl05_candidate_audit_v4')
    p.add_argument('--bootstrap', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--raw-k', type=int, default=8)
    p.add_argument('--proxy-p', type=int, default=2)
    p.add_argument('--expected-n', type=int, default=4405)
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()
    selftest()
    if args.selftest:
        return
    require(args.bootstrap >= 100 and args.raw_k >= 8 and (args.proxy_p > 0), 'bootstrap>=100 raw-k>=8 proxy-p>0')
    (ids, _) = base.load_split(args.split)
    require(len(ids) == args.expected_n, 'holdout N mismatch')
    freeze = base.verify_retrieval_manifest(args.retrieval, args.split)
    gold = base.load_gold(args.gold, ids)
    index = list(base.read_rows(args.index))
    require(len(index) == 90447, f'index length unexpected: {len(index)} (verify actual search index before changing)')
    require(all((isinstance(x.get('gold_titles'), list) for x in index)), 'invalid index title metadata')
    predpaths = {c: _pkg_resolve(Path(_pkg_resolve(args.pred_dir)) / (c + '.jsonl')) for c in CONDS}
    preds = {c: base.load_pred(predpaths[c], ids, c) for c in CONDS}
    retrieval = {}
    for r in base.read_rows(args.retrieval):
        q = base.qid(r)
        require(q in set(ids), f'unexpected retrieval qid {q}')
        require(q not in retrieval, f'duplicate retrieval qid {q}')
        retrieval[q] = r
    require(set(retrieval) == set(ids), 'retrieval QID mismatch')
    classifications = Counter()
    reference_rows = []
    rows = []
    for q in ids:
        r = retrieval[q]
        items = r.get('qi_top_items')
        require(isinstance(items, list) and len(items) >= args.raw_k, f'{q} missing raw top items')
        seen = set()
        checked = []
        for (rank, x) in enumerate(items, 1):
            ix = x.get('item')
            require(type(ix) is int and 0 <= ix < len(index), f'{q} rank{rank} invalid index {ix}')
            require(ix not in seen, f'{q} duplicate index {ix}')
            seen.add(ix)
            cls = classify(index[ix]['gold_titles'], x.get('titles'))
            classifications[cls] += 1
            trusted = cls in CERTAIN
            checked.append({'item': ix, 'rank': rank, 'trusted': trusted, 'answers': answer_values(index[ix].get('answer'))})
            if not trusted:
                reference_rows.append({'qid': q, 'rank': rank, 'item': ix, 'classification': cls, 'index_question': index[ix].get('question', ''), 'index_answers': json.dumps(answer_values(index[ix].get('answer')), ensure_ascii=False), 'index_titles': json.dumps(index[ix]['gold_titles'], ensure_ascii=False), 'retrieved_titles': json.dumps(x.get('titles'), ensure_ascii=False)})
        golds = base.gold_answers(gold[q])
        (qt, vt) = title_hit(gold[q], r)
        rr = {'qid': q, 'qi_gold_title_hits': qt, 'van_gold_title_hits': vt}
        for k in KS:
            (lo, hi, unc) = bounds(checked, k, golds)
            rr[f'raw{k}_lower'] = lo
            rr[f'raw{k}_upper'] = hi
            rr[f'raw{k}_uncertain'] = unc
        (lo, hi, unc) = bounds(checked, args.proxy_p, golds)
        rr['proxy_lower'] = lo
        rr['proxy_upper'] = hi
        rr['proxy_uncertain'] = unc
        for c in CONDS:
            response = preds[c][q]['pred']
            answered = not base.abstain(response)
            correct = int(answered and base.em(response, golds) > 0)
            rr[c + '_answered'] = int(answered)
            rr[c + '_correct_em'] = correct
            rr[c + '_wrong_em'] = int(answered and (not correct))
            rr[c + '_state'] = 'abstain' if not answered else 'correct_em' if correct else 'wrong_em'
        rows.append(rr)
    require(sum(classifications.values()) == len(ids) * args.raw_k, 'Unexpected references count: raw results may not equal raw-k (check --raw-k)')
    metrics = {}
    for label in [f'raw{k}' for k in KS] + ['proxy']:
        low = label + '_lower'
        up = label + '_upper'
        uncertain = label + '_uncertain'
        metrics[label] = {'lower': boot([r[low] for r in rows], args.bootstrap, args.seed), 'upper': boot([r[up] for r in rows], args.bootstrap, args.seed + 1), 'uncertain_qids': sum((r[uncertain] for r in rows)), 'width': float(np.mean([r[up] - r[low] for r in rows]))}
    outcomes = []
    for criterion in ('raw8', 'proxy'):
        for status in ('definitely_hit', 'definitely_miss', 'uncertain'):
            subset = [r for r in rows if (r[criterion + '_lower'] == 1 if status == 'definitely_hit' else r[criterion + '_upper'] == 0 if status == 'definitely_miss' else r[criterion + '_lower'] != r[criterion + '_upper'])]
            for c in CONDS:
                outcomes.append({'criterion': criterion, 'group': status, 'condition': c, 'n': len(subset), 'correct_em': sum((r[c + '_correct_em'] for r in subset)), 'wrong_em': sum((r[c + '_wrong_em'] for r in subset)), 'abstain': sum((1 - r[c + '_answered'] for r in subset)), 'em': float(np.mean([r[c + '_correct_em'] for r in subset])) if subset else None, 'coverage': float(np.mean([r[c + '_answered'] for r in subset])) if subset else None})
    cross = Counter(((r['qi_gold_title_hits'], r['raw8_lower'], r['raw8_upper']) for r in rows))
    crossrows = [{'qi_gold_title_hits': a, 'raw8_lower': b, 'raw8_upper': c, 'n': n} for ((a, b, c), n) in sorted(cross.items())]
    contrasts = {}
    for (i, j) in ((1, 2), (2, 4), (4, 8), (1, 8)):
        contrasts[f'raw{j}_minus_raw{i}_lower'] = boot([r[f'raw{j}_lower'] - r[f'raw{i}_lower'] for r in rows], args.bootstrap, args.seed)
        contrasts[f'raw{j}_minus_raw{i}_upper'] = boot([r[f'raw{j}_upper'] - r[f'raw{i}_upper'] for r in rows], args.bootstrap, args.seed)
    for (a, b) in (('QI_STRICT', 'QA_STRICT'), ('QI_CAND', 'QA_STRICT'), ('QI_STRICT', 'QI_CAND')):
        contrasts[f'{a}_minus_{b}_EM'] = boot([r[a + '_correct_em'] - r[b + '_correct_em'] for r in rows], args.bootstrap, args.seed)
    nref = sum(classifications.values())
    nunc = sum((v for (k, v) in classifications.items() if k not in CERTAIN))
    report = {'experiment': 'ABL-05 v4 conservative Q-A candidate audit', 'n': len(ids), 'raw_k': args.raw_k, 'proxy_p': args.proxy_p, 'freeze': freeze, 'index_audit': {'status': 'DIAGNOSTIC_NOT_PROVEN', 'index_rows': len(index), 'references': nref, 'classification_counts': dict(classifications), 'uncertain_references': nunc, 'uncertain_qids': len({r['qid'] for r in reference_rows}), 'note': 'Metadata agreement is not a cryptographic assertion of original search index row identity.'}, 'candidate_recall_bounds': metrics, 'paired_contrasts': contrasts, 'conditional_outputs': outcomes, 'gold_title_x_candidate': crossrows, 'delivered_candidates': {'status': 'NOT_VERIFIED', 'recall': None, 'reason': 'QA_STRICT frozen prompt bytes and exact candidate-extraction policy were not independently audited.'}, 'sha256': {'index': base.sha(args.index), 'gold': base.sha(args.gold), 'split': base.sha(args.split), 'retrieval': base.sha(args.retrieval), **{c: base.sha(path) for (c, path) in predpaths.items()}}, 'limitations': ['Lower bound treats all uncertain references as absent. Upper bound treats an uncertain reference in prefix as possibly correct.', 'Upper/lower are identification bounds under unresolved metadata identity, NOT confidence limits. Bootstrap CIs separately quantify question sampling variability.', 'The conservative lower bound is not guaranteed if an ostensibly aligned item index is wrong despite matching title metadata.', 'Gold answer normalized EM can miss semantically correct alternate answers.', 'Top-P candidate subset is an unverified proxy, not actual delivered QA_STRICT payload.', 'Gold-title hits do not measure stored-answer candidate recall or entailment.', 'Output correct_EM / wrong_EM is not factual correctness / hallucination.', 'No causal mediation or LLM candidate-choice claim.']}
    out = Path(_pkg_resolve(args.out))
    out.mkdir(parents=True, exist_ok=True)
    _pkg_output(_pkg_resolve(out / 'report.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(_pkg_resolve(out / 'candidate_audit.csv'), rows, list(rows[0]))
    write_csv(_pkg_resolve(out / 'uncertain_references.csv'), reference_rows, ['qid', 'rank', 'item', 'classification', 'index_question', 'index_answers', 'index_titles', 'retrieved_titles'])
    write_csv(_pkg_resolve(out / 'conditional_metrics.csv'), outcomes, list(outcomes[0]))
    write_csv(_pkg_resolve(out / 'gold_title_vs_candidate.csv'), crossrows, list(crossrows[0]))
    print('ABL-05 v4 READ ONLY; FREEZE=', freeze['status'], 'N=', len(ids), 'INDEX STATUS=DIAGNOSTIC_NOT_PROVEN')
    print('REFS=', nref, 'UNCERTAIN REFS=', nunc, 'UNCERTAIN QIDS=', len({r['qid'] for r in reference_rows}))
    for (label, v) in metrics.items():
        print(label, 'LOWER=', round(v['lower']['mean'], 6), 'UPPER=', round(v['upper']['mean'], 6), 'UNCERTAIN_QIDS=', v['uncertain_qids'])
    print('DELIVERED=NOT_VERIFIED; NO CAUSAL SELECTION CLAIM')
    print('OUTPUTS:', _pkg_resolve(out / 'report.json'), _pkg_resolve(out / 'candidate_audit.csv'), _pkg_resolve(out / 'uncertain_references.csv'), _pkg_resolve(out / 'conditional_metrics.csv'), _pkg_resolve(out / 'gold_title_vs_candidate.csv'))
if __name__ == '__main__':
    main()
