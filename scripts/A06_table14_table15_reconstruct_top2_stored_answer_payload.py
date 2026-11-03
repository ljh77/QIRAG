"""ABL-05 v6.1: deterministically reconstruct QA_STRICT candidate payload.

Read-only on all source data. No LLM requests. The original generator's
QA_STRICT construction is reproduced byte-for-byte for the text prompt when
its frozen template/config hashes match. No original per-QID prompt bytes or
HTTP request log survives: status is RECONSTRUCTED, never TRANSMISSION_VERIFIED.

The original QA_STRICT offers stored ANSWERS; QI_STRICT offers PASSAGES.
Those interventions are different and must not be conflated.
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse, ast, collections, csv, hashlib, json, re
from pathlib import Path
import numpy as np
import S00_dependency_qi_ablation_integrated_v3 as base
import S00_dependency_qi_abl05_candidate_audit_v4 as v4
COND = 'QA_STRICT'

def check(x, msg):
    if not x:
        raise ValueError(msg)

def sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def value_forms(stored):
    """Evaluate an EXACT text candidate and optional safe Python list repr.

    We do not change the actual delivered text; the parsed-list estimate is
    labelled semantic-proxy and is separate from literal candidate EM.
    """
    out = [str(stored)]
    try:
        value = ast.literal_eval(str(stored))
    except (ValueError, SyntaxError, TypeError):
        return out
    if isinstance(value, (list, tuple)) and value and all((isinstance(x, str) for x in value)):
        out.extend(value)
    return list(dict.fromkeys(out))

def row_state(pred, gold):
    answered = not base.abstain(pred)
    correct = answered and base.em(pred, gold) > 0
    return 'correct_em' if correct else 'wrong_em' if answered else 'abstain'

def boot(vals, B, seed):
    if not vals:
        return {'n': 0, 'mean': None, 'ci95': None}
    a = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    simulations = []
    for start in range(0, B, 128):
        size = min(128, B - start)
        ix = rng.integers(0, len(a), size=(size, len(a)))
        simulations.extend(a[ix].mean(axis=1))
    return {'n': len(a), 'mean': float(a.mean()), 'ci95': list(map(float, np.quantile(simulations, [0.025, 0.975])))}

def dump_csv(path, data, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(data)

def selftest():
    ans = ["['Alpha']", "['Alpha']", "['Beta']"]
    assert list(dict.fromkeys(ans))[:2] == ["['Alpha']", "['Beta']"]
    assert value_forms("['Alpha']") == ["['Alpha']", 'Alpha']
    assert value_forms('Alpha') == ['Alpha']
    assert row_state("I don't know.", ['Alpha']) == 'abstain'
    assert row_state('Alpha', ['Alpha']) == 'correct_em'
    assert int(bool(0) or bool([])) == 0
    assert int(bool(0) or bool([3, 7])) == 1
    assert int(bool(1) or bool([])) == 1
    print('SELFTEST PASS: dedupe, list-repr, EM states, uncertain-list bounds')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='llm_experiments_v2.py')
    p.add_argument('--freeze', default='out/holdout_locked/llm_prompts_FREEZE.json')
    p.add_argument('--index', default='data/_split/index_items.jsonl')
    p.add_argument('--gold', default='data/hotpotqa/hotpot_dev_distractor.jsonl')
    p.add_argument('--split', default='out/holdout_locked/split_ids.json')
    p.add_argument('--retrieval', default='out/holdout_locked/retrieval_predictions.jsonl')
    p.add_argument('--pred-dir', default='out/holdout_llm')
    p.add_argument('--out', default='out/qi_abl05_delivery_reconstruct_v6')
    p.add_argument('--bootstrap', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--expected-n', type=int, default=4405)
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()
    selftest()
    if args.selftest:
        return
    check(args.bootstrap >= 100, 'bootstrap<100')
    import importlib.util
    source = Path(_pkg_resolve(args.source)).resolve()
    check(source.is_file(), 'Original generator missing')
    spec = importlib.util.spec_from_file_location('_abl05_original_generator', str(source))
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    freeze = json.loads(Path(_pkg_resolve(args.freeze)).read_text(encoding='utf-8'))
    frozen = freeze['config']
    config = {'model': orig.VLLM_MODEL, 'temperature': orig.TEMPERATURE, 'max_tokens': orig.MAX_TOKENS, 'P': orig.P, 'conds': orig.CONDS, 'strict_sha': sha_text(orig.STRICT), 'cand_sha': sha_text(orig.CAND)}
    check(config == frozen, 'FROZEN CONFIG DOES NOT MATCH source constants/templates')
    returned = orig.cfg_hash()
    check(isinstance(returned, tuple) and len(returned) == 2, 'original cfg_hash return contract changed')
    check(returned[1] == config, 'original cfg_hash config disagrees')
    check(returned[0] == freeze['config_sha256'], 'ORIGINAL CFG HASH != FREEZE')
    check(base.sha(args.retrieval) == freeze['retrieval_predictions_sha256'], 'LLM freeze retrieval SHA mismatch')
    retrieval_freeze = base.verify_retrieval_manifest(args.retrieval, args.split)
    (ids, _) = base.load_split(args.split)
    check(len(ids) == args.expected_n, 'Holdout N incorrect')
    gold = base.load_gold(args.gold, ids)
    index = list(base.read_rows(args.index))
    check(len(index) == 90447, 'Index length differs')
    ret = {}
    for r in base.read_rows(args.retrieval):
        q = base.qid(r)
        check(q not in ret, 'Duplicate retrieval QID ' + q)
        ret[q] = r
    check(set(ret) == set(ids), 'Retrieval/holdout QID disagreement')
    pred = {c: base.load_pred(_pkg_resolve(Path(_pkg_resolve(args.pred_dir)) / (c + '.jsonl')), ids, c) for c in ('QA_STRICT', 'QI_STRICT', 'QI_CAND')}
    counts = collections.Counter()
    rows = []
    suspects = []
    for q in ids:
        r = ret[q]
        goldvals = base.gold_answers(gold[q])
        question = gold[q].get('question')
        check(isinstance(question, str) and question.strip(), f'Missing question {q}')
        items = r['qi_top_items']
        check(isinstance(items, list) and len(items) == 8, f'Expected Top8 {q}')
        candidates = []
        first_candidates = []
        seen_items = set()
        uncertain_refs = []
        for (rank, item) in enumerate(items, 1):
            j = item.get('item')
            check(type(j) == int and 0 <= j < len(index), f'Bad item {q} rank{rank}')
            check(j not in seen_items, f'Duplicate item {q} {j}')
            seen_items.add(j)
            classif = v4.classify(index[j]['gold_titles'], item.get('titles'))
            trusted = classif in v4.CERTAIN
            if not trusted:
                uncertain_refs.append(rank)
            stored = str(index[j].get('answer', ''))
            if stored:
                candidates.append((rank, j, stored, trusted))
            if rank <= 2 and stored:
                first_candidates.append(stored)
        unique = []
        used = set()
        chosen = []
        for (rank, j, stored, trusted) in candidates:
            if stored not in used:
                used.add(stored)
                unique.append(stored)
                chosen.append((rank, j, stored, trusted))
            if len(chosen) >= orig.P:
                break
        prompt = orig.STRICT.format(ctx='\n'.join(('Stored answer: ' + a for a in unique[:orig.P])), q=question)
        check(prompt == orig.STRICT.format(ctx='\n'.join(('Stored answer: ' + c[2] for c in chosen)), q=question), 'prompt mismatch')
        qa_record = pred['QA_STRICT'][q]
        word_match = 'not_available'
        if qa_record.get('prompt_words') is not None:
            check(int(qa_record['prompt_words']) == len(prompt.split()), f'prompt_words mismatch {q}')
            word_match = 'matched'
        gold_norm = {base.norm(g) for g in goldvals}

        def hit(refs, only_trusted=True, parse=False):
            forms = []
            for entry in refs:
                if only_trusted and (not entry[3]):
                    continue
                forms.extend(value_forms(entry[2]) if parse else [entry[2]])
            return int(any((base.norm(x) in gold_norm for x in forms)))
        raw_candidates = [(rank, j, stored, trusted) for (rank, j, stored, trusted) in candidates]
        raw8_lower = hit(raw_candidates)
        raw8_upper = int(bool(raw8_lower) or bool(uncertain_refs))
        deliver_lower = hit(chosen)
        deliver_upper = int(deliver_lower or any((not x[3] for x in chosen)))
        delivered_parsed_lower = hit(chosen, parse=True)
        delivered_parsed_upper = int(delivered_parsed_lower or any((not x[3] for x in chosen)))
        gold_title = v4.title_hit(gold[q], r)[0]
        row = {'qid': q, 'raw8_lower': raw8_lower, 'raw8_upper': raw8_upper, 'reconstructed_delivered_lower': deliver_lower, 'reconstructed_delivered_upper': deliver_upper, 'reconstructed_delivered_parsed_lower': delivered_parsed_lower, 'reconstructed_delivered_parsed_upper': delivered_parsed_upper, 'raw8_correct_but_delivered_not': int(raw8_lower == 1 and deliver_upper == 0), 'chosen_ranks': ','.join((str(x[0]) for x in chosen)), 'chosen_items': ','.join((str(x[1]) for x in chosen)), 'chosen_answers': json.dumps([x[2] for x in chosen], ensure_ascii=False), 'chosen_n': len(chosen), 'uncertain_raw_ranks': ','.join(map(str, uncertain_refs)), 'uncertain_selected': int(any((not x[3] for x in chosen))), 'prompt_sha256_reconstructed': sha_text(prompt), 'prompt_words': len(prompt.split()), 'saved_prompt_words_comparison': word_match, 'qi_gold_title_hits': gold_title, 'QA_STRICT_state': row_state(pred['QA_STRICT'][q]['pred'], goldvals), 'QI_STRICT_state': row_state(pred['QI_STRICT'][q]['pred'], goldvals), 'QI_CAND_state': row_state(pred['QI_CAND'][q]['pred'], goldvals)}
        rows.append(row)
        counts['prompt_word_match_' + word_match] += 1
        counts['selected_with_uncertain_reference'] += row['uncertain_selected']
        counts['top2_selected_ranks_not_1_2'] += int(row['chosen_ranks'] != '1,2')
        if uncertain_refs and len(suspects) < 100:
            suspects.append({'qid': q, 'uncertain_raw_ranks': row['uncertain_raw_ranks'], 'chosen_ranks': row['chosen_ranks'], 'selected_uncertain': row['uncertain_selected']})

    def subset_metric(group, col):
        return boot([r[col] for r in group], args.bootstrap, args.seed)
    metric_cols = ('raw8_lower', 'raw8_upper', 'reconstructed_delivered_lower', 'reconstructed_delivered_upper', 'reconstructed_delivered_parsed_lower', 'reconstructed_delivered_parsed_upper', 'raw8_correct_but_delivered_not')
    metrics = {c: subset_metric(rows, c) for c in metric_cols}
    conditional = {}
    for (label, group) in [('definitely_delivered_hit', [r for r in rows if r['reconstructed_delivered_lower']]), ('definitely_delivered_miss', [r for r in rows if not r['reconstructed_delivered_upper']]), ('delivery_uncertain', [r for r in rows if r['reconstructed_delivered_lower'] != r['reconstructed_delivered_upper']])]:
        conditional[label] = {'n': len(group)}
        for cond in ('QA_STRICT', 'QI_STRICT', 'QI_CAND'):
            for state in ('correct_em', 'wrong_em', 'abstain'):
                conditional[label][cond + '_' + state] = boot([int(r[cond + '_state'] == state) for r in group], args.bootstrap, args.seed)
    report = {'experiment': 'ABL-05 v6 deterministic QA_STRICT reconstruction', 'provenance_status': 'DETERMINISTIC_RECONSTRUCTION_FROM_FROZEN_CONFIG_NOT_NETWORK_PAYLOAD_PROOF', 'n_holdout': len(rows), 'frozen_config': 'PASS', 'frozen_retrieval': retrieval_freeze, 'generator_sha256': base.sha(args.source), 'frozen_prompt_meta_sha256': base.sha(args.freeze), 'index_sha256': base.sha(args.index), 'gold_sha256': base.sha(args.gold), 'model': frozen['model'], 'P': orig.P, 'qa_delivered_rule': 'unique str(index[item].answer) from Top8, first P', 'integrity_counts': dict(counts), 'metrics': metrics, 'conditional_output': conditional, 'limitations': ['Per-question original prompt bytes and HTTP request payloads were NOT preserved. Deterministic reconstruction is not byte-level delivery proof.', 'Source code hash is provided but the frozen artifact freezes config, not complete original source bytes.', 'Index row identity is verified only by title consistency; uncertain references retained with conservative bounds.', 'QA_STRICT receives answer candidate strings; QI_STRICT receives documents. Conditional QA_STRICT metrics do not prove QI_STRICT selection fidelity.', 'Raw list string is the actual prompt payload; optional parsed-list values are a sensitivity analysis, not what model necessarily interpreted.', 'Gold exact match is not semantic correctness or entailment and Wrong-EM is not hallucination.']}
    out = Path(_pkg_resolve(args.out))
    out.mkdir(parents=True, exist_ok=True)
    _pkg_output(_pkg_resolve(out / 'report.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    dump_csv(_pkg_resolve(out / 'reconstructed_delivery.csv'), rows, list(rows[0]))
    dump_csv(_pkg_resolve(out / 'uncertain_references.csv'), suspects, ['qid', 'uncertain_raw_ranks', 'chosen_ranks', 'selected_uncertain'])
    print('ABL-05 v6 READ-ONLY N=', len(rows), 'FROZEN CONFIG=PASS RETRIEVAL=', retrieval_freeze['status'])
    print('PROVENANCE: RECONSTRUCTED; original prompt bytes/HTTP logs unavailable')
    print('P=', orig.P, ' QA_STRICT is stored ANSWER payload; QI_STRICT uses PASSAGES')
    print('COUNTS:', dict(counts))
    for key in metric_cols:
        x = metrics[key]
        print(f"{key}: {x['mean']:.6f} CI95={x['ci95']}")
    for (k, v) in conditional.items():
        print('CONDITIONAL', k, 'N=', v['n'], 'QA_EM=', v['QA_STRICT_correct_em']['mean'])
    print('OUTPUTS:', _pkg_resolve(out / 'report.json'), _pkg_resolve(out / 'reconstructed_delivery.csv'), _pkg_resolve(out / 'uncertain_references.csv'))
if __name__ == '__main__':
    main()
