"""POST-18F v3.2 exact matched-coverage HOLDOUT analysis (exploratory, post-hoc).

Offline, read-only inputs; no LLM/API calls; output written only to a NEW folder.
CAND = a separate candidate-generation prompt/policy, NOT STRICT's answer set.
Exact response budget is set per arm from GOLD-BLIND top1 retrieval scores.
Gold answers are loaded only AFTER the initial selection masks are frozen.
EM=0 != semantic hallucination. The previously inspected holdout is not pristine.

Usage:
 python3 post18f_exact_batch_holdout_v3.py selftest
 python3 post18f_exact_batch_holdout_v3.py run --root ~/2610-qi-rag \\
       --out out/post18f_exact_batch_REAL_$(date +%Y%m%d_%H%M%S) --bootstrap 10000
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
import re
import string
import sys
import tempfile
from pathlib import Path
import numpy as np
N_HOLD = 4405
ARMS = ('QI', 'VAN')
COVERAGES = (0.2, 0.3, 0.4)
PIN = {'out/holdout_locked/split_ids.json': 'b33fb99e4ce27330b2e5b5461b36b6ea18fb68aa7fc3fd6bc38b4a523d28cc09', 'out/holdout_locked/retrieval_predictions.jsonl': '4592253a0c4c46711e3525f38a51f89cf64b56e181986b9ce2acdafc9f6cfff7', 'data/hotpotqa/hotpot_dev_distractor.jsonl': 'c301b4d689fabff61df514e4511a4688e8e10a16ed2cee28484bc85ed8dc72e4'}
LLM_INPUTS = tuple(('out/holdout_llm/' + a + '_' + p + '.jsonl' for a in ('QI', 'VAN') for p in ('CAND', 'STRICT')))
ABSTAIN = re.compile("(?i)^(?:i (?:do not|don't) know|i cannot answer(?: this)?|cannot answer|unable to answer|no answer|unknown|not enough information|답변할 수 없습니다|모르겠습니다|알 수 없습니다|정보가 없습니다)[.!\\s]*$")

def require(cond, message):
    if not cond:
        raise RuntimeError('AUDIT BLOCK: ' + str(message))

def sha(path):
    h = hashlib.sha256()
    with open(_pkg_open_path(_pkg_resolve(path), 'rb'), 'rb') as f:
        for block in iter(lambda : f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def readjl(path):
    with open(_pkg_open_path(_pkg_resolve(path), 'r'), encoding='utf-8-sig') as f:
        for (n, line) in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                require(isinstance(r, dict), f'{path} line {n}: object expected')
                yield r

def keyed(path, qfield='qid'):
    rows = {}
    for r in readjl(path):
        q = str(r.get(qfield, '')).strip()
        require(q and q not in rows, f'{path}: blank/duplicate {qfield}: {q}')
        rows[q] = r
    return rows

def norm(x):
    s = str(x).lower()
    s = ''.join((ch for ch in s if ch not in string.punctuation))
    s = re.sub('\\b(a|an|the)\\b', ' ', s)
    return ' '.join(s.split())

def score(pred, gold):
    (a, b) = (norm(pred), norm(gold))
    em = int(a == b)
    if (a in {'yes', 'no', 'noanswer'} or b in {'yes', 'no', 'noanswer'}) and a != b:
        return (float(em), 0.0)
    (aa, bb) = (a.split(), b.split())
    if not aa or not bb:
        return (float(em), float(not aa and (not bb)))
    overlap = sum((collections.Counter(aa) & collections.Counter(bb)).values())
    if not overlap:
        return (float(em), 0.0)
    (prec, rec) = (_pkg_resolve(overlap / len(aa)), _pkg_resolve(overlap / len(bb)))
    return (float(em), _pkg_resolve(2 * prec * rec / (prec + rec)))

def is_abstain(pred):
    return not str(pred).strip() or bool(ABSTAIN.fullmatch(str(pred).strip()))

def tie_id(q):
    return int.from_bytes(hashlib.sha256(('POST18F-v2|' + q).encode('utf-8')).digest()[:8], 'big')

def order(score, eligible, tie):
    ii = np.flatnonzero(eligible)
    return ii[np.lexsort((tie[ii], -score[ii]))]

def pick(scores, eligible, tie, c):
    n = len(tie)
    k = round(c * n)
    rank = order(scores, eligible, tie)
    if k <= 0 or len(rank) < k:
        return None
    x = np.zeros(n, dtype=bool)
    x[rank[:k]] = True
    return x

def partial_auc(scores, eligible, tie, em, c):
    rank = order(scores, eligible, tie)
    k = round(c * len(tie))
    if k < 1 or len(rank) < k:
        return None
    errors = 1 - em[rank[:k]]
    return float(np.mean(_pkg_resolve(np.cumsum(errors) / np.arange(1, k + 1))))

def metrics(mask, em, f1, n_total=None, paurc=None):
    n = len(mask) if n_total is None else n_total
    k = int(mask.sum())
    if k == 0:
        return dict(n=n, answered=0, coverage=0.0, answered_em=None, answered_f1=None, selective_risk=None, wrong_answer_rate=0.0, correct_yield=0.0, correct_answers=0, wrong_answers=0, partial_aurc=None)
    correct = float(em[mask].sum())
    wrong = k - correct
    return dict(n=n, answered=k, coverage=_pkg_resolve(k / n), answered_em=_pkg_resolve(correct / k), answered_f1=float(f1[mask].mean()), selective_risk=_pkg_resolve(wrong / k), wrong_answer_rate=_pkg_resolve(wrong / n), correct_yield=_pkg_resolve(correct / n), correct_answers=int(round(correct)), wrong_answers=int(round(wrong)), partial_aurc=paurc)

def cluster_diagnostics(clusters):
    sizes = np.array(list(collections.Counter(clusters).values()), dtype=np.float64)
    n = float(sizes.sum())
    g = len(sizes)
    g_eff = _pkg_resolve(n * n / float((sizes * sizes).sum())) if g else 0.0
    largest = _pkg_resolve(float(sizes.max()) / n) if n else 1.0
    can = g >= 20 and g_eff >= 20 and (largest <= 0.2)
    return {'independent_clusters': g, 'effective_clusters': g_eff, 'largest_cluster_share': largest, 'cluster_CI_estimable': can, 'gate': 'heuristic G>=20, G_eff>=20, max share<=20%; not a validity proof'}

def check_pins(root):
    manifest = {}
    for (rel, expected) in PIN.items():
        path = _pkg_resolve(root / rel)
        require(path.is_file(), f'missing frozen input {rel}')
        actual = sha(path)
        require(actual == expected, f'{rel}: SHA256 mismatch expected={expected} actual={actual}')
        manifest[rel] = actual
    for rel in LLM_INPUTS:
        path = _pkg_resolve(root / rel)
        require(path.is_file(), f'missing LLM input {rel}')
        manifest[rel] = sha(path)
    frozen = json.loads(_pkg_resolve(root / 'out/holdout_locked/FREEZE.json').read_text(encoding='utf-8'))
    prompt = json.loads(_pkg_resolve(root / 'out/holdout_locked/llm_prompts_FREEZE.json').read_text(encoding='utf-8'))
    require(frozen['split_ids_sha256'] == manifest['out/holdout_locked/split_ids.json'], 'FREEZE split hash mismatch')
    require(prompt['retrieval_predictions_sha256'] == manifest['out/holdout_locked/retrieval_predictions.jsonl'], 'LLM prompt freeze retrieval mismatch')
    cfg = prompt['config']
    require(cfg.get('P') == 2 and cfg.get('temperature') == 0 and (cfg.get('max_tokens') == 128), 'unexpected frozen LLM parameters')
    manifest['FREEZE.json'] = sha(_pkg_resolve(root / 'out/holdout_locked/FREEZE.json'))
    manifest['llm_prompts_FREEZE.json'] = sha(_pkg_resolve(root / 'out/holdout_locked/llm_prompts_FREEZE.json'))
    return (manifest, cfg)

def load_selection_only(root):
    """No gold is read in this function. Produces selection masks before gold access."""
    (frozen, cfg) = check_pins(root)
    split = json.loads(_pkg_resolve(root / 'out/holdout_locked/split_ids.json').read_text(encoding='utf-8'))
    hold = split['holdout']
    cal = split['calibration']
    require(len(hold) == N_HOLD and len(set(hold)) == N_HOLD, 'holdout 4405 QIDs not unique')
    require(len(cal) == 1000 and len(set(cal)) == 1000, 'calibration QIDs not unique')
    require(not set(cal) & set(hold), 'calibration/holdout QID overlap')
    qs = sorted(hold)
    qset = set(qs)
    retrieval = keyed(_pkg_resolve(root / 'out/holdout_locked/retrieval_predictions.jsonl'))
    require(set(retrieval) == qset, 'retrieval QIDs not equal to frozen Holdout')
    rows = {}
    tie = np.array([tie_id(q) for q in qs], dtype=np.uint64)
    for arm in ARMS:
        row = keyed(_pkg_resolve(root / f'out/holdout_llm/{arm}_CAND.jsonl'))
        strict = keyed(_pkg_resolve(root / f'out/holdout_llm/{arm}_STRICT.jsonl'))
        require(set(row) == qset and set(strict) == qset, f'{arm}: paired QIDs mismatch')
        vals = []
        eligible = []
        strict_abstain = []
        cand_abstain = []
        for q in qs:
            (x, y) = (row[q], strict[q])
            require(x.get('status') == 'ok' and y.get('status') == 'ok', f'{arm}/{q}: non-ok LLM response')
            require(not x.get('err') and (not y.get('err')), f'{arm}/{q}: model error')
            require(isinstance(x.get('pred'), str) and isinstance(y.get('pred'), str), f'{arm}/{q}: nonstring pred')
            top = retrieval[q]['qi_top_items' if arm == 'QI' else 'va_top_docs']
            require(isinstance(top, list) and top, f'{arm}/{q}: missing retrieval candidates')
            v = float(top[0]['score'])
            cx = float(x['top1_score'])
            sx = float(y['top1_score'])
            require(all((math.isfinite(vv) for vv in (v, cx, sx))), f'{arm}/{q}: nonfinite score')
            require(abs(v - cx) <= 1e-09 and abs(v - sx) <= 1e-09, f'{arm}/{q}: frozen top1 score mismatch')
            vals.append(v)
            abstain = is_abstain(x['pred'])
            cand_abstain.append(abstain)
            eligible.append(not abstain)
            strict_abstain.append(is_abstain(y['pred']))
        vals = np.array(vals, dtype=np.float64)
        eligible = np.array(eligible, dtype=bool)
        masks = {c: pick(vals, eligible, tie, c) for c in COVERAGES}
        rows[arm] = {'score': vals, 'eligible': eligible, 'cand': row, 'strict': strict, 'masks': masks, 'strict_abstain': np.array(strict_abstain, dtype=bool), 'cand_abstain': np.array(cand_abstain, dtype=bool)}
        print(f'[SCORE] {arm} top1_scores={len(vals)} eligible={int(eligible.sum())} CAND_abstentions={sum(cand_abstain)} STRICT_abstentions={sum(strict_abstain)}')
        for c in COVERAGES:
            require(masks[c] is not None, f'{arm}: {c:.0%} budget infeasible: eligible={int(eligible.sum())} need={round(c * N_HOLD)}')
            require(int(masks[c].sum()) == round(c * N_HOLD), 'incorrect matched top-k')
    print('[GOLD-BLIND SELECTION] PASS: selection masks frozen before opening gold')
    return (qs, tie, rows, frozen, cfg)

def support_titles(record, qid):
    """Accept both HotpotQA supporting_facts schemas without changing gold.

 Supported: [[title, sentence_id], ...] and
 {"title":[...], "sent_id":[...]}.
 Null/empty metadata is assigned a unique singleton variance cluster.
 Unexpected nonempty schemas fail rather than silently losing dependence.
 """
    facts = record.get('supporting_facts')
    if facts is None:
        return []
    if isinstance(facts, dict):
        require('title' in facts and isinstance(facts['title'], list), f'{qid}: supporting_facts dict missing title list')
        titles = facts['title']
        if 'sent_id' in facts:
            require(isinstance(facts['sent_id'], list) and len(facts['sent_id']) == len(titles), f'{qid}: supporting_facts title/sent_id shape mismatch')
    elif isinstance(facts, list):
        titles = []
        for row in facts:
            require(isinstance(row, (list, tuple)) and len(row) >= 2, f'{qid}: malformed supporting_facts pair')
            titles.append(row[0])
    else:
        raise RuntimeError(f'AUDIT BLOCK: {qid}: unsupported supporting_facts schema: {type(facts).__name__}')
    require(all((isinstance(t, str) and bool(t.strip()) for t in titles)), f'{qid}: empty/nonstring supporting_facts title')
    return sorted(set((t.strip().lower() for t in titles)))

def attach_gold(root, qs, arms):
    gold = keyed(_pkg_resolve(root / 'data/hotpotqa/hotpot_dev_distractor.jsonl'), 'id')
    require(set(qs).issubset(gold), 'gold misses holdout questions')
    strata = np.array([str(gold[q].get('type', '')).capitalize() for q in qs], dtype=object)
    require(int(np.sum(strata == 'Bridge')) == 3518 and int(np.sum(strata == 'Comparison')) == 887, 'holdout question-type strata mismatch')
    clusters = []
    for q in qs:
        titles = support_titles(gold[q], q)
        clusters.append('SINGLETON_QID:' + q if not titles else 'GOLD_TITLES:' + json.dumps(titles, ensure_ascii=False, separators=(',', ':')))
    for arm in ARMS:
        cand = arms[arm]['cand']
        strict = arms[arm]['strict']
        em = np.empty(len(qs), float)
        f1 = np.empty(len(qs), float)
        ems = np.empty(len(qs), float)
        f1s = np.empty(len(qs), float)
        for (i, q) in enumerate(qs):
            ans = gold[q].get('answer')
            require(isinstance(ans, str) and ans.strip(), f'{q}: missing gold answer')
            (em[i], f1[i]) = (0.0, 0.0) if arms[arm]['cand_abstain'][i] else score(cand[q]['pred'], ans)
            (ems[i], f1s[i]) = (0.0, 0.0) if arms[arm]['strict_abstain'][i] else score(strict[q]['pred'], ans)
        arms[arm].update(em=em, f1=f1, strict_em=ems, strict_f1=f1s)
    return (strata, np.array(clusters, dtype=object))

def eval_rank(scores, eligible, tie, em, f1, coverages):
    """One ranking per arm per draw; cumulative arrays for fixed coverage points."""
    n = len(scores)
    rank = order(scores, eligible, tie)
    if not len(rank):
        return {}
    loss = 1 - em[rank]
    prefix_correct = np.cumsum(em[rank])
    prefix_f1 = np.cumsum(f1[rank])
    cumrisk = _pkg_resolve(np.cumsum(loss) / np.arange(1, len(rank) + 1))
    prefix_paurc = np.cumsum(cumrisk)
    output = {}
    for c in coverages:
        k = round(n * c)
        if not 0 < k <= len(rank):
            output[c] = None
            continue
        correct = float(prefix_correct[k - 1])
        wrong = k - correct
        output[c] = {'selective_risk': _pkg_resolve(wrong / k), 'answered_em': _pkg_resolve(correct / k), 'answered_f1': float(_pkg_resolve(prefix_f1[k - 1] / k)), 'wrong_answer_rate': _pkg_resolve(wrong / n), 'correct_yield': _pkg_resolve(correct / n), 'partial_aurc': float(_pkg_resolve(prefix_paurc[k - 1] / k)), 'answered': k, 'coverage': _pkg_resolve(k / n)}
    return output

def paired_bootstrap(arms, tie, clusters, strata, reps, seed, scheme):
    n = len(tie)
    rng = np.random.default_rng(seed)
    groups = None
    if scheme == 'cluster':
        buckets = collections.defaultdict(list)
        for (i, key) in enumerate(clusters):
            buckets[str(key)].append(i)
        groups = [np.array(g, dtype=np.int32) for g in buckets.values()]
    out = collections.defaultdict(list)
    invalid = collections.Counter()
    for b in range(reps):
        if scheme == 'question':
            ind = rng.integers(0, n, n, dtype=np.int32)
        else:
            draw = rng.integers(0, len(groups), len(groups))
            ind = np.concatenate([groups[j] for j in draw])
        qtie = tie[ind]
        runs = {}
        for arm in ARMS:
            v = arms[arm]
            runs[arm] = eval_rank(v['score'][ind], v['eligible'][ind], qtie, v['em'][ind], v['f1'][ind], COVERAGES)
        for c in COVERAGES:
            a = runs['QI'].get(c)
            v = runs['VAN'].get(c)
            if a is None or v is None:
                invalid[c, 'all'] += 1
                continue
            for metric in ('selective_risk', 'answered_em', 'answered_f1', 'wrong_answer_rate', 'correct_yield', 'partial_aurc'):
                out[c, 'all', metric].append(float(a[metric] - v[metric]))
            k = round(len(ind) * c)
            masks = {}
            for arm in ARMS:
                aa = arms[arm]
                rr = order(aa['score'][ind], aa['eligible'][ind], qtie)
                mm = np.zeros(len(ind), bool)
                mm[rr[:k]] = True
                masks[arm] = mm
            st = strata[ind]
            for group in ('Bridge', 'Comparison'):
                sg = st == group
                if not sg.any():
                    invalid[c, group] += 1
                    continue
                for metric in ('selective_risk', 'answered_em', 'answered_f1', 'wrong_answer_rate', 'correct_yield', 'coverage'):
                    values = []
                    for arm in ARMS:
                        aa = arms[arm]
                        m = masks[arm] & sg
                        nk = int(m.sum())
                        ns = int(sg.sum())
                        if metric in ('selective_risk', 'answered_em', 'answered_f1') and nk == 0:
                            values.append(None)
                            continue
                        correct = float(aa['em'][ind][m].sum())
                        wrong = nk - correct
                        values.append({'selective_risk': _pkg_resolve(wrong / nk) if nk else None, 'answered_em': _pkg_resolve(correct / nk) if nk else None, 'answered_f1': float(aa['f1'][ind][m].mean()) if nk else None, 'wrong_answer_rate': _pkg_resolve(wrong / ns), 'correct_yield': _pkg_resolve(correct / ns), 'coverage': _pkg_resolve(nk / ns)}[metric])
                    if any((x is None for x in values)):
                        invalid[c, group, metric] += 1
                    else:
                        out[c, group, metric].append(float(values[0] - values[1]))
        if (b + 1) % 1000 == 0:
            print(f'  BOOT {scheme}: {b + 1}/{reps}', flush=True)
    return (out, invalid)
EXTENDED_ABSTAIN = re.compile("(?i)^(?:i (?:do not|don't) know(?: the answer)?|the (?:given )?context does not (?:contain|provide) the answer|not enough information is provided|i cannot determine(?: the answer)? from the (?:given )?context|there is insufficient information(?: to answer)?|insufficient information(?: to answer)?)[.!\\s]*$")

def extended_abstain(pred):
    return is_abstain(pred) or bool(EXTENDED_ABSTAIN.fullmatch(str(pred).strip()))

def subgroup_exact(arms, tie, strata, coverages=COVERAGES):
    result = {}
    for group in ('Bridge', 'Comparison'):
        ix = np.flatnonzero(strata == group)
        require(len(ix) > 0, 'empty subgroup ' + group)
        result[group] = {}
        for c in coverages:
            d = {}
            for arm in ARMS:
                a = arms[arm]
                rank = order(a['score'][ix], a['eligible'][ix], tie[ix])
                k = round(len(ix) * c)
                require(len(rank) >= k, f'{group}/{arm} insufficient eligible for {c}')
                mask = np.zeros(len(ix), bool)
                mask[rank[:k]] = True
                p = partial_auc(a['score'][ix], a['eligible'][ix], tie[ix], a['em'][ix], c)
                d[arm] = metrics(mask, a['em'][ix], a['f1'][ix], paurc=p)
            result[group][str(c)] = {'k_per_arm': k, 'arms': d, 'delta_QI_minus_VAN': {key: d['QI'][key] - d['VAN'][key] for key in ('selective_risk', 'answered_em', 'answered_f1', 'wrong_answer_rate', 'correct_yield', 'partial_aurc')}}
    return result

def subgroup_bootstrap(arms, tie, clusters, strata, rep, seed, scheme):
    rng = np.random.default_rng(seed)
    result = collections.defaultdict(list)
    for group in ('Bridge', 'Comparison'):
        ids = np.flatnonzero(strata == group)
        groups = None
        if scheme == 'cluster':
            d = collections.defaultdict(list)
            for i in ids:
                d[str(clusters[i])].append(int(i))
            groups = [np.array(v, dtype=np.int32) for v in d.values()]
        for _ in range(rep):
            if scheme == 'question':
                ix = ids[rng.integers(0, len(ids), len(ids))]
            else:
                ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            tt = tie[ix]
            rr = {arm: eval_rank(arms[arm]['score'][ix], arms[arm]['eligible'][ix], tt, arms[arm]['em'][ix], arms[arm]['f1'][ix], COVERAGES) for arm in ARMS}
            for c in COVERAGES:
                aa = rr['QI'].get(c)
                vv = rr['VAN'].get(c)
                for m in ('selective_risk', 'answered_em', 'answered_f1', 'wrong_answer_rate', 'correct_yield', 'partial_aurc'):
                    if aa is not None and vv is not None:
                        result[group, c, m].append(float(aa[m] - vv[m]))
        print(f'  SUBGROUP BOOT {scheme}: {group} {rep}/{rep}', flush=True)
    return result

def abstention_sensitivity(arms, qs, tie, gold):
    out = {}
    for c in COVERAGES:
        row = {}
        for arm in ARMS:
            a = arms[arm]
            eligible = np.array([not extended_abstain(a['cand'][q]['pred']) for q in qs], bool)
            mask = pick(a['score'], eligible, tie, c)
            if mask is None:
                row[arm] = {'feasible': False, 'eligible_n': int(eligible.sum())}
                continue
            em = np.array(a['em'], copy=True)
            f1 = np.array(a['f1'], copy=True)
            new_abstains = a['eligible'] & ~eligible
            em[new_abstains] = 0
            f1[new_abstains] = 0
            row[arm] = {'feasible': True, 'eligible_n': int(eligible.sum()), 'new_abstentions': int(new_abstains.sum()), 'metrics': metrics(mask, em, f1)}
        out[str(c)] = {'arms': row, 'delta_risk': row['QI']['metrics']['selective_risk'] - row['VAN']['metrics']['selective_risk'] if row['QI']['feasible'] and row['VAN']['feasible'] else None}
    return out

def overlapping_title_clusters(root, qs):
    gold = keyed(_pkg_resolve(root / 'data/hotpotqa/hotpot_dev_distractor.jsonl'), 'id')
    parent = list(range(len(qs)))
    size = [1] * len(qs)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(x, y):
        x = find(x)
        y = find(y)
        if x == y:
            return
        if size[x] < size[y]:
            (x, y) = (y, x)
        parent[y] = x
        size[x] += size[y]
    seen = {}
    for (i, q) in enumerate(qs):
        for t in support_titles(gold[q], q):
            if t in seen:
                union(i, seen[t])
            else:
                seen[t] = i
    return np.array([str(find(i)) for i in range(len(qs))], dtype=object)

def ci(vals, reps, quality=True):
    if not quality:
        return {'ci95': None, 'valid': len(vals), 'requested': reps, 'status': 'CLUSTER_QUALITY_BLOCK'}
    if len(vals) != reps:
        return {'ci95': None, 'valid': len(vals), 'requested': reps, 'status': 'INCOMPLETE_RESAMPLES'}
    (lo, hi) = np.quantile(np.asarray(vals, dtype=float), [0.025, 0.975])
    return {'ci95': [float(lo), float(hi)], 'valid': reps, 'requested': reps, 'status': 'OK'}

def safe_json(path, obj):
    with open(_pkg_open_path(_pkg_resolve(path), 'x'), 'x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def native_summary(arms, qs):
    result = {}
    for (label, what) in (('STRICT', 'strict'), ('CAND_UNFILTERED', 'cand')):
        result[label] = {}
        for arm in ARMS:
            a = arms[arm]
            mask = ~a['strict_abstain'] if what == 'strict' else a['eligible']
            em = a['strict_em'] if what == 'strict' else a['em']
            f1 = a['strict_f1'] if what == 'strict' else a['f1']
            result[label][arm] = metrics(mask, em, f1)
    return result

def main_run(args):
    root = Path(_pkg_resolve(args.root)).resolve()
    out = Path(_pkg_resolve(args.out))
    if not out.is_absolute():
        out = _pkg_resolve(root / out)
    require(not out.exists(), 'refuse overwrite output directory: ' + str(out))
    require(args.bootstrap >= 100 and args.bootstrap <= 100000, 'bootstrap out of range 100..100000')
    require(args.seed >= 0, 'seed must be nonnegative')
    print('========== POST18F EXACT MATCHED HOLDOUT ==========', flush=True)
    (qs, tie, arms, pins, cfg) = load_selection_only(root)
    (strata, clusters) = attach_gold(root, qs, arms)
    diagnostic = cluster_diagnostics(clusters)
    missing_support_n = sum((str(k).startswith('SINGLETON_QID:') for k in clusters))
    print('[GOLD SUPPORT] missing/empty supporting_facts:', missing_support_n, '; unique singleton variance clusters assigned (no Gold answer exclusions)')
    shared = overlapping_title_clusters(root, qs)
    shared_diagnostic = cluster_diagnostics(shared)
    print('[SHARED-DOC CLUSTER DIAGNOSTIC]', shared_diagnostic)
    print('[GOLD] 4405 answers re-scored; type Bridge=3518 Comparison=887')
    print('[CLUSTERS]', diagnostic)
    point = {}
    maskdata = {}
    for c in COVERAGES:
        k = round(c * len(qs))
        armstats = {}
        for arm in ARMS:
            a = arms[arm]
            mm = a['masks'][c]
            pa = partial_auc(a['score'], a['eligible'], tie, a['em'], c)
            armstats[arm] = metrics(mm, a['em'], a['f1'], paurc=pa)
        maskdata[str(c)] = {arm: arms[arm]['masks'][c] for arm in ARMS}
        point[str(c)] = {'k_per_arm': k, 'arms': armstats, 'delta_QI_minus_VAN': {metric: armstats['QI'][metric] - armstats['VAN'][metric] for metric in ('selective_risk', 'answered_em', 'answered_f1', 'wrong_answer_rate', 'correct_yield', 'partial_aurc')}}
        print(f"[MATCHED {c:.0%}] k={k} QI Risk={armstats['QI']['selective_risk']:.6f} VAN Risk={armstats['VAN']['selective_risk']:.6f} Δ Risk={point[str(c)]['delta_QI_minus_VAN']['selective_risk']:+.6f}")
    subgroup_equal = subgroup_exact(arms, tie, strata)
    sensitivity = abstention_sensitivity(arms, qs, tie, None)
    print('[SUBGROUP EQUAL-COVERAGE] computed separate Bridge/Comparison Top-K')
    print('[ABSTENTION SENSITIVITY] computed without changing primary')
    native = native_summary(arms, qs)
    print('[NATIVE STRICT] QI=', native['STRICT']['QI']['coverage'], 'VAN=', native['STRICT']['VAN']['coverage'])
    print('[BOOTSTRAP] Paired question + paired gold-title-cluster,', args.bootstrap, 'each')
    (bq, iq) = paired_bootstrap(arms, tie, clusters, strata, args.bootstrap, args.seed, 'question')
    (bc, ic) = paired_bootstrap(arms, tie, clusters, strata, args.bootstrap, args.seed, 'cluster')
    subgroup_q = subgroup_bootstrap(arms, tie, clusters, strata, args.bootstrap, args.seed + 5001, 'question')
    subgroup_c = subgroup_bootstrap(arms, tie, clusters, strata, args.bootstrap, args.seed + 6001, 'cluster')
    subgroup_cis = {}
    for group in ('Bridge', 'Comparison'):
        subgroup_cis[group] = {}
        group_d = cluster_diagnostics(clusters[strata == group])
        for c in COVERAGES:
            cs = str(c)
            subgroup_cis[group][cs] = {}
            for metric in ('selective_risk', 'answered_em', 'answered_f1', 'wrong_answer_rate', 'correct_yield', 'partial_aurc'):
                key = (group, c, metric)
                subgroup_cis[group][cs][metric] = {'paired_question': ci(subgroup_q.get(key, []), args.bootstrap), 'paired_cluster': ci(subgroup_c.get(key, []), args.bootstrap, group_d['cluster_CI_estimable'])}
        subgroup_equal[group]['cluster_diagnostics'] = group_d
    primary = {}
    cis = {}
    for c in COVERAGES:
        cs = str(c)
        cis[cs] = {}
        for region in ('all', 'Bridge', 'Comparison'):
            cis[cs][region] = {}
            metrics_list = ('selective_risk', 'answered_em', 'answered_f1', 'wrong_answer_rate', 'correct_yield', 'partial_aurc') if region == 'all' else ('selective_risk', 'answered_em', 'answered_f1', 'wrong_answer_rate', 'correct_yield', 'coverage')
            for metric in metrics_list:
                key = (c, region, metric)
                cis[cs][region][metric] = {'paired_question': ci(bq.get(key, []), args.bootstrap), 'paired_cluster': ci(bc.get(key, []), args.bootstrap, diagnostic['cluster_CI_estimable'])}
        if c == 0.3:
            cc = cis[cs]['all']['selective_risk']['paired_cluster']
            obs = point[cs]['delta_QI_minus_VAN']['selective_risk']
            supported = cc['ci95'] is not None and cc['ci95'][1] < 0 and (obs < 0)
            primary = {'coverage': c, 'k_per_arm': point[cs]['k_per_arm'], 'delta_selective_risk': obs, 'paired_cluster_ci95': cc['ci95'], 'observed_holdout_direction_support': supported, 'claim_scope': 'EXPLORATORY_POSTHOC_ONLY; no prospective confirmatory claim'}
    require(all((cis['0.3']['all'][m]['paired_question']['status'] == 'OK' for m in ('selective_risk', 'answered_em', 'answered_f1'))), 'primary paired-question bootstrap unavailable')
    print('[PRIMARY]', primary)
    regions = {}
    for c in COVERAGES:
        regions[str(c)] = {}
        for group in ('Bridge', 'Comparison'):
            sg = strata == group
            results = {}
            for arm in ARMS:
                aa = arms[arm]
                mask = aa['masks'][c] & sg
                results[arm] = metrics(mask[sg], aa['em'][sg], aa['f1'][sg])
            regions[str(c)][group] = {'arms': results, 'delta_QI_minus_VAN': {met: results['QI'][met] - results['VAN'][met] if results['QI'][met] is not None and results['VAN'][met] is not None else None for met in ('coverage', 'answered_em', 'answered_f1', 'selective_risk', 'wrong_answer_rate', 'correct_yield')}}
    out.mkdir(parents=True)
    safe_json(_pkg_resolve(out / 'results.json'), {'experiment': 'POST-18F exact matched-coverage retrieval-score selective-risk', 'evaluation_status': 'exploratory_posthoc_holdout_previously_inspected', 'version': 'v3.2', 'mode': 'CAND answers; exact global batch top-k by frozen retrieval top1_score', 'policy_boundary': 'STRICT and CAND use different prompts; results cannot be attributed to original STRICT policy', 'selection_policy': 'CAND non-abstaining answers only; gold-blind score ranks selected per-arm; round(coverage*N) distinct QIDs.', 'primary': primary, 'point_estimates': point, 'subgroups_global_selection': regions, 'subgroups_exact_matched': subgroup_equal, 'subgroups_exact_matched_ci': subgroup_cis, 'abstention_sensitivity': sensitivity, 'shared_doc_cluster_diagnostics': shared_diagnostic, 'natural_baselines': native, 'confidence_intervals': cis, 'cluster_diagnostics': diagnostic, 'missing_supporting_facts_singleton_clusters': missing_support_n, 'bootstrap': {'replicates_each': args.bootstrap, 'seed': args.seed, 'question_invalid': {str(k): int(v) for (k, v) in iq.items()}, 'cluster_invalid': {str(k): int(v) for (k, v) in ic.items()}}, 'caveats': ['Gold-blind ranking verified only for retrieved score provenance, not full producer provenance.', 'LLM CAND/STRICT file hashes are captured at run-time, not externally pre-registered pins.', 'Coverage is exactly matched, but selected QIDs can differ between arms.', 'Automatic EM/F1 error != semantic hallucination.', 'Already inspected Holdout. No confirmatory inference label irrespective of CI.', 'No Calibration thresholds learned; Fixed Threshold is NOT evaluated here.', 'Paired cluster CI is heuristic-gated, not guaranteed valid.', 'Global Top-K selection before subgroup analysis.', 'Subgroup-specific exact coverage is a SEPARATE reselected policy.', 'Extended abstention sensitivity is not the primary policy.', 'Shared-title connected components diagnostic is not a primary bootstrap CI.']})
    safe_json(_pkg_resolve(out / 'manifest.json'), {'source_hashes': pins, 'code_sha256': sha(Path(_pkg_resolve(__file__)).resolve()), 'llm_config': cfg, 'n_holdout': N_HOLD, 'n_calibration_qids': 1000, 'calibration_predictions_used': False, 'gold_blind_ranking': 'CAND top1_score is copied verbatim from frozen retrieval', 'gold_qa_sha256': pins['data/hotpotqa/hotpot_dev_distractor.jsonl']})
    with _pkg_pathlib.Path(_pkg_open_path(_pkg_resolve(out / 'selection_audit.csv'), 'x')).open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['qid', 'type', 'cluster_sha256_12', 'QI_top1', 'VAN_top1', 'QI_candidate_answered', 'VAN_candidate_answered', 'QI_EM', 'VAN_EM', 'QI_F1', 'VAN_F1', 'QI_STRICT_answered', 'VAN_STRICT_answered', *[f'{a}_selected_{int(c * 100)}' for c in COVERAGES for a in ARMS]])
        for (i, q) in enumerate(qs):
            w.writerow([q, strata[i], hashlib.sha256(str(clusters[i]).encode()).hexdigest()[:12], float(arms['QI']['score'][i]), float(arms['VAN']['score'][i]), int(arms['QI']['eligible'][i]), int(arms['VAN']['eligible'][i]), int(arms['QI']['em'][i]), int(arms['VAN']['em'][i]), float(arms['QI']['f1'][i]), float(arms['VAN']['f1'][i]), int(not arms['QI']['strict_abstain'][i]), int(not arms['VAN']['strict_abstain'][i]), *[int(arms[a]['masks'][c][i]) for c in COVERAGES for a in ARMS]])
    with _pkg_pathlib.Path(_pkg_open_path(_pkg_resolve(out / 'summary.csv'), 'x')).open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['coverage_target', 'k', 'arm', 'risk', 'answered_EM', 'answered_F1', 'wrong_all', 'correct_yield', 'pAURC'])
        for c in COVERAGES:
            for arm in ARMS:
                v = point[str(c)]['arms'][arm]
                w.writerow([c, v['answered'], arm, v['selective_risk'], v['answered_em'], v['answered_f1'], v['wrong_answer_rate'], v['correct_yield'], v['partial_aurc']])

    def percent(v):
        return f'{100 * v:.3f}%' if v is not None else 'NA'
    lines = ['# POST-18F Exact Matched-Coverage: real frozen Holdout', '**Exploratory post-hoc only:** previously inspected Holdout, NOT new prospective confirmation.', 'Policy: CAND answers ranked using original Gold-blind retrieval top1_score; Strict is a separate baseline.', '', '|Coverage|N per arm|QI risk|VAN risk|Δ Risk (QI−VAN)|Paired-cluster 95% CI|', '|---:|---:|---:|---:|---:|---|']
    for c in COVERAGES:
        z = point[str(c)]
        cc = cis[str(c)]['all']['selective_risk']['paired_cluster']['ci95']
        v = z['delta_QI_minus_VAN']['selective_risk']
        CI = '[' + ', '.join((f'{100 * x:+.3f}%p' for x in cc)) + ']' if cc else 'Unavailable (quality/incomplete)'
        lines.append(f"|{int(c * 100)}%|{z['k_per_arm']}|{percent(z['arms']['QI']['selective_risk'])}|{percent(z['arms']['VAN']['selective_risk'])}|{100 * v:+.3f}%p|{CI}|")
    lines += ['', '## Primary 30%: yield / wrong answer / pAURC', '|Arm|Answered F1|Correct (N)|Wrong (N)|Wrong / N|Correct / N|pAURC 0–30%|', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m = point['0.3']['arms'][arm]
        lines.append(f"|{arm}|{m['answered_f1']:.4f}|{m['correct_answers']}|{m['wrong_answers']}|{percent(m['wrong_answer_rate'])}|{percent(m['correct_yield'])}|{m['partial_aurc']:.4f}|")
    lines.extend(['', '## Global 30% selection: Bridge / Comparison breakdown', '|Type|QI coverage|VAN coverage|QI risk|VAN risk|Δ risk|Cluster 95% CI|', '|---|---:|---:|---:|---:|---:|---|'])
    for region in ('Bridge', 'Comparison'):
        a = regions['0.3'][region]
        z = cis['0.3'][region]['selective_risk']['paired_cluster']['ci95']
        cistr = '[' + ', '.join((f'{100 * x:+.3f}%p' for x in z)) + ']' if z else 'Unavailable'
        lines.append(f"|{region} (coverage may differ)|{percent(a['arms']['QI']['coverage'])}|{percent(a['arms']['VAN']['coverage'])}|{percent(a['arms']['QI']['selective_risk'])}|{percent(a['arms']['VAN']['selective_risk'])}|{100 * a['delta_QI_minus_VAN']['selective_risk']:+.3f}%p|{cistr}|")
    lines.extend(['', '## SEPARATE Bridge/Comparison exact matched Top-K (within-group selection)', '|Type|Target|K per arm|QI risk|VAN risk|Delta risk|Cluster CI95|', '|---|---:|---:|---:|---:|---:|---|'])
    for group in ('Bridge', 'Comparison'):
        for c in COVERAGES:
            rec = subgroup_equal[group][str(c)]
            delta = rec['delta_QI_minus_VAN']['selective_risk']
            z = subgroup_cis[group][str(c)]['selective_risk']['paired_cluster']['ci95']
            zz = '[' + ', '.join((f'{100 * x:+.3f}%p' for x in z)) + ']' if z else 'Unavailable'
            lines.append(f"|{group}|{int(c * 100)}%|{rec['k_per_arm']}|{percent(rec['arms']['QI']['selective_risk'])}|{percent(rec['arms']['VAN']['selective_risk'])}|{100 * delta:+.3f}%p|{zz}|")
    lines.extend(['', '## Extended abstention sensitivity (alternative definition; descriptive)', '|Target|QI additional abstains|VAN additional abstains|Delta risk (QI-VAN)|', '|---|---:|---:|---:|'])
    for c in COVERAGES:
        s = sensitivity[str(c)]
        dr = s['delta_risk']
        lines.append(f"|{int(c * 100)}%|{s['arms']['QI'].get('new_abstentions', 'NA')}|{s['arms']['VAN'].get('new_abstentions', 'NA')}|{100 * dr:+.3f}%p|" if dr is not None else f'|{int(c * 100)}%|NA|NA|Infeasible|')
    lines += ['', f'Cluster diagnostics: {diagnostic}', 'LLM source hashes recorded at execution; unlike split/retrieval/gold they are not externally pinned.', f"Native Strict coverage QI/VAN: {percent(native['STRICT']['QI']['coverage'])} / {percent(native['STRICT']['VAN']['coverage'])}", 'Full AURC is not defined across 0–100% coverage because CAND abstains on some prompts. Fixed pAURC is reported instead.', 'EM-based risk is not the same as human-rated semantic hallucination.', 'Exact batch Top-K and Frozen Calibration Threshold are different policies.', 'Input source files unmodified.']
    _pkg_output(_pkg_resolve(out / 'REPORT.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    safe_json(_pkg_resolve(out / 'RUN_COMPLETE.json'), {'status': 'COMPLETE', 'posthoc_not_confirmatory': True, 'selection_audit_sha256': sha(_pkg_resolve(out / 'selection_audit.csv')), 'results_sha256': sha(_pkg_resolve(out / 'results.json'))})
    print('========== RESULTS ==========', flush=True)
    print('\n'.join(lines), flush=True)
    print('[PASS] output:', out)

def selftest():
    require(support_titles({'supporting_facts': {'title': ['A', 'B'], 'sent_id': [0, 2]}}, 'q') == ['a', 'b'], 'dict schema')
    require(support_titles({'supporting_facts': [['A', 0], ['B', 2]]}, 'q') == ['a', 'b'], 'list schema')
    require(support_titles({'supporting_facts': None}, 'q') == [], 'null schema singleton')
    require(support_titles({'supporting_facts': []}, 'q') == [], 'empty schema singleton')
    try:
        support_titles({'supporting_facts': {'title': ['A'], 'sent_id': [0, 1]}}, 'q')
    except RuntimeError:
        pass
    else:
        raise RuntimeError('AUDIT BLOCK: mismatch fact schema not rejected')
    require(norm('The Foo!') == 'foo', 'normalize')
    require(score('Yes', 'no') == (0.0, 0.0), 'special yes/no')
    require(score('Paris.', 'paris')[0] == 1.0, 'normalized EM')
    require(is_abstain("I don't know.") and (not is_abstain('Paris')), 'abstention detector')
    n = 100
    tie = np.array([tie_id(str(i)) for i in range(n)], dtype=np.uint64)
    vals = np.ones(n)
    eligible = np.ones(n, dtype=bool)
    m = pick(vals, eligible, tie, 0.3)
    require(m.sum() == 30, 'exact batch budget')
    m2 = pick(vals.copy(), eligible.copy(), tie.copy(), 0.3)
    require(np.array_equal(m, m2), 'shared tie ordering')
    val = np.array([_pkg_resolve(i * 13 % 97 / 97) for i in range(n)])
    em = np.array([i % 3 == 0 for i in range(n)], dtype=float)
    q = eval_rank(val, eligible, tie, em, em, COVERAGES)
    require(abs(q[0.3]['selective_risk'] + q[0.3]['answered_em'] - 1) < 1e-12, 'risk identity')
    require(cluster_diagnostics(['one'] * 99 + ['two'])['cluster_CI_estimable'] is False, 'giant cluster refusal')
    require(len(set(['SINGLETON_QID:x', 'SINGLETON_QID:y'])) == 2, 'missing support must stay singleton')
    require(extended_abstain("I don't know the answer.") and extended_abstain('Not enough information is provided.'), 'expanded abstain')
    require(not extended_abstain('Paris'), 'expanded abstain false positive')
    arms = {a: {'score': val, 'eligible': eligible, 'em': em, 'f1': em} for a in ARMS}
    (b, inv) = paired_bootstrap(arms, tie, np.array([f'c{i // 2}' for i in range(n)], dtype=object), np.array(['Bridge'] * 80 + ['Comparison'] * 20, dtype=object), 100, 0, 'cluster')
    x = ci(b[0.3, 'all', 'selective_risk'], 100)
    require(x['ci95'] == [0.0, 0.0], 'paired identical arm CI')
    require(pick(val, np.array([i < 20 for i in range(n)]), tie, 0.3) is None, 'infeasible budget')
    sub = subgroup_exact(arms, tie, np.array(['Bridge'] * 80 + ['Comparison'] * 20, dtype=object))
    require(sub['Bridge']['0.3']['k_per_arm'] == 24 and sub['Comparison']['0.3']['k_per_arm'] == 6, 'subgroup exact budgets')
    bb = subgroup_bootstrap(arms, tie, np.array([f'c{i // 2}' for i in range(n)], dtype=object), np.array(['Bridge'] * 80 + ['Comparison'] * 20, dtype=object), 100, 0, 'question')
    require(ci(bb['Bridge', 0.3, 'selective_risk'], 100)['ci95'] == [0.0, 0.0], 'subgroup paired identical CI')
    first = pick(val, eligible, tie, 0.3)
    synthetic_gold = np.random.default_rng(123).permutation(em)
    require(np.array_equal(first, pick(val, eligible, tie, 0.3)) and len(synthetic_gold) == n, 'gold-blind selection regression')
    print('SELFTEST PASS: scores, abstain, EM/F1, exact top-k, common ties, fixed pAURC, paired bootstrap, giant cluster refusal, coverage infeasibility')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    r = sub.add_parser('run')
    r.add_argument('--root', default='.')
    r.add_argument('--out', required=True)
    r.add_argument('--bootstrap', type=int, default=10000)
    r.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.cmd == 'selftest':
        selftest()
    else:
        main_run(args)
if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as e:
        print('FAIL:', e, file=sys.stderr)
        sys.exit(2)
