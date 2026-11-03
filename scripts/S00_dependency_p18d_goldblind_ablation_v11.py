"""POST-18D v11: gold-blind frozen Top-8 document-selection bottleneck audit.

Exploratory analysis only: the 2,000 POST-18A evaluation QIDs have already been
examined while developing these selection policies. No confirmatory p-values or
claims of index superiority are warranted from these data.

Policies (new names, NEVER replace v5 D0-D4 or POST-18E v5 D2):
 G0: original QI top-question-first, P=2 (byte-frozen reference)
 G1: max-score document dedup (negative control, mathematically G0-equivalent)
 G2: rank-weighted document voting (exact v5 D1 selection rule)
 G3: relevance-bounded Top-1-question escape (fixed --score-drop)
 G4: unbounded Top-1-question escape (stress/sensitivity)
 V0: frozen original Vanilla policy (reference only)

Selection gets only frozen ranked indexed-question IDs/scores and item->document
key links. Gold labels/supporting facts are introduced AFTER selection, solely
for evaluation. No embeddings or LLMs are recomputed. Output is non-overwriting
and records fingerprints. Default --score-drop=.05 is an exploratory heuristic,
NOT a calibrated or preregistered threshold.
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import tempfile
from pathlib import Path
EXPECTED_SHA = '4976962eacc1d7a1e8398657939729795a252e4f27e44f2693b3cc13709e5872'
EXPECTED_RETRIEVAL_SHA = '8db2d9c3e1bbfc6d96cde34ed6ecad70894ff320d27b154a8a819114ad73fc9f'
EXPECTED_DATASET_SHA = '86558b9e2e8b94eb6702b3a84e6a18188cf92e9d62097e03c20090bd1ff7167e'
EXPECTED_N = 2000
EXPECTED_D0_FULL = 262
EXPECTED_D4_FULL = 481
TOKEN = re.compile('(?u)\\b\\w+\\b')
POLICIES = ('G0', 'G1', 'G2', 'G3', 'G4', 'V0')
ALTERNATIVES = ('G1', 'G2', 'G3', 'G4', 'V0')
METRICS = ('recall', 'full', 'sentence_full', 'sentence_strict_full')

def readjl(path):
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda : f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def unique(xs):
    (seen, out) = (set(), [])
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def strict_index(xs, key):
    out = {}
    for x in xs:
        k = str(x[key])
        if k in out:
            raise ValueError(f'duplicate {key}={k}')
        out[k] = x
    return out

def tokens(s):
    return TOKEN.findall(str(s).casefold())

def choose_policies(ranked_items, item_doc_keys, p=2, score_drop=0.05, return_audit=False):
    """Gold-blind P=2 selection; no question text, labels, answers, support.

    Policy G3: choose the original highest-ranked document, then choose the
    highest-scoring candidate OUTSIDE the Top-1 question's linked document set
    only if its maximum source-question score >= top_question_score-score_drop.
    Otherwise use highest-ranked eligible remaining document. If no remaining
    doc meets the bound, fill P=2 with a ranked candidate and explicitly mark
    bound_infeasible_fallback; never misrepresent fallback as a hard bound.

    G1 is a rank-stable mathematical negative control. A tolerated <=1e-8
    score inversion is clamped for G1 ordering ONLY. The frozen raw scores
    remain unchanged for G2/G3/G4 and all saved input hashes.

    return_audit=True returns (policy_document_keys, gold-blind_selection_audit).
    """
    if p != 2 or isinstance(p, bool):
        raise ValueError('POST-18D v11 fixes P=2 exactly')
    if isinstance(score_drop, bool) or not isinstance(score_drop, (int, float)) or (not math.isfinite(score_drop)) or (score_drop < 0):
        raise ValueError('score_drop must be non-negative finite')
    if not isinstance(ranked_items, list) or not ranked_items:
        raise ValueError('ranked_items must be a nonempty list')
    (pool, sources, maxscore) = ([], collections.defaultdict(set), {})
    canonical_maxscore = {}
    first_keys = []
    prev = float('inf')
    monotone_prev = float('inf')
    tiny_inversions = 0
    for (rank, (j, score)) in enumerate(ranked_items):
        if isinstance(j, bool) or not isinstance(j, int) or j < 0 or (j >= len(item_doc_keys)):
            raise ValueError(f'invalid item ID at rank={rank}')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or (not math.isfinite(score)):
            raise ValueError(f'invalid candidate score at rank={rank}')
        if score > prev:
            if score - prev > 1e-08:
                raise ValueError('ranking score inversion exceeds 1e-8 tolerance')
            tiny_inversions += 1
        prev = score
        monotone_score = min(score, monotone_prev)
        monotone_prev = monotone_score
        keys = unique(item_doc_keys[j])
        if rank == 0:
            first_keys = keys
        for key in keys:
            if key not in maxscore:
                pool.append(key)
                maxscore[key] = score
                canonical_maxscore[key] = monotone_score
            else:
                maxscore[key] = max(maxscore[key], score)
                canonical_maxscore[key] = max(canonical_maxscore[key], monotone_score)
            sources[key].add(rank)
    audit = {'top1_question_score': float(ranked_items[0][1]), 'tiny_score_inversions': tiny_inversions, 'G3_status': 'unassigned', 'G3_bound_relaxed': 0, 'G3_eligible_outside_n': 0, 'G3_eligible_remaining_n': 0, 'G3_has_outside_candidate': 0, 'G3_chosen_second_score': None, 'G3_chosen_second_gap': None, 'G3_bound_satisfied': None, 'G3_top1_escape': 0, 'G4_status': 'unassigned', 'G4_top1_escape': 0, 'G3_eligible_outside_best_score': None}
    if not pool:
        out = {key: [] for key in POLICIES if key != 'V0'}
        audit['G3_status'] = 'no_candidates'
        audit['G4_status'] = 'no_candidates'
        return (out, audit) if return_audit else out
    earliest = {key: i for (i, key) in enumerate(pool)}
    g0 = pool[:p]
    g1 = sorted(pool, key=lambda k: (-canonical_maxscore[k], earliest[k]))[:p]
    if g1 != g0:
        raise AssertionError('G1 stable dedup failed G0 mathematical equivalence')
    votes = {k: sum((_pkg_resolve(1.0 / (rank + 1)) for rank in sources[k])) for k in pool}
    g2 = sorted(pool, key=lambda k: (-votes[k], earliest[k]))[:p]
    firstset = set(first_keys)
    bestfirst = pool[0]
    remain = pool[1:]
    scorefloor = ranked_items[0][1] - score_drop
    eligible = [k for k in remain if maxscore[k] >= scorefloor]
    outside_eligible = [k for k in eligible if k not in firstset]
    outside_all = [k for k in remain if k not in firstset]
    audit['G3_eligible_remaining_n'] = len(eligible)
    audit['G3_eligible_outside_n'] = len(outside_eligible)
    audit['G3_has_outside_candidate'] = int(bool(outside_all))
    if outside_eligible:
        audit['G3_eligible_outside_best_score'] = float(max((maxscore[k] for k in outside_eligible)))

    def high_first(candidates):
        return min(candidates, key=lambda k: (-maxscore[k], earliest[k]))
    if not remain:
        g3 = [bestfirst]
        g4 = [bestfirst]
        audit['G3_status'] = 'single_document'
        audit['G4_status'] = 'single_document'
    else:
        if outside_eligible:
            second3 = high_first(outside_eligible)
            audit['G3_status'] = 'bounded_top1_escape'
        elif eligible:
            second3 = high_first(eligible)
            audit['G3_status'] = 'no_eligible_outside'
        else:
            second3 = high_first(remain)
            audit['G3_status'] = 'bound_infeasible_fallback'
            audit['G3_bound_relaxed'] = 1
        g3 = [bestfirst, second3]
        second4 = high_first(outside_all) if outside_all else high_first(remain)
        g4 = [bestfirst, second4]
        audit['G4_status'] = 'unbounded_top1_escape' if outside_all else 'no_outside_candidate'
        audit['G3_chosen_second_score'] = float(maxscore[second3])
        audit['G3_chosen_second_gap'] = float(ranked_items[0][1] - maxscore[second3])
        audit['G3_bound_satisfied'] = int(maxscore[second3] >= scorefloor)
        audit['G3_top1_escape'] = int(second3 not in firstset)
        audit['G4_top1_escape'] = int(second4 not in firstset)
        if audit['G3_bound_relaxed'] != 1 - audit['G3_bound_satisfied']:
            raise AssertionError('G3 relaxation audit inconsistent')
        if audit['G3_top1_escape'] and (not outside_eligible) and (not audit['G3_bound_relaxed']):
            raise AssertionError('escaped without an eligible outside candidate')
        if outside_eligible and (not audit['G3_top1_escape']):
            raise AssertionError('failed to escape despite eligible outside candidate')
    out = {'G0': g0, 'G1': g1, 'G2': g2, 'G3': g3, 'G4': g4}
    for (name, keys) in out.items():
        if len(keys) != min(p, len(pool)) or len(keys) != len(set(keys)):
            raise AssertionError('invalid selected document count: ' + name)
    return (out, audit) if return_audit else out

def classify_top1_gold_bottleneck(first_keys, candidate_keys, baseline_keys, docmap, gold_titles):
    """EVALUATION ONLY, never pass labels/results to choose_policies.

    Separates two disjoint reasons for an Oracle-feasible G0 miss:
    * INTERNAL: both gold titles already linked to the first indexed question;
      changing the choice/order WITHIN that question could retrieve them.
    * EXTERNAL: at least one gold title has no linked document in Top1;
      some document OUTSIDE Top1 is required to retrieve both gold titles.

    'Feasible' is a title-level candidate oracle, not semantic evidence or a
    claim that G2/G3 can in fact repair the miss. Duplicate corpus titles
    are not treated as unique document keys.
    """
    gold = set(gold_titles)
    if len(gold) != 2:
        raise ValueError('Hotpot title oracle requires two distinct gold titles')
    first = set(first_keys)
    pool = set(candidate_keys)
    chosen = set(baseline_keys)
    if not chosen.issubset(pool) or not first.issubset(pool):
        raise ValueError('diagnostic selection/first keys not within candidate pool')
    if any((k not in docmap for k in pool)):
        raise ValueError('diagnostic candidate key missing from corpus')
    first_titles = {docmap[k]['title'] for k in first}
    pool_titles = {docmap[k]['title'] for k in pool}
    baseline_titles = {docmap[k]['title'] for k in chosen}
    in_first = len(gold & first_titles)
    oracle_possible = gold.issubset(pool_titles)
    baseline_both = gold.issubset(baseline_titles)
    exposed = len(first) >= 2
    miss = bool(exposed and oracle_possible and (not baseline_both))
    internal = bool(miss and in_first == 2)
    external = bool(miss and in_first < 2)
    if int(internal) + int(external) != int(miss):
        raise AssertionError('internal/external bottleneck partition failed')
    return {'top1_gold_titles_available_n': in_first, 'top1_gold_titles_missing_n': 2 - in_first, 'top1_recoverable_miss': int(miss), 'top1_internal_selection_miss': int(internal), 'top1_external_evidence_needed': int(external)}

def diagnostic_failure_subsets(rows):
    """Descriptive ONLY: selection on G0 outcome invalidates net-effect CI.

    These subsets are conditioned on gold/oracle and a baseline failure;
    therefore no risk difference, paired confidence interval or p-value
    is reported here. Primary inferential effects are estimated using ALL
    questions and baseline-outcome-independent structural subgroups.
    """
    definitions = {'oracle_two_titles_G0_miss': lambda r: bool(r['oracle_both'] and (not r['G0_full'])), 'top1_multi_oracle_two_G0_miss': lambda r: bool(r['top1_recoverable_miss']), 'top1_internal_selection_miss': lambda r: bool(r['top1_internal_selection_miss']), 'top1_external_evidence_needed': lambda r: bool(r['top1_external_evidence_needed'])}
    result = {}
    for (name, test) in definitions.items():
        subset = [r for r in rows if test(r)]
        n = len(subset)
        policies = {}
        for p in ('G1', 'G2', 'G3', 'G4'):
            both = sum((r[p + '_full'] for r in subset))
            gained = sum((r[p + '_recall'] > r['G0_recall'] for r in subset))
            equal = sum((r[p + '_recall'] == r['G0_recall'] for r in subset))
            lost = sum((r[p + '_recall'] < r['G0_recall'] for r in subset))
            policies[p] = {'both_gold_recovered_n': both, 'both_gold_recovered_fraction': _pkg_resolve(both / n) if n else None, 'more_gold_titles_n': gained, 'same_gold_titles_n': equal, 'fewer_gold_titles_n': lost, 'selection_changed_n': sum((r['changed_' + p] for r in subset)), 'actual_top1_break_n': sum((r['effective_top1_break_' + p] for r in subset))}
        result[name] = {'n': n, 'denominator': 'outcome-selected diagnostic subset', 'inference': 'descriptive counts only; no valid unbiased policy superiority inference from selected subset', 'policies': policies}
    return result

def top1_intervention_flags(original_keys, alternative_keys, first_keys):
    """Disentangle outside-source selection from a genuine P=2 monopoly break.

    A routine second document from a later question is not an intervention if
    the first question had <=1 document or if the selected document *set* did
    not change. The function has NO gold, answer, or evidence inputs.
    """
    first = set(first_keys)
    (baseline, chosen) = (set(original_keys), set(alternative_keys))
    if len(original_keys) != len(baseline) or len(alternative_keys) != len(chosen):
        raise ValueError('duplicate document key in selection')
    changed = int(chosen != baseline)
    outside = int(any((k not in first for k in alternative_keys)))
    dominant = int(len(first) >= 2 and len(original_keys) == 2 and (baseline <= first))
    effective = int(bool(dominant and changed and outside))
    return {'changed': changed, 'outside_top1_source': outside, 'baseline_top1_dominant': dominant, 'effective_top1_break': effective, 'outside_without_effective_break': int(bool(outside and (not effective)))}

def verify_frozen_dev_source(dev_path, prepare_manifest, expected_sha=None, allow_manifest_only=False):
    """Check DEV bytes against the manifest and (preferably) an EXTERNAL pin.

    A manifest supplied beside its DEV file is not an independent trust anchor:
    both can be rewritten together. Without an external trusted SHA256 this
    function fails closed unless manifest-only exploratory mode is explicitly
    opted into. Never claim an independently verified freeze in that mode.
    """
    manifest_sha = prepare_manifest.get('dev_sha256')
    if not isinstance(manifest_sha, str) or not re.fullmatch('[a-fA-F0-9]{64}', manifest_sha):
        raise SystemExit('MISSING/INVALID DEV SHA256 IN PREPARE MANIFEST')
    if expected_sha is not None:
        if not isinstance(expected_sha, str) or not re.fullmatch('[a-fA-F0-9]{64}', expected_sha):
            raise SystemExit('INVALID INDEPENDENT FROZEN DEV SHA256')
        if manifest_sha.lower() != expected_sha.lower():
            raise SystemExit('MANIFEST DEV SHA DOES NOT MATCH INDEPENDENT FROZEN SHA')
    elif not allow_manifest_only:
        raise SystemExit('INDEPENDENT DEV SHA256 REQUIRED: pass --frozen-dev-sha256 from a trusted original freeze; or explicitly choose --allow-manifest-only-dev (weaker exploratory verification)')
    actual = sha(dev_path)
    if actual.lower() != manifest_sha.lower():
        raise SystemExit('DEV SOURCE SHA MISMATCH: hotpot_dev_distractor.jsonl')
    if expected_sha is not None and actual.lower() != expected_sha.lower():
        raise SystemExit('DEV SOURCE SHA MISMATCH AGAINST INDEPENDENT FROZEN PIN')
    return actual

def title_change_audit(original_keys, alternative_keys, docmap):
    """Gold-blind title-level audit IN ADDITION TO unique document-key changes.

    Different corpus document IDs may have identical titles. A changed doc
    set or Top-1-source escape is not proof that a new gold title was recovered.
    Title-level novelty is also not proof of semantic novelty or correctness.
    """
    for (name, keys) in (('original', original_keys), ('alternative', alternative_keys)):
        if len(keys) != len(set(keys)):
            raise ValueError(f'{name} contains duplicate document keys')
        if any((k not in docmap for k in keys)):
            raise ValueError(f'{name} contains unknown document keys')
    before = {docmap[k]['title'] for k in original_keys}
    after = {docmap[k]['title'] for k in alternative_keys}
    return {'title_set_changed': int(before != after), 'new_title_count': len(after - before), 'removed_title_count': len(before - after), 'unique_title_gain': len(after) - len(before), 'original_unique_title_n': len(before), 'alternative_unique_title_n': len(after)}

def sentence_coverage_strict(selected, supports):
    """Sensitivity: normalized whole-token-boundary matching, not semantic entailment."""
    normalized = [(normalize(d['title']), ' ' + normalize(d['text']) + ' ') for d in selected]
    if not supports:
        raise ValueError('missing supports')
    found = sum((any((t == dt and ' ' + sent + ' ' in body for (dt, body) in normalized)) for (t, sent) in supports))
    return (found, len(supports))

def doc_signature(key, d):
    content = json.dumps({'key': key, 'title': d['title'], 'text': d['text']}, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return {'key': key, 'title': d['title'], 'sha256': hashlib.sha256(content.encode('utf-8')).hexdigest()}

def load_legacy_v5(root):
    path = _pkg_resolve(root / 'p18d_goldblind_ablation_v5.py')
    if not path.is_file():
        return (None, None)
    spec = importlib.util.spec_from_file_location('p18d_v5_readonly_compare', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (module, sha(path))

def make_summary(rows, nboot, seed, bonferroni=True):
    """Paired gold-title-pair cluster bootstrap, reusing cluster draws across
    all policies/metrics to preserve the covariance and paired comparisons.
    """
    import numpy as np
    if not rows:
        return None
    n = len(rows)
    summary = {'n': n, 'question_type_counts': dict(collections.Counter((r['question_type'] for r in rows))), 'policies': {}, 'paired_cluster_delta_vs_G0': {}, 'score_drop': None, 'mean_top_question_degree': _pkg_resolve(sum((r['top_question_degree'] for r in rows)) / n), 'top1_dominance_n': sum((r['top1_dominates_P2'] for r in rows)), 'top1_structural_multi_edge_n': sum((r['top1_structural_multi_edge'] for r in rows)), 'top1_empty_n': sum((r['top1_empty'] for r in rows)), 'top1_single_edge_n': sum((r['top1_single_edge'] for r in rows)), 'top1_recoverable_miss_n': sum((r['top1_recoverable_miss'] for r in rows)), 'top1_recoverable_miss_rate': _pkg_resolve(sum((r['top1_recoverable_miss'] for r in rows)) / n), 'top1_internal_selection_miss_n': sum((r['top1_internal_selection_miss'] for r in rows)), 'top1_external_evidence_needed_n': sum((r['top1_external_evidence_needed'] for r in rows)), 'top1_gold_available_in_first_counts': dict(collections.Counter((r['top1_gold_titles_available_n'] for r in rows))), 'top1_recoverable_miss_given_multi_edge': _pkg_resolve(sum((r['top1_recoverable_miss'] for r in rows)) / sum((r['top1_structural_multi_edge'] for r in rows))) if any((r['top1_structural_multi_edge'] for r in rows)) else None, 'top1_internal_miss_given_multi_edge': _pkg_resolve(sum((r['top1_internal_selection_miss'] for r in rows)) / sum((r['top1_structural_multi_edge'] for r in rows))) if any((r['top1_structural_multi_edge'] for r in rows)) else None, 'top1_external_needed_given_multi_edge': _pkg_resolve(sum((r['top1_external_evidence_needed'] for r in rows)) / sum((r['top1_structural_multi_edge'] for r in rows))) if any((r['top1_structural_multi_edge'] for r in rows)) else None, 'top1_recoverable_miss_partition_check': 'PASS', 'G3_selection_status_counts': dict(collections.Counter((r['G3_status'] for r in rows))), 'G3_bound_relaxed_n': sum((r['G3_bound_relaxed'] for r in rows)), 'G3_eligible_outside_n': sum((bool(r['G3_eligible_outside_n']) for r in rows)), 'G3_questions_with_eligible_outside_n': sum((bool(r['G3_eligible_outside_n']) for r in rows)), 'G3_total_eligible_outside_docs': sum((r['G3_eligible_outside_n'] for r in rows)), 'G3_top1_escape_n': sum((r['G3_top1_escape'] for r in rows)), 'G4_top1_escape_n': sum((r['G4_top1_escape'] for r in rows)), 'G3_effective_top1_break_n': sum((r['effective_top1_break_G3'] for r in rows)), 'G4_effective_top1_break_n': sum((r['effective_top1_break_G4'] for r in rows)), 'G3_ordinary_outside_without_break_n': sum((r['outside_without_effective_break_G3'] for r in rows)), 'G4_ordinary_outside_without_break_n': sum((r['outside_without_effective_break_G4'] for r in rows)), 'G3_effective_break_given_dominant': _pkg_resolve(sum((r['effective_top1_break_G3'] for r in rows)) / sum((r['top1_dominates_P2'] for r in rows))) if any((r['top1_dominates_P2'] for r in rows)) else None, 'G4_effective_break_given_dominant': _pkg_resolve(sum((r['effective_top1_break_G4'] for r in rows)) / sum((r['top1_dominates_P2'] for r in rows))) if any((r['top1_dominates_P2'] for r in rows)) else None, 'baseline_two_doc_single_title_n': sum((r['G0_n'] == 2 and r['G0_distinct_titles_n'] == 1 for r in rows)), 'near_tie_score_inversion_questions_n': sum((r['tiny_score_inversions'] > 0 for r in rows)), 'candidate_oracle_both_gold_n': sum((r['oracle_both'] for r in rows)), 'candidate_oracle_mean_recall': _pkg_resolve(sum((r['oracle_recall'] for r in rows)) / n)}
    for name in POLICIES:
        summary['policies'][name] = {'recall_at_2': _pkg_resolve(sum((r[name + '_recall'] for r in rows)) / n), 'both_gold_hit_at_2': _pkg_resolve(sum((r[name + '_full'] for r in rows)) / n), 'sentence_full_legacy': _pkg_resolve(sum((r[name + '_sentence_full'] for r in rows)) / n), 'sentence_full_token_boundary': _pkg_resolve(sum((r[name + '_sentence_strict_full'] for r in rows)) / n), 'mean_documents': _pkg_resolve(sum((r[name + '_n'] for r in rows)) / n)}
    for name in ('G1', 'G2', 'G3', 'G4'):
        summary['policies'][name].update({'changed_doc_sets_n': sum((r['changed_' + name] for r in rows)), 'changed_doc_sets_rate': _pkg_resolve(sum((r['changed_' + name] for r in rows)) / n), 'changed_title_sets_n': sum((r['title_set_changed_' + name] for r in rows)), 'changed_title_sets_rate': _pkg_resolve(sum((r['title_set_changed_' + name] for r in rows)) / n), 'new_title_introduced_questions_n': sum((r['new_title_count_' + name] > 0 for r in rows)), 'new_titles_total_n': sum((r['new_title_count_' + name] for r in rows)), 'mean_unique_title_gain': _pkg_resolve(sum((r['unique_title_gain_' + name] for r in rows)) / n), 'effective_top1_break_with_new_title_n': sum((r['effective_top1_break_with_new_title_' + name] for r in rows)), 'changed_both_docs_n': sum((r['replaced_' + name] == 2 for r in rows)), 'replaced_doc_fraction': _pkg_resolve(sum((r['replaced_' + name] for r in rows)) / (2 * n)), 'recovered_both_gold_n': sum((r['G0_full'] == 0 and r[name + '_full'] == 1 for r in rows)), 'lost_both_gold_n': sum((r['G0_full'] == 1 and r[name + '_full'] == 0 for r in rows)), 'recovered_recall_n': sum((r[name + '_recall'] > r['G0_recall'] for r in rows)), 'lost_recall_n': sum((r[name + '_recall'] < r['G0_recall'] for r in rows)), 'mean_min_score_drop_vs_G0': _pkg_resolve(sum((r['score_loss_' + name] for r in rows)) / n), 'outside_first_item_n': sum((r['outside_first_' + name] for r in rows)), 'top1_dominance_break_n': sum((r['effective_top1_break_' + name] for r in rows)), 'outside_top1_without_effective_break_n': sum((r['outside_without_effective_break_' + name] for r in rows)), 'recoverable_d0_miss_recovered_n': sum((r['top1_recoverable_miss'] and r[name + '_full'] for r in rows)), 'internal_selection_miss_recovered_n': sum((r['top1_internal_selection_miss'] and r[name + '_full'] for r in rows)), 'external_evidence_needed_recovered_n': sum((r['top1_external_evidence_needed'] and r[name + '_full'] for r in rows)), 'internal_selection_miss_recovery_rate_diagnostic': _pkg_resolve(sum((r['top1_internal_selection_miss'] and r[name + '_full'] for r in rows)) / sum((r['top1_internal_selection_miss'] for r in rows))) if any((r['top1_internal_selection_miss'] for r in rows)) else None, 'external_evidence_needed_recovery_rate_diagnostic': _pkg_resolve(sum((r['top1_external_evidence_needed'] and r[name + '_full'] for r in rows)) / sum((r['top1_external_evidence_needed'] for r in rows))) if any((r['top1_external_evidence_needed'] for r in rows)) else None, 'recoverable_d0_miss_lost_n': sum((r['top1_recoverable_miss'] and r[name + '_recall'] < r['G0_recall'] for r in rows))})
    pairs = [(p, m) for p in ALTERNATIVES for m in METRICS]
    matrix = np.asarray([[r[p + '_' + m] - r['G0_' + m] for (p, m) in pairs] for r in rows], dtype=np.float64)
    blocks = collections.defaultdict(list)
    for (i, r) in enumerate(rows):
        blocks[r['cluster']].append(i)
    size = np.asarray([len(idx) for idx in blocks.values()], dtype=np.int64)
    block_sum = np.asarray([matrix[idx, :].sum(axis=0) for idx in blocks.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty((nboot, len(pairs)), dtype=np.float64)
    for start in range(0, nboot, 64):
        end = min(nboot, start + 64)
        idx = rng.integers(0, len(blocks), size=(end - start, len(blocks)))
        draws[start:end, :] = _pkg_resolve(block_sum[idx, :].sum(axis=1) / size[idx].sum(axis=1)[:, None])
    q95 = np.quantile(draws, [0.025, 0.975], axis=0)
    q975 = np.quantile(draws, [0.0125, 0.9875], axis=0)
    delta = matrix.mean(axis=0)
    for (i, (p, m)) in enumerate(pairs):
        entry = {'delta': float(delta[i]), 'ci95': [float(q95[0, i]), float(q95[1, i])], 'n': n, 'clusters': len(blocks), 'reps': nboot, 'method': 'paired gold-title-pair cluster bootstrap, percentile'}
        if bonferroni and p in ('G2', 'G3') and (m == 'recall'):
            entry['familywise_ci95_two_primary_recall_contrasts'] = [float(q975[0, i]), float(q975[1, i])]
            entry['multiplicity_note'] = 'Bonferroni coverage: 97.5% per contrast for G2 and G3 recall only'
        summary['paired_cluster_delta_vs_G0'].setdefault(p, {})[m] = entry
    if summary['top1_recoverable_miss_n'] != summary['top1_internal_selection_miss_n'] + summary['top1_external_evidence_needed_n']:
        raise AssertionError('top1 bottleneck subtypes do not partition recoverable misses')
    return summary

def paired_cluster_stratum_interaction(rows, label_fn, level_a, level_b, policy, metric, nboot, seed):
    """Effect modification: (policy-G0)_B minus (policy-G0)_A.

    Resample WHOLE gold-title-pair clusters jointly over both strata. This
    handles a cluster present in both groups and preserves QID pairing.
    Exploratory, multiplicity UNADJUSTED. Not a causal interaction estimate.
    """
    import numpy as np
    observed = collections.Counter((label_fn(r) for r in rows))
    if not observed[level_a] or not observed[level_b]:
        return None
    groups = collections.defaultdict(list)
    for (i, r) in enumerate(rows):
        groups[r['cluster']].append(i)
    cats = (level_a, level_b)
    sizes = np.zeros((len(groups), 2), dtype=np.int64)
    totals = np.zeros((len(groups), 2), dtype=np.float64)
    for (j, idxs) in enumerate(groups.values()):
        for i in idxs:
            cat = label_fn(rows[i])
            if cat in cats:
                k = cats.index(cat)
                sizes[j, k] += 1
                totals[j, k] += rows[i][policy + '_' + metric] - rows[i]['G0_' + metric]
    delta = _pkg_resolve(totals.sum(axis=0) / sizes.sum(axis=0))
    point = float(delta[1] - delta[0])
    rng = np.random.default_rng(seed)
    draws = []
    invalid = 0
    for start in range(0, nboot, 64):
        n = min(64, nboot - start)
        picks = rng.integers(0, len(groups), size=(n, len(groups)))
        ns = sizes[picks].sum(axis=1)
        sums = totals[picks].sum(axis=1)
        valid = (ns[:, 0] > 0) & (ns[:, 1] > 0)
        invalid += int((~valid).sum())
        if valid.any():
            draws.extend((_pkg_resolve(sums[valid, 1] / ns[valid, 1]) - _pkg_resolve(sums[valid, 0] / ns[valid, 0])).tolist())
    if len(draws) < max(100, int(nboot * 0.9)):
        return {'point': point, 'ci95': None, 'valid_draws': len(draws), 'invalid_draws': invalid, 'note': 'insufficient valid bootstrap draws'}
    (low, high) = np.quantile(draws, [0.025, 0.975])
    return {'delta_difference_b_minus_a': point, 'ci95': [float(low), float(high)], 'level_a': level_a, 'level_b': level_b, 'n_a': observed[level_a], 'n_b': observed[level_b], 'clusters': len(groups), 'valid_draws': len(draws), 'invalid_draws': invalid, 'method': 'joint paired gold-title-pair cluster bootstrap (descriptive unadjusted)'}

def self_test():
    import numpy as np
    items = [['a', 'b'], ['a', 'b'], ['c', 'd'], ['d', 'c']]
    ranked = [(0, 0.99), (1, 0.98), (2, 0.96), (3, 0.93)]
    d = choose_policies(ranked, items, score_drop=0.05)
    assert d['G0'] == d['G1'] == ['a', 'b']
    assert d['G2'] == ['a', 'b']
    assert d['G3'] == ['a', 'c']
    assert d['G4'] == ['a', 'c']
    low = choose_policies([(0, 0.99), (1, 0.98), (2, 0.01)], [['a', 'b'], ['a', 'b'], ['c']], score_drop=0.05)
    assert low['G3'] == ['a', 'b'], low
    assert low['G4'] == ['a', 'c'], low
    assert choose_policies([(0, 0.99)], [['a']], score_drop=0.05)['G0'] == ['a']
    import random
    random.seed(33)
    for _ in range(2000):
        ss = sorted([random.random() for _ in range(8)], reverse=True)
        keys = [[random.choice('abcde') for i in range(random.randint(1, 4))] for j in range(8)]
        pol = choose_policies(list(enumerate(ss)), keys)
        assert pol['G0'] == pol['G1']
        assert all((len(set(k)) == len(k) for k in pol.values()))
    for bad in [float('nan'), -1.0, float('inf')]:
        try:
            choose_policies(ranked, items, score_drop=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('score_drop validation missed ' + str(bad))
    assert sentence_coverage([{'title': 'A', 'text': 'cluster'}], [('a', 'lust')]) == (1, 1)
    assert sentence_coverage_strict([{'title': 'A', 'text': 'cluster'}], [('a', 'lust')]) == (0, 1)
    assert sentence_coverage_strict([{'title': 'A', 'text': 'The good sentence.'}], [('a', 'good sentence')]) == (1, 1)
    escape = choose_policies([(0, 0.99), (1, 0.98), (2, 0.97)], [['a', 'b'], ['a', 'c'], ['a', 'c']], score_drop=0.05, return_audit=True)
    assert escape[0]['G0'] == ['a', 'b']
    assert escape[0]['G3'] == ['a', 'c'] and escape[1]['G3_status'] == 'bounded_top1_escape'
    assert escape[1]['G3_top1_escape'] == 1 and escape[1]['G3_bound_relaxed'] == 0
    fail = choose_policies([(0, 0.99), (1, 0.01)], [['a'], ['b']], score_drop=0.05, return_audit=True)
    assert fail[0]['G3'] == ['a', 'b']
    assert fail[1]['G3_status'] == 'bound_infeasible_fallback'
    assert fail[1]['G3_bound_relaxed'] == 1 and fail[1]['G3_bound_satisfied'] == 0
    exact = choose_policies([(0, 0.99), (1, 0.98), (2, 0.01)], [['a', 'b'], ['a', 'b'], ['c']], return_audit=True)
    assert exact[0]['G3'] == ['a', 'b'] and exact[0]['G4'] == ['a', 'c']
    near = [(0, 0.9), (1, 0.900000005)]
    assert checked_ranking([{'item': j, 'score': v} for (j, v) in near], 'item', 8, 'near') == near
    (near_result, near_audit) = choose_policies(near, [['a'], ['b']], return_audit=True)
    assert near_result['G0'] == near_result['G1'] == ['a', 'b']
    assert near_audit['tiny_score_inversions'] == 1
    import importlib.util
    v5path = Path(_pkg_resolve(__file__)).with_name('p18d_goldblind_ablation_v5.py')
    if v5path.is_file():
        (old, _) = load_legacy_v5(v5path.parent)
        mock = {'a': {'title': 'A', 'text': 'x'}, 'b': {'title': 'B', 'text': 'y'}, 'c': {'title': 'C', 'text': 'z'}, 'd': {'title': 'D', 'text': 'w'}}
        assert choose_policies(ranked, items)['G2'] == old.choose_policies('hello', ranked, items, mock)['D1']
    sample = []
    for (i, (g2, g3, base)) in enumerate([(1, 1, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1)]):
        r = {'qid': str(i), 'cluster': 'a' if i < 2 else 'b', 'question_type': 'bridge', 'top_question_degree': 2, 'top1_dominates_P2': 1, 'top1_structural_multi_edge': 1, 'top1_empty': 0, 'top1_single_edge': 0, 'top1_recoverable_miss': int(not base), 'top1_internal_selection_miss': int(not base), 'top1_external_evidence_needed': 0, 'top1_gold_titles_available_n': 2, 'G0_distinct_titles_n': 2, 'top1_gold_titles_missing_n': 0, 'oracle_both': 1, 'oracle_recall': 1, 'G3_status': 'bounded_top1_escape', 'G3_bound_relaxed': 0, 'G3_eligible_outside_n': 1, 'G3_top1_escape': 1, 'G4_top1_escape': 1, 'tiny_score_inversions': 0}
        for p in POLICIES:
            val = {'G0': base, 'G1': base, 'G2': g2, 'G3': g3, 'G4': g3, 'V0': base}[p]
            for met in METRICS:
                r[p + '_' + met] = val
            r[p + '_n'] = 2
            if p not in ('G0', 'V0'):
                r['changed_' + p] = int(p != 'G1')
                r['replaced_' + p] = int(p != 'G1')
                r['score_loss_' + p] = 0.0
                r['outside_first_' + p] = int(p != 'G1')
                r['title_set_changed_' + p] = 0
                r['new_title_count_' + p] = 0
                r['unique_title_gain_' + p] = 0
                r['effective_top1_break_with_new_title_' + p] = 0
                r['effective_top1_break_' + p] = int(p != 'G1')
                r['outside_without_effective_break_' + p] = 0
        sample.append(r)
    joint = [dict(sample[i % 4], cluster='unit' + str(i), question_type='bridge' if i < 20 else 'comparison') for i in range(40)]
    it = paired_cluster_stratum_interaction(joint, lambda r: r['question_type'], 'bridge', 'comparison', 'G2', 'recall', 101, 0)
    assert it is not None and it['n_a'] == 20 and (it['n_b'] == 20)
    stat = make_summary(sample, 101, 0)
    assert abs(stat['paired_cluster_delta_vs_G0']['G2']['recall']['delta'] - 0.25) < 1e-12
    assert 'familywise_ci95_two_primary_recall_contrasts' in stat['paired_cluster_delta_vs_G0']['G3']['recall']
    assert len(stat['paired_cluster_delta_vs_G0']['G3']['recall']['ci95']) == 2
    assert top1_intervention_flags(['a', 'b'], ['a', 'c'], ['a', 'b']) == {'changed': 1, 'outside_top1_source': 1, 'baseline_top1_dominant': 1, 'effective_top1_break': 1, 'outside_without_effective_break': 0}
    for firstkeys in ([], ['a']):
        flags = top1_intervention_flags(['a', 'b'], ['a', 'b'], firstkeys)
        assert flags['effective_top1_break'] == 0 and flags['changed'] == 0
        assert flags['outside_top1_source'] == 1
        assert flags['outside_without_effective_break'] == 1
    assert top1_intervention_flags(['a', 'b'], ['a', 'c'], ['a'])['effective_top1_break'] == 0
    docs = {k: {'title': k, 'text': 'test text'} for k in 'ABCDX'}
    internal = classify_top1_gold_bottleneck(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D', 'X'], ['A', 'B'], docs, {'C', 'D'})
    assert internal['top1_recoverable_miss'] == 1
    assert internal['top1_internal_selection_miss'] == 1
    assert internal['top1_external_evidence_needed'] == 0
    external = classify_top1_gold_bottleneck(['A', 'B'], ['A', 'B', 'C', 'D'], ['A', 'B'], docs, {'B', 'C'})
    assert external['top1_recoverable_miss'] == 1
    assert external['top1_external_evidence_needed'] == 1
    assert external['top1_internal_selection_miss'] == 0
    unavailable = classify_top1_gold_bottleneck(['A', 'B'], ['A', 'B', 'C'], ['A', 'B'], docs, {'C', 'D'})
    assert not unavailable['top1_recoverable_miss']
    single = classify_top1_gold_bottleneck(['A'], ['A', 'B', 'C'], ['A', 'B'], docs, {'B', 'C'})
    assert single['top1_recoverable_miss'] == 0
    diagnostics = diagnostic_failure_subsets(sample)
    assert diagnostics['top1_internal_selection_miss']['n'] == 2
    assert 'ci95' not in str(diagnostics).lower()
    assert diagnostics['top1_external_evidence_needed']['n'] == 0
    assert stat['top1_internal_selection_miss_n'] == 2
    assert stat['top1_external_evidence_needed_n'] == 0
    assert stat['top1_recoverable_miss_partition_check'] == 'PASS'
    with tempfile.TemporaryDirectory() as tmp:
        path = _pkg_resolve(Path(_pkg_resolve(tmp)) / 'hotpot_dev_distractor.jsonl')
        _pkg_output(path).write_text('frozen\n', encoding='utf-8')
        manifest = {'dev_sha256': sha(path)}
        assert verify_frozen_dev_source(path, manifest, expected_sha=manifest['dev_sha256']) == manifest['dev_sha256']
        for mutated in ({}, {'dev_sha256': 'bad'}):
            try:
                verify_frozen_dev_source(path, mutated, expected_sha=manifest['dev_sha256'])
            except SystemExit:
                pass
            else:
                raise AssertionError('invalid DEV manifest accepted')
        _pkg_output(path).write_text('altered\n', encoding='utf-8')
        try:
            verify_frozen_dev_source(path, manifest, expected_sha=manifest['dev_sha256'])
        except SystemExit:
            pass
        else:
            raise AssertionError('changed DEV bytes accepted')
    with tempfile.TemporaryDirectory() as tmp:
        p = _pkg_resolve(Path(_pkg_resolve(tmp)) / 'dev.jsonl')
        _pkg_output(p).write_text('original\n', encoding='utf-8')
        trusted = sha(p)
        m = {'dev_sha256': trusted}
        assert verify_frozen_dev_source(p, m, expected_sha=trusted) == trusted
        try:
            verify_frozen_dev_source(p, m)
        except SystemExit:
            pass
        else:
            raise AssertionError('non-independent DEV check silently accepted')
        _pkg_output(p).write_text('tampered\n', encoding='utf-8')
        m['dev_sha256'] = sha(p)
        try:
            verify_frozen_dev_source(p, m, expected_sha=trusted)
        except SystemExit:
            pass
        else:
            raise AssertionError('coordinated DEV/manifest tampering not detected')
        assert verify_frozen_dev_source(p, m, allow_manifest_only=True) == sha(p)
    cl1 = json.dumps(sorted(['A | B', 'C']), ensure_ascii=False, separators=(',', ':'))
    cl2 = json.dumps(sorted(['A', 'B | C']), ensure_ascii=False, separators=(',', ':'))
    assert cl1 != cl2
    mockdocs = {'a': {'title': 'A'}, 'b': {'title': 'B'}, 'c': {'title': 'B'}}
    ta = title_change_audit(['a', 'b'], ['a', 'c'], mockdocs)
    assert ta['title_set_changed'] == 0 and ta['new_title_count'] == 0
    assert ta['unique_title_gain'] == 0
    mockdocs['c'] = {'title': 'C'}
    ta = title_change_audit(['a', 'b'], ['a', 'c'], mockdocs)
    assert ta['title_set_changed'] == 1 and ta['new_title_count'] == 1
    test_rank_validation()
    test_source_support_scope()
    print('SELF-TEST PASS: DEV independent pin, title-key audit, cluster collision,')
    print('                Top1 escape/fallback, near ties, 2000 random controls,')
    print('                legacy token-boundary sensitivity and cluster bootstrap')

def normalize(s):
    return ' '.join(TOKEN.findall(str(s).casefold()))

def load_supporting_facts(dev_path, target_qids=None):
    """Validate source support for evaluation QIDs only.

    Unrelated DEV examples are outside the frozen 2000-question experiment and
    cannot abort it. Selected examples are validated strictly; no correction,
    clipping or silent omission of malformed sentence indices is permitted.
    """
    raw = dev_path.read_text(encoding='utf-8-sig').lstrip()
    records = json.loads(raw) if raw.startswith('[') else [json.loads(x) for x in raw.splitlines() if x.strip()]
    targets = None if target_qids is None else set(map(str, target_qids))
    out = {}
    skipped = 0
    for r in records:
        qid = str(r.get('_id', r.get('id')))
        if targets is not None and qid not in targets:
            skipped += 1
            continue
        if qid in out:
            raise ValueError(f'duplicate DEV qid: {qid}')
        ctx = r.get('context', [])
        if isinstance(ctx, dict):
            (titles, sent_lists) = (ctx.get('title', []), ctx.get('sentences', []))
            if len(titles) != len(sent_lists):
                raise ValueError(f'Bad DEV context columns: {qid}')
            ctx_pairs = zip(titles, sent_lists)
        else:
            ctx_pairs = ctx
        by_title = {}
        for entry in ctx_pairs:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(f'Bad DEV context entry: {qid}')
            (title, sentences) = entry
            if not isinstance(sentences, list):
                raise ValueError(f'Bad DEV sentences: {qid}, {title}')
            normalized_title = normalize(title)
            by_title[normalized_title] = sentences
        sf = r.get('supporting_facts', [])
        if isinstance(sf, dict):
            (titles, ids) = (sf.get('title', []), sf.get('sent_id', []))
            if len(titles) != len(ids):
                raise ValueError(f'Bad supporting fact columns: {qid}')
            pairs = list(zip(titles, ids))
        else:
            pairs = sf
        support = []
        for (title, sid) in pairs:
            sentences = by_title.get(normalize(title), [])
            if isinstance(sid, bool) or not isinstance(sid, int) or sid < 0 or (sid >= len(sentences)):
                raise ValueError(f'Bad supporting fact in EVALUATION QID: {qid}, {title}, {sid}, available_sentences={len(sentences)}')
            sentence = normalize(sentences[sid])
            if not sentence:
                raise ValueError(f'Empty supporting fact in EVALUATION QID: {qid}')
            support.append((normalize(title), sentence))
        if not support:
            raise ValueError(f'No supporting facts in EVALUATION QID: {qid}')
        out[qid] = support
    if targets is not None and set(out) != targets:
        missing = sorted(targets - set(out))
        raise ValueError(f'Missing DEV evaluation QIDs ({len(missing)}): {missing[:5]}')
    return out

def sentence_coverage(selected, supports):
    normalized = [(normalize(d['title']), normalize(d['text'])) for d in selected]
    if not supports:
        raise ValueError('missing supporting facts')
    found = sum((any((t == dt and sent in body for (dt, body) in normalized)) for (t, sent) in supports))
    return (found, len(supports))

def boot_pair(a, b, nboot=10000, seed=0):
    import numpy as np
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if not len(diffs) or nboot < 1:
        raise ValueError('empty paired dataset or invalid bootstrap count')
    rng = np.random.default_rng(seed)
    draws = np.empty(nboot)
    for start in range(0, nboot, 256):
        n = min(256, nboot - start)
        sampled = rng.integers(0, len(diffs), size=(n, len(diffs)))
        draws[start:start + n] = diffs[sampled].mean(axis=1)
    return {'delta': float(diffs.mean()), 'ci95': [float(x) for x in np.quantile(draws, [0.025, 0.975])], 'resampling': 'paired question bootstrap', 'n': len(diffs), 'reps': nboot}

def boot_pair_cluster(a, b, clusters, nboot=10000, seed=0):
    import numpy as np
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if len(diffs) != len(clusters) or not len(diffs) or nboot < 1:
        raise ValueError('invalid paired cluster bootstrap inputs')
    group = collections.defaultdict(list)
    for (i, key) in enumerate(clusters):
        group[key].append(i)
    blocks = list(group.values())
    meansums = np.asarray([diffs[inds].sum() for inds in blocks])
    sizes = np.asarray([len(inds) for inds in blocks])
    rng = np.random.default_rng(seed)
    draws = np.empty(nboot)
    for start in range(0, nboot, 256):
        n = min(256, nboot - start)
        picked = rng.integers(0, len(blocks), size=(n, len(blocks)))
        draws[start:start + n] = _pkg_resolve(meansums[picked].sum(axis=1) / sizes[picked].sum(axis=1))
    return {'delta': float(diffs.mean()), 'ci95': [float(v) for v in np.quantile(draws, [0.025, 0.975])], 'resampling': 'paired gold-title-pair cluster bootstrap', 'clusters': len(blocks), 'n': len(a), 'reps': nboot}

def checked_ranking(entries, id_field, maximum, qid, expected_len=None):
    if not isinstance(entries, list) or not 1 <= len(entries) <= maximum:
        raise SystemExit(f'INVALID {id_field} RANK LENGTH: {qid}')
    if expected_len is not None and len(entries) != expected_len:
        raise SystemExit(f'INVALID {id_field} TOP-K LENGTH: {qid}')
    (ids, scores) = ([], [])
    for row in entries:
        if not isinstance(row, dict) or id_field not in row or 'score' not in row:
            raise SystemExit(f'INVALID {id_field} RANK RECORD: {qid}')
        identifier = row[id_field]
        if isinstance(identifier, bool) or not isinstance(identifier, int if id_field == 'item' else str):
            raise SystemExit(f'INVALID {id_field} TYPE: {qid}')
        if id_field != 'item' and (not identifier):
            raise SystemExit(f'EMPTY {id_field}: {qid}')
        try:
            score = float(row['score'])
        except (TypeError, ValueError, OverflowError):
            raise SystemExit(f'INVALID {id_field} SCORE: {qid}')
        if not math.isfinite(score):
            raise SystemExit(f'NONFINITE {id_field} SCORE: {qid}')
        ids.append(identifier)
        scores.append(score)
    if id_field == 'item' and len(set(ids)) != len(ids):
        raise SystemExit(f'DUPLICATE {id_field} RANK: {qid}')
    if any((scores[i] < scores[i + 1] - 1e-08 for i in range(len(scores) - 1))):
        raise SystemExit(f'UNSORTED {id_field} SCORES: {qid}')
    return list(zip(ids, scores))

def run(args):
    root = Path(_pkg_resolve(args.root)).resolve()
    base = _pkg_resolve(root / 'out/p18_evidence_qa_v1')
    paths = {name: _pkg_resolve(base / name) for name in ('dataset.jsonl', 'retrieval.jsonl', 'prepare_manifest.json')}
    paths['index_items.jsonl'] = _pkg_resolve(root / 'data/_split/index_items.jsonl')
    paths['hotpot_dev_distractor.jsonl'] = _pkg_resolve(root / 'data/hotpotqa/hotpot_dev_distractor.jsonl')
    for (name, path) in paths.items():
        if not path.is_file():
            raise SystemExit(f'MISSING {name}: {path}')
    manifest = json.loads(paths['prepare_manifest.json'].read_text(encoding='utf-8'))
    if manifest.get('corpus_sha256') != EXPECTED_SHA:
        raise SystemExit('MANIFEST CORPUS SHA MISMATCH')
    checks = [('dataset.jsonl', EXPECTED_DATASET_SHA), ('retrieval.jsonl', EXPECTED_RETRIEVAL_SHA), ('index_items.jsonl', manifest.get('index_sha256'))]
    for (name, expected) in checks:
        if not expected or sha(paths[name]) != expected:
            raise SystemExit(f'FROZEN INPUT SHA MISMATCH: {name}')
    if manifest.get('dataset_sha256') != EXPECTED_DATASET_SHA:
        raise SystemExit('MANIFEST DATASET SHA MISMATCH')
    frozen_dev_sha = verify_frozen_dev_source(paths['hotpot_dev_distractor.jsonl'], manifest, expected_sha=args.frozen_dev_sha256, allow_manifest_only=args.allow_manifest_only_dev)
    sys.path.insert(0, str(root))
    from S00_qi_common import build_corpus, corpus_fingerprint, para_key, split_paragraphs
    from S00_retrieval_policy import qi_policy, vanilla_policy
    index = list(readjl(paths['index_items.jsonl']))
    all_docs = build_corpus(index)
    if corpus_fingerprint(all_docs) != EXPECTED_SHA:
        raise SystemExit('RECOMPUTED CORPUS SHA MISMATCH')
    docmap = {para_key(d): d for d in all_docs}
    if len(docmap) != len(all_docs):
        raise SystemExit('CORPUS KEY COLLISION')
    item_doc_keys = [[para_key(d) for d in split_paragraphs(x.get('context', ''))] for x in index]
    if any((k not in docmap for keys in item_doc_keys for k in keys)):
        raise SystemExit('ITEM EDGE REFERENCES DOC OUTSIDE FROZEN CORPUS')
    data = strict_index(readjl(paths['dataset.jsonl']), 'qid')
    retrieval = strict_index(readjl(paths['retrieval.jsonl']), 'qid')
    if len(data) != EXPECTED_N or set(data) != set(retrieval):
        raise SystemExit('FROZEN QID COUNT OR ALIGNMENT MISMATCH')
    supports = load_supporting_facts(paths['hotpot_dev_distractor.jsonl'], data.keys())
    bytitle = collections.defaultdict(list)
    for doc in all_docs:
        bytitle[doc['title']].append(doc)
    (v5, v5sha) = (None, None)
    if args.verify_v5:
        (v5, v5sha) = load_legacy_v5(root)
        if v5 is None:
            raise SystemExit('CANNOT VERIFY LEGACY v5: p18d_goldblind_ablation_v5.py missing')
    outrows = []
    for (qid, d) in data.items():
        r = retrieval[qid]
        ranked = checked_ranking(r.get('qi_top_items', []), 'item', 8, qid, expected_len=8)
        if any((j < 0 or j >= len(index) for (j, _) in ranked)):
            raise SystemExit(f'INVALID INDEX ITEM ID {qid}')
        for (raw, (j, _)) in zip(r['qi_top_items'], ranked):
            titles = [docmap[k]['title'] for k in item_doc_keys[j]]
            if raw.get('titles') != titles:
                raise SystemExit(f'FROZEN TOP-ITEM TITLES NOT EQUAL {qid}')
        (pmap, selection_audit) = choose_policies(ranked, item_doc_keys, p=2, score_drop=args.score_drop, return_audit=True)
        (frozen, short) = qi_policy([j for (j, _) in ranked], item_doc_keys, 2)
        if pmap['G0'] != frozen or [docmap[k]['title'] for k in frozen] != r.get('qi_titles') or short != r.get('qi_short'):
            raise SystemExit(f'FROZEN ORIGINAL QI P=2 POLICY MISMATCH: {qid}')
        if pmap['G0'] != pmap['G1']:
            raise SystemExit(f'INVALID NEGATIVE CONTROL G0!=G1: {qid}')
        if args.verify_v5:
            old = v5.choose_policies(d['question'], ranked, item_doc_keys, docmap, p=2)
            if old['D0'] != pmap['G0'] or old['D1'] != pmap['G2']:
                raise SystemExit(f'LEGACY v5 D0/D1 COMPARISON FAILED: {qid}')
        vranked = checked_ranking(r.get('va_top_docs', []), 'title', 10, qid, expected_len=10)
        (v_titles, v_short) = vanilla_policy([t for (t, _) in vranked], 2)
        if v_titles != r.get('va_titles') or v_short != r.get('va_short'):
            raise SystemExit(f'FROZEN VANILLA P=2 POLICY MISMATCH: {qid}')
        vd = []
        for title in v_titles:
            candidates = bytitle[title]
            if len(candidates) != 1:
                raise SystemExit(f'AMBIGUOUS VANILLA DOC TITLE (fail-closed): qid={qid}, title={title!r}, n={len(candidates)}, candidate_keys={[para_key(x) for x in candidates[:10]]}')
            vd.append(candidates[0])
        gold_titles = [str(t) for t in d.get('gold_titles', []) if t]
        if len(gold_titles) != 2 or len(set(gold_titles)) != 2:
            raise SystemExit(f'UNEXPECTED GOLD FORMAT: {qid}')
        gold = set(gold_titles)
        qtype = str(d.get('type', '')).strip().lower()
        if qtype not in ('bridge', 'comparison'):
            raise SystemExit(f'INVALID HOTPOT QUESTION TYPE: {qid}: {qtype}')
        firstset = set(unique(item_doc_keys[ranked[0][0]]))
        pool = unique((key for (j, _) in ranked for key in item_doc_keys[j]))
        pool_titles = {docmap[k]['title'] for k in pool}
        oracle_hits = len(gold & pool_titles)
        row = {'qid': qid, 'question_type': qtype, 'evidence_group': d.get('evidence_group'), 'cluster': json.dumps(sorted(gold), ensure_ascii=False, separators=(',', ':')), 'top8_pool_size': len(pool), 'top_question_degree': len(firstset), 'top1_dominates_P2': int(len(firstset) >= 2 and len(set(pmap['G0']) & firstset) == 2), 'oracle_recall': _pkg_resolve(oracle_hits / 2), 'oracle_both': int(oracle_hits == 2), 'G0_distinct_titles_n': len({docmap[k]['title'] for k in pmap['G0']}), 'first_item_id': ranked[0][0], 'top_item_scores_json': json.dumps([float(s) for (_, s) in ranked]), 'first_item_keys_json': json.dumps(sorted(firstset), ensure_ascii=False), 'top1_structural_multi_edge': int(len(firstset) >= 2), 'top1_empty': int(len(firstset) == 0), 'top1_single_edge': int(len(firstset) == 1), **selection_audit}
        row.update(classify_top1_gold_bottleneck(firstset, pool, pmap['G0'], docmap, gold))
        if row['oracle_both'] == 0 and row['top1_recoverable_miss']:
            raise AssertionError(f'oracle-infeasible miss labeled recoverable: {qid}')
        selections = {name: [docmap[k] for k in keys] for (name, keys) in pmap.items()}
        selections['V0'] = vd
        row['V0_titles'] = json.dumps(v_titles, ensure_ascii=False)
        for name in POLICIES:
            selected = selections[name]
            keys = pmap[name] if name != 'V0' else [para_key(doc) for doc in vd]
            titles = [d0['title'] for d0 in selected]
            nhit = len(set(titles) & gold)
            row[name + '_n'] = len(keys)
            row[name + '_hit'] = nhit
            row[name + '_recall'] = _pkg_resolve(nhit / 2)
            row[name + '_full'] = int(nhit == 2)
            row[name + '_titles'] = json.dumps(titles, ensure_ascii=False)
            row[name + '_keys_json'] = json.dumps(keys, ensure_ascii=False)
            row[name + '_documents_json'] = json.dumps([doc_signature(k, docmap[k]) for k in keys], ensure_ascii=False)
            (hit, total) = sentence_coverage(selected, supports[qid])
            (strict_hit, strict_total) = sentence_coverage_strict(selected, supports[qid])
            if total != strict_total or strict_hit > hit:
                raise SystemExit(f'SENTENCE COVERAGE INTEGRITY ERROR {qid}')
            row[name + '_sentence_hit'] = hit
            row[name + '_sentence_total'] = total
            row[name + '_sentence_full'] = int(hit == total)
            row[name + '_sentence_recall'] = _pkg_resolve(hit / total)
            row[name + '_sentence_strict_hit'] = strict_hit
            row[name + '_sentence_strict_full'] = int(strict_hit == total)
        scores = {key: next((s for (j, s) in ranked if key in item_doc_keys[j])) for key in pool}
        g0_min = min([scores[k] for k in pmap['G0']]) if pmap['G0'] else float('nan')
        for name in ('G1', 'G2', 'G3', 'G4'):
            keyset = set(pmap[name])
            row['changed_' + name] = int(keyset != set(pmap['G0']))
            row['replaced_' + name] = len(set(pmap['G0']) - keyset)
            row['outside_first_' + name] = int(any((k not in firstset for k in pmap[name])))
            flags = top1_intervention_flags(pmap['G0'], pmap[name], firstset)
            row['effective_top1_break_' + name] = flags['effective_top1_break']
            row['outside_without_effective_break_' + name] = flags['outside_without_effective_break']
            ta = title_change_audit(pmap['G0'], pmap[name], docmap)
            for (field, value) in ta.items():
                row[field + '_' + name] = value
            row['effective_top1_break_with_new_title_' + name] = int(flags['effective_top1_break'] and ta['new_title_count'] > 0)
            if name == 'G1' and (ta['title_set_changed'] or ta['new_title_count']):
                raise AssertionError(f'G1 negative control changed title evidence: {qid}')
            if flags['changed'] != row['changed_' + name] or flags['outside_top1_source'] != row['outside_first_' + name]:
                raise AssertionError(f'TOP1 INTERVENTION AUDIT INCONSISTENCY {qid} {name}')
            score_min = min([scores[k] for k in pmap[name]]) if pmap[name] else float('nan')
            row['score_loss_' + name] = max(0.0, g0_min - score_min) if math.isfinite(score_min) and math.isfinite(g0_min) else 0.0
        for name in ('G3', 'G4'):
            if row[name + '_top1_escape'] != row['outside_first_' + name]:
                raise AssertionError(f'RAW TOP1 ESCAPE FLAG INCONSISTENCY {qid} {name}')
            if row['effective_top1_break_' + name] and (not row['top1_dominates_P2']):
                raise AssertionError(f'FALSE EFFECTIVE TOP1 BREAK {qid} {name}')
        outrows.append(row)
    if sum((r['G0_sentence_full'] for r in outrows)) != EXPECTED_D0_FULL:
        raise SystemExit('FROZEN G0 LEGACY SUPPORT BASELINE FAILED: expected 262')
    if sum((r['V0_sentence_full'] for r in outrows)) != EXPECTED_D4_FULL:
        raise SystemExit('FROZEN V0 LEGACY SUPPORT BASELINE FAILED: expected 481')
    if any((r['G1_full'] != r['G0_full'] or r['G1_recall'] != r['G0_recall'] for r in outrows)):
        raise SystemExit('NEGATIVE CONTROL METRIC DIFFERENCE')
    observed_types = collections.Counter((r['question_type'] for r in outrows))
    if not all((observed_types.get(k, 0) > 0 for k in ('bridge', 'comparison'))):
        raise SystemExit('MISSING BRIDGE OR COMPARISON STRATUM')
    group_evidence = collections.defaultdict(list)
    group_type = collections.defaultdict(list)
    group_dominance = collections.defaultdict(list)
    group_top1_degree = collections.defaultdict(list)
    for row in outrows:
        group_evidence[str(row['evidence_group'])].append(row)
        group_type[row['question_type']].append(row)
        group_dominance['top1_dominates_P2' if row['top1_dominates_P2'] else 'top1_not_dominant'].append(row)
        degree_category = 'empty' if row['top1_empty'] else 'single_edge' if row['top1_single_edge'] else 'multi_edge'
        group_top1_degree[degree_category].append(row)

    def groupsummary(groups):
        return {name: make_summary(rr, args.bootstrap, args.seed) for (name, rr) in sorted(groups.items())}
    report = {'title': 'POST-18D v11 Gold-Blind P=2 document selection bottleneck audit', 'experiment_stage': 'exploratory; NOT a preregistered new holdout', 'analysis_n': len(outrows), 'question_type_counts': dict(observed_types), 'policy_definitions': {'G0': 'Original QI rank-first unique document keys P2; v5 D0', 'G1': 'rank-stable max-source-score dedup negative control; G0-equivalent by construction including tolerated 1e-8 score inversions', 'G2': 'sum of reciprocal item-rank votes; same algorithm as v5 D1', 'G3': 'P2 Top1 escape: prefer highest-score outside-Top1 document if its linked-question score >= top1score-score_drop; else eligible highest; explicit forced-fill fallback', 'G4': 'P2 Top1 escape regardless of score; exploratory sensitivity only', 'V0': 'original frozen Vanilla top-10 titles -> P2; v5 D4'}, 'score_drop': args.score_drop, 'bootstrap_reps': args.bootstrap, 'seed': args.seed, 'primary_endpoint': 'mean per-question gold-title Recall@2 (2 titles per question)', 'secondary_endpoints': ['Both-Gold Hit@2', 'legacy exact-normalized support full', 'token-boundary support full'], 'multiple_testing': 'Bonferroni 2 comparisons, G2/G3 versus G0, Recall@2 endpoint only; exploratory due to reused evaluation data', 'gold_blind': 'no gold labels, answers or supporting facts passed to choose_policies', 'warnings': ['Evaluation-set policies are exploratory, NOT independent confirmatory evidence', 'Gold title Recall and support substring matching do NOT establish semantic answer sufficiency', 'Legacy support match uses unbounded substring and can yield false positives; token-boundary sensitivity included', f'G3 threshold {args.score_drop} is exploratory and not calibrated; do not tune using these evaluated QIDs', 'Score drop limits linked indexed-question embedding similarity, NOT direct document semantic relevance', 'G3_top1_escape is a raw outside-source flag; G3_effective_top1_break requires a changed document set AND baseline Top1 P2 monopoly', 'Top1 empty/single-edge questions do not qualify as monopolies even if a later-question document is selected', 'G3 is relevance-bounded only when at least one eligible second document exists; P2 forced-fill fallback is explicitly recorded', 'Top1 multi-edge is a structural exposure, not evidence of causal bottleneck by itself', 'Top1 recoverable misses are partitioned into first-item INTERNAL selection failure versus EXTERNAL required title', 'Outcome-selected Oracle-feasible G0 failures are descriptive only; never use conditional CIs to assert policy superiority', 'No new retrieval, learned document scores, answer generation or abstention evaluation', 'V0 is a comparator, not a gold-blind QI selection policy; no QI-index superiority can be concluded', 'Stratified and secondary endpoint CIs are descriptive; no familywise correction over all endpoints and strata', 'A changed document-key set does not necessarily change the set of document titles', 'Title novelty is not sufficient to conclude semantic evidence novelty or answer correctness', 'Manifest-only DEV SHA verification is NOT independent if the DEV and manifest are jointly altered'], 'source_sha256': {name: sha(path) for (name, path) in paths.items()}, 'corpus_sha256_verified': EXPECTED_SHA, 'dev_sha256_verified_against_manifest': frozen_dev_sha, 'dev_freeze_verification_level': 'independent_pin_and_manifest' if args.frozen_dev_sha256 is not None else 'manifest_only_NOT_independent', 'independent_dev_sha256_pin': args.frozen_dev_sha256, 'source_v5_code_sha256': v5sha, 'v5_policy_identity_checked': bool(args.verify_v5), 'script_sha256': sha(Path(_pkg_resolve(__file__)).resolve()), 'reference_sentence_full_counts': {'G0': EXPECTED_D0_FULL, 'V0': EXPECTED_D4_FULL}, 'baseline_audit': 'PASS: corpus, source data, every QID, per-QID QI/Vanilla selection and sentence counts', 'overall': make_summary(outrows, args.bootstrap, args.seed), 'by_question_type': groupsummary(group_type), 'by_evidence_group': groupsummary(group_evidence), 'by_top1_dominance': groupsummary(group_dominance), 'by_top1_degree': groupsummary(group_top1_degree), 'outcome_selected_failure_diagnostics': diagnostic_failure_subsets(outrows), 'outcome_selected_failure_diagnostics_note': 'Counts and conditional recovery fractions ONLY. No paired confidence intervals or net policy superiority claims in outcome-selected G0 failure subsets.', 'effect_modification': {'comparison_minus_bridge': {p: paired_cluster_stratum_interaction(outrows, lambda r: r['question_type'], 'bridge', 'comparison', p, 'recall', args.bootstrap, args.seed) for p in ('G2', 'G3')}, 'top1_multiedge_minus_not_multiedge': {p: paired_cluster_stratum_interaction(outrows, lambda r: 'multi_edge' if r['top1_structural_multi_edge'] else 'not_multi_edge', 'not_multi_edge', 'multi_edge', p, 'recall', args.bootstrap, args.seed) for p in ('G2', 'G3')}}}
    if sum((x['n'] for x in report['by_question_type'].values())) != EXPECTED_N:
        raise SystemExit('STRATUM COUNT DOES NOT SUM TO TOTAL')
    if sum((x['n'] for x in report['by_top1_dominance'].values())) != EXPECTED_N:
        raise SystemExit('DOMINANCE STRATUM COUNT DOES NOT SUM TO TOTAL')
    if sum((x['n'] for x in report['by_top1_degree'].values())) != EXPECTED_N:
        raise SystemExit('TOP1 DEGREE STRATUM COUNT DOES NOT SUM TO TOTAL')
    if any((r['top1_empty'] + r['top1_single_edge'] + r['top1_structural_multi_edge'] != 1 for r in outrows)):
        raise SystemExit('TOP1 DEGREE LABELS DO NOT FORM A PARTITION')
    if any((r['top1_recoverable_miss'] != r['top1_internal_selection_miss'] + r['top1_external_evidence_needed'] for r in outrows)):
        raise SystemExit('TOP1 INTERNAL/EXTERNAL BOTTLENECK LABELS DO NOT PARTITION')
    if any((r['top1_internal_selection_miss'] and r['top1_gold_titles_available_n'] != 2 for r in outrows)):
        raise SystemExit('INTERNAL TOP1 MISS WITHOUT BOTH TITLES INSIDE TOP1')
    if any((r['top1_external_evidence_needed'] and r['top1_gold_titles_available_n'] >= 2 for r in outrows)):
        raise SystemExit('EXTERNAL NEEDED LABEL WITH BOTH TITLES ALREADY IN TOP1')
    report['paired_question_bootstrap_sensitivity'] = {p: boot_pair([r[p + '_recall'] for r in outrows], [r['G0_recall'] for r in outrows], args.bootstrap, args.seed) for p in ('G2', 'G3')}
    report['paired_question_bootstrap_sensitivity_note'] = 'Secondary QID-paired bootstrap; possible QID dependence remains. Primary inferential summary uses frozen gold-title-pair cluster bootstrap.'
    target = Path(_pkg_resolve(args.output))
    if not target.is_absolute():
        target = _pkg_resolve(root / target)
    if target.exists():
        raise SystemExit('REFUSE TO OVERWRITE ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(_pkg_resolve(tempfile.mkdtemp(prefix='.p18d_v11_stage_', dir=str(target.parent))))
    try:
        with _pkg_pathlib.Path(_pkg_open_path(_pkg_resolve(stage / 'items.csv'), 'x')).open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(outrows[0]))
            writer.writeheader()
            writer.writerows(outrows)
        _pkg_output(_pkg_resolve(stage / 'report.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        lines = ['# POST-18D v11: Gold-Blind document selection', '', f'Frozen dataset: N={EXPECTED_N}, score_drop={args.score_drop}, seed={args.seed}, paired cluster bootstrap={args.bootstrap}', f'Strict D0 + Vanilla legacy evidence baseline PASS ({EXPECTED_D0_FULL} / {EXPECTED_D4_FULL}).', 'Gold-title Recall@2 is a title retrieval proxy; Both-Gold Hit@2 requires both titles.', 'G3 score-drop constrains linked QUESTION similarity, not direct document relevance.', 'Top1 outside-source flags are NOT effective monopoly breaks unless Top1 originally consumed both P2 slots.', 'Exploratory results only. No LLM outcomes or causal index superiority claim.', f"DEV freeze assurance: {report['dev_freeze_verification_level']}.", '']
        for (groupname, entry) in [('ALL', report['overall'])] + [('TYPE/' + k, v) for (k, v) in report['by_question_type'].items()] + [('TOP1/' + k, v) for (k, v) in report['by_top1_dominance'].items()] + [('TOP1_DEGREE/' + k, v) for (k, v) in report['by_top1_degree'].items()] + [('EVIDENCE/' + k, v) for (k, v) in report['by_evidence_group'].items()]:
            lines += [f"## {groupname} (n={entry['n']})", '', '|Policy|Recall@2|Both-Gold@2|Delta Recall vs G0 (95% paired-cluster CI)|Changed document sets|Recovered/Lost both-gold|', '|---|---:|---:|---|---:|---:|']
            for p in POLICIES:
                vals = entry['policies'][p]
                if p == 'G0':
                    delta = 'reference'
                    changed = '–'
                    recover = '–'
                else:
                    val = entry['paired_cluster_delta_vs_G0'][p]['recall']
                    delta = f"{val['delta']:+.4f} [{val['ci95'][0]:+.4f}, {val['ci95'][1]:+.4f}]"
                    if p == 'V0':
                        changed = '–'
                        recover = '–'
                    else:
                        changed = f"{vals['changed_doc_sets_n']}/{entry['n']}"
                        recover = f"{vals['recovered_both_gold_n']}/{vals['lost_both_gold_n']}"
                lines.append(f"|{p}|{vals['recall_at_2']:.4f}|{vals['both_gold_hit_at_2']:.4f}|{delta}|{changed}|{recover}|")
            lines += ['', f"Oracles (titles in Top8): both={entry['candidate_oracle_both_gold_n']}/{entry['n']}; top1 multi-edge={entry['top1_structural_multi_edge_n']}/{entry['n']}; potentially recoverable Top1 miss={entry['top1_recoverable_miss_n']}/{entry['n']}.", '']
            if groupname == 'ALL':
                lines.append(f"Top1 candidate-oracle recoverable baseline misses (TITLE proxy): {entry['top1_recoverable_miss_n']} = INTERNAL selection {entry['top1_internal_selection_miss_n']} + EXTERNAL evidence needed {entry['top1_external_evidence_needed_n']}.")
                lines.append(f"Among multi-edge Top1 (n={entry['top1_structural_multi_edge_n']}): potential miss fraction={entry['top1_recoverable_miss_given_multi_edge']}, internal fraction={entry['top1_internal_miss_given_multi_edge']}, external-needed fraction={entry['top1_external_needed_given_multi_edge']}.")
                for p in ('G2', 'G3', 'G4'):
                    vals = entry['policies'][p]
                    lines.append(f"{p} outcome-selected recovery counts (DESCRIPTIVE ONLY): internal {vals['internal_selection_miss_recovered_n']}/{entry['top1_internal_selection_miss_n']}; external-needed {vals['external_evidence_needed_recovered_n']}/{entry['top1_external_evidence_needed_n']}.")
                lines.append('No confidence intervals or net superiority claims for Gold/Oracle/G0-failure selected subsets.')
                lines.append(f"Top1 source degree: empty={entry['top1_empty_n']}, single={entry['top1_single_edge_n']}, multi={entry['top1_structural_multi_edge_n']}.")
                lines.append(f"G3 bound-relaxed P2 fallback: {entry['G3_bound_relaxed_n']}/{entry['n']}; questions with eligible outside-Top1 document: {entry['G3_questions_with_eligible_outside_n']}/{entry['n']}; total eligible outside-document candidates across questions: {entry['G3_total_eligible_outside_docs']}.")
                lines.append(f"G3 outside-Top1 selected (raw): {entry['G3_top1_escape_n']}/{entry['n']}; effective Top1 P2 monopoly breaks: {entry['G3_effective_top1_break_n']}/{entry['n']}; raw outside without effective break: {entry['G3_ordinary_outside_without_break_n']}/{entry['n']}.")
                lines.append(f"G4 outside-Top1 selected (raw): {entry['G4_top1_escape_n']}/{entry['n']}; effective Top1 P2 monopoly breaks: {entry['G4_effective_top1_break_n']}/{entry['n']}.")
                lines.append(f"G3 effective break conditional on baseline monopoly: {entry['G3_effective_break_given_dominant']}; G4: {entry['G4_effective_break_given_dominant']}.")
                for pol in ('G2', 'G3', 'G4'):
                    c = entry['policies'][pol]
                    lines.append(f"{pol} document-key set changed {c['changed_doc_sets_n']}/{entry['n']}; title set changed {c['changed_title_sets_n']}/{entry['n']}; new title added {c['new_title_introduced_questions_n']}/{entry['n']}; top1 key-break with new title {c['effective_top1_break_with_new_title_n']}/{entry['n']}.")
                lines.append(f"QID-paired bootstrap sensitivity for Recall@2 (G2/G3): {report['paired_question_bootstrap_sensitivity']}")
                lines.append(f"G3 statuses: {entry['G3_selection_status_counts']}")
                lines.append(f"Exploratory effect modification (unadjusted): {report['effect_modification']}")
                for p in ('G2', 'G3'):
                    c = entry['paired_cluster_delta_vs_G0'][p]['recall']
                    v = c['familywise_ci95_two_primary_recall_contrasts']
                    lines.append(f'{p} vs G0 Recall@2 Bonferroni two-contrast CI: [{v[0]:+.4f},{v[1]:+.4f}]')
                lines.append('')
        lines += ['', '## Outcome-selected failure diagnostics (descriptive ONLY)', '', 'Counts are conditioned on baseline failure and/or candidate Gold. These do NOT estimate an unbiased net policy effect.', '']
        for (groupname, entry) in report['outcome_selected_failure_diagnostics'].items():
            lines.append(f"### {groupname} (n={entry['n']})")
            lines.append('|Policy|Both-Gold recovered|Gold-title gains/same/loss|Selection changed|')
            lines.append('|---|---:|---|---:|')
            for (p, x) in entry['policies'].items():
                lines.append(f"|{p}|{x['both_gold_recovered_n']}/{entry['n']}|{x['more_gold_titles_n']}/{x['same_gold_titles_n']}/{x['fewer_gold_titles_n']}|{x['selection_changed_n']}|")
            lines.append('')
        _pkg_output(_pkg_resolve(stage / 'report.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        artifact_sha = {p.name: sha(p) for p in (_pkg_resolve(stage / 'items.csv'), _pkg_resolve(stage / 'report.json'), _pkg_resolve(stage / 'report.md'))}
        _pkg_output(_pkg_resolve(stage / 'audit_manifest.json')).write_text(json.dumps({'script_sha256': report['script_sha256'], 'source_sha256': report['source_sha256'], 'output_sha256': artifact_sha, 'n': EXPECTED_N, 'v5_policy_identity_checked': bool(args.verify_v5), 'dev_sha256_verified_against_manifest': frozen_dev_sha, 'dev_freeze_verification_level': report['dev_freeze_verification_level'], 'independent_dev_sha256_pin': args.frozen_dev_sha256, 'gold_selection_separation': 'choose_policies operates strictly on ranked IDs/scores and corpus adjacency'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        stage.rename(target)
    except Exception:
        print('FAILED WHILE WRITING; partial stage at', stage, file=sys.stderr)
        raise
    print('\n'.join(lines[:22]))
    print('Saved:', target)
    print('Files: items.csv report.json report.md audit_manifest.json')

def test_rank_validation():
    valid = [{'item': 0, 'score': 1.0}, {'item': 1, 'score': 0.5}]
    assert checked_ranking(valid, 'item', 8, 'mock') == [(0, 1.0), (1, 0.5)]
    for invalid in ([{'item': 0, 'score': 1.0}, {'item': 0, 'score': 0.5}], [{'item': 0, 'score': float('nan')}], [{'item': 0, 'score': 0.3}, {'item': 1, 'score': 0.7}], [{'item': True, 'score': 0.7}]):
        try:
            checked_ranking(invalid, 'item', 8, 'mock')
        except SystemExit:
            pass
        else:
            raise AssertionError('invalid ranking was accepted')

def test_source_support_scope():
    with tempfile.TemporaryDirectory() as tmp:
        path = _pkg_resolve(Path(_pkg_resolve(tmp)) / 'dev.jsonl')
        records = [{'_id': 'unused', 'context': [['X', ['x']]], 'supporting_facts': [['X', 902]]}, {'_id': 'eval', 'context': [['Y', ['good sentence']]], 'supporting_facts': [['Y', 0]]}]
        _pkg_output(path).write_text('\n'.join((json.dumps(r) for r in records)) + '\n', encoding='utf-8')
        assert load_supporting_facts(path, {'eval'})['eval'] == [('y', 'good sentence')]
        try:
            load_supporting_facts(path, {'unused'})
        except ValueError:
            pass
        else:
            raise AssertionError('invalid support did not raise')
        try:
            load_supporting_facts(path, {'missing'})
        except ValueError:
            pass
        else:
            raise AssertionError('missing evaluated QID did not raise')

def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--root', default='.', help='QI-RAG project root, default current directory')
    pa.add_argument('--output', default='out/p18d_goldblind_ablation_v11', help='non-overwriting result folder')
    pa.add_argument('--frozen-dev-sha256', default=None, help='SHA256 pinned from an independently trusted ORIGINAL DEV freeze; do not calculate from potentially changed current files')
    pa.add_argument('--allow-manifest-only-dev', action='store_true', help='explicit weaker exploratory verification, if independent DEV SHA256 is unavailable')
    pa.add_argument('--bootstrap', type=int, default=10000)
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--score-drop', type=float, default=0.05, help='G3 linked-question similarity score-drop bound (not document relevance; exploratory default .05)')
    pa.add_argument('--verify-v5', action='store_true', help='also compare every G0/G2 selection against v5 D0/D1')
    pa.add_argument('--self-test', action='store_true')
    args = pa.parse_args()
    if args.bootstrap < 1000:
        pa.error('--bootstrap must be >=1000 for reported confidence intervals')
    if args.frozen_dev_sha256 and args.allow_manifest_only_dev:
        pa.error('choose --frozen-dev-sha256 or --allow-manifest-only-dev, not both')
    if not args.self_test and (not args.frozen_dev_sha256) and (not args.allow_manifest_only_dev):
        pa.error('pass independently pinned --frozen-dev-sha256, or explicitly --allow-manifest-only-dev')
    if not math.isfinite(args.score_drop) or args.score_drop < 0:
        pa.error('--score-drop must be finite and nonnegative')
    if args.self_test:
        self_test()
    else:
        run(args)
if __name__ == '__main__':
    main()
