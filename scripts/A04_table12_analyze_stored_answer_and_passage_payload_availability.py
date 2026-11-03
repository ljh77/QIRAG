"""
p17  POST-17 Passage-Payload Information Reuse Audit v12 (탐색적 구조 분석, 신규 LLM 생성 없음)

입력 (probe 로 확인된 구조)
  data/_split/index_items.jsonl            : qid, answer(list), context(str). 행 번호 = 색인 item id
  out/holdout_locked/split_ids.json        : 'holdout' 키에 Holdout qid 목록 (4,405개)
  out/holdout_locked/retrieval_predictions.jsonl : qid, qi_top_items[{item, titles}], qi_titles
  data/hotpotqa/hotpot_dev_distractor.jsonl: id, answer(str), type, level  (Holdout 정답 출처)
  retrieval_policy.qi_policy, qi_common.split_paragraphs / para_key 는 import 해서 그대로 호출

두 가지 Holdout 조건 (둘 다 보고, 역할을 구분)
  A2 (정식 주 분석, 동결 재현 조건):
     Q-C = 동결 전체 순위 qi_top_items 로 qi_policy(P=2) 가 고른 문서 (동결 기록 qi_titles 와 제목 일치 검증)
     Q-A = 그 P=2 문서를 공급한 순위 항목들(문서마다 순위에서 처음 등장한 항목)의 정답 문자열
     두 조건은 같은 항목 집합에서 나온 정보이므로 정보 출처가 대응한다.
  A1 (보조, Top-1 조건):
     Q-A = 상위 1개 항목의 정답, Q-C = qi_policy([top1], P=2). 동결 재현 조건이 아니다.

LOO 보조 분석
  색인 항목마다 근거 문서 키를 공유하는 이웃 후보(정답 비어 있지 않음) 중 해시로 1개 선택 (정답 미사용).
  Q-A = 이웃 정답, Q-C = qi_policy([이웃], P=2). 공유 문서가 선정 문서에 포함되는 비율을 따로 기록.

검증 범위
  - 동결 검증은 문서 '제목' 수준이다. 제목이 같은 서로 다른 문서는 구분되지 않는다.
  - Gold 출처 검증은 dev 파일의 id 연결과 정답 형식까지다. 평가 분할 원본과의 동일성은 manifest 의 해시로만 기록한다.
  - 정답 포함은 문자열 포함 여부일 뿐 생성 답변의 정확도가 아니다.

명령
  probe   입력 파일 존재 확인
  run     분석 실행 (--limit-holdout N, --limit-loo N 은 양의 정수만 허용, 스모크는 smoke_ 폴더)
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
import numpy as np
HOME = pathlib.Path.home()
ROOT = _PACKAGE_ROOT
INDEX_FILE = _pkg_resolve(_pkg_resolve(_pkg_resolve(ROOT / 'data') / '_split') / 'index_items.jsonl')
SPLIT_IDS_FILE = _pkg_resolve(_pkg_resolve(_pkg_resolve(ROOT / 'out') / 'holdout_locked') / 'split_ids.json')
PRED_FILE = _pkg_resolve(_pkg_resolve(_pkg_resolve(ROOT / 'out') / 'holdout_locked') / 'retrieval_predictions.jsonl')
HOTPOT_DEV_FILE = _pkg_resolve(_pkg_resolve(_pkg_resolve(ROOT / 'data') / 'hotpotqa') / 'hotpot_dev_distractor.jsonl')
OUT_ROOT = _pkg_resolve(_pkg_resolve(ROOT / 'out') / 'p17_post_payload_reuse')
SEED = 0
BOOT_ITERS = 10000
BOOT_CHUNK = 500
SHORT_ANSWER_CHARS = 3
EXPECT_HOLDOUT_N = 4405
P_BUDGET = 2
COLLISION_KEYS = set()
MIN_CLUSTERS = 30
MAX_CLUSTER_SHARE = 0.5
SALTS = [0, 1, 2, 3, 4]
MAIN_SALT = 0
AUDIT_DISAGREE_N = 30
AUDIT_RANDOM_N = 30
PRE09_REFERENCE = {'answer_payload_accuracy': 0.423, 'passage_payload_accuracy': 0.69, 'note': '참고값. 이 실행 결과와 강제로 맞추지 않는다.'}

def sha256_file(p):
    h = hashlib.sha256()
    with open(_pkg_open_path(_pkg_resolve(p), 'rb'), 'rb') as fh:
        for chunk in iter(lambda : fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(t):
    return hashlib.sha256(t.encode('utf-8')).hexdigest()

def load_jsonl(p):
    with open(_pkg_open_path(_pkg_resolve(p), 'r'), encoding='utf-8') as fh:
        return [json.loads(l) for l in fh if l.strip()]

def load_module(name):
    path = _pkg_resolve(ROOT / name)
    if not path.exists():
        sys.exit(f'모듈 없음: {path}')
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return (mod, path)

def norm(s):
    return re.sub('[\\s.,;:\'\\"!?()\\-]', '', str(s).lower())

def tokens(s):
    return re.findall('[0-9a-z\\uac00-\\ud7a3]+', str(s).lower())

def golds_of(x):
    out = []

    def rec(y):
        if isinstance(y, str):
            out.append(y)
        elif isinstance(y, list):
            for z in y:
                rec(z)
    rec(x)
    return [g for g in out if norm(g)]

def trivial_kind(golds):
    if not golds:
        return 'no_gold'
    for g in golds:
        n = norm(g)
        if len(n) < SHORT_ANSWER_CHARS:
            return 'short'
        if re.fullmatch('[0-9]+', n):
            return 'numeric'
        if n in ('yes', 'no'):
            return 'yesno'
    return None

def hit_boundary(payload, golds):
    pt = tokens(payload)
    for g in golds:
        gt = tokens(g)
        if not gt:
            continue
        L = len(gt)
        for i in range(len(pt) - L + 1):
            if pt[i:i + L] == gt:
                return True
    return False

def hit_substring(payload, golds):
    p = norm(payload)
    return any((norm(g) in p for g in golds))

def answer_set(golds):
    return frozenset((norm(g) for g in golds))

def positive_int(s):
    v = int(s)
    if v <= 0:
        raise argparse.ArgumentTypeError('양의 정수만 허용')
    return v

def sample_trust(cluster_list):
    sizes = Counter(cluster_list)
    n_cl = len(sizes)
    total = len(cluster_list)
    largest = max(sizes.values()) if sizes else 0
    share = _pkg_resolve(largest / total) if total else 1.0
    trusted = n_cl >= MIN_CLUSTERS and share <= MAX_CLUSTER_SHARE
    reason = None if trusted else f'표본 연결요소 {n_cl}개 (최소 {MIN_CLUSTERS}) 또는 최대 점유율 {share:.3f} (상한 {MAX_CLUSTER_SHARE}) 기준 미달'
    return {'n_clusters': n_cl, 'largest_cluster': largest, 'largest_share': round(share, 4), 'trusted': trusted, 'reason': reason}

def boot_ci_question(d):
    rng = np.random.default_rng(SEED)
    n = len(d)
    (out, done) = ([], 0)
    while done < BOOT_ITERS:
        m = min(BOOT_CHUNK, BOOT_ITERS - done)
        out.append(d[rng.integers(0, n, size=(m, n))].mean(axis=1))
        done += m
    means = np.concatenate(out)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]

def boot_ci_cluster(d, clusters):
    uniq = sorted(set(clusters))
    pos = {c: i for (i, c) in enumerate(uniq)}
    k = len(uniq)
    sums = np.zeros(k)
    cnts = np.zeros(k)
    for (di, c) in zip(d, clusters):
        sums[pos[c]] += di
        cnts[pos[c]] += 1
    rng = np.random.default_rng(SEED)
    (out, done) = ([], 0)
    while done < BOOT_ITERS:
        m = min(BOOT_CHUNK, BOOT_ITERS - done)
        idx = rng.integers(0, k, size=(m, k))
        out.append(_pkg_resolve(sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)))
        done += m
    means = np.concatenate(out)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]

def crosstab(a, b):
    return {'both': int(((a == 1) & (b == 1)).sum()), 'QA_only': int(((a == 1) & (b == 0)).sum()), 'QC_only': int(((a == 0) & (b == 1)).sum()), 'neither': int(((a == 0) & (b == 0)).sum())}

def point_summary(rows, qa_key, qc_key, label):
    n = len(rows)
    if n == 0:
        return {'label': label, 'n': 0}
    a = np.asarray([r[qa_key] for r in rows], dtype=float)
    b = np.asarray([r[qc_key] for r in rows], dtype=float)
    return {'label': label, 'n': n, 'coverage_QA': round(float(a.mean()), 4), 'coverage_QC': round(float(b.mean()), 4), 'diff_QC_minus_QA': round(float((b - a).mean()), 4), 'crosstab': crosstab(a, b)}

def paired_summary(rows, qa_key, qc_key, cluster_key, label):
    n = len(rows)
    if n == 0:
        return {'label': label, 'n': 0, 'note': '대상 없음'}
    a = np.asarray([r[qa_key] for r in rows], dtype=float)
    b = np.asarray([r[qc_key] for r in rows], dtype=float)
    d = b - a
    trust = sample_trust([r[cluster_key] for r in rows])
    if trust['trusted']:
        (ci_cl, cl_note) = (boot_ci_cluster(d, [r[cluster_key] for r in rows]), None)
    else:
        (ci_cl, cl_note) = (None, f"클러스터 CI 미보고: {trust['reason']}")
    return {'label': label, 'n': n, 'coverage_QA': round(float(a.mean()), 4), 'coverage_QC': round(float(b.mean()), 4), 'diff_QC_minus_QA': round(float(d.mean()), 4), 'ci95_question_bootstrap': boot_ci_question(d), 'ci95_cluster_bootstrap': ci_cl, 'cluster_note': cl_note, 'sample_cluster_trust': trust, 'crosstab': crosstab(a, b), 'bootstrap': {'iters': BOOT_ITERS, 'seed': SEED, 'method': 'paired percentile, 95%'}}

def bins_by(rows, value_key, qa_key, qc_key, label):
    if not rows:
        return {'note': '대상 없음'}
    vals = np.asarray([r[value_key] for r in rows], dtype=float)
    edges = np.unique(np.percentile(vals, [0, 25, 50, 75, 100]))
    if len(edges) < 2:
        return {'note': '값의 분산이 없어 층화 불가', 'label': label}
    (bins, total_n, weighted) = ([], 0, 0.0)
    for i in range(len(edges) - 1):
        (lo, hi) = (edges[i], edges[i + 1])
        last = i == len(edges) - 2
        sel = [r for r in rows if (lo <= r[value_key] <= hi if last else lo <= r[value_key] < hi)]
        if not sel:
            continue
        a = float(np.mean([r[qa_key] for r in sel]))
        b = float(np.mean([r[qc_key] for r in sel]))
        bins.append({'range': f'{round(float(lo), 2)}-{round(float(hi), 2)}', 'n': len(sel), 'coverage_QA': round(a, 4), 'coverage_QC': round(b, 4), 'diff': round(b - a, 4)})
        total_n += len(sel)
        weighted += (b - a) * len(sel)
    return {'label': label, 'bins': bins, 'weighted_mean_diff': round(_pkg_resolve(weighted / total_n), 4) if total_n else None, 'note': '층화 차이의 가중 평균. 순수 재사용 효과가 아니다.'}

def write_csv(path, rows, fields):
    with open(_pkg_open_path(_pkg_resolve(path), 'w'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

def cmd_probe(_args):
    for p in [INDEX_FILE, SPLIT_IDS_FILE, PRED_FILE, HOTPOT_DEV_FILE]:
        print(f"{p}: {('있음' if p.exists() else '없음')}")
    for m in ['qi_common.py', 'retrieval_policy.py']:
        print(f"{m}: {('있음' if _pkg_resolve(ROOT / m).exists() else '없음')}")

def load_inputs(args):
    for p in [INDEX_FILE, SPLIT_IDS_FILE, PRED_FILE, HOTPOT_DEV_FILE]:
        if not p.exists():
            sys.exit(f'필수 파일 없음: {p}')
    split = json.loads(SPLIT_IDS_FILE.read_text(encoding='utf-8'))
    if not isinstance(split, dict) or 'holdout' not in split:
        sys.exit("split_ids.json 에 'holdout' 키가 없음")
    hold_ids = list(split['holdout'])
    if len(hold_ids) != len(set(hold_ids)):
        sys.exit('Holdout ID 중복')
    if not all((isinstance(q, str) and q for q in hold_ids)):
        sys.exit('Holdout ID 는 비지 않은 문자열이어야 함')
    if args.expect_n and len(hold_ids) != args.expect_n:
        sys.exit(f'Holdout 수 {len(hold_ids)} != 기대값 {args.expect_n}')
    index_rows = load_jsonl(INDEX_FILE)
    qid_to_item = {}
    for (i, r) in enumerate(index_rows):
        if not isinstance(r.get('qid'), str):
            sys.exit(f'색인 행 {i} 의 qid 가 문자열이 아님')
        if r['qid'] in qid_to_item:
            sys.exit(f"색인 qid 중복: {r['qid']}")
        qid_to_item[r['qid']] = i
    dev_by_id = {}
    for r in load_jsonl(HOTPOT_DEV_FILE):
        if r['id'] in dev_by_id:
            sys.exit(f"dev id 중복: {r['id']}")
        dev_by_id[r['id']] = r
    gold_map = {}
    for q in hold_ids:
        if q not in dev_by_id:
            sys.exit(f'Holdout qid 가 dev 파일에 없음: {q}')
        d = dev_by_id[q]
        if not isinstance(d['answer'], str) or not d['answer'].strip():
            sys.exit(f'dev 정답이 비었거나 문자열이 아님 (id={q})')
        gold_map[q] = {'answer': [d['answer']], 'type': d.get('type'), 'level': d.get('level')}
    preds = {}
    for r in load_jsonl(PRED_FILE):
        if r.get('qid') in preds:
            sys.exit(f"예측 파일 qid 중복: {r.get('qid')}")
        preds[r.get('qid')] = r
    missing_pred = [q for q in hold_ids if q not in preds]
    if missing_pred:
        sys.exit(f'Holdout 중 동결 예측이 없는 qid {len(missing_pred)}건')
    n_items = len(index_rows)
    for q in hold_ids:
        p = preds[q]
        top = p.get('qi_top_items')
        if not isinstance(top, list) or not top:
            sys.exit(f'빈 순위 또는 형식 오류: {q}')
        items = []
        for t in top:
            it = t.get('item')
            if not isinstance(it, int) or it < 0 or it >= n_items:
                sys.exit(f'순위 item 범위 오류 (qid={q}): {it}')
            items.append(it)
        if len(items) != len(set(items)):
            sys.exit(f'순위 item 중복 (qid={q})')
        qt = p.get('qi_titles')
        if not isinstance(qt, list) or len(qt) != P_BUDGET or (not all((isinstance(x, str) for x in qt))):
            sys.exit(f'qi_titles 형식 오류 (qid={q}): {qt!r}')
    overlap = set(hold_ids) & set(qid_to_item)
    if overlap and (not args.allow_overlap):
        sys.exit(f'Holdout 과 색인의 qid 겹침 {len(overlap)}건. 중단합니다.')
    inputs = {'index_items': {'path': str(INDEX_FILE), 'sha256': sha256_file(INDEX_FILE), 'rows': n_items}, 'split_ids': {'path': str(SPLIT_IDS_FILE), 'sha256': sha256_file(SPLIT_IDS_FILE), 'holdout_n': len(hold_ids)}, 'frozen_predictions': {'path': str(PRED_FILE), 'sha256': sha256_file(PRED_FILE), 'rows': len(preds)}, 'gold_source_hotpot_dev': {'path': str(HOTPOT_DEV_FILE), 'sha256': sha256_file(HOTPOT_DEV_FILE), 'dev_rows': len(dev_by_id), 'matched_holdout': len(gold_map), 'validation_scope': 'id 연결과 정답 문자열 형식만 검사'}, 'overlap_holdout_index': len(overlap), 'holdout_qid_list_sha256': sha256_text('\n'.join(sorted(hold_ids)))}
    return (index_rows, gold_map, preds, hold_ids, inputs)

def build_doc_structs(index_rows, qc):
    (item_doc_keys, key_text, key_title) = ([], {}, {})
    collisions = 0
    collision_keys = set()
    empty_paras = 0
    for r in index_rows:
        keys = []
        for p in qc.split_paragraphs(r.get('context', '')):
            if not p.get('text', '').strip():
                empty_paras += 1
                continue
            k = qc.para_key(p)
            keys.append(k)
            if k in key_text and (key_text[k] != p['text'] or key_title[k] != p['title']):
                collisions += 1
                collision_keys.add(k)
            if k not in key_text:
                key_text[k] = p['text']
                key_title[k] = p['title']
        item_doc_keys.append(keys)
    return (item_doc_keys, key_text, key_title, {'key_collisions_diff_text': collisions, 'collision_unique_keys': len(collision_keys), 'collision_keys': sorted(collision_keys), 'empty_paragraphs_skipped': empty_paras})

def make_policy(rp, item_doc_keys, key_text):
    """strict=True: P 개 문서를 확보하지 못하면 중단. strict=False: 부족하면 None 을 돌려준다."""
    cache = {}

    def select(item_list, strict=True):
        cache_key = tuple(item_list)
        if cache_key in cache:
            out = cache[cache_key]
        else:
            (out, _short) = rp.qi_policy(item_list, item_doc_keys, P=P_BUDGET)
            out = list(out)
            cache[cache_key] = out
        if len(out) != len(set(out)):
            sys.exit(f'P=2 선정 결과 중복: {out}')
        if len(out) != P_BUDGET:
            if strict:
                sys.exit(f'P=2 선정 결과 길이 이상: {out}')
            return None
        for k in out:
            if k not in key_text:
                sys.exit(f'선정 문서 키가 본문 사전에 없음: {k[:80]!r}')
        return out
    return select

def meta_clusters(rows, comp_item):
    """결정적 Union-Find 메타 클러스터."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in rows:
        labs = sorted({r['cluster']} | {comp_item[i] for i in r['supp_items']})
        for other in labs[1:]:
            (ra, rb) = (find(labs[0]), find(other))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    for r in rows:
        r['cluster'] = find(r['cluster'])
    return rows

def validate_frozen_all(hold_ids, preds, select, key_title, item_doc_keys):
    """평가 대상 필터링 이전에, Holdout 전체에 대해 동결 정책과 참조 제목을 검증한다."""
    (title_bad, frozen_bad) = ([], [])
    for qid in hold_ids:
        p = preds[qid]
        ranking = [int(t['item']) for t in p['qi_top_items']]
        keys = select(ranking)
        if [key_title[k] for k in keys] != list(p['qi_titles']):
            frozen_bad.append(qid)
        top1 = ranking[0]
        if not set(p['qi_top_items'][0].get('titles', [])) <= {key_title[k] for k in item_doc_keys[top1]}:
            title_bad.append(qid)
    if title_bad:
        sys.exit(f'상위 항목 제목이 색인 행과 불일치 {len(title_bad)}건 (전체 Holdout 검사). 중단합니다.')
    if frozen_bad:
        sys.exit(f'동결 기록 qi_titles 와 재계산 제목 불일치 {len(frozen_bad)}건 (전체 Holdout 검사). 중단합니다.')
    return {'checked_all_holdout': len(hold_ids), 'title_mismatch': 0, 'frozen_mismatch': 0}

def supplying_items(ranking, keys, item_doc_keys):
    """선정된 각 문서 키를 처음 공급한 순위 항목 (순위 순서, 중복 제거)."""
    (order, seen) = ([], set())
    for k in keys:
        for it in ranking:
            if k in item_doc_keys[it]:
                if it not in seen:
                    seen.add(it)
                    order.append(it)
                break
    return order

def supplier_audit(ranking, keys, item_doc_keys):
    counts = []
    for k in keys:
        suppliers = [it for it in ranking if k in item_doc_keys[it]]
        if not suppliers:
            raise ValueError(f'공급자 없는 문서: {k!r}')
        counts.append(len(suppliers))
    return {'supplier_counts': counts, 'n_multi_supplier_docs': sum((c > 1 for c in counts)), 'max_suppliers_per_doc': max(counts, default=0)}

def supplier_summary(rows):
    n = len(rows)
    multi = sum((r['n_multi_supplier_docs'] > 0 for r in rows))
    return {'n': n, 'rows_with_multi_supplier': multi, 'rate_with_multi_supplier': round(_pkg_resolve(multi / n), 4) if n else None, 'total_multi_supplier_docs': sum((r['n_multi_supplier_docs'] for r in rows)), 'max_suppliers_per_doc': max((r['max_suppliers_per_doc'] for r in rows), default=0)}

def supplier_sensitivity(rows, index_rows, gold_map, preds, item_doc_keys, select):
    (first_hits, last_hits) = ([], [])
    for r in rows:
        qid = r['holdout_qid']
        ranking = [int(t['item']) for t in preds[qid]['qi_top_items']]
        keys = select(ranking)
        golds = golds_of(gold_map[qid]['answer'])
        (first_items, last_items) = ([], [])
        for k in keys:
            suppliers = [it for it in ranking if k in item_doc_keys[it]]
            if not suppliers:
                raise ValueError(f'공급자 누락: {qid}')
            first_items.append(suppliers[0])
            last_items.append(suppliers[-1])
        first_items = list(dict.fromkeys(first_items))
        last_items = list(dict.fromkeys(last_items))
        first_text = ' | '.join((g for it in first_items for g in golds_of(index_rows[it]['answer'])))
        last_text = ' | '.join((g for it in last_items for g in golds_of(index_rows[it]['answer'])))
        first_hit = int(hit_boundary(first_text, golds))
        last_hit = int(hit_boundary(last_text, golds))
        if first_hit != r['qa_hit']:
            raise ValueError(f'기존 A2 결과 불일치: {qid}')
        first_hits.append(first_hit)
        last_hits.append(last_hit)
    n = len(rows)
    if not n:
        return {'n': 0, 'note': '대상 없음'}
    return {'n': n, 'first_supplier_coverage': round(_pkg_resolve(sum(first_hits) / n), 4), 'last_supplier_coverage': round(_pkg_resolve(sum(last_hits) / n), 4), 'delta_last_minus_first': round(_pkg_resolve((sum(last_hits) - sum(first_hits)) / n), 4), 'n_changed_hit': sum((a != b for (a, b) in zip(first_hits, last_hits))), 'method': 'first_vs_last_ranked_supplier'}

def common_meta_clusters(filtered_rows, all_rows, comp_item):
    meta_clusters(all_rows, comp_item)
    mapping = {r['holdout_qid']: r['cluster'] for r in all_rows}
    if len(mapping) != len(all_rows):
        raise ValueError('Holdout ID 중복')
    for r in filtered_rows:
        r['cluster'] = mapping[r['holdout_qid']]
    return {'unfiltered_clusters': len(set(mapping.values())), 'filtered_clusters': len(set((r['cluster'] for r in filtered_rows))), 'mapping_consistent': True}

def holdout_frozen(hold_ids, gold_map, preds, index_rows, select, key_text, key_title, item_doc_keys, comp_item, apply_filter=True):
    (rows, excluded) = ([], Counter())
    (title_mismatch, frozen_mismatch) = (0, 0)
    for qid in hold_ids:
        golds = golds_of(gold_map[qid]['answer'])
        if apply_filter:
            kind = trivial_kind(golds)
            if kind:
                excluded[kind] += 1
                continue
        p = preds[qid]
        ranking = [int(t['item']) for t in p['qi_top_items']]
        keys = select(ranking)
        if any((k in COLLISION_KEYS for k in keys)):
            excluded['collision_key_selected'] += 1
            continue
        titles = [key_title[k] for k in keys]
        if titles != list(p['qi_titles']):
            frozen_mismatch += 1
        top1 = ranking[0]
        if not set(p['qi_top_items'][0].get('titles', [])) <= {key_title[k] for k in item_doc_keys[top1]}:
            title_mismatch += 1
        sup = supplying_items(ranking, keys, item_doc_keys)
        if any((not any((k in item_doc_keys[it] for it in sup)) for k in keys)):
            sys.exit(f'공급 항목이 없는 선정 문서가 있음 (qid={qid})')
        supplier_info = supplier_audit(ranking, keys, item_doc_keys)
        qa_golds = []
        for it in sup:
            qa_golds.extend(golds_of(index_rows[it]['answer']))
        qa_text = ' | '.join(qa_golds)
        qc_text = ' '.join((key_text[k] for k in keys))
        rows.append({'holdout_qid': qid, 'type': gold_map[qid]['type'], 'level': gold_map[qid]['level'], 'ranking_len': len(ranking), 'supplying_items': ' '.join(map(str, sup)), 'supp_items': sup, 'n_selected_docs': len(keys), 'supplier_counts': ' '.join(map(str, supplier_info['supplier_counts'])), 'n_multi_supplier_docs': supplier_info['n_multi_supplier_docs'], 'max_suppliers_per_doc': supplier_info['max_suppliers_per_doc'], 'cluster': comp_item[top1], 'gold': ' | '.join(golds), 'qa_text': qa_text, 'qc_text': qc_text, 'qa_hit': int(hit_boundary(qa_text, golds)), 'qc_hit': int(hit_boundary(qc_text, golds)), 'qa_hit_substr': int(hit_substring(qa_text, golds)), 'qc_hit_substr': int(hit_substring(qc_text, golds)), 'qa_chars': len(qa_text), 'qc_chars': len(qc_text), 'qa_words': len(qa_text.split()), 'qc_words': len(qc_text.split())})
    return (rows, excluded, title_mismatch, frozen_mismatch)

def holdout_top1(hold_ids, gold_map, preds, index_rows, select, key_text, item_doc_keys, comp_item, apply_filter=True):
    (rows, excluded) = ([], Counter())
    for qid in hold_ids:
        golds = golds_of(gold_map[qid]['answer'])
        if apply_filter:
            kind = trivial_kind(golds)
            if kind:
                excluded[kind] += 1
                continue
        top1 = int(preds[qid]['qi_top_items'][0]['item'])
        ref_golds = golds_of(index_rows[top1]['answer'])
        if not ref_golds:
            excluded['empty_reference_answer'] += 1
            continue
        qa_text = ' | '.join(ref_golds)
        sel = select([top1], strict=False)
        if sel is None:
            excluded['top1_fewer_than_P_docs'] += 1
            continue
        if any((k in COLLISION_KEYS for k in sel)):
            excluded['collision_key_selected'] += 1
            continue
        qc_text = ' '.join((key_text[k] for k in sel))
        rows.append({'holdout_qid': qid, 'type': gold_map[qid]['type'], 'level': gold_map[qid]['level'], 'ref_item': top1, 'cluster': comp_item[top1], 'gold': ' | '.join(golds), 'qa_text': qa_text, 'qc_text': qc_text, 'qa_hit': int(hit_boundary(qa_text, golds)), 'qc_hit': int(hit_boundary(qc_text, golds)), 'qa_hit_substr': int(hit_substring(qa_text, golds)), 'qc_hit_substr': int(hit_substring(qc_text, golds)), 'qa_chars': len(qa_text), 'qc_chars': len(qc_text)})
    return (rows, excluded)

def build_doc_sharing(item_doc_keys):
    key_to_items = defaultdict(set)
    for (j, keys) in enumerate(item_doc_keys):
        for k in keys:
            key_to_items[k].add(j)
    return key_to_items

def components(n, key_to_items):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for items in key_to_items.values():
        items = list(items)
        for other in items[1:]:
            (ra, rb) = (find(items[0]), find(other))
            if ra != rb:
                parent[rb] = ra
    return [str(find(j)) for j in range(n)]

def loo_analysis(index_rows, item_doc_keys, key_to_items, comp_item, key_text, select, salt, limit=None):
    (rows, excluded) = ([], Counter())
    targets = range(len(index_rows)) if limit is None else range(min(limit, len(index_rows)))
    for j in targets:
        golds = golds_of(index_rows[j]['answer'])
        kind = trivial_kind(golds)
        if kind:
            excluded[f'target_{kind}'] += 1
            continue
        neighbors = set()
        for k in item_doc_keys[j]:
            neighbors |= key_to_items[k]
        neighbors.discard(j)
        if not neighbors:
            excluded['no_neighbor'] += 1
            continue
        eligible = sorted((n for n in neighbors if golds_of(index_rows[n]['answer']) and select([n], strict=False) is not None and (not any((k in COLLISION_KEYS for k in select([n], strict=False))))))
        if not eligible:
            excluded['no_eligible_neighbor'] += 1
            continue
        h = int(hashlib.sha1(f"{salt}|{index_rows[j]['qid']}".encode('utf-8')).hexdigest()[:12], 16)
        nb = eligible[h % len(eligible)]
        nb_golds = golds_of(index_rows[nb]['answer'])
        layer = 'exact_answer_set_match' if answer_set(nb_golds) == answer_set(golds) else 'exact_answer_set_diff'
        keys = select([nb], strict=True)
        qa_text = ' | '.join(nb_golds)
        qc_text = ' '.join((key_text[k] for k in keys))
        shared = any((k in set(item_doc_keys[j]) for k in keys))
        rows.append({'qid': index_rows[j]['qid'], 'item': j, 'neighbor_item': nb, 'cluster': comp_item[j], 'layer': layer, 'n_eligible_neighbors': len(eligible), 'salt': salt, 'selected_shares_target_doc': int(shared), 'gold': ' | '.join(golds), 'qa_text': qa_text, 'qc_text': qc_text, 'qa_hit': int(hit_boundary(qa_text, golds)), 'qc_hit': int(hit_boundary(qc_text, golds)), 'qa_hit_substr': int(hit_substring(qa_text, golds)), 'qc_hit_substr': int(hit_substring(qc_text, golds)), 'qa_chars': len(qa_text), 'qc_chars': len(qc_text)})
    return (rows, excluded)

def sharing_stats(item_doc_keys, key_to_items, index_rows):
    rows = []
    per_doc = Counter((len(v) for v in key_to_items.values()))
    for k in sorted(per_doc):
        rows.append({'type': 'questions_per_doc', 'value': k, 'count': per_doc[k]})
    variants = Counter()
    for items in key_to_items.values():
        variants[len({answer_set(golds_of(index_rows[j]['answer'])) for j in items})] += 1
    for k in sorted(variants):
        rows.append({'type': 'answer_variants_per_doc', 'value': k, 'count': variants[k]})
    per_ans = Counter((answer_set(golds_of(r['answer'])) for r in index_rows if golds_of(r['answer'])))
    dist = Counter(per_ans.values())
    for k in sorted(dist):
        rows.append({'type': 'questions_per_identical_answer', 'value': k, 'count': dist[k]})
    return rows

def index_cluster_rows(comp_item):
    dist = Counter(Counter(comp_item).values())
    return [{'cluster_size': k, 'n_clusters': dist[k]} for k in sorted(dist)]

def length_stats(rows, keys):
    out = []
    for k in keys:
        vals = np.asarray([r[k] for r in rows if k in r], dtype=float)
        if vals.size == 0:
            continue
        out.append({'metric': k, 'n': int(vals.size), 'mean': round(float(vals.mean()), 2), 'median': round(float(np.median(vals)), 2), 'p25': round(float(np.percentile(vals, 25)), 2), 'p75': round(float(np.percentile(vals, 75)), 2)})
    return out

def with_ratio(rows):
    out = []
    for r in rows:
        rr = dict(r)
        rr['qc_over_qa'] = _pkg_resolve(r['qc_chars'] / r['qa_chars']) if r['qa_chars'] > 0 else None
        out.append(rr)
    return [r for r in out if r['qc_over_qa'] is not None]
AUDIT_FIELDS = ['holdout_qid', 'gold', 'qa_text', 'qc_text', 'qa_boundary', 'qa_substring', 'qc_boundary', 'qc_substring']

def audit_rows(sample):
    return [{'holdout_qid': r['holdout_qid'], 'gold': r['gold'], 'qa_text': r['qa_text'], 'qc_text': r['qc_text'], 'qa_boundary': r['qa_hit'], 'qa_substring': r['qa_hit_substr'], 'qc_boundary': r['qc_hit'], 'qc_substring': r['qc_hit_substr']} for r in sample]

def exclusion_table(hold_excl, n_hold, loo_excl, n_loo):
    rows = []
    for (k, v) in sorted(hold_excl.items()):
        rows.append({'analysis': 'holdout', 'reason': k, 'count': v, 'share': round(_pkg_resolve(v / n_hold), 4) if n_hold else None})
    for (k, v) in sorted(loo_excl.items()):
        rows.append({'analysis': 'loo', 'reason': k, 'count': v, 'share': round(_pkg_resolve(v / n_loo), 4) if n_loo else None})
    return rows

def verify_outputs(tmp, expected):
    """산출물 존재, 행 수, JSON 집계, 교차표, 분석 수 합계를 검사한다."""
    problems = []
    for name in expected['files']:
        p = _pkg_resolve(tmp / name)
        if not p.exists() or p.stat().st_size == 0:
            problems.append(f'파일 없음 또는 비어 있음: {name}')
    for (name, n) in expected['csv_rows'].items():
        with open(_pkg_open_path(_pkg_resolve(_pkg_resolve(tmp / name)), 'r'), encoding='utf-8') as fh:
            got = sum((1 for _ in csv.DictReader(fh)))
        if got != n:
            problems.append(f'{name}: 행 {got} != 기대 {n}')
    stats = json.loads(_pkg_resolve(tmp / 'paired_statistics.json').read_text(encoding='utf-8'))
    for (key, n) in expected['json_n'].items():
        if stats[key].get('n') != n:
            problems.append(f"{key}.n={stats[key].get('n')} != {n}")
    for (key, ct) in expected['json_crosstab'].items():
        if stats[key].get('crosstab') != ct:
            problems.append(f"{key} crosstab 불일치: csv={ct} json={stats[key].get('crosstab')}")
    for (name, total, parts) in expected['sum_checks']:
        if total != parts:
            problems.append(f'{name} 합계 불일치: {total} != {parts}')
    if not expected['salt_sets_equal']:
        problems.append('salt 간 평가 질문 집합이 다름')
    if problems:
        sys.exit('출력 교차 검증 실패: ' + '; '.join(problems))
    return 'PASS'

def crosstab_from_rows(rows, a_key, b_key):
    return crosstab(np.asarray([r[a_key] for r in rows]), np.asarray([r[b_key] for r in rows]))

def cmd_run(args):
    (qc, qc_path) = load_module('qi_common.py')
    (rp, rp_path) = load_module('retrieval_policy.py')
    (index_rows, gold_map, preds, hold_ids, inputs) = load_inputs(args)
    all_hold = list(hold_ids)
    if args.limit_holdout:
        hold_ids = hold_ids[:args.limit_holdout]
    (item_doc_keys, key_text, key_title, doc_checks) = build_doc_structs(index_rows, qc)
    COLLISION_KEYS.clear()
    COLLISION_KEYS.update(doc_checks['collision_keys'])
    print(f"충돌 고유 키: {len(COLLISION_KEYS)}개 / 충돌 출현: {doc_checks['key_collisions_diff_text']}회")
    select = make_policy(rp, item_doc_keys, key_text)
    frozen_all = validate_frozen_all(all_hold, preds, select, key_title, item_doc_keys)
    key_to_items = build_doc_sharing(item_doc_keys)
    comp_item = components(len(index_rows), key_to_items)
    smoke = bool(args.limit_holdout or args.limit_loo)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = ('debug_' if args.allow_overlap else '') + ('smoke_' if smoke else '')
    base = _pkg_resolve(OUT_ROOT / f'{prefix}{ts}')
    tmp = _pkg_resolve(base.parent / f'.tmp_{base.name}')
    tmp.mkdir(parents=True, exist_ok=False)
    if args.allow_overlap:
        print('경고: --allow-overlap 디버깅 실행. debug_ 폴더에 저장되며 정식 결과가 아님.')
    try:
        (a2_rows, a2_excl, title_mis, frozen_mis) = holdout_frozen(hold_ids, gold_map, preds, index_rows, select, key_text, key_title, item_doc_keys, comp_item, apply_filter=True)
        if title_mis:
            sys.exit(f'상위 항목 제목이 색인 행과 불일치 {title_mis}건. 중단합니다 (행 번호 해석 오류 의심).')
        if frozen_mis:
            sys.exit(f'동결 기록 qi_titles 와 재계산 제목 불일치 {frozen_mis}건. 중단합니다.')
        (a2_all, a2_all_excl, _, _) = holdout_frozen(hold_ids, gold_map, preds, index_rows, select, key_text, key_title, item_doc_keys, comp_item, apply_filter=False)
        cluster_audit = common_meta_clusters(a2_rows, a2_all, comp_item)
        supplier_stats = supplier_summary(a2_rows)
        supplier_sensitivity_stats = supplier_sensitivity(a2_rows, index_rows, gold_map, preds, item_doc_keys, select)
        write_csv(_pkg_resolve(tmp / 'holdout_frozen_A2.csv'), a2_rows, ['holdout_qid', 'type', 'level', 'ranking_len', 'supplying_items', 'supplier_counts', 'n_multi_supplier_docs', 'max_suppliers_per_doc', 'cluster', 'gold', 'qa_text', 'qc_text', 'qa_hit', 'qc_hit', 'qa_hit_substr', 'qc_hit_substr', 'qa_chars', 'qc_chars', 'qa_words', 'qc_words'])
        a2_stats = paired_summary(a2_rows, 'qa_hit', 'qc_hit', 'cluster', 'holdout_frozen_A2_boundary')
        a2_sub = paired_summary(a2_rows, 'qa_hit_substr', 'qc_hit_substr', 'cluster', 'holdout_frozen_A2_substring')
        a2_nofilter = paired_summary(a2_all, 'qa_hit', 'qc_hit', 'cluster', 'holdout_frozen_A2_sensitivity_no_trivial_filter')
        dis_a2 = [r for r in a2_rows if r['qa_hit'] != r['qa_hit_substr'] or r['qc_hit'] != r['qc_hit_substr']]
        dis_sample = dis_a2 if len(dis_a2) <= AUDIT_DISAGREE_N else random.Random(SEED).sample(dis_a2, AUDIT_DISAGREE_N)
        rnd_sample = a2_rows if len(a2_rows) <= AUDIT_RANDOM_N else random.Random(SEED + 1).sample(a2_rows, AUDIT_RANDOM_N)
        write_csv(_pkg_resolve(tmp / 'boundary_disagreement_sample.csv'), audit_rows(dis_sample), AUDIT_FIELDS)
        write_csv(_pkg_resolve(tmp / 'random_audit_sample.csv'), audit_rows(rnd_sample), AUDIT_FIELDS)
        (a1_rows, a1_excl) = holdout_top1(hold_ids, gold_map, preds, index_rows, select, key_text, item_doc_keys, comp_item, apply_filter=True)
        write_csv(_pkg_resolve(tmp / 'holdout_top1_A1.csv'), a1_rows, ['holdout_qid', 'type', 'level', 'ref_item', 'cluster', 'gold', 'qa_text', 'qc_text', 'qa_hit', 'qc_hit', 'qa_hit_substr', 'qc_hit_substr', 'qa_chars', 'qc_chars'])
        a1_stats = paired_summary(a1_rows, 'qa_hit', 'qc_hit', 'cluster', 'holdout_top1_A1_boundary')
        (loo_rows, loo_excl) = loo_analysis(index_rows, item_doc_keys, key_to_items, comp_item, key_text, select, MAIN_SALT, args.limit_loo)
        write_csv(_pkg_resolve(tmp / 'loo_payload_comparison.csv'), loo_rows, ['qid', 'item', 'neighbor_item', 'cluster', 'layer', 'n_eligible_neighbors', 'salt', 'selected_shares_target_doc', 'gold', 'qa_text', 'qc_text', 'qa_hit', 'qc_hit', 'qa_hit_substr', 'qc_hit_substr', 'qa_chars', 'qc_chars'])
        loo_all = paired_summary(loo_rows, 'qa_hit', 'qc_hit', 'cluster', 'loo_all_boundary')
        loo_all_sub = paired_summary(loo_rows, 'qa_hit_substr', 'qc_hit_substr', 'cluster', 'loo_all_substring')
        diff_rows = [r for r in loo_rows if r['layer'] == 'exact_answer_set_diff']
        match_rows = [r for r in loo_rows if r['layer'] == 'exact_answer_set_match']
        loo_diff = paired_summary(diff_rows, 'qa_hit', 'qc_hit', 'cluster', 'loo_diff_boundary')
        loo_match = paired_summary(match_rows, 'qa_hit', 'qc_hit', 'cluster', 'loo_match_boundary')
        shared_rate = _pkg_resolve(sum((r['selected_shares_target_doc'] for r in loo_rows)) / len(loo_rows)) if loo_rows else None
        (salt_rows, salt_sets) = ([], [])
        for s in SALTS:
            (rows_s, _) = loo_analysis(index_rows, item_doc_keys, key_to_items, comp_item, key_text, select, s, args.limit_loo)
            salt_sets.append(frozenset((r['qid'] for r in rows_s)))
            pt = point_summary(rows_s, 'qa_hit', 'qc_hit', f'salt_{s}')
            salt_rows.append({'salt': s, 'n': pt['n'], 'coverage_QA': pt.get('coverage_QA'), 'coverage_QC': pt.get('coverage_QC'), 'diff': pt.get('diff_QC_minus_QA')})
        salt_sets_equal = all((x == salt_sets[0] for x in salt_sets))
        write_csv(_pkg_resolve(tmp / 'loo_salt_sensitivity.csv'), salt_rows, ['salt', 'n', 'coverage_QA', 'coverage_QC', 'diff'])
        write_csv(_pkg_resolve(tmp / 'index_cluster_sizes.csv'), index_cluster_rows(comp_item), ['cluster_size', 'n_clusters'])
        write_csv(_pkg_resolve(tmp / 'exclusion_composition.csv'), exclusion_table(a2_excl, len(hold_ids), loo_excl, len(index_rows)), ['analysis', 'reason', 'count', 'share'])
        write_csv(_pkg_resolve(tmp / 'grounding_sharing_stats.csv'), sharing_stats(item_doc_keys, key_to_items, index_rows), ['type', 'value', 'count'])
        len_rows = length_stats(a2_rows, ['qa_chars', 'qc_chars', 'qa_words', 'qc_words'])
        write_csv(_pkg_resolve(tmp / 'payload_length_stats.csv'), len_rows, ['metric', 'n', 'mean', 'median', 'p25', 'p75'])
        n_zero_qa_chars = sum((r['qa_chars'] == 0 for r in a2_rows))
        ratio_rows = with_ratio(a2_rows)
        assert len(ratio_rows) + n_zero_qa_chars == len(a2_rows)
        ratio_strata = bins_by(ratio_rows, 'qc_over_qa', 'qa_hit', 'qc_hit', 'A2_ratio_strata')
        length_strata = bins_by(a2_rows, 'qc_chars', 'qa_hit', 'qc_hit', 'A2_qc_chars_strata')
        bins_flat = [dict(bin=f"qc_chars:{b['range']}", **{k: b[k] for k in ('n', 'coverage_QA', 'coverage_QC')}) for b in length_strata.get('bins', [])] + [dict(bin=f"qc_over_qa:{b['range']}", **{k: b[k] for k in ('n', 'coverage_QA', 'coverage_QC')}) for b in ratio_strata.get('bins', [])]
        write_csv(_pkg_resolve(tmp / 'length_bins_A2.csv'), bins_flat, ['bin', 'n', 'coverage_QA', 'coverage_QC'])
        by_type = {t: point_summary([r for r in a2_rows if r['type'] == t], 'qa_hit', 'qc_hit', t) for t in sorted({r['type'] for r in a2_rows if r['type']})}
        by_level = {t: point_summary([r for r in a2_rows if r['level'] == t], 'qa_hit', 'qc_hit', t) for t in sorted({r['level'] for r in a2_rows if r['level']})}
        g_shared = [r for r in loo_rows if r['selected_shares_target_doc'] == 1]
        g_not = [r for r in loo_rows if r['selected_shares_target_doc'] == 0]
        loo_by_shared = {'selected_shares_target_doc_1': paired_summary(g_shared, 'qa_hit', 'qc_hit', 'cluster', 'loo_shared_doc_selected'), 'selected_shares_target_doc_0': paired_summary(g_not, 'qa_hit', 'qc_hit', 'cluster', 'loo_shared_doc_not_selected')}
        stats_obj = {'analysis_note': '탐색적 구조 분석. 확증 결과 아님. 신규 LLM 생성 없음.', 'A2_frozen_main': a2_stats, 'A2_supplier_audit': supplier_stats, 'A2_supplier_sensitivity': supplier_sensitivity_stats, 'A2_cluster_mapping': cluster_audit, 'A2_zero_length_QA_count': n_zero_qa_chars, 'A2_ratio_strata_n': len(ratio_rows), 'A2_aux_substring': a2_sub, 'A2_sensitivity_no_trivial_filter': a2_nofilter, 'A2_by_type': by_type, 'A2_by_level': by_level, 'A2_ratio_strata': ratio_strata, 'A2_length_strata': length_strata, 'A2_excluded': dict(a2_excl), 'A2_unfiltered_excluded': dict(a2_all_excl), 'collision_policy': 'exclude_selected_collision_keys', 'collision_unique_keys': len(COLLISION_KEYS), 'A2_boundary_disagreement': {'total': len(dis_a2), 'sampled': len(dis_sample)}, 'A1_top1_supplementary': a1_stats, 'A1_excluded': dict(a1_excl), 'LOO_definition': 'shared_document_cross_question_reuse', 'LOO_document_removed': False, 'loo_all': loo_all, 'loo_all_substring': loo_all_sub, 'loo_diff_supplementary': loo_diff, 'loo_match': loo_match, 'loo_selected_shares_target_doc_rate': round(shared_rate, 4) if shared_rate is not None else None, 'loo_by_shared_doc_selection': loo_by_shared, 'A2_frozen_validation_all_holdout': frozen_all, 'loo_excluded': dict(loo_excl), 'loo_salt_qid_sets_equal': salt_sets_equal, 'pre09_reference': PRE09_REFERENCE}
        _pkg_output(_pkg_resolve(tmp / 'paired_statistics.json')).write_text(json.dumps(stats_obj, ensure_ascii=False, indent=2), encoding='utf-8')
        files = ['holdout_frozen_A2.csv', 'holdout_top1_A1.csv', 'loo_payload_comparison.csv', 'boundary_disagreement_sample.csv', 'random_audit_sample.csv', 'loo_salt_sensitivity.csv', 'index_cluster_sizes.csv', 'exclusion_composition.csv', 'grounding_sharing_stats.csv', 'payload_length_stats.csv', 'length_bins_A2.csv', 'paired_statistics.json']
        consistency = verify_outputs(tmp, {'files': files, 'csv_rows': {'holdout_frozen_A2.csv': len(a2_rows), 'holdout_top1_A1.csv': len(a1_rows), 'loo_payload_comparison.csv': len(loo_rows), 'loo_salt_sensitivity.csv': len(SALTS), 'boundary_disagreement_sample.csv': len(dis_sample), 'random_audit_sample.csv': len(rnd_sample)}, 'json_n': {'A2_frozen_main': len(a2_rows), 'A1_top1_supplementary': len(a1_rows), 'loo_all': len(loo_rows), 'loo_diff_supplementary': len(diff_rows), 'loo_match': len(match_rows)}, 'json_crosstab': {'A2_frozen_main': crosstab_from_rows(a2_rows, 'qa_hit', 'qc_hit'), 'loo_all': crosstab_from_rows(loo_rows, 'qa_hit', 'qc_hit')}, 'sum_checks': [('holdout 분석+제외', len(a2_rows) + sum(a2_excl.values()), len(hold_ids)), ('loo 분석+제외', len(loo_rows) + sum(loo_excl.values()), min(args.limit_loo, len(index_rows)) if args.limit_loo is not None else len(index_rows))], 'salt_sets_equal': salt_sets_equal})
        report = ['p17 POST-17 v12 감사 보고서', f"실행 시각: {ts}  [{('스모크' if smoke else '정식')}]", '주 분석 A2 = 동결 전체 순위 P=2 재현 조건. 보조 A1 = Top-1 조건 (동결 재현 아님).', f"A2 복수 공급자 포함 질문: {supplier_stats['rows_with_multi_supplier']}건", f"A2 복수 공급자 질문 비율: {supplier_stats['rate_with_multi_supplier']}", f"A2 공급자 변경 시 QA 판정 변화: {supplier_sensitivity_stats['n_changed_hit']}건", f"A2 첫/마지막 공급자 Coverage: {supplier_sensitivity_stats['first_supplier_coverage']} / {supplier_sensitivity_stats['last_supplier_coverage']}", f'A2 QA 길이 0 표본: {n_zero_qa_chars}건', f"A2 공통 클러스터 수(전체/필터): {cluster_audit['unfiltered_clusters']}/{cluster_audit['filtered_clusters']}", 'A2 Q-A는 선정 문서별 최초 순위 공급 항목의 정답을 사용함.', f'충돌 고유 키: {len(COLLISION_KEYS)}개', f"A2 충돌 제외: {a2_excl.get('collision_key_selected', 0)}건", f"A2 무필터 충돌 제외: {a2_all_excl.get('collision_key_selected', 0)}건", f"A1 충돌 제외: {a1_excl.get('collision_key_selected', 0)}건", 'LOO: 충돌 키가 선정되는 이웃을 Salt 선택 전에 제외함.', '충돌 본문은 임의 복원하지 않고 분석에서 격리함.', 'LOO는 문서를 제거한 실험이 아닌 공유 문서 기반 질문 간 재사용 분석임.', f'동결 기록 대조(제목 수준): 불일치 {frozen_mis}건 (0이어야 함, 아니면 중단됨)', f'상위 항목 제목과 색인 행 제목 불일치: {title_mis}건 (0이어야 함, 아니면 중단됨)', '검증 범위: 동결 대조는 제목 수준. 같은 제목의 다른 문서는 구분되지 않는다.', f'Holdout 행: {len(hold_ids)} / A2 분석: {len(a2_rows)} / 제외: {dict(a2_excl)}', f"A2 표본 클러스터 게이트: {('보고 가능' if a2_stats['sample_cluster_trust']['trusted'] else a2_stats['cluster_note'])}", f'LOO 분석: {len(loo_rows)} / 제외: {dict(loo_excl)}', f"LOO 공유 문서가 선정 문서에 포함된 비율: {stats_obj['loo_selected_shares_target_doc_rate']}", f"LOO 표본 클러스터 게이트: {('보고 가능' if loo_all['sample_cluster_trust']['trusted'] else loo_all['cluster_note'])}", f"색인 문단 키 충돌(같은 키, 다른 본문): {doc_checks['key_collisions_diff_text']}건", f'salt 간 평가 질문 집합 동일: {salt_sets_equal}', 'exact_answer_set_* 는 정규화된 정답 집합의 완전 일치일 뿐 의미 동일성이 아님.', 'LOO 주 결과는 전체(all). diff 층은 보조 해석만 한다.', '클러스터 기준은 사전 지정 진단 휴리스틱이며 검증된 임계값이 아니다.', '정답 판정: primary=토큰 경계, auxiliary=부분 문자열. 불일치와 무작위 감사 표본을 수동 검토용으로 저장.', 'Holdout 정답 출처: HotpotQA dev 의 id/answer. 평가 분할 원본과의 동일성은 manifest 해시로만 기록.', '길이 층화는 순수 재사용 효과가 아니다. 길이 비율 층화(A2_ratio_strata)도 함께 본다.', f'출력 교차 검증: {consistency}', 'PRE-09 참고값은 비교 기준이 아니다.']
        _pkg_output(_pkg_resolve(tmp / 'audit_report.txt')).write_text('\n'.join(report) + '\n', encoding='utf-8')
        _pkg_output(_pkg_resolve(tmp / 'run_manifest.json')).write_text(json.dumps({'created': datetime.datetime.now().isoformat(timespec='seconds'), 'script_sha256': sha256_file(pathlib.Path(_pkg_resolve(__file__)).resolve()), 'qi_common': {'path': str(qc_path), 'sha256': sha256_file(qc_path)}, 'retrieval_policy': {'path': str(rp_path), 'sha256': sha256_file(rp_path), 'func': 'qi_policy', 'P': P_BUDGET}, 'inputs': inputs, 'frozen_validation_scope': 'title_level', 'frozen_document_identity_verified': False, 'frozen_validation_limit': 'Document titles match frozen qi_titles; paragraph identity is not independently verified.', 'A2_supplier_audit': supplier_stats, 'A2_supplier_sensitivity': supplier_sensitivity_stats, 'A2_cluster_mapping': cluster_audit, 'A2_zero_length_QA_count': n_zero_qa_chars, 'A2_ratio_strata_n': len(ratio_rows), 'A2_supplier_rule': 'first_ranked_supplier', 'LOO_definition': 'shared_document_cross_question_reuse', 'LOO_document_removed': False, 'doc_checks': doc_checks, 'collision_policy': 'exclude_selected_collision_keys', 'collision_selected_A2': a2_excl.get('collision_key_selected', 0), 'collision_selected_A2_unfiltered': a2_all_excl.get('collision_key_selected', 0), 'collision_selected_A1': a1_excl.get('collision_key_selected', 0), 'title_mismatch_top1': title_mis, 'frozen_title_mismatch': frozen_mis, 'excluded_A2': dict(a2_excl), 'excluded_A1': dict(a1_excl), 'excluded_loo': dict(loo_excl), 'analyzed_A2_qid_sha256': sha256_text('\n'.join(sorted((r['holdout_qid'] for r in a2_rows)))), 'analyzed_loo_qid_sha256': sha256_text('\n'.join(sorted((r['qid'] for r in loo_rows)))), 'consistency': consistency, 'smoke': smoke, 'limit_holdout': args.limit_holdout, 'limit_loo': args.limit_loo, 'salts': SALTS, 'main_salt': MAIN_SALT, 'seed': SEED, 'bootstrap_iters': BOOT_ITERS, 'min_clusters': MIN_CLUSTERS, 'max_cluster_share': MAX_CLUSTER_SHARE, 'expect_holdout_n': args.expect_n, 'allow_overlap': args.allow_overlap}, ensure_ascii=False, indent=2), encoding='utf-8')
        out_hashes = {}
        for p in sorted(tmp.iterdir()):
            if p.is_file():
                out_hashes[p.name] = {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
        _pkg_output(_pkg_resolve(tmp / 'output_hashes.json')).write_text(json.dumps(out_hashes, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.rename(base)
        recheck = json.loads(_pkg_resolve(base / 'output_hashes.json').read_text(encoding='utf-8'))
        present = {p.name for p in base.iterdir() if p.is_file() and p.name != 'output_hashes.json'}
        bad = [n for (n, v) in recheck.items() if not _pkg_resolve(base / n).exists() or sha256_file(_pkg_resolve(base / n)) != v['sha256']]
        if bad or present != set(recheck):
            shutil.rmtree(base, ignore_errors=True)
            sys.exit(f'최종 산출물 해시 재검증 실패: {bad}, 파일 집합 일치: {present == set(recheck)}')
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f'저장 완료: {base}')
    print(f'A2 주 분석: {json.dumps(a2_stats, ensure_ascii=False)}')
    print(f'LOO 전체: {json.dumps(loo_all, ensure_ascii=False)}')

def main():
    ap = argparse.ArgumentParser(description='p17 POST-17 payload reuse audit v12')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('probe', help='입력 파일 존재 확인')
    r = sub.add_parser('run', help='분석 실행')
    r.add_argument('--expect-n', type=int, default=EXPECT_HOLDOUT_N)
    r.add_argument('--limit-holdout', type=positive_int, default=None, help='스모크: Holdout 앞 N개')
    r.add_argument('--limit-loo', type=positive_int, default=None, help='스모크: LOO 대상 앞 N개')
    r.add_argument('--allow-overlap', action='store_true', help='디버깅 전용 (debug_ 폴더)')
    args = ap.parse_args()
    if args.cmd == 'probe':
        cmd_probe(args)
    else:
        cmd_run(args)
if __name__ == '__main__':
    main()
