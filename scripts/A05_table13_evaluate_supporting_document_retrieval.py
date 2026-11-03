"""
e1a_score.py  (E1-A 채점기: Gold Document Recall@2, H1 판정)
=============================================================
실행 위치: ~/2610-qi-rag

  python e1a_score.py --sample     탐색 50문항 채점 배선 확인 -> out/e1a/sample_scores/ (탐색 결과)
  python e1a_score.py --holdout    동결 Holdout 채점 -> out/holdout_final/ (확증 결과)

Gold 사용 시점
  - Gold 는 채점 단계에서만 읽는다. 검색 단계는 Gold 를 읽지 않는다.
  - --holdout 은 아래 게이트를 모두 통과해야 실행된다.
      1) FREEZE.json 존재, 분할 해시 일치
      2) retrieval_manifest.json 의 예측 파일 해시가 현재 파일과 일치
      3) 분할 해시가 manifest 와 FREEZE 에서 같음
      4) 결과 파일이 아직 없음 (덮어쓰기 금지)

판정 규칙 (protocol_v2.1 H1)
  R_i = |Top2(q_i) ∩ G_i| / |G_i|        (G_i: 공식 Gold 제목 집합)
  Delta = mean(R_QI - R_Vanilla)
  95% 양측 percentile CI 하한 > 0 이면 H1 지지, 아니면 미지지로 보고
  Bootstrap: 10,000회, 시드 0, 별도 RNG 인스턴스 (PCG64)

Gold 처리
  - 공식 Gold 제목이 있으면 코퍼스 존재 여부와 무관하게 포함. 검색되지 않으면 0.
  - 공식 Gold 목록이 비어 있는 문항만 제외하고 개수를 보고.
  - 코퍼스 접근 가능성은 별도 분포로만 보고 (분석 제외 기준으로 쓰지 않음).
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse
import csv
import datetime
import json
import os
import sys
from collections import Counter
import numpy as np
from S00_qi_common import DEV, EVAL, INDEX, LOCKED, build_corpus, load_jsonl, rid, sha256_file
ROOT = str(_PACKAGE_ROOT)
N_BOOT = 10000
BOOT_SEED = 0
BOOT_CHUNK = 1000
CI = 0.95
SPLIT = _pkg_resolve(os.path.join(LOCKED, 'split_ids.json'))
FREEZE = _pkg_resolve(os.path.join(LOCKED, 'FREEZE.json'))
HOLD_PRED = _pkg_resolve(os.path.join(LOCKED, 'retrieval_predictions.jsonl'))
HOLD_MAN = _pkg_resolve(os.path.join(LOCKED, 'retrieval_manifest.json'))
SAMPLE_PRED = _pkg_resolve(os.path.join(ROOT, 'out', 'e1a', 'sample_predictions.jsonl'))
OUT_SAMPLE = _pkg_resolve(os.path.join(ROOT, 'out', 'e1a', 'sample_scores'))
OUT_HOLD = _pkg_resolve(os.path.join(ROOT, 'out', 'holdout_final'))

def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')

def recall(returned, gold):
    g = set((x for x in gold if x))
    if not g:
        return None
    return _pkg_resolve(len(g & set((x for x in returned if x))) / len(g))

def holdout_gate():
    problems = []
    if not os.path.exists(FREEZE):
        problems.append('FREEZE.json 없음')
        return problems
    fz = json.load(open(_pkg_open_path(_pkg_resolve(FREEZE), 'r'), encoding='utf-8'))
    if not os.path.exists(SPLIT) or sha256_file(SPLIT) != fz['split_ids_sha256']:
        problems.append('분할 파일 해시가 동결 기록과 다름')
    if not os.path.exists(HOLD_MAN):
        problems.append('retrieval_manifest.json 없음 (e1a_retrieve.py --holdout 필요)')
        return problems
    man = json.load(open(_pkg_open_path(_pkg_resolve(HOLD_MAN), 'r'), encoding='utf-8'))
    if not os.path.exists(HOLD_PRED) or sha256_file(HOLD_PRED) != man['predictions_sha256']:
        problems.append('검색 결과 파일 해시가 기록과 다름 (결과가 바뀌었을 수 있음)')
    if man.get('split_ids_sha256') != fz['split_ids_sha256']:
        problems.append('manifest 의 분할 해시가 동결 기록과 다름')
    if man.get('gold_used_in_retrieval') is not False:
        problems.append('검색 단계에서 Gold 사용 기록이 있음')
    return problems

def gold_for_holdout(ids):
    out = {}
    with open(_pkg_open_path(_pkg_resolve(DEV), 'r'), encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            i = rid(r)
            if i in ids:
                out[i] = sorted(set(r['supporting_facts']['title']))
    return out

def gold_for_sample(ids):
    out = {}
    for e in load_jsonl(_pkg_resolve(os.path.join(ROOT, 'data', '_split', 'eval_queries.jsonl'))):
        q = str(e['qid'])
        if q in ids:
            out[q] = sorted(set(e.get('gold_titles', [])))
    return out

def bootstrap_ci(d):
    rng = np.random.Generator(np.random.PCG64(BOOT_SEED))
    n = len(d)
    boots = np.empty(N_BOOT)
    for s in range(0, N_BOOT, BOOT_CHUNK):
        idx = rng.integers(0, n, (BOOT_CHUNK, n))
        boots[s:s + BOOT_CHUNK] = d[idx].mean(axis=1)
    (lo, hi) = np.percentile(boots, [_pkg_resolve(100 * (1 - CI) / 2), _pkg_resolve(100 * (1 + CI) / 2)])
    return (float(lo), float(hi))

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--sample', action='store_true')
    g.add_argument('--holdout', action='store_true')
    args = ap.parse_args()
    if args.holdout:
        problems = holdout_gate()
        if problems:
            print('[채점 안 함] 게이트 실패:')
            for p in problems:
                print('  -', p)
            sys.exit(1)
        (pred_path, out_dir, label) = (HOLD_PRED, OUT_HOLD, '확증 (Holdout)')
        out_names = ['e1a_scores.csv', 'e1a_statistics.json', 'e1a_report.txt']
        if any((os.path.exists(_pkg_resolve(os.path.join(out_dir, n))) for n in out_names)):
            print('[채점 안 함] out/holdout_final 에 결과가 이미 있음 (덮어쓰기 금지)')
            sys.exit(1)
    else:
        (pred_path, out_dir, label) = (SAMPLE_PRED, OUT_SAMPLE, '탐색 샘플 (배선 확인용, 확증 아님)')
    rows = load_jsonl(pred_path)
    ids = set((str(r['qid']) for r in rows))
    gold = gold_for_holdout(ids) if args.holdout else gold_for_sample(ids)
    corpus_titles = set((d['title'] for d in build_corpus(load_jsonl(INDEX)) if d['title']))
    (recs, excluded) = ([], 0)
    (gold_n_dist, avail_dist) = (Counter(), Counter())
    for r in rows:
        qid = str(r['qid'])
        g_list = gold.get(qid, [])
        gold_n_dist[len(g_list)] += 1
        if not g_list:
            excluded += 1
            continue
        rq = recall(r['qi_titles'], g_list)
        rv = recall(r['va_titles'], g_list)
        in_corpus = sum((1 for t in g_list if t in corpus_titles))
        avail_dist[in_corpus] += 1
        recs.append({'qid': qid, 'gold_n': len(g_list), 'gold_in_corpus': in_corpus, 'qi_recall': rq, 'va_recall': rv, 'diff': rq - rv, 'qi_short': int(r['qi_short']), 'va_short': int(r['va_short'])})
    if not recs:
        print('채점할 문항이 없습니다.')
        sys.exit(1)
    qi = np.array([x['qi_recall'] for x in recs])
    va = np.array([x['va_recall'] for x in recs])
    d = qi - va
    (lo, hi) = bootstrap_ci(d)
    decision = 'H1 지지 (CI 하한 > 0)' if lo > 0 else 'H1 미지지 (CI 하한 <= 0). 결과 그대로 보고'
    stats = {'label': label, 'created_at': now(), 'n_scored': len(recs), 'n_excluded_no_gold': excluded, 'qi_mean': float(qi.mean()), 'va_mean': float(va.mean()), 'delta_mean': float(d.mean()), 'delta_ci95': [lo, hi], 'bootstrap': {'n': N_BOOT, 'seed': BOOT_SEED, 'rng': 'PCG64'}, 'gold_count_distribution': {str(k): v for (k, v) in sorted(gold_n_dist.items())}, 'gold_in_corpus_distribution': {str(k): v for (k, v) in sorted(avail_dist.items())}, 'qi_short_count': int(sum((x['qi_short'] for x in recs))), 'va_short_count': int(sum((x['va_short'] for x in recs))), 'decision': decision, 'pred_sha256': sha256_file(pred_path)}
    os.makedirs(out_dir, exist_ok=True)
    with open(_pkg_open_path(_pkg_resolve(_pkg_resolve(os.path.join(out_dir, 'e1a_scores.csv'))), 'w'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    with open(_pkg_open_path(_pkg_resolve(_pkg_resolve(os.path.join(out_dir, 'e1a_statistics.json'))), 'w'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    lines = [f'E1-A Gold Document Recall@2  [{label}]', '=' * 60, f'채점 문항: {len(recs):,}   제외(공식 Gold 없음): {excluded}', f"QI-RAG 평균 Recall@2:  {stats['qi_mean']:.4f}", f"Vanilla 평균 Recall@2: {stats['va_mean']:.4f}", f"차이 (QI - Vanilla):   {stats['delta_mean']:+.4f}", f'95% CI (paired bootstrap {N_BOOT:,}회, 시드 {BOOT_SEED}): [{lo:+.4f}, {hi:+.4f}]', f'판정: {decision}', '', f"공식 Gold 개수 분포: {stats['gold_count_distribution']}", f"코퍼스에 존재하는 Gold 개수 분포: {stats['gold_in_corpus_distribution']}", f"부족 사례 QI {stats['qi_short_count']} / Vanilla {stats['va_short_count']}", '', '해석 제한: 이 결과는 질문 색인 검색 구조의 차이만 보며, 생성 답변 정확도가 아니다.']
    with open(_pkg_open_path(_pkg_resolve(_pkg_resolve(os.path.join(out_dir, 'e1a_report.txt'))), 'w'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n저장: {out_dir}')
if __name__ == '__main__':
    main()
