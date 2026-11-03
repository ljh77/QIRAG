"""
e1a_retrieve.py  (E1-A 검색 실행기, Gold 미사용)
================================================
실행 위치: ~/2610-qi-rag

  python e1a_retrieve.py --sample     탐색 50문항 검색 점검 -> out/e1a/sample_*.{jsonl,json}
  python e1a_retrieve.py --holdout    동결 Holdout 4,405문항 검색 -> out/holdout_locked/

Gold 미사용 원칙
  - 질문 텍스트(question, original)만 읽는다. 정답, supporting_facts, gold_titles 는 읽지 않는다.
  - 결과 파일에 정답 관련 필드가 없다. 채점은 e1a_score.py 가 동결 후에 한다.

정책 (protocol_v2.1 과 동일)
  - 질문 인덱스 상위 8개 -> 연결 문서를 순회 -> 고유 문서 P=2
  - 문서 인덱스 상위 10개 -> 고유 문서 P=2
  - 부족은 기록만 하고 임의로 채우지 않는다

--holdout 게이트 (하나라도 실패하면 실행 안 함)
  1) out/holdout_locked/FREEZE.json 존재, 분할 파일 해시 일치
  2) out/e1a/sample_status.json 이 PASS, 기록된 코드 해시가 현재 파일과 일치
  3) 예측 파일이 아직 없음 (덮어쓰기 금지)
  명세 위반(불변식 실패)이 있으면 예측 파일을 저장하지 않는다.
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import argparse
import datetime
import json
import os
import sys
import numpy as np
from S00_qi_common import DEV, ENCODER, EVAL, INDEX, LOCKED, build_corpus, corpus_fingerprint, find_cache, load_jsonl, para_key, rid, sha256_file, split_paragraphs
from S00_retrieval_policy import qi_policy, vanilla_policy
ROOT = str(_PACKAGE_ROOT)
P = 2
TOP_ITEMS = 8
TOP_DOCS = 10
SAMPLE_N = 50
SAMPLE_SEED = 0
HOLDOUT_N = 4405
OUT_E1A = _pkg_resolve(os.path.join(ROOT, 'out', 'e1a'))
SPLIT = _pkg_resolve(os.path.join(LOCKED, 'split_ids.json'))
FREEZE = _pkg_resolve(os.path.join(LOCKED, 'FREEZE.json'))
SAMPLE_PRED = _pkg_resolve(os.path.join(OUT_E1A, 'sample_predictions.jsonl'))
SAMPLE_STATUS = _pkg_resolve(os.path.join(OUT_E1A, 'sample_status.json'))
HOLD_PRED = _pkg_resolve(os.path.join(LOCKED, 'retrieval_predictions.jsonl'))
HOLD_MAN = _pkg_resolve(os.path.join(LOCKED, 'retrieval_manifest.json'))
CODE = ['e1a_retrieve.py', 'retrieval_policy.py', 'qi_common.py']

def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')

def code_hashes():
    return {f: sha256_file(_pkg_resolve(os.path.join(ROOT, f))) for f in CODE}

def write_json(path, obj):
    with open(_pkg_open_path(_pkg_resolve(path), 'w'), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path, rows):
    with open(_pkg_open_path(_pkg_resolve(path), 'w'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')

def topn(S, k):
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    rows = np.arange(S.shape[0])[:, None]
    order = np.argsort(-S[rows, part], axis=1, kind='stable')
    return part[rows, order]

class Retriever:

    def __init__(self):
        index_items = load_jsonl(INDEX)
        docs = build_corpus(index_items)
        self.docs_by_key = {para_key(d): d for d in docs}
        self.doc_keys = [para_key(d) for d in docs]
        self.item_doc_keys = [[para_key(p) for p in split_paragraphs(r.get('context', ''))] for r in index_items]
        self.E_q = np.load(find_cache('qidx')).astype('float32')
        self.E_d = np.load(find_cache('docs_index')).astype('float32')
        if self.E_q.shape[0] != len(index_items) or self.E_d.shape[0] != len(docs):
            raise RuntimeError('임베딩 행 수가 인덱스·코퍼스와 다릅니다. 실행 중단.')
        self.corpus_sha = corpus_fingerprint(docs)
        from sentence_transformers import SentenceTransformer
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'
        self.model = SentenceTransformer(ENCODER, device=device)

    def run(self, pairs, batch=256):
        rows = []
        for s in range(0, len(pairs), batch):
            chunk = pairs[s:s + batch]
            Q = self.model.encode(['query: ' + q for (_, q) in chunk], convert_to_numpy=True, normalize_embeddings=True, batch_size=64).astype('float32')
            Sq = Q @ self.E_q.T
            Sd = Q @ self.E_d.T
            tq = topn(Sq, TOP_ITEMS)
            td = topn(Sd, TOP_DOCS)
            for (b, (qid, _)) in enumerate(chunk):
                rows.append(self._row(qid, Sq[b], tq[b], Sd[b], td[b]))
        return rows

    def _row(self, qid, sq, tq, sd, td):
        tq = [int(j) for j in tq]
        td = [int(j) for j in td]
        title = lambda k: self.docs_by_key[k]['title']
        (qi_keys, qi_short) = qi_policy(tq, self.item_doc_keys, P)
        (va_keys, va_short) = vanilla_policy([self.doc_keys[j] for j in td], P)
        return {'qid': str(qid), 'qi_titles': [title(k) for k in qi_keys], 'qi_short': bool(qi_short), 'qi_top_items': [{'item': j, 'score': float(sq[j]), 'titles': [title(k) for k in self.item_doc_keys[j]]} for j in tq], 'va_titles': [title(k) for k in va_keys], 'va_short': bool(va_short), 'va_top_docs': [{'title': title(self.doc_keys[j]), 'score': float(sd[j])} for j in td]}

def check_invariants(rows):
    viol = []
    for r in rows:
        for name in ('qi_titles', 'va_titles'):
            t = r[name]
            if len(t) > P:
                viol.append((r['qid'], name, 'I2 개수 초과'))
            if len(set(t)) != len(t):
                viol.append((r['qid'], name, 'I1 중복'))
            if any((x == '' for x in t)):
                viol.append((r['qid'], name, '빈 제목 반환'))
        for name in ('qi_top_items', 'va_top_docs'):
            s = [x['score'] for x in r[name]]
            if any((s[i] < s[i + 1] - 1e-09 for i in range(len(s) - 1))):
                viol.append((r['qid'], name, 'I6 점수 정렬 오류'))
        if r['qi_short'] != (len(r['qi_titles']) < P):
            viol.append((r['qid'], 'qi_short', '부족 표시 불일치'))
        if r['va_short'] != (len(r['va_titles']) < P):
            viol.append((r['qid'], 'va_short', '부족 표시 불일치'))
    return viol

def run_sample():
    evals = load_jsonl(EVAL)
    rng = np.random.default_rng(SAMPLE_SEED)
    pick = sorted(rng.choice(len(evals), SAMPLE_N, replace=False).tolist())
    pairs = [(str(evals[i]['qid']), evals[i]['original']) for i in pick]
    del evals
    R = Retriever()
    rows1 = R.run(pairs)
    rows2 = R.run(pairs)
    reproducible = json.dumps(rows1, sort_keys=True) == json.dumps(rows2, sort_keys=True)
    viol = check_invariants(rows1)
    os.makedirs(OUT_E1A, exist_ok=True)
    write_jsonl(SAMPLE_PRED, rows1)
    status = 'PASS' if not viol and reproducible and (len(rows1) == SAMPLE_N) else 'FAIL'
    write_json(SAMPLE_STATUS, {'status': status, 'n': len(rows1), 'invariant_violations': len(viol), 'reproducible': reproducible, 'code_sha256': code_hashes(), 'predictions_sha256': sha256_file(SAMPLE_PRED), 'corpus_sha256': R.corpus_sha, 'created_at': now()})
    print(f'샘플 검색 {len(rows1)}문항 (탐색 데이터, Gold 미사용)')
    print(f"불변식 위반 {len(viol)}건 / 재현성 {('일치' if reproducible else '불일치')}")
    for v in viol[:10]:
        print('   ', v)
    print(f'판정: {status}')
    print(f'저장: {SAMPLE_PRED}')
    print(f'저장: {SAMPLE_STATUS}')
    print('다음: e1a_score.py --sample 로 채점 배선 확인 (탐색용)')

def gate_holdout():
    problems = []
    if not os.path.exists(FREEZE):
        problems.append('FREEZE.json 없음 (동결 필요)')
    else:
        fz = json.load(open(_pkg_open_path(_pkg_resolve(FREEZE), 'r'), encoding='utf-8'))
        if not os.path.exists(SPLIT) or sha256_file(SPLIT) != fz['split_ids_sha256']:
            problems.append('분할 파일 해시가 동결 기록과 다름')
    if not os.path.exists(SAMPLE_STATUS):
        problems.append('sample_status.json 없음 (--sample 먼저 실행)')
    else:
        st = json.load(open(_pkg_open_path(_pkg_resolve(SAMPLE_STATUS), 'r'), encoding='utf-8'))
        if st.get('status') != 'PASS':
            problems.append('샘플 검증이 PASS가 아님')
        if st.get('code_sha256') != code_hashes():
            problems.append('샘플 검증 이후 검색 코드가 바뀜 (--sample 다시 실행)')
    if os.path.exists(HOLD_PRED):
        problems.append('retrieval_predictions.jsonl 이미 있음 (덮어쓰기 금지)')
    return problems

def load_holdout_pairs():
    split = json.load(open(_pkg_open_path(_pkg_resolve(SPLIT), 'r'), encoding='utf-8'))
    want = set((str(x) for x in split['holdout']))
    qmap = {}
    with open(_pkg_open_path(_pkg_resolve(DEV), 'r'), encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            i = rid(r)
            if i in want:
                qmap[i] = r['question']
    if len(want) != HOLDOUT_N or len(qmap) != HOLDOUT_N:
        raise RuntimeError(f'Holdout 개수 불일치: 기대 {HOLDOUT_N}, 분할 {len(want)}, 찾음 {len(qmap)}')
    return [(i, qmap[i]) for i in sorted(qmap)]

def run_holdout():
    problems = gate_holdout()
    if problems:
        print('[실행 안 함] 게이트 실패:')
        for p in problems:
            print('  -', p)
        sys.exit(1)
    pairs = load_holdout_pairs()
    R = Retriever()
    rows = R.run(pairs)
    viol = check_invariants(rows)
    if viol:
        print(f'[저장 안 함] 명세 위반 {len(viol)}건. 원인을 확인하고 명세 8번 기준으로만 수정하십시오.')
        for v in viol[:10]:
            print('   ', v)
        sys.exit(1)
    write_jsonl(HOLD_PRED, rows)
    fz = json.load(open(_pkg_open_path(_pkg_resolve(FREEZE), 'r'), encoding='utf-8'))
    man = {'created_at': now(), 'n': len(rows), 'predictions_file': os.path.basename(HOLD_PRED), 'predictions_sha256': sha256_file(HOLD_PRED), 'split_ids_sha256': fz['split_ids_sha256'], 'corpus_sha256': R.corpus_sha, 'code_sha256': code_hashes(), 'sample_status_sha256': sha256_file(SAMPLE_STATUS), 'gold_used_in_retrieval': False, 'policy': {'P': P, 'top_items': TOP_ITEMS, 'top_docs': TOP_DOCS}}
    write_json(HOLD_MAN, man)
    print(f'Holdout 검색 완료: {len(rows)}문항 (Gold 미사용)')
    print(f"예측 파일 해시 {man['predictions_sha256'][:16]}...")
    print(f'저장: {HOLD_PRED}')
    print(f'저장: {HOLD_MAN}')
    print('다음: e1a_score.py --holdout 로 채점. 이 단계 이후 검색 결과를 바꾸지 않습니다.')

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--sample', action='store_true')
    g.add_argument('--holdout', action='store_true')
    args = ap.parse_args()
    if args.sample:
        run_sample()
    else:
        run_holdout()
if __name__ == '__main__':
    main()
