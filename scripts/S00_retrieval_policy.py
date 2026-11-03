"""
retrieval_policy.py  (2단계: 고유 문서 P=2 정책과 Recall@2 채점)
================================================================
검색 정책 함수는 정답을 입력으로 받지 않는다. 채점 함수만 Gold 제목을 받는다.

  unique_take      순위 목록을 앞에서부터 훑어 서로 다른 키를 P개까지 취함
  qi_policy        QI-RAG: 상위 질문들의 연결 문서를 검색 순서대로 순회 -> 고유 P개
  vanilla_policy   Vanilla: 상위 문서 목록 -> 고유 P개
  recall_at_p      Gold Document Recall@P (채점 전용)

반환 부족(후보 소진으로 P개를 못 채움)은 임의로 채우지 않고 short=True 로 표시한다.

테스트: python retrieval_policy.py --test
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import sys

def unique_take(seq, P):
    (out, seen) = ([], set())
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) == P:
            break
    return (out, len(out) < P)

def qi_policy(top_item_ids, item_doc_keys, P=2):
    """top_item_ids: 질문 인덱스 상위 순위.
    item_doc_keys[j]: 질문 j 가 연결한 문서 키 목록 (원래 순서 유지).
    순위대로 연결 문서를 펼친 뒤 고유 P개를 취한다. 질문 개수와 무관하게 문서 P개를 맞춘다."""
    seq = []
    for j in top_item_ids:
        seq.extend(item_doc_keys[j])
    return unique_take(seq, P)

def vanilla_policy(top_doc_keys, P=2):
    return unique_take(top_doc_keys, P)

def recall_at_p(returned_titles, gold_titles):
    """Gold 문서 제목 중 반환된 것의 비율. Gold 가 비어 있으면 None."""
    gold = set((t for t in gold_titles if t))
    if not gold:
        return None
    return _pkg_resolve(len(gold & set((t for t in returned_titles if t))) / len(gold))

def _tests():
    fails = []

    def check(name, cond):
        print(('PASS ' if cond else 'FAIL ') + name)
        if not cond:
            fails.append(name)
    (out, short) = unique_take(['A', 'A', 'B', 'C'], 2)
    check('unique_take: 중복 제거 후 2개', out == ['A', 'B'] and (not short))
    (out, short) = unique_take(['A'], 2)
    check('unique_take: 후보 부족은 short 표시', out == ['A'] and short)
    items = [['A', 'B'], ['C', 'D']]
    (out, short) = qi_policy([0, 1], items, 2)
    check('qi_policy: 첫 질문의 문서 2개로 충족', out == ['A', 'B'] and (not short))
    items = [['A'], ['A', 'B']]
    (out, short) = qi_policy([0, 1], items, 2)
    check('qi_policy: 같은 문서 중복은 건너뛰고 다음 문서 채움', out == ['A', 'B'] and (not short))
    items = [['A']]
    (out, short) = qi_policy([0], items, 2)
    check('qi_policy: 연결 문서가 부족하면 short 표시(임의 보충 없음)', out == ['A'] and short)
    (out, short) = vanilla_policy(['A', 'A', 'B'], 2)
    check('vanilla_policy: 같은 규칙으로 고유 2개', out == ['A', 'B'] and (not short))
    check('recall: 두 Gold 중 1개 회수 = 0.5', recall_at_p(['A', 'C'], ['A', 'B']) == 0.5)
    check('recall: 두 Gold 모두 회수 = 1.0', recall_at_p(['A', 'B'], ['A', 'B']) == 1.0)
    check('recall: Gold 없음 = None', recall_at_p(['A'], []) is None)
    check('recall: 빈 제목은 매칭하지 않음', recall_at_p(['', ''], ['A', 'B']) == 0.0)
    print(f"\n결과: {('전부 통과' if not fails else f'실패 {len(fails)}건')}")
    return 0 if not fails else 1
if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.exit(_tests())
    print('사용법: python retrieval_policy.py --test')
