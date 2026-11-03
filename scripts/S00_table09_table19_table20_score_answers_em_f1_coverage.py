"""
02_score_report_v2.py  (qi-rag-test)
====================================
01_run_generate_v2.py 가 만든 out/<모드>/*.jsonl 을 채점하고,
사전 등록한 주장별로 지지 / 반증 / 불확실을 판정한다.

두 집합에서 판정한다
  [주] 정답 문단 전부 코퍼스에 존재하는 문항 (gold_full = True)
       상한(ORACLE)과 비교가 공정한 집합. 판정은 여기서 한다.
  [참고] 전체 공통 문항
       코퍼스에 없는 정답 문단이 있는 문항까지 포함. 방향만 본다.

지표
  주 지표 F1 (보조 EM, contains)
  유보율 = 유보/전체, 오답률 = 오답/전체, 선택위험 = 오답/답변
  근거 내 정답 = contains 로 맞고 정답 문자열이 제시 문맥 안에 있는 정답 비율
  달성률 = 조건 F1 / ORACLE F1 (같은 집합에서 계산)

판정 규칙 (사전 등록)
  비교 쌍 (A -> B, 차이 = B - A, B 는 항상 QI-RAG)
    지지   : F1 차이 95% 부트스트랩 CI 하한 > 0  AND  Holm 보정 McNemar(EM) p < 0.05
    반증   : F1 차이 95% CI 상한 < 0  AND  Holm 보정 p < 0.05
    불확실 : 그 밖의 모든 경우
  Holm 보정은 같은 집합 안의 주장들끼리 적용한다.
  주장 목록 CLAIMS 는 전체 실행 전에 고정한다.

층화
  정답 문서 개수(0, 1, 2)별 F1. 같은 개수에서도 우위가 남는지 본다.

실행 (qi-rag-test 폴더에서)
  python 02_score_report_v2.py
  메뉴: 1) 샘플 결과 보고  2) 전체 결과 보고  3) 종료
산출물: out/<모드>/report_v2.txt
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import ast
import json
import math
import os
import re
from collections import Counter
import numpy as np
OUT_ROOT = 'out'
VARIANTS = ['original', 'keyword', 'noisy', 'paraphrase_llm']
BUDGETS = [2, 4]
QAGEN_K = 2
N_BOOT = 2000
SEED = 0
ALPHA = 0.05
MAIN_VARIANT = 'original'
CLAIMS = []
for _P in BUDGETS:
    CLAIMS.append({'id': f'A{_P}', 'A': f'VANILLA_P{_P}', 'B': f'QIRAG_P{_P}', 'text': f'같은 예산 P={_P} 에서 vanilla 대비 QI-RAG (문서공간 -> 질의공간)'})
    CLAIMS.append({'id': f'B{_P}', 'A': f'RANDOM_P{_P}', 'B': f'QIRAG_P{_P}', 'text': f'같은 문단 수 무작위 문맥 대비 QI-RAG (1:N 구조 효과)'})
    CLAIMS.append({'id': f'D{_P}', 'A': 'REPAQ_TOP1', 'B': f'QIRAG_P{_P}', 'text': 'RePAQ top-1 대비 QI-RAG (검색기가 달라 참고용)'})
CLAIMS.append({'id': 'C2', 'A': 'E5_TOP1', 'B': 'QIRAG_P2', 'text': '같은 E5 검색기에서 답만 반환 대비 문맥 생성 (페이로드 효과)'})
CLAIMS.append({'id': 'E2', 'A': f'QAGEN_K{QAGEN_K}', 'B': 'QIRAG_P2', 'text': '저장 답 페이로드 대비 문단 페이로드 (페이로드 효과)'})
LAYER_PREFIX = ('QIRAG', 'VANILLA', 'RANDOM', 'ORACLE')
_PUNC = re.compile('[^\\w\\s]')
_ART = re.compile('\\b(a|an|the)\\b')
ABSTAIN = ("I DON'T KNOW", 'I DONT KNOW', 'I DO NOT KNOW', 'INSUFFICIENT')

def load_jsonl(path):
    out = []
    with open(_pkg_open_path(_pkg_resolve(path), 'r'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def as_list(a):
    if a is None:
        return []
    if isinstance(a, list):
        return [str(x) for x in a]
    return [str(a)]

def norm(s):
    s = (s or '').lower()
    s = _PUNC.sub(' ', s)
    s = _ART.sub(' ', s)
    return ' '.join(s.split())

def parse_pred(p):
    """답이 "['Japan']" 같은 리스트 문자열이면 원소를 풀어서 돌려준다."""
    s = (p or '').strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return ' '.join((str(x) for x in v))
        except Exception:
            pass
    return s

def is_abstain(t):
    if not t or not str(t).strip():
        return True
    up = str(t).upper()
    return any((a in up for a in ABSTAIN))

def f1(pred, gold):
    pt = norm(pred).split()
    gt = norm(gold).split()
    if not pt or not gt:
        return 0.0
    c = Counter(pt) & Counter(gt)
    n = sum(c.values())
    if n == 0:
        return 0.0
    (prec, rec) = (_pkg_resolve(n / len(pt)), _pkg_resolve(n / len(gt)))
    return _pkg_resolve(2 * prec * rec / (prec + rec))

def score_row(r):
    gold = [g for g in as_list(r.get('gold')) if norm(g)]
    pred = parse_pred(r.get('pred', ''))
    ab = is_abstain(pred)
    em_v = 0.0 if ab else float(any((norm(pred) == norm(g) for g in gold)))
    ct_v = 0.0 if ab else float(any((norm(g) in norm(pred) for g in gold)))
    f1_v = 0.0 if ab else max((f1(pred, g) for g in gold), default=0.0)
    ctx = r.get('ctx') or ''
    if ctx:
        ctx_n = norm(ctx)
        grounded = 1.0 if ct_v == 1.0 and any((norm(g) in ctx_n for g in gold)) else 0.0
    else:
        grounded = None
    gt = [str(t) for t in r.get('gold_titles', [])]
    rt = set((str(t) for t in r.get('retrieved_titles', [])))
    k_gold = sum((1 for t in gt if t in rt)) if gt else None
    return {'ab': 1.0 if ab else 0.0, 'EM': em_v, 'contains': ct_v, 'F1': f1_v, 'grounded': grounded, 'k_gold': k_gold, 'words': float(r.get('words', 0)), 'paras': float(r.get('n_paras', 0)), 'hit': 1.0 if r.get('hit_all') else 0.0}

def mcnemar_exact(n01, n10):
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = _pkg_resolve(sum((math.comb(n, i) for i in range(k + 1))) / 2 ** n)
    return min(1.0, 2.0 * tail)

def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for (rank, i) in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj

def paired(a_f1, b_f1, a_em, b_em, seed=SEED):
    """차이 = B - A. F1 차이의 부트스트랩 CI, EM 기준 정확 McNemar."""
    diff = b_f1 - a_f1
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (N_BOOT, n))
    boot = diff[idx].mean(axis=1)
    (lo, hi) = np.percentile(boot, [2.5, 97.5])
    a_bin = (a_em >= 1).astype(int)
    b_bin = (b_em >= 1).astype(int)
    n01 = int(((a_bin == 0) & (b_bin == 1)).sum())
    n10 = int(((a_bin == 1) & (b_bin == 0)).sum())
    return {'a': float(a_f1.mean()), 'b': float(b_f1.mean()), 'diff': float(diff.mean()), 'lo': float(lo), 'hi': float(hi), 'n01': n01, 'n10': n10, 'p': mcnemar_exact(n01, n10), 'n': n}

def verdict(lo, hi, p_adj):
    if p_adj is None:
        return '데이터 없음'
    if p_adj < ALPHA and lo > 0:
        return '지지 (QI 우위)'
    if p_adj < ALPHA and hi < 0:
        return '반증 (QI 열세)'
    return '불확실'

def fmt(x, spec, na=False):
    if na or x is None or (isinstance(x, float) and math.isnan(x)):
        return '-'
    return format(x, spec)

def stats(arr):
    n = len(arr)
    if n == 0:
        return None
    ab = np.array([s['ab'] for s in arr])
    ans = 1.0 - ab
    answered = ans.sum()
    em_ = np.array([s['EM'] for s in arr])
    wrong = float(((ans == 1) & (em_ == 0)).sum())
    grd = [s['grounded'] for s in arr if s['grounded'] is not None]
    return {'n': n, 'abstain': float(ab.mean()), 'F1': float(np.mean([s['F1'] for s in arr])), 'EM': float(em_.mean()), 'contains': float(np.mean([s['contains'] for s in arr])), 'wrong': _pkg_resolve(wrong / n), 'risk': _pkg_resolve(wrong / answered) if answered else float('nan'), 'grounded': float(np.mean(grd)) if grd else None, 'words': float(np.mean([s['words'] for s in arr])), 'paras': float(np.mean([s['paras'] for s in arr])), 'hit': float(np.mean([s['hit'] for s in arr]))}

def P_of(name):
    if '_P' in name:
        try:
            return int(name.rsplit('_P', 1)[1])
        except ValueError:
            return None
    return None

def block(v, cmap, S, qs, label, L):
    """한 집합(qs)에 대해 표, 층화, 주장 판정을 출력하고 판정 결과를 돌려준다."""
    L(f'\n  ── 집합: {label}  (n = {len(qs)}) ──')
    if not qs:
        L('  문항이 없습니다.')
        return []
    arr = {c: [S[c][q] for q in qs] for c in cmap}
    st = {c: stats(arr[c]) for c in cmap}
    oracle = st.get('ORACLE')
    L(f"  {'조건':<18}{'유보':>7}{'F1':>7}{'EM':>7}{'ct':>7}{'오답률':>7}{'선택위험':>9}{'근거내':>7}{'단어':>7}{'문단/답':>8}{'적중':>7}{'달성률':>8}")
    has_qa = False
    for c in cmap:
        s = st[c]
        if s is None:
            continue
        is_rep = c in ('REPAQ_TOP1', 'E5_TOP1')
        is_qa = c.startswith('QAGEN')
        has_qa = has_qa or is_qa
        name = c + ('*' if is_qa else '') + ('-' if is_rep else '')
        ach = None
        if oracle and oracle['F1'] > 0 and (not is_rep) and (not is_qa):
            ach = _pkg_resolve(s['F1'] / oracle['F1'])
        paras_txt = f"{s['paras']:.0f}답" if is_qa else fmt(s['paras'], '.2f', na=is_rep)
        hit_txt = '*' if is_qa else fmt(s['hit'], '.3f', na=is_rep)
        L(f"  {name:<18}{s['abstain']:>7.3f}{s['F1']:>7.3f}{s['EM']:>7.3f}{s['contains']:>7.3f}{s['wrong']:>7.3f}{fmt(s['risk'], '.3f'):>9}{fmt(s['grounded'], '.3f', na=is_rep or is_qa or s['grounded'] is None):>7}{fmt(s['words'], '.1f', na=is_rep):>7}{paras_txt:>8}{hit_txt:>7}{fmt(ach, '.3f', na=ach is None):>8}")
    if has_qa:
        L('  * QAGEN: 문단이 아니라 저장 답 개수. 적중은 항목 gold 제목 기준이라 문단 조건과 직접 비교 안 함')
    L("  - REPAQ_TOP1, E5_TOP1: 생성하지 않고 답만 반환. 문맥 관련 칸은 '-'")
    L('\n  [층화] 회수한 정답 문서 개수별 F1')
    L(f"  {'조건':<16}{'0개':>14}{'1개':>14}{'2개':>14}")
    for c in cmap:
        if not c.startswith(LAYER_PREFIX):
            continue
        cells = []
        for k in (0, 1, 2):
            grp = [S[c][q]['F1'] for q in qs if S[c][q]['k_gold'] is not None and min(S[c][q]['k_gold'], 2) == k]
            cells.append(f'{np.mean(grp):.3f}({len(grp)})' if grp else '-')
        L(f'  {c:<16}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}')
    L('\n  [사전 등록 주장 판정]  차이 = B - A, 지지 조건: CI 하한 > 0 AND Holm p < 0.05')
    results = []
    for cl in CLAIMS:
        if cl['A'] not in cmap or cl['B'] not in cmap:
            continue
        a = np.array([S[cl['A']][q]['F1'] for q in qs])
        b = np.array([S[cl['B']][q]['F1'] for q in qs])
        ae = np.array([S[cl['A']][q]['EM'] for q in qs])
        be = np.array([S[cl['B']][q]['EM'] for q in qs])
        r = paired(a, b, ae, be)
        r.update({'id': cl['id'], 'A': cl['A'], 'B': cl['B'], 'text': cl['text']})
        results.append(r)
    adj = holm([r['p'] for r in results])
    for (r, pa) in zip(results, adj):
        r['p_adj'] = pa
        r['verdict'] = verdict(r['lo'], r['hi'], pa)
        if r['id'].startswith('D'):
            r['verdict'] += ' (검색기 상이, 참고)'
        L(f"  {r['id']}: {r['text']}")
        L(f"     {r['A']} {r['a']:.3f} -> {r['B']} {r['b']:.3f}   차이 {r['diff']:+.4f}  95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]  n={r['n']}")
        L(f"     McNemar(EM) B만 {r['n01']} / A만 {r['n10']}  p={r['p']:.2e}  Holm p={pa:.2e}  -> {r['verdict']}")
    return results

def analyze(v, cmap, L):
    common = None
    for (c, m) in cmap.items():
        ok = {q for (q, r) in m.items() if r.get('status') == 'ok'}
        common = ok if common is None else common & ok
    common = sorted(common or set())
    L(f'\n[{v}]  공통 문항 {len(common)}  (모든 조건이 정상 응답한 문항만 비교)')
    for (c, m) in cmap.items():
        L(f'    {c:<16} 전체 행 {len(m)}, 공통에서 제외 {len(m) - len(common)}')
    S = {c: {q: score_row(cmap[c][q]) for q in common} for c in cmap}
    if 'ORACLE' in cmap:
        subset = [q for q in common if bool(cmap['ORACLE'][q].get('gold_full'))]
        L(f'    정답 문단 전부 코퍼스에 있는 문항: {len(subset)}/{len(common)} ({_pkg_resolve(len(subset) / max(len(common), 1)):.3f})')
    else:
        subset = []
        L('    ORACLE 결과가 없어 부분집합 판정을 건너뜁니다.')
    res_sub = block(v, cmap, S, subset, '정답 문단 전부 코퍼스 존재 (판정 기준)', L)
    res_all = block(v, cmap, S, common, '전체 공통 문항 (참고)', L)
    return (res_sub, res_all)

def load_mode(mode):
    d = _pkg_resolve(os.path.join(OUT_ROOT, mode))
    by_var = {}
    for fn in sorted(_pkg_listdir(d)):
        if not fn.endswith('.jsonl'):
            continue
        base = fn[:-len('.jsonl')]
        for v in VARIANTS:
            if base.endswith('_' + v):
                cond = base[:-len(v) - 1]
                rows = {str(r['qid']): r for r in load_jsonl(_pkg_resolve(os.path.join(d, fn)))}
                by_var.setdefault(v, {})[cond] = rows
                break
    return by_var

def run_report(mode):
    d = _pkg_resolve(os.path.join(OUT_ROOT, mode))
    if not _pkg_isdir(d):
        print(f'  {d} 폴더가 없습니다. 01_run_generate_v2.py 를 먼저 실행하세요.')
        return
    by_var = load_mode(mode)
    lines = []

    def L(s=''):
        print(s)
        lines.append(s)
    L(f'==== 보고서 v2: {mode} ====')
    L('  지표: 유보 = 유보/전체, 오답률 = 오답/전체, 선택위험 = 오답/답변, 근거내 = 문맥 안 정답 비율')
    L(f'  판정 기준 집합 = 정답 문단 전부 코퍼스 존재. 주 변형 = {MAIN_VARIANT}')
    if not by_var:
        L('  결과 파일이 없습니다.')
    main_sub = []
    for v in VARIANTS:
        if v not in by_var:
            continue
        (res_sub, _) = analyze(v, by_var[v], L)
        if v == MAIN_VARIANT:
            main_sub = res_sub
    if main_sub:
        L('\n  [요약] 주 변형, 판정 기준 집합')
        for r in main_sub:
            L(f"    {r['id']:<4} {r['verdict']:<22} {r['A']} -> {r['B']}  (차이 {r['diff']:+.4f})")
    L('\n  해석 메모')
    L('  - 판정은 정답 문단 전부 코퍼스 존재 집합에서만 한다. 나머지는 방향 참고용이다.')
    L('  - 달성률은 같은 집합의 ORACLE F1 로 나눈 값이다. 1 에 가까울수록 정답 문맥 활용이 상한에 가깝다.')
    L('  - 층화에서 같은 개수일 때 우위가 사라지면, 우위는 정답 문서 수 때문이다.')
    L('  - 근거내 정답률은 문맥 밖 정답을 뺀다. 문맥적 정답을 보는 지표다.')
    L('  - 주장 목록과 판정 규칙은 전체 실행 전에 고정했다. 결과를 보고 바꾸지 않는다.')
    fp = _pkg_resolve(os.path.join(d, 'report_v2.txt'))
    with open(_pkg_open_path(_pkg_resolve(fp), 'w'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  보고서 저장: {fp}')

def main():
    print('qi-rag-test 채점·판정 (v2)')
    while True:
        print('\n  1) 샘플 결과 보고')
        print('  2) 전체 결과 보고')
        print('  3) 종료')
        c = input('  선택: ').strip()
        if c == '1':
            run_report('sample')
        elif c == '2':
            run_report('full')
        elif c == '3':
            break
if __name__ == '__main__':
    main()
