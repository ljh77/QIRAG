"""
qi_common.py  - 여러 실행 파일이 함께 쓰는 공통 함수 (읽기 전용 유틸리티)

  split_paragraphs / build_corpus : 01_run_generate_v2.py 와 같은 규칙으로 코퍼스 재구성
  content_present                 : E0-03 과 같은 내용 기준 Gold 존재 판정
  corpus_fingerprint              : 코퍼스 본문 해시
  find_cache                      : out/cache 아래 임베딩 파일 찾기
"""
import pathlib as _pkg_pathlib
from S00_package_paths import output_path as _pkg_output, open_path as _pkg_open_path
from S00_package_paths import PACKAGE_ROOT as _PACKAGE_ROOT, resolve as _pkg_resolve, listdir as _pkg_listdir, isdir as _pkg_isdir
import glob
import hashlib
import json
import os
import re
from difflib import SequenceMatcher
ROOT = str(_PACKAGE_ROOT)
INDEX = _pkg_resolve(os.path.join(ROOT, 'data', '_split', 'index_items.jsonl'))
EVAL = _pkg_resolve(os.path.join(ROOT, 'data', '_split', 'eval_queries.jsonl'))
DEV = _pkg_resolve(os.path.join(ROOT, 'data', 'hotpotqa', 'hotpot_dev_distractor.jsonl'))
CACHE = _pkg_resolve(os.path.join(ROOT, 'out', 'cache'))
LOCKED = _pkg_resolve(os.path.join(ROOT, 'out', 'holdout_locked'))
MIN_CHARS = 20
CONTENT_MIN = 0.9
ENCODER = 'intfloat/e5-base-v2'
PARA_RX = re.compile('\\[(.*?)\\]\\s*(.*?)(?=\\s*\\[[^\\]]+\\]\\s|$)', re.S)
_PUNC = re.compile('[^\\w\\s]')

def norm(s):
    s = (s or '').lower()
    s = _PUNC.sub(' ', s)
    return ' '.join(s.split())

def load_jsonl(path):
    with open(_pkg_open_path(_pkg_resolve(path), 'r'), encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def sha256_file(path):
    h = hashlib.sha256()
    with open(_pkg_open_path(_pkg_resolve(path), 'rb'), 'rb') as f:
        for chunk in iter(lambda : f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def rid(r):
    return str(r.get('_id', r.get('id')))

def split_paragraphs(ctx):
    """01_run_generate_v2.py 의 split_paragraphs 와 같은 규칙."""
    ctx = (ctx or '').strip()
    out = []
    for (title, body) in PARA_RX.findall(ctx):
        (title, body) = (title.strip(), body.strip())
        if len(body) >= MIN_CHARS:
            out.append({'title': title, 'text': body})
    if not out and len(ctx) >= MIN_CHARS:
        out.append({'title': '', 'text': ctx})
    return out

def para_key(p):
    return p['title'] or p['text'][:200]

def build_corpus(index_items):
    """01 의 build_corpus_index 와 같은 순서·중복 규칙으로 코퍼스를 만든다."""
    (docs, seen) = ([], set())
    for r in index_items:
        for p in split_paragraphs(r.get('context', '')):
            k = para_key(p)
            if k not in seen:
                seen.add(k)
                docs.append(p)
    return docs

def corpus_fingerprint(docs):
    h = hashlib.sha256()
    for d in docs:
        h.update((d['title'] + '\t' + d['text'] + '\n').encode('utf-8'))
    return h.hexdigest()

def index_passages_content(index_items):
    """제목 -> 정규화 본문 집합 (길이 필터 적용, E0-03 과 같은 기준)."""
    passages = {}
    for r in index_items:
        for (t, b) in PARA_RX.findall(r.get('context', '')):
            if len(b.strip()) < MIN_CHARS:
                continue
            passages.setdefault(t.strip(), set()).add(norm(b))
    return passages

def content_present(title, body_norm, passages):
    if title not in passages or not body_norm:
        return False
    return any((body_norm == b or SequenceMatcher(None, body_norm, b).ratio() >= CONTENT_MIN for b in passages[title]))

def find_cache(prefix):
    hits = sorted(glob.glob(_pkg_resolve(os.path.join(CACHE, f'{prefix}_*.npy'))))
    if not hits:
        raise FileNotFoundError(f'임베딩 캐시 없음: out/cache/{prefix}_*.npy')
    return hits[-1]
