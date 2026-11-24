"""책 전체(모든 권)에서 겹치는 \\label을 찾는다. 권을 따로 빌드하면 권 사이의 중복은 LaTeX가 잡지 못한다.
쓰기: python3 tools/check_labels.py"""
import re, glob, collections, sys
seen = collections.defaultdict(list)
for f in sorted(glob.glob('chapters/*.tex')):
    for m in re.finditer(r'\\label\{([^}]*)\}', open(f, encoding='utf-8').read()):
        seen[m.group(1)].append(f)
dup = {k: v for k, v in seen.items() if len(v) > 1}
for k, v in dup.items():
    print(k, v)
print(f'labels {len(seen)}, duplicated {len(dup)}')
sys.exit(1 if dup else 0)
