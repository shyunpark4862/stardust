"""humanize-korean 조각 결과를 합치고(요약 주석 제거) LaTeX 구조가 원문과 같은지 대조한다.
사용: python3 tools/hk_merge.py _workspace/<run_id>"""
import collections, glob, json, os, re, sys
run = sys.argv[1]
m = json.load(open(os.path.join(run, 'chunk_manifest.json')))
parts = []
for c in m['chunks']:
    t = open(os.path.join(run, c['rewritten_file'])).read()
    parts.append(re.sub(r'<!-- HUMANIZE-SUMMARY.*?-->\s*', '', t, flags=re.S))
out = ''.join(parts)
open(os.path.join(run, 'final.md'), 'w').write(out)
def segs(t):
    c = collections.Counter()
    for x in re.finditer(r'\$[^$]+\$|\\\[.*?\\\]', t, re.S): c['M:' + x.group()] += 1
    for x in re.finditer(r'\\[A-Za-z@]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', t): c['C:' + x.group()] += 1
    for x in re.finditer(r'\\begin\{(algorithmic|sdtable|tikzpicture|proof|theorem|lemma|proposition|corollary|definition)\}.*?\\end\{\1\}', t, re.S): c['B:' + x.group()] += 1
    return c
a = open(os.path.join(run, '01_input.txt')).read()
A, B = segs(a), segs(out)
print('chars', len(a), len(out))
print('missing', [k[:100] for k in (A - B)][:10])
print('added', [k[:100] for k in (B - A)][:10])
