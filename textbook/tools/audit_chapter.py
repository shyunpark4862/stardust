"""장 원고를 CHECKLIST.md의 기계로 셀 수 있는 항목으로 점검한다.

쓰기: python3 tools/audit_chapter.py chapters/…tex [더…]
실패 항목이 있으면 종료 코드 1. 사람이 판단할 항목(증명의 충실함, 세 축의 균형)은
수치로만 보여 주고, 판단은 검토 에이전트와 내가 한다.
"""
import re
import sys

HANGUL = re.compile(r'[가-힣]')


def strip_tex(s):
    s = re.sub(r'\$[^$]*\$', 'X', s)
    s = re.sub(r'\\[a-zA-Z@]+\*?(\[[^\]]*\])?', '', s)
    return re.sub(r'[{}]', '', s)


def braced(t, start):
    """t[start]가 '{'일 때 짝이 맞는 '}'까지의 내용."""
    depth = 0
    for i in range(start, len(t)):
        if t[i] == '{':
            depth += 1
        elif t[i] == '}':
            depth -= 1
            if depth == 0:
                return t[start + 1:i]
    return t[start + 1:]


def audit(path):
    t = open(path, encoding='utf-8').read()
    body = re.sub(r'(?m)^%.*$', '', t)
    fails, notes = [], []

    # 개요: 5줄 이내 (본문 폭 한 줄에 한글 약 45자로 어림)
    i = body.find('\\abstract{')
    if i < 0:
        fails.append('개요(\\abstract)가 없다')
    else:
        n = len(strip_tex(braced(body, i + len('\\abstract'))))
        lines = n / 45
        (fails if lines > 5.2 else notes).append(f'개요 약 {lines:.1f}줄 ({n}자)')

    # 제목은 영어
    for m in re.finditer(r'\\(chapter|section|subsection|subsubsection)\*?\{', body):
        title = braced(body, m.end() - 1)
        if HANGUL.search(title):
            fails.append(f'한글 제목: \\{m.group(1)}{{{title}}}')

    # 인라인 \frac
    for m in re.finditer(r'(?<!\\)\$([^$]+)\$', body):
        if re.search(r'\\[dt]?frac', m.group(1)):
            fails.append(f'인라인 \\frac: ${m.group(1)[:50]}$')

    # \term의 영어
    for m in re.finditer(r'\\term\{([^}]*)\}\{([^}]*)\}', body):
        if not m.group(2).strip():
            fails.append(f'\\term{{{m.group(1)}}}의 영어가 비었다')

    # 인라인 행렬 (별행으로)
    for m in re.finditer(r'(?<![\\$])\$([^$]+)\$', body):
        if re.search(r'\\begin\{[pbvBV]?matrix\}', m.group(1)):
            fails.append(f'인라인 행렬: ${m.group(1)[:50]}$')

    # 본문이 없는 절 (표·그림·알고리즘만)
    for m in re.finditer(r'\\section\{([^}]*)\}(.*?)(?=\\section\{|\\chapterreferences|\Z)', body, re.S):
        seg = re.sub(r'\\begin\{(figure|table|algorithm)\*?\}.*?\\end\{\1\*?\}', '', m.group(2), flags=re.S)
        seg = re.sub(r'\\(label|subsection)\{[^}]*\}', '', seg)
        if len(HANGUL.findall(seg)) < 30:
            fails.append(f'본문이 없는 절: {m.group(1)}')

    # 절과 그림·표·알고리즘
    sections = [m.start() for m in re.finditer(r'\\section\{', body)] + [len(body)]
    titles = [braced(body, m.end() - 1) for m in re.finditer(r'\\section\{', body)]
    figs = len(re.findall(r'\\begin\{figure\*?\}', body))
    tabs = len(re.findall(r'\\begin\{table\*?\}', body))
    algs = len(re.findall(r'\\begin\{algorithm\}', body))
    proofs = len(re.findall(r'\\begin\{proof\}', body))
    notes.append(f'절 {len(titles)}, 그림 {figs}, 표 {tabs}, 알고리즘 {algs}, 증명 {proofs}, 글자 {len(body)}')
    if figs < max(3, (len(titles) - 2) // 2):
        fails.append(f'그림이 {figs}개뿐이다(절 {len(titles)}개)')
    if algs == 0:
        fails.append('알고리즘(유사코드)이 없다')
    bare = []
    for k, title in enumerate(titles):
        seg = body[sections[k]:sections[k + 1]]
        if title in ('Concluding Remarks',) or title.startswith('APIs in'):
            continue
        if not re.search(r'\\begin\{(figure|table|algorithm)\*?\}', seg):
            bare.append(title)
    if bare:
        notes.append('그림·표·알고리즘이 없는 절: ' + '; '.join(bare))

    # 세 축의 분량(어림): 알고리즘 환경과 Implementation 절의 글자 수
    alg_chars = sum(len(m.group(0)) for m in re.finditer(r'\\begin\{algorithm\}.*?\\end\{algorithm\}', body, re.S))
    notes.append(f'유사코드 글자 {alg_chars} ({alg_chars / max(len(body), 1):.0%})')

    # instardust 상자: 짧게
    for m in re.finditer(r'\\begin\{instardust\}(.*?)\\end\{instardust\}', body, re.S):
        n = len(strip_tex(m.group(1)))
        if n > 700:
            fails.append(f'instardust 상자가 길다({n}자): 구현 설명은 본문으로')

    # 장 끝 API 표
    if not re.search(r'\\section\{APIs in This Chapter\}', body):
        fails.append('장 끝 API 표(\\section{APIs in This Chapter})가 없다')

    # 인용 없이 넘긴 흔적: "증명은 …를 보라" 꼴
    for pat in (r'증명은[^.]{0,40}(보라|참고|참조)', r'자세한 것은[^.]{0,40}(보라|참고|참조)'):
        for m in re.finditer(pat, body):
            notes.append(f'증명·설명을 인용으로 넘긴 듯한 곳: "{m.group(0)[:60]}"')

    # 증명을 싣지 않는다는 변명 (GUIDE §5: 줄거리와 참고할 곳을 쓴다)
    for m in re.finditer(r'[^.]{0,40}(증명은[^.]{0,20}싣지 않|싣지 않는다|싣지 않았다|범위 밖이다|증명을 생략)[^.]{0,20}', body):
        fails.append(f'증명·설명을 빼는 변명: "{m.group(0).strip()[:70]}" → 핵심 아이디어·줄거리·참고할 곳으로')
    print(f'== {path}')
    for n in notes:
        print('  ·', n)
    for f in fails:
        print('  ✗', f)
    print('  ' + ('통과' if not fails else f'실패 {len(fails)}'))
    return not fails


if __name__ == '__main__':
    ok = all([audit(p) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
