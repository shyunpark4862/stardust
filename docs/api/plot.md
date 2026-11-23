# sd.plot API

> 상태: **확정** (PL0–PL9, 2026-11-23). 아래의 모든 이름이 공개되어 있다(`sd.plot.__all__`과 `Figure`의 메서드). 시그니처는 `tests/plot/test_api_surface.py`가 코드와 견준다. 결정과 그 까닭은 [설계 문서](../design/plot.md)에 있다.

- 설계와 결정의 근거는 [docs/design/plot.md](../design/plot.md)(§1.1 PL1–PL22), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/plot/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 이름(`name(...)`, 클래스는 `name`), 클래스의 메서드(`Class.method(...)`)와 속성(`Class.attr`)이다.
- 길이는 mm, 글자 크기와 선 굵기는 pt, 각은 도(°)다.

## §1. 공통 규칙

### §1.1 호출 모양
- **모든 그림 함수는 `Figure`를 돌려준다**(PL6). 전역 상태(지금 그림)가 없다. 화면에 띄우는 `show()`가 없다. 노트북은 셀의 마지막 값인 `Figure`를 그려 보인다(§2.4).
- **자료와 주 인자는 위치로만, 옵션은 키워드로만** 받는다(API.md, DESIGN §8.1).
- **틀린 호출 모양**은 함수 이름으로 시작한다(API.md §2.1): `line(): unexpected keyword argument 'foo'`, `Figure(): takes 0 positional arguments but 2 were given`, `grid(): missing 1 required positional argument: 'figures'`. 모듈 함수와 `Figure(...)`가 그렇고, `Figure`의 메서드는 Python의 꼴 그대로다(`Figure.xlabel() missing ...`).
- **모듈 함수와 같은 이름의 `Figure` 메서드가 있다.** `fig.line(x, y)`는 `sd.plot.line(x, y)`와 같은 인자를 받고, 새 그림을 만드는 대신 `fig`의 칸에 층을 더한 뒤 `fig`를 돌려준다. 모듈 함수 `f(...)`는 `Figure(그림 옵션).f(...)`와 같다. 예외는 여러 칸을 스스로 만드는 함수(`grid`, `diagnostics`, `pairs`, `facet`)다. 이것들은 모듈 함수만 있다.
- **그림 옵션**: 새 그림을 만드는 모든 모듈 함수가 `width=`, `height=`, `font=`, `theme=`, `palette=`를 받는다(§2.1의 `Figure`와 같은 뜻과 기본값). 아래 시그니처에는 `**figure`로 줄여 적는다. 메서드는 받지 않는다. 자기 인자로 `width`를 가진 함수(선 굵기: `line step hline vline function parametric polar implicit ecdf curve3d`, 막대·상자 너비: `bar boxplot`)에서는 `width=`가 그 인자다. 이 함수들의 그림 너비는 `Figure(width=...)`로 시작하거나 `size()`로 정한다.
- **별칭이 없다.** matplotlib의 `plot`, `hist`, `imshow`, 짧은 색 이름 `'r'`, `lw=`·`ls=`·`c=` 같은 줄임 인자는 없다.

### §1.2 받는 자료
- 자료는 API.md §5의 상위 함수 입력이다: `sd.ndarray`, 수의 시퀀스, numpy 배열, frame의 `Series`. float64로 바꿔 쓴다. 범주 축의 자료는 문자열의 시퀀스나 string 열이다.
- **`data=`**: `DataFrame`을 주면 자료 인자가 열 이름이다: `sd.plot.scatter("height", "weight", data=df)`. `data=`가 없는데 문자열을 주면 `TypeError`다(범주 축은 문자열의 시퀀스다. 문자열 하나가 아니다).
- **`group=`**: 무리를 나누는 값(열 이름이거나 자료와 길이가 같은 시퀀스)이다. 무리마다 팔레트의 색을 받고 범례가 생긴다. `color=`는 색 하나이고 열 이름이 아니다. 둘을 함께 주면 `ValueError`다.
- 유한하지 않은 값: NaN인 점은 그리지 않고 선은 그 자리에서 끊긴다. 에러가 아니다. frame의 결측값은 NaN이다.

### §1.2a 표본의 조절: `samples`와 `refine`
- 함수를 받는 그림은 모두 표본을 알아서 고르고, 두 인자로 그것을 조절한다. 둘 다 `None`이면 아래의 기본값이다.
- **`samples`**: 첫 격자의 점의 수(한 방향)다. 두 변수의 그림은 수 하나(두 방향 같게)거나 `(가로, 세로)` 쌍이다. 3 이상이고, 곡선은 20만, 격자는 한 방향 4000(곡면은 2000)까지다.
- **`refine`**: 첫 격자의 한 조각을 더 쪼개는 횟수의 한계다. `0`이면 적응형 표본을 끄고 첫 격자만 쓴다(극점에서 선을 끊는 것도 하지 않는다).

| 그림 | `samples`의 기본 | `refine`의 기본(최대) | 한 번 쪼개면 |
|---|---|---|---|
| `function` `parametric` `polar` `curve3d` | 257 | 10 | 구간을 반으로 |
| `surface` `parametric_surface` | 121 × 121 | 2 | 칸을 4 × 4로 |
| `implicit` `region` `contour` | 321 × 241 | 1 | 칸을 4 × 4로 |
| `density` `phase` | 480 × 360 (화소) | 없음 | |

### §1.3 색, 선, 점
- **색**은 이름(`black white gray grey silver red green blue yellow orange purple brown pink cyan magenta`, CSS의 값), `'#rrggbb'`·`'#rrggbbaa'`, 0과 1 사이의 수 셋이나 넷의 튜플, 팔레트의 번호 `'C0'`–`'C9'`다.
- **선의 꼴** `style=`: `'-'`, `'--'`, `':'`, `'-.'`. **점의 꼴** `shape=`: `'o'`, `'s'`, `'^'`, `'v'`, `'D'`, `'+'`, `'x'`.
- **글자**는 LaTeX 원문이다(PL7; §10의 8). 수식은 `$...$` 안에 쓴다. `%`, `_`, `&`, `#`은 사용자가 `\`로 숨긴다. 한글을 그대로 쓸 수 있다.
- **범례의 이름** `label=`: 층에 이름을 주면 범례가 생긴다. 함수 그림에 함수를 여럿 주면 `labels=`로 이름의 목록을 준다.

### §1.4 에러
- 메시지는 API.md §2.1의 꼴이다: `function(): 'a' must be less than 'b'`, `scatter(): 'x' and 'y' must have the same length: 10 and 12`, `save(): 'path' must end in '.pdf', '.svg' or '.png', not '.jpg'`.
- 조판이 실패하면 `ValueError`이고 LaTeX의 첫 오류 줄을 붙인다: `xlabel(): LaTeX could not typeset 'x_1 %': ! Missing $ inserted.` TeX가 없으면 `RuntimeError`다: `save(): sd.plot needs 'latex', 'xelatex', 'tex', 'xetex', 'dvisvgm' and 'kpsewhich' (TeX Live) on PATH; 'latex' was not found`.
- 조판은 그릴 때 한다(§2.3의 내보내기와 노트북 표시). 글자를 주는 메서드(`xlabel`)가 조판을 기다리지 않게 하려는 것이다. 그래서 조판 오류는 그리는 메서드의 이름으로 난다(`save(): LaTeX could not typeset ...`). 묶음을 깨뜨리는 문자열(짝이 맞지 않는 중괄호, 숨기지 않은 `%`)은 TeX에 넘기기 전에 같은 꼴의 에러다: `to_svg(): LaTeX could not typeset '50% of': an unescaped % (write \%)`.
- 경고를 내지 않는다. 함수 그림이 한 일(잘린 범위, 찾은 극점)은 `Figure.info`가 알린다(§4.6).

## §2. Figure

### §2.1 만들기와 칸
```
Figure(*, width=89.0, height=None, nrows=1, ncols=1, font='serif', theme='paper', palette='jco')
Figure.width
Figure.height
Figure.shape
Figure.axes
Figure.info
Figure.__getitem__(key, /)
grid(figures, /, *, nrows=None, ncols=None, labels=None, sharex=False, sharey=False, width=None, height=None)
facet(data, by, /, *, ncols=None, sharex=True, sharey=True, width=None, height=None, font='serif', theme='paper', palette='jco')
Figure.twin()
Figure.inset(figure, rect, /)
```
- `width`, `height`는 그림 전체의 바깥 크기(mm)다. 저장한 PDF의 크기가 정확히 이 값이다. `height=None`이면 칸의 가로세로 비가 4:3이 되게 정한다. 기본 너비 89 mm는 학술지의 한 단 너비다.
- `font`: `'serif'`(TeX Gyre Termes, KoPubWorld 바탕, newtxmath) 또는 `'sans'`(TeX Gyre Heros, Apple SD Gothic Neo, newtxsf)다(PL8). 글자 크기는 테마가 정한다: `'paper'`는 라벨 8, 눈금 7, 제목 9, 범례 7 bp다.
- `theme`: 기본 `'paper'`는 왼쪽과 아래의 축 선만, 격자 없음, 바깥쪽 눈금이다. `'boxed'`는 네 변의 축 선, `'grid'`는 눈금마다 옅은 격자를 더한다.
- `palette`: 범주 색의 이름이다. 기본 `'jco'`는 파랑, 노랑, 회색, 벽돌색으로 시작하는 학술지 꼴의 팔레트다(사용자 결정 2026-11-22). 선과 점은 그 색 그대로, 채운 도형(막대, 상자)은 그 색을 옅게 한 채움에 그 색의 가는 테두리다. `'tableau'`, `'deep'`, `'bright'`(색각 이상에 안전), `'okabe-ito'`(색각 이상에 안전), 학술지 꼴의 `'npg'`, `'aaas'`, `'lancet'`, `'nejm'`가 있다. 연속 색은 층의 `cmap=`이 정한다(§4.4).
- `fig[i, j]`(여러 칸)와 `fig[i]`(한 줄이나 한 열)는 그 칸만 가리키는 `Figure`의 보기다. 그 보기의 메서드는 그 칸에 그린다. 칸을 고르지 않은 메서드는 첫 칸에 그린다(칸이 하나면 그 칸이다).
- `grid`는 이미 지은 그림들을 칸으로 모아 새 `Figure`를 만든다. 칸은 복사해 넣으므로, 모은 뒤에 원래 그림이나 모은 그림을 고쳐도 다른 쪽은 그대로다. `labels='A'`는 칸마다 A, B, C…를, `'a'`는 a, b, c…를 왼쪽 위에 굵게 붙인다. 같은 열의 칸은 축의 왼쪽 끝이 한 줄에 선다.
- `grid`의 `sharex`, `sharey`는 칸들의 축 범위를 맞춘다: `True`면 같은 열(x)·같은 줄(y)끼리, `'all'`이면 모든 칸이 한 범위다(범위를 준 칸과 범주 축은 빼고). 맞춘 축의 눈금 숫자는 바깥 칸(열의 맨 아래, 줄의 맨 왼쪽)에만 적는다.
- `facet`은 DataFrame의 열 `by`의 값마다 칸 하나를 만든 `Facets`를 돌려준다(수면 정렬, 아니면 처음 나온 순서; 칸의 제목이 그 값). `Facets`의 그림 메서드(`scatter`, `line`, `histogram` 등)는 모든 칸에 그리고, 칸마다 그 값의 행을 `data`로 준다: 인자는 열 이름이다(`data`는 주지 않는다). 축은 기본으로 모든 칸이 맞추고, `group=`의 무리는 칸마다 같은 색이며 범례는 마지막 칸에만 선다. `xlabel`은 맨 아래 칸들에, `ylabel`은 맨 왼쪽 칸들에, `xlim ylim xscale yscale xticks yticks legend`는 모든 칸에 준다. `save`, `to_svg` 등과 노트북 표시는 그 그림의 것이고, `Facets.figure`가 그 `Figure`다.
- `twin()`은 그 칸의 오른쪽에 두 번째 y축을 두고 그 축을 가리키는 보기를 돌려준다. 보기의 그림 메서드와 `ylabel ylim yscale yticks`는 두 번째 축의 것이고 x축은 함께 쓴다. 색은 첫 축의 층들 다음 색부터 이어지고, 범례는 두 축의 이름을 한데 모은다.
- `inset(figure, rect)`는 칸 하나짜리 그림 `figure`를 이 칸의 그림 위에 얹는다. `rect = (x, y, w, h)`는 그림 사각형에 대한 몫(왼쪽 아래 기준)이고, 작은 그림의 눈금과 글자도 그 안에 든다(흰 바탕). `figure`는 복사해 넣는다.
- 모든 그림의 기본값을 한 번에 바꾸는 전역 설정은 없다. 그림 옵션은 그림마다 준다(전역 상태를 두지 않는다, PL6).

### §2.2 고치기
```
Figure.title(text, /)
Figure.xlabel(text, /)
Figure.ylabel(text, /)
Figure.zlabel(text, /)
Figure.xlim(low, high, /)
Figure.ylim(low, high, /)
Figure.xscale(kind, /)
Figure.yscale(kind, /)
Figure.xticks(values, /, *, labels=None)
Figure.yticks(values, /, *, labels=None)
Figure.legend(*, loc='best', title=None, ncols=1)
Figure.size(width, height=None, /)
```
- 모두 `fig` 자신을 돌려준다(§10의 6): `fig.xlabel(r"$x$").ylabel(r"$\Gamma(x)$").xlim(-4, 4)`.
- `xlim`·`ylim`의 `None`은 그 끝을 자동으로 둔다. `kind`는 `'linear'`와 `'log'`다. 로그 축에서 0 이하의 값은 그리지 않고, 눈금은 10의 거듭제곱이다.
- `xticks(values)`의 `values`가 `'pi'`면 π 단위의 눈금이다(§10의 17).
- `legend(loc=None)`은 범례를 없앤다. `loc`은 `'best'`, `'top'`, `'right'`, `'bottom'`과 칸 안의 `'upper left'` 등이다. `'best'`는 그려진 것을 가리지 않는 구석을 고르고, 그런 구석이 없으면 칸 밖(한 줄에 들어가면 위, 아니면 오른쪽)에 둔다.

### §2.3 내보내기
```
Figure.save(path, /, *, dpi=300)
Figure.to_svg()
Figure.to_pdf()
Figure.to_png(*, dpi=300)
Figure.scene()
```
- `save`는 확장자로 형식을 정한다: `.pdf`, `.svg`, `.png`. `to_svg`는 `str`, `to_pdf`와 `to_png`는 `bytes`다.
- `dpi`는 PNG의 해상도와, PDF·SVG 안에 그림으로 들어가는 층(3D 곡면, 점이 아주 많은 층)의 해상도다.
- 같은 `Figure`는 같은 기계, 같은 스레드 수에서 같은 바이트를 낸다(파일에 시각을 적지 않는다).
- `scene()`은 그리기 전의 장면(pt 좌표의 도형 목록)을 dict로 준다. 테스트와 디버깅에 쓴다.

### §2.4 노트북
- `Figure`는 `_repr_mimebundle_`를 가진다. 3D 칸이 있거나 그림으로 들어가는 층(`density`, `phase`)이 있으면 PNG(200 dpi)를, 아니면 SVG를 한 꼴만 만들어 보인다.
- `repr(fig)`는 한 줄이다: `Figure(89 x auto mm, 1 axes, 2 layers)`. 높이를 주지 않은 그림은 `auto`다(높이는 배치가 정하고, `repr`은 그림을 짓지 않는다). 배치가 정한 높이는 `fig.height`다.

## §3. 기본 그림
```
line(x, y, /, *, data=None, group=None, color=None, width=None, style='-', alpha=1.0, label=None, **figure)
scatter(x, y, /, *, data=None, group=None, color=None, size=None, shape='o', alpha=1.0, label=None, **figure)
bar(x, height, /, *, data=None, group=None, stack=False, gap=0.1, err=None, annotate=False, color=None, width=0.8, horizontal=False, label=None, **figure)
area(x, y1, y2=0.0, /, *, data=None, color=None, alpha=0.3, label=None, **figure)
step(x, y, /, *, data=None, group=None, where='post', color=None, width=None, style='-', label=None, **figure)
errorbar(x, y, /, *, yerr=None, xerr=None, data=None, group=None, color=None, cap=2.0, label=None, **figure)
hline(y, /, *, color=None, width=None, style='-', label=None, **figure)
vline(x, /, *, color=None, width=None, style='-', label=None, **figure)
text(x, y, s, /, *, ha='center', va='center', size=None, color=None, rotation=0.0, **figure)
annotate(s, xy, xytext, /, *, arrow=True, size=None, color=None, **figure)
```
- `bar`의 `x`는 범주(문자열의 시퀀스)이거나 수다. `group=`이 있으면 무리의 막대가 나란히 서고, 막대 사이를 막대 너비의 `gap`(0 이상 1 미만, 기본 0.1)만큼 띄운다. `stack=True`면 무리를 한 자리에 쌓는다(양수는 위로, 음수는 아래로, 무리의 순서대로). `err`는 높이의 오차다: 배열 하나(대칭)이거나 `(아래, 위)` 쌍이고, 열 이름도 받는다. 쌓은 막대에는 줄 수 없다. `annotate`는 막대의 값을 적는다: `True`면 유효숫자 세 자리, 문자열이면 Python의 서식(`'.1f'`, `'.0%'`, `',.0f'`)이다. 값은 막대(오차 막대가 있으면 그 끝) 바깥에, 쌓은 막대는 조각 안 가운데에 선다. `horizontal=True`는 가로 막대다. `width`는 한 자리의 막대들이 차지하는 너비(자료 단위)다.
- `yerr`·`xerr`는 수 하나, 길이 n의 시퀀스(위아래 같은 길이), (2, n) 배열(아래와 위)이다.
- 위 이름 모두 같은 인자의 `Figure` 메서드가 있다(§1.1).

## §4. 함수 그림

### §4.1 함수를 받는 법
- `f`는 float64 `sd.ndarray`를 받아 같은 모양의 배열을 돌려주는 호출 가능한 객체다: `sd.sin`, `sd.special.gamma`, `lambda x: sd.exp(-x**2)`. 그림은 표본을 묶어 몇 번만 부른다(PL14). 수 하나를 돌려주는 함수(`lambda x: 0.5`)는 상수 함수다. 리스트나 다른 모양의 배열을 돌려주면 에러다(`function(): 'f' must return an sd.ndarray (it is called with one), not list`). 스칼라만 받는 함수는 받지 않는다: 점마다 부르면 표본 수만큼 Python 호출이 든다(PL14).
- 함수의 목록을 주면 한 칸에 겹쳐 그리고 색을 차례로 준다. 범위는 함께 잡는다.
- 범위는 언제나 준다(§10의 16). `a < b`가 아니거나 유한하지 않으면 `ValueError`다.

### §4.2 곡선
```
function(f, a, b, /, *, ylim=None, asymptotes=False, samples=None, refine=None, color=None, width=None, style='-', label=None, labels=None, **figure)
parametric(fx, fy, a, b, /, *, samples=None, refine=None, color=None, width=None, style='-', label=None, **figure)
polar(r, a, b, /, *, samples=None, refine=None, color=None, width=None, style='-', label=None, **figure)
```
- `function`은 y = f(x)를 [a, b]에서 그린다. 표본은 굽은 곳에서 알아서 촘촘해지고, 극점과 뜀에서 선이 끊기며, y의 범위는 본체에 맞춰 잡힌다(설계 §4). `ylim=(low, high)`를 주면 자동 범위를 쓰지 않는다.
- `asymptotes=True`는 찾은 극점마다 가는 세로 점선을 그린다(§10의 15).
- `parametric`은 (fx(t), fy(t)), `polar`는 반지름 r(θ)의 곡선이다(θ는 라디안). 가로세로 비는 1:1로 둔다. 곡선이 한 축에 나란한 직선이면(`polar(1/cos θ)`는 직선 x = 1) 폭이 없는 쪽의 축은 직선을 가운데 두고 다른 축과 같은 길이로 잡는다. 점 하나뿐이면 두 축 모두 길이 1이다.

### §4.3 음함수와 영역
```
implicit(f, x0, x1, y0, y1, /, *, level=0.0, samples=None, refine=None, color=None, width=None, style='-', label=None, **figure)
region(f, x0, x1, y0, y1, /, *, samples=None, refine=None, color=None, alpha=0.3, label=None, **figure)
```
- `f`는 같은 모양의 배열 둘(x, y)을 받아 배열을 돌려준다. `implicit`는 f(x, y) = level인 곡선, `region`은 f(x, y) > 0인 곳을 칠한다.
- **이미 계산한 값**: `implicit`, `region`, `contour`, `density`, `surface`의 `f` 자리에 함수 대신 2차원 배열 `Z`를 줄 수 있다. `Z[j, i]`는 상자를 고르게 나눈 격자의 (x_i, y_j)의 값이고 첫 줄이 `y0`다(`sd.meshgrid(x, y, indexing='xy')`로 만든 격자의 값이 그대로 맞는다). `density`는 값 하나를 화소 하나의 가운데 값으로 본다. 값에는 적응형 표본이 없으므로 `samples`와 `refine`을 주면 `ValueError`다.

### §4.4 두 변수의 함수
```
contour(f, x0, x1, y0, y1, /, *, levels=10, filled=False, labels=True, samples=None, refine=None, cmap=None, color=None, **figure)
density(f, x0, x1, y0, y1, /, *, samples=None, cmap=None, vmin=None, vmax=None, colorbar=True, **figure)
phase(f, re0, re1, im0, im1, /, *, samples=None, contours=True, **figure)
vectorfield(fu, fv, x0, x1, y0, y1, /, *, samples=None, scale=1.0, color=None, cmap=None, width=None, colorbar=True, label=None, **figure)
streamlines(fu, fv, x0, x1, y0, y1, /, *, cells=25, samples=None, color=None, cmap=None, width=None, arrows=True, colorbar=True, label=None, **figure)
```
- `levels`는 개수(int)이거나 값의 시퀀스다. 개수면 눈금과 같은 규칙으로 "고운" 값을 고른다.
- `contour`의 `levels`가 개수면 수준은 값의 대부분이 있는 범위에서 고르게 뽑는다(극점 둘레의 큰 값은 뺀다). 함수가 빨리 변하는 곳은 알아서 4배 촘촘히 잰다.
- `contour`의 `labels`는 수준선 위에 그 값을 적는다. 숫자는 칸 안에서 서로 닿지 않는 자리를 골라 서고, 자리가 없는 수준은 숫자 없이 둔다. `filled=True`면 숫자 대신 색 막대가 붙는다.
- `implicit`와 `region`은 곡선이 지나는 곳 가까이만 가로세로 4배로 다시 잰다: 스치는 곡선, 뾰족점, 두 가지가 만나는 점이 또렷하다.
- `cmap`은 연속 색의 이름이다. 기본은 어디서나 `'cool'`이다(아래 하늘색, 가운데 보라, 위 빨강; 양 끝이 어둡지 않다). 그 밖에 밝은 `'spectral'`, `'coolwarm'`, `'sunrise'`, `'lagoon'`, 지각적으로 고른 `'viridis'`, `'magma'`, `'cividis'`, 발산형 `'rdbu'`가 있다.
- `phase`는 복소 함수의 위상 그림이다. `f`는 complex128 배열을 받아 complex128 배열을 돌려준다. 위상이 색상, 크기가 밝기의 띠다.
- `vectorfield`는 벡터장 (fu(x, y), fv(x, y))를 칸의 가운데마다 화살표로 그린다. `samples`는 화살표의 수(가로, 또는 `(가로, 세로)`; 기본은 가로 21개에 상자 비율대로)다. 가장 긴 화살표가 칸의 90%이고 `scale`을 곱한다. `streamlines`는 같은 벡터장의 흐름선을 상자에 고르게 퍼뜨려 그린다: 상자를 가로 `cells`칸의 정사각형으로 나눠 한 칸에 선 하나까지 지나게 하고, 선마다 가운데에 흐름 방향의 화살촉을 단다(`arrows=False`로 뺀다). 장은 `samples`의 격자(기본 가로 `6 × cells + 1`)에서 잰 값을 겹선형으로 읽는다. 둘 다 색은 크기에 따른 연속 색(기본)이거나 `color` 한 색이고, 크기의 색 범위는 값의 대부분이 있는 범위다(극점 둘레의 큰 값은 끝 색). `fu`, `fv`에 2차원 값 배열 둘을 줄 수도 있다. 칸은 같은 비율이다.
- 격자 자료(이미 계산한 값의 배열)의 열지도는 §5의 `heatmap`이다. 격자 자료의 등고선은 아직 없다(함수를 받는 `contour`만 있다).

### §4.5 3D
```
surface(f, x0, x1, y0, y1, /, *, zlim=None, mesh=True, contours=False, color=None, cmap=None, alpha=1.0, samples=None, refine=None, azimuth=None, elevation=None, perspective=True, **figure)
parametric_surface(fx, fy, fz, u0, u1, v0, v1, /, *, mesh=True, color=None, cmap=None, alpha=1.0, samples=None, refine=None, azimuth=None, elevation=None, perspective=True, **figure)
curve3d(fx, fy, fz, a, b, /, *, samples=None, refine=None, color=None, width=None, label=None, azimuth=None, elevation=None, perspective=True, **figure)
scatter3d(x, y, z, /, *, data=None, group=None, color=None, size=None, label=None, azimuth=None, elevation=None, perspective=True, **figure)
```
- 곡면은 음영과 격자선(`mesh`)이 있는 그림으로 굽고, 상자 축과 눈금과 글자는 벡터로 얹는다(PL9). `cmap`을 주면 높이로 색을 정하고, `color`를 주면 한 색에 음영만 준다. 둘 다 주지 않으면 칸에 곡면이 하나일 때는 `cmap='cool'`(아래 하늘색, 가운데 보라, 위 빨강)이고, 여럿일 때는 곡면마다 팔레트의 색을 차례로 받는다. 빛은 고정이다: 왼쪽 위 앞의 주광, 반대쪽의 보조광, 뒤쪽의 테두리광 셋이고, 넓은 광택과 또렷한 반사광이 얹힌다. 곡면의 두 면을 똑같이 밝힌다. 상자의 꼭짓점에서 두 축의 끝 눈금 숫자가 가까이 서면(가로로 눈금 글자 크기, 세로로 그 0.6배 안) 뒤에 그리는 축의 숫자를 뺀다. 눈금 선은 남는다.
- `mesh`는 격자선이다: `True`(한 방향 24칸), `False`, 또는 칸의 수. `contours`는 상자의 바닥에 그리는 등고선이다: `True`(수준 약 14개), `False`, 또는 수준의 수(눈금 규칙으로 고른다). 등고선은 곡면과 같은 색을 받는다.
- `alpha`는 곡면의 불투명도다(0보다 크고 1 이하). 반투명한 곡면은 뒤의 것이 비치고, 여러 장이 겹치거나 서로 뚫고 지나가도 깊이 순서대로 비친다(12겹까지).
- 꺾이는 곳, 극점의 옆구리, 정의역의 가장자리는 알아서 더 촘촘히 잰다(첫 격자의 16배까지). `f`는 그때마다 배열 하나로 불린다.
- `curve3d`는 곡선이 꺾이는 곳을 알아서 더 촘촘히 잰다. 값이 달아나는 곳(극점)에서는 선을 끊고, 축은 값의 대부분이 있는 범위로 잡아 상자의 면에서 선을 자른다.
- 어디나 거친 곡면(칸의 40%가 넘게 모자람)은 격자 전체를 3배로 다시 잰다.
- 3D 칸의 범례: `scatter3d`의 무리와 `label`을 준 `curve3d`·`scatter3d`가 범례에 들고, 범례는 상자 오른쪽 밖에 선다(`legend(loc=None)`으로 없앤다). 3D 칸에는 `xlim ylim xscale yscale xticks yticks`가 없고(`ValueError`), 2D 그림을 더할 수도 없다(`line(): the cell already holds a 3D plot`).
- `azimuth`, `elevation`은 보는 방향(도)이다. 주지 않은 각은 알아서 고른다: 방위각 여덟(±30°, ±60°, ±120°, ±150°: 상자의 축을 정면으로 보는 방향은 뺀다), 고도 18°·30°·42° 가운데 곡면의 넓이가 가장 많이 보이고(다른 부분에 가리거나 옆으로 누워 보이는 것이 적고) 곡선과 점이 화면에 가장 넓게 퍼지는 방향이다. 차이가 작으면 늘 쓰던 방향(−60°, 30°)이다. 한 각만 주면 다른 각만 고른다. 한 칸에 3D 층을 여럿 겹칠 수 있고 서로 가리는 것은 깊이로 정한다. 시점은 그 칸의 첫 층이 정한다.

### §4.6 `Figure.info`
- 함수 그림이 한 일을 곡선마다 적은 기록의 튜플이다. 기록은 namedtuple `PlotInfo(kind, n, poles, jumps, ylim, clipped, undefined, oscillating)`다: 그림의 종류(`'function'`, `'parametric'`, `'polar'`), 쓴 표본 수 `n`, 찾은 극점과 뜀의 자리(`poles`, `jumps`; 표본의 가장 좁은 간격까지, 범위의 약 4×10⁻⁶), 그림에 보이는 값의 범위 `ylim`, 그 범위가 아래와 위에서 값을 잘랐는지 `clipped`, 유한하지 않은 값이 나온 구간들 `undefined`, 선 대신 띠로 그린 구간들 `oscillating`.

## §5. 통계 그림
```
histogram(x, /, *, data=None, group=None, bins=10, range=None, density=False, color=None, alpha=None, label=None, **figure)
hist2d(x, y, /, *, data=None, bins=40, range=None, density=False, cmap=None, colorbar=True, **figure)
hexbin(x, y, /, *, data=None, size=30, range=None, cmap=None, colorbar=True, **figure)
ecdf(x, /, *, data=None, group=None, color=None, width=None, label=None, **figure)
boxplot(x, y, /, *, data=None, group=None, points=False, color=None, width=0.6, **figure)
strip(x, y, /, *, data=None, group=None, jitter=0.2, rng=None, color=None, size=None, alpha=1.0, **figure)
qq(x, /, *, dist=None, data=None, line=True, color=None, **figure)
heatmap(z, /, *, xlabels=None, ylabels=None, cmap=None, vmin=None, vmax=None, annotate=False, colorbar=True, **figure)
pairs(data, /, *, columns=None, group=None, width=None, height=None, font='serif', theme='paper', palette='jco')
```
- **그림의 수는 계산 영역의 수다.** `histogram`의 `bins`, `range`, `density`는 `sd.histogram`의 것이고 막대의 높이가 그 결과다. `boxplot`의 상자는 `sd.quantile(method='linear')`의 사분위, 수염은 1.5 IQR 안의 가장 먼 값, 그 밖은 점이다. `qq`의 이론 분위수는 분포의 `icdf`다(`dist=None`이면 표준정규).
- `boxplot`과 `strip`의 `x`는 범주, `y`는 값이다. `x`를 `None`으로 주면 상자 하나다.
- `strip`의 흩음은 `rng`(`sd.random`의 생성기나 씨)로 정해진다. `rng=None`은 고정된 씨다. 같은 호출은 같은 그림이다.
- `sd.plot.histogram`은 그림을, `sd.histogram`은 값을 돌려준다(API.md §6에 적는다; §10의 4).
- `hist2d`는 (x, y) 짝을 격자의 칸에 세어 그림으로 그린다. `bins`는 수 하나(두 방향 같게)이거나 `(가로, 세로)`, `range`는 `((x0, x1), (y0, y1))`이다(밖의 짝은 세지 않는다). `density=True`면 넓이로 나눈 밀도다. `hexbin`은 같은 일을 육각형 칸으로 하고, `size`는 가로로 놓이는 육각형의 수다(4 : 3 칸에서 정육각형으로 보인다). 둘 다 빈 칸은 비워 두고 색 막대를 붙인다. 점이 수만 개 넘어 산점도가 뭉칠 때 쓴다.
- `heatmap`의 `annotate`는 칸마다 값을 적는다: `True`면 값마다 유효숫자 세 자리, 문자열이면 `bar`와 같은 서식(`'.2f'` 등)이다. 글자는 칸의 밝기에 따라 검정이나 흰색이다.
- **아직 없는 것**: 밀도 곡선과 바이올린은 `sd.stats`에 핵밀도추정이 생기면 넣는다(PL15). 평균과 오차 막대는 `sd.stats`의 값으로 `errorbar`를 부른다.

## §6. 결과 객체의 그림
```
Figure.compare(result, /, *, groups=None, format='p', y=None)
Figure.fit(result, /, *, band=True, alpha=0.05, equation=True, color=None)
diagnostics(result, /, *, width=None, height=None, font='serif', theme='paper', palette='jco')
coefficients(result, /, *, alpha=0.05, intercept=False, color=None, **figure)
distribution(dist, /, *, kind='pdf', a=None, b=None, color=None, label=None, **figure)
```
- `compare`는 `sd.htest`의 결과를 받아 범주 축 그림(`boxplot`, `strip`, `bar`) 위에 괄호와 p값을 그린다. 그림이 검정을 고르거나 다시 계산하지 않는다. `groups`는 괄호가 잇는 두 범주의 라벨이다(범주가 둘뿐이면 생략한다). 다중 비교의 결과(k × k의 `pvalue`: `tukey_hsd` 등)는 범주의 순서대로 모든 짝의 괄호를 그리고 `groups`를 받지 않는다. `dunnett`의 결과는 첫 범주를 대조군으로 보고 나머지 범주마다 괄호를 그린다. 괄호는 자료 위에 쌓인다: 가까운 짝이 아래, 먼 짝이 위이고, `compare`를 다시 부르면 그 괄호가 위에 선다. 그만큼 축이 위로 늘어난다(로그 축이면 로그로). p값이 NaN이면 `n/a`라고 적는다. `format`은 `'p'`(`p = 0.012`, 0.001보다 작으면 `p < 0.001`)나 `'stars'`(`ns`, `*`, `**`, `***`: 0.05, 0.01, 0.001)다. `y`는 아직 받지 않는다(`None`만).
- `fit`은 `sd.regression`의 결과를 받아 산점도 위에 회귀선과 신뢰 띠(평균의 구간, `conf_int`와 같은 `alpha`; 결과에 `predict_interval`이 있을 때), 식과 R²를 그린다. 설명 변수가 하나인 모형만 받는다. 계수는 유효숫자 세 자리이고 아주 작거나 크면 `1.2\times10^{-5}`로 적는다. GLM은 연결 함수의 척도로 적는다(`log μ = 0.2x − 0.02`).
- `diagnostics`는 회귀 결과의 진단 넷(잔차–적합값, QQ, 척도–위치, 잔차–지렛값)을 2 × 2 칸으로 그린다.
- `coefficients`는 계수와 그 구간을 점과 가로 막대로 그린다. 구간이 없는 결과(`conf_int`가 없는 `ridge` 등)는 점만 그린다.
- `distribution`은 `sd.stats`의 분포를 그린다. `kind`는 `'pdf'`(이산 분포는 pmf의 막대), `'cdf'`, `'sf'`다. `a`, `b`가 `None`이면 분포의 분위수로 범위를 잡는다. 밀도의 세로 범위는 봉우리까지 닿는다(밀도가 범위의 끝에서 달아나면 함수 그림의 규칙으로 자른다). 이산 분포의 가로 눈금은 정수다.
- 받는 것은 형이 아니라 가진 것으로 가린다: `compare`는 `pvalue`가 있는 결과(값 하나이거나 k × k 행렬), `fit`은 `params`와 `predict`가 있는 회귀 결과, `coefficients`는 `params`가 있는 회귀 결과(`conf_int`가 있으면 구간도), `diagnostics`는 `fittedvalues`·`resid`·`resid_studentized_internal`·`leverage`가 있는 선형 회귀 결과, `distribution`은 `cdf`와 `icdf`가 있는 분포다. 배치 결과(여러 적합)는 `ValueError`다.

## §7. matplotlib과 다른 곳 (모음)
| matplotlib | sd.plot | 이유 |
|---|---|---|
| `plt.plot(x, y)`와 "지금 그림" | `sd.plot.line(x, y)`가 `Figure`를 돌려준다 | 전역 상태가 없다(PL6) |
| `plt.show()` | 없다. 노트북이 값을 보이거나 `save`한다 | 창을 만들지 않는다(PL3) |
| 크기는 인치(`figsize`) | mm(`width`, `height`) | 학술지가 mm로 정한다 |
| 수식은 mathtext, `usetex`는 옵션 | 모든 글자가 LaTeX | 글자의 품질(PL7) |
| `hist`, `imshow`, `plot`, `'r'`, `lw=` | `histogram`, `heatmap`·`density`, `line`, `'red'`, `width=` | 별칭과 줄임을 두지 않는다 |
| `tight_layout`, `constrained_layout` | 없다. 언제나 글자를 재서 여백을 푼다 | 바깥 크기가 고정이다 |
| `ax.set_xlabel(...)` | `fig.xlabel(...)`, 칸은 `fig[i, j]` | `Figure` 하나로 한 줄에 쓴다 |
| 3D는 도형을 정렬해 덮어 그린다 | 깊이 버퍼 | 교차하는 곡면이 옳다(PL5, PL9) |
