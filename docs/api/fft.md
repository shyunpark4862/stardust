# sd.fft API

> 상태: **초안** (FT0 마일스톤, 2026-11-23). FT1 마일스톤(2026-11-23)에서 `fft ifft fftfreq rfftfreq fftshift ifftshift next_fast_len prev_fast_len`이, FT2 마일스톤(2026-11-23)에서 `rfft irfft hfft ihfft`가, FT3 마일스톤(2026-11-23)에서 `fft2 ifft2 fftn ifftn rfft2 irfft2 rfftn irfftn hfft2 ihfft2 hfftn ihfftn`이 공개되었다. 아래 시그니처는 numpy 2.5.4의 `numpy.fft`와 scipy 1.18.1의 `scipy.fft`의 이름과 인자 이름에 stardust의 규칙과 FT1–FT34의 결정을 입힌 안이고, 마일스톤 FT1–FT5에서 하나씩 공개된다(docs/design/fft.md §8: 1차원 복소 변환과 도우미는 FT1, 실수·에르미트 1차원은 FT2, 다차원은 FT3, DCT·DST는 FT4, FFTLog는 FT5).

- 설계와 결정의 근거는 [docs/design/fft.md](../design/fft.md)(§1.1 FT1–FT52), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/fft/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 이름과 시그니처(`name(...)`)다. 이 이름들이 계획한 공개 이름 전부이고, 공개된 것은 시그니처가 블록과 같아야 한다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양
- **이름공간은 평평하다**(FT3): 모든 함수가 `sd.fft`에 바로 있다(`sd.fft.fft`, `sd.fft.dct`, `sd.fft.fftfreq`). 하위 모듈이 없다. 이름은 `scipy.fft`의 것이고(FT2) 겹치지 않는다. `sd`의 다른 이름과도 겹치지 않는다(`sd.fft`는 이 모듈 하나다).
- **데이터는 위치로만, 옵션은 키워드로만** 받는다(FT6). `n`, `axis`, `s`, `axes`, `norm`, `type`, `orthogonalize`, `d`, `real`은 키워드만이다. numpy·scipy 코드의 `fft(x, 8)`, `dct(x, 2)`, `fftfreq(n, 0.1)`은 `TypeError`다(`[numpy와 다름]`, `[scipy와 다름]`). 변환의 필수 인자는 위치다: `fftfreq`의 `n`, `next_fast_len`의 `target`, `fht`의 `dln`과 `mu`(FT32; `sd.spatial.mahalanobis(u, v, VI)`의 `VI`와 같은 자리).
- **별칭이 없다**: `numpy.fft`에만 있는 이름이 없다(`numpy.fft`의 이름은 모두 `scipy.fft`에도 있다).
- `workers`(FT8; 스레드 수는 `sd.set_num_threads` 하나다), `overwrite_x`(FT9), `plan`(FT10), `out`(FT11), `device`는 없다. `set_workers get_workers set_backend skip_backend set_global_backend register_backend`도 없다. 주면 모르는 키워드의 `TypeError`다.
- 틀린 호출 모양은 함수 이름으로 시작한다(API.md §2.1): `fft(): unexpected keyword argument 'workers'`, `fft(): takes 1 positional argument but 2 were given`.

### §1.2 입력과 결과의 dtype (FT5, FT15, FT16)
- **데이터** `x`는 `sd.ndarray`만 받는다(FT5). 리스트와 numpy 배열은 `TypeError`다(`fft(): 'x' must be an sd.ndarray, not list`). 0차원 배열은 sd에 없으므로 입력은 언제나 1차원 이상이다.
- 계산은 float64와 complex128이다. float32 판은 없다(FT15). 결과는 언제나 새 배열이고 C 순서다. 입력을 고치지 않는다.

| 함수 | bool·int64·float64 입력 | complex128 입력 | 결과 dtype |
|---|---|---|---|
| `fft ifft fft2 ifft2 fftn ifftn` | float64로 바꿔 실수 입력의 길로 계산한다(FT16) | 그대로 | complex128 |
| `rfft rfft2 rfftn ihfft ihfft2 ihfftn` | float64로 바꾼다 | `TypeError` (numpy·scipy와 같다) | complex128 |
| `irfft irfft2 irfftn hfft hfft2 hfftn` | 허수부가 0인 complex128로 본다 | 그대로 | float64 |
| `dct idct dst idst dctn idctn dstn idstn` | float64로 바꾼다 | 실수부와 허수부를 따로 변환한다(FT28) | 입력이 complex128이면 complex128, 아니면 float64 |
| `fftshift ifftshift` | dtype 그대로 | 그대로 | 입력과 같다 |
| `fftfreq rfftfreq` | — | — | float64 |
| `next_fast_len prev_fast_len` | — | — | Python `int` |
| `fht ifht` | float64로 바꾼다 | `TypeError` (scipy와 같다) | float64 |

- **실수 입력의 `fft`**(FT16): 실수 dtype이면 실수 변환을 하고 에르미트 대칭으로 채운다(FT2 마일스톤부터; FT1 마일스톤에서는 허수부가 0인 복소 입력으로 계산했다, FT38). 그래서 `fft(x)`의 앞 n//2 + 1개는 `rfft(x)`와 비트까지 같고 y_{n−k} = conj(y_k)가 정확하며 DC의 허수부는 +0이다(numpy와 같고 `[scipy와 다름]`: scipy는 −0). 실수 입력의 `ifft`는 `conj(fft(x))`에 역방향 인자를 곱한 것과 비트까지 같다(FT46). 끝자리는 `fft(x.astype(complex128))`와 다를 수 있다(둘 다 정확도 계약 안이다). scipy도 실수 입력에 실수 변환을 쓴다(`rfft`가 `fft`의 앞쪽과 같은 비트다). numpy는 같지 않다(길이 6 이상에서 다르다).
- **int 인자**(`n`, `s`의 원소, `axis`, `axes`의 원소, `type`, `target`)는 API.md §2의 규칙이다: `operator.index`로 읽고 bool은 `TypeError`(`[scipy와 다름]`: scipy는 `n=True`를 1로 받는다), int64 밖은 `OverflowError`다.

### §1.3 결정성 (FT22, FT25, FT30)
- **같은 입력이면 스레드 수, 배치 모양, 메모리 레이아웃과 상관없이 같은 결과다.** 한 줄(변환 축을 따라 놓인 원소들)의 결과는 그 줄의 값과 변환의 종류·길이·`norm`으로만 정해진다: `fft(X, axis=a)`의 한 줄은 그 줄만 꺼내 `fft`한 것과 바이트까지 같다.
- 한 (종류, 길이)는 한 길로만 계산한다. 길은 종류와 길이로만 정한다(FT22). 첫 판은 모든 길이를 자체 C++로 계산한다(FT30: vDSP가 받는 길이 꼴에서도 자체 길이 더 빨랐다). 그래서 결과가 Accelerate의 스레드 모드와도 무관하다.
- 다차원 변환은 `axes`의 뒤에서 앞으로 한 축씩 계산한다(FT25; numpy와 같은 순서, 레이아웃과 무관). 큰 1차원 변환 하나는 길이로만 정한 4단계 분해로 풀에서 나누므로 결과가 스레드 수와 무관하다.

### §1.4 에러와 경계
- API.md §2의 종류와 꼴이다. 메시지는 부른 함수의 이름으로 시작한다(`fft(): ...`, `dctn(): ...`). 경고는 내지 않는다.
- **길이 0**(FT12): 변환 축의 길이(입력, `n`, `s`)가 0이면 `ValueError`다(numpy·scipy와 같다). 변환하지 않는 축의 길이가 0인 배열은 받고 빈 결과를 낸다(`fft(zeros((0, 4)))`는 (0, 4)). `n`·`s`의 0과 음수(−1 뺌)는 `ValueError`다(`[scipy와 다름]`: scipy의 `irfft(x, n=0)`은 길이 1의 결과를 낸다).
- **`s`**(FT51): 원소가 int가 아니면(bool, None 포함) `TypeError`(`fftn(): 's' must be an int or a tuple or list of ints, not NoneType`; `[numpy와 다름]`: numpy는 None 원소를 폐기 예고와 함께 받는다), 0과 −2 이하는 `ValueError`(`fftn(): the entries of 's' must be at least 1 (or -1), not 0`), `axes` 없이 준 `s`가 축보다 많으면 `fftn(): 's' has 3 entries for an array of dimension 2`, 빈 `s`는 `fftn(): 's' must not be empty`, `axes`와 길이가 다르면 `fftn(): 's' and 'axes' must have the same length, not 3 and 2`(모두 `ValueError`; numpy·scipy도 `ValueError`)다.
- **축**: 범위 밖의 `axis`·`axes`는 `ValueError`다(`[numpy와 다름]`, `[scipy와 다름]`: 둘 다 `IndexError`). 중복된 축은 `ValueError`다(FT13; scipy와 같고 `[numpy와 다름]`: numpy는 같은 축을 두 번 변환한다). 빈 `axes`는 `ValueError`다(FT23: `fftn(): 'axes' must not be empty`; `[numpy와 다름]`, `[scipy와 다름]`: 둘 다 입력 객체 자체를 돌려준다. scipy의 `rfftn`도 막는다).
- **유한하지 않은 값**(FT17): IEEE 산술을 그대로 거친 결과다. 어느 칸이 NaN이나 inf가 되는지는 약속하지 않는다(같은 기계·같은 입력이면 같은 비트다). numpy와 scipy도 서로 다르다: 길이 2–64에서 원소 하나가 NaN·inf·−inf인 실수 입력의 `fft`·`rfft` 12,474가지 가운데 4,018가지에서 칸마다의 무늬가 달랐다.
- 요청한 결과(`n`·`s`의 0 채우기 포함)가 물리 메모리보다 크면 할당 전에 `MemoryError`다(API.md §2).

### §1.5 정규화 `norm` (FT7)
| `norm` | 정변환(`fft`, `rfft`, `ihfft`, `dct`, `dst`, `fht`) | 역변환(`ifft`, `irfft`, `hfft`, `idct`, `idst`, `ifht`) |
|---|---|---|
| `'backward'` (기본) | 1 | 1/N |
| `'ortho'` | 1/√N | 1/√N |
| `'forward'` | 1/N | 1 |

- N은 변환의 길이다: 복소·실수 변환은 결과 쪽 길이 n(다차원은 `s`의 곱), DCT·DST는 scipy의 논리 길이(DCT-I은 2(N − 1), DST-I은 2(N + 1), 그 밖은 2N)다. `hfft`는 역변환 쪽, `ihfft`는 정변환 쪽이다(numpy와 같다).
- 값은 `'backward'`, `'ortho'`, `'forward'` 가운데 하나다. None은 `TypeError`, 다른 문자열은 `ValueError`다(`fft(): 'norm' must be 'backward', 'ortho' or 'forward', not 'bad'`; `[numpy와 다름]`: numpy는 None을 `'backward'`로 받는다).
- 정규화는 결과에 인자 하나를 곱한다(1/√N은 fl(1/fl(√N))). 인자가 2의 거듭제곱이면 정확하다.

### §1.6 numpy·scipy와 다른 점 (모음)
- 호출 모양: 옵션은 키워드만, 리스트·numpy 입력 없음, `norm=None` 없음, `workers`·`overwrite_x`·`plan`·`out`·`device`·백엔드 제어 없음, int 인자에 bool은 `TypeError`.
- 축: 범위 밖은 `ValueError`, 중복은 `ValueError`(numpy의 `fftn`·`fftshift`는 두 번 변환하거나 민다), 빈 `axes`는 `ValueError`, `s`의 None 원소는 `TypeError`.
- 경계에서 틀린 값이나 내부 메시지를 내는 곳은 모두 `ValueError`다: scipy의 `irfft(n=0)`(길이 1의 결과), scipy의 DCT-I N = 1(DUCC0의 `no zero-sized FFTs`가 새어 나온다), numpy·scipy의 `fftfreq(0)`과 `d=0`(`ZeroDivisionError`), scipy의 `next_fast_len(0)`·`prev_fast_len(0)`(0).
- 결과의 dtype: float32 입력이 없으므로 언제나 complex128·float64다.

## §2. 복소 변환 (FT1, FT3)
```
fft(x, /, *, n=None, axis=-1, norm='backward')
ifft(x, /, *, n=None, axis=-1, norm='backward')
fft2(x, /, *, s=None, axes=(-2, -1), norm='backward')
ifft2(x, /, *, s=None, axes=(-2, -1), norm='backward')
fftn(x, /, *, s=None, axes=None, norm='backward')
ifftn(x, /, *, s=None, axes=None, norm='backward')
```
- 정의는 numpy와 같다: `fft`는 Y_k = Σⱼ xⱼ·e^(−2πijk/n), `ifft`는 지수의 부호가 +이고 정규화는 §1.5다.
- **`n`**: None이면 입력의 그 축 길이다. 주면 1 이상의 int이고, 입력보다 짧으면 앞에서 자르고 길면 끝에 0을 채운다(numpy·scipy와 같다).
- **`axes`**: None이면 모든 축(`fft2` 계열은 (−2, −1))이다. int의 tuple이나 list다.
- **`s`**: None이면 각 축의 입력 길이다. 주면 원소가 1 이상의 int이거나 −1(그 축의 입력 길이; numpy·scipy와 같다)인 tuple이나 list, 또는 int 하나다(FT51; `axes`와 같은 꼴, scipy와 같고 `[numpy와 다름]`). `axes`를 함께 주면 길이가 같아야 한다(`ValueError`). `axes` 없이 주면 마지막 len(s)개 축이다(FT14; numpy가 폐기 예고했지만 scipy와 지금의 numpy가 함께 하는 동작이다).
- 다차원 판은 `axes`의 축마다 1차원 변환을 거듭한 것이고 `axes`의 뒤에서 앞으로 계산한다(FT25, FT47). 정규화는 N = `s`의 곱의 인자 하나다(FT48; numpy는 축마다 곱한다). 그래서 축 하나의 `fftn`은 `fft`와, `fftn(x)`(`'backward'`)와 `ifftn(x, norm='forward')`는 1차원 변환을 뒤에서 앞으로 이은 것과 비트까지 같다. 실수 입력의 `fftn`은 첫 패스가 실수 길이라 앞쪽 n//2 + 1칸이 `rfftn`과 비트까지 같다. 1차원 입력의 `fft2`는 `ValueError`다(`fft2(): axis -2 is out of bounds for an array of dimension 1`; `[numpy와 다름]`: numpy는 `IndexError`).
- **길이 1**: `fft`·`ifft`는 입력을 complex128로 바꾼 복사본과 정확히 같다(`norm`의 인자도 1이다).
- **메시지**(FT42): `fft(): 'n' must be an int or None, not float`, `fft(): 'n' must be at least 1, not 0`, `fft(): 'axis' must be an int, not NoneType`, `fft(): axis 1 is out of bounds for an array of dimension 1`, `fft(): cannot transform an axis of length 0`.
- **길과 결정성**(FT35–FT37): 소인수가 모두 11 이하인 길이는 자체 혼합 기수, 그 밖은 Bluestein(안쪽 길이는 2n − 1 이상의 가장 작은 2ᵃ3ᵇ5ᶜ)으로 계산한다. 길은 길이로만 정해진다. `ifft(x, norm='forward')`는 실수부와 허수부를 바꾼 입력의 `fft`에서 다시 실수부와 허수부를 바꾼 것과 비트까지 같다.
- R과의 관계: R의 `fft(z)`는 `fft(z)`, `fft(z, inverse=TRUE)`는 `ifft(z, norm='forward')`(R의 역변환은 1/n을 곱하지 않는다), `mvfft(Z)`는 `fft(Z, axis=0)`이다.

## §3. 실수와 에르미트 변환 (FT2, FT3)
```
rfft(x, /, *, n=None, axis=-1, norm='backward')
irfft(x, /, *, n=None, axis=-1, norm='backward')
hfft(x, /, *, n=None, axis=-1, norm='backward')
ihfft(x, /, *, n=None, axis=-1, norm='backward')
rfft2(x, /, *, s=None, axes=(-2, -1), norm='backward')
irfft2(x, /, *, s=None, axes=(-2, -1), norm='backward')
rfftn(x, /, *, s=None, axes=None, norm='backward')
irfftn(x, /, *, s=None, axes=None, norm='backward')
hfft2(x, /, *, s=None, axes=(-2, -1), norm='backward')
ihfft2(x, /, *, s=None, axes=(-2, -1), norm='backward')
hfftn(x, /, *, s=None, axes=None, norm='backward')
ihfftn(x, /, *, s=None, axes=None, norm='backward')
```
- **`rfft`**: 실수 x의 `fft` 가운데 앞 n//2 + 1개다. DC와(짝수 n이면) 나이퀴스트 칸의 허수부는 정확히 +0이다(numpy·scipy와 같다).
- **`irfft`**: 결과 길이 n의 기본값은 2(m − 1)이다(m은 입력의 그 축 길이; numpy·scipy와 같다). m = 1이면 기본 n이 0이라 `ValueError`다. 입력에서 n//2 + 1개만 쓰고(길면 자르고 짧으면 0을 채운다), 쓰는 칸 가운데 DC와(짝수 n이면) 나이퀴스트 칸의 허수부는 버린다(numpy·scipy와 같다: 에르미트 확장에 그 칸의 짝이 없다). 홀수 n은 명시해야 한다.
- **`hfft(x, n)`**은 `irfft(conj(x), n)`에 정규화를 바꾼 것이고(`'backward'`면 `hfft` 쪽이 1/n 없음), **`ihfft(x)`**는 `conj(rfft(x))/n`이다(numpy와 같다). 그래서 `ihfft`의 DC 허수부는 −0이다(numpy와 같고 `[scipy와 다름]`: scipy는 +0이다).
- **길과 결정성**(FT43): 짝수 n은 길이 n/2의 복소 변환과 회전 인자 한 번의 처리, 홀수 n은 길이 n의 복소 변환과 그 에르미트 부분 (y_k + conj y_{n−k})/2다. 길은 길이로만 정해지고, 한 줄의 결과는 배치·레이아웃·스레드 수와 무관하다. `ihfft(x, norm=a)`는 `conj(rfft(x, norm=a′))`, `hfft(x, n, norm=a)`는 `irfft(conj(x), n, norm=a′)`와 비트까지 같다(a′는 방향을 바꾼 `norm`: `'backward'` ↔ `'forward'`).
- **버리는 부분**: `irfft`·`hfft`는 버리는 허수부를 읽지 않으므로 그 자리의 NaN도 결과에 들어가지 않는다(numpy·scipy와 같다).
- **메시지**(FT46): `rfft(): 'x' must be real, not complex128`, `irfft(): the default 'n' = 2 (m - 1) is 0 for an axis of length m = 1; give 'n'`, `irfft(): cannot transform an axis of length 0`(`n`을 주면 0을 채운다), `irfft(): 'n' must be at least 1, not 0`(`[scipy와 다름]`: scipy는 길이 1의 결과를 낸다).
- **다차원**: `rfftn`은 `axes`의 마지막 축에 `rfft`, 나머지 축에 `fft`다. `irfftn`은 마지막 축을 뺀 축에 `ifft`를 한 뒤 마지막 축에 `irfft`다. 이 순서가 에르미트 대칭이 아닌 입력의 어느 부분을 버리는지를 정하므로 numpy·scipy와 같은 순서를 지킨다: 복소 패스 뒤에 마지막 축의 DC와(짝수 n이면) 나이퀴스트 칸의 허수부를 버리고, 마지막 축의 n//2 + 1칸 밖은 읽지 않는다(그 자리의 NaN도 결과에 들어가지 않는다). 마지막 축의 기본 길이는 2(m − 1)이고 s[−1] = −1은 입력 길이 m이다(numpy·scipy와 같다). 기본 길이가 0이면 `irfftn(): the default s[-1] = 2 (m - 1) is 0 for an axis of length m = 1; give 's'`다. `hfft2 ihfft2 hfftn ihfftn`은 scipy의 정의(켤레와 정규화를 바꾼 `irfftn`·`rfftn`)이고 비트까지 그렇다: `hfftn(x, norm=a)`는 `irfftn(conj(x), norm=a′)`, `ihfftn(x, norm=a)`는 `conj(rfftn(x, norm=a′))`다(FT52). 그래서 `ihfftn`의 DC 허수부는 −0이다(`[scipy와 다름]`: scipy는 +0).

## §4. DCT·DST (FT4)
```
dct(x, /, *, type=2, n=None, axis=-1, norm='backward', orthogonalize=None)
idct(x, /, *, type=2, n=None, axis=-1, norm='backward', orthogonalize=None)
dst(x, /, *, type=2, n=None, axis=-1, norm='backward', orthogonalize=None)
idst(x, /, *, type=2, n=None, axis=-1, norm='backward', orthogonalize=None)
dctn(x, /, *, type=2, s=None, axes=None, norm='backward', orthogonalize=None)
idctn(x, /, *, type=2, s=None, axes=None, norm='backward', orthogonalize=None)
dstn(x, /, *, type=2, s=None, axes=None, norm='backward', orthogonalize=None)
idstn(x, /, *, type=2, s=None, axes=None, norm='backward', orthogonalize=None)
```
- 정의는 scipy와 같다(형 1–4). `idct(type=k)`는 DCT의 역(형 1과 4는 자기 자신, 2와 3은 서로의 역)에 정규화를 붙인 것이다.
- **`type`**: 1, 2, 3, 4의 int다. 다른 int는 `ValueError`, int가 아니면 `TypeError`다(`[scipy와 다름]`: scipy는 `type=2.0`에서 바인딩의 시그니처 덤프를 낸다).
- **`orthogonalize`**(FT19): None(기본)은 `norm == 'ortho'`일 때 True다. True면 형 1–3의 첫(과 끝) 칸에 √2를 곱하고 나눠 행렬을 직교로 만든다. 형 4와 DST-I에는 영향이 없다(scipy와 같다).
- **길이의 하한**(FT20): DCT-I은 N ≥ 2다(`dct(): type=1 needs at least 2 points, not 1`; `[scipy와 다름]`: scipy는 DUCC0의 내부 메시지가 새어 나온다). DST-I은 N = 1에서 2x다(scipy와 같다). 다른 형은 N ≥ 1이다.
- **복소 입력**(FT28): 실수부와 허수부를 따로 변환해 complex128로 돌려준다. 그래서 `dct(x)`는 `dct(x.real) + 1j·dct(x.imag)`와 비트까지 같다(scipy는 값으로 같고 비트는 다를 수 있다).
- DST-II·III·IV는 부호를 바꾸고 뒤집은 입력의 같은 형 DCT와 비트까지 같다(docs/design/fft.md §3.5).

## §5. 도우미 (FT1)
```
fftfreq(n, /, *, d=1.0)
rfftfreq(n, /, *, d=1.0)
fftshift(x, /, *, axes=None)
ifftshift(x, /, *, axes=None)
next_fast_len(target, /, *, real=False)
prev_fast_len(target, /, *, real=False)
```
- **`fftfreq`·`rfftfreq`**(FT26): 정수 k = 0, 1, …, −1에 fl(1/fl(n·d))를 곱한 numpy의 계산식 그대로라 numpy와 비트까지 같다. n은 1 이상의 int, d는 0이 아닌 유한한 실수다(d < 0은 받는다; numpy와 같다). n = 0과 d = 0은 `ValueError`다(`fftfreq(): 'n' must be at least 1, not 0`, `fftfreq(): 'd' must be finite and nonzero, not 0.0`; `[numpy와 다름]`: numpy와 scipy는 `ZeroDivisionError`).
- **`fftshift`·`ifftshift`**: 값을 옮기기만 하므로 `sd.roll`과 비트가 같고 dtype을 지킨다. `axes`는 int, tuple, list, None이다. 중복된 축은 `ValueError`다(`fftshift(): repeated axis`; `[numpy와 다름]`: numpy와 scipy는 그 축을 두 번 민다). 빈 `axes`는 복사본이다.
- **`next_fast_len`·`prev_fast_len`**(FT18): scipy와 같은 값이다. 복소(`real=False`)는 11-매끄러운 수(소인수가 모두 11 이하), `real=True`는 5-매끄러운 수다. `next_fast_len`은 `target` 이상의 가장 작은 그런 수, `prev_fast_len`은 이하의 가장 큰 그런 수다. `target`은 1 이상의 int다. 0과 음수는 `ValueError`(`next_fast_len(): 'target' must be at least 1, not 0`; `[scipy와 다름]`: scipy는 0에서 0을 돌려준다), 결과가 int64를 넘으면 `ValueError`(`next_fast_len(): the next fast length after … is beyond int64`)다. int64의 모든 `target`을 받는다(`[scipy와 다름]`: scipy는 약 2⁶¹보다 큰 `target`을 `ValueError`, 2⁶³ 이상을 `OverflowError`로 막는다; FT42). 이 길이들이 정확히 Bluestein을 타지 않는 길이다(FT36). R의 `nextn(n)`(기본 인수 2, 3, 5)은 `next_fast_len(n, real=True)`와 같은 값이다.

## §6. FFTLog (FT5)
```
fht(a, dln, mu, /, *, offset=0.0, bias=0.0)
ifht(A, dln, mu, /, *, offset=0.0, bias=0.0)
fhtoffset(dln, mu, /, *, initial=0.0, bias=0.0)
```
- 정의와 결과는 scipy와 같다(로그 간격 표본의 빠른 Hankel 변환; FT21). `dln`(로그 간격)과 `mu`(Bessel 함수의 차수)는 변환의 필수 인자라 위치로 받는다(FT32). 입력 `a`는 1차원 이상이고 마지막 축을 변환한다(scipy와 같다).
- 세부(복소 `loggamma`의 경계, `bias`의 범위, 에러)는 FT5에서 정해 이 절에 적는다.

## §7. numpy·scipy·R → sd.fft

**numpy.fft**
| numpy | sd.fft | 마일스톤 |
|---|---|---|
| `fft ifft` | 같은 이름(옵션은 키워드만) | FT1 |
| `rfft irfft hfft ihfft` | 같은 이름 | FT2 |
| `fft2 ifft2 fftn ifftn rfft2 irfft2 rfftn irfftn` | 같은 이름 | FT3 |
| `fftfreq rfftfreq fftshift ifftshift` | 같은 이름(`device` 없음) | FT1 |
| `out=` | 없음 | — |
| `norm=None` | 없음(기본이 `'backward'`) | — |

**scipy.fft**
| scipy | sd.fft | 마일스톤 |
|---|---|---|
| numpy와 같은 이름의 변환 | 같은 이름 | FT1–FT3 |
| `hfft2 ihfft2 hfftn ihfftn` | 같은 이름 | FT3 |
| `dct idct dst idst dctn idctn dstn idstn` | 같은 이름(형 1–4, `orthogonalize`) | FT4 |
| `next_fast_len prev_fast_len` | 같은 이름 | FT1 |
| `fht ifht fhtoffset` | 같은 이름(`dln`, `mu`는 위치) | FT5 |
| `workers`, `overwrite_x`, `plan` | 없음 | — |
| `set_workers get_workers` | 없음(스레드 수는 `sd.set_num_threads`) | — |
| `set_backend skip_backend set_global_backend register_backend` | 없음(백엔드는 하나다) | — |
| `scipy.signal.fftconvolve`·`oaconvolve` | 없음(나중의 signal·smooth 영역) | — |

**R**
| R | sd.fft |
|---|---|
| `fft(z)` | `fft(z)` |
| `fft(z, inverse=TRUE)` | `ifft(z, norm='forward')` |
| `mvfft(Z)`, `mvfft(Z, inverse=TRUE)` | `fft(Z, axis=0)`, `ifft(Z, axis=0, norm='forward')` |
| `nextn(n)` | `next_fast_len(n, real=True)` |

## §8. 정확도
- 참값은 Arb다. 계약은 노름 상대오차의 한계다(FT4, FT27): 길이 n의 복소 변환은 ‖ŷ − y‖ ≤ (8·L(n) + 4)·u·‖y‖ + 8·n·2⁻¹⁰⁷⁴이고(L(n) = ⌈log₂ n⌉, u = 2⁻⁵³; 11보다 큰 소인수가 있는 길이는 L(n) 대신 2·L(m) + 2를 쓰고 한계에 ρ(n) ≈ 2를 곱한다, FT37), 실수·에르미트·다차원·DCT·DST는 같은 꼴을 그 변환에 맞춘 것이다(docs/design/fft.md §4.2, FT31). 원소마다의 상대 정확도는 약속하지 않는다: 참값이 0에 가까운 칸은 다른 칸의 크기만큼 움직일 수 있다(docs/design/fft.md §4.1).
- FT1 마일스톤의 `fft`·`ifft`는 저장한 모든 Arb 참값과 긴 줄 넷(2¹⁶, 65537, 3·10007, 2²⁰)에서 계약의 0.054 이하였다(numpy·scipy와 같은 크기; docs/design/fft.md §4.4). FT2 마일스톤의 `rfft`, 실수 입력 `fft`, `irfft`는 저장한 실수·에르미트 참값 전부와 긴 실수 줄 셋(2¹⁶, 65537, 2²⁰)에서 계약의 0.049 이하였다. 실수·에르미트 변환의 안쪽 길이(짝수 n은 n/2, 홀수 n은 n)에 11보다 큰 소인수가 있으면 그 길이의 ρ와 안쪽 길이로 계약을 센다(FT45). 반올림 크기의 결과는 numpy·scipy와 끝자리가 다를 수 있다(API.md §4의 규칙 3).
- FT3 마일스톤의 다차원 변환은 저장한 다차원 참값과 Arb를 그 자리에서 구한 점(Bluestein 축, 짝수·홀수·Bluestein의 실수 마지막 축, 에르미트가 아닌 `irfftn` 입력, 넘침 직전과 비정상 수)에서 계약의 0.045 이하였다. 다차원 계약은 축마다의 항을 더한 꼴이다(FT49: Bluestein 축은 그 ρ, `irfftn`·`hfftn`은 마지막 축에 무게 1·√2를 준 입력의 노름).
- 비트로 지키는 성질: 길이 1의 변환, 2의 거듭제곱 배율(`fft(2·x)`는 `2·fft(x)`), `rfft(x)`와 실수 입력 `fft(x)`의 앞쪽(다차원은 `rfftn`과 `fftn`), 축 하나의 n차원 함수와 1차원 함수, DST와 같은 형 DCT의 관계, `fftshift`와 `roll`, `fftfreq`와 numpy.
