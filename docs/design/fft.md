# 고속 푸리에 변환 영역 설계 (sd.fft)

> 상태: **확정** (2026-11-23). 큰 결정(§1.1 FT1–FT4)과 §9의 18번(FT22)은 사용자가 정했고, 나머지 결정(FT5–FT21, FT23–FT68)은 추천안이다(사용자 위임). FT0–FT7 마일스톤은 완료·승인(위임)이다. 모든 공개 이름(`docs/api/fft.md`의 35개: `fft ifft rfft irfft hfft ihfft fft2 ifft2 fftn ifftn rfft2 irfft2 rfftn irfftn hfft2 ihfft2 hfftn ihfftn dct idct dst idst dctn idctn dstn idstn fht ifht fhtoffset fftfreq rfftfreq fftshift ifftshift next_fast_len prev_fast_len`)이 구현되어 있고 이 문서는 구현과 맞다.

`sd.fft`(이산 푸리에 변환, 실수·에르미트 변환, 이산 코사인·사인 변환, FFTLog와 그 도우미)의 확정 사항과 내부 구조를 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처는 [docs/api/fft.md](../api/fft.md)에 있다.
- 비교 대상은 numpy 2.5.4의 `numpy.fft`(pocketfft의 C++ 판), scipy 1.18.1의 `scipy.fft`(1.18.1부터 DUCC0의 FFT: `scipy/fft/_duccfft`), R 4.6.2의 `stats::fft`·`mvfft`·`nextn`(Singleton의 혼합 기수)이다. R에는 `fftw`, `fftwtools`, `gsignal`, `signal`, `dtt` 패키지가 설치되어 있지 않다. 설치하지 않는다(R과는 복소 변환만 견준다).
  - pocketfft와 DUCC0의 소스는 설치본에 없다(컴파일된 확장과 라이선스 파일만 있다). 두 라이브러리의 알고리즘은 공개된 기술(혼합 기수 Cooley–Tukey, Bluestein, 실수 전용 패스)로만 적고 그 이상은 기대지 않는다.
- 바닥은 배열(`csrc/array/`), 스레드풀(`csrc/parallel/`), 같은 x의 sin·cos(`csrc/base/sincos.hpp`)다. Accelerate의 vDSP DFT(`vDSP_DFT_zop_CreateSetupD`, `vDSP_DFT_zrop_CreateSetupD`, `vDSP_DFT_ExecuteD`)는 쓰지 않는다: FT22의 측정과 마무리한 커널로 다시 잰 측정에서 모든 길이 꼴이 자체 길보다 느렸다(FT30, FT66; §3.1). 그래서 결과가 Accelerate와 무관하다.
- vDSP에 대한 사실은 Xcode의 macOS SDK 헤더 `Accelerate.framework/…/vecLib.framework/Headers/vDSP.h`(Command Line Tools의 `MacOSX26.sdk`와 바이트까지 같다)와 그 헤더로 짠 탐침에서 확인했다(§3.2). Accelerate의 바이너리는 역분석하지 않는다.
- 이 문서의 잰 값은 Apple M4 Pro, macOS 26.6.2에서 본 것이다. 다른 작업과 기기를 나눠 썼다(부하 평균 1.6–2.6).
- ε은 float64의 기계 엡실론 2⁻⁵², u는 단위 반올림 2⁻⁵³(= ε/2), γ_k = ku/(1 − ku), L(n) = ⌈log₂ n⌉(L(1) = 0)이다. ‖·‖는 2-노름이다(spatial.md, sparse.md §4와 같은 기호).

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-23)
| 번호 | 결정 |
|---|---|
| FT1 계산의 바탕 | vDSP가 지원하는 길이는 Accelerate vDSP의 DFT로, 나머지 길이는 자체 C++(NEON, 혼합 기수, 큰 소인수는 Bluestein의 chirp-z)로 계산한다. 두 길 모두 같은 Arb 기반 정확도 계약(FT4)을 지킨다. vDSP가 받는 길이와 변환의 종류는 설치된 SDK의 헤더로 확인해 적는다: 확인한 결과는 §3.1이다(복소 2ᵏ와 f·2ᵏ(f ∈ {3, 5, 15}, k ≥ 3), 실수 짝수 길이 2ᵏ와 f·2ᵏ(k ≥ 4); DCT는 float 판만 있어 쓸 수 없다) (사용자 결정 2026-11-23) |
| FT2 범위 | `numpy.fft`와 `scipy.fft`의 모든 변환이다: `fft ifft rfft irfft hfft ihfft fft2 ifft2 fftn ifftn rfft2 irfft2 rfftn irfftn hfft2 ihfft2 hfftn ihfftn`, `dct idct dst idst`(형 1–4)와 그 n차원 판 `dctn idctn dstn idstn`, `fftfreq rfftfreq fftshift ifftshift next_fast_len prev_fast_len`. `norm`(`'backward'`, `'ortho'`, `'forward'`), `axis`/`axes`, `s`/`n`을 받는다. 지금 넣지 않는 것: `fftconvolve`·`oaconvolve`(나중의 signal·smooth 영역). scipy의 `workers`, `plan`, `overwrite_x`는 §9에서 정한다(라이브러리의 스레드 설정은 `sd.set_num_threads` 하나다) (사용자 결정 2026-11-23) |
| FT3 위치 | 최상위 `sd.fft`, 평평한 이름공간이고 함수마다 이름이 하나다(별칭 없음) (사용자 결정 2026-11-23) |
| FT4 정확도 | 참값은 Arb다. 계약은 c·log₂(n)·u·‖x‖ 꼴의 노름 상대오차 한계이고, 변환마다 맞는 꼴을 유도해 적는다(§4). 원소마다의 상대 정확도는 FFT에서 일반적으로 불가능하다(§4.1에 이유) (사용자 결정 2026-11-23) |
| FT5 받는 입력 | §9의 1번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT6 옵션의 자리 | §9의 2번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT7 `norm`의 기본과 None | §9의 3번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT8 `workers=` | §9의 4번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT9 `overwrite_x=` | §9의 5번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT10 `plan=` | §9의 6번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT11 `out=` | §9의 7번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT12 길이 0 | §9의 8번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT13 중복된 축 | §9의 9번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT14 `axes` 없이 준 `s` | §9의 10번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT15 float32 판 | §9의 11번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT16 실수 입력의 `fft` | §9의 12번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT17 NaN·inf의 무늬 | §9의 13번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT18 `next_fast_len`의 뜻 | §9의 14번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT19 `orthogonalize` | §9의 15번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT20 DCT-I의 N = 1 | §9의 16번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT21 FFTLog(`fht ifht fhtoffset`) | §9의 17번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT22 vDSP가 받지만 자체 길이 더 빠른 길이 | §9의 18번 (a)를 따른다 (사용자 결정 2026-11-23) |
| FT23 빈 `axes` | §9의 19번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT24 계획 캐시의 상한 | §9의 20번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT25 큰 1차원 변환과 다차원 축의 순서 | §9의 21번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT26 `fftfreq`의 계산식 | §9의 22번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT27 계약의 상수 | §9의 23번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT28 복소 입력의 DCT·DST | §9의 24번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| FT29 FT22를 재는 방법 | 길이 꼴(복소·실수 × 2ᵏ, 3·2ᵏ, 5·2ᵏ, 15·2ᵏ)마다 vDSP와 자체 길을 sd의 레이아웃에서 끝까지 잰다: complex128을 받아 complex128을 내고(vDSP는 split으로 나누고 다시 엮는 것을 포함, n ≤ 4096은 `Interleaved` 설정과 둘 중 빠른 쪽), 실수는 float64를 받아 반쪽 스펙트럼 complex128을 낸다. 16 ≤ n ≤ 2²⁰인 그 꼴의 모든 길이를 1스레드로, 세 번 따로 돌린 중앙값으로 잰다. 그 꼴의 길은 시간 비의 기하평균이 작은 쪽이다. FT6 마일스톤은 같은 방법으로 다시 잰다(`tools/fft_probe.py`의 `paths`) (추천안, 사용자 위임 2026-11-23) |
| FT30 길의 표 (FT22의 결과) | 여덟 꼴 모두 자체 길이다(§3.1의 표; vDSP/자체 시간의 기하평균 복소 1.20–1.68, 실수 1.27–1.94). 그래서 첫 판은 vDSP를 부르지 않고 모든 길이가 자체 혼합 기수나 Bluestein이다. 한 (종류, 길이)가 한 길이라는 원칙은 그대로이고 결과가 Accelerate와 무관해진다. vDSP 설정의 메모리(§3.2: 원소마다 11–35바이트)도 계획 캐시에 들지 않는다. `csrc/fft/vdsp.hpp`는 FT6 마일스톤의 다시 재기에만 남긴다 (추천안, 사용자 위임 2026-11-23) |
| FT31 계약의 상수 | κ = 8, c₀ = 4, c₁ = 2, c₂ = 8로 정한다(§4.2). DCT·DST의 안쪽 길이는 모든 형에 M = 2N + 2를 쓴다(가장 긴 안쪽 실수 FFT의 위 한계). 바닥 τ는 2⁻¹⁰⁷⁴·√(출력 원소 수)가 아니라 κ·m·2⁻¹⁰⁷⁴(m은 변환의 길이)다: 비정상 수의 곱은 단마다 절대오차를 더하고 뒤 단이 그것을 키운다(numpy·scipy가 비정상 수 입력의 n = 127에서 2⁻¹⁰⁷⁴·√n의 12.5배를 냈다). 증명은 회전 인자의 오차를 μ ≤ 1.5u로 둔다(FT0 마일스톤의 회전 인자가 n ≤ 2²⁰에서 1.31u). 자체 나비마다의 증명은 FT1 마일스톤에서 코드 주석과 §3.3에 쓴다(FT1 마일스톤이 §4.2와 `csrc/fft/cfft.hpp`의 주석에 썼다) (추천안, 사용자 위임 2026-11-23) |
| FT32 FFTLog의 시그니처 | `fht(a, dln, mu, /, *, offset=0.0, bias=0.0)`, `ifht(A, dln, mu, /, *, ...)`, `fhtoffset(dln, mu, /, *, initial=0.0, bias=0.0)`: `dln`과 `mu`는 변환의 필수 인자라 위치다(`sd.spatial.mahalanobis(u, v, VI)`의 `VI`와 같다; 기본값이 없는 옵션을 키워드로 두지 않는다) (추천안, 사용자 위임 2026-11-23) |
| FT33 참값 파일의 범위 | 각 칸은 출력의 2-노름의 2⁻¹²⁰ 안으로 정하고 double-double로 둔다. 파일에는 n ≤ 2¹⁴의 줄만 둔다(다섯 묶음 약 10 MB). 그보다 긴 줄(2¹⁶, 65537, 3·10007, 2²⁰)은 정확도 체계가 그 자리에서 Arb로 잰다(`acb.dft`는 2¹⁶에서 0.1초다) (추천안, 사용자 위임 2026-11-23) |
| FT34 자체 길의 시제품 | `csrc/fft/`의 헤더(`twiddle.hpp`, `cfft.hpp`, `rfft.hpp`, `vdsp.hpp`)에 두고 바인딩하지 않는다. 복소 FFT는 Stockham 혼합 기수이고 기수 순서는 4, …, 4, (2), 3, …, 5, …다. 입력과 출력은 complex128의 엮은 꼴이고 단 사이는 split 버퍼다(첫 단이 `vld2q`로 풀고 마지막 단이 `vst2q`로 엮어 따로 엮는 단계가 없다). 첫 단은 p를, 나머지 단은 q를 안쪽 고리로 돈다. 실수 FFT는 길이 n/2의 복소 FFT와 회전 인자 한 번이다. FT1 마일스톤은 이 꼴 위에 계획, 일반 소인수, Bluestein을 세운다 (추천안, 사용자 위임 2026-11-23) |
| FT35 혼합 기수의 꼴 | 기수는 4를 되는 대로, 남으면 2, 그다음 3, 5, 7, 11이다(FT34의 순서에 7과 11을 더함). 홀수 기수 p는 모두 한 꼴이다: t_j = a_j + a_{p−j}, d_j = a_j − a_{p−j}로 A_k = a₀ + Σ cos(2πjk/p)·t_j, B_k = Σ sin(2πjk/p)·d_j를 FMA 사슬로 만들고 y_k = A_k − iB_k, y_{p−k} = A_k + iB_k다(상수는 Arb로 구해 한 번 반올림한 값; 시제품의 3·5 나비를 이 꼴로 바꿨다: 곱셈 수가 같고 반올림이 적다). 고리의 길이가 홀수면 마지막 자리를 NEON 레지스터의 두 레인에 같은 값으로 계산해 첫 레인만 쓴다(짝을 이룬 자리와 같은 연산이라 비트가 같다). 역변환은 첫 읽기와 마지막 쓰기에서 실수부와 허수부를 바꾼다(swap(z) = i·conj(z)이므로 swap(DFT(swap(x)))가 부호 +의 DFT다). 그래서 역변환은 정변환과 연산이 같고 `ifft(x, norm='forward')`는 swap(`fft`(swap(x)))와 비트까지 같다. 순서와 꼴은 FT6 마일스톤에서 다시 잰다 (추천안, 사용자 위임 2026-11-23) |
| FT36 Bluestein의 문턱 p₀ | p₀ = 11이다: 소인수가 모두 11 이하인 길이(11-매끄러운 수, `next_fast_len`의 집합)는 혼합 기수, 11보다 큰 소인수가 하나라도 있으면 길이 전체를 Bluestein으로 계산한다. 이유는 증명이다: 기수 13의 단은 log₂13 한 수준마다 8.22u가 들어 κ = 8을 넘는다(기수 11은 7.62u, §4.2). 속도로도 충분하다(소수 1009, 10007, 100003에서 numpy·scipy보다 1.8–3.4배 빠르다, §5) (추천안, 사용자 위임 2026-11-23) |
| FT37 Bluestein의 안쪽 길이와 핵 | 안쪽 길이 m은 2n − 1 이상인 가장 작은 2ᵃ3ᵇ5ᶜ다. chirp wⱼ = e^(−πij²/n)(j² mod 2n은 정확한 정수 연산)과 핵의 변환 β = DFT_m(b)/m(b_l = conj(w_l), \|l\| < n)은 계획을 만들 때 double-double로 계산해 double로 한 번 반올림한다(`csrc/fft/ddmath.hpp`). double로 핵을 변환하면 β의 오차를 u·log m·√m·max\|β\|로만 묶을 수 있어(핵의 2n − 1개 원소가 모두 크기 1이고 β의 원소는 그것들이 상쇄한 합이다) 증명이 무너진다. 잰 ρ(n) = m·max\|βₖ\|/√n은 시험 길이에서 1.90–2.15다. 계획 시간은 65537에서 약 0.1초다 (추천안, 사용자 위임 2026-11-23) |
| FT38 FT1의 실수 입력 | bool·int64·float64 입력의 `fft`·`ifft`는 FT2 마일스톤 전까지 허수부가 0인 복소 입력으로 계산한다(값으로는 같은 변환이다). FT2 마일스톤이 FT16의 실수 길로 바꾸면 그 결과의 끝자리가 바뀌므로 골든 케이스에 실수 입력의 `fft`를 넣지 않는다(FT2 마일스톤이 실수 길로 바꾸고 골든 케이스에 다시 넣었다) (추천안, 사용자 위임 2026-11-23) |
| FT39 회전 인자 표 | 단마다 온전한 표를 `twiddle()`로 만든다(√n 크기의 두 표의 곱은 쓰지 않는다). 계획의 메모리는 원소마다 약 16바이트(2²⁰에서 16.8 MB)라 256 MB의 캐시에 2²⁴ 아래의 길이가 들어간다. 계획들의 표를 Arb와 견준 \|ŵ − w\|는 1.04u 이하였다(`tests/fft/test_proof.py`). 증명은 μ = 1.5u를 그대로 쓴다 (추천안, 사용자 위임 2026-11-23) |
| FT40 축 구동기의 크기 | 줄이 이미 계획이 읽는 꼴(complex128, 변환 축이 마지막이고 연속, 자르기만 있음)이면 입력에서 바로 읽고 결과의 줄에 바로 쓴다(같은 연산이라 비트가 같다). 아니면 이웃한 줄을 32개까지(줄 버퍼 합 2¹⁹ double 이하) 한 묶음으로 모으고 흩는다. 한 태스크는 ⌈2¹²/비용⌉개의 줄(묶음의 배수로 올림)이고 비용은 한 줄의 n·L(n)(Bluestein은 2m·L(m) + 4m)이다. 호출의 일이 2¹⁵ 이상이면 나누고 GIL을 놓는다. 묶음의 크기는 FT6 마일스톤에서 다시 잰다 (추천안, 사용자 위임 2026-11-23) |
| FT41 큰 1차원 변환의 4단계 | FT1 마일스톤에는 넣지 않는다: 한 줄은 한 태스크가 처음부터 끝까지 계산하고, 큰 줄 하나는 1스레드다. 4단계와 문턱 n₄는 FT6 마일스톤이 재서 넣는다(§6의 파일 계획대로). 그때 n ≥ n₄의 결과의 끝자리가 바뀐다 (추천안, 사용자 위임 2026-11-23) |
| FT42 도우미의 경계와 메시지 | `n`·`target`의 0과 음수는 `'n' must be at least 1, not 0`, 변환 축의 길이 0은 `cannot transform an axis of length 0`, `d`는 `'d' must be finite and nonzero, not 0.0`, 축은 배열의 `axes`와 같은 메시지(`repeated axis`, `axis 2 is out of bounds for an array of dimension 2`)다. `next_fast_len`·`prev_fast_len`은 int64의 모든 `target`을 받는다(`[scipy와 다름]`: scipy는 약 2⁶¹보다 큰 `target`을 "too large to perform an FFT"로 막는다). 다음 빠른 길이가 int64를 넘으면 `next_fast_len(): the next fast length after … is beyond int64`(`ValueError`)다 (추천안, 사용자 위임 2026-11-23) |
| FT43 실수 길의 꼴 | 짝수 n은 실수 계획이다: 길이 n/2의 복소 계획과 회전 인자 한 번의 사후 처리(역변환은 그 반대의 사전 처리, §3.4). 홀수 n은 길이 n의 복소 계획을 허수부 0의 줄에 돌리고 결과의 에르미트 부분 (y_k + conj y_{n−k})/2(y_0은 실수부와 +0)를 쓴다. 이유: 사영은 정확한 결과에 대해 축약이라 오차를 늘리지 않고(반올림 한 번이 c₀ 안), 대칭이 정확하고 DC의 허수부가 +0이다. 그래서 짝수·홀수 모두 `rfft(x)`가 `fft(x)`의 앞쪽과 비트까지 같다. 홀수 길이의 실수 전용 나비(두 배 빠를 수 있다)는 FT6 마일스톤의 후보다. 두 줄을 한 복소 줄로 묶는 기법은 한 줄의 비트가 짝에 달려 §3.7을 깨므로 쓰지 않는다 (추천안, 사용자 위임 2026-11-23) |
| FT44 실수 계획의 안쪽 계획 | 실수 계획은 길이 n/2의 복소 계획을 캐시에서 빌리지 않고 스스로 만든다(`Plan`의 `half_`). 한 (종류, 길이)가 캐시의 한 칸이고, 계획을 만드는 동안 캐시의 잠금과 GIL을 다시 잡지 않는다. 같은 길이의 복소 계획이 따로 있으면 메모리가 두 번 들지만 같은 비트다(복소 계획은 길이로만 정해진다). 홀수 실수 길이는 실수 계획이 없고 캐시의 복소 계획을 그대로 쓴다 (추천안, 사용자 위임 2026-11-23) |
| FT45 실수·에르미트 계약의 Bluestein 꼴 | 안쪽 길이 h(짝수 n은 n/2, 홀수 n은 n)가 Bluestein이면 계약은 h의 ρ와 안쪽 길이 m으로 L(n) 대신 2·L(m) + 2(짝수 n은 사후 처리의 몫으로 + 1), 바닥은 τ(max(m, n))다. 실수 입력 `fft`의 전체 스펙트럼은 `rfft` 줄의 계약(같은 꼴, ‖y‖ 대신 s√n‖x‖)을 지킨다: 짝수 n의 사후 처리 오차를 거울에 비춰도 그 한계 안이고(§4.2), 홀수 n은 사영이다. 복소 입력의 Bluestein 꼴(ρ(n), m(n))과는 길이가 달라 두 한계의 크고 작음이 정해지지 않으므로 실수 입력은 실수 길의 한계로 판정한다 (추천안, 사용자 위임 2026-11-23) |
| FT46 실수·에르미트 1차원의 경계 | `rfft`·`ihfft`의 복소 입력은 `TypeError`(`rfft(): 'x' must be real, not complex128`; 다른 영역의 같은 메시지 꼴). `irfft`·`hfft`의 기본 길이가 0이 되는 길이 1의 축은 `ValueError`(`irfft(): the default 'n' = 2 (m - 1) is 0 for an axis of length m = 1; give 'n'`), 길이 0의 축은 `cannot transform an axis of length 0`이다(`n`을 주면 0을 채운다). 버리는 허수부(DC와 짝수 n의 나이퀴스트)는 읽지 않는다: NaN이 있어도 결과에 들어가지 않는다(numpy·scipy와 같다). 실수 입력의 `ifft`는 `conj(fft(x))`에 역방향 인자를 곱한 것과 비트까지 같다. 실수 입력 `fft`의 DC 허수부는 +0이다(numpy와 같고 `[scipy와 다름]`: scipy는 −0) (추천안, 사용자 위임 2026-11-23) |
| FT47 다차원 패스의 꼴 | 축마다 구동기를 한 번, `axes`의 뒤에서 앞으로 돈다(FT25). 첫 패스는 입력을 읽어 새 complex128 배열(C 순서)에 쓰고, 뒤의 패스는 그 축의 길이가 그대로면 앞 패스의 배열을 제자리에서 변환한다(한 줄을 다 모은 뒤 흩으므로 바로 읽기와 같은 비트다; 바로 읽기는 제자리에서 끈다). 길이가 바뀌는 패스만 새 배열을 쓴다. 그래서 중간 배열은 많아야 둘이다. `rfftn`은 첫 패스가 실수 패스(`axes`의 마지막 축의 `rfft`)이고, `irfftn`은 마지막 축을 뺀 축의 복소 패스를 뒤에서 앞으로 한 뒤 실수 패스(`irfft`)가 마지막이다. numpy의 `irfftn`은 복소 패스를 앞에서 뒤로 돌지만 그 순서는 반올림만 바꾸고, 버리는 부분(복소 패스 뒤의 DC·나이퀴스트 허수부, n//2 + 1칸 밖)은 같은 정의다. `irfftn`의 첫 복소 패스는 마지막 축을 n//2 + 1칸으로 잘라(짧으면 0을 채워) 읽는다. 실수 입력의 `fftn`·`ifftn`은 첫 패스가 실수 길이라(FT16) `rfftn(x)`가 `fftn(x)`의 앞쪽 n//2 + 1칸과 비트까지 같다. 결과가 비면 계획을 만들지 않는다(FT2의 `rfft`·`irfft`는 빈 배치에서도 길이 n의 계획부터 만들었다: `rfft(zeros((0, 3)), n=2**40)`이 2⁴⁰의 계획을 시도했다. FT3 마일스톤이 고쳤다) (추천안, 사용자 위임 2026-11-23) |
| FT48 다차원의 정규화 | N은 길이의 곱이다(`irfftn`·`hfftn`은 마지막 축의 출력 길이 n을 센다). 인자 하나(fl(1/N), fl(1/fl(√N)))를 마지막 패스가 곱하고 다른 패스는 곱하지 않는다(1차원과 같은 반올림 한 번). 그래서 축 하나의 n차원 함수는 1차원 함수와 비트까지 같고, `'backward'`의 정변환과 `'forward'`의 역변환(인자 1)은 1차원 변환을 뒤에서 앞으로 이은 것과 비트까지 같다. numpy는 축마다 1/nᵢ을 곱한다(값으로 같고 끝자리가 다를 수 있다) (추천안, 사용자 위임 2026-11-23) |
| FT49 다차원 계약의 꼴 | 패스마다 그 길이의 줄 변환이므로 축의 항을 더한다: 11-매끄러운 축은 κ·L(nᵢ), Bluestein 축은 ρᵢ·κ(2·L(mᵢ) + 2), 실수 축은 안쪽 길이로 FT45의 꼴(ρ·κ(2·L(m) + 2 + [n 짝수])). 상수 항은 maxᵢρᵢ·c₀ + (축 수 − 1)·c₁, 바닥은 τ(N)·Πᵢ(ρᵢ·m′ᵢ/nᵢ)(m′는 Bluestein의 안쪽 길이, 실수 축은 max(m, n); 패스마다의 바닥의 합의 위 한계)다. 축이 하나면 1차원의 꼴과 같다. ‖y‖의 자리는 s·√N·‖x‖이고(`rfftn`의 반쪽은 전체 스펙트럼의 일부다), `irfftn`·`hfftn`은 ‖X‖_W다: 마지막 축의 칸에 무게 1(k = 0과 짝수 n의 k = n/2)이나 √2(나머지)를 준 2-노름이고 허수부를 버리지 않는다(복소 패스가 버리기 전에 오므로; 축이 하나면 1차원의 ‖X̃‖_H). 판정은 `tests/fft/refdata.bound_nd`다 (추천안, 사용자 위임 2026-11-23) |
| FT50 구동기의 크기 (FT40을 고침) | 묶음의 버퍼는 스레드의 scratch다(슬롯마다 2¹⁶ double = 512 KB까지 호출 사이에 남기고, 더 크면 태스크가 할당한다). 줄의 간격은 16 double의 배수에 16을 더해 2의 거듭제곱에서 비킨다. 묶음은 min(2¹⁶/줄 간격, ⌈줄 수/64⌉, 128)줄이고 태스크는 max(⌈2¹²/비용⌉, ⌈줄 수/1024⌉)줄(묶음의 배수로 올림)이다. 모두 크기로만 정하고 한 줄의 비트를 바꾸지 않는다(1차원 골든 케이스의 변화 0). 잰 이유: 태스크마다 버퍼를 할당하면 페이지 폴트를 스레드들이 줄 서서 (64, 64, 64)의 축 1 패스가 10스레드에서 285 µs(1스레드 463 µs)였고 고친 뒤 63 µs다. (2048, 2048)의 축 0 패스는 32 KB 간격의 줄 버퍼가 같은 캐시 집합에 몰려 1스레드 26.2 ms였고 간격을 비킨 뒤 17.6 ms다. 몇 줄짜리 태스크 16,384개가 공유 카운터에 몰려 (256, 256, 256)의 `rfft` 패스가 10스레드에서 3.6 ms였고 태스크 수의 상한 뒤 2.0 ms다 (추천안, 사용자 위임 2026-11-23) |
| FT51 `s`의 꼴과 메시지 | None, int, int의 tuple·list다(`axes`와 같은 꼴; scipy는 int를 받고 numpy는 받지 않는다). 원소는 1 이상이거나 −1(그 축의 입력 길이)이다. bool과 None 원소는 `TypeError`(`fftn(): 's' must be an int or a tuple or list of ints, not NoneType`), 0과 −2 이하는 `ValueError`(`fftn(): the entries of 's' must be at least 1 (or -1), not 0`), int64 밖은 `OverflowError`다. `axes` 없이 준 `s`가 축보다 많으면 `'s' has 3 entries for an array of dimension 2`, 빈 `s`는 `'s' must not be empty`, `axes`와 길이가 다르면 `'s' and 'axes' must have the same length, not 3 and 2`(모두 `ValueError`)다. `irfftn`의 s[−1] = −1은 입력 길이 m이다(numpy·scipy와 같다; 기본값 2(m − 1)이 아니다). 기본 길이가 0이면 `irfftn(): the default s[-1] = 2 (m - 1) is 0 for an axis of length m = 1; give 's'`다 (추천안, 사용자 위임 2026-11-23) |
| FT52 다차원 에르미트의 비트 | `hfftn(x, norm=a)`는 `irfftn(conj(x), norm=a′)`, `ihfftn(x, norm=a)`는 `conj(rfftn(x, norm=a′))`와 비트까지 같다(a′는 방향을 바꾼 `norm`). 첫 복소 패스가 줄을 모으면서 켤레를 취하고(0 채우기 전이라 채운 칸은 +0), 마지막 패스가 결과의 켤레를 취한다. `hfft2`·`ihfft2`는 `axes=(-2, -1)`의 `hfftn`·`ihfftn`이다 (추천안, 사용자 위임 2026-11-23) |
| FT53 DCT·DST 계약의 Bluestein 꼴 | 형마다 안쪽 복소 변환이 하나다: DCT-I은 길이 2(N − 1)의 실수 FFT라 N − 1, DST-I은 N + 1, 형 2·3은 길이 N의 실수 FFT라 짝수 N이면 N/2·홀수 N이면 N, 형 4는 짝수 N이면 길이 N/2의 복소 FFT·홀수 N이면 길이 N의 실수 FFT라 N/2와 N이다(`tests/fft/refdata.r2r_inner`). 그 길이 h가 11-매끄러우면 계약은 §4.2의 κ·L(M) + c₂ 그대로이고, Bluestein이면 h의 ρ와 안쪽 길이 m으로 L(M) 대신 2·L(m) + 3, 한계에 ρ(h)를 곱하고 바닥은 ρ(h)·τ(max(m, M))다(실수 변환의 FT45와 같은 꼴에 실수 패스의 + 1을 모든 형에 똑같이 붙였다). 이유: FT31의 κ·L(M)은 11-매끄러운 안쪽 길이의 증명이라 소수 N(127, 1009)과 2ᵏ의 DCT-I·DST-I(안쪽 2ᵏ ∓ 1)은 그 꼴로 증명이 서지 않는다. 잰 값은 Bluestein 영역에서 계약의 0.012 이하다(§4.4) (추천안, 사용자 위임 2026-11-23) |
| FT54 형마다의 알고리즘과 계획 | §3.5대로다. DCT-I·DST-I은 짝·홀 확장의 실수 FFT(캐시의 실수 종류 2(N ∓ 1)), DCT-II·III은 Makhoul의 순서 바꿈과 길이 N의 실수 FFT, 짝수 N의 DCT-IV는 길이 N/2의 복소 FFT와 앞뒤의 회전 인자 e^(−iπ(8j + 1)/8N) 한 표, 홀수 N의 DCT-IV는 부호를 붙인 순열 위의 길이 N 실수 FFT 하나(§3.5의 지표 χ₁, χ₄; 길이 2N이나 4N의 확장보다 2–4배 적은 일이고, 1155에서 DCT-II와 같은 3.9 µs다). DST-II·III·IV는 같은 형 DCT에 정확한 부호 바꿈과 뒤집기다. 계획은 종류 `Dct2`(형 2·3, DST 2·3이 함께 쓴다)와 `Dct4`(형 4)를 더해 (종류, N)마다 하나이고, 안쪽 계획은 FT44처럼 스스로 만든다(캐시를 빌리지 않는다) (추천안, 사용자 위임 2026-11-23) |
| FT55 `orthogonalize`의 √2 | 입력 칸에 fl(√2)를 곱하고 결과 칸을 fl(√2)로 나눈다(fl(√2)/2를 곱하지 않는다). 정규화 인자와 따로 한 번씩 반올림한다. 이유: DCT-I의 x₀ → y₀처럼 한 칸이 앞뒤 둘을 다 거치면 상수의 오차가 정확히 지워진다. scipy는 fl(√2)·0.5를 곱한다(값으로 같고 끝자리가 다를 수 있다) (추천안, 사용자 위임 2026-11-23) |
| FT56 DCT·DST의 인자 검사와 메시지 | 순서는 시그니처대로 x, `type`, `n`·`s`, `axis`·`axes`, `norm`, `orthogonalize`다. `type`은 int 규칙(API.md §2)이라 bool·float·None은 `TypeError`(`dct(): 'type' must be an int, not float`), int64 밖은 `OverflowError`(`'type' is out of range`), 1–4 밖은 `ValueError`(`dct(): 'type' must be 1, 2, 3 or 4, not 5`; scipy는 "invalid DCT type")다. `orthogonalize`는 True, False, None만이고 그 밖(1, numpy.bool_ 포함)은 `TypeError`(`'orthogonalize' must be True, False or None, not int`; 배열의 `copy=`와 같은 꼴)다. DCT-I의 N = 1은 `n`·`s`로 정한 길이에도 같은 메시지다(`idctn(): type=1 needs at least 2 points, not 1`) (추천안, 사용자 위임 2026-11-23) |
| FT57 DCT·DST의 다차원과 복소 입력 | 축마다 구동기를 한 번, `axes`의 뒤에서 앞으로 돈다. 첫 패스가 입력을 읽고, 길이가 그대로인 가운데 패스는 제자리에서, 마지막 패스는 결과에 쓴다(중간 배열은 많아야 하나). 정규화는 논리 길이 N′의 곱의 인자 하나를 마지막 패스가 곱한다(FT48의 규칙). 그래서 축 하나의 `dctn`은 `dct`와, 인자 1의 `dctn`은 `dct`를 뒤에서 앞으로 이은 것과 비트까지 같다. 복소 입력(FT28)은 실수부·허수부의 float64 보기를 구동기가 바로 읽고 결과의 같은 부분에 쓴다(복사 없음) (추천안, 사용자 위임 2026-11-23) |
| FT58 FFTLog 계수의 계산 (§9의 17번의 "special의 `loggamma` 위에"를 고침) | 계수 U_m = 2^q·Γ(x₊ + iy)/Γ(x₋ − iy)·e^(2iy(ln 2 − offset))의 지수 Z를 fft 안의 double-double log Γ(반사식, 위로 옮기기, Stirling 8항; `csrc/fft/fftlog.hpp`)로 만들고 2πi로 줄인 뒤 지수만 한 번 반올림한다. `sd.special.loggamma`는 쓰지 않는다. 이유: 허수부 2 arg Γ(x + iy)가 2y ln y만큼 커서(n = 4096, dln = 0.01이면 y = 157에서 약 1600) double로 구한 Z는 반 ulp만 틀려도 계수에 u의 수백 배가 되고, special의 계약은 \|log Γ\|에 대한 상대 1e-14다(같은 자리에서 약 6×10⁴u). double-double 계수는 Arb와 견주어 u 이하(잰 최대 0.98u, 부분마다 한 번의 반올림)이고, scipy는 이 영역 계약의 4,600배까지 틀린다(§4.4). 계수 계산은 1스레드에서 계수 하나에 약 0.25 µs(q = 0)·0.45 µs(q ≠ 0)다 (추천안, 사용자 위임 2026-11-23) |
| FT59 FFTLog의 계획과 캐시 | 계수와 바이어스(키: n, `dln`, `mu`, `offset`, `bias`의 비트, 방향)를 계획으로 두고 전용 캐시(LRU, 16개·128 MB)에 둔다. Hamilton의 FFTLog가 `fhti`의 표를 두는 것과 같다. 계수는 m마다 따로 계산하므로 풀에서 나눠도 비트가 같다. 이유: 같은 매개변수로 거듭 부르는 쓰임에서 호출은 실수 FFT 두 번과 곱 한 번이다(2¹²에서 11 µs, scipy 150 µs). 새 계획은 1스레드 2¹²에서 0.48 ms(scipy 0.15 ms), 10스레드 75 µs다 (추천안, 사용자 위임 2026-11-23) |
| FT60 특이한 변환과 극점 | x₊ 하나만 Γ의 극점(0, −1, …)이면 U₀ = ∞라 `fht`가, x₋ 하나만이면 U₀ = 0이라 `ifht`가 특이하다. 두 경우 모두 그 계수를 0으로 둔다(scipy의 고침과 같은 값). 경고는 내지 않는다(docs/api/fft.md §1.4; `[scipy와 다름]`: scipy는 경고한다). 둘 다 극점이면 극한 2^q(−1)^(j+k)k!/j!(x₊ = −j, x₋ = −k)이다(scipy의 `poch`와 같은 값). 극점은 정확한 (mu + 1 ± q)/2로 가린다(scipy는 double로 반올림한 값으로 가린다; 그 둘이 다른 입력만 다르다) (추천안, 사용자 위임 2026-11-23) |
| FT61 FFTLog의 인자 검사와 메시지 | 순서는 시그니처대로 데이터(`fht`는 `a`, `ifht`는 `A`), `dln`, `mu`, `offset`(`fhtoffset`은 `initial`), `bias`다. 데이터는 `sd.ndarray`이고 bool·int64·float64(float64로 바꾼다)만 받는다. complex128은 `TypeError`(`fht(): 'a' must be real, not complex128`; scipy도 `TypeError`)다. 실수 인자는 int·float·numpy 실수 스칼라이고(bool은 `TypeError`), `dln`은 유한한 0 아닌 값(`ValueError: 'dln' must be finite and nonzero, not 0.0`; `[scipy와 다름]`: scipy는 경고와 NaN), 나머지는 유한한 값(`'mu' must be finite, not nan`)이다. 마지막 축의 길이 0은 `ValueError`, 다른 축의 0은 빈 결과다. `fhtoffset`은 Python float를 돌려준다(scipy는 numpy.float64) (추천안, 사용자 위임 2026-11-23) |
| FT62 FFTLog의 계약 | ‖Â − A‖ ≤ (2ε_F + 8)·u·‖b_out‖_∞·M·‖B_in a‖ + (‖b_out‖_∞(M + 1) + 1)·2τ_F다. ε_F·τ_F는 길이 n 실수 변환의 수준 항과 바닥(FT45; 안쪽 길이가 Bluestein이면 그 꼴), M = max\|U_m\|(`ifht`는 max 1/\|U_m\|; 실수 칸은 Re U_m), B_in·b_out은 입력·출력 바이어스다. q = 0이면 M = 1, 바이어스가 없어 노름 상대오차 (2ε_F + 8)u다. 정의역은 넘침이 없고 (\|mu\| + \|q\| + 1)/2, π/(2\|dln\|), \|ln 2 − offset\|/(2\|dln\|)가 2³⁶ 이하인 곳이다(double-double의 오차가 그 안에서 2⁻⁶⁰ 아래). 이유: 두 실수 변환의 계약을 대각 사상과 이은 것이 증명이고(§4.2), 바이어스는 문제 자체의 조건수라 한계에 그대로 둔다 (추천안, 사용자 위임 2026-11-23) |
| FT63 FFTLog의 실수 칸과 바이어스의 반올림 | m = 0과 (짝수 n의) m = n/2는 scipy처럼 U의 허수부를 먼저 버린다: `fht`는 Re U_m/n, `ifht`는 1/(n·Re U_m)이다(1/(n conj U_m)의 실수부가 아니다; Re U_m이 0에 가까우면 둘이 크게 다르다). 바이어스 e^(sk + c)의 지수는 정확한 곱 q·dln, q·offset의 double-double이고 Tang의 표(2^(i/64))와 6차 다항식으로 한 번 반올림한다(상대 u/2 + 2⁻⁵⁹). 1/n은 계수에 넣어 줄마다 곱을 하나 줄인다 (추천안, 사용자 위임 2026-11-23) |
| FT64 기수 13–31과 Bluestein의 문턱 (FT36의 p₀ = 11을 고침) | 혼합 기수의 기수에 13, 17, 19, 23, 29, 31을 더한다(FT35의 홀수 꼴 그대로, 상수 cos·sin(2πm/p)는 Arb로 구해 한 번 반올림). 그래서 Bluestein은 31보다 큰 소인수가 있는 길이만이다(p₀ = 31). 이런 기수가 하나라도 있는 길이(안쪽 길이)의 계약은 κ = 8 대신 κ = 17이고 바닥도 17·m·2⁻¹⁰⁷⁴다(그 단은 log₂ r 한 수준마다 8.22u(13)–16.83u(31), `tests/fft/proof.py`). 계약을 넓히는 것이 아니다: 그 길이들은 Bluestein 꼴(ρ(2L(m) + 2)κ, 잰 ρ ≥ 1.9)을 지키던 곳이고 새 꼴이 모든 길이에서 더 좁다(n = 1023에서 약 390u가 174u; ρ = 1로 쳐도 L < 32이면 좁다). 속도는 2–5배다(1001에서 10.7 → 3.0 µs, 4095에서 58.9 → 12.0 µs, 32736 = 3·11·31·32에서 576 → 191 µs, (34, 1009)의 축 0 패스(길이 34)에서 267 → 73 µs; 모두 scipy보다 빠르다). 37 이상은 Bluestein으로 둔다: 일반 기수의 일은 p에 비례해 늘고 한 수준의 오차도 19.9u(37)부터 κ = 17을 넘는다. `next_fast_len`의 집합(11-매끄러운 수, FT18)은 그대로다(가장 싼 기수만 쓰는 길이). 바뀌는 것: 그런 길이의 결과 끝자리(골든 케이스) (추천안, 사용자 위임 2026-11-23) |
| FT65 4단계 (FT41) | 소인수가 31 이하인 길이 n ≥ n₄ = 2¹⁵의 복소 변환은 4단계다: n = n₁·n₂(n₁은 √n 이하의 가장 큰 약수; n으로만 정함), 열 16개씩 모아 길이 n₁의 변환과 회전 인자 w_n^(j₂k₁)(온전한 표 하나, FT39; j₂k₁ = 0이면 곱하지 않음)을 거쳐 열 묶음 단위로 작업 버퍼에 쓰고, 행 16개씩 모아 길이 n₂의 변환 뒤 결과에 흩는다. 회전 인자 표와 작업 버퍼를 열 묶음 단위로 놓아 열 패스의 읽기·쓰기가 차례대로다(행 우선이면 2²⁰의 회전 인자 곱이 2.7 ms, 묶음 단위면 0.4 ms). 묶음은 풀의 태스크라 긴 줄 하나가 모든 스레드에서 돌고, 분해와 연산이 n으로만 정해지므로 스레드 수와 무관한 비트다. 같은 길은 그 길이를 안쪽에 쓰는 모든 변환(실수 n ≥ 2¹⁶, DCT·DST, Bluestein의 안쪽 길이, FFTLog)에 그대로 간다. 긴 줄의 O(n) 패스(실수 처리, Bluestein의 chirp 곱, 정규화, 켤레·거울, DCT의 앞뒤 처리, FFTLog의 곱)는 2¹⁴개씩 풀에서 나눈다(칸마다 따로 계산하므로 비트가 같다). 그런 줄이 여럿이면 차례대로 하나씩 돈다. 계약은 그대로다(한 수준에 나비 한 단과 회전 인자 곱 하나라는 셈이 같다; 기수의 순서만 n₁의 것, n₂의 것이다). 잰 값(복소 한 줄, 같은 길이의 직접 Stockham에 대한 속도 비): 1스레드 2¹⁴ 0.76, 2¹⁵ 0.96, 2¹⁶ 0.91, 2¹⁷ 1.05, 2¹⁸ 0.99, 2²⁰ 0.83, 2²² 0.78, 3·2¹⁶·5·2¹⁵·15·2¹⁴ 0.97–0.98; 10스레드 2¹⁴ 2.8, 2¹⁵ 4.5, 2¹⁶–2²² 4.8–7.3. 그래서 n₄ = 2¹⁵이다(그 아래는 1스레드에서 잃는 것이 크다). Bluestein의 5-매끄러운 안쪽 길이(131220, 202500)는 직접 Stockham이 2의 거듭제곱 간격의 캐시 충돌이 없어 유난히 빨라 1스레드 0.63–0.68배지만 10스레드에서 2.4–3.9배다. 묶음 크기(열 8·16·32·64, 행 16·32)는 열 16·행 16이 10스레드에서 가장 좋았고 1스레드에서 4% 안이었다. 바뀌는 것: n ≥ 2¹⁵인 줄의 끝자리(FT41이 예고한 대로) (추천안, 사용자 위임 2026-11-23) |
| FT66 길의 표 다시 재기 (FT22, FT29의 방법) | FT6 마일스톤의 커널(4단계 포함)로 다시 재도 여덟 꼴 모두 자체 길이다(vDSP/자체 시간의 기하평균 복소 2ᵏ 1.09, 3·2ᵏ 1.43, 5·2ᵏ 1.58, 15·2ᵏ 1.52, 실수 2ᵏ 1.17, 3·2ᵏ 1.84, 5·2ᵏ 1.78, 15·2ᵏ 1.78; 기록 `tests/fft/data/prototype.json`의 `paths/…`). vDSP는 계속 부르지 않고 `csrc/fft/vdsp.hpp`는 탐침에만 남는다. 꼴 안에서 vDSP가 빠른 길이는 짧은 것(복소 16·32·128·8192, 실수 16·16384)뿐이고 꼴 단위로만 옮기므로 그대로다 (추천안, 사용자 위임 2026-11-23) |
| FT67 마지막 축이 아닌 축의 패스 (FT50에 더함) | 복소 입력의 줄이 마지막 축이 아니면 (a) 위치 간격이 L1 캐시에 머무는 줄(n ≤ 8·16384/max(gcd(간격, 16384), 128))은 두 줄 커널로 바로 변환한다: 레지스터의 두 레인이 두 줄의 같은 위치이고 단마다 한 줄 커널이 그 위치에 하는 연산을 그대로 하므로(첫 단의 p = 0 회전 인자 1과 뒤 단의 p = 0 건너뛰기까지) 비트가 같고, 모으고 흩는 복사가 없다; 태스크는 8줄 이상(이웃한 줄이 128바이트 캐시 줄을 나누므로). (b) 그 밖의 줄(2의 큰 거듭제곱의 배수 간격)은 지금처럼 묶음으로 모으고 흩되, 복사를 128바이트 타일(복소 8칸, 실수 16칸) 단위로 한다. 모두 레이아웃으로 고르지만 비트가 같은 길이라 FT1의 원칙(한 길이는 한 길)에 어긋나지 않는다. 잰 값(1스레드): (256, 256, 129)의 축 1 패스 22.9 → 16.5 ms(두 줄), 축 0 패스 23.3 → 18.1 ms(타일), (1024, 1000)의 축 0 2.66 → 2.15 ms(두 줄), (1000, 1024)의 축 0 3.04 → 2.84 ms(타일), (64, 64, 64)의 축 1 500 → 323 µs; 10스레드는 같거나 빠르고 (256, 256, 129)의 축 1만 3.1 → 3.4 ms다. 타일을 두 배 넘게 키우면 16 KB 간격에서 두 배 느려진다 (추천안, 사용자 위임 2026-11-23) |
| FT68 재 보고 넣지 않은 것과 그대로 둔 상수 | (a) 실수 사후·사전 처리를 두 자리씩 split 레지스터로: 따로 잰 패스는 1024에서 0.50 → 0.26 µs였지만 변환 전체에서 차이가 잡음 안이라 넣지 않는다(FT2와 같은 결론). (b) 홀수 길이의 실수 전용 나비(FT43의 후보): 넣지 않는다. 홀수 실수 길이의 느린 곳은 Bluestein 길이였고(999, 1001, 4095가 scipy의 0.25–0.65배) FT64가 그 대부분을 고쳤다(1001·4095는 scipy보다 빠르다); 실수 기수 나비 전체는 일이 크고 홀수 11-매끄러운 길이는 이미 numpy의 1.9배다. (c) DCT-I의 다른 꼴: 길이 N − 1의 실수 FFT로 하는 FFTPACK의 방법은 수치적으로 불안정하다고 알려져(FFTW가 그래서 길이 2(N − 1)을 쓴다) 계약을 지킬 수 없다. 2(N − 1)은 DCT-I의 주기라 바꿀 수 없고, 개선은 FT64(1023 = 3·11·31)와 FT65(65535의 Bluestein 안쪽 131072가 4단계)에서 온다. (d) Bluestein의 안쪽 길이(2ᵃ3ᵇ5ᶜ, FT37), 줄 묶음(128줄, 줄 수의 1/64, 512 KB), 태스크 크기, 계획 캐시(32개·256 MB)와 FFTLog 캐시(16개·128 MB)는 그대로다(묶음 크기를 1/16·256 KB·128 KB로 바꿔 보면 같거나 느렸다). (e) FFTLog의 새 계획(1스레드 scipy의 2.8–5배 느림): double-double log Γ(FT58)가 계수 하나에 약 0.2 µs이고 정확도를 위한 것이라 그대로 둔다(10스레드는 scipy보다 2배 빠르고, 계획이 있으면 6.7–14배 빠르다). (f) 작은 n의 호출 비용(`fft` 16에서 0.20 µs, numpy의 8배)은 그대로다. 기수 13–31을 더할 때 단 함수를 제네릭 람다로 고르면 작은 n이 20% 느려져 `switch`로 둔다 (추천안, 사용자 위임 2026-11-23) |

### 1.2 범위
마일스톤은 §8이다. 아래 표의 "FTn"은 그 마일스톤에서 만들었다는 뜻이고 "나중"은 첫 판에 넣지 않은 후보다.

**이름공간**: 모든 함수를 `sd.fft`에 바로 둔다. 하위 모듈이 없다. 이름이 겹치지 않는다(scipy.fft의 이름 그대로).

**numpy.fft → sd.fft**
| numpy | sd.fft | 마일스톤 |
|---|---|---|
| `fft ifft` | 같은 이름 (옵션은 키워드만, §9의 2번) | FT1 |
| `rfft irfft hfft ihfft` | 같은 이름 | FT2 |
| `fft2 ifft2 fftn ifftn rfft2 irfft2 rfftn irfftn` | 같은 이름 | FT3 |
| `fftfreq rfftfreq fftshift ifftshift` | 같은 이름 (`device` 없음) | FT1 |
| `out=` | 없음 (§9의 7번) | — |
| `norm=None` | 없음: 기본이 `'backward'`이고 None은 `TypeError` (§9의 3번) | — |

**scipy.fft → sd.fft**
| scipy | sd.fft | 마일스톤 |
|---|---|---|
| numpy와 같은 이름의 변환 | 같은 이름 | FT1–FT3 |
| `hfft2 ihfft2 hfftn ihfftn` | 같은 이름 | FT3 |
| `dct idct dst idst dctn idctn dstn idstn` (형 1–4, `orthogonalize`) | 같은 이름 | FT4 |
| `next_fast_len prev_fast_len` | 같은 이름 (§9의 14번) | FT1 |
| `workers`, `overwrite_x`, `plan` | 없음 (§9의 4–6번) | — |
| `set_workers get_workers` | 없음: 스레드 수는 `sd.set_num_threads` 하나다 | — |
| `set_backend skip_backend set_global_backend register_backend` | 없음: 백엔드는 하나다 | — |
| `fht ifht fhtoffset` (FFTLog) | §9의 17번, FT58–FT63 | FT5 |
| `fftconvolve`·`oaconvolve` (`scipy.signal`) | 없음, 나중 (signal·smooth 영역, FT2) | 나중 |

**R → sd.fft**
| R | sd.fft |
|---|---|
| `fft(z)` | `fft(z)` |
| `fft(z, inverse=TRUE)` | `ifft(z, norm='forward')` (R의 역변환은 1/n을 곱하지 않는다) |
| `mvfft(Z)`, `mvfft(Z, inverse=TRUE)` | `fft(Z, axis=0)`, `ifft(Z, axis=0, norm='forward')` |
| `nextn(n)`(기본 인수 2, 3, 5) | `next_fast_len(n, real=True)`와 같은 집합(5-매끄러운 수, §9의 14번) |

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 데이터는 위치로, 옵션은 키워드로만 받는다(§9의 2번). 별칭이 없다.
- **같은 기계, 같은 스레드 수, 같은 입력이면 실행마다 같은 비트다**(API.md §3). 스레드 수와 무관한 비트는 약속이 아니지만 이 영역은 지킨다: 분할은 크기로만 정하고, 한 줄(변환 축을 따라 놓인 원소들)의 결과는 그 줄의 값과 변환의 종류·길이·`norm`으로만 정해진다(§3.7). 그래서 배치의 모양, 메모리 레이아웃, 스레드 수, Accelerate의 스레드 모드와도 무관하다.
- **한 (종류, 길이)는 한 길로만 계산한다.** 길(혼합 기수, 4단계, Bluestein, 실수 길)마다 알고리즘이 달라 끝자리가 다르다. 그래서 어느 길로 갈지는 변환의 종류와 길이로만 정하고(§3.1), 정렬, stride, 배치 크기, 스레드 수, 값으로 고르지 않는다. 같은 길이가 두 길로 가는 일이 없다. 레이아웃으로 고르는 것(두 줄 커널, 묶음 복사)은 비트가 같은 실행 방식뿐이다(FT67).
- 반올림이 아닌 동작(결과의 모양과 dtype, 자르기와 0 채우기, 에르미트 입력에서 버리는 부분, 정규화 인자, 기본값)은 numpy 2.5.4·scipy 1.18.1과 같다(API.md §4의 규칙 2). 둘이 다르거나 sd의 공통 규칙과 맞지 않는 곳은 `[numpy와 다름]`, `[scipy와 다름]`으로 이유와 함께 적는다(§2.9). 반올림 크기의 결과는 정확도 계약(§4) 안이면 된다.
- 성능이 먼저이고 정확도가 그다음이다(API.md §4). FFT의 오차는 길이의 로그로만 자라므로 빠른 알고리즘과 계약이 부딪히지 않는다. 계약을 늦춰야 하는 길(예: 회전 인자를 덜 정확하게 만드는 빠른 길)은 이득과 손실을 재서 §1.1에 적는다.
- 경고를 내지 않는다. numpy가 `RuntimeWarning`을 내는 유한하지 않은 입력(§2.8)도 조용히 IEEE 결과를 낸다.
- 에러 종류와 메시지는 API.md §2·§2.1이다. 메시지는 부른 함수의 이름으로 시작한다(`fft(): ...`, `dctn(): ...`).
- 참값은 Arb(python-flint 0.9.0의 `acb.dft`와 구간 산술)와 정확한 유리수만 쓴다. mpmath는 쓰지 않는다.
- 모든 호출은 정해진 시간 안에 끝난다. 반복이 없는 영역이라 작업량은 입력과 결과의 크기로 정해진다(소수 길이도 Bluestein으로 O(n log n)).

### 1.4 다른 영역과의 관계
- **배열**: `fftshift`·`ifftshift`는 배열의 `roll`과 같은 결과다(비트가 같다: 값을 옮기기만 한다). dtype의 승격(bool·int64 → float64 → complex128)은 배열의 규칙이다.
- **linalg**: 쓰지 않는다. FFT는 BLAS를 부르지 않으므로 결과가 Accelerate의 스레드 모드(`VECLIB_MAXIMUM_THREADS`)와 무관하다(vDSP의 DFT는 안에서 스레드를 쓰지 않는다, §3.2).
- **special**: 쓰지 않는다. FFTLog의 계수는 fft 안의 double-double log Γ로 만든다(FT58: special의 복소 `loggamma`는 \|log Γ\|에 대한 상대 계약이라 큰 위상의 계수에 모자란다).
- **나중의 영역**: signal·smooth의 `fftconvolve`·`oaconvolve`, 보간, 스펙트럼 추정이 이 영역의 계획(§3.6)과 축 구동기(§3.7)를 C++에서 직접 쓴다. 약속이 아니라 메모다. 그래서 계획과 구동기는 공개 함수와 떨어진 헤더로 둔다(§6).

---

## 2. API
시그니처는 §9의 답(옵션은 키워드만, `norm`의 기본은 `'backward'`, `workers`·`overwrite_x`·`plan`·`out` 없음)이고 [docs/api/fft.md](../api/fft.md)와 같다.

### 2.1 입력과 dtype
- **데이터** `x`는 `sd.ndarray`만 받는다(§9의 1번). 리스트와 numpy 배열은 `TypeError`다(`fft(): 'x' must be an sd.ndarray, not list`). 0차원 배열은 sd에 없으므로(array.md §1.2) 입력은 언제나 1차원 이상이다.
- **계산의 dtype은 float64와 complex128뿐이다.** float32 판은 없다(sd에 float32 dtype이 없다, API.md §1; §9의 11번). numpy가 float32에서 complex64를 내는 것과 달리 sd의 결과는 언제나 complex128·float64다.

| 함수 | bool·int64·float64 입력 | complex128 입력 | 결과 dtype |
|---|---|---|---|
| `fft ifft fft2 ifft2 fftn ifftn` | float64로 바꿔 실수 입력의 길(§9의 12번) | 그대로 | complex128 |
| `rfft rfft2 rfftn ihfft ihfft2 ihfftn` | float64로 바꾼다 | `TypeError` (numpy·scipy와 같다; scipy의 메시지 뜻은 "x must be a real sequence") | complex128 |
| `irfft irfft2 irfftn hfft hfft2 hfftn` | 허수부가 0인 complex128로 본다 | 그대로 | float64 |
| `dct idct dst idst dctn idctn dstn idstn` | float64로 바꾼다 | 실수부와 허수부를 따로 변환한다(scipy와 같다) | 입력이 complex128이면 complex128, 아니면 float64 |
| `fftshift ifftshift` | dtype 그대로(bool, int64, float64) | 그대로 | 입력과 같다 |
| `fftfreq rfftfreq` | — | — | float64 |
| `next_fast_len prev_fast_len` | — | — | Python `int` |

- int64를 float64로 바꾸는 것은 배열의 `astype`과 같다(2⁵³을 넘는 정수는 반올림한다; numpy와 같다).
- **int 인자**(`n`, `s`의 원소, `axis`, `axes`의 원소, `type`, `target`)는 API.md §2의 규칙이다: `operator.index`로 읽고 bool은 `TypeError`(`[scipy와 다름]`: scipy는 `n=True`를 1로 받는다), int64 밖은 `OverflowError`다.
- 결과는 언제나 새 배열이고 C 순서다. 입력을 고치지 않는다.

### 2.2 복소 변환
```
fft(x, /, *, n=None, axis=-1, norm='backward')
ifft(x, /, *, n=None, axis=-1, norm='backward')
fft2(x, /, *, s=None, axes=(-2, -1), norm='backward')
ifft2(x, /, *, s=None, axes=(-2, -1), norm='backward')
fftn(x, /, *, s=None, axes=None, norm='backward')
ifftn(x, /, *, s=None, axes=None, norm='backward')
```
- 정의는 numpy와 같다: `fft`는 Y_k = Σⱼ xⱼ·e^(−2πijk/n), `ifft`는 부호가 +이고, 정규화는 §2.5다.
- 다차원 판은 `axes`의 축마다 1차원 변환을 거듭한 것이다. 축의 순서는 수학적으로 결과를 바꾸지 않지만 반올림을 바꾼다. 계산 순서는 `axes`의 뒤에서 앞으로다(numpy와 같은 순서, 레이아웃과 무관하게 고정; §9의 21번).
- `fft2`는 `fftn`의 `axes` 기본값만 다르다(두 축). 1차원 입력의 `fft2`는 `ValueError`다(축이 범위 밖; numpy는 `IndexError`).

### 2.3 실수와 에르미트 변환
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
- `rfft`: 실수 x의 `fft` 가운데 앞 n//2 + 1개다. DC와(짝수 n이면) 나이퀴스트 칸의 허수부는 정확히 +0이다(numpy·scipy와 같다).
- `irfft`: 결과 길이 n의 기본값은 2(m − 1)이다(m은 입력의 그 축 길이; numpy·scipy와 같다). m = 1이면 기본 n이 0이라 `ValueError`다. 입력에서 n//2 + 1개만 쓰고(길면 자르고 짧으면 0을 채운다), 쓰는 칸 가운데 DC의 허수부와(짝수 n이면) 나이퀴스트 칸의 허수부는 버린다(numpy·scipy와 같다; 에르미트 확장에 그 칸의 짝이 없다). 홀수 n은 명시해야 한다.
- `hfft(x, n)`은 `irfft(conj(x), n)`에 정규화를 바꾼 것이고(`'backward'`면 hfft 쪽이 1/n 없음), `ihfft(x)`는 `conj(rfft(x))/n`이다(numpy와 같다). 그래서 `ihfft`의 DC 허수부는 −0이다(numpy와 같고 `[scipy와 다름]`: scipy는 +0이다).
- **다차원**: `rfftn`은 `axes`의 마지막 축에 `rfft`, 나머지 축에 `fft`다. `irfftn`은 마지막 축을 뺀 축에 `ifft`를 한 뒤 마지막 축에 `irfft`다. 이 순서는 반올림만의 문제가 아니다: `irfftn`에서 에르미트 대칭이 아닌 입력의 어느 부분이 버려지는지가 이 순서로 정해지므로 numpy·scipy와 같은 순서(다른 축 먼저, 마지막 축의 c2r이 마지막)를 지킨다. 마지막 축의 기본 길이는 2(m − 1)이다. `hfftn`·`ihfftn`은 scipy의 정의(켤레와 정규화를 바꾼 `irfftn`·`rfftn`)다.

### 2.4 축과 길이 (`n`, `s`, `axes`)
- **`axis`**: int 하나, 음수는 뒤에서 센다. 범위 밖은 `ValueError`다(`[numpy와 다름]`: numpy와 scipy는 `IndexError: tuple index out of range`. API.md §2는 범위 밖의 축을 `ValueError`로 정하고 배열의 모든 함수가 그렇다).
- **`n`**: None이면 입력의 그 축 길이(`irfft`·`hfft`는 2(m − 1)). 주면 1 이상의 int다. 입력보다 짧으면 앞에서 자르고 길면 끝에 0을 채운다(numpy·scipy와 같다). 0과 음수는 `ValueError`다(§2.8, numpy·scipy와 같다; `[scipy와 다름]`: scipy 1.18.1의 `irfft(x, n=0)`은 에러 없이 길이 1의 결과를 낸다).
- **`axes`**: None이면 모든 축(`fft2` 계열은 (−2, −1)). int의 tuple이나 list를 받는다(DESIGN §7: 배열의 `axes`는 리스트도 받는다). 중복된 축은 `ValueError`다(§9의 9번; scipy와 같고 `[numpy와 다름]`: numpy는 같은 축을 두 번 변환한다). 빈 `axes`는 §9의 19번.
- **`s`**: None이면 각 축의 입력 길이(`irfftn`의 마지막 축은 2(m − 1)). 주면 원소가 1 이상의 int이거나 −1(그 축의 입력 길이를 그대로 쓴다; numpy·scipy가 함께 받는다)인 tuple이나 list, 또는 int 하나다(FT51). numpy가 폐기 예고한 None 원소는 받지 않는다(`TypeError`; scipy도 받지 않는다). `axes`를 함께 주면 길이가 같아야 한다(`ValueError`). `axes` 없이 주면 마지막 len(s)개 축이다(§9의 10번).

### 2.5 정규화 `norm`
| `norm` | 정변환(`fft`, `rfft`, `ihfft`, `dct`, …) | 역변환(`ifft`, `irfft`, `hfft`, `idct`, …) |
|---|---|---|
| `'backward'` (기본) | 1 | 1/N |
| `'ortho'` | 1/√N | 1/√N |
| `'forward'` | 1/N | 1 |

- N은 변환의 길이다: 복소·실수 변환은 결과 쪽 길이 n(다차원은 `s`의 곱), DCT·DST는 scipy의 "논리 길이"(DCT-I은 2(N − 1), DST-I은 2(N + 1), 그 밖은 2N)다. `hfft`·`ihfft`는 numpy처럼 방향을 거꾸로 센다(`hfft`가 역변환 쪽).
- 값은 문자열 셋 가운데 하나다. None은 `TypeError`, 다른 문자열은 `ValueError`(`fft(): 'norm' must be 'backward', 'ortho' or 'forward', not 'bad'`; API.md §2.1의 문자열 옵션 꼴)다(§9의 3번).
- 정규화는 결과에 인자 하나를 곱한다(1/√N은 fl(1/fl(√N))이다; numpy의 `'ortho'`도 곱한다). 인자가 2의 거듭제곱이면 정확하다.

### 2.6 DCT·DST
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
- 정의는 scipy와 같다(형 1–4, scipy 문서 `_realtransforms.py`의 식). `idct(type=k)`는 DCT의 역(형 1, 4는 자기 자신, 2와 3은 서로의 역)에 정규화를 붙인 것이다.
- `type`은 1, 2, 3, 4의 int다. 다른 int는 `ValueError`(scipy의 "invalid DCT type"), int가 아니면 `TypeError`다(`[scipy와 다름]`: scipy는 `type=2.0`에서 바인딩의 시그니처 덤프를 낸다).
- `orthogonalize`: None(기본)은 `norm == 'ortho'`일 때 True다. True면 형 1–3의 첫(와 끝) 칸에 √2를 곱하고 나눠 행렬을 직교로 만든다. 형 4와 DST-I은 영향이 없다(scipy와 같다; §9의 15번).
- **길이의 하한**: DCT-I은 N ≥ 2다(정의가 N − 1로 나눈다). N = 1은 `ValueError`다(§9의 16번; `[scipy와 다름]`: scipy 1.18.1은 `dct(x, type=1)`·`idct(x, type=1)`에서 DUCC0의 내부 단언이 `RuntimeError: ... no zero-sized FFTs`로 새어 나온다). DST-I은 N = 1에서 2x다(scipy와 같다). 다른 형은 N ≥ 1이다.
- **복소 입력**(FT28): 실수부와 허수부를 따로 변환하므로 `dct(x)`는 `dct(x.real) + 1j·dct(x.imag)`와 비트까지 같다. scipy의 결과는 그것과 값으로 같지만 비트는 다르다(`tests/fft/data/scipy.json`의 `dct/complex is parts`).

### 2.7 도우미
```
fftfreq(n, /, *, d=1.0)
rfftfreq(n, /, *, d=1.0)
fftshift(x, /, *, axes=None)
ifftshift(x, /, *, axes=None)
next_fast_len(target, /, *, real=False)
prev_fast_len(target, /, *, real=False)
```
- **`fftfreq`·`rfftfreq`**: numpy의 정의와 계산식 그대로다: 정수 k = 0, 1, …, −1에 fl(1/fl(n·d))를 곱한다. 그래서 numpy와 비트까지 같다(§9의 22번). n은 1 이상의 int, d는 0이 아닌 유한한 실수다. n = 0과 d = 0은 `ValueError`다(`[numpy와 다름]`: numpy와 scipy는 `ZeroDivisionError`. API.md §2에 없는 예외이고 값이 틀린 것이다; §9의 8번). d < 0은 받는다(numpy와 같다).
- **`fftshift`·`ifftshift`**: 값을 옮기기만 하므로 비트가 같고 dtype을 지킨다. `axes`는 int, tuple, list, None이다. 중복된 축은 `ValueError`다(`[numpy와 다름]`: numpy와 scipy는 그 축을 두 번 민다; §9의 9번과 같은 규칙).
- **`next_fast_len`·`prev_fast_len`**: scipy와 같은 값이다(§9의 14번: 복소는 11-매끄러운 수, `real=True`는 5-매끄러운 수). `target`은 1 이상의 int다. bool은 `TypeError`(`[scipy와 다름]`: scipy는 `True`를 1로 받는다), 0과 음수는 `ValueError`(`[scipy와 다름]`: scipy는 0에서 0을 돌려준다, 문서는 "양의 정수"라고 한다), 결과가 int64를 넘으면 `ValueError`(scipy와 같은 뜻)다. 뜻은 "이 길이 이상의 가장 작은 빠른 길이"이고, 빠른 길이의 집합은 scipy와 같은 집합이다. 그 집합은 자체 혼합 기수의 가장 싼 기수(2–11)만 쓰는 길이다(§3.3; 13–31의 소인수도 혼합 기수지만 더 느리다, FT64).

### 2.8 유한하지 않은 값과 빈 입력
- **NaN·inf**: IEEE 산술을 그대로 거친 결과다(경고 없음). 어느 출력 칸의 실수부·허수부가 NaN이나 inf가 되는지는 계산 순서에 달려 있어 약속하지 않는다(§9의 13번). numpy와 scipy도 서로 다르다: 길이 2–64에서 원소 하나가 NaN·inf·−inf인 실수 입력의 `fft`·`rfft` 12,474가지 가운데 4,018가지에서 칸마다의 NaN·inf 무늬가 달랐다(`tests/fft/data/scipy.json`). 같은 기계, 같은 입력이면 무늬도 같은 비트다.
- **길이 0의 변환**: 변환 축의 길이(입력이나 `n`·`s`)가 0이면 `ValueError`다(numpy·scipy와 같다; §9의 8번). 변환하지 않는 축의 길이가 0인 배열은 받고 빈 결과를 낸다(numpy·scipy와 같다: `fft(ones((0, 4)))`는 (0, 4)).
- **길이 1**: `fft`·`ifft`는 입력을 복사해 complex128로 바꾼 것과 정확히 같다(`'ortho'`·`'forward'`도 인자가 1이다). `irfft(n=1)`은 실수부다.

### 2.9 numpy·scipy와 다른 곳 (모음)
위의 `[numpy와 다름]`, `[scipy와 다름]`을 모은다. 이유는 그 자리에 있다.
- 공통 규칙에서 오는 것: 옵션은 키워드만, `norm=None`과 리스트 입력 없음, 범위 밖의 축은 `ValueError`, 중복된 축은 `ValueError`(numpy의 `fftn`·`fftshift`는 두 번 변환한다), int 인자에 bool은 `TypeError`, `workers`·`overwrite_x`·`plan`·`out`·`device` 없음, 백엔드 제어 없음.
- 경계에서 틀린 결과나 내부 메시지를 내는 곳: scipy의 `irfft(n=0)`(길이 1의 결과)과 DCT-I N = 1(`RuntimeError`), numpy·scipy의 `fftfreq(0)`과 `d=0`(`ZeroDivisionError`), scipy의 `next_fast_len(0)`(0)은 모두 `ValueError`다.
- 결과의 dtype: float32 입력이 없으므로 결과는 언제나 complex128·float64다.
- FFTLog: `offset`·`bias`·`initial`은 키워드만(scipy는 위치도 받는다), `dln`의 0·무한대와 다른 실수 인자의 NaN·무한대는 `ValueError`(scipy는 경고와 NaN), 특이한 변환에 경고 없음(FT60), `fhtoffset`은 Python float(FT61), 계수의 위상을 double-double로 만들어 값이 scipy와 위상 오차만큼 다르다(n = 4096이면 노름 상대 1e-11까지; FT58).

---

## 3. 수치 방법

### 3.1 길 고르기 (FT1, FT22, FT30, FT64–FT66)
**한 (종류, 길이)는 한 길이다.** 길은 이 절 끝의 표처럼 종류와 길이로만 정한다. 같은 길이의 결과가 두 알고리즘에서 나오는 일이 없으므로, 어떤 배치 모양·레이아웃·스레드 수에서도 한 줄의 비트는 그 줄 혼자 변환한 것과 같다(§3.7).

**모든 길이가 자체 C++이고 vDSP는 쓰지 않는다**(FT30, FT66). FT1은 vDSP가 받는 길이를 vDSP로 계산하기로 했고 FT22는 그것을 길이 꼴마다 빠른 쪽으로 좁혔다. 아래의 두 측정(시제품과 마무리한 커널)에서 vDSP가 받는 여덟 꼴 모두 자체 길이 빨랐다. 아래는 그 근거로 남기는 vDSP의 길이표와 측정이다.

**vDSP가 받는 길이** (헤더의 "implemented cases"를 n = 1–4096의 모든 길이와 f·2ᵏ(f = 1, 3, 5, 9, 15, 25; 설정이 512 MB를 넘지 않는 2²²까지)에서 설정 함수로 확인했다; 설정이 NULL이 아니면 받는다. 기록 `tests/fft/data/vdsp.json`)
| vDSP 함수(double) | 변환 | 헤더가 적는 길이 | 확인한 길이 |
|---|---|---|---|
| `vDSP_DFT_zop_CreateSetupD` + `vDSP_DFT_ExecuteD` | 복소 ↔ 복소, 실수부·허수부를 나눈 배열(split), stride 1 | 2ᵏ; f·2ᵏ (f = 3, 5, 15, k ≥ 3) | 헤더와 같다: 1, 2, 4, …; 24, 48, …; 40, 80, …; 120, 240, …(4096까지 34개, 그 위 2²²까지 40개) |
| `vDSP_DFT_zrop_CreateSetupD` + `vDSP_DFT_ExecuteD` | 실수 → 복소(정), 복소 → 실수(역); 길이는 실수 쪽, 짝수 | 2ᵏ; f·2ᵏ (f = 3, 5, 15, k ≥ 4) | 헤더와 같다: 2, 4, …; 48, 96, …; 80, …; 240, … |
| `vDSP_DFT_Interleaved_CreateSetupD` + `…_ExecuteD` | 복소 ↔ 복소, 실수부·허수부가 번갈아 놓인 배열(complex128의 꼴) | f·2ᵏ (f = 2, 3, 5, 9, 15, 25, k ≥ 2) | **헤더와 다르다**: 2ᵏ (8 ≤ 2ᵏ ≤ 4096)과 f·2ᵏ (f = 3, 5, 9, 15, k ≥ 2)이고 25·2ᵏ는 없다. 4096보다 긴 길이는 하나도 받지 않는다(헤더에 상한이 없다) |
| `vDSP_DCT_CreateSetup` + `vDSP_DCT_Execute` | DCT-II, III, IV | f·2ᵏ (f = 1, 3, 5, 15, k ≥ 4) | **float 판만 있다**(double 판이 없다). 쓰지 않는다. 덧붙여 받는 길이가 헤더와 다르고(헤더의 f·2ᵏ(k ≥ 4) 외에 120 = 15·2³과 45·2ᵏ(180, 360)도 받는다), n ≡ 2 (mod 4)인 18 이상의 모든 길이(4096까지 1,020개)에서 설정 호출이 `SIGSEGV`·`SIGBUS`로 프로세스를 죽였다 |
| `vDSP_fft_zopD`, `vDSP_fftm_zopD`, `vDSP_fft3_zopD`, `vDSP_fft5_zopD` | 옛 FFT(log₂n, stride, 여러 신호) | 2ᵏ; 3·2ᵏ, 5·2ᵏ | 헤더가 DFT 함수를 쓰라고 하고 `fft3`·`fft5`는 macOS 10.11부터 폐기다. 쓰지 않는다 |

**FT22의 측정** (FT29의 방법: sd의 레이아웃에서 끝까지, 1스레드, 세 번의 중앙값; 자체 길은 FT0 마일스톤의 시제품 `csrc/fft/cfft.hpp`·`rfft.hpp`, vDSP는 split과 엮기를 포함하고 n ≤ 4096은 `Interleaved`와 빠른 쪽; 기록 `tests/fft/data/prototype.json`의 `paths/…`)
| 종류 | 꼴 | 잰 길이 | vDSP/자체 시간의 기하평균 | 범위 | n ≤ 4096 | n > 4096 | 길 |
|---|---|---|---:|---|---:|---:|---|
| 복소 | 2ᵏ | 16–2²⁰ (17개) | 1.20 | 0.83–1.56 | 1.10 | 1.34 | 자체 |
| 복소 | 3·2ᵏ | 24–786432 (16개) | 1.57 | 0.75–2.81 | 1.27 | 1.95 | 자체 |
| 복소 | 5·2ᵏ | 40–655360 (15개) | 1.68 | 0.85–3.00 | 1.25 | 2.18 | 자체 |
| 복소 | 15·2ᵏ | 120–983040 (14개) | 1.60 | 1.00–2.81 | 1.22 | 1.96 | 자체 |
| 실수 | 2ᵏ | 16–2²⁰ (17개) | 1.27 | 0.96–1.56 | 1.27 | 1.26 | 자체 |
| 실수 | 3·2ᵏ | 48–786432 (15개) | 1.94 | 1.52–2.62 | 2.00 | 1.88 | 자체 |
| 실수 | 5·2ᵏ | 80–655360 (14개) | 1.84 | 1.57–2.85 | 1.72 | 1.93 | 자체 |
| 실수 | 15·2ᵏ | 240–983040 (13개) | 1.82 | 1.45–2.68 | 1.77 | 1.85 | 자체 |

- vDSP가 빠르거나 같았던 길이는 여덟 꼴의 121개 가운데 여덟이다: 복소 2ᵏ의 16·32·128·8192, 3·2ᵏ의 24·96, 5·2ᵏ의 40, 15·2ᵏ의 120, 실수 2ᵏ의 16384(0.75–1.00배; 짧은 것은 한 번에 수십 ns다). 긴 길이에서는 자체 길이 1.3–3배 빨랐다. 자체 시제품이 vDSP보다 빠른 까닭은 둘이다: 엮기와 풀기를 첫 단과 마지막 단에 합쳐 따로 도는 패스가 없고(vDSP는 2²⁰에서 split 변환 7.2 ms에 엮기 0.25 ms가 더 든다), 2의 거듭제곱이 아닌 길이에서 vDSP의 커널 자체가 느리다(3·2ᵏ·5·2ᵏ에서 같은 크기의 2ᵏ보다 원소당 1.5–2배).
- 두 길의 결과는 노름 상대 차이 10⁻¹⁵ 안에서 같았고 자체 길이 더 정확했다(§4.4).
- 그래서 FT30: **모든 길이를 자체 C++로 계산한다.**

**마무리한 커널로 다시 잰 것** (FT66; 같은 방법, 커널은 지금의 혼합 기수이고 2¹⁵ 이상은 4단계를 1스레드로; 기록 `tests/fft/data/prototype.json`의 `paths/…`, 2026-11-23)
| 종류 | 꼴 | 잰 길이 | vDSP/자체 시간의 기하평균 | 범위 | n ≤ 4096 | n > 4096 | 길 |
|---|---|---|---:|---|---:|---:|---|
| 복소 | 2ᵏ | 16–2²⁰ (17개) | 1.09 | 0.59–1.53 | 0.96 | 1.25 | 자체 |
| 복소 | 3·2ᵏ | 24–786432 (16개) | 1.43 | 0.60–2.52 | 1.11 | 1.83 | 자체 |
| 복소 | 5·2ᵏ | 40–655360 (15개) | 1.58 | 0.71–2.73 | 1.20 | 2.01 | 자체 |
| 복소 | 15·2ᵏ | 120–983040 (14개) | 1.52 | 0.92–2.57 | 1.21 | 1.81 | 자체 |
| 실수 | 2ᵏ | 16–2²⁰ (17개) | 1.17 | 0.87–1.54 | 1.14 | 1.20 | 자체 |
| 실수 | 3·2ᵏ | 48–786432 (15개) | 1.84 | 1.51–2.39 | 1.89 | 1.80 | 자체 |
| 실수 | 5·2ᵏ | 80–655360 (14개) | 1.78 | 1.59–2.65 | 1.69 | 1.86 | 자체 |
| 실수 | 15·2ᵏ | 240–983040 (13개) | 1.78 | 1.50–2.50 | 1.77 | 1.78 | 자체 |

- 여덟 꼴 모두 자체 길로 남는다. 가장 가까운 꼴은 복소 2ᵏ(1.09)이고, vDSP가 빠른 것은 짧은 길이(복소 16·32·128·8192: 수십 ns, 8192는 0.97배; 실수 16·16384)다. 4단계를 1스레드로 잰 2²⁰도 vDSP의 1.22배다(4단계는 1스레드에서 직접 Stockham의 0.83배라 시제품의 1.34배보다 작다). 길은 꼴 단위로만 옮기므로(FT22) 짧은 몇 길이 때문에 vDSP를 부르지 않는다.

**표** (FT30, FT64–FT66; 언제나 종류와 길이의 함수다)
| 종류 | 길이 | 길 |
|---|---|---|
| 복소 | 소인수가 모두 31 이하이고 n < 2¹⁵ (FT36, FT64) | 자체 혼합 기수(§3.3) |
| 복소 | 소인수가 모두 31 이하이고 n ≥ n₄ = 2¹⁵ (FT65) | 4단계(§3.3): n = n₁·n₂로 나눈 안쪽 변환은 위 줄의 길 |
| 복소 | 31보다 큰 소인수가 있는 길이 | 자체 Bluestein(§3.3; 안쪽은 길이 m의 혼합 기수나 4단계, FT37) |
| 실수 | 짝수 n | 길이 n/2의 복소 변환(위 표의 길) + 회전 인자 한 번의 사후 처리(§3.4) |
| 실수 | 홀수 n | 길이 n의 복소 변환과 에르미트 부분(FT43; 실수 전용 나비는 넣지 않았다, FT68) |
| DCT·DST | 모든 형 | 실수 변환 위의 자체 알고리즘(§3.5) |
| FFTLog | 모든 n | 길이 n의 실수 변환과 그 역(위 표의 길)과 계수의 곱(§3.9) |

- 4단계와 사후 처리처럼 한 길이 안에서 다른 길이의 변환을 부르는 것도 그 다른 길이의 표대로 간다. 그래도 바깥 길이의 결과는 바깥 길이로만 정해진다.
- **입력값으로 고르지 않는다.** 예: 허수부가 모두 0인 complex128 입력을 실수 길로 보내지 않는다(그러면 같은 dtype, 같은 길이의 결과가 값에 따라 다른 알고리즘에서 나온다). 실수 입력의 길은 dtype으로 정한다(§9의 12번).

### 3.2 vDSP에 대해 잰 사실
vDSP는 쓰지 않는다(FT30, FT66). 아래는 길을 고를 때 잰 사실이고, 나중에 vDSP를 다시 쓰게 되면 그 전제다(`tools/fft_probe.py`의 `vdsp` 구역, 기록 `tests/fft/data/vdsp.json`).

- **실행마다, 스레드마다, 정렬과 무관하게 같은 비트다.** 길이 1024, 3072, 4096, 15360, 20480, 40960, 2¹⁶, 2²⁰의 `zop`과 `zrop`에서 (a) 두 번 실행, (b) 10스레드가 같은 설정으로 50번씩 동시에 실행, (c) 입력과 출력을 8바이트 어긋나게 둔 실행, (d) 제자리 실행이 모두 첫 실행과 바이트까지 같았다. linalg의 LAPACK과 달리 16바이트 정렬이 반올림을 바꾸지 않았다(linalg.md §6.4와 대조).
- **설정을 다시 만들어도 같은 비트다.** 설정을 지우고 다시 만든 것, 다른 길이의 설정과 메모리를 나눠(`Previous`) 만든 것이 모두 같은 비트를 냈다(같은 여덟 길이의 `zop`과 `zrop`). 그래서 vDSP 설정을 계획 캐시에 두어도 쫓겨났다 다시 만든 설정이 같은 결과를 낸다.
- **안에서 스레드를 쓰지 않는다.** 2¹⁶, 2²⁰, 15·2¹⁶에서 CPU 시간/경과 시간이 1.00이었고 `VECLIB_MAXIMUM_THREADS=1`에서도 같은 시간이었다. 그래서 결과와 속도가 Accelerate의 스레드 모드와 무관하고, 병렬은 sd의 풀이 맡는다.
- **동시 실행은 헤더가 허락한다**: 설정은 실행 중에 읽기만 하므로 같은 설정(이나 메모리를 나눈 설정들)을 여러 스레드가 동시에 실행해도 된다. 메모리를 나눈 설정들 가운데 하나라도 실행 중이면 그 무리의 설정을 만들거나 지우면 안 된다. `csrc/fft/vdsp.hpp`의 `VdspPair`는 정방향 설정과 그것을 `Previous`로 만든 역방향 설정을 함께 만들고 함께 지우며 다른 설정과 메모리를 나누지 않으므로 이 규칙을 지킨다.
- **역변환의 관계**: 2의 거듭제곱 길이에서는 역변환이 conj(정변환(conj(x)))와 같은 비트였지만 3072, 15360, 20480, 40960에서는 아니었다. 그래서 vDSP의 역변환은 따로 만든 역방향 설정으로 해야 하고 이 관계에 기댈 수 없다.
- **정확도**: 균등 난수 입력에서 Arb와 견준 노름 상대오차가 2ᵏ에서 0.23–0.31, f·2ᵏ에서 0.27–0.44 × u·log₂n이었다(§4.4). numpy·scipy는 0.13–0.28, 자체 시제품은 0.15–0.33이다. 2의 거듭제곱이 아닌 길이에서 vDSP가 numpy보다 약 두 배 크다.
- **설정의 비용**: malloc이 쥔 바이트로 잰 설정 하나의 메모리는 n ≥ 2¹⁴에서 원소마다 `zop` 2ᵏ 14–15, 3·2ᵏ 26–34, 5·2ᵏ 25–29, 15·2ᵏ 24–35바이트, `zrop` 2ᵏ 11–12, f·2ᵏ 16–21바이트였다(역방향 설정을 `Previous`로 나누면 더 들지 않는다). 시간은 `zop` 2²⁰에서 5.8 ms·14 MB, 2²²에서 14–16 ms·60 MB, 15·2¹⁸에서 108 ms·92 MB였다. 자체 계획(회전 인자 표)은 원소마다 약 16바이트이고 2²⁰에서 약 5 ms다(§3.6). 큰 길이의 vDSP 설정은 변환만큼 비싸고(2²⁰의 변환은 7.3 ms), 2²⁸ 같은 길이의 설정은 수 GB다(설계 초안의 탐침에서 2³¹ 둘레의 설정이 20 GB를 넘게 잡아 멈췄다; 지금의 탐침은 512 MB를 넘을 설정을 만들지 않는다). vDSP를 다시 쓰게 되면 계획 캐시의 바이트 상한이 그 설정을 센다(`csrc/fft/vdsp.hpp`의 `vdsp_setup_bytes`, 원소마다 48바이트로 어림).
- **확인하지 못한 것**: P 코어와 E 코어에서 같은 비트인지(배경 QoS 스레드에서는 같았지만 그 스레드가 E 코어에서 돌았는지 알 수 없다), macOS를 올린 뒤에도 같은 비트인지(LAPACK처럼 같은 기계·같은 판의 약속으로만 둔다, API.md §3), vDSP 안의 알고리즘과 오차의 최악값(닫힌 소스다; 계약은 잰 값으로만 뒷받침한다, §4.3).

### 3.3 자체 복소 FFT (FT34–FT37, FT39, FT64, FT65, FT67)
`csrc/fft/cfft.hpp`(혼합 기수, 4단계, 두 줄 커널), `bluestein.hpp`, `ddmath.hpp`, `plan.cpp`다. 길은 길이로만 정한다: 31-매끄러운 길이는 혼합 기수(2¹⁵ 이상은 4단계), 나머지는 Bluestein이다(FT36, FT64, FT65).

- **혼합 기수**(FT34, FT35, FT64): Stockham 꼴이다. 입력과 출력은 complex128의 엮은 꼴이고 단 사이는 split 버퍼다(첫 단이 `vld2q`로 풀고 마지막 단이 `vst2q`로 엮는다). 첫 단(s = 1)은 이웃한 p 둘을 한 레지스터에 두고 출력을 `vst2q`·`vst3q`·`vst4q`로 엮어 쓰고, 나머지 단은 q를 안쪽 고리로 돌아 회전 인자가 p마다 한 쌍이다. 기수는 4, …, 4, (2), 3, …, 5, …, 7, …, 11, …, 13, …, 17, …, 19, …, 23, …, 29, …, 31, …의 순이다. 4와 2는 곱셈이 없는 나비, 홀수 기수는 FT35의 실수 꼴(상수 cos·sin(2πm/p)를 Arb로 구해 한 번 반올림)이다. 단 함수는 기수마다 `switch`로 고른다(FT68). 복소 곱은 실수부·허수부마다 곱 하나와 FMA 하나다. 고리의 길이가 홀수인 자리(홀수 n의 q 고리, m이 홀수인 첫 단)는 두 레인에 같은 값을 넣어 계산하고 첫 레인만 쓴다. 단이 하나뿐인 길이(n ∈ {2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31})는 엮은 꼴에서 엮은 꼴로 바로 간다. 역변환은 실수부와 허수부를 첫 읽기와 마지막 쓰기에서 바꾼다(FT35).
- **회전 인자**(FT39): e^(−2πik/n)를 k/n의 정확한 유리수에서 구한다(`csrc/fft/twiddle.hpp`: k mod n을 정확한 정수 연산으로 [0, π/4]에 접고 각을 double-double로 만든 뒤 `base/sincos.hpp`의 sincos와 1차 보정). 단마다 온전한 표다. 열두 계획 길이(12–4620의 모든 k, 2¹⁶·15·2¹⁰·2²⁰·131220에서 3000개씩)를 Arb와 견준 \|ŵ − w\|는 1.04u 이하였다. 증명은 μ ≤ 1.5u로 둔다(FT31).
- **Bluestein**(FT36, FT37): jk = (j² + k² − (k − j)²)/2로 yₖ = wₖ·Σⱼ(xⱼwⱼ)·conj(w_{k−j})이고(wⱼ = e^(−πij²/n)), 길이 m의 원형 합성곱이다. m은 2n − 1 이상인 가장 작은 2ᵃ3ᵇ5ᶜ다(안쪽 변환은 기수 4, 2, 3, 5만 쓴다). 호출마다 chirp 곱, 길이 m의 정변환, 핵의 변환 β와의 곱, 길이 m의 역변환(정규화 없음, 1/m은 β에 있다), chirp 곱이다. chirp의 지수 j² mod 2n은 128비트 정수 연산이고, chirp와 β는 계획을 만들 때 double-double로 계산해(Taylor 급수의 sin·cos, √N 크기의 두 표, 기수 4·2·3·5의 재귀 DFT) 한 번 반올림한다. 계획 시간은 1009에서 0.5 ms, 10007에서 5.6 ms, 65537에서 39 ms, 100003에서 60 ms다(2²⁰의 혼합 기수 계획은 5.0 ms; 2026-11-23, 한 번씩 잰 값). 작업 메모리는 줄마다 6m개의 double이다. 안쪽 길이 m이 2¹⁵ 이상이면 안쪽 변환이 4단계이고(65537의 m = 131220), 긴 줄의 chirp 곱은 2¹⁴칸씩 풀에서 나눈다(§3.7).
- **어느 소인수부터 Bluestein인가**(FT36, FT64): p₀ = 31이다. 기수 2–11의 단은 log₂ r 한 수준마다 3.75–7.62u라 κ = 8 안이고, 13–31의 단은 8.22–16.83u라 그런 기수가 하나라도 있는 길이는 κ = 17로 센다(§4.2). 그 꼴은 그 길이들을 Bluestein으로 계산할 때의 한계(ρ(2L(m) + 2)κ, ρ ≥ 1.9)보다 모든 길이에서 좁다. `next_fast_len`의 집합(11-매끄러운 수, §2.7)은 가장 싼 기수(2–11)만 쓰는 길이다. 일반 기수의 일은 p에 비례하고 37의 단은 한 수준에 약 20u가 들어 κ = 17도 넘으므로 37 이상은 Bluestein이다.
- **4단계**(FT41, FT65): 31-매끄러운 n ≥ 2¹⁵는 n = n₁·n₂(n₁은 √n 이하의 가장 큰 약수)로 나눠 열 패스(열 16개씩 모아 길이 n₁의 변환, 회전 인자 w_n^(j₂k₁) 곱, 열 묶음 단위로 작업 버퍼에)와 행 패스(행 16개씩 모아 길이 n₂의 변환, 결과에 흩기)로 한다. 묶음이 풀의 태스크이고 분해가 n으로만 정해지므로 긴 줄 하나가 스레드 수와 무관한 비트로 모든 스레드에서 돈다. 회전 인자 표는 온전한 표 하나(16n바이트, FT39)이고 작업 버퍼는 직접 Stockham과 같은 2n개의 double이다. 오차의 셈은 단의 셈 그대로다(열 변환의 마지막 단에 없는 회전 인자 곱을 4단계의 회전 인자 곱이 대신한다). n₁, n₂가 다시 2¹⁵ 이상이면(n ≥ 2³⁰) 안쪽도 4단계다.
- **두 줄 커널**(FT67): 단을 두 레인이 두 줄의 같은 위치인 꼴로 돈다(`detail::Lines`). 각 레인이 한 줄 커널과 같은 연산을 하므로 비트가 같고, 줄을 간격이 있는 메모리에서 바로 읽고 쓴다(마지막 축이 아닌 축의 패스, §3.7).
- 무엇이든 n과 계획으로만 정해진다. 정렬, stride, 배치, 스레드 수, 값으로 고르는 것이 없다.

### 3.4 실수 변환과 에르미트 변환 (FT16, FT43–FT46)
`csrc/fft/rfft.hpp`, `plan.hpp`의 실수 계획, `driver.cpp`의 `r2c_axis`·`c2r_axis`다. 길은 길이의 홀짝과 안쪽 길이로만 정한다(FT43).

- **짝수 n**(실수 계획, FT43·FT44): n개의 실수를 그대로 길이 h = n/2의 complex128로 읽어(복사 없이: 짝·홀 원소가 실수부·허수부다) 길이 h의 복소 계획(혼합 기수나 Bluestein, §3.3)으로 변환하고 회전 인자 w = e^(−2πik/n) 한 번의 사후 처리를 한다: e = (Z_k + conj Z_{h−k})/2, o = −i(Z_k − conj Z_{h−k})/2(x의 짝·홀 원소의 DFT)에서 X_k = e + w·o, X_{h−k} = conj(e − w·o)를 한 번에 만든다. X_0 = Re Z_0 + Im Z_0, X_h = Re Z_0 − Im Z_0이고 둘의 허수부는 정확히 +0이다. 자리 h/2(짝수 h)는 제 짝이라 한 번만 쓴다. 회전 인자는 k = 0, …, h/2의 표 하나다(`real_twiddles`, §3.3의 `twiddle()`).
- **짝수 n의 역변환**(`irfft`·`hfft`): 같은 단계를 거꾸로 한다. a = X_k + conj X_{h−k}(= 2E_k), b = X_k − conj X_{h−k}(= 2wO_k), o′ = conj(w)·b에서 Z′_k = a + i·o′, Z′_{h−k} = conj(a − i·o′), Z′_0 = (X_0 + X_h) + i(X_0 − X_h)이고, 길이 h의 정규화 없는 역변환이 n·z(z_j = x_{2j} + i·x_{2j+1})를 결과의 메모리에 바로 쓴다. 1/2을 곱하지 않아 정변환보다 곱셈이 하나 적다. X_0과 X_h의 허수부는 읽지 않는다(버리는 부분, §2.3). `hfft`는 같은 단계에서 입력의 켤레를 읽는다(곱셈 −1, 정확).
- **홀수 n**(FT43): 줄을 허수부 0의 complex128로 펴서 길이 n의 복소 계획(캐시의 것)으로 변환하고 에르미트 부분 y_k = (t_k + conj t_{n−k})/2(y_0 = Re t_0 + 0i)를 쓴다. 역변환은 쓰는 n//2 + 1칸에서 에르미트 확장(X_0의 허수부 버림)을 만들어 복소 역변환하고 실수부를 쓴다. 홀수 길이의 실수 전용 나비는 넣지 않았다(FT68: 홀수 길이의 느린 곳은 Bluestein 길이였고 기수 13–31이 대부분을 고쳤다).
- **긴 줄**: 짝수 n ≥ 2¹⁶이면 안쪽 길이 n/2가 4단계이고, 회전 인자 한 번의 사후·사전 처리와 거울 채우기도 2¹⁴칸씩 풀에서 나눈다(칸마다 따로 계산하므로 비트가 같다; §3.7).
- **실수 입력의 `fft`·`ifft`**(FT16): 실수 변환의 반쪽 결과를 에르미트 대칭으로 채운다(y_{n−k} = conj(y_k)). 대칭이 정확하고 DC의 허수부가 +0이다. 그래서 `rfft(x)`는 `fft(x)[:n//2 + 1]`과 비트까지 같다(numpy에서는 같지 않고 scipy에서는 같았다, `tests/fft/data/{numpy,scipy}.json`). 실수 입력의 `ifft`는 그 켤레에 역방향 인자를 곱한 것이다(FT46). `ihfft`는 `conj(rfft)`, `hfft`는 `irfft(conj(x))`에 반대 방향의 인자를 곱한 것이고 같은 연산이라 비트까지 같다(numpy의 정의).
- **정규화와 켤레**는 결과에 인자 하나를 곱한 뒤(복소 변환과 같은 `scale_line`) 허수부의 부호를 바꾼다(정확해서 순서가 비트를 바꾸지 않는다).
- **구동기**: 복소 변환의 구동기(§3.7)와 같은 나누기(줄 수의 덩어리, 크기로만)와 같은 줄 묶음이다. 줄이 이미 계획이 읽는 꼴이면(짝수 n의 연속 float64 입력, 연속 complex128 반쪽 스펙트럼) 입력에서 바로 읽고, 결과의 줄이 연속이면 바로 쓴다. 아니면 모아서(float64·int64·bool의 변환, 자르기와 0 채우기 포함) 변환하고 흩는다. 어느 쪽이든 같은 연산이라 한 줄의 비트는 배치·레이아웃·스레드 수와 무관하다.
- vDSP `zrop`를 다시 쓰게 되면: 정변환이 결과에 2를 곱해 내놓으므로(헤더의 C = 2) 정규화 인자에 1/2을 합쳐 한 번 곱하고(2의 거듭제곱이라 정확하다), 결과의 꼴(DC와 나이퀴스트가 한 칸에 실수부·허수부로 담김)을 풀어서 n//2 + 1개의 complex128로 쓴다.

### 3.5 DCT·DST (FT28, FT53–FT57)
vDSP의 DCT는 float 판뿐이라(§3.1) 모든 형을 자체 알고리즘으로 실수 FFT 위에 만든다: `csrc/fft/r2r.hpp`의 줄 커널과 표, `r2r.cpp`의 인자 검사와 다차원 패스, `driver.cpp`의 `r2r_axis`, `plan.cpp`의 계획 종류 `Dct2`·`Dct4`다(FT54). 길은 형과 N으로만 정한다.
- **DCT-I**(N ≥ 2): 짝 확장 v = (x₀, …, x_{N−1}, x_{N−2}, …, x₁)(길이 2(N − 1))의 실수 FFT V에서 y_k = Re V_k다. 계획은 캐시의 실수 종류 2(N − 1)이다.
- **DST-I**: 홀 확장 (0, x₀, …, x_{N−1}, 0, −x_{N−1}, …, −x₀)(길이 2(N + 1))의 실수 FFT에서 y_k = −Im V_{k+1}이다. 계획은 실수 종류 2(N + 1)이다. 그래서 N = 1은 2x다(scipy와 같다).
- **DCT-II**: 짝 원소를 앞에, 홀 원소를 뒤집어 뒤에 둔 순서 바꿈 v = (x₀, x₂, …, x₃, x₁)(Makhoul)과 길이 N의 실수 FFT V(짝수 N은 실수 계획, 홀수 N은 복소 계획과 에르미트 부분; §3.4), 그리고 y_k = 2·Re(e^(−iπk/2N)·V_k)다. (c, s) = (cos, sin)(πk/2N)로 V_k 하나에서 y_k = 2(c·Re V_k + s·Im V_k)와 y_{N−k} = 2(s·Re V_k − c·Im V_k)를 만든다(곱 하나와 FMA 하나). 표는 k = 0, …, N/2의 (c, s)이고 `twiddle(k, 4N)`에서 온다.
- **DCT-III**: 그 역순이다. V₀ = x₀, V_k = (x_k − i·x_{N−k})·e^(+iπk/2N)(k = N/2에서는 x_{N−k} = x_k라 저절로 맞다)이 에르미트 반쪽 스펙트럼이고, 정규화 없는 역 실수 FFT v의 순서를 되돌려 y_{2j} = v_j, y_{2j+1} = v_{N−1−j}다. 계획은 DCT-II와 같은 `Dct2`다.
- **DCT-IV, 짝수 N**: z_j = (x_{2j} + i·x_{N−1−2j})·w_j, w_j = e^(−iπ(8j + 1)/8N)(j < N/2)의 길이 N/2 복소 FFT Z에 다시 w_p를 곱해 u_p = Z_p·w_p, y_{2p} = 2·Re u_p, y_{N−1−2p} = −2·Im u_p다. k = 2p, n = 2j의 위상 π(2k + 1)(2n + 1)/4N이 2πpj/(N/2)에 π(8p + 1)/8N과 π(8j + 1)/8N을 더한 것이고, 홀수 칸의 코사인은 그 위상의 사인이다. 표는 w_j 하나다(`twiddle(8j + 1, 16N)`).
- **DCT-IV, 홀수 N**(FT54): 부호를 붙인 순열 위의 길이 N 실수 FFT 하나다. a = 2k + 1, b = 2n + 1, 1/8N = α/8 + β/N(αN + 8β = 1, N이 홀수라 있다)이면 위상 2πab/8N = A + B, A = 2παab/8, B = 2πβab/N이다. 곱셈 지표 χ₁(r ≡ ±1 mod 8에서 +1, ±3에서 −1)과 χ₄(r ≡ 1 mod 4에서 +1, 3에서 −1)로 cos A = χ₁(αab)/√2, sin A = χ₁χ₄(αab)/√2이므로 y_k = √2·χ₁(αa)·Σₙ χ₁(b)xₙ(cos B − χ₄(αa)χ₄(b) sin B)다. u_j = χ₁(b)xₙ를 χ₄(b) = 1이면 j = b mod N, −1이면 j = −b mod N에 두면(b ↦ 2N − b가 χ₄를 지키는 대합이라 j는 전단사다) U = DFT_N(u)로 y_k = √2·χ₁(αa)(Re U_l + χ₄(αa)·Im U_l), l = βa mod N이다(N/2 위의 l은 U_l = conj U_{N−l}). 순열과 부호는 계획을 만들 때 표(원소마다 16바이트)로 만든다. 이 꼴은 √2 × 등거리라 실수 FFT의 오차를 그대로 옮긴다(§4.2). 길이 2N이나 4N의 확장에 비해 일이 2–4배 적다.
- **DST-II·III·IV**는 같은 형의 DCT에 부호 바꾸기와 뒤집기를 더한 것이다: DST-II(x)_k = DCT-II((−1)ⁿxₙ)_{N−1−k}, DST-III(x)_k = (−1)ᵏ·DCT-III(x_{N−1−n})_k, DST-IV(x)_k = (−1)ᵏ·DCT-IV(x_{N−1−n})_k. 줄 커널이 정확히 그렇게 계산하므로 DST의 비트는 그 DCT의 비트에서 바로 나온다(§4.5).
- **정규화와 `orthogonalize`**(FT55): 결과에 인자 s 하나를 곱한다(1, 1/N′, fl(1/fl(√N′)); N′은 §2.5의 논리 길이). `orthogonalize`는 DCT-III의 x₀(DCT-I은 x₀과 x_{N−1})에 변환 전에 fl(√2)를 곱하고, DCT-II의 y₀(DCT-I은 y₀과 y_{N−1})을 인자를 곱한 뒤 fl(√2)로 나눈다. 모두 한 번씩 따로 반올림한다. DST-II·III은 뒤집기를 거쳐 y_{N−1}·x_{N−1}이다. DCT-IV와 DST-I에는 없다(scipy와 같다). `idct`·`idst`는 형 2와 3을 바꿔 계산하고 역방향 인자를 곱한다.
- **다차원**(FT57): 축마다 1차원 판을 `axes`의 뒤에서 앞으로 거듭한다. 인자 하나를 마지막 패스가 곱한다.
- **구동기**(`r2r_axis`): 복소·실수 변환과 같은 나누기와 줄 묶음(§3.7)이다. 줄을 언제나 묶음 버퍼에 모은 뒤(커널이 제 사본을 고쳐 쓴다) 변환하고, 결과의 줄이 연속이면 바로 쓰고 아니면 흩는다. 묶음 전체를 모은 뒤 쓰므로 결과가 입력 자신이어도 된다(다차원의 제자리 패스). 복소 입력은 실수부·허수부의 float64 보기(같은 stride, 시작이 0 또는 8바이트)를 읽고 결과의 같은 부분에 쓴다.
- **긴 줄**: 앞뒤 처리(확장, 순서 바꿈, 회전 인자 곱)는 2¹⁴칸씩 풀에서 나누고 안쪽 실수·복소 변환은 길이가 2¹⁵ 이상이면 4단계다(§3.7).
- **잰 시간**(§5): 형 2–4는 2¹⁶·2²⁰의 1스레드에서 scipy의 1.4–1.7배다. DCT-I은 안쪽 길이 2(N − 1)의 반 N − 1에 37 이상의 소인수가 있으면 Bluestein이라(2¹⁶ − 1 = 3·5·17·257의 257, 2²⁰ − 1 = 3·5²·11·31·41의 41) 1스레드에서 2¹⁶ 0.99배, 2²⁰ 0.51배다. scipy는 큰 소인수를 일반 기수로 계산한다. 1023 = 3·11·31은 혼합 기수라 n = 1024 줄 1000개의 DCT-I은 1스레드에서 scipy의 0.84배, 10스레드에서 scipy `workers=10`의 0.93배다. 길이 N − 1의 실수 FFT로 하는 꼴은 수치적으로 불안정해 쓰지 않는다(FT68).

### 3.6 계획과 캐시 (FT24, FT39, FT44)
- **계획**은 (종류, 길이)마다 하나다(`fft/plan.hpp`의 `Plan`): 종류는 `Complex`, `Real`(짝수 n), `Dct2`(DCT·DST 형 2·3), `Dct4`(형 4)다. DCT-I·DST-I은 실수 종류 2(N ∓ 1)을 쓴다. 계획은 길(§3.1), 소인수 분해, 회전 인자 표, 4단계의 열·행 계획, Bluestein의 chirp와 β와 안쪽 계획, 실수·DCT 종류의 안쪽 복소 계획(스스로 만든다, FT44)과 표를 쥔다. vDSP 설정은 없다(FT30). 만든 뒤에는 바뀌지 않는다(읽기만 한다).
- **잰 값으로 계획하지 않는다.** FFTW의 측정 계획처럼 실행 시간을 재서 알고리즘을 고르면 결과가 그때의 기기 부하에 따라 달라진다. 계획은 길이로만 정한다. 문턱(p₀ = 31, n₄ = 2¹⁵)과 묶음 크기는 재서 정한 상수다(FT64, FT65, FT68).
- **캐시**: 프로세스에 하나, 계획을 `shared_ptr`로 쥐는 LRU다. 찾기와 넣기는 뮤텍스 하나로 지킨다. 같은 계획을 두 스레드가 동시에 만들면 먼저 넣은 것을 쓰고 다른 하나는 버린다(계획은 (종류, 길이)로만 정해져 어느 것이나 같은 비트다). 쫓겨난 계획은 쓰는 호출이 끝나면 풀린다. 상한은 계획 32개, 합 256 MB다(FT24, `kCachePlans`·`kCacheBytes`; 그보다 큰 계획 하나는 캐시에 넣지 않고 호출마다 만든다: 2²⁴의 계획이 269 MB라 그 둘레부터다). 혼합 기수 계획은 원소마다 약 16바이트(2²⁰에서 16.8 MB), Bluestein 계획은 chirp(16n)·β(16m)·안쪽 계획이라 n의 원소마다 약 81바이트다. 캐시는 32칸의 선형 탐색과 마지막 사용 시각의 LRU다(`tests/fft/test_plan_cache.py`). FFTLog의 계획은 따로 전용 캐시(16개·128 MB)에 둔다(§3.9).
- **계획을 만드는 동안은 GIL을 놓는다**(2²⁰의 계획은 약 5 ms, 65537의 Bluestein 계획은 약 39 ms다). 계획을 만드는 데 실패하면(메모리) `MemoryError`다.

### 3.7 배치, 축, 병렬화와 결정성
**같은 입력이면 스레드 수, 배치 모양, 메모리 레이아웃, Accelerate의 스레드 모드와 상관없이 같은 결과다.** 한 줄의 결과는 그 줄의 값과 (종류, 길이, `norm`)으로만 정해진다: `fft(X, axis=a)`의 한 줄은 그 줄만 꺼내 `fft`한 것과 바이트까지 같다. 테스트는 1, 3, 10스레드와 C·F·띄엄띄엄한 레이아웃, 두 Accelerate 모드에서 같은 바이트를 확인한다(§7.3).

- **축 구동기**(`csrc/fft/driver.cpp`, FT40, FT50, FT67): 줄이 이미 계획이 읽는 꼴(complex128, 변환 축의 간격 16바이트, 자르기만 있음; 실수 변환은 연속한 float64 줄과 연속한 반쪽 스펙트럼)이면 입력에서 바로 읽고 결과의 줄에 바로 쓴다. 아니면 이웃한 줄을 한 묶음으로 태스크의 버퍼에 모아(gather; dtype 변환, 자르기와 0 채우기 포함) 한 줄씩 변환하고 결과의 자리에 흩는다(scatter). 줄의 원래 stride와 정렬은 계산에 들어가지 않는다(버퍼의 줄은 128바이트로 정렬한다).
- **두 줄 커널과 타일 복사**(FT67): 복소 입력의 줄이 바로 읽기의 꼴이 아니면(마지막 축이 아닌 축, 띄엄띄엄한 stride) 위치 간격이 L1 캐시에 머무는 줄(n ≤ 8·16384/max(gcd(간격, 16384), 128))은 두 줄 커널(§3.3)이 바로 읽고 쓴다: 레지스터의 두 레인이 두 줄의 같은 위치이고 각 레인이 한 줄 커널과 같은 연산을 하므로 비트가 같다. 태스크는 8줄 이상이다(이웃한 줄이 배열의 128바이트 캐시 줄을 나누므로). Bluestein·4단계 길이, 0을 채우거나 켤레를 취하는 입력은 두 줄 커널을 쓰지 않는다. 그 밖의 줄(2의 큰 거듭제곱의 배수 간격)은 묶음으로 모으고 흩되 복사를 128바이트 타일(복소 8칸, 실수 16칸) 단위로 한다. 둘 다 레이아웃으로 고르는 실행 방식이고 결과의 비트는 같다.
- **다차원**(FT47, FT48, FT57): 축마다 구동기를 한 번씩, `axes`의 뒤에서 앞으로 돈다(§2.2). 첫 패스가 입력을 읽어 새 complex128 배열에 쓰고, 축의 길이가 그대로인 뒤 패스는 그 배열을 제자리에서 변환한다(모은 줄을 흩으므로 바로 읽기와 비트가 같다). `rfftn`은 첫 패스, `irfftn`은 마지막 패스가 실수 쪽이다. 정규화는 마지막 패스의 인자 하나다. 패스마다 줄을 태스크로 나누므로 각 패스가 병렬이고, 한 줄의 비트는 그 줄의 값과 길이로만 정해지므로 결과가 스레드 수와 입력의 레이아웃과 무관하다(중간 배열은 언제나 C 순서다).
- **묶음과 태스크**(FT50): 묶음의 버퍼는 스레드의 scratch(슬롯 넷, 슬롯마다 2¹⁶ double = 512 KB까지 호출 사이에 남김)이고 줄 간격을 16 double의 배수에 16을 더해 2의 거듭제곱에서 비킨다. 묶음은 min(2¹⁶/줄 간격, ⌈줄 수/64⌉, 128)줄(바로 읽기는 1줄)이고 태스크는 max(⌈2¹²/비용⌉, ⌈줄 수/1024⌉)줄(묶음의 배수로 올림)이다. 비용은 한 줄의 n·L(n)(Bluestein은 2m·L(m) + 4m)이다. 모두 크기로만 정하고 한 줄의 비트를 바꾸지 않는다. 마지막 축이 아닌 축의 패스는 큰 배열에서 메모리에 묶인다: (256, 256, 129)의 축 1 패스가 1스레드에서 16.5 ms(두 줄 커널), 축 0 패스가 18.1 ms(타일)다(FT67).
- **나누기**: 줄들을 태스크 크기의 고정된 덩어리로 나눈다(위의 규칙, array.md §10.4의 `kParallelTask`와 같은 2¹²). 한 줄은 한 태스크가 처음부터 끝까지 계산한다(4단계 줄은 아래).
- **큰 1차원 변환 하나**(§9의 21번, FT65): 31-매끄러운 n ≥ n₄ = 2¹⁵이면 4단계(Bailey, §3.3)로 n = n₁·n₂(n으로만 정함)를 길이 n₁의 변환 n₂개, 회전 인자 곱, 길이 n₂의 변환 n₁개로 나누고 그 변환들을 16개씩 풀에서 나눈다. 분해가 스레드 수와 무관하므로 1스레드에서도 같은 4단계를 같은 순서로 하고, 결과는 스레드 수와 무관하다. 그런 줄(안쪽 길이가 4단계인 실수·Bluestein·DCT 줄 포함)이 여럿이면 구동기는 줄을 차례대로 하나씩 돌린다(`Plan::parallel_inside`). 긴 줄의 O(n) 패스(실수 처리, Bluestein의 chirp 곱, 정규화, 켤레·거울, DCT의 앞뒤 처리, FFTLog의 곱)도 2¹⁴칸씩 나눈다(`run_chunks`; 태스크 안에서는 차례대로). 1스레드에서는 빨라지지 않는다(직접 Stockham에 대한 속도 비가 2¹⁵–2¹⁸에서 0.91–1.05, 2²⁰ 0.83, 2²² 0.78; 열과 행을 모으고 흩는 간격 있는 접근이 직접 Stockham의 순차 패스보다 비싸다).
- **병렬 기준값**: 호출의 작업량(줄 수 × 비용)이 2¹⁵ 이상이면 나누고, 그 이상이면 스레드 수와 상관없이 GIL을 놓는다(API.md §3, array.md §10.4). 풀이 다른 Python 스레드의 일로 바쁘면 1스레드로 돈다(결과는 같다).

| 병렬 경로 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| 줄의 배치(1차원, 다차원의 각 패스) | 줄 덩어리(크기로만) | 한 줄은 어디서든 같은 계획과 같은 연산이다. 두 줄 커널의 레인도 한 줄 커널과 같은 연산이다 |
| 큰 1차원 변환 | 4단계의 열·행 묶음(16개씩, n으로만 정한 분해) | 안쪽 변환은 서로 독립이고 분해가 고정이다 |
| 긴 줄의 O(n) 패스 | 2¹⁴칸씩 | 칸마다 따로 계산한다 |
| 계획 만들기 | 복소·실수·DCT 계획은 하지 않는다(호출 스레드). FFTLog의 계수는 16개씩 | 계수는 m마다 따로 계산한다 |
| `fftshift` | 배열의 `roll`과 같은 복사 | 복사뿐이다 |

### 3.8 메모리
- 사용자가 모양으로 요청한 결과는 할당 전에 물리 메모리와 견주어 넘으면 `MemoryError`다(API.md §2).
- 태스크마다의 작업 메모리는 줄 묶음의 버퍼와 계획의 작업 버퍼(혼합 기수 2n개 double, Bluestein 6m개)다. 묶음은 버퍼 하나가 2¹⁶개 double(512 KB)에 드는 줄 수이고(한 줄이 그보다 길면 한 줄), 그 크기까지는 스레드의 scratch를 다시 쓰고 더 크면 태스크가 할당한다. 4단계의 중간 버퍼는 계획의 작업 버퍼(2n개 double, 직접 Stockham과 같다)이고, 그 태스크는 열·행 16개의 묶음 버퍼 둘과 안쪽 작업 버퍼를 스레드의 몫으로 둔다(2¹⁷개 double까지 호출 사이에 남기고, 깊이마다 슬롯 셋; 2²⁰에서 스레드마다 약 0.6 MB). 두 줄 커널은 줄 짝마다 8n개 double이다.
- 계획 캐시는 §3.6의 상한 안이다. 혼합 기수 계획은 회전 인자 표(원소마다 약 16바이트), Bluestein 계획은 chirp(16n바이트), β(16m바이트)와 안쪽 계획이다.
- 다차원 변환의 중간 배열은 많아야 둘이다(FT47): 길이가 그대로인 패스는 제자리다. `rfftn`의 중간 결과는 결과 배열 자체, `irfftn`의 중간 결과는 마지막 축이 n//2 + 1칸인 complex128 하나다. 묶음 버퍼는 스레드마다 슬롯 넷 × 512 KB까지 호출 사이에 남는다(FT50).
- FFTLog의 계획(FT59)은 계수 n/2 + 1개(16바이트)와 바이어스 둘(q ≠ 0이면 n개씩 8바이트)이다. 2²⁰에서 8 MB(q ≠ 0이면 24 MB)이고 캐시는 128 MB까지다.

### 3.9 FFTLog (FT21, FT58–FT63)
`csrc/fft/fftlog.hpp`의 double-double 함수와 계획, `fftlog.cpp`의 계획 만들기·캐시·`fhtoffset`·바인딩, `driver.cpp`의 `fht_axis`, `src/stardust/fft/_fftlog.py`의 인자 검사다. 정의는 scipy의 것(Hamilton 2000의 FFTLog)이다. 길이 n의 줄 a(간격 dln, 차수 mu, offset = ln k_c r_c, 바이어스 q, j_c = (n − 1)/2)에 대해

- v_j = a_j·e^(−q(j − j_c)dln), V = v의 실수 DFT의 반쪽 스펙트럼, W_m = U_m·V_m/n(m = 0, …, n/2), w = 에르미트 W의 정규화 없는 역실수 DFT, A_j = w_(n−1−j)·e^(−q((j − j_c)dln + offset))다.
- U_m = 2^q·Γ(x₊ + iy_m)/Γ(x₋ − iy_m)·e^(2iy_m(ln 2 − offset)), x± = (mu + 1 ± q)/2, y_m = πm/(n·dln)다. `ifht`는 곱 대신 conj U_m으로 나누고 바이어스는 역수다(입력 e^(q((j − j_c)dln + offset)), 출력 e^(q(j − j_c)dln)).
- **줄의 계산**(FT59, FT63): 줄마다 입력 바이어스를 곱하며 모으고(q = 0이면 바이어스가 없고 짝수 n의 이어진 float64 줄은 실수 계획이 바로 읽는다), 실수 변환(짝수 n은 실수 계획, 홀수 n은 복소 계획과 에르미트 부분; FT43), 계수와의 FMA 복소 곱(1/n은 계수에 들어 있다), 역실수 변환, 뒤집으며 출력 바이어스를 곱해 결과에 쓴다. 구동기는 다른 변환과 같은 `drive_lines`이고 줄의 비트는 줄의 값과 계획으로만 정해진다.
- **계수**(FT58): Z = log Γ(x₊ + iy) − conj log Γ(x₋ + iy) + q ln 2 + 2πi·m·(ln 2 − offset)/(n·dln)을 double-double로 만들고 허수부를 2π로 줄인 뒤 e^(±Re Z − ln n)·(cos, sin)(Im Z)의 부분마다 한 번 반올림한다. log Γ는 2πi의 배수를 빼고 구한다(e^Z만 쓰므로 가지는 상관없다): Re w < 1/2는 반사식 log π − log sin(πw) − log Γ(1 − w), \|w\| < 16은 log Γ(w + N) − log(w(w + 1)⋯(w + N − 1))(Re(w + N) ≥ 16), 그 위는 Stirling 8항((w − ½)log w − w + log √(2π)은 double-double, 급수(1/192 아래)는 double; 나머지는 2⁹\|B₁₈\|/(18·17·16¹⁷) < 2⁻⁶¹). log z는 libm의 log\|z\|·arg z에서 시작해 z·e^(−L₀−iθ₀) = 1 + e를 double-double로 만들고 e − e²/2를 더한다. 지수와 사인·코사인은 표(2^(i/64), (sin, cos)(jπ/64); 처음에 급수로 만든다)와 줄인 인자의 테일러 급수(앞 몇 항은 double-double, 나머지는 double)다. log sin(πw)는 \|Im w\| > 20이면 π\|b\| − ln 2 + i·sgn(b)(π/2 − πf)(f = Re w mod 2; 나머지 항은 2⁻¹⁸⁰ 아래), 그 밖은 sin(πf)cosh(πb) + i cos(πf)sinh(πb)의 double-double 로그다. q = 0이면 log Γ 하나로 Z = 2i·Im log Γ(x + iy)이고 \|U_m\| = 1이다. 특이한 계수와 극점은 FT60이다.
- **계수 계산의 비용**: 계수 하나에 log Γ 하나(q ≠ 0이면 둘)이고 log Γ 하나가 double-double 로그 하나다(작은 \|w\|만 옮기기의 곱과 로그 하나를 더한다). 계수는 m마다 따로라 16개씩 풀에서 나눈다. 1스레드 2¹²에서 새 계획이 0.48 ms(q ≠ 0이면 0.88 ms), 10스레드 75 µs다(§5).
- **`fhtoffset`**: initial + (arg − round(arg))·dln, arg = (ln 2 − initial)/dln + (Im log Γ(x₊ + iy) + Im log Γ(x₋ + iy))/π, y = π/(2·dln)이고 round는 짝수 쪽 반올림이다(scipy와 같다). arg를 double-double로 만들고(log Γ의 가지는 arg를 짝수만큼 옮기므로 반올림이 정확히 지운다) 결과를 한 번 반올림한다.

---

## 4. 정확도 계약
참값은 Arb다. 입력은 double이라 Arb의 `acb.dft`(구간 산술이라 알고리즘과 무관하게 엄밀하다)가 참 변환을 담는 구간을 준다. 정밀도는 결과의 반지름이 측정에 충분할 때까지(상대 2⁻¹²⁰ 아래) 늘린다. 오차의 노름도 Arb에서 구한다(돌려준 double과 참값의 차이를 정확히 빼서 제곱합). 계약 단위(오차 ÷ 한계)로 적고 경계는 1이다(spatial.md §4와 같은 꼴). 판정은 `tests/fft/refdata.py`의 `bound()`·`units()` 하나이고 테스트와 정확도 체계가 같이 쓴다.

### 4.1 왜 노름 계약인가
- 원소마다의 상대오차는 FFT에서 일반적으로 지킬 수 없다. 출력 yₖ = Σⱼ xⱼwʲᵏ는 크기 ‖x‖인 항들의 상쇄로 0에 가까울 수 있다. 회전 인자 하나를 올바르게 반올림해도(상대 u) yₖ는 약 u·Σ|xⱼ|만큼 움직이므로, yₖ가 그보다 작으면 상대오차는 1을 넘는다. yₖ의 원소별 조건수 Σ|xⱼ|/|yₖ|에 상한이 없기 때문이다. 예: `fft([1, 2, 3, 4], n=6)`의 넷째 칸은 참값이 −2 + 0i인데 numpy는 허수부 8.9×10⁻¹⁶을 낸다(상대오차 무한대). 어떤 O(n log n) 알고리즘도 이 칸을 정확한 0으로 만들 수 없다.
- FFT가 지키는 것은 노름의 역방향 안정성이다: ŷ = F(x + Δx), ‖Δx‖ ≤ c·L(n)·u·‖x‖. F가 √n × 유니터리이므로 이것은 ‖ŷ − y‖ ≤ c·L(n)·u·‖y‖와 같다. 그래서 계약은 노름 상대오차이고, 각 원소는 그 노름 한계를 절대오차의 한계로 갖는다(|ŷₖ − yₖ| ≤ ‖ŷ − y‖).

### 4.2 변환마다의 꼴
T는 정확한 선형 변환(정규화 인자 s 포함), y = Tx는 참값, ŷ는 돌려준 값이다. 모든 한계에 τ = κ·m·2⁻¹⁰⁷⁴를 더한다(m은 변환의 길이: n, 다차원은 원소 수, DCT·DST는 2N + 2; FT31). 비정상 수끼리의 곱은 부분마다 2⁻¹⁰⁷⁵까지 틀리므로 한 단이 노름 √2·2⁻¹⁰⁷⁴·√n까지의 절대오차를 더하고, 기수 rᵢ의 뒤 단들이 그것을 √(Πrᵢ)배 한다. 단마다 더하면 3.5·√2·n·2⁻¹⁰⁷⁴ 아래라 κ·n·2⁻¹⁰⁷⁴ 안이고, 비정상 수인 결과의 마지막 반올림도 그 안이다(2⁻¹⁰⁷⁴·√(출력 원소 수)만 두면 numpy·scipy가 비정상 수 입력의 n = 127에서 그 12.5배를 냈다).

| 변환 | 계약 | ‖T‖·‖x‖의 뜻 |
|---|---|---|
| 복소 `fft ifft` | ‖ŷ − y‖ ≤ (κ·L(n) + c₀)·u·‖y‖ + τ | ‖y‖ = s·√n·‖x‖ (정확히) |
| 복소 다차원 (FT49) | 같은 꼴, L(n) 대신 Σ L(nᵢ)(축마다; Bluestein 축은 ρᵢ·(2·L(mᵢ) + 2)), c₀ 대신 maxᵢρᵢ·c₀ + (축 수 − 1)·c₁, 바닥 τ(N)·Πᵢ(ρᵢ·m′ᵢ/nᵢ) | ‖y‖ = s·√(Πnᵢ)·‖x‖ |
| `rfft ihfft` (반쪽 스펙트럼) | ‖ŷ − y‖ ≤ (κ·L(n) + c₀)·u·s·√n·‖x‖ + τ | 반쪽의 노름은 전체 스펙트럼 노름 s√n‖x‖의 1/√2와 1배 사이다 |
| `irfft hfft` | ‖ŷ − y‖ ≤ (κ·L(n) + c₀)·u·s·√n·‖X̃‖_H + τ | X̃는 정리한 입력(n//2 + 1칸, DC·나이퀴스트의 허수부 버림), ‖X̃‖_H² = \|X̃₀\|² + 2Σ\|X̃ₖ\|² (+ \|X̃_{n/2}\|², 짝수 n): 에르미트 확장의 노름 |
| `rfftn irfftn hfftn ihfftn` (FT49) | 복소 다차원의 꼴에서 마지막 축의 항이 실수 길의 것(안쪽 길이가 Bluestein이면 FT45의 꼴) | `rfftn ihfftn`: s·√N·‖x‖. `irfftn hfftn`: s·√N·‖X‖_W(마지막 축의 칸에 무게 1 또는 √2, 허수부를 버리지 않는다: 복소 패스가 먼저다). 증명: 복소 패스는 마지막 축의 칸마다(열마다) 작용하므로 ‖·‖_W를 √(Πnᵢ)배 하고, 실수 패스의 c2r은 ‖R(v)‖ ≤ √n·‖v‖_W(허수부 버리기는 ‖·‖_W에서 축약)이라 앞 패스들의 오차를 √n배까지만 옮긴다 |
| `dct dst idct idst` | ‖ŷ − y‖ ≤ (κ·L(M) + c₂)·u·‖T‖·‖x‖ + τ | M = 2N + 2(모든 형의 안쪽 실수 FFT 길이의 위 한계, FT31). ‖T‖는 형과 `norm`·`orthogonalize`마다 닫힌 꼴이다: 직교 판은 1, `'backward'`의 정변환은 DCT-II·DST-II 2√N, DCT-III·IV·DST-III·IV √(2N), DST-I √(2(N + 1)), DCT-I 2√(N − 1) 이하(대각 배율 곱하기 직교 행렬이다; `refdata.norm_r2r`). `orthogonalize`의 행렬은 √N′ × 직교라 이 값 아래다 |
| DCT·DST의 Bluestein 안쪽 길이 (FT53) | 안쪽 복소 길이 h(`refdata.r2r_inner`: DCT-I N − 1, DST-I N + 1, 형 2·3과 4는 짝수 N이면 N/2, 홀수 N이면 N)가 Bluestein이면 L(M) 대신 2·L(m) + 3, 한계에 ρ(h)를 곱하고 바닥은 ρ(h)·τ(max(m, M)) | m은 h의 안쪽 길이 |
| `fht ifht` (FT62) | ‖Â − A‖ ≤ (2ε_F + c_h)·u·‖b_out‖_∞·M·‖B_in a‖ + (‖b_out‖_∞(M + 1) + 1)·2τ_F | ε_F·τ_F는 길이 n 실수 변환의 (κ·L(n) + c₀)와 τ(n)(안쪽 길이가 Bluestein이면 FT45의 꼴), M = max\|U_m\|(`ifht`는 max 1/\|U_m\|), B_in·b_out은 바이어스(q = 0이면 1). q = 0이면 ‖A‖ = ‖a‖라 노름 상대오차다 |
| 실수 입력 `fft` (전체 스펙트럼, FT45) | `rfft` 줄과 같은 꼴 | 사후 처리의 오차를 거울에 비춘 것도 그 한계 안이다(아래) |
| 실수·에르미트의 Bluestein 안쪽 길이 (FT45) | 안쪽 길이 h(짝수 n은 n/2, 홀수 n은 n)의 ρ(h)와 m(h)로 L(n) 대신 2·L(m) + 2 + [n 짝수], 한계에 ρ(h)를 곱하고 바닥은 ρ(h)·τ(max(m, n)) | |
| Bluestein의 길이 (FT36, FT37) | 위의 꼴에서 L(n) 대신 L_B(n) = 2·L(m) + 2, 한계와 τ(τ는 m으로 센다)에 ρ(n)을 곱한다 | m은 안쪽 길이(2n − 1 이상인 가장 작은 2ᵃ3ᵇ5ᶜ), ρ(n) = maxₖ\|(F_m b)ₖ\|/√n(chirp 핵 b의 변환의 최대 크기; 계획이 double-double로 구한 상수이고 `tests/fft/refdata.bluestein_rho`가 따로 구해 견준다; 1–128의 Bluestein 길이와 1009, 2018, 4097, 10007, 30021, 65537, 100003에서 1.98–2.20, 가장 큰 값은 37)이다 |

- **κ와 c** (FT31, FT64): κ = 8, c₀ = 4, c₁ = 2, c₂ = 8. 기수 13–31이 하나라도 있는 (안쪽) 길이는 모든 꼴에서 κ 대신 κ_w = 17이고 그 바닥도 17·m·2⁻¹⁰⁷⁴다(`refdata.kappa`, `KAPPA_WIDE`). FFTLog의 c_h = 8(FT62)이다.
- **증명의 틀**: 단 j가 길이 L의 부분 변환들에 기수 rⱼ의 나비와 회전 인자의 대각을 곱하는 것이면, 정확한 단은 √rⱼ × 유니터리다. 계산한 단이 ẑ = 단(z) + e, ‖e‖ ≤ ηⱼ·√rⱼ·‖z‖를 지키면 ‖ŷ − y‖ ≤ (Πⱼ(1 + ηⱼ) − 1)·‖y‖이다. ηⱼ ≤ κ·log₂(rⱼ)·u이면 Σⱼ log₂ rⱼ = log₂ n ≤ L(n)이라 곱이 (κ·L(n) + O(L²u))·u 아래다(어느 인수분해든 log₂ rⱼ의 합은 log₂ n이다). 그래서 증명할 것은 나비마다의 ηⱼ/log₂ rⱼ ≤ κ·u 하나다. 회전 인자의 오차는 μ ≤ 1.5u로 둔다(계획들의 표는 1.04u 이하였다, §3.3).
  - **나비마다의 증명** (`tests/fft/proof.py`, `tests/fft/test_proof.py`): 나비를 C++가 계산하는 연산 그대로 따라가며 모든 중간값 v(실수부나 허수부 하나)에 정확한 선형식 L_v(v = L_v·x)와 오차 한계 E_v ≥ 0(1차까지 \|v̂ − v\| ≤ E_v·\|x\|)을 둔다. 덧셈은 E_v = E_a + E_b + u\|L_v\|, 반올림한 상수 ĉ = c + e의 곱은 \|c\|E_a + \|e\|·\|L_a\| + u\|L_v\|, FMA c·a + b는 \|c\|E_a + E_b + \|e\|·\|L_a\| + u\|L_v\|이다(−i 곱과 부호는 정확하다; 상수의 오차 e는 Arb로 구한 정확한 값). 출력의 E를 모은 음이 아닌 행렬 M에 대해 ‖ŷ − y‖ ≤ ‖M\|x\|‖ ≤ ‖M‖₂‖x‖이고 정확한 나비는 ‖y‖ = √r‖x‖이므로 η_b = ‖M‖₂/√r다. 단은 그 뒤에 회전 인자를 곱한다: FMA 복소 곱이 2u(Jeannerod·Kornerup·Louvet·Muller 2017), 회전 인자의 오차가 μ = 1.5u라 η_단 = η_b + (2u + μ)(1 + η_b)다. 2차 항(반올림마다의 (1 + δ)의 곱)은 한 경로의 반올림이 30개 아래라 한계를 10⁻¹⁴만큼도 바꾸지 않는다.

| 기수 r | η_b | η_단 | η_단/log₂ r |
|---:|---:|---:|---:|
| 2 | 1.414u | 4.914u | 4.914u |
| 4 | 4.000u | 7.500u | 3.750u |
| 3 | 5.768u | 9.268u | 5.847u |
| 5 | 9.912u | 13.412u | 5.776u |
| 7 | 13.669u | 17.169u | 6.116u |
| 11 | 22.875u | 26.375u | 7.624u |
| 13 | 26.921u | 30.421u | 8.221u |
| 17 | 37.424u | 40.924u | 10.012u |
| 19 | 42.930u | 46.430u | 10.930u |
| 23 | 54.338u | 57.838u | 12.786u |
| 29 | 72.666u | 76.166u | 15.679u |
| 31 | 79.861u | 83.361u | 16.826u |

  - 그래서 혼합 기수의 길이 n은 ‖ŷ − y‖ ≤ (κ·log₂ n + O(u))·u·‖y‖ ≤ κ·L(n)·u·‖y‖다. 기수 13–31은 κ = 8을 넘고 κ_w = 17 안이라 그런 기수가 있는 길이는 κ_w·L(n)으로 센다(FT64). 4단계는 열 변환의 마지막 단에 없는 회전 인자 곱을 4단계의 회전 인자 곱이 대신하므로 단의 셈이 그대로다(§3.3). 정규화는 c₀의 몫이다: fl(1/n)과 곱(2u), `'ortho'`의 fl(1/fl(√n))과 곱(3u)이 c₀ = 4 안이다. 들어오고 나가는 엮기와 풀기는 정확하다.
  - **Bluestein** (FT37): 정확한 분해 y = P·D_w·F*·D_β·F·E·D_w·x(F는 정규화 없는 길이 m의 DFT, F*F = mI, E는 0 채우기, P는 앞 n개)의 다섯 단계를 따라간다. (1) chirp 곱: chirp는 double-double에서 한 번 반올림해 원소마다 u 안이고 복소 곱이 2u라 ‖â − a‖ ≤ 3.5u‖x‖. (2) 정변환: η_m√m‖a‖(η_m = κ·log₂ m·u). (3) β와의 곱: β̂ₖ는 βₖ의 u 안(β의 double-double 오차는 2·L(m)·√n·2⁻¹⁰⁴·max\|β\|/ρ 정도라 계획을 만들 수 있는 n < 2³⁰에서 2⁻⁸⁰·max\|β\| 아래)이고 복소 곱이 2u라 3u·max\|β\|·‖Fa‖. (4) 역변환: η_m√m‖D_βFa‖. (5) chirp 곱: 3.5u‖y‖. ‖F‖ = ‖F*‖ = √m, ‖D_β‖ = max\|β\|이고 ρ = m·max\|β\|/√n로 쓰면 ‖ŷ − y‖ ≤ (ρ(2κ·log₂ m + 6.5) + 3.5)·u·‖y‖ ≤ ρ·(κ(2L(m) + 2) + c₀)·u·‖y‖다(ρ ≥ 1: ‖y‖ ≤ ‖D_β‖·m·‖x‖). β를 double로 변환하면 (3)의 오차를 u·log m·√m·max\|β\|로만 묶을 수 있어 이 증명이 서지 않는다: double-double로 계산하는 이유다.
  - **실수 길** (`csrc/fft/rfft.hpp`의 주석): 짝수 n에서 Z ↦ (X_0, …, X_{n−1})는 √2 × 유니터리다(|X_k|² + |X_{k+h}|² = 2(|E_k|² + |O_k|²), Σ(|E_k|² + |O_k|²) = ‖Z‖²). 그래서 안쪽 변환의 오차 ε_h‖Z‖(‖Z‖ = √h‖x‖)는 전체 스펙트럼에서 ε_h·√n·‖x‖이고, 혼합 기수 h면 ε_h = κ·L(h)·u = κ(L(n) − 1)u라 계약의 κ·L(n)·u에서 κ·u 하나가 사후 처리의 몫으로 남는다. 사후 처리는 짝마다 |X̂_k − X_k| ≤ 2u|e| + 5.5u|d|(합의 반올림 u(|e| + |d|), e의 u|e|, d의 u|d|가 곱을 거친 것, FMA 복소 곱 2u, 회전 인자 μ = 1.5u)이고 짝들에 대해 Σ(|e_k|² + |d_k|²) = ‖Z‖²/2이므로 전체 스펙트럼에 √(2(2² + 5.5²))·u·‖Z‖ = 5.85u·√h·‖x‖ 이하를 더한다: κ·u·√n·‖x‖ 아래다. 반쪽 스펙트럼(`rfft`)은 그 일부이고, 실수 입력 `fft`의 거울 채우기는 정확한 켤레라 전체 스펙트럼의 오차가 그대로다. 역변환은 같은 곱과 합을 거꾸로 하므로 같은 셈이 선다(입력의 노름이 ‖X̃‖_H). Bluestein h는 §4.2의 Bluestein 증명에 남는 여유(ρ ≥ 1이라 ρ(κ(2L(m) + 2) + c₀) − (ρ(2κ·log₂ m + 6.5) + 3.5) ≥ 2κ − 6 = 10u)에 κ·u 하나를 더한 것(FT45의 + 1)이 사후 처리와 정규화를 덮는다. 홀수 n은 길이 n의 복소 변환 뒤의 에르미트 사영 P(‖P‖ = 1, Py = y)라 ‖Pŷ − y‖ = ‖P(ŷ − y)‖ ≤ ‖ŷ − y‖이고, 사영의 반올림(칸마다 u)이 c₀ 안이다. 저장한 실수 참값 282줄과 에르미트 참값 113줄의 잰 값은 §4.4다.
  - **DCT·DST** (`csrc/fft/r2r.hpp`의 주석): 안쪽 실수 FFT는 전체 스펙트럼에서 제 계약(κ·L(n) + c₀)u·√n·‖v‖을 지키고(FT45), 스펙트럼에서 y로 가는 사상은 에르미트 스펙트럼 위에서 등거리의 배수이며 그 배수가 ‖T‖/(√n·‖v‖)을 넘지 않는다. DCT-I은 앞 N칸의 실수부라 ‖v‖ ≤ √2‖x‖, ‖T‖ ≤ 2√(N − 1)이고, DST-I은 N칸의 허수부인데 그 거울 칸이 같은 오차를 가지므로 전체 스펙트럼 오차의 1/√2 이하다. DCT-II는 2·Re(w_k V_k)라 2 × 등거리(‖T‖ = 2√N), DCT-III은 그 역이고 ‖V‖_H ≤ √2‖x‖, 홀수 N의 DCT-IV는 √2 × 등거리(‖T‖ = √(2N)), 짝수 N의 DCT-IV는 크기 1의 두 대각 사이의 길이 N/2 복소 FFT다. 안쪽 길이는 M 이하이고 그 수준 수는 L(M) − 1 이하라 κu 하나가 남고, 앞뒤 처리(회전 인자 곱 2u + μ = 3.5u, 덧셈 하나, √2 곱)와 `orthogonalize`의 √2가 그것과 c₂ = 8 안이다. 넘침의 정의역도 같다: 확장과 2배가 중간값을 ‖x‖₁의 4배까지만 키운다.
  - c₀는 정규화 곱셈, 실수 변환의 사후 처리(회전 인자 한 번과 덧셈 둘), 에르미트 대칭 채우기의 몫이고 c₂는 DCT·DST의 앞뒤 처리(순서 바꾸기와 2e^(−iπk/2N)의 곱)의 몫이다. 상수를 고치면 이 표와 `tests/fft/refdata.py`를 함께 고친다.
  - **FFTLog** (`csrc/fft/fftlog.cpp`의 주석): v = B_in a, X = Fv(실수 DFT의 반쪽), y = R(U′X)(U′ = U/n, R은 에르미트 반쪽 스펙트럼의 정규화 없는 역실수 DFT), A = B_out·P·y(P는 뒤집기)로 쓰면 ‖Fv‖_H = √n‖v‖, ‖RW‖ = √n‖W‖_H라 ‖y‖ ≤ M‖v‖다. 계산한 v̂는 ‖v̂ − v‖ ≤ e₁‖v‖(e₁ = 바이어스의 u/2 + 2⁻⁵⁹과 곱의 u/2), 실수 변환은 ‖X̂ − Fv̂‖_H ≤ ε_F√n‖v̂‖, 계수의 곱은 칸마다 (1 + d_m), \|d_m\| ≤ u/2(계수의 반올림) + 2u(FMA 복소 곱) + 2⁻⁵⁹, 역변환은 ‖ŷ − RŴ‖ ≤ ε_F√n‖Ŵ‖_H, 출력의 곱이 다시 e₁이다. 더하면 ‖Â − A‖ ≤ ‖b_out‖_∞·M·‖v‖·(2ε_F + 2e₁ + max\|d_m\| + O(u²))이고 2e₁ + max\|d_m\| ≤ 4.5u + 2⁻⁵⁷이 c_h = 8 안이다. 바닥은 두 변환의 τ_F(M·‖b_out‖_∞와 ‖b_out‖_∞배)와 세 번의 곱의 언더플로(칸마다 2⁻¹⁰⁷⁵)를 모은 것이다. 계수의 double-double 오차(로그와 지수 2⁻⁹⁶, 정의역 FT62 안에서 \|w log w\|·2⁻¹⁰⁰ ≤ 2⁻⁶⁰)는 계수의 u/2에 비해 보이지 않는다.
- **넘침의 정의역**: 계약은 중간값이 넘치지 않을 때다. 모든 중간값은 부분합이라 ‖x‖₁(실수부·허수부의 절댓값 합) ≤ 2¹⁰²⁰이면 넘치지 않는다. 그 밖에서 inf·NaN이 나오는 것은 계약 밖이다(참값이 넘칠 수도 있다).

### 4.3 vDSP 길의 계약
vDSP는 쓰지 않으므로(FT30, FT66) 모든 길이의 계약이 증명된 한계다. vDSP는 닫힌 소스라 최악의 한계를 증명할 수 없다: 다시 쓰게 되면 그 꼴은 같은 계약(FT1)을 측정으로만 뒷받침하게 되고(정확도 체계가 vDSP가 받는 모든 길이 꼴과 적대적 입력에서 계약 단위를 재고 기준선보다 나빠지면 실패), 탐침에서 잰 vDSP의 값은 계약의 0.025–0.049다(§4.4의 끝).

### 4.4 잰 값
정확도 체계(`bench/accuracy/fft.py`, 기준선 `bench/accuracy/results/fft.json`, 표 `bench/fft/accuracy.md`; 2026-11-23의 빌드)가 Arb와 견준 계약 단위(오차 ÷ §4.2의 한계)의 최댓값이다. 점은 한 줄이다. 영역은 길의 꼴이다: `pow2`(2ᵏ), `f2k`(f·2ᵏ, f ∈ {3, 5, 15}, k ≥ 3), `mixed`(그 밖의 31-매끄러운 길이), `bluestein`(31보다 큰 소인수), 실수는 안쪽 길이로 `even`·`odd`·`bluestein`, 다차원은 모든 축이 31-매끄러우면 `smooth`, DCT·DST는 안쪽 복소 길이로 `even`·`odd`·`bluestein`이다.

| 스윕 | 점 | sd | numpy | scipy | R `fft` |
|---|---:|---:|---:|---:|---:|
| `c2c` 정방향 pow2 / f2k / mixed / bluestein | 81 / 40 / 144 / 52 | 0.040 / 0.041 / 0.054 / 0.035 | 0.041 / 0.033 / 0.054 / 0.032 | 0.041 / 0.033 / 0.054 / 0.032 | 0.084 / 0.086 / 0.108 / 2.86 |
| `c2c` 역방향 pow2 / f2k / mixed / bluestein | 79 / 40 / 144 / 50 | 0.037 / 0.042 / 0.054 / 0.030 | 0.042 / 0.034 / 0.054 / 0.031 | 0.042 / 0.034 / 0.054 / 0.031 | — |
| `real`: `rfft` pow2 / even / odd / bluestein | 77 / 94 / 74 / 40 | 0.042 / 0.030 / 0.030 / 0.016 | 0.042 / 0.034 / 0.030 / 0.0087 | 0.042 / 0.034 / 0.030 / 0.0083 | — |
| `real`: 실수 입력 `fft` pow2 / even / odd / bluestein | 77 / 94 / 74 / 40 | 0.042 / 0.042 / 0.039 / 0.023 | 0.042 / 0.039 / 0.051 / 0.029 | 0.042 / 0.048 / 0.039 / 0.012 | — |
| `hermitian`: `irfft` pow2 / even / odd / bluestein | 37 / 32 / 38 / 9 | 0.049 / 0.049 / 0.033 / 0.0079 | 0.030 / 0.043 / 0.038 / 0.011 | 0.030 / 0.043 / 0.038 / 0.0099 | — |
| `nd`: `fftn` / `ifftn` / `rfftn` / `irfftn` (smooth) | 26 / 26 / 32 / 32 | 0.034 / 0.034 / 0.029 / 0.044 | 0.036 / 0.036 / 0.032 / 0.042 | 0.036 / 0.036 / 0.032 / 0.042 | — |
| `nd`: `fftn` / `ifftn` / `rfftn` / `irfftn` (bluestein)¹ | 16 / 16 / 16 / 16 | 0.042 / 0.044 / 0.040 / 0.037 | 0.012 / 0.012 / 0.012 / 0.020 | 0.012 / 0.012 / 0.012 / 0.020 | — |
| `fftlog`: `fht` / `ifht`, 바이어스 없음 | 95 / 95 | 0.030 / 0.030 | — | 2,743 / 4,615 | — |
| `fftlog`: `fht` / `ifht`, 바이어스 | 74 / 74 | 0.069 / 0.036 | — | 31.3 / 31.3 | — |

| `r2r` 형 | 점 even / odd / bluestein | sd | scipy |
|---|---|---:|---:|
| DCT-I | 65 / 62 / 12 | 0.029 / 0.025 / 0.0065 | 0.024 / 0.025 / 0.0055 |
| DCT-II | 75 / 58 / 11 | 0.033 / 0.028 / 0.0057 | 0.029 / 0.030 / 0.0075 |
| DCT-III | 75 / 58 / 11 | 0.040 / 0.036 / 0.0098 | 0.038 / 0.043 / 0.011 |
| DCT-IV | 75 / 58 / 11 | 0.052 / 0.035 / 0.0086 | 0.045 / 0.042 / 0.011 |
| DST-I | 58 / 66 / 20 | 0.039 / 0.042 / 0.0083 | 0.029 / 0.034 / 0.0098 |
| DST-II | 75 / 58 / 11 | 0.031 / 0.042 / 0.0066 | 0.027 / 0.045 / 0.0080 |
| DST-III | 75 / 58 / 11 | 0.061 / 0.052 / 0.0084 | 0.045 / 0.046 / 0.011 |
| DST-IV | 75 / 58 / 11 | 0.052 / 0.048 / 0.0099 | 0.052 / 0.044 / 0.011 |

- 점은 저장한 참값(`c2c` 정방향 313줄·역방향 309줄, 실수 282줄, 에르미트 113줄, 다차원 10개, r2r 140줄)과 Arb를 그 자리에서 구한 긴 줄(c2c 2¹⁶·65537·3·10007·2²⁰, 실수 2¹⁶·65537·2²⁰과 그 반쪽 스펙트럼, r2r 1009·2·1009·2¹⁶·65537), 다차원 점(`refdata.nd_points`: 복소 모양 여덟, 실수 모양 열둘, 반쪽 스펙트럼 열둘; 그 가운데 넷씩은 31보다 큰 소인수의 축 37, 41, 43, 74 = 2·37이 다른 축, 마지막 축, 실수 축의 안쪽 길이에 있는 모양), FFTLog 338점(`refdata.fht_points`: 길이 1–4096과 2¹⁶ × `refdata.FHT_PARAMS`의 아홉 묶음 × 두 방향)이다.
- ¹ 다차원의 bluestein 영역은 2026-11-23의 실행(`--compare`, 기준선 승격 전)에서 잰 값이다. 기수 13–31이 혼합 기수가 된 뒤(FT64) 비었던 영역을 31보다 큰 소인수의 축으로 채웠다. 가장 큰 값은 비정상 수 입력의 (3, 41)이고, 노름 상대오차는 sd 약 10⁻⁴, numpy·scipy 3×10⁻⁵다(계약이 ρ로 넓어 단위는 작다).
- 가장 큰 값은 모두 비정상 수 입력(`tiny`), 넘침 직전(`huge`), 짧은 줄이다(바닥 τ와 언더플로가 정한다). 균등 난수 입력만 보면 0.04 아래다. 계약을 넓힌 곳은 없고, 기수 13–31의 길이(κ_w = 17)는 `mixed`·`odd`·`smooth` 영역에 들어 있다.
- Bluestein 영역은 계약이 ρ로 넓어 단위가 작다. 노름 상대오차의 최댓값은 비정상 수 입력의 짧은 줄에서 10⁻⁴ 크기다(c2c 정방향 sd 9.5×10⁻⁵, numpy·scipy 8.5×10⁻⁵; 실수 `rfft`는 sd 8.3×10⁻⁵, numpy·scipy 2.8×10⁻⁵).
- `irfftn`의 점은 에르미트 대칭이 아닌 입력이다: 참값은 numpy·scipy의 순서(복소 패스 뒤에 DC·나이퀴스트 허수부를 버리고 n//2 + 1칸만 읽음)로 Arb에서 구했고(`refdata.arb_irfftn`), 세 라이브러리가 모두 그 참값의 계약 안이다.
- DCT·DST의 노름 상대오차는 sd 2.0–4.5×10⁻¹⁶, scipy 2.0–8.3×10⁻¹⁶이다(Bluestein 영역에서 sd가 작다). 저장한 묶음에 없는 값(비정상 수, 넘침 직전, 칸 위와 칸 사이의 사인파; 길이 2, 5, 16, 26, 127)은 테스트가 Arb를 그 자리에서 구해 재고 계약의 0.052 이하다(`tests/fft/test_realtransforms.py`). 모든 `norm`과 `orthogonalize`(참값은 `refdata.r2r_reference`가 정확한 √2와 인자로 저장한 참값에서 구한다)에서 계약의 0.063 이하다.
- FFTLog의 계수 자체는 Arb와 견주어 u 이하다(부분마다 한 번의 반올림, 잰 최대 0.98u; `tests/fft/test_fftlog.py`). `fhtoffset`은 Arb 값의 반 ulp 안이다. scipy의 오차는 계수의 위상(double의 `loggamma`와 2y(ln 2 − offset))에서 온다: n = 4096, dln = ln(10⁸)/4095, offset = −6 ln 10의 정규 값에서 이 계약의 28.9배, dln = 10⁻⁴(y가 15,708까지)에서 4,615배다. 같은 점에서 sd는 0.014–0.03이다. 노름 상대오차로 sd 3.3×10⁻¹⁶, scipy 10⁻¹⁴–3×10⁻¹¹이다. 바이어스가 있는 `ifht`의 노름 상대오차가 큰 점(최대 9.7)은 결과가 바이어스의 동적 범위에 비해 작은 곳이다(조건수가 큰 문제; scipy도 9.2). 계약은 ‖b_out‖_∞·M·‖B_in a‖로 재므로 그 안이다.
- R의 `fft`는 큰 소인수를 O(p²)의 합으로 계산해 Bluestein 영역에서 sd의 계약을 넘는다(2.86배; sd의 계약은 R에 대한 약속이 아니다).

**vDSP·시제품과 견준 탐침** (`tools/fft_probe.py`의 `accuracy`; vDSP를 쓰지 않기로 한 근거의 하나다)

균등 [−1, 1)의 실수부·허수부, 한 줄의 변환, 저장한 참값(`tests/fft/ref`)과 견준 값이다. 계약 단위(괄호 안은 ‖ŷ − y‖/(u·log₂n·‖y‖))이고 기록은 `tests/fft/data/prototype.json`의 `accuracy/…`다.

| n | 종류 | numpy | scipy | 자체 시제품 | vDSP |
|---:|---|---:|---:|---:|---:|
| 1024 | 복소 | 0.022 (0.19) | 0.022 (0.19) | 0.020 (0.17) | 0.030 (0.25) |
| 4096 | 복소 | 0.021 (0.18) | 0.021 (0.18) | 0.019 (0.16) | 0.028 (0.23) |
| 16384 | 복소 | 0.021 (0.17) | 0.021 (0.17) | 0.018 (0.15) | 0.027 (0.23) |
| 1536 | 복소 | 0.024 (0.21) | 0.024 (0.21) | 0.022 (0.19) | 0.043 (0.37) |
| 1280 | 복소 | 0.021 (0.19) | 0.021 (0.19) | 0.020 (0.18) | 0.046 (0.41) |
| 3840 | 복소 | 0.023 (0.19) | 0.023 (0.19) | 0.021 (0.18) | 0.044 (0.37) |
| 4096 | 실수 | 0.014 (0.16) | 0.015 (0.18) | 0.014 (0.16) | 0.020 (0.23) |
| 2560 | 실수 | 0.014 (0.17) | 0.015 (0.18) | 0.014 (0.18) | 0.029 (0.37) |

- 클래스 길이 전부(복소 29개, 실수 13개)의 최댓값: numpy·scipy 0.031, 자체 시제품 0.035, vDSP 0.049 단위다(u·log₂n으로 0.13–0.28, 0.15–0.33, 0.17–0.44).
- 저장한 모든 참값(다섯 묶음, 858줄: 넘침 직전·비정상 수·넓은 크기의 입력 포함)에서 numpy·scipy의 최댓값은 복소 0.136(비정상 수 입력의 n = 127), 실수 0.042, 에르미트 0.044, scipy의 DCT·DST 0.052 단위다(`tests/fft/test_reference_data.py`).
- R의 `fft`는 n = 8, 24, 1000에서 0.05–0.09 단위이고 소수 n = 1009에서 0.46 단위다(Singleton의 혼합 기수가 큰 소인수를 O(p²)의 합으로 계산한다).
- 증명된 한계는 무작위 입력의 실제 오차보다 20–40배 느슨하다(오차는 무작위 걸음처럼 √L로 자라고 한계는 L로 센다). 그래서 계약만으로는 작은 정확도 회귀를 못 잡는다. 정확도 체계의 기준선(최대와 제곱평균의 계약 단위)이 그 몫을 한다(bench/README.md §2). 더 촘촘한 경험적 계약은 두지 않았다(§9의 23번).
- 비교 패키지(같은 입력에서 그 자리에서): numpy, scipy, R `fft`. 계약이 아니라 정확도 표의 비교 열로 보인다.

### 4.5 비트로 지키는 성질 (반올림 계약이 아니라 정확한 약속)
- 한 줄의 결과는 배치, 레이아웃, 스레드 수, Accelerate 모드와 무관하다(§3.7).
- 길이 1의 변환은 정확하다. 2의 거듭제곱 배율은 넘침·언더플로가 없으면 정확히 옮겨진다: `fft(2·x)`는 `2·fft(x)`와 비트까지 같다(같은 연산에 지수만 다르다).
- `rfft(x)`는 실수 입력의 `fft(x)`의 앞쪽과, `ihfft`·`hfft`는 정의의 켤레·정규화와(`ihfft(x, norm=a)`는 `conj(rfft(x, norm=a′))`, `hfft(x, norm=a)`는 `irfft(conj(x), norm=a′)`, a′는 방향을 바꾼 `norm`), 실수 입력의 `ifft(x, norm=a′)`는 `conj(fft(x, norm=a))`와 비트까지 같다(§3.4). 실수 입력 `fft`의 y_{n−k} = conj(y_k)는 정확하다. `irfft`의 결과는 버리는 허수부(DC와 짝수 n의 나이퀴스트)와 무관하다.
- 다차원: 축 하나의 n차원 함수는 1차원 함수와, `fftn(x)`(`'backward'`)와 `ifftn(x, norm='forward')`는 1차원 변환을 `axes`의 뒤에서 앞으로 이은 것과, `rfftn(x)`는 실수 입력 `fftn(x)`의 앞쪽 n//2 + 1칸과, `hfftn`·`ihfftn`은 켤레와 방향을 바꾼 `irfftn`·`rfftn`과 비트까지 같다(FT48, FT52). `irfftn`의 결과는 마지막 축의 n//2 + 1칸 밖과 무관하다.
- DST-II·III·IV는 부호를 바꾸고 뒤집은 입력의 DCT와 비트까지 같다(§3.5). `idct`·`idst`의 형 k′(2 ↔ 3)에 `norm='forward'`는 `dct`·`dst`의 형 k(`'backward'`)와, 복소 입력의 `dct(x)`는 `dct(x.real) + 1j·dct(x.imag)`와, 축 하나의 `dctn`은 `dct`와, 인자 1의 `dctn`(`'backward'`)·`idctn`(`'forward'`)은 1차원 변환을 `axes`의 뒤에서 앞으로 이은 것과 비트까지 같다(FT57).
- FFTLog: `fht`·`ifht`의 한 줄은 그 줄만 변환한 것과 비트까지 같다(배치, 레이아웃, 정수·bool 입력, 스레드 수, 계획을 만든 스레드 수와 무관하다). 계획은 키로만 정해져 캐시에서 쫓겨난 뒤 다시 만들어도 같은 비트다.
- `fftshift`·`ifftshift`는 `roll`과, `fftfreq`·`rfftfreq`는 numpy와 비트까지 같다.

---

## 5. 성능
시간은 µs이고 가장 좋은 값이다. numpy·scipy는 Python 호출과 결과 할당을 포함하고 따로 적지 않으면 1스레드다(numpy의 pocketfft는 언제나 1스레드, scipy는 기본 `workers=1`). sd는 기본 10스레드이고, 4단계 아래의 한 줄은 한 태스크라 1스레드다. 기기를 다른 작업과 나눠 썼다. 속도 사례는 `bench/speed/cases/fft.py`의 66개이고(이 절의 크기를 담는다; scipy는 `workers=1`과 `workers=10` 둘 다), 공식 측정은 `bench/speed`로 사용자가 다시 한다. 그때까지는 이 절의 표가 기준선이다.

**지금의 빌드** (2026-11-23; 같은 프로세스 안의 가장 좋은 값을 두 번 따로 돌린 것 가운데 작은 값. "뒤"가 지금의 빌드이고 "전"은 기수 13–31, 4단계, 두 줄 커널과 타일 복사(FT64, FT65, FT67)를 넣기 전의 빌드다. 둘을 번갈아 쟀고 numpy·scipy는 같은 날 같은 방법이다)

| 일 | 전 1스레드 | 뒤 1스레드 | 전 10스레드 | 뒤 10스레드 | numpy | scipy | scipy `workers=10` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fft` 2¹⁶ | 232 | 259 | 242 | 47.4 | 450 | 429 | 178 |
| `fft` 2¹⁸ | 1,137 | 1,164 | 1,183 | 175 | 2,112 | 1,316 | 392 |
| `fft` 2²⁰ | 5,163 | 6,143 | 5,349 | 882 | 10,432 | 7,136 | 1,619 |
| `fft` 2²² | 22,674 | 29,452 | 23,557 | 4,035 | 51,052 | 35,066 | 6,516 |
| `fft` 3·2¹⁸ / 5·2¹⁷ | 3,791 / 3,046 | 4,150 / 3,303 | 3,920 / 3,170 | 604 / 449 | 7,328 / 5,392 | 4,792 / 4,046 | 1,144 / 960 |
| `fft` 36960 (11-매끄러운, 4단계) | 107 | 168 | 110 | 35.8 | 210 | 244 | 124 |
| `fft` 100003 (Bluestein, 안쪽 4단계) | 1,348 | 2,141 | 1,392 | 349 | 4,390 | 4,846 | 2,708 |
| `fft` 13 / 34 (FT64) | 0.27 / 0.43 | 0.21 / 0.28 | — | — | 1.62 / 1.79 | 1.80 / 1.91 | — |
| `fft` 1001 / 4095 (기수 13) | 9.85 / 53.9 | 2.83 / 11.3 | — | — | 7.40 / 26.3 | 6.74 / 24.0 | — |
| `fft` 32736 = 3·11·31·32 | 522 | 181 | 555 | 186 | 347 | 275 | 113 |
| `rfft` 1001 / 4095 (기수 13) | 10.5 / 54.1 | 3.21 / 13.1 | — | — | 5.82 / 17.1 | 4.42 / 13.2 | — |
| `rfft` 2¹⁶ / 2²⁰ | 134 / 2,851 | 142 / 2,776 | 141 / 2,996 | 40.8 / 409 | 212 / 4,105 | 249 / 5,379 | 136 / 1,521 |
| `irfft` 2¹⁶ | 149 | 157 | 153 | 53.1 | 245 | 249 | 143 |
| `fft` (1000, 1024) 축 0 (타일) | 3,113 | 2,962 | 406 | 404 | 5,469 | 2,945 | 571 |
| `fft` (1024, 1000) 축 0 (두 줄) | 2,506 | 2,202 | 410 | 269 | 2,560 | 2,461 | 388 |
| `fft` (256, 256, 129) 축 1 (두 줄) | 22,673 | 16,254 | 3,055 | 2,908 | 22,271 | 20,733 | 3,455 |
| `fft` (34, 1009) 축 0 (길이 34, FT64) | 253 | 67.2 | 35.5 | 11.9 | 113 | 97.0 | 48.0 |
| `fft2` (2048, 2048) | 23,694 | 22,799 | 3,333 | 3,252 | 42,173 | 23,719 | 3,231 |
| `fftn` (64, 64, 64) / (32, 32, 32) | 1,210 / 165 | 1,064 / 142 | 197 / 33.5 | 174 / 30.8 | 1,909 / 255 | 1,218 / 153 | 319 / 109 |
| `rfftn` (256, 256, 256) | 55,741 | 46,544 | 7,658 | 6,370 | 67,911 | 43,777 | 6,800 |
| `irfftn` (128, 128, 65) | 6,441 | 5,692 | 982 | 906 | 8,213 | 6,217 | 994 |
| `fft2` (34, 1009) | 578 | 394 | 85.5 | 60.7 | 482 | 443 | 137 |
| `dct` 형 1, 2¹⁶ / 2²⁰ | 1,283 / 26,180 | 1,273 / 30,387 | 1,376 / 27,223 | 258 / 5,291 | — | 1,265 / 15,372 | 1,133 / 4,476 |
| `dct` 형 2, 2¹⁶ / 2²⁰ | 173 / 3,434 | 179 / 3,574 | 177 / 3,625 | 69.1 / 773 | — | 258 / 6,236 | 147 / 2,411 |
| `dst` 형 1, 2¹⁶ | 980 | 1,442 | 1,015 | 274 | — | 3,323 | 2,000 |
| `dct` 형 1, (1000, 1024) | 9,195 | 6,125 | 1,191 | 751 | — | 5,150 | 699 |
| `fht` 2²⁰ | 6,379 | 6,639 | 6,736 | 1,128 | — | 44,744 | 46,671 |
| `fht` 새 계획 2¹⁶ | 7,375 | 7,399 | 1,212 | 962 | — | 2,476 | 2,536 |
| 작은 호출: `fft` 16 / 1024, `rfft` 16, `fftn` (8, 8) | 0.19 / 1.51, 0.22, 0.73 | 0.20 / 1.54, 0.22, 0.63 | — | — | 1.59 / 5.15, 1.58, 4.77 | 1.70 / 4.26, 1.86, 3.08 | — |

- 긴 한 줄(4단계, FT65)은 10스레드에서 전의 5–7배이고 scipy `workers=10`의 1.6–3.8배 빠르다(2²⁰ 882 대 1,619 µs, 2²² 4.0 대 6.5 ms, 실수 2²⁰ 409 대 1,521 µs, 소수 100003 349 대 2,708 µs, DCT-II 2²⁰ 773 대 2,411 µs). 1스레드는 2¹⁶–2¹⁸에서 0.90–0.98배, 2²⁰ 0.84배, 2²² 0.77배, Bluestein의 5-매끄러운 안쪽 길이(100003, DST-I 2¹⁶)와 36960에서 0.63–0.68배로 느려졌지만 모두 scipy `workers=1`보다는 빠르다(DCT-I 2²⁰만 빼고: 30.4 대 15.4 ms, 안쪽 2²⁰ − 1에 41이 있어 Bluestein).
- 기수 13–31(FT64)은 그런 길이에서 1.3–4.8배다(`fft` 4095 53.9 → 11.3 µs, scipy 24.0). `dct` 형 1 (1000, 1024)은 안쪽 1023 = 3·11·31이라 1.5배이고 scipy의 0.56배가 0.84배가 됐다.
- 마지막 축이 아닌 축(FT67)은 1스레드에서 1.05–1.47배다(`rfftn` (256, 256, 256) 55.7 → 46.5 ms, scipy 43.8; `fft2` (34, 1009) 578 → 394 µs, scipy 443). `rfft2` (2048, 2048)의 1스레드(11.0 → 11.8 ms)는 잡음 폭 안에서 오르내렸다(타일만 바뀐 경로이고 따로 잰 그 패스는 9.2 → 8.7 ms).
- 짧은 한 줄과 4단계 아래의 배치는 전과 3% 안이다(기수 13–31을 `switch`로 고른 단 함수; FT68).

**4단계와 기수 13–31이 닿지 않는 크기** (n < 2¹⁵이고 기수가 2–11인 한 줄, 안쪽 길이가 2¹⁵ 아래인 Bluestein 줄, 마지막 축 줄의 FFTLog. 앞의 빌드에서 잰 값이고 이 경로들은 지금의 빌드와 3% 안이다, FT68)

| 일 | sd | numpy | scipy | scipy `workers=10` |
|---|---:|---:|---:|---:|
| `fft` n = 16 / 64 | 0.20 / 0.26 | 1.46 / 1.63 | 1.60 / 1.64 | — |
| `fft` n = 1024 / 4096 | 1.53 / 8.29 | 4.84 / 17.7 | 3.99 / 14.2 | — |
| `fft` n = 3072 / 15360 / 20480 | 5.48 / 40.0 / 56.4 | 14.6 / 73.0 / 95.1 | 11.6 / 91.2 / 126 | — |
| `fft` n = 1000 | 1.75 | 5.16 | 4.50 | — |
| `fft` n = 1009 / 10007 (Bluestein) | 9.96 / 137 | 28.6 / 294 | 17.5 / 251 | — / 180 |
| `ifft` n = 1024 | 1.74 | 5.56 | 4.24 | — |
| `rfft` n = 16 / 64 / 1024 / 4096 | 0.21 / 0.26 / 1.11 / 4.60 | 1.60 / 1.77 / 3.72 / 10.9 | 1.87 / 1.96 / 3.50 / 9.65 | — |
| `rfft` n = 1000 / 1155(홀수) | 1.14 / 3.31 | 3.83 / 5.82 | 3.54 / 4.70 | — |
| `rfft` n = 10007(홀수 Bluestein) / 20014(반이 Bluestein) | 146 / 145 | 304 / 823 | 270 / 271 | — |
| `irfft` n = 1024 / 1155 | 1.36 / 3.29 | 4.09 / 6.21 | 4.17 / 5.49 | — |
| 실수 입력 `fft` n = 1024 (같은 값의 complex128 입력) | 1.27 (1.60) | 5.66 (5.22) | 3.81 (4.23) | — |
| `fftfreq(1024)` / `fftshift` (1000, 1024) | 0.61 / 85 | 1.68 / 518 | — | — |

| FFTLog | sd 1스레드 캐시 / 새 계획 | sd 10스레드 캐시 / 새 계획 | scipy |
|---|---:|---:|---:|
| `fht` 2¹² | 11.1 / 479 | 11.1 / 74.7 | 150 |
| `ifht` 2¹², `bias=0.5` | 12.5 / 883 | 12.4 / 173 | 176 |
| `fht` (1000, 1024) | 2,140 / 2,350 | 283 / 304 | 3,040 |
| `fhtoffset` | 0.96 | — | 1.04 |

- 이 크기의 한 줄은 numpy의 1.7–7.6배, scipy의 1.4–9배 빠르다. 가장 작은 쪽은 홀수 실수 길이(1155: 길이 n의 복소 변환과 에르미트 부분, FT43)와 Bluestein 길이다. 실수 입력의 `fft`는 실수 길(FT16) 덕에 같은 값의 complex128 입력보다 1.26배 빠르다.
- 실수 한 줄의 커널은 n = 1024에서 안쪽 복소 FFT(길이 512) 약 0.77 µs와 사후 처리 약 0.25 µs다(C++에서 잰 값).
- FFTLog의 입력은 logspace(−4, 4, n)의 r^1.5·e^(−r²/2), mu = 0.5, scipy의 최적 offset이다. 계획이 있으면 한 줄은 실수 FFT 두 번과 곱 한 번이라 scipy의 6.7–14배 빠르다(scipy는 부를 때마다 `loggamma` 둘과 `exp`로 계수를 만든다; 2²⁰은 위 표). 새 계획은 1스레드에서 scipy의 2.8–5배 느리고(double-double의 log Γ; FT58, FT68) 10스레드에서 2–2.6배 빠르다. 배치는 계획 하나를 모든 줄이 쓴다: (1000, 1024)는 1스레드 1.4배, 10스레드 11배 빠르다.

**vDSP·시제품과 견준 탐침** (`tools/fft_probe.py`의 `time`, `paths`; vDSP를 쓰지 않기로 한 근거의 하나다. 자체 시제품과 vDSP 열은 C++에서 잰 한 줄의 변환이고 complex128(실수는 float64)을 받아 complex128을 내기까지다(vDSP는 split으로 나누고 다시 엮는 것과, n ≤ 4096이면 `Interleaved`와 빠른 쪽; 세 번의 중앙값). R 열은 설계 초안의 한 R 세션의 `fft`다)

| 일 | numpy | scipy | scipy `workers=10` | R | 자체 시제품 | vDSP |
|---|---:|---:|---:|---:|---:|---:|
| `fft` n = 1024 | 4.9 | 4.0 | — | 3.6 | 1.22 | 1.58 |
| `fft` n = 4096 | 18.0 | 14.1 | — | — | 7.64 | 8.17 |
| `fft` n = 3072 | 14.7 | 11.8 | — | — | 4.55 | 9.98 |
| `fft` n = 15360 | 73.2 | 91.4 | — | — | 38.4 | 68.4 |
| `fft` n = 20480 | 90.4 | 121.8 | — | — | 54.5 | 123.4 |
| `fft` n = 2¹⁶ | 589 | 414 | 154 | 1,767 | 225 | 316 |
| `fft` n = 2²⁰ | 10,406 | 7,053 | 1,179 | 64,000 | 5,012 | 7,437 |
| `fft` n = 1000 | 5.1 | 4.5 | — | 5.0 | — | — |
| `fft` n = 1009 (소수) | 27.8 | 17.3 | — | 250 | — | — |
| `fft` n = 10007 (소수) | 289 | 249 | 182 | — | — | — |
| `fft` n = 100003 (소수) | 4,331 | 4,701 | 2,417 | — | — | — |
| `rfft` n = 1024 / 2¹⁶ / 2²⁰ | 3.5 / 198 / 3,989 | 3.4 / 236 / 5,207 | — | — | 0.83 / 129 / 2,717 | 1.02 / 161 / 3,691 |
| `dct` 형 2, n = 2¹⁶ / 2²⁰ | — | 260 / 6,257 | — | — | — | — |
| `fft` (1000, 1024), 마지막 축 | 2,082 | 2,004 | 330 | — | — | — |
| `fft2` (2048, 2048) | 38,271 | 22,756 | 3,217 | — | — | — |

- 자체 시제품은 한 줄에서 scipy의 1스레드보다 n = 1024에서 3.3배(호출 비용을 뺀 커널끼리는 아니다), 2²⁰에서 1.4배, `rfft` 2²⁰에서 1.9배 빨랐다.
- vDSP는 2의 거듭제곱이 아닌 길이에서 numpy보다도 느린 곳이 있다(n = 20480에서 123 대 90 µs). 그래서 FT30이다.
- scipy(DUCC0)는 `workers`로 1차원 변환 하나도 나눈다(2²⁰에서 10 workers가 4.7배). sd는 같은 일을 4단계(FT65)로 기본에서 한다.
- R의 `fft`는 큰 소인수에서 O(n²)라 소수 길이에서 느리다(1009에서 250 µs).

- **목표** (§9를 정할 때 둔 것)
  - 작은 1차원(n ≤ 4096, 한 줄): numpy의 2배 이상. 호출 비용(입력 검사, 결과 할당, 계획 찾기)은 1 µs 아래를 목표로 하고 계획 찾기는 잠금 하나와 해시 찾기다.
  - 큰 1차원(2¹⁶–2²⁰): 1스레드에서 scipy `workers=1` 이상, 10스레드(4단계)에서 scipy `workers=10` 이상.
  - 자체 혼합 기수(1000, 7·2ᵏ, 11-매끄러운 수): 1스레드에서 scipy의 1.2배 안, numpy 이상.
  - 소수 길이(Bluestein): scipy `workers=1` 이상.
  - 배치와 다차원((1000, 1024), (2048, 2048), `rfftn` (256, 256, 256)): 10스레드에서 scipy `workers=10` 이상.
  - DCT·DST(2¹⁶, 2²⁰): 1스레드에서 scipy의 1.2배 이상(자체 실수 FFT 위의 Makhoul).
- **확인** (위 표들): 작은 1차원(n ≤ 4096)은 복소와 짝수 실수 길이에서 numpy의 2.1–8배이고 호출은 0.2 µs다. **홀수 실수 길이는 2배에 못 미친다**(`rfft` 1155·1001은 1.8배, 4095는 1.3배: 길이 n의 복소 변환과 에르미트 부분이라 짝수 길이의 두 배 일이다; 실수 전용 나비는 넣지 않았다, FT68). 큰 1차원은 1스레드에서 scipy `workers=1`의 1.1–1.7배, 10스레드에서 scipy `workers=10`의 1.6–3.8배다. 자체 혼합 기수(1000, 7·2¹⁰)는 scipy의 1.7–2.7배다. 소수 길이(Bluestein)는 scipy `workers=1`의 1.8–2.3배다. 배치와 다차원은 10스레드에서 scipy `workers=10`과 대체로 같거나 빠르다(`fft` (1000, 1024) 마지막 축 331 대 320 µs와 `fft2` (2048, 2048) 3.25 대 3.23 ms는 3% 안, `rfftn` (256, 256, 256)·`irfftn`·`ifft2`·축 0 배치·`fftn`은 1.07–3.5배 빠르다); **`rfft2` (2048, 2048)는 1.69 대 1.55 ms로 못 미친다**(두 번째 패스의 간격 16400바이트가 두 줄 커널의 캐시 조건 밖이다). DCT·DST는 형 2–4가 2¹⁶·2²⁰의 1스레드에서 scipy의 1.4–1.7배이고, **DCT-I은 못 미친다**(2¹⁶ 0.99배, 2²⁰ 0.51배: 안쪽 길이 2ᵏ − 1의 소인수 257·41이 Bluestein이다; 10스레드는 2¹⁶ 4.4배, 2²⁰ 0.85배).
- **나중에 볼 것**: 다차원에서 축 패스를 합친 블록 전치, 작은 n(≤ 64)의 코드 생성 나비, 큰 소인수(37 이상)의 Rader 단(DCT-I 2ᵏ의 안쪽 길이), 4단계의 1스레드 손실(열·행의 간격 있는 접근), 실수·DCT 패스의 두 줄 커널.

---

## 6. 층과 파일
- **층**: `fft/`는 `array/`, `base/`(`sincos.hpp`), `parallel/` 위에 있다(인자 검사와 바인딩은 `bind/`의 `pyutil`·`errors`). Accelerate의 vDSP는 부르지 않는다(FT30, FT66). vDSP의 래퍼 `fft/vdsp.hpp`는 라이브러리에 빌드되지 않고 탐침(`tools/fft_probe_kernels.cpp`)만 쓴다. 다른 영역은 `fft/`를 모른다(나중의 signal·smooth가 계획과 구동기를 쓰게 되면 그때 위에 붙는다). Python 층은 `stardust.fft` 하나이고 다른 영역을 쓰지 않는다(FFTLog도 쓰지 않는다: FT58).
- 등록: `bench/areas.py`의 `fft`는 `interpolate` 뒤, `plot` 앞이다(`plot`은 맨 위 층이다). ARCHITECTURE §1의 그림과 §2의 목록, API.md의 색인 표, README.md의 표, tests/README.md·bench/README.md의 영역 목록에 한 줄씩 있다. `sd.fft`는 `sd`의 다른 이름과 겹치지 않는다(배열 영역에 `fft`라는 이름이 없다).
- `csrc/fft/`
  - `cfft.hpp`: 혼합 기수 나비(기수 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31; 홀수 기수의 상수 `OddConst`)와 Stockham 구동, 4단계(`kFourStepMin` = 2¹⁵, 열·행 16개 묶음), 두 줄 커널(`detail::Lines`), `CfftPlan`, `run_tasks`·`run_chunks`(2¹⁴칸). 헤더뿐이라 `tools/fft_probe_kernels.cpp`도 그대로 쓴다
  - `twiddle.hpp`: 회전 인자 e^(−2πik/n)
  - `ddmath.hpp`: double-double 산술, e^(−2πit/N)의 double-double 값과 그 두 표(`CisTable`), 재귀 DFT의 선언
  - `bluestein.hpp`: `BluesteinPlan`, `smooth5_at_least`
  - `plan.hpp`, `plan.cpp`: 계획(`Plan`: 종류 `Complex`·`Real`·`Dct2`·`Dct4`, 길이, 길, `parallel_inside`), 길 고르기(FT36, FT64), Bluestein의 계획과 실행(chirp 곱은 `run_chunks`), double-double DFT, 캐시(LRU, 뮤텍스, `kCachePlans` = 32·`kCacheBytes` = 256 MB), 풀의 `run_tasks`
  - `driver.hpp`, `driver.cpp`: 축 구동기 `c2c_axis`(바로 읽기, 모으기, 묶음, 흩기, 정규화, 제자리 변환과 켤레, 두 줄 경로와 캐시 조건, 타일 복사), 실수 변환의 `r2c_axis`·`c2r_axis`(짝수는 실수 계획, 홀수는 복소 계획과 에르미트 부분·확장, 거울 채우기, 켤레), DCT·DST의 `r2r_axis`, FFTLog의 `fht_axis`, 공용 줄 구동 `drive_lines`, 스레드의 scratch(`kScratchKeep` = 2¹⁶ double, 슬롯 넷), 묶음·태스크 크기(`kBlockLines` = 128, `kBlockShare` = 64, `kTaskWork` = 2¹², `kMaxTasks` = 1024, `kParallelWork` = 2¹⁵)
  - `args.hpp`: 변환들이 함께 쓰는 인자 검사(`norm`, `n`, `axis`, `s`, `axes`)와 정규화 인자·결과 모양
  - `basic.cpp`: `_c2c`(`fft`·`ifft`; 실수 입력은 실수 길), `_r2c`(`rfft`·`ihfft`), `_c2r`(`irfft`·`hfft`), `_c2cn`(`fft2 ifft2 fftn ifftn`), `_r2cn`(`rfft2 rfftn ihfft2 ihfftn`), `_c2rn`(`irfft2 irfftn hfft2 hfftn`; 패스의 순서와 제자리, 정규화)의 인자 검사와 실행, 시험용 `_plan_info`(`real=True`면 실수 계획; 4단계면 `n1`)·`_cache_state`·`_cache_clear`·`_twiddles`
  - `helper.cpp`: `_freq`(`fftfreq`·`rfftfreq`), `_fast_len`(`next_fast_len`·`prev_fast_len`)
  - `rfft.hpp`: 짝수 n의 실수 길의 사후 처리 `r2c_post`와 역변환의 사전 처리 `c2r_pre`(에르미트 입력의 버리는 허수부를 읽지 않음, 켤레; 긴 줄은 `run_chunks`), 회전 인자 표 `real_twiddles`, 에르미트 부분 `hermitian_part`, 탐침용 시제품 `RfftPlan`
  - `r2r.hpp`: DCT·DST 형 1–4의 줄 커널(`r2r_line`), `Dct2`·`Dct4` 계획의 표(회전 인자, 홀수 DCT-IV의 순열과 부호)
  - `r2r.cpp`: `_r2r`(`dct idct dst idst`), `_r2rn`(`dctn idctn dstn idstn`; 다차원 패스, 복소 입력의 부분 보기)의 인자 검사와 실행
  - `fftlog.hpp`: double-double의 지수·사인·코사인(표와 급수)·로그·log Γ(반사식, 옮기기, Stirling), Tang의 반올림 지수, FFTLog의 계획 `FhtPlan`과 키, 전용 캐시의 상한(`kFhtCachePlans` = 16, `kFhtCacheBytes` = 128 MB)(§3.9, FT58–FT63)
  - `fftlog.cpp`: 계획 만들기(계수와 바이어스, 계수는 16개씩 풀에서), 전용 캐시, `fhtoffset`, `_fht`·`_fhtoffset`과 시험용 `_fht_plan`·`_fht_cache_state`·`_fht_cache_clear`
  - `fft.hpp`: 파일마다의 `bind_*`
  - `vdsp.hpp`: vDSP 설정의 RAII 묶음(`VdspPair`, `VdspInterleaved`)과 설정 메모리의 어림 `vdsp_setup_bytes`. 라이브러리는 쓰지 않는다(FT30, FT66)
- 바인딩 `bind/fft_bind.cpp`·`fft_bind.hpp`: `_core.fft` 하위 모듈(`bind_basic`, `bind_helper`, `bind_r2r`, `bind_fftlog`).
- `src/stardust/fft/` (하위 패키지 없음): `__init__.py`(공개 이름 35개), `_common.py`(도우미와 FFTLog의 입력, `axes`, int·실수 인자의 검사와 메시지; 변환의 검사는 `csrc/fft/args.hpp`·`basic.cpp`·`r2r.cpp`), `_basic.py`(복소·실수·에르미트 변환), `_realtransforms.py`(DCT·DST), `_helper.py`(`fftfreq rfftfreq fftshift ifftshift next_fast_len prev_fast_len`), `_fftlog.py`(`fht ifht fhtoffset`의 인자 검사).
- 도구: `tools/fft_ref.py`(참값, §7.2), `tools/fft_probe.py`와 `tools/fft_probe_kernels.cpp`(탐침, §7.4).

---

## 7. 참값과 검증
다른 영역과 같은 체계다(ARCHITECTURE §6, tests/README.md, bench/README.md). 모든 도구는 `bench/cli.py`의 공통 옵션(영역, `--match`, `--jobs`)을 쓴다. 긴 참값 생성은 프로세스 10개까지, 메모리를 지켜보며(RSS가 평평한지) 돈다.

### 7.1 시험 신호 (`tests/fft/signals.py`)
- **길이**(`LENGTHS`): 1–128의 모든 길이, vDSP 집합과 2ᵏ의 경계(23, 24, 25, 39, 40, 41, 47, 48, 49, 119, 120, 121, 239, 240, 241, 2ᵏ ± 1), 소수(127, 1009, 10007, 65537), 큰 소인수가 있는 합성수(2·127, 4·257, 2·1009, 3·10007), 7·2ᵏ와 11-매끄러운 수, 1–2²⁰의 2의 거듭제곱(4단계 문턱 2¹⁵의 양쪽)과 f·2ᵏ(f = 3, 5, 15).
- **값**: 균등·정규 난수(실수, 복소), 임펄스(한 칸만 1), 상수(결과의 대부분이 참 0), 한 주파수의 사인파(정수 k와 k + 1/2), 에르미트 대칭 입력과 대칭이 아닌 입력(`irfftn`이 버리는 부분), 크기가 1e−300–1e300에 걸친 입력(노름 계약이 작은 칸을 보지 않음을 보인다), 넘침 직전(‖x‖₁ ≈ 2¹⁰²⁰)과 비정상 수, NaN·inf의 모든 자리.
- **배치와 레이아웃**: 변환 축이 처음·가운데·끝, C·F 순서, 음수와 띄엄띄엄한 stride, 길이 0인 배치 축.
- 신호의 id는 `<영역>/<값>/<길이>`다(영역 `c` 복소, `r` 실수, `h` 길이 n의 출력을 낼 반쪽 스펙트럼, `nd` 다차원 `AxBxC`). 값은 `uniform normal impulse constant sine sine_half wide huge tiny`이고, `huge`는 ‖x‖₁이 2¹⁰²⁰ 둘레, `tiny`는 2⁻¹⁰⁶⁰배(비정상 수)다. 유한하지 않은 값(`nonfinite(n)`)과 레이아웃(`layouts()`)은 따로 만든다. 모두 정한 씨앗의 PCG64에서 나온다.

### 7.2 참값 (`tools/fft_ref.py` → `tests/fft/ref/`, `tests/reffile.py`의 npz)
- Arb의 `acb.dft`로 참 변환을 구하고 각 칸을 double-double(hi, lo)로 저장한다. 노름 상대오차가 u 크기라서 RN(y)만 저장하면 참값의 반올림(칸마다 u·\|yₖ\| 이하)이 재려는 오차와 같은 크기가 되기 때문이다.
- DCT·DST의 참값은 짝·홀 확장의 `acb.dft`(구간 산술이라 엄밀하다)에서 구한다. 실수·에르미트 변환은 확장한 복소 입력의 `acb.dft`다.
- 파일은 작게 둔다(FT33): n ≤ 2¹⁴의 줄만 둔다. 묶음은 `c2c`(313줄: 균등 난수는 1–128의 모든 길이, 경계 길이, 2ᵏ, f·2ᵏ, 소수와 큰 소인수, 7·2ᵏ와 11-매끄러운 수; 다른 값 여덟 가지는 짧은 목록; `backward`는 n ≤ 4096), `real`(282줄), `hermitian`(113줄), `r2r`(140줄, 여덟 변환), `nd`(10줄)이고 합해 약 10 MB다. 2¹⁶, 65537, 3·10007, 2²⁰의 줄은 정확도 체계가 그 자리에서 Arb로 잰다(메모리를 지켜보며).
- 각 칸은 출력의 2-노름의 2⁻¹²⁰ 안으로 정해진다(그 뒤 double-double의 반올림이 칸의 2⁻¹⁰⁶을 더한다). 정밀도는 192비트에서 시작해 두 배씩 올린다. 858줄이 모두 192비트에서 정해졌고 3프로세스로 몇 초 걸린다.
- FFTLog는 참값 파일이 없다: `refdata.arb_fht`가 정확한 매개변수에서 `acb.gamma`·`acb.rgamma`(계수)와 `acb.dft`로 그 자리에서 구한다(길이 4096에서 0.1초). 계수만은 `refdata.fht_coefficients`, `fhtoffset`은 `refdata.arb_fhtoffset`이다.
- 참값이 스스로를 검사한다(`tests/fft/test_reference_data.py`): 파스발 항등식, 정·역 방향이 지수를 뒤집은 관계(따로 계산했다), DC와 나이퀴스트 칸의 정확한 유리수 합, 상수·임펄스 신호의 정확한 꼴, numpy·scipy가 모든 참값에서 계약의 0.25 안.

### 7.3 테스트 (`tests/fft/`)
- `test_api_surface.py`: API 문서의 시그니처 블록과 `sd.fft.__all__`, scipy.fft의 이름, 데이터는 위치·옵션은 키워드.
- `test_basic.py`: 복소 변환. numpy·scipy와 그 자리에서 계약 안, R의 기록, 모양·dtype·자르기·0 채우기·`norm`, 길이 1, 빈 배치 축, 2의 거듭제곱 배율, 역변환이 실수부·허수부를 바꾼 정변환과 같은 비트, 유한하지 않은 값.
- `test_real.py`: 실수·에르미트 1차원. 버리는 허수부, 기본 길이, 홀수 n, `rfft` = 실수 입력 `fft`의 앞쪽 비트와 켤레 관계.
- `test_nd.py`: 다차원. `s`·`axes`의 모든 꼴(기수 13의 축과 31보다 큰 소인수의 축 37, 41, 43, 74 포함)을 numpy·scipy와 FT49의 계약으로, `irfftn`이 버리는 부분, 비트 관계, 레이아웃, 빈 배치, 에러, numpy·scipy의 기록.
- `test_realtransforms.py`: DCT·DST. 저장한 r2r 참값의 모든 줄 × `norm` × `orthogonalize`, Arb를 그 자리에서 구한 긴 줄과 비정상 수·넘침 직전의 값, scipy와의 계약, 경계, DST와 DCT의 비트 관계, 레이아웃, 에러.
- `test_fftlog.py`: FFTLog. `refdata.fht_points`의 모든 점이 Arb와 계약 안, 계수가 Arb와 u 안, `fhtoffset`이 반 ulp 안, scipy의 테스트 값(Fortran FFTLog의 출력, 항등식, 특이한 경우, 거듭제곱의 정확한 변환), scipy와의 차이, 레이아웃과 배치, 계획 캐시, 에러.
- `test_helper.py`: `fftfreq`가 numpy와 같은 비트, `fftshift` = `roll`, `next_fast_len`·`prev_fast_len`이 scipy와 같은 값(R의 `nextn`).
- `test_errors.py`: 1차원 변환과 도우미의 틀린 인자와 메시지, 호출 모양, numpy·scipy가 조용히 넘기거나 내부 메시지를 내는 경우.
- `test_reference.py`: 저장한 c2c·실수·에르미트·다차원 참값의 모든 줄과 긴 줄(Arb를 그 자리에서), `refdata.nd_points`(Bluestein 축 포함)가 계약 안.
- `test_reference_data.py`: 참값과 기록이 시험 신호와 맞음(digest), 참값의 자체 검사(§7.2), numpy·scipy가 모든 참값에서 계약 안.
- `test_paths.py`: 한 줄의 비트가 배치·레이아웃(C, F, 음수와 띄엄띄엄한 stride, 축의 위치)·자르기와 무관함, 혼합 기수와 Bluestein 경계의 양쪽, 바로 읽기와 모으기, 실수 길의 홀짝.
- `test_threads.py`: 1, 3, 10스레드에서 같은 바이트(1차원, 다차원 패스, DCT·DST, FFTLog의 스레드 수마다 새로 만든 계획, 긴 줄의 4단계); `VECLIB_MAXIMUM_THREADS=1`과 기본 모드의 자식 프로세스가 같은 결과의 SHA-256.
- `test_plan_cache.py`: 상한과 쫓아내기, 쫓겨난 뒤 다시 만든 계획의 같은 비트, Python 스레드 여럿이 같은 길이와 다른 길이를 동시에 부를 때.
- `test_proof.py`(`proof.py`): 나비의 상수가 가장 가까운 double, 단마다의 한계(기수 2–11은 κ = 8, 13–31은 κ_w = 17 안), 계획들의 회전 인자가 Arb와 μ = 1.5u 안, Bluestein의 ρ, 길이 `n`의 길(31-매끄러운 길이만 혼합 기수)과 4단계의 분해.
- 도우미: `signals.py`(§7.1), `refdata.py`(계약의 판정 `bound`·`bound_nd`·`bound_fht`, Arb를 그 자리에서 구하는 함수), `proof.py`, `conftest.py`. hypothesis는 쓰지 않는다(길이와 모양은 정한 목록이다).
- numpy·scipy와는 그 자리에서 견준다: 같은 입력에서 둘 다 계약 안이고 반올림이 아닌 동작(모양, dtype, 에러의 종류)이 같다. `[numpy와 다름]`·`[scipy와 다름]`은 하나하나 테스트가 확인한다.

### 7.4 numpy·scipy·R·vDSP의 기록 (`tools/fft_probe.py` → `tests/fft/data/`)
- 설계 초안의 탐침(경계 동작, NaN·inf 무늬, vDSP의 길이표·결정성·설정 비용·정확도·시간)을 도구 하나로 옮겼다: 구역 `paths`(FT22·FT66의 측정), `twiddle`, `vdsp`, `accuracy`, `numpy`, `scipy`, `r`, `time`. C++ 쪽은 `tools/fft_probe_kernels.cpp`이고 sd의 플래그로 `build/fft_probe/`에 공용 빌드 잠금 아래 빌드한다. 프로세스를 죽일 수 있는 호출(scipy의 DUCC0 단언, vDSP DCT의 n ≡ 2 (mod 4))은 자식 프로세스에서 부르고 죽으면 다음 길이부터 다시 띄운다. C++ 자식은 하나씩, 시간 제한과 함께 돌고 상주 메모리를 0.1초마다 읽어 3 GB를 넘으면 죽인다. 512 MB를 넘을 vDSP 설정은 만들지 않는다.
- 기록: `numpy.json`, `scipy.json`(경계와 에러, `[numpy와 다름]`·`[scipy와 다름]`의 근거), `r.json`(R 4.6.2 `fft`·`mvfft`·`nextn`), `vdsp.json`(길이표, 결정성, 설정의 시간과 메모리, CPU/경과 시간, split과 `Interleaved`의 비용), `prototype.json`(FT22의 측정, 회전 인자의 오차, 네 길의 계약 단위). 신호에서 나온 기록은 입력의 `digest`를 가진다.

### 7.5 정확도·속도 체계와 골든
- **정확도** (`bench/accuracy/fft.py`, 표 `bench/fft/accuracy.md`): 스윕 `c2c`(영역은 방향 × 길의 꼴: pow2, f2k, mixed, bluestein), `real`(`rfft`·실수 입력 `fft` × pow2, even, odd, bluestein), `hermitian`(`irfft`), `nd`(`fftn ifftn rfftn irfftn` × smooth, bluestein), `r2r`(형 여덟, `norm='backward'`, × even, odd, bluestein), `fftlog`(방향 × 바이어스의 유무). 점은 한 줄이고 판정은 §4의 계약 단위다. 비교 패키지는 numpy, scipy, R(복소만; FFTLog와 DCT·DST는 scipy만). 기준선은 최대와 제곱평균의 계약 단위를 둔다(§4.4).
- **속도** (`bench/speed/cases/fft.py`): 사례 66개. §5의 크기 전부와 `ifft`·`irfft`·`ihfft`·`hfft`, 축 0의 배치, 다차원, DCT·DST 형 1–4와 `dctn`, FFTLog(캐시와 새 계획), 기수 13–31과 4단계의 길이, `fftfreq`·`fftshift`(호출 비용). scipy는 `workers=1`과 `workers=10` 둘 다 비교 대상이다(sd는 기본이 10스레드라서).
- **골든** (`tests/golden/cases_fft.py`, 기록 `tests/golden/data/fft.json`의 2,605케이스): 모든 함수 × 길이의 꼴(2ᵏ, f·2ᵏ, 그 밖의 혼합 기수, 기수 13–31, Bluestein, 4단계) × `norm`, 다차원의 `s`·`axes`, DCT·DST 형 1–4, FFTLog, 도우미, 틀린 호출. 1·3·10스레드의 결과가 모두 같아야 한다. 큰 케이스는 병렬 길과 4단계를 탈 크기로 두되 결과가 2¹⁷개 복소수를 넘지 않는다.

---

## 8. 마일스톤
| 마일스톤 | 내용 | 끝났다는 뜻 |
|---|---|---|
| FT0 준비 — **완료·승인(위임, 2026-11-23)** | API 문서 초안(`docs/api/fft.md`: 모든 공개 이름의 시그니처, numpy·scipy·R → sd 대응표, §9의 답), 영역 등록(`bench/areas.py`, ARCHITECTURE, API.md의 색인), 시험 신호(§7.1), 참값 도구(§7.2), 탐침 도구와 기록(§7.4), vDSP의 split·`Interleaved` 비용과 설정 비용 다시 재기, 자체 나비의 오차 한계 증명의 틀(§4.2의 κ, c), 속도 사례의 틀 | 문서와 도구가 커밋되고 탐침 기록이 있다. §9의 답이 §1.1의 결정으로 옮겨졌다. 결과: FT22의 측정(§3.1, FT29·FT30), 계약의 상수(§4.2, FT31), 참값(FT33), 시제품(FT34) |
| FT1 1차원 복소와 도우미 — **완료·승인(위임, 2026-11-23)** | 계획과 캐시(§3.6), 자체 혼합 기수·회전 인자·Bluestein(§3.3), 축 구동기와 나누기(§3.7), `fft ifft`, `fftfreq rfftfreq fftshift ifftshift next_fast_len prev_fast_len` | 정확도 스윕이 계약 안, 한 줄의 비트가 배치·레이아웃·스레드와 무관함, Accelerate 모드와 무관함, 골든, numpy·scipy·R과의 속도 사례. 결과: 계약 단위 최대 0.054(§4.4), 1·3·10스레드와 레이아웃·Accelerate 모드에서 같은 바이트, 나비마다의 증명(§4.2), 결정 FT35–FT42 |
| FT2 실수와 에르미트 1차원 — **완료·승인(위임, 2026-11-23)** | `rfft irfft hfft ihfft`, 자체 실수 길(§3.4), 실수 입력의 `fft`(§9의 12번) | 계약 안, 버리는 허수부와 기본 길이가 numpy·scipy와 같음, `rfft` = `fft`의 앞쪽 비트, 속도 사례. 결과: 계약 단위 최대 0.049(§4.4), `rfft` = 실수 입력 `fft`의 앞쪽과 켤레 관계가 비트까지 같음, 1·3·10스레드와 레이아웃에서 같은 바이트, 결정 FT43–FT46 |
| FT3 다차원 — **완료·승인(위임, 2026-11-23)** | `fft2 ifft2 fftn ifftn rfft2 irfft2 rfftn irfftn hfft2 ihfft2 hfftn ihfftn`, `s`·`axes`의 규칙(§2.4) | `irfftn`의 버리는 부분이 numpy·scipy와 같음, 계약 안, 축 패스의 병렬과 결정성, 속도 사례. 결과: 계약 단위 최대 0.045(§4.4), 에르미트가 아닌 입력의 `irfftn`이 numpy·scipy의 순서로 구한 Arb 참값의 계약 안, 1·3·10스레드와 레이아웃에서 같은 바이트, 10스레드의 큰 배열은 scipy `workers=10`의 ±15% 안이고 중간·작은 배열은 더 빠름(§5), 결정 FT47–FT52 |
| FT4 DCT·DST — **완료·승인(위임, 2026-11-23)** | `dct idct dst idst dctn idctn dstn idstn`(형 1–4, `orthogonalize`; §3.5) | 계약 안, DST와 DCT의 비트 관계, scipy와 경계가 같음(다른 곳은 `[scipy와 다름]`), 속도 사례. 결과: 계약 단위 최대 0.061(§4.4), DST-II·III·IV가 부호·뒤집기의 DCT와 비트까지 같음, 1·3·10스레드와 레이아웃에서 같은 바이트, 2¹⁶·2²⁰의 형 2–4는 1스레드에서 scipy의 1.4–1.9배이고 DCT-I(2¹⁶)은 0.96배(안쪽 길이 2¹⁶ − 1이 Bluestein, §3.5), 결정 FT53–FT57 |
| FT5 FFTLog — **완료·승인(위임, 2026-11-23)** | `fht ifht fhtoffset`(§3.9) | scipy와 계약 안, 속도 사례. 결과: 계약 단위 최대 0.069(§4.4; scipy는 같은 계약의 4,615배까지), 계수가 Arb와 u 안(double-double의 log Γ, FT58), 1·3·10스레드와 레이아웃에서 같은 바이트, 계획이 있으면 scipy의 6.7–14배 빠르고 새 계획은 1스레드에서 2.8–5배 느림(§5), 결정 FT58–FT63 |
| FT6 성능과 일관성 점검 — **완료·승인(위임, 2026-11-23)** | 모든 공개 함수를 크기마다 numpy·scipy·R과 견주어 고치기, 길의 표(vDSP 대 자체를 FT29의 방법으로 다시 잼, FT22), Bluestein 문턱, 4단계 문턱 n₄, 줄 묶음의 크기, 계획 캐시의 상한, 메시지·시그니처의 영역 간 일관성, 코드 리뷰 | 속도 표와 정확도 표가 기준선이 되고 §5의 목표를 적은 대로 확인한다. 결과: 길의 표는 여덟 꼴 모두 자체 그대로(FT66), 기수 13–31과 p₀ = 31(FT64; 그런 길이의 계약 κ = 17은 전의 Bluestein 꼴보다 좁고 잰 최대 0.027), 4단계 n₄ = 2¹⁵(FT65; 긴 한 줄이 10스레드에서 5–7배, scipy `workers=10`의 1.6–3.8배), 마지막 축이 아닌 축의 두 줄 커널과 타일 복사(FT67; 같은 비트), 재 보고 넣지 않은 것(FT68), §5의 목표는 DCT-I만 못 미침, 골든 케이스 371개가 바뀜(기수 13–31과 n ≥ 2¹⁵의 길이; 다시 기록은 메인 세션), 결정 FT64–FT68 |
| FT7 문서 — **완료·승인(위임, 2026-11-23)** | `docs/api/fft.md`의 확정, 이 문서를 구현대로 고치기(잰 값, 정한 상수, 길의 표, 파일, 계약) | 문서가 코드와 맞고 `test_api_surface`가 통과한다 |

표의 결과는 그 마일스톤을 마칠 때의 측정이다(지금의 값은 §4.4와 §5다).

진행 원칙은 다른 영역과 같았다: 마일스톤마다 시작 전에 §9의 질문을 정하고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달했다. FT1 마일스톤이 계획과 구동기를 만들므로 맨 앞이었다. FT2 마일스톤은 FT1 마일스톤 뒤, FT3·FT4 마일스톤은 FT2 마일스톤 뒤이고 서로 순서를 바꿀 수 있었다. FT5 마일스톤은 FT2 마일스톤 뒤였다. 기존 영역의 결과는 비트까지 같아야 한다(이 영역은 다른 영역을 고치지 않는다).

**나중 후보** (첫 판 밖): `fftconvolve`·`oaconvolve`(signal·smooth 영역), 사용자에게 보이는 계획 객체(§9의 6번), float32 판(§9의 11번), 실수 대칭 변환의 형 5–8, 비균등 FFT.

---

## 9. 마일스톤을 시작하기 전에 물을 것
모두 (a)로 정했다(§1.1 FT5–FT28; 18번은 사용자 결정, 나머지는 추천안, 사용자 위임 2026-11-23).

추천을 첫째에 둔다.

**FT0 전**
1. **받는 입력**. (a) `sd.ndarray`만 받는다(리스트, numpy 배열은 `TypeError`). 추천: 배열·linalg·special의 함수처럼 다른 계산을 짜는 부품이라 작은 입력으로 자주 불리고, 암묵적 변환이 복사와 dtype 추론을 숨긴다(API.md §5). (b) API.md §5의 상위 함수 입력(시퀀스, numpy, frame의 열): numpy·scipy 코드가 바뀌지 않고 돈다. spatial이 이쪽이지만 spatial의 함수는 사용자가 밖의 자료를 처음 넣는 자리였다.
2. **옵션의 자리**. (a) `n`, `axis`, `s`, `axes`, `norm`, `type`, `d`, `real`은 키워드만이다. 추천: 옵션은 키워드만이라는 공통 원칙이고 spatial의 `metric`·`k`(SPT15), sparse의 `eigsh(A, /, *, k=6)`와 같다. numpy·scipy 코드의 `fft(x, 8)`, `dct(x, 2)`, `fftfreq(n, 0.1)`은 `TypeError`다(`[numpy와 다름]`). (b) numpy처럼 위치로도 받는다: 다른 영역에 없는 꼴이 된다.
3. **`norm`의 기본과 None**. (a) 기본은 `'backward'`이고 None은 `TypeError`다. 추천: 문자열 옵션의 공통 규칙(None이 뜻이 없는 옵션이면 다른 틀린 값과 같다, API.md §2.1)이고, Array API 표준의 fft 확장도 기본을 `'backward'`로 적는다고 알고 있다(설치본에 표준 문서가 없어 확인하지 못했다; (a)는 공통 규칙만으로 선다). (b) numpy처럼 None을 `'backward'`로 받는다: numpy 코드의 `norm=None`이 그대로 돈다.
4. **`workers=`**. (a) 두지 않는다: 스레드 수는 `sd.set_num_threads` 하나다(ARCHITECTURE §4.1). 기본으로 병렬이고 결과는 스레드 수와 무관하다. 추천: spatial(SPT10)·integrate·htest·optimize와 같다. 주면 모르는 키워드의 `TypeError`다(`[scipy와 다름]`). (b) 받되 무시한다: 틀린 기대를 남긴다. (c) 호출마다의 스레드 수로 받는다: 하위 모듈에 스레드 API를 두지 않는 규칙과 어긋난다.
5. **`overwrite_x=`**. (a) 두지 않는다. 추천: linalg가 `overwrite_a`를 두지 않은 것과 같다(docs/api/linalg.md). sd의 결과는 언제나 새 배열이고, 입력을 덮어써도 되는 길(같은 크기의 complex128 입력)에서 얻는 것은 할당 하나다. (b) scipy처럼 받되 "덮어쓸 수도 있다"는 힌트로만 쓴다: scipy의 뜻에도 맞지만(약속이 아니라 허락이다) 사용자가 입력이 바뀌는지 알 수 없게 된다.
6. **`plan=`**. (a) 두지 않는다: 계획은 안의 캐시가 맡는다(§3.6). 추천: scipy 1.18.1도 `plan`에 무엇을 주든 `NotImplementedError`다(이 초안에서 확인). (b) FFTW식 계획 객체를 공개한다: 캐시를 사용자가 쥐게 되지만 지금 쓰는 곳이 없다(나중 후보).
7. **`out=`**. (a) 두지 않는다. 추천: spatial(SPT14)·linalg와 같다. 배열의 원소별 함수에만 있는 꼴이고, 변환의 결과는 입력과 모양·dtype이 다른 일이 많다. (b) numpy처럼 둔다: 큰 배치에서 할당 하나를 아낀다.
8. **길이 0**. (a) 변환 축의 길이 0(입력, `n`, `s`)은 `ValueError`, 변환하지 않는 축의 0은 빈 결과다. `fftfreq(0)`·`rfftfreq(0)`과 `d=0`도 `ValueError`다. 추천: 변환 쪽은 numpy·scipy가 함께 하는 동작이고(규칙 2), 도우미는 같은 뜻의 실수를 같은 예외로 맞춘다(`ZeroDivisionError`는 API.md §2의 목록에 없다; `[numpy와 다름]`). (b) 길이 0의 변환은 빈 결과(수학적으로는 빈 벡터의 DFT가 정의된다): numpy·scipy와 달라진다.
9. **중복된 축**. (a) `fftn` 계열·`dctn` 계열·`fftshift`의 중복 축은 `ValueError`다. 추천: API.md §2(중복된 축 번호는 `ValueError`)와 배열의 모든 함수가 그렇고, scipy의 변환도 그렇다(`[numpy와 다름]`: numpy는 같은 축을 두 번 변환하거나 민다). (b) numpy처럼 거듭한다.
10. **`axes` 없이 준 `s`**. (a) 마지막 len(s)개 축이다. 추천: 지금의 numpy와 scipy가 함께 하는 동작이다. numpy 2.0은 이것을 폐기 예고했지만(나중에 에러) scipy는 그대로 정의한다. (b) `ValueError`(`axes`를 함께 주라): numpy가 갈 쪽이지만 지금의 두 라이브러리 코드가 깨진다. (c) len(s)가 차원 수와 같을 때만 받는다: 세 해석이 모두 같은 경우만 남긴다.
11. **float32 판**. (a) 두지 않는다. 추천: sd에 float32 dtype이 없고(API.md §1), vDSP의 float 판을 쓰려면 라이브러리 전체에 dtype을 더해야 한다. (b) 나중 후보로 적어 두기만 한다(위의 나중 후보에 넣었다).
12. **실수 입력의 `fft`**. (a) dtype이 실수(bool·int64·float64)면 실수 변환을 하고 에르미트 대칭으로 채운다(§3.4). 추천: 약 두 배 빠르고, 대칭이 정확하고, `rfft`와 비트가 같다. 끝자리가 `fft(x.astype(complex128))`와 다를 수 있다(계약 안; scipy도 실수 입력에 실수 변환을 쓴다). (b) numpy처럼 complex128로 바꿔 복소 변환을 한다: 두 dtype의 결과가 같은 비트다.
13. **NaN·inf의 무늬**. (a) IEEE 산술을 그대로 거친 결과이고 어느 칸이 NaN·inf가 되는지는 약속하지 않는다(같은 기계·같은 입력이면 같은 비트). 추천: numpy와 scipy도 서로 다르고(§2.8), 막으려면 줄마다 훑기와 분기가 든다. (b) 줄에 유한하지 않은 값이 하나라도 있으면 그 줄의 결과를 모두 NaN으로: 길과 무관하게 깔끔하지만 inf의 뜻(예: 임펄스 inf의 변환은 모든 칸이 inf)을 잃는다. (c) `ValueError`로 막는다: 원소별 함수처럼 값 때문에 실패하지 않는다는 배열의 원칙과 어긋난다.
14. **`next_fast_len`의 뜻**. (a) scipy와 같은 값(복소 11-매끄러운 수, 실수 5-매끄러운 수)이고, 자체 혼합 기수가 7과 11을 빠르게 한다(§3.3). 추천: 반올림이 아닌 동작이라 scipy 코드가 같은 길이로 채운다(규칙 2). (b) sd가 잰 빠른 길이(vDSP 집합과 자체 기수의 합집합 가운데 실제로 가장 빠른 것): sd에서 더 빠를 수 있지만 scipy와 다른 길이로 채워 결과의 모양이 달라지고, 튜닝할 때마다 바뀐다. (c) vDSP 집합만(2ᵏ, 3·2ᵏ, 5·2ᵏ, 15·2ᵏ): 채움이 커진다.
15. **`orthogonalize`**. (a) scipy의 세 값(None은 `norm == 'ortho'`일 때 True)을 그대로 둔다. 추천: None에 뜻이 있는 옵션이다(API.md §2.1). (b) bool만 받고 기본을 `False`로: `norm='ortho'`의 결과가 scipy와 달라진다.
16. **DCT-I의 N = 1**. (a) `ValueError`(`dct(): type=1 needs at least 2 points, not 1`). 추천: 정의가 N − 1로 나눈다. scipy는 내부 단언이 새어 나오므로(§2.6) 맞출 동작이 없다. (b) 극한으로 y₀ = x₀(`'backward'`)를 정의한다: 다른 라이브러리에 근거가 없다.

**FT1 전**
17. **FFTLog(`fht ifht fhtoffset`)**. (a) FT5 마일스톤으로 넣는다. 추천: scipy.fft에 있는 이름이고(FT2의 "scipy.fft 전부"), `rfft`·`irfft`와 special의 복소 `loggamma` 위에 얇게 선다(FT5 마일스톤에서 계수는 fft 안의 double-double log Γ로 바꿨다: FT58). (b) 나중(signal 영역): FT2의 목록에 이름이 없었다.
18. **vDSP가 받지만 자체 길이 더 빠른 길이**. (a) FT6 마일스톤에서 길이의 꼴마다(예: 5·2ᵏ, 15·2ᵏ) 재서 더 빠른 쪽을 표(§3.1)에 적는다. 언제나 (종류, 길이)의 함수이고 한 길이가 두 길로 가지 않는다. 추천: 성능이 먼저이고(n = 20480에서 vDSP가 numpy보다 느렸다) 결정성은 표가 지킨다. FT1(vDSP가 받는 길이는 vDSP)을 길이 꼴 단위로 좁히는 것이라 사용자의 승인이 필요하다. (b) FT1 그대로 vDSP가 받는 길이는 언제나 vDSP다.
19. **빈 `axes`**. (a) 모든 n차원 함수에서 `ValueError`다(`fftn(): 'axes' must not be empty`). 추천: numpy와 scipy의 `fftn(x, axes=())`는 입력 객체 자체를 돌려준다(복사도 dtype 변환도 없다; 이 초안에서 확인). sd의 결과는 언제나 새 배열이고 dtype이 함수로 정해진다. scipy의 `rfftn`도 "at least 1 axis must be transformed"로 막는다(`[numpy와 다름]`). (b) 변환 없이 complex128 복사본을 돌려준다.
20. **계획 캐시의 상한**. (a) 계획 32개, 합 256 MB의 LRU이고 그보다 큰 계획(2²⁴ 둘레부터) 하나는 캐시에 넣지 않고 호출마다 만든다. 추천: 2²⁰의 vDSP 설정이 13 MB·3.6 ms라 흔한 길이는 다 담고, 메모리가 입력과 무관하게 묶인다. (b) 개수만(16개; pocketfft의 C++ 판이 이렇게 한다고 알려져 있지만 소스가 설치본에 없어 확인하지 않았다): 큰 길이 몇 개가 GB를 잡을 수 있다. (c) 상한을 `sd.set_*` 같은 설정으로 둔다: 영역마다의 설정 API를 두지 않는다.

**FT1·FT3 전**
21. **큰 1차원 변환과 다차원 축의 순서**. (a) 큰 n은 n으로만 정한 4단계로 풀에서 나누고(§3.7), 다차원의 축 순서는 `axes`의 뒤에서 앞으로 고정한다(레이아웃과 무관). 추천: scipy가 `workers`로 1차원 하나를 나누는 것(2²⁰에서 4.7배)을 기본으로 얻고, 결과는 스레드 수·레이아웃과 무관하다. (b) 1차원 하나는 언제나 한 스레드, 축 순서는 메모리 순서(빠른 축 먼저): 단순하지만 큰 1차원이 느리고 레이아웃에 따라 끝자리가 달라진다(API.md §3이 허락은 한다).
22. **`fftfreq`의 계산식**. (a) numpy의 식 그대로(k · fl(1/fl(n·d))): numpy와 비트까지 같다. 추천: 값을 축의 눈금으로 쓰고 `==`로 견주는 코드가 많다. 비용이 없고 비트가 같은 것은 공짜로 얻는 검사다(DESIGN §1.6의 4번). (b) k/(n·d)를 올바르게 반올림: 더 정확하지만 numpy와 끝자리가 다르다.

**FT1 끝에서**
23. **계약의 상수**. (a) 증명한 한계(§4.2의 κ = 8 꼴; 자체 나비마다 증명)를 계약으로 두고, 작은 회귀는 정확도 기준선(최대와 제곱평균)이 잡는다. 추천: spatial·sparse처럼 계약이 증명된 한계라 어떤 입력에서도 지켜야 할 약속이 된다. (b) 잰 값에 여유를 둔 경험적 계약(예: 1.0·L(n)·u): 촘촘하지만 적대적 입력에서 깨질 수 있고 vDSP 길은 증명도 없다.

**FT4 전**
24. **복소 입력의 DCT·DST**. (a) scipy처럼 실수부와 허수부를 따로 변환해 complex128로 돌려준다. 추천: scipy와 같은 결과이고 비용은 실수 변환 두 번이다. (b) `TypeError`: 실수 변환이라는 이름에 맞지만 scipy 코드가 깨진다.
