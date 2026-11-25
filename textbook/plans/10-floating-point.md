# 10장 Floating-Point Numbers — 계획

GUIDE.md와 CHECKLIST.md는 8장 시작 때 다시 읽은 판에서 바뀐 것이 없다(2026-11-25, 마지막 수정 11:01). 옛 8장(legacy/chapters/08-floating-point.tex, 3.4만 자)의 이론(집합과 간격, 비트 배치와 단조성, 반올림과 표준 모형, ULP, Sterbenz, 점진적 언더플로, 특수값, 정수 변환, 컴파일러와 축약)은 살리고 instardust 상자 다섯 개를 본문의 유사코드와 측정으로 옮긴다. 배열 API(`finfo iinfo astype`은 22장, `nextafter copysign maximum`은 27장, `round`는 27–28장, 합의 순서는 29장)는 3권이 맡으므로 여기서는 그 밑의 스칼라 의미론과 커널만 다룬다. 반올림 오차의 해석은 11장, 오차 없는 변환은 12장이 맡는다.

## 이 장이 맡는 커널 (구현까지)
- array/kernels/scalar_ops.hpp: `Nextafter`(비트 패턴 걸음), `fmax_nan`·`fmin_nan`(IEEE maximum, −0 < +0, NEON FMAX와 같은 비트), `fmax`·`fmin`(maximumNumber)
- array/kernels/cast.hpp: float64 → int64(FCVTZS: 0 쪽 자름, 포화, NaN → 0, 벡터 둘씩)
- bench/accuracy/common.py `ulp_distance`(순서를 지키는 정수, uint64 뺄셈) — 정확도 계약의 단위
- CMakeLists.txt: -ffp-contract=off, -fno-builtin-sin/cos, fast-math 없음, 비트 정책 D-33(FMA는 코드에 명시)
- dtype이 float64와 complex128 하나씩인 설계(22장으로 연결)

## 절 배치
1. The Set of Floating-Point Numbers — 정의, 간격과 흔들림(그림), 정수 2^53, 비트 배치, 단조성(증명), 상수 표 / 알고리즘: 비트로 이웃 찾기(nextafter), ULP 거리
2. Rounding — 반올림 함수, 짝수로의 반올림과 치우침(실험 그림), 올바른 반올림, 표준 모형(증명), ULP와 흔들림(그림: 무작위 실수의 상대오차), 정확한 연산, Sterbenz(증명), 이중 반올림(정리와 예)
3. Subnormal Numbers and Gradual Underflow — 정리, 증명, 표준 모형의 확장, 하드웨어 비용(측정: M4에서 서브노멀 곱의 속도)
4. Special Values — ±0(표), ±∞, NaN(조용·신호, 페이로드), maximum/maximumNumber(유사코드), 예외와 기본 결과 / 측정: stardust의 특수값 표
5. Conversion to and from Integers — 정수 → 실수, 실수 → 정수의 세 문제, FCVTZS(유사코드), round의 규칙(Python·numpy·C), 측정
6. Other Formats — binary32/16, bfloat16, 확장 정밀도와 이중 반올림, stardust가 float64 하나인 이유
7. Compilers and Floating-Point Expressions — 결합법칙, fast-math, 축약(FMA)과 비트 재현성, 측정: 축약이 바꾸는 비트
8. Testing, 9. APIs, 10. Concluding Remarks

## 직접 얻을 수치
0.1 + 0.2의 정확한 값(Fraction), 간격·흔들림 자료, 무작위 실수의 반올림 상대오차, 짝수 반올림 대 위로 올림의 치우침, 서브노멀 연산 시간(복제본), 특수값과 변환 결과(복제본), FMA 축약 예(Python math.fma)

## 진행 (2026-11-25)
- 초고 3.9만 자(본문 글자) → 검토 반영 뒤 약 6.1만 자, 절 10, 그림 7, 표 12, 알고리즘 9, 증명 8. 2권 40쪽 경고 0.
- 측정(복제본, 참값은 Fraction·Decimal): fp.py(간격, 반올림 상대오차 최대 0.978u, 치우침 11.5 대 508.5), special_sd.py(특수값·변환·서브노멀), review_sd.py(NaN 비트 전파, 신호 NaN의 maximum 경로 차이, 부호 있는 영 리덕션, 정수 반올림과 −0, 복소 특수값, cmul의 FMA 24.4%, 좁은 형식 가져오기, nextafter·astype·isinf 속도, L1 안 서브노멀 곱), double_round_random.py·double_round_exhaustive.py(연산별 안전한 최소 q: ×÷ 2p, +− 2p+1, √ 2p+2), knuth.py, sterbenz.py(쐐기 안 1755/1755 정확, 밖 1038/2245).
- 검토 반영: 윤문 70쌍(+2차), 수학 21항(κ가 1장 정렬 키와 다름, expm1 범위, +0 항등원 모순, NaN 비트와 결합법칙, FMAX·FMAXNM은 조용한 NaN에서만 같음, 그림 캡션, cos 홀수배, u/(1+u), 2^-54, ULP 정의 출처, Figueroa 가정, AArch64 long double, binary16 넘침 문턱 등), 깊이(비트 판정·지수 분리 소절과 유사코드 둘, 정수→실수 올바른 반올림 유사코드, 마법 상수 반올림 명제·유사코드와 exp의 kExpShift, ULP 거리와 상대오차 명제, 이중 반올림 표, Sterbenz 그림, NaN 비트 표, 예외 표와 부동소수점 환경 가정, 복소 특수값 소절, cmul FMA 사례, IntSum 정정과 두 번 반올림 반례, 정수 반올림 표, 넓히기 명제와 버퍼 가져오기, bool 변환, 시험 유사코드와 신호 NaN).
- LIBRARY-NOTES: 신호 NaN에서 maximum의 스칼라·NEON 경로 차이(우선순위 중간), nextafter 주석 수치, sd.log(-1) 페이로드.
