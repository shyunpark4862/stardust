# 2장 Integers and Modular Arithmetic — 계획

GUIDE.md 전체와 CHECKLIST.md를 다시 읽고 쓴다(2026-11-25). 옛 2장(legacy/chapters/02-integers.tex, 3.7만 자)의 Theory(나눗셈 정리, 혼합 기수, 유클리드·Lamé·Bézout, 합동·가역원, 2의 보수, Hensel, Lemire, 전단사, F2[x])는 살리고, 상자 5개(groupby 키 접기, copy.cpp 겹침, philox bounded, SeedSequence hashmix, CSV prefix xor)에 들어 있던 구현을 본문의 Algorithms·Implementation으로 옮긴다. 원소별 정수 연산의 배열 엔진(루프·SIMD)은 27장, 해시 테이블은 19장, 난수 생성기 전체는 X권, CSV 토크나이저는 45장의 몫이다. 이 장은 그 구현들이 기대는 정수론과 그 정수론을 그대로 옮긴 스칼라 알고리즘을 맡는다.

## 이 장이 맡는 API·규칙 (구현까지)
- `floor_divide`, `remainder`(정수): 바닥 나눗셈을 자르는 나눗셈에서 얻는 보정, `b=0`·`b=-1`·`INT_MIN` 경계(scalar_ops.hpp FloorDivide/Remainder)
- 정수 `add/subtract/multiply/negative/abs`의 감김(wrap_*), `pow`(int_pow: 제곱 거듭, 음의 지수)
- `bitwise_left_shift/right_shift`(셈이 64 이상·음수일 때), 산술 오른쪽 밀기 = 바닥 나눗셈
- 혼합 기수로 키 접기(frame groupby 다중 키, 넘침 전 다시 번호 매기기) — 42장이 집계를 맡고 여기서는 접기 알고리즘
- `may_share_memory`/대입 겹침 판정의 gcd 조건(copy.cpp)
- FFT의 캐시 집합 판정 `cache_friendly`(fft/driver.cpp)
- 정확 검정 분포의 격자 축소(htest rank.cpp·gof2.cpp: 점수들의 gcd로 DP 상태 수 줄이기)
- 난수: `bounded`(Lemire), `to_unit`, Philox 라운드의 전단사성, SeedSequence hashmix
- 해시: fmix64(정수 키), HashBytes의 mum 섞기, float_key(NaN·−0 정규화) — 표 구조는 19장
- PNG 쓰기의 CRC-32·Adler-32(plot/_png.py, zlib) — 수학과 알고리즘
- CSV의 prefix xor(PMULL) — 수학과 커널

## 절 배치 (세 축)
1. Division and Positional Notation — T: 나눗셈 정리, 세 가지 나눗셈 약속(T·F·E) / A: 바닥 나눗셈 보정 알고리즘(증명), 혼합 기수 인코딩·디코딩, 키 접기와 다시 번호 매기기, 불변 수로 나누기(Granlund–Montgomery 마법수, 정리와 증명) / I: stardust FloorDivide·Remainder의 경계, numpy와 비교 표, 나눗셈 속도 측정(sdiv vs 마법 곱)
2. Greatest Common Divisors — T: 유클리드 보조정리, Lamé, Bézout, 이진 gcd의 정확성 / A: 유클리드, 확장 유클리드, 이진 gcd 유사코드 / I: std::gcd의 음수·INT_MIN, 속도 측정, stardust의 gcd 사용처 셋(겹침 판정, 캐시 집합, 정확 분포의 격자 축소)
3. Congruences and Residue Rings — T: 합동, 가환환, 가역원, CRT / A: CRT 재구성(Garner) / I: Adler-32(소수 65521, 미룬 나머지의 한계 5552 유도, 이어 붙이기 공식), PNG 쓰기
4. Machine Integers — T: Z/2^64, 2의 보수, 부호 있는 순서, 밀기와 2의 거듭제곱 나눗셈, 넘침 판정 / A: 감김 연산, 제곱 거듭 pow, 2^64 법의 역원(Newton), 곱의 위쪽 절반(32비트 조각), Lemire / I: scalar_ops의 UB 회피, 밀기 셈 경계, philox bounded·to_unit, 128비트 정수 누산(htest)
5. Bijections and Bit Mixing — T: 전단사 부품, xorshift 역함수, 눈사태 기준 / A: fmix64와 그 역함수, Philox 라운드 역산 / I: hash.hpp(Fmix64, float_key, HashBytes mum은 전단사가 아님), SeedSequence hashmix, 눈사태 측정 그림
6. Binary Polynomials and Finite Fields — T: F2[x], 나눗셈 정리, 기약다항식과 체, CRC가 짧은 폭주 오류를 모두 잡는다는 정리 / A: prefix xor, CRC 비트·표 알고리즘 / I: CSV 따옴표 마스크(PMULL), PNG 청크 CRC-32, zlib과 비교
7. Testing Integer Code — 작은 단어(8·16비트)에서 전수 시험, 큰 정수(Python int)를 기준으로, numpy와 차이
8. APIs in This Chapter (본문 + 표)
9. Concluding Remarks

## 그림
- 그림 1: a에 대한 몫·나머지 계단(자르기 vs 바닥, b=3) — 직접 계산
- 그림 2: 유클리드 반복 횟수의 최댓값과 Lamé 한계(b ≤ N) — 직접 계산
- 그림 3: x[::2]와 x[1::2]의 바이트 배치와 gcd 판정 — TikZ
- 그림 4: 보폭에 따른 캐시 집합 수 16384/gcd — 직접 계산
- 그림 5: Newton 역원의 맞는 비트 수(반복마다 두 배) — 직접 계산
- 그림 6: 16비트 단어에서 Lemire의 버림 확률 t/2^16과 w mod r의 치우침 — 직접 계산
- 그림 7: 눈사태 행렬(곱 한 번 vs fmix64) — 직접 계산, 회색 열지도
- 그림 8: CSV 따옴표 마스크와 prefix xor — TikZ, 비트는 직접 계산

## 표
- 세 나눗셈 약속의 몫·나머지(±7, ±2)
- 정수 경계값: stardust 규칙 vs numpy(직접 실행)
- 나눗셈 속도(C 측정)
- Lemire 3비트 예(옛 표)
- CRC-32 매개변수와 검증값
- 시험 표, API 표

## 유사코드
바닥 나눗셈·나머지, 혼합 기수 인코딩/디코딩, 키 접기(다시 번호 매기기), 마법수 계산·불변 나눗셈, 유클리드, 확장 유클리드, 이진 gcd, 겹침 판정, CRT, Adler-32, 제곱 거듭 pow, 밀기, 2^64 역원, mulhi(32비트 조각), Lemire bounded, fmix64 역함수, CRC-32(표)

## 직접 얻을 수치
- 경계값(numpy 실행), 몫·나머지 표, 유클리드 반복 수, 캐시 집합 수, 역원 비트, Lemire 전수(8·16비트), 눈사태 행렬, CRC/Adler 검증(zlib과), 따옴표 마스크, 나눗셈·gcd 속도(C)

## 진행 (2026-11-25)
- 초고 8만 자, 절 9, 그림 7, 표 9, 알고리즘 21, 증명 32. audit 통과, 1권 92쪽 경고 0. 그림 7개 100dpi로 확인(겹침 판정 그림의 화살표 글자, 캐시 그림 눈금, Lemire 그림을 로그-선형으로, 눈사태 그림은 TeX 메모리 초과로 PNG+addplot graphics, CSV 그림 따옴표 글리프).
- 측정: figures/02/divbench.cpp(바닥 나눗셈 SDIV 0.46ns 대 마법수 0.21ns, gcd 유클리드 109 대 이진 58 대 std::gcd 59ns, 유클리드 평균 37.45번), numpy_edges.py(%0, 음의 지수 차이), euclid_steps.py, cache_sets.py, lemire_bias.py(8·12비트 전수), avalanche.py, newton_inverse.py, crc_adler.py(zlib 비교), quote_mask.py, fmix_inverse.py, conventions.py.
- LIBRARY-NOTES: 스칼라 제수 floor_divide의 마법수 경로(2.2배), factorize_rows 넘침 가드(행 2^31 초과).
- sdbook.cls에 pgfplots groupplots 라이브러리를 더함.
- 검토 반영(2026-11-25): 윤문 처방 80건과 쉼표 62곳(복원 스크립트·게이트 통과, 각주 경고는 오탐), 수학 검토 18항(T-나머지 식의 |b|→b, 마법수 m<2^64 근거, Heilbronn 진술과 어림 구분, alg:int-pow 8행, mum 충돌, 해시 예, Philox 사슬 깊이 10, CRC XOR 역할 등), 깊이 검토 21항(integers의 범위 하나 경로 유사코드 alg:integers, Euclid 보조정리와 binom_u128 alg:binom, 곱셈 넘침 명제, sign·bitwise_invert·square, CRT 격자 그림과 Garner 예, NP-어려움 환원, 캐시 줄·집합 보조 논증, to_unit 증명, 정수 키 직접 표, CSV 조각 홀짝, 1장과 맞추기). 9만 자, 그림 8, 표 9, 알고리즘 23, 증명 34. 1권 98쪽 경고 0, 바뀐 그림 다시 확인.
- 점검표: 구성·Theory·Algorithms·Implementation·그림과 수치·문체와 조판 모두 통과(audit 통과, 라벨 중복 0). LIBRARY-NOTES에 composite_keys 부호 있는 넘침, offset_of 나눗셈 두 항목을 더함. figures/02/crt.py 추가.
