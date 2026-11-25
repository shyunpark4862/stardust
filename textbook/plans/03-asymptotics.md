# 3장 Asymptotic Analysis — 계획

GUIDE.md·CHECKLIST.md를 다시 읽고 쓴다(2026-11-25, 2장 직후; 두 파일은 2장 시작 뒤 바뀌지 않았다). 옛 3장(legacy/chapters/03-asymptotics.tex, 3만 자)의 Theory(점근 기호, 점근 급수와 erfc 꼬리, Laplace·Stirling·Watson, Euler–Maclaurin, Stirling 급수, 연분수와 Lentz)는 살리고, instardust 상자 셋(erf.hpp, zeta.hpp, cf.hpp)의 내용을 본문의 Algorithms·Implementation으로 옮긴다. 특수함수 하나하나의 설계는 VIII권의 몫이고, 이 장은 점근 도구와 그 도구를 그대로 옮긴 커널을 맡는다.

## 이 장이 맡는 API·경로 (구현까지)
- `special.erfcx`(x ≥ 8의 점근 급수 17항, 안쪽부터 겹친 계산)
- `special.gammaln`/`loggamma`의 Stirling 영역(lgamma_pos x ≥ 13, Cephes의 맞춘 계수 kLgamA; 복소 loggamma 8항), digamma의 점근 급수
- `special.zeta`(Hurwitz, Euler–Maclaurin: 직접 합 + 적분 + Bernoulli 보정, sm1을 정확히 받기, q > 1e8의 세 항)
- 연분수: `special/cf.hpp`(수정 Lentz, tiny = 2^-1000, b0 = 0 함정, lentz_backward), `cf_fast.hpp`(근사분수 점화식 + 2의 거듭제곱 척도 맞춤 + 뒤에서부터 P/Q) — `gammainc`/`gammaincc`, `betainc`, Bessel 비
- 이항·음이항·Poisson pmf의 안장점 꼴(Loader): Stirling 보정 δ(z)와 μ(t) = t − log(1+t) 급수(stats/dist/discrete_basic.hpp)

## 절 배치 (세 축)
1. Asymptotic Notation — T: 정의·규칙·균등성, 합과 적분의 비교 / A: 측정에서 증가 차수 추정(로그–로그 기울기) / I: 직접 짠 C 커널의 측정 그림(정렬 n² 대 n log n, 캐시 단계)
2. Asymptotic Series — T: 정의·유일성·erfc 꼬리 정리·최적 절단 / A: 겹친 계산과 반올림 오차 분석, 항 수 정하기 / I: stardust erfcx, 정확한 참값(고정소수점 정수 산술)으로 오차 측정
3. Laplace's Method — T: Laplace 정리, Stirling, Watson / A·I: 이항 pmf의 안장점 꼴(정확한 항등식 증명, 유사코드, |t| ≤ 1/8 급수, Stirling 보정), 순진한 lgamma 차 계산과 비교 그림
4. The Euler–Maclaurin Formula — T: Bernoulli, 정리와 증명, Bernoulli 수의 증가 / A: Hurwitz 제타 유사코드, 직접 합 길이와 보정 수, 오차 한계 / I: zeta.hpp(sm1, 큰 q, 음의 q 일부), 그림: 보정 수에 따른 오차
5. The Stirling Series — T: 유도, 실수에서 나머지가 첫 버린 항보다 작다는 정리(Binet 적분의 줄거리) / A: lgamma의 영역 나누기 / I: Cephes 맞춤 계수와 정확한 계수 비교, 그림: 항 수별 절단 오차
6. Continued Fractions — T: 정의, 점화식, 수렴(Śleszyński–Pringsheim 줄거리) / A: Lentz, 수정 Lentz, 뒤에서부터, 척도 맞춘 점화식 / I: cf.hpp·cf_fast.hpp, 그림·표: 앞으로 대 뒤로의 ULP 오차(직접 측정)
7. Testing Asymptotic Code — 나머지 정리를 보증된 오차 한계로 쓰기, 정확한 유리수 참값, 영역 경계 시험
8. APIs in This Chapter, 9. Concluding Remarks

## 그림
- 측정한 시간과 기울기(삽입 정렬, 병합/std::sort), 보폭 합의 캐시 단계
- erfcx 점근 급수 항 크기(x = 2, 3, 4, 8), 최적 절단
- 이항 pmf 상대오차: 순진한 lgamma 차 대 안장점 꼴(직접 구현, 정확한 유리수 참값)
- Euler–Maclaurin 보정 수에 따른 ζ(2)·Hurwitz 오차
- Stirling 급수 항 수별 절단 오차(정확한 Bernoulli)
- 연분수 앞으로/뒤로 오차 분포

## 유사코드
증가 차수 추정, 점근 급수 겹친 계산(erfcx), 이항 pmf 안장점 꼴, Hurwitz 제타(EM), lgamma_pos, 수정 Lentz, 뒤에서부터 연분수, 척도 맞춘 근사분수 점화식

## 직접 얻을 수치
erfc 최소 항 표(정확 유리수), erfcx 오차(정확 참값), 이항 pmf 오차, ζ(2) 보정 표, Stirling 절단 오차, Lentz 대 뒤로의 ULP, 정렬·캐시 측정, tan(1)·√2 연분수 수렴

## 진행 (2026-11-25)
- 초고 5만 자, 절 9, 그림 7, 표 5, 알고리즘 7, 증명 12. audit 통과, 1권 128쪽 경고 0. 그림 7개 90dpi로 확인(포인터 추적 그림의 '메모리' 글자 위치 고침).
- 측정: growth.c(삽입 기울기 1.92, 병합 1.09, T/(n log n) 2.83–2.84ns; 포인터 추적 L1 0.89ns, L2 4–11ns, 메모리 약 120ns), erfc_asym.py(erfcx 급수 최대 1.13ULP), binom_pmf.py(순진 1e-10 대 안장점 D 정확 2e-15), euler_maclaurin.py(ζ(2) 표, 최적 m≈πN, Cephes 꼴 3.5–9.1ULP), stirling_series.py(맞춘 계수 x=13에서 7.6e-19 대 정확 5개 1.0e-15), cf_errors.py(Lentz 15.6ULP 대 뒤에서부터 4.8ULP), small_checks.py.
- 남은 것: 깊이·수학·문체 검토 반영(알고리즘·구현 축 보강 예상).
- 검토 반영(2026-11-25): 윤문 처방 74건과 쉼표 8곳(게이트 통과), 수학 검토 14항(EM 나머지는 마지막 더한 항 이하 + 부호 조건 아래 첫 버린 항 이하, Binet 줄거리의 유한 등식, μ 급수의 t<0, erfcx 반올림 4.1u, 줄 번호, A_j 유리수, Lentz b0=0의 τ+f, Kahan 2u 등), 깊이 검토 13항(라이브러리 복제본으로 직접 측정: erfcx 비트 일치·zeta 같은 최대 오차·이항 1.4e-16; 오차 예산 명제; Temme 앞 인수와 double-double μ 알고리즘; 적응형 절단(expint, Hankel); lnΓ(2+e) 급수 명제; lbeta_large·dd::loggamma; Hurwitz a+=1 원인과 LIBRARY-NOTES; hurwitz_relative; Γ(a,x) 연분수 유도 줄거리와 층 수; 연분수 반올림 명제와 lentz_backward 알고리즘; Laplace·근사분수 그림; API 표 확장). 6.7만 자, 그림 9, 표 5, 알고리즘 11, 증명 17, 1권 138쪽 경고 0.
- CHECKLIST에 복제본으로 확인하는 항목을 더함.
