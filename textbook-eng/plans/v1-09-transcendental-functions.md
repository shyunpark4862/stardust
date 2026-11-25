# 1권 9장 Elementary Transcendental Functions — 얼개

예비 권(리뷰 방식). Part II Calculus. **이 장은 앞 장들이 미뤄 둔 빚을 갚는 장이다.** plans/PROMISES.md의 9장 약속을 하나도 빠짐없이 지킨다: exp·ln의 정의와 성질(8장의 급수 E, 7장의 Λ를 받아 ln := Λ, e^x := E(x)로 확정), 실수 지수 a^x와 x^s(7장 감마 절, 6장 Bernoulli 실수 지수판, 8장 p-급수·이항 급수의 일반 꼴), 삼각함수의 엄밀한 정의와 3장·5장의 삼각 Assumption 전부(홀짝성, sin(π/2)=1, sin(kπ)=0, 주기, π>3, 부채꼴 넓이 x/2, 단조성, (0,π/2)에서 양수, sin x < x < tan x), e^{iθ}의 정의(3장), 1 − t ≤ e^{−t}(4장), ln의 오목성과 Jensen으로 AM–GM(6장), 도함수 표의 e^x·ln x(6장), 부채꼴 넓이를 적분으로, 역삼각함수. 순환이 없도록 정의 순서를 엄격히 한다: 삼각함수는 급수(8장의 사인·코사인 급수)나 적분(7장의 arctan·arcsin 적분 표현) 가운데 하나로 정의하고, 학교 기하의 사실은 정리로 증명한다.

## Abstract (한 문단, 120단어 이내, 동기부여)
exp, log, sin, cos는 계산기 단추처럼 당연해 보이지만 정의부터 따져야 성질이 증명되고, 성질이 있어야 컴퓨터가 이들을 정확히 계산하는 방법(인자 축소, 항등식)이 선다는 점.

## 절 구성
9.1 The Exponential Function — e^x := E(x)(8장), 함수방정식 E(x+y) = E(x)E(y)(Cauchy 곱 또는 미분으로 증명), 양수·증가·볼록, E' = E와 초기값 문제의 유일한 해(증명; 미분방정식 맛보기, Part IV 예고), 극한 (1+x/n)^n → e^x(증명), 성장 비교 x^k/e^x → 0. 부등식 e^x ≥ 1 + x(4장 약속).
9.2 The Natural Logarithm — ln := Λ가 exp의 역함수(8장 prop:log-exp 인용하여 완결), 성질, 도함수와 급수, 오목성, 부등식 x/(1+x) ≤ ln(1+x) ≤ x, Jensen으로 AM–GM(6장 약속), 성장 비교 ln x/x^s → 0. 그림: exp와 ln의 그래프와 대칭.
9.3 General Powers and Logarithms — a^x := e^{x ln a}, 유리 지수 정의(3·6장)와의 일치(증명), 지수 법칙, x^s의 도함수와 실수 지수 Bernoulli(6장 약속), log_a, 밑 변환, log₂와 자릿수·비트 수(뒤 권 암시). 7장 감마 절이 앞당겨 쓴 사실을 여기서 확정.
9.4 Numerical Aspects of exp and log — x가 0에 가까울 때 e^x − 1과 ln(1+x)의 상쇄와 그 조건수(6장의 κ), 인자 축소의 생각 e^x = 2^k e^r(순수 수학으로; 구현은 10권 예고), 로그 눈금과 지수적 자람·감쇠(반감기), 표: 직접 계산과 급수 계산의 오차(직접 계산한 수치).
9.5 The Trigonometric Functions — 정의(급수로: 8장의 사인·코사인 급수를 정의로 삼는다), cos² + sin² = 1(증명), 덧셈 정리(증명), π의 정의(cos의 가장 작은 양의 영점의 두 배; 존재 증명), 주기성·홀짝성·특수값(증명), 단조성과 부호, sin x < x < tan x(증명), 원의 매개화와 호의 길이·부채꼴 넓이(7장의 길이·넓이로 증명 — 기하와의 일치), π > 3과 π의 수치. **3장·5장 Assumption의 각 항목이 어느 정리로 증명되는지 대응 표를 둔다.**
9.6 Inverse Trigonometric Functions — arcsin, arccos, arctan의 정의(5장의 역함수 정리), 도함수, 7장의 적분 표현과 일치, 급수(8장), atan2와 주편각 Arg(3장과 연결). 그림: 가지와 그래프.
9.7 The Complex Exponential — e^{z}의 급수 정의(복소 급수의 수렴은 절대 수렴으로 필요한 만큼), Euler 공식 e^{iθ} = cos θ + i sin θ(증명; 3장의 표기를 정리로 격상), 덧셈 정리의 재해석, de Moivre, 단위근, 복소 로그와 가지는 맛보기(2권). 그림: 단위원과 e^{iθ}.
9.8 Hyperbolic Functions — sinh, cosh, tanh와 역함수, 항등식, 삼각함수와의 관계(cos(ix) = cosh x), 현수선은 한 문단. 표: 삼각·쌍곡 항등식 대조.
9.9 Identities and Argument Reduction — 배각·반각·합을 곱으로, 인자 축소(주기와 대칭으로 [0, π/4]로 줄이기)의 수학과 큰 인자에서 π의 근삿값이 만드는 오차(왜 정밀한 π가 필요한가; 구현은 10권 예고), 표: 축소 규칙. 알고리즘: 인자 축소(유사코드, 순수 수학판).

## note 상자 후보
- e와 π의 초월성(진술과 문헌)
- 감마 함수와 계승의 보간(7장 연결) 한 문단
- Chebyshev 다항식 cos(n arccos x) 맛보기(뒤 권 예고)

## Notation 장에 넣을 기호
exp, e^x, ln(이제 Λ를 대체), log_a, a^x, sin cos tan, π, arcsin arccos arctan, atan2, sinh cosh tanh, e^{iθ}, e^z

## 분량
제한 없음. 65쪽 안팎 예상.

## 장을 마친 뒤 할 일(이 장의 집필자가)
- 앞 장들의 Assumption(3장 삼각 가정, 5장 asm:trig-geometry)과 "Chapter 9" 글자 참조를 \cref로 바꾸고, Assumption 본문은 그대로 두되 "proved in \cref{…}"가 정확한지 확인한다(순수 참조 변경만).
- 7장의 Λ 표기는 그대로 둔다(그 장의 관례). 이 장에서 ln = Λ임을 정리로 밝힌다.
