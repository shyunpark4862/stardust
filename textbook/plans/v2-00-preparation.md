# 2권 Analysis — 준비 (1권과의 중복 정리, 받은 약속, 권장 장 구성)

2026-11-27 작성. 읽은 것: GUIDE.md, OUTLINE.md(2·3·10·11권 목차와 권 사이 의존), plans/PROMISES.md의 "- 2권:" 줄 11개, vol1.tex(Notation), 1권 1–27장 원고(정리 꼴 환경 1395개를 꼬리표·증명 상태와 함께 뽑아 해석 관련 장을 전수 대조, note 상자와 본문의 "Volume II" 언급 88곳을 문맥째 읽음). 27장은 검토 중이라 꼬리표가 바뀔 수 있다.

상태 표기: **증명** = `\begin{proof}`(완전), **줄거리** = `\begin{proof}[sketch]`, **진술** = 정리 환경은 있으나 증명 환경이 없음, **note** = note 상자에서만 다룸(진술·짧은 논증·인용), **문단** = 본문 문단에서 언급만, **없음** = 1권에 없음. "(5장)"은 1권 장 번호.

---

## A. 1권이 이미 증명한 해석학

### A.0 요약
- 1권은 일변수·다변수 미적분을 공리(asm:reals)에서 출발해 사실상 Spivak 수준으로 완전히 증명했다: 극한·연속·EVT·IVT·고른 연속, 평균값 정리 일가, l'Hôpital, Taylor(Peano·Lagrange·Cauchy·적분 나머지), 볼록·Jensen(유한 꼴), Darboux 적분 전부(FTC 두 부분, 이상 적분), 거듭제곱 급수(고른 수렴 없이 직접), 초등 함수의 엄밀한 정의(급수로 exp·sin·cos, π의 정의), R^n의 미분 전부(연쇄 법칙, Schwarz, Taylor, Lagrange 승수, Newton), 다중 Riemann 적분(Jordan 내용, Fubini, 이상 다중 적분, Gauss 적분, 감마·베타), 벡터 미적분(Green·발산·Stokes).
- **뜻밖에 이미 된 것**: 단순 연결 영역에서 닫힌 C^1 장은 기울기장이라는 정리 thm:simply-connected(12장, 연속 호모토피로 정의, 완전 증명) — CR 방정식과 합치면 C^1 정칙 함수의 Cauchy 정리가 바로 나온다. 선형계의 대역 존재·유일성 thm:linear-existence(24장, Picard 반복 완전), 해석 계수 정상점 급수 해 thm:ordinary-point와 Frobenius thm:frobenius(26장, 완전), Gelfand 공식 thm:gelfand(19장), 매개변수 이상 적분의 고른 수렴 thm:parameter-improper(11장), Bessel 부등식·최선 근사가 **임의의** 내적 공간에서(18장), 무한 차원 Riesz 표현의 반례(18장), ∫sin x/x = π/2(11장 thm:sinc-integral, 실해석 증명), Gauss 함수의 Fourier 변환(11장 prop:gaussian-fourier), 변수 변환의 선형 경우 thm:linear-volume(11장, 완전).
- **빠진 것**: 거리 공간·컴팩트(열린 덮개)·Heine–Borel·연결성 일반, 일반 고른 수렴 이론(함수열의 극한 교환, 일반 M-판정은 note뿐), Weierstrass 근사(인용만), Arzelà–Ascoli(이름만), Baire, 축약 사상 정리 일반(1차원 미분 가능 꼴만), 역함수·음함수 정리(줄거리), 변수 변환 일반(줄거리), Picard–Lindelöf·Peano·최대 해(줄거리), Fourier 급수의 수렴(L²·점별·고른 모두 없음), Riemann–Stieltjes·유계 변동(note), Lebesgue 판정법(note), 실수의 구성(note), 복소 미분과 복소해석 전부(CR은 언급, 복소 거듭제곱 급수는 note), 측도론 전부.

### A.1 실수와 완비성
- def:complete (3장) — 정의: 완비 순서체 = 상한 공리.
- asm:reals (3장) — **공리**. 구성은 note 'Dedekind cuts and Cauchy sequences'(3장 1693행)에서 두 구성의 개요만(Rudin·Landau·Ebbinghaus 인용); note는 "완비화는 Volume II"라고 적음.
- thm:r-unique (3장) — 줄거리(Spivak 인용). 완비 순서체의 유일성.
- prop:q-gap (3장) — 증명(Q는 완비가 아님). lem:sup-approx, prop:inf — 증명.
- thm:archimedes, cor:archimedes, thm:floor, thm:roots(n제곱근 존재), thm:q-dense(유리수·무리수 조밀), prop:intervals(구간 ⇔ 사이 성질) (3장) — 증명.
- thm:monotone (3장, 단조 수렴) — 증명. thm:real-expansion, lem:expansion-basic (3장, b진 전개) — 증명.
- 완비성의 동치 꼴: note 'Equivalent forms of completeness'(5장 616행) — (a)상한 ⇒ (b)단조 수렴 ⇒ (c)BW·축소 구간 ⇒ Cauchy 수렴, (c) ⇒ Archimedes는 본문 증명으로 읽음; 원을 닫는 나머지 함의와 비Archimedes 반례(형식 Laurent 급수체, 초현실수)는 Propp 인용만. 본문 3장 1655행·5장 614행이 "Volume II가 Cauchy 수열로 완비성을 동치로 말한다"고 약속.
- 가산성 (2장): thm:q-countable, thm:binary-sequences, thm:r-uncountable, thm:cantor — 증명; thm:schroeder-bernstein — 줄거리; 선택 공리·Zorn은 note 'Axiomatic set theory and the axiom of choice'(2장 189행).
- 확장 실수: def:diverge-infinity, def:limits-infinity (5장) — 정의. def:limsup(liminf 포함)·lem:limsup (8장) — 정의·증명.

### A.2 수열, Cauchy 판정법, Bolzano–Weierstrass, 축소 구간
- prop:limit-variants, prop:seq-basic, prop:convergent-bounded, thm:limit-laws, prop:limit-order, thm:squeeze, cor:squeeze-zero, prop:infinite-laws, thm:standard-limits (5장) — 증명.
- cor:monotone-dichotomy, prop:heron, thm:e·def:e (5장) — 증명.
- thm:nested (5장, 축소 구간) — 증명.
- def:subsequence, prop:subsequence, lem:monotone-subsequence, thm:bolzano-weierstrass (5장) — 증명.
- def:cauchy, thm:cauchy-criterion (5장) — 증명.
- def:convergence-rn, prop:componentwise, thm:bolzano-weierstrass-rn (10장) — 증명.
- 급수 (8장): prop:geometric-series, thm:series-rules, thm:term-test, thm:cauchy-series, prop:grouping, prop:nonnegative-series, thm:comparison-series, cor:limit-comparison-series, thm:integral-test, cor:p-series, thm:euler-constant, thm:ratio-test, thm:root-test, thm:absolute-series, prop:positive-negative-parts, thm:leibniz, thm:rearrangement-absolute, thm:riemann-rearrangement, thm:cauchy-product, prop:tail-bounds, prop:richardson — 증명. 복소 급수 lem:complex-series (9장) — 증명. 이중 급수의 순서 교환은 11장 note 'The Lebesgue integral and the Fubini–Tonelli theorem' 끝 문장(counting measure)에서만.

### A.3 R과 R^n의 위상, 컴팩트, Heine–Borel, 연결
- R의 열린 집합: note 'Continuity and open sets'(5장 995행) — 정의와 "R→R 연속 ⇔ 열린 집합의 역상이 열림"을 note 안에서 논증; "거리 공간에서는 Volume II"(2장 682행·5장 996행 문단도 같은 약속).
- def:open-closed (10장) — 정의: 열린·닫힌 공, 내부점, 열린·닫힌 집합, 경계점, 유계, 근방. 폐포·내부의 기호는 없음. prop:closed-sequences(닫힘 ⇔ 수열 극한을 포함), prop:level-sets-closed — 증명.
- **컴팩트**: 정의 없음. thm:evt-rn 뒤 문단(10장 479행)이 "닫힌 유계 집합 = Volume II의 컴팩트 집합"이라고만 함. 열린 덮개, Heine–Borel, 순차 컴팩트의 동치 없음(실제로는 BW + prop:closed-sequences로 순차 컴팩트성을 씀). lem:exhaustion-compact (11장, 닫힌 유계 집합이 소진열의 한 항에 들어감) — 증명(덮개 논법의 특수형).
- **연결**: prop:intervals (3장), cor:interval-image (5장, 연속상은 구간) — 증명. R^n에서는 def:connected-open (12장: 꺾은선으로 잇는 열린 집합)과 cor:zero-gradient 뒤 문단(10장)만; 일반 연결성·경로 연결성·성분 없음. note 'The two components of the invertible matrices'(17장 992행) — 줄거리(기본 행렬 분해로 경로 구성; 2권 약속).
- 단순 연결: def:simply-connected (12장, 연속 호모토피) — 정의.
- note 'Functions with many discontinuities'(5장 991행): Thomae 함수(note 안 논증), "유리수에서만 연속인 함수는 없다"는 Baire 인용(2권).
- 거리 공간·노름 공간 일반, 완비화, Cantor 집합(7장 note에서 이름만): 없음.

### A.4 연속, 고른 연속, 최대·최소, 중간값
- def:continuity, thm:continuity-forms(ε-δ ⇔ 수열), thm:continuity-algebra, cor:rational-continuous, thm:composition-continuous, lem:sign-persistence (5장) — 증명.
- thm:bolzano, cor:ivt, cor:interval-image, prop:bisection, prop:odd-degree, thm:fixed-point(1차원 Brouwer) (5장) — 증명.
- thm:bounded, thm:evt, cor:image-closed-interval, cor:positive-lower-bound (5장) — 증명. thm:evt-rn (10장) — 증명(BW로).
- def:uniform-continuity, prop:uniform-cauchy, thm:heine-cantor (5장) — 증명. lem:heine-cantor-rn (11장) — 증명. def:lipschitz, prop:grid-max (5장) — 증명.
- 단조 함수: thm:monotone-one-sided, prop:monotone-countable, thm:monotone-continuity, thm:inverse-continuous (5장) — 증명; prop:injective-monotone — 줄거리.
- thm:cauchy-functional (9장, 연속 가법 함수는 cx) — 증명. 불연속 가법 함수는 note 'Bases of infinite-dimensional spaces'(15장, Hamel)에서만.

### A.5 미분, 평균값 정리, Taylor, 볼록, Jensen
- def:derivative, thm:linear-approx, lem:caratheodory, prop:diff-continuous, prop:one-sided-derivatives, thm:diff-algebra, prop:power-rule, thm:chain-rule, thm:inverse-derivative, cor:root-derivative, thm:trig-derivatives, cor:inverse-trig-derivatives (6장) — 증명.
- thm:fermat-extremum, prop:candidates, thm:rolle, thm:mvt, thm:cauchy-mvt, cor:zero-derivative, cor:monotone-derivative, cor:lipschitz-derivative, lem:comparison (6장) — 증명.
- thm:lhopital, cor:lhopital-infinity, thm:lhopital-infinite (6장) — 증명.
- def:higher-derivative, thm:leibniz-rule, def:ck, prop:ck-algebra, thm:darboux, cor:darboux, prop:derivative-limit, prop:x2sin (6장) — 증명.
- note 'Continuous functions that are nowhere differentiable'(6장 229행) — Weierstrass·du Bois-Reymond 예, 연속성은 "8장과 Volume II", 미분 불가능성 증명 없음, Banach의 범주 정리는 "Volume II가 Baire로 유도". note 'Which functions are derivatives?' — Volterra 함수(인용), "Lebesgue 적분의 동기".
- 볼록: def:convex, lem:slopes, thm:convex-derivative, cor:convex-second, prop:tangent-below, cor:convex-minimum, thm:jensen(**유한 꼴**), cor:power-means, thm:second-derivative-test (6장) — 증명. 볼록 함수의 연속성·한쪽 도함수의 존재·단조성은 정리로 없음.
- Taylor (8장): def:taylor-polynomial, lem:taylor-matching, thm:taylor-peano, thm:taylor-lagrange(Lagrange·Cauchy 나머지), thm:taylor-integral, prop:newton-ratio, prop:difference-taylor, prop:exactness-order, def:taylor-series, thm:taylor-convergence, prop:flat-function(e^{−1/x²}) — 증명. bump 함수는 prop:flat-function 뒤 문단(8장 1380행)에서 "Volume II".
- 부등식 (9장): prop:exp-tangent, prop:ln-inequalities, thm:amgm-log, prop:bernoulli-real, thm:weighted-amgm, cor:young — 증명. Hölder는 cor:young 뒤 문단(9장 437행)과 note 'The p-norms'(13장)에서 진술만.
- Newton: thm:newton-quadratic, cor:newton-local (6장) — 증명. def:condition-number, prop:condition-number (6장) — 증명.

### A.6 Riemann/Darboux 적분, 적분 가능성, 기본 정리, 이상 적분
- def:interval-partition, def:darboux-sums, lem:refinement, cor:lower-upper, def:riemann-integral, thm:riemann-criterion, cor:riemann-sequence, prop:dirichlet, def:riemann-sum, thm:riemann-sums (7장) — 증명.
- thm:monotone-integrable, thm:continuous-integrable, prop:piecewise-integrable, thm:integrable-algebra(곱·몫·연속 함수와의 합성 포함) (7장) — 증명.
- **Lebesgue 판정법**: note 'Lebesgue's criterion'(7장 475행) — 측도 0의 정의, 가산 집합이 측도 0(note 안 계산), Cantor 집합(이름), 판정법 진술(Abbott·Rudin 인용).
- thm:integral-linear, cor:change-finitely-many, thm:integral-additive, thm:integral-monotone, prop:positive-integral, def:mean-value, thm:mvt-integral, thm:cauchy-schwarz-integral, prop:area-function-lipschitz (7장) — 증명.
- thm:ftc1, cor:antiderivative, thm:ftc2(F 미분 가능 **그리고 f 적분 가능** 가정), cor:variable-limits, prop:arctan-integral (7장) — 증명. 'The Hypotheses of the Second Part' 문단(Volterra; "유계 도함수면 ∫F'=F(b)−F(a)는 Lebesgue에서" — 7장 829행).
- thm:substitution, prop:log-additive, thm:parts, prop:wallis (7장) — 증명. thm:partial-fractions — 7장 줄거리, 16장 sec:rank-nullity-applications에서 완전 증명. thm:liouville(초등 원시함수 없음) — 줄거리.
- 이상 적분: def:improper, prop:p-integrals, lem:monotone-limit-infinity, thm:comparison-test, cor:limit-comparison, def:absolute-convergence, thm:absolute-convergence, prop:sum-integral (7장) — 증명. Cauchy 주값은 문단 정의(7장 1128행, "복소해석 Volume II"). ∫sin x/x 수렴은 문단, 값 π/2는 11장 thm:sinc-integral(증명, 매개변수 미분) + "Volume II가 유수로 다시"(7장 1229행·11장 1458행).
- def:gamma, thm:gamma (7장) — 증명. 구적 오차(def:quadrature-rules … cor:asymptotic-errors), thm:arc-length (7장) — 증명.
- **Riemann–Stieltjes**: note 'Riemann--Stieltjes integrals'(7장 368행)만. **유계 변동**: 없음(12장 def:path-length가 분할 위 상한으로 곡선 길이를 정의하고 thm:path-length 증명 — 개념의 씨앗만).

### A.7 함수열과 함수급수, 고른 수렴, 거듭제곱 급수, Weierstrass 근사
- **일반 고른 수렴**: 정의 없음. note 'Uniform convergence'(8장 824행) — 정의, M-판정(한 줄 이유), "고른 극한은 연속·항별 적분 가능"은 진술, 항별 미분 조건은 문장, "Volume II develops this theory".
- lem:vector-functions (24장) — 증명: 닫힌 유계 구간 위 연속 함수열(벡터값)의 M-판정, 합의 연속성, 꼬리 한계, 항별 적분. 24장 147행이 "Volume II develops the theory of uniform convergence in general"이라고 명시. 이것이 1권의 유일한 일반 함수급수 정리.
- 매개변수 적분 (11장): lem:parameter-continuous, thm:diff-under-integral, cor:leibniz-variable, def:uniform-convergence-integral, prop:weierstrass-test, thm:parameter-improper, prop:gamma-smooth — 증명.
- 거듭제곱 급수 (8장): def:power-series, lem:abel(Abel **보조정리**), def:radius, thm:radius, thm:cauchy-hadamard, lem:derived-radius, thm:termwise-diff, cor:termwise-int, cor:power-series-unique, prop:formal-analytic — 증명(고른 수렴 없이 직접). Abel **정리**(경계 연속): note 'Complex power series'에서 진술. "합은 수렴 구간의 모든 점에서 해석적": thm:taylor-convergence 뒤 문단(8장 1076행) 진술 + Krantz–Parks 인용, "Volume II proves this in the setting of complex analysis".
- def:analytic (26장, 실변수 실·복소값) — 정의. thm:power-series-matrix (22장) — 증명.
- **Weierstrass 근사 정리**: note 'Complements in infinite dimensions and Hilbert spaces'(18장 452행)에서 Rudin Thm 7.26 인용으로 사용만. Bernstein 다항식, Stone–Weierstrass, 등연속, Arzelà–Ascoli(23장 thm:peano 줄거리에 이름만), Dini: 없음.

### A.8 R^n의 미분, 역함수·음함수 정리
- 증명 (10장): def:differentiable-rn, prop:derivative-unique, thm:jacobian, thm:c1-differentiable, prop:directional, thm:steepest-ascent, lem:caratheodory-rn, thm:chain-rule-rn, cor:ck-composition, thm:schwarz, thm:gradient-level, thm:mvt-rn(스칼라값), cor:zero-gradient, thm:mean-value-inequality(벡터값, 부등식 꼴), prop:mvt-integral-rn, lem:linear-remainder-rn, thm:taylor-rn, thm:taylor-multi, thm:first-order, lem:definite-bound, thm:second-derivative-test-rn, prop:definite-2x2, thm:convex-c2, thm:implicit-scalar(**한 방정식**), thm:lagrange-one, thm:lagrange(여러 제약; 증명이 thm:implicit-function을 씀), thm:newton-n, cor:newton-n-local.
- **줄거리**: thm:inverse-function("contraction mapping theorem of Volume II", Rudin 9.24 인용), thm:implicit-function. thm:invertible-matrices·thm:definiteness-criteria(줄거리)는 14·17·20장에서 완결(thm:det-smooth: 역행렬 사상이 C^∞, 증명).
- note 'Derivatives in infinite dimensions'(10장 778행) — Fréchet·Gâteaux 정의, H ↦ −A⁻¹HA⁻¹, "Volume II returns to it".

### A.9 고정점, 축약 사상, 상미분방정식의 존재
- thm:fixed-point (5장, [a,b]→[a,b] 연속이면 고정점) — 증명. prop:fixed-point-contraction (6장; |g'| ≤ q < 1인 미분 가능 꼴) — 증명. 6장 750행·5장 1166행·10장 1379행이 "축약 사상 정리는 Volume II".
- thm:neumann, cor:perturbed-inverse (13장), cor:neumann-spectral (19장), prop:stationary-iteration·thm:map-linearised (27장), thm:power-method (19장) — 증명.
- **일반 축약 사상 정리(거리·Banach 공간)**: 없음.
- 23장: prop:integral-equation — 증명; thm:peano — 줄거리(Euler 다각형 + 등연속 + Arzelà–Ascoli, "Volume II proves the theorem"); def:lipschitz-y, prop:lipschitz-test, lem:gronwall, thm:uniqueness(연속 의존 포함), cor:local-uniqueness — 증명; thm:picard-lindelof — 줄거리(반복의 정의역·오차 한계 귀납은 완전, 극한 통과는 줄거리; "Volume II proves the theorem in full, as an application of the contraction mapping theorem"); thm:maximal-solution — 줄거리; cor:global-existence(최대 해 정리를 가정) — 증명; thm:comparison, lem:discrete-gronwall, thm:euler-convergence — 증명. note 'Funnels'(Kneser, Osgood 인용).
- 24장: thm:linear-existence(선형계, 실·복소, 구간 전체) — 증명; cor:higher-existence — 증명.
- 25장: 비선형 연립계 y'=f(t,y)의 존재·유일성·최대 해는 sec:ode-systems 'Existence and Uniqueness' 끝 문단(25장 108행) 진술 + "Volume II". lem:integral-norm, thm:linearised-stability-systems — 증명; thm:linearised-instability, thm:hartman-grobman, thm:poincare-bendixson, thm:routh-hurwitz — 줄거리.
- 26장: lem:coefficient-growth, thm:ordinary-point, prop:analytic-systems, thm:frobenius — 증명(해석 계수, 실변수); thm:rational-coefficients(반지름 ≥ 가장 가까운 복소 영점까지 거리) — 줄거리.
- 초기값·매개변수에 대한 **미분 가능** 의존성: 없음(연속 의존 eq:continuous-dependence만).

### A.10 Fourier 급수
- 18장: 삼각 정규직교계, Fourier 계수 a_k, b_k, 부분합 S_Nf(sec:orthonormal-sets 끝); thm:bessel(임의 내적 공간의 유한 정규직교족) — 증명 → Σ_{k≤N}1/k² ≤ π²/6; thm:best-approximation(유한 차원 부분공간, 임의 V) — 증명 → S_Nf는 차수 N 삼각 다항식 중 L² 최선 근사; thm:orthonormal-expansion(유한 Parseval) — 증명. note 'Fourier series and Parseval's identity'(18장 397행) — 연속 f의 L² 수렴, 무한 Parseval, π²/6은 진술(Rudin 8장·Folland 8장 인용).
- 점별·고른 수렴, Dirichlet·Fejér 핵, Gibbs 현상, Riemann–Lebesgue 보조정리: 없음. (12·13장의 "Gibbs"는 J. W. Gibbs의 벡터 해석 역사.)
- 이산 Fourier: prop:fourier-basis (18장), thm:circulant (19장) — 증명. note 'The trapezoidal rule for periodic functions'(7장).
- prop:gaussian-fourier (11장, ∫e^{−x²}cos tx = √π e^{−t²/4}) — 증명. Fourier 변환 일반: 없음.
- 경계값·고유함수: prop:boundary-value (24장, y''+λy=g, Dirichlet 조건, 고유값 n²) — 증명; note 'Sturm–Liouville problems'(24장); thm:legendre-orthogonality (26장), thm:periodic-response (25장) — 증명.

### A.11 노름과 거리 개념
- R^n: def:inner-product, prop:inner-product, thm:cauchy-schwarz-vectors, prop:norm (10장); thm:cauchy-schwarz (3장, 유한 합); thm:cauchy-schwarz-integral (7장) — 증명.
- prop:standard-norms (13장, ‖·‖_∞ ≤ ‖·‖_2 ≤ ‖·‖_1 ≤ √n‖·‖_2 ≤ n‖·‖_∞) — 증명. "R^n의 모든 노름은 동치, Volume II"는 그 뒤 문단(13장 243행). note 'The p-norms' — Hölder·Minkowski 진술.
- 행렬 노름: prop:matrix-norms (10장), thm:norm-formulas, prop:operator-norms (13장), thm:two-norm, prop:norm-comparison (21장), prop:spectral-radius-norm, thm:gelfand (19장), prop:norm-near-radius (22장), prop:exp-norm-bounds (22장) — 증명.
- 내적 공간 (18장): thm:cauchy-schwarz-general, cor:inner-product-norm, prop:parallelogram(극화 항등식) — 증명; note 'Norms that come from inner products'(Jordan–von Neumann 진술).
- 무한 차원: note 'Linear operators in infinite dimensions'(16장 384행) — 유계 ⇔ 연속 진술, D의 비유계성 계산, "Volume II develops bounded operators on Banach and Hilbert spaces". thm:riesz (18장, 유한 차원) — 증명, 뒤 문단에서 C[−1,1]의 f ↦ f(1)이 표현되지 않음을 증명. thm:adjoint, prop:adjoint-matrix, prop:adjoint-rules (18장) — 증명; sec:adjoints의 D* = −D는 형식적 계산(18장 1411행 "Volume II develops it"). note 'Complements in infinite dimensions and Hilbert spaces'(18장), note 'The spectral theorem in infinite dimensions'(20장, x곱 연산자에 고유벡터 없음을 note 안에서 보임), note 'Singular values of integral operators'(21장, Schmidt 1907).
- 거리 공간, 완비 노름 공간, ℓ^p, C(K)의 sup 노름 완비성: 없음.

### A.12 (Part II 관련) 복소수와 복소 함수
- 3장: thm:c-field, prop:conjugate, prop:modulus, prop:polar-mult, thm:de-moivre, thm:complex-roots, cor:factor, prop:c-not-ordered — 증명. thm:fta — **줄거리**(d'Alembert–Argand 최솟값 논법; "details are carried out in Volume II, which also gives the shortest proof, by Liouville").
- 9장: lem:complex-series, def:complex-exp, thm:complex-exp-properties, thm:euler-formula, cor:euler-consequences, prop:roots-of-unity, def:atan2·prop:atan2(Arg), prop:complex-log, def:complex-log(Log, 주가지), prop:inverse-trig-log — 증명. 가지·가지 자름은 문단(Log의 불연속, Log(zw) ≠ Log z + Log w, i^i; 9장 1356행 "Volume II"). 복소 거듭제곱의 일반론 없음.
- 8장 note 'Complex power series' — 수렴 원판(실수 증명이 그대로 된다는 진술), 반지름 = 가장 가까운 비정칙점까지 거리("Volume II proves"), Abel 정리. note 'The complex exponential'(8장) — 형식 계산.
- **복소 미분**: 정의 없음. 22장 sec:matrix-functions 첫 문단(837행)이 "복소 증분의 차분몫, Volume II"라고만. lem:complex-calculus (24장, 실변수 복소값 함수의 미적분) — 증명.
- 대수학의 기본 정리 사용처: thm:eigenvalues-exist (19장), cor:complex-partial-fractions·thm:rational-gf (27장, 복소 부분분수로 유리 생성함수와 극; 복소해석 없이) — 증명.
- 12장: lem:green-boundary, thm:green-standard, thm:green(admissible region) — **증명**; cor:green-area — 진술; thm:ftc-line, thm:path-independence, prop:closed-condition, thm:poincare-lemma(별꼴) — 증명; def:simply-connected + thm:simply-connected — **증명**(격자로 쪼개고 볼록 공마다 Poincaré; 뒤 문단 "homotopy invariance는 Volume II"). note 'Holes and topology' — CR 방정식이 두 장 (u,−v), (v,u)의 닫힘이라는 것, Cauchy 정리 ∮f dz = 0, ∮dz/z = 2πi, de Rham 코호몰로지(언급, 2권 약속 아님).
- 감김 수: def:winding-number(닫힌 **다각형**, 꼭짓각의 합), prop:winding(정수, 각도 장의 선적분), thm:point-in-polygon, prop:shoelace-winding — 증명; thm:jordan-polygon(다각형 Jordan 곡선 정리), thm:triangulation — 줄거리(Hales 등).
- 조화 함수: def:laplacian (12장) — 정의, e^x cos y 문단(12장 1061행 "Volume II shows that real and imaginary parts are harmonic"); cor:green-identities, prop:dirichlet-unique(3차원 standard solid) — 증명; note 'The mean value property of harmonic functions'(3차원, note 안 계산, 최대 원리 언급, "Volume II … two variables").
- 편각 원리·Rouché는 인용만: thm:gershgorin-separated (19장, 근이 계수에 연속) — 줄거리; thm:schur-cohn (27장) — 줄거리.
- 행렬 함수: def:matrix-function(Hermite 보간), thm:function-polynomial, thm:function-properties, prop:spectral-projections-general, thm:function-composition (22장) — 증명; note 'The Cauchy integral and the holomorphic functional calculus'(22장 973행) — 레졸벤트 적분 정의 진술, Higham 인용.
- 감마: thm:gamma (7장), prop:gamma-smooth·thm:beta-gamma·cor:gaussian-consequences (11장) — 증명; def:gamma-extension (26장, 음의 비정수 실수로 확장) — 정의; note 'The gamma function and the factorials'(9장 469행) — Bohr–Mollerup, Gauss 공식, 반사 공식 진술. 26장 813행 "반사 공식은 Volume II에서 증명".

### A.13 (Part III 관련) 내용, 측도 0, 다중 적분
- 측도 0: note 'Lebesgue's criterion'(7장) 안의 정의와 계산만.
- 11장 (모두 Riemann 적분): def:rectangle-partition, def:double-integral, lem:refinement-2d, thm:riemann-criterion-2d, thm:riemann-sums-2d, thm:continuous-integrable-2d — 증명; thm:double-properties — 진술(증명 환경 없음).
- def:content-zero, prop:content-zero-examples, thm:integrable-content-zero, cor:change-content-zero — 증명; 유리점 집합은 내용 0이 아님(400행 문단, 증명 + "Lebesgue measure of Volume II, which allows countable covers").
- def:integral-over-set, def:jordan-measurable(경계가 내용 0), thm:integral-over-region, def:elementary-region, thm:elementary-regions — 증명. "Jordan 가측 ⇔ 1_S 적분 가능"은 def:jordan-measurable 뒤 문단에서 한쪽만.
- thm:fubini(직사각형, 내부 적분 존재 가정; 없으면 상·하적분), cor:product-integral, thm:multiple-summary(n차원 확장, "2차원 증명 그대로" + 그래프의 내용 0), thm:fubini-n, cor:triple-elementary — 증명. note 'The Lebesgue integral and the Fubini--Tonelli theorem' — Tonelli·Fubini 진술(Folland 인용).
- prop:scaling, thm:cavalieri, prop:revolution, prop:simplex, lem:lipschitz-content, thm:linear-volume(|det A| 배율) — 증명; thm:det-volume (17장) — 증명.
- thm:change-variables — **줄거리**(상의 Jordan 가측성과 국소 부피 추정은 쓰고 나머지는 Spivak·Munkres·Lax). 17장 966행 "complete proof is given in Volume II". eq:polar-integral, 구면 좌표 Jacobian — 이 정리로.
- 이상 다중 적분: def:exhaustion, lem:exhaustion-compact, thm:exhaustion, def:improper-multiple, thm:absolute-multiple, prop:radial, cor:power-integrals-rn — 증명.
- thm:gaussian, cor:gaussian-consequences, prop:gaussian-n, thm:ball-volume (11장), prop:sphere-area (12장), thm:gaussian-general (20장; 11장 thm:gaussian-quadratic의 완결) — 증명.
- 곡면 적분·발산·Stokes (12장): def:parametrised-surface … thm:divergence, thm:stokes — 증명(standard solid·standard region 범위).
- σ-대수, 외측도, Lebesgue 측도, 가측 함수, Lebesgue 적분, 수렴 정리, L^p: **없음**.

---

## B. 2권 앞으로 온 약속

PROMISES.md의 "- 2권:" 줄 11개(101·115·121·126·132·140·145·147·149·153·156행)를 항목으로 쪼개 **현재 OUTLINE.md 번호**의 장에 다시 묶었다. ⚑ = 현재 목차의 어느 장에도 맞지 않음, △ = 맞는 장은 있으나 그 장의 줄에 없는 내용.

### 현재 목차의 장별
- **1 The Real Numbers**: (본문 약속, PROMISES.md에 없음) Cauchy 수열로 완비성의 동치 꼴(3장 1655행, 5장 614행); 27장 맺음말 "Its real analysis proves the completeness of the real numbers"(27장 1732행).
- **2 Metric Spaces**: R^n의 모든 노름의 동치(13장 prop:standard-norms 뒤; 101행) — 증명에 컴팩트성이 필요하므로 실제로는 3장 △; p-노름과 Hölder·Minkowski(13장 note 'The p-norms'; 101행) △; Young에서 Hölder(9장 cor:young 뒤; 101행) △; 열린 집합(5장; 101행); (본문) 거리 공간 사이 연속 ⇔ 열린 집합의 역상(2장 682행, 5장 996행); (본문) 거리 공간의 완비화(3장 1696행) △; 노름 공간의 유계 선형 사상·연산자 노름·미분 연산자의 비유계성(16장 note; 101행) △.
- **3 Continuity and Compactness**: 컴팩트성과 Heine–Borel(10장 thm:evt-rn 뒤; 101행); 열린 집합과 컴팩트(5장; 101행); 연결 집합(10장 cor:zero-gradient 뒤; 101행); 가역 실행렬의 두 경로 성분(17장 note; 101행) △; (본문) 컴팩트 집합 위 EVT(5장 1270행).
- **4 Differentiation**: 해당 약속 없음.
- **5 The Riemann Integral**: 측도 0과 Lebesgue 판정법(7장; 101행) △(5장 줄에 "적분 가능성"만); Jordan 가측 ⇔ 1_S 적분 가능(11장 def:jordan-measurable 뒤; 101행) △(R^n이라 1차원 장 밖).
- **6 Sequences and Series of Functions**: 함수열의 고른 수렴, M-판정, 항별 미분·적분의 일반 정리(8장; 101행); 고른 수렴의 일반 이론(24장 lem:vector-functions; 147행); Weierstrass 함수의 연속성(6장; 101행); Abel 정리(8장 note; 101행) △; 매끄러운 bump 함수(8장 prop:flat-function 뒤; 101행) △.
- **7 Differentiation in R^n**: 역함수·음함수 정리의 완전한 증명(10장 thm:inverse-function·thm:implicit-function; 101행); 노름 공간의 Fréchet·Gâteaux 미분(10장 note; 101행) △(장 줄은 R^n만).
- **8 Fixed Points and Existence Theorems**: R^n과 Banach 공간의 축약 사상 정리(10장; 101행) — 7장의 역함수 정리 증명이 먼저 필요로 하므로 순서 문제; 축약 사상(5장; 101행); 6장 prop:fixed-point-contraction을 1차원 경우로 인용(101행); 27장 thm:map-linearised가 쓰는 축약 사상 정리(156행); Picard–Lindelöf 존재와 유일성(101행), 완전한 증명(23장 thm:picard-lindelof; 145행); 최대 해와 폭발 양자택일(23장 thm:maximal-solution; 145행) △; 비선형 연립계 y'=f(y)의 존재·유일성·최대 해(25장; 149행); Peano 존재 정리의 증명(Euler 다각형, Arzelà–Ascoli; 23장 thm:peano; 145행) — Arzelà–Ascoli를 둘 장이 없음 ⚑.
- **9 Fourier Series**: Fourier 급수로 Σ1/n² = π²/6(5장; 101행); Fourier 급수의 L² 수렴과 무한 Parseval(18장 note; 115행); Fourier 급수 같은 무한 결합 전개(L²; 15장 note; 101행) — L²는 Part III라 9장과 20장에 걸침.
- **10 Holomorphic Functions**: 복소 미분(22장 sec:matrix-functions; 140행); 복소 로그의 가지와 가지 자름(9장; 101행); 복소 거듭제곱(9장; 101행); Cauchy–Riemann 방정식과 조화 함수(12장 sec:div-curl; 101행); 해석 함수 이론(8장; 101행; Part II 전체); (본문) principal values(2장 918행).
- **11 Complex Integration**: Green(thm:green)과 Poincaré(thm:poincare-lemma)로 Cauchy 정리(12장; 101행); 닫힌 장의 선적분의 호모토피 불변성 일반(12장 thm:simply-connected·note 'Holes and topology'; 101행); 감김 수(12장; 101행; 편각 원리는 13장); 대수학의 기본 정리의 증명(3장 thm:fta; 101행; d'Alembert–Argand 세부와 Liouville 증명 둘 다 약속); 일반 연속 곡선의 Jordan 곡선 정리(12장 thm:jordan-polygon; 101행) ⚑(어느 장의 줄에도 없음, 표준 복소해석 과목도 대개 증명하지 않음); Cauchy 적분으로 정의한 f(A)(22장 note; 140행) △.
- **12 Series Representations**: 복소 거듭제곱 급수의 수렴 반지름 = 가장 가까운 특이점까지 거리(8장 note; 101행); 거듭제곱 급수의 합이 수렴 구간의 모든 점에서 해석적(8장; 101행); 해석적 연속(9장; 101행); 복소 영역에서 급수 해의 반지름 ≥ 가장 가까운 복소 특이점까지 거리, 해석 계수 일반(26장 thm:rational-coefficients; 153행) △(정칙 계수 ODE가 장 줄에 없음); Fuchs 이론·monodromy·Riemann P-기호(26장 note; 153행) ⚑.
- **13 The Residue Calculus**: ∫_0^∞ sin x/x = π/2의 유수 정리 증명(11장·7장; 101행 두 번); Cauchy 주값(7장; 101행); 편각 원리(12장; 101행); 다항식의 근이 계수에 연속(편각 원리/Rouché; 19장 thm:gershgorin-separated; 121행); Rouché로 Schur–Cohn 판정(27장 thm:schur-cohn; 156행).
- **14 Conformal Maps and Harmonic Functions**: 조화 함수의 평균값 성질과 최대 원리(12장 note; 101행); CR과 조화 함수(12장; 101행, 10장과 공유).
- **15 Entire and Meromorphic Functions**: 감마 함수의 반사 공식과 Gauss 공식(9장 note; 101행); 감마 함수의 복소 확장과 반사 공식(26장; 153행).
- **16 Measures**: 측도 0과 가산 덮개(11장, 유리점 집합의 내용; 101행); Riemann–Stieltjes를 측도 적분으로(7장 note; 101행) △(Lebesgue–Stieltjes 측도가 장 줄에 없음).
- **17 Measurable Functions and the Lebesgue Integral**: Lebesgue 적분과 Dirichlet 함수의 Lebesgue 적분 0(7장; 101행); Volterra 함수와 Lebesgue 적분의 동기(6장; 101행); Jordan 가측 ⇔ 1_S 적분 가능(11장)은 여기서 "Riemann 적분과의 관계"로도 받을 수 있음.
- **18 Convergence Theorems**: 유계 도함수에 대한 ∫F' = F(b) − F(a)(7장; 101행) — 지배 수렴으로 증명되므로 18장, 또는 21장.
- **19 Product Measures**: Lebesgue 적분의 Fubini–Tonelli(11장 note; 101행).
- **20 L^p Spaces**: L² 공간(7장; 101행); Young → Hölder와 L^p(9장; 101행); 거의 어디서나 같은 함수의 몫(15장 note; 101행); Hilbert 공간과 닫힌 부분공간의 직교 분해(18장 note; 115행) △(장 줄은 "L²와 직교 사영"뿐, 추상 Hilbert 공간 아님); 무한 차원 Riesz 표현(18장 thm:riesz 뒤; 115행) △.
- **21 Signed Measures and Differentiation**: 변수 변환 공식의 완전한 증명(11장 thm:change-variables, 17장 thm:det-volume 뒤 문단; 101행 두 번) — 21장 줄의 "변수 변환"이 이것을 뜻한다면 해당, 다만 Lebesgue 측도의 C^1 변환은 19장(곱측도·R^n) 쪽이 자연스러움; ∫F' = F(b)−F(a)(18장과 공유).

### 어느 장에도 맞지 않는 약속 (⚑)
- Hamel 기저(15장 note; 101행)와 연속이 아닌 가법 함수(9장; 101행) — 선택 공리의 응용, 해석학 장 줄 어디에도 없음.
- Hilbert 공간의 Fredholm 대안(15장 cor:fredholm-square 뒤 −u''=f 예; 101행).
- Baire(5장; 101행)와 Banach의 범주 정리(6장, 어디서도 미분 불가능한 함수가 전형적; 101행); (본문) 유리수에서만 연속인 함수는 없다(5장 992행).
- Arzelà–Ascoli(23장 thm:peano; 145행).
- 유계 연산자의 Banach·Hilbert 공간 이론(16장 note; 101행).
- 경계 조건이 있는 미분 연산자의 수반 D* = −D, −D²의 자기 수반성(18장 sec:adjoints; 115행).
- 컴팩트 자기 수반 연산자의 스펙트럼 정리와 Sturm–Liouville 연산자의 고유함수 기저(20장 note; 126행), Sturm–Liouville 고유함수 전개(24장 note 147행, 26장 153행 — 같은 약속 세 번; 26장 1148행 Bessel 고유함수 전개 문단도).
- 컴팩트 연산자의 특이값 전개(Schmidt)와 Hilbert–Schmidt 연산자(21장 note; 132행).
- 해석 함수 연산(holomorphic functional calculus)과 작용소의 스펙트럼 이론(22장 note; 140행).
- 일반 연속 곡선의 Jordan 곡선 정리(12장; 101행).
- Fuchs 이론·monodromy·Riemann P-기호(26장 note; 153행).
- 매끄러운 bump 함수(8장), Abel 정리(8장), Fréchet·Gâteaux(10장), 노름 공간의 유계 연산자(16장), 완비화(3장), GL_n의 두 성분(17장)은 △로 표시한 대로 장은 있으나 장 줄에 없음.
- 약속이 아닌 것: Abel–Ruffini·Galois(19장 note 'Solving equations by radicals', PROMISES.md 121행이 "권 배정 없음"이라 적음), de Rham 코호몰로지(12장 note, 2권 약속 없음), Favard 정리의 측도 부분(27장 1312행은 "requires the measure theory of Volume II, and Chihara gives the proof" — 증명 약속 아님), Kneser·Osgood(23장 note, Hartman 인용).

### OUTLINE.md "함수해석, 초함수는 다루지 않는다"와의 충돌
- **충돌한다.** 컴팩트 자기 수반 연산자의 스펙트럼 정리, Schmidt 전개·Hilbert–Schmidt 연산자, Hilbert 공간의 Fredholm 대안, 유계 연산자의 Banach·Hilbert 공간 이론, 작용소의 스펙트럼 이론·해석 함수 연산, (약하게) Banach 공간의 축약 사상 정리와 Fréchet 미분은 함수해석의 첫 과목 내용이다.
- 충돌하지 **않는** 것: Baire, Arzelà–Ascoli, Stone–Weierstrass, L²의 직교 사영, L^p의 완비성, 무한 차원 Riesz 표현(Hilbert 공간)은 표준 실해석 과목(Rudin PMA 7장, Folland 4–5장, Stein–Shakarchi III 4장)의 범위다.
- 초함수에 대한 약속은 없다.
- 결론(D3): 금지문을 "함수해석 **일반**(Hahn–Banach, 열린 사상·균등 유계성 원리, 약위상, 비유계 연산자의 스펙트럼 정리)과 초함수는 다루지 않는다"로 좁히고, Hilbert 공간과 컴팩트 연산자 한 장을 둔다(C 22장). 작용소의 스펙트럼 이론 일반·해석 함수 연산의 작용소 판은 약속을 고친다(D4).

### PROMISES.md에 없는 본문 약속 (관리자가 PROMISES.md에 더할 것)
3장 1655행·5장 614행(Cauchy 수열로 완비성), 3장 1696행(거리 공간의 완비화), 2장 682행·5장 996행(거리 공간에서 연속 ⇔ 역상), 2장 918행(복소 로그 등의 주값), 5장 992행(Baire: 유리수에서만 연속인 함수 없음), 5장 1270행·10장 479행(컴팩트 집합 위 EVT), 5장 1395행(Lipschitz 조건이 축약 사상·존재 정리의 가정으로), 8장 198행(복소 급수 심화), 10장 51행(함수의 내적), 12장 1205행(2변수 조화 함수), 15장 96행(무한 차원 공간을 그 자체로), 26장 766행(Γ의 복소 확장, 153행과 같음), 26장 1148행(Bessel 고유함수 전개 = 스펙트럼 이론), 27장 1732행(2권 내용 요약: 실수의 완비성 증명, 함수열과 고른 수렴, 축약 사상 + 존재·유일성, Cauchy·Laurent·유수·편각·Rouché, Lebesgue 적분과 L^p).

---

## C. 권장 장 구성 (23장)

원칙: 1권이 증명한 것은 "Volume I, Theorem x.y"로 인용하고 다시 증명하지 않는다(다만 일반화의 특수 경우로 한 줄 확인은 한다). 각 장은 1권이 증명하지 않은 것에서 출발해 표준 학부 과목의 범위까지 간다. 세 부(Real Analysis 9장, Complex Analysis 6장, Measure Theory 8장)를 유지한다.

### Part I Real Analysis

**1. The Real Numbers** (이름 유지, 범위 바꿈)
1권은 완비성을 공리로 두고(asm:reals) 그 결과(상한, Archimedes, 단조 수렴, BW, Cauchy, 축소 구간)를 모두 증명했으므로, 이 장은 공리를 정리로 바꾸는 일을 한다: Q에서 Cauchy 수열의 동치류로 R을 구성하고 완비 순서체임을 증명(완전), Dedekind 절단 구성은 줄거리, 완비 순서체의 유일성(1권 thm:r-unique 줄거리 → 완전), 순서체에서 완비성의 다섯 꼴의 동치(1권 5장 note의 원을 닫음; 비Archimedes 반례는 note), 확장 실수와 limsup·liminf의 성질(1권 def:limsup 인용), 연속체의 기수(2장 인용)와 대수적·초월수 note. 1권 3·5장의 결과는 인용만. 약속: 3장 1655행·27장 1732행. Cauchy 수열 구성을 택하는 것은 2장의 거리 공간 완비화가 같은 구성이기 때문.
중심 정리: 완비 순서체의 존재(Cantor 구성), 유일성(순서와 연산을 보존하는 유일한 동형), 완비성의 동치 정리.

**2. Metric and Normed Spaces** (이름 바꿈: Metric Spaces → Metric and Normed Spaces)
거리·노름의 정의와 예(R^n의 p-노름, ℓ^p, 유계 함수의 sup 노름, C[a,b]의 ‖·‖_∞·‖·‖_1·‖·‖_2), 유한 합의 Hölder·Minkowski(9·13장 약속; 1권 cor:young 인용), 열린·닫힌 집합·내부·폐포·경계·조밀(1권 def:open-closed의 일반화), 수렴과 연속, 연속 ⇔ 열린 집합의 역상(2·5장 약속), 균등 연속·Lipschitz·등거리, Cauchy 수열과 완비성, R^n·ℓ^p·C_b(X)의 완비성, 거리 공간의 완비화(3장 약속; 1장과 같은 구성), **Banach 고정점 정리**(5·6·10·27장 약속; 1권 prop:fixed-point-contraction과 thm:map-linearised는 특수 경우로 한 줄), 노름 공간 사이 선형 사상의 유계 ⇔ 연속, 연산자 노름, B(X,Y)의 완비성, Banach 공간의 Neumann 급수(1권 thm:neumann의 일반화), D의 비유계성(16장 약속). 축약 사상 정리를 현재 8장에서 여기로 옮긴다: 7장의 역함수 정리가 먼저 쓰기 때문(현재 목차는 7장이 8장의 정리를 앞서 씀).
중심 정리: 완비화 정리, Banach 고정점 정리(오차 한계 포함), ℓ^p와 C_b(X)의 완비성, 유계 ⇔ 연속.

**3. Compactness and Connectedness** (이름 바꿈: Continuity and Compactness)
연속의 기초는 1권(5·10장)과 2장으로 끝났으므로 이 장은 컴팩트와 연결에 집중: 열린 덮개 컴팩트, 거리 공간에서 컴팩트 ⇔ 순차 컴팩트 ⇔ 완비 + 완전 유계, Heine–Borel(10장 약속), 컴팩트 위의 연속 함수(EVT·고른 연속의 일반판; 1권 thm:evt·thm:heine-cantor·lem:heine-cantor-rn은 인용), Lebesgue 수, **유한 차원 노름 공간의 모든 노름은 동치**(13장 약속; 현재 목차의 2장에서 옮김 — 단위구의 컴팩트성이 필요), 유한 차원 부분공간은 닫힘, Riesz 보조정리(무한 차원 단위 공은 컴팩트가 아님; 1권 20장 note의 "단위구가 컴팩트가 아니다"를 정리로), 연결·경로 연결·성분, R^n의 연결 열린 집합은 꺾은선 연결(1권 def:connected-open과 일치시킴; 10장 약속), 연결 집합의 연속상(1권 cor:interval-image 일반화), GL_n(R)의 두 경로 성분(17장 note 약속 — 1권 줄거리를 완결), Cantor 집합(완전, 비가산, 측도 0의 예고), **Baire 범주 정리**와 응용(유리수에서만 연속인 함수 없음 — 5장 약속; 범주 개념). 불연속 가법 함수와 Hamel 기저는 note(D5).
중심 정리: Heine–Borel, 거리 공간 컴팩트의 세 동치, 노름 동치, Baire 범주 정리.

**4. Uniform Convergence** (현재 6장을 둘로 나눈 앞 장)
점별·고른 수렴, 고른 Cauchy 판정, 극한 교환 정리 셋(연속성, Riemann 적분, 미분 — 1권 8장 note가 진술만 한 것), 일반 M-판정(1권 lem:vector-functions는 특수 경우로 인용), Dini 정리, 극한 교환의 반례, 이중 급수와 이중 수열의 교환(절대 수렴), 거듭제곱 급수의 재정리: 수렴 구간의 모든 점에서 해석적(8장 약속, 실변수 증명 — 복소판은 12장), **Abel 정리**(8장 약속) 와 Tauber 정리 note, Weierstrass 함수의 연속성(6장 약속)과 어디서도 미분 불가능한 연속 함수의 완전한 증명(Takagi 함수 또는 van der Waerden 꼴; Weierstrass 원래 함수는 줄거리), 매개변수 적분은 1권 11장(thm:parameter-improper) 인용. 약속: 8장(일반 고른 수렴·M-판정·항별 미분·적분), 24장 147행, 6장.
중심 정리: 고른 극한의 연속성·적분·미분 교환 정리, Abel 정리, 연속이지만 어디서도 미분 불가능한 함수의 존재.

**5. Approximation and Equicontinuity** (새 장, 현재 6장의 Weierstrass 근사를 옮겨 넓힘)
Weierstrass 근사 정리(Bernstein 다항식, 연속도 ω(f,δ)로 정량 한계 — 11권이 쓴다), Stone–Weierstrass(실·복소; 삼각 다항식이 연속 주기 함수에서 조밀 — 9장이 씀), 1권 18장 note가 인용한 Weierstrass 정리를 여기서 증명, 등연속, **Arzelà–Ascoli**(23장 약속; Peano와 22장 컴팩트 적분 연산자가 씀), C[0,1]에서 Baire: 어디서도 미분 불가능한 함수가 전형적(Banach 1931; 6장 약속), 매끄러운 bump 함수와 잘라내기 함수(8장 약속; 20장 mollifier가 씀), Chebyshev 최소최대 근사는 note로 10·11권 예고.
중심 정리: Weierstrass·Stone–Weierstrass, Arzelà–Ascoli, Banach의 범주 정리.

**6. Monotone Functions, Bounded Variation, and the Riemann–Stieltjes Integral** (현재 4장 Differentiation과 5장 The Riemann Integral을 대신함)
1권 6·7·8장이 미분과 Darboux 적분을 이미 완전히 증명했으므로 두 장을 다시 쓰지 않고, 1권이 note로 남긴 것을 채운다: 측도 0과 **Lebesgue 판정법**(7장 약속; 진동 함수로 증명), R^n의 Jordan 내용과 "Jordan 가측 ⇔ 1_S 적분 가능"(11장 약속), 단조 함수(1권 prop:monotone-countable 인용)와 볼록 함수의 연속성·한쪽 도함수(1권에 없음; 20장 Jensen이 씀), 유계 변동 함수와 Jordan 분해, 연속점, 곡선의 길이 = BV(1권 def:path-length 인용), Riemann–Stieltjes 적분(1권 7장 note를 완전히: 존재, 부분적분, C^1 적분자, 계단 적분자 = 가중합), 적분의 둘째 평균값 정리(Bonnet; 9장 Dirichlet 판정과 ∫sin x/x의 수렴에 씀), Volterra 함수(진술 + 줄거리, 6장 약속의 "Lebesgue 적분의 동기"), 양의 선형 범함수 = RS 적분(Riesz, note; 16장과 1권 27장 Favard로 이어짐). 1권의 미분 정리(MVT, Taylor, l'Hôpital, Darboux, 볼록·Jensen)와 적분 정리(FTC, 치환, 이상 적분)는 인용만.
중심 정리: Lebesgue 판정법, BV의 Jordan 분해, 연속 f·BV α의 RS 적분 존재와 부분적분.

**7. Differential Calculus in Normed Spaces** (이름 바꿈: Differentiation in R^n)
1권 10장이 R^n의 미분(정의, 연쇄 법칙, 평균값 부등식, Taylor, 극값, Lagrange)을 모두 증명했으므로, 이 장은 Banach 공간으로 올려 같은 증명을 한 번에 하고(1권 증명은 그대로 읽힌다는 확인만) 1권이 미룬 것을 증명한다: Fréchet 미분과 Gâteaux 미분(10장 note 약속; 정의·예·관계), Banach 공간 연쇄 법칙, 평균값 부등식, 역사상 A ↦ A⁻¹의 미분(1권 note의 H ↦ −A⁻¹HA⁻¹), 고계 미분 = 다중선형 사상 꼴(간단히), **역함수 정리**(Banach 공간, 2장의 고정점 정리로; 10장 약속), **음함수 정리**(10장 약속), 그 결과로 1권 thm:lagrange의 증명이 완결됨을 한 문단, 계수 정리와 R^n의 매끄러운 곡면(줄거리), 단순 고윳값의 매끄러운 의존(음함수 정리의 응용, 8권 예고), Newton–Kantorovich(진술 + 줄거리, 12권 예고).
중심 정리: Banach 공간의 평균값 부등식, 역함수 정리, 음함수 정리.

**8. Fixed Points and Differential Equations** (이름 바꿈: Fixed Points and Existence Theorems)
축약 사상 정리는 2장에 있으므로 이 장은 그 응용과 ODE 이론: **Picard–Lindelöf**(R^n 값, 완전; Bielecki 노름 또는 반복; 23·25장 약속 — 1권 thm:picard-lindelof의 극한 통과 완결), 선형계의 대역 존재(1권 thm:linear-existence를 특수 경우로 인용), **최대 해와 폭발 양자택일**(23·25장 약속), **Peano**(Euler 다각형 + 5장 Arzelà–Ascoli; 23장 약속), 비유일성과 Kneser(note), 초기값·매개변수에 대한 연속 의존(1권 thm:uniqueness 인용)과 **미분 가능 의존**(변분 방정식; 1권에 없음, 25장 선형화의 근거), 자율계의 흐름과 군 성질, 반복법의 수렴(고정점 반복·Newton; 1권 thm:newton-n 인용). Brouwer 고정점 정리는 진술 + 참고(note; 1권 thm:fixed-point가 1차원).
중심 정리: Picard–Lindelöf, Peano, 최대 해 정리, 초기값에 대한 미분 가능 의존.

**9. Fourier Series** (이름 유지)
Riemann 적분 수준에서(Part III 전에) 1권 18장의 유한 이론(Bessel, 최선 근사, π²/6 ≤ 반쪽)을 무한으로 완성: 복소 지수 꼴과 실수 꼴, Dirichlet 핵, Riemann–Lebesgue 보조정리(Riemann 적분 가능), 국소화, Dirichlet–Jordan 판정(구간별 C^1과 BV; 6장 사용), Fejér 핵과 Fejér 정리, 삼각 다항식의 조밀성(5장 Stone–Weierstrass와 대조), Riemann 적분 가능 함수의 평균 제곱 수렴과 **Parseval**(15·18장 약속), Σ1/n² = π²/6(5장 약속)과 ζ(2k), 계수의 감소와 매끄러움, 항별 미분·적분, 고른 수렴 조건, **Gibbs 현상**(계산), 연속 함수의 Fourier 급수가 발산할 수 있음(du Bois-Reymond; Baire로 존재 증명 또는 note), Poisson 핵과 Abel 합(14장 예고), Weyl 등분포(11권 준-Monte Carlo 예고, 선택), 이산 Fourier와 계수의 에일리어싱(1권 prop:fourier-basis 인용, 11권 예고). L²의 완비성(Riesz–Fischer)과 정규직교 기저로서의 완비성은 22장에서 닫는다.
중심 정리: Dirichlet 정리, Fejér 정리, Riemann 적분 가능 함수의 Parseval 등식.

### Part II Complex Analysis

**10. Holomorphic Functions** (이름 유지)
복소 미분(22장 약속), Cauchy–Riemann 방정식과 실미분 가능성의 관계(완전; 1권은 언급만), 조화 성분(12장 약속, C² 가정은 11장에서 제거), 복소 거듭제곱 급수가 정칙이고 항별 미분됨(1권 8장 note가 "같은 증명"이라고만 했으므로 복소판을 다시 증명 — 짧음), 지수·삼각·로그·거듭제곱(1권 9장 정의 인용) 위에 가지·가지 자름·정칙 가지의 존재(9장 약속; 단순 연결 영역의 로그는 11장 뒤 절로), 복소 거듭제곱 z^α와 그 법칙의 실패, 등각성(f' ≠ 0), Riemann 구.
중심 정리: CR 판정, 거듭제곱 급수의 정칙성, 정칙 로그 가지의 존재(원시함수와 함께).

**11. Complex Integration** (이름 유지)
경로 적분(1권 12장 선적분 인용), 원시함수와 경로 독립(1권 thm:path-independence 인용), Goursat 정리(C^1 가정 제거), Cauchy 정리: 별꼴·단순 연결 영역에서는 1권 thm:poincare-lemma·thm:simply-connected를 CR과 합쳐 바로 얻고(12장 약속), Goursat로 일반 정칙 함수까지, 호모토피 판과 일반 닫힌 곡선의 감김 수(12장 약속; 1권 다각형 감김 수 def:winding-number와 일치 확인), Cauchy 적분 공식(호몰로지 판은 Dixon 증명), 정칙 ⇒ 해석·C^∞, Cauchy 부등식, **Liouville**과 **대수학의 기본 정리**(3장 약속: Liouville 증명 + 1권 d'Alembert–Argand 줄거리의 세부를 3장 컴팩트로 완결), Morera, 최대 절댓값 원리, Jordan 곡선 정리는 진술 + 줄거리 + 참고(D6).
중심 정리: Cauchy–Goursat, Cauchy 적분 공식, Liouville(과 FTA), 최대 절댓값 원리.

**12. Series Representations and Analytic Continuation** (이름 바꿈: Series Representations)
Taylor 급수가 가장 큰 원판에서 수렴(반지름 = 가장 가까운 특이점까지 거리; 8·26장 약속), 실변수 해석 함수와의 관계(8장 약속 복소판), 영점의 고립·항등 정리, Laurent 급수, 고립 특이점의 분류와 Casorati–Weierstrass, 열린 사상 정리, 경로를 따른 해석적 연속과 **monodromy 정리**(9장 약속, 26장 약속의 "도구"), 정칙 계수 선형 ODE의 원판 위 정칙 해 존재와 반지름 ≥ 특이점까지 거리(26장 thm:rational-coefficients 완결, 해석 계수 일반; 1권 lem:coefficient-growth 인용), 정칙 특이점 둘레의 해의 연속과 monodromy 행렬, Fuchs 방정식과 Riemann P-기호는 note + Whittaker–Watson(D4).
중심 정리: Taylor·Laurent 전개, 항등 정리, monodromy 정리, 정칙 계수 ODE의 해의 반지름.

**13. The Residue Calculus** (이름 유지, 응용 추가)
유수 정리, 실적분 계산(유리 함수, 삼각, Jordan 보조정리로 Fourier 형; ∫_0^∞ sin x/x = π/2 — 7·11장 약속), Cauchy 주값(7장 약속), 급수의 합(π cot πz; Σ1/n² 다시), 편각 원리, **Rouché**, Hurwitz 정리, 다항식의 근이 계수에 연속(19장 약속 → 1권 thm:gershgorin-separated 완결), Schur–Cohn 판정의 완결(27장 약속), (선택) Routh–Hurwitz 완결(1권 thm:routh-hurwitz 줄거리; 약속 없음), 행렬의 해석 함수 연산 f(A) = (1/2πi)∮f(z)(zI−A)⁻¹dz가 1권 def:matrix-function과 같음(22장 약속의 행렬 부분; 레졸벤트의 극과 지표).
중심 정리: 유수 정리, 편각 원리, Rouché, 레졸벤트 적분과 행렬 함수의 일치.

**14. Conformal Maps and Harmonic Functions** (이름 유지)
등각 사상, Möbius 변환과 원판·반평면의 자기동형, Schwarz 보조정리, Riemann 사상 정리(진술 + 줄거리, Montel), 평면 조화 함수: 평균값 성질과 최대 원리(12장 약속; 1권 note는 3차원), 단순 연결 영역의 조화 켤레(12장 약속), Poisson 적분과 원판의 Dirichlet 문제(완전), 9장 Fourier 급수의 Abel–Poisson 합과의 연결, 평균값 성질 ⇒ 조화(역).
중심 정리: Schwarz 보조정리, Poisson 공식이 Dirichlet 문제를 푼다, 평균값 성질의 특성화.

**15. Entire and Meromorphic Functions** (이름 유지, zeta 추가)
무한곱, Weierstrass 인수분해, Mittag-Leffler(π²/sin²πz, π cot πz), sin의 곱 공식(Euler)과 Wallis(1권 prop:wallis 인용), 감마 함수: Gauss·Weierstrass 곱(9장 약속), C \ {0, −1, …}로의 해석적 연속(1권 def:gamma-extension 인용), **반사 공식**(9·26장 약속), 배가 공식, 복소 베타, Bohr–Mollerup(1권 note가 Artin 인용; 넣을지 D9), **Riemann zeta**: Euler 곱, 해석적 연속(η 또는 Γ·Hankel), 함수 방정식(진술 + 줄거리), ζ(2k)와 Bernoulli 수(1권 8장 note 'The Euler--Maclaurin formula and Bernoulli numbers' 연결). Stirling 급수와 점근 전개는 10권(OUTLINE 결정).
중심 정리: Weierstrass 곱 정리, 반사 공식, Γ와 ζ의 해석적 연속.

### Part III Measure Theory

**16. Measures** (이름 유지, Lebesgue–Stieltjes 추가)
σ-대수, Borel 집합, 측도, π–λ(Dynkin) 정리와 유일성, 외측도와 Carathéodory 구성, Lebesgue 측도(R과 R^n), **Lebesgue–Stieltjes 측도** ↔ 오른쪽 연속 증가 함수(7장 약속 "Riemann–Stieltjes를 측도 적분으로"; 3권의 분포함수), 정칙성, 평행이동 불변성, 선형 사상의 |det| 배율(1권 thm:linear-volume의 Lebesgue판), 측도 0과 가산 덮개(11장 약속), Jordan 내용과 Lebesgue 측도의 관계, Cantor 집합, 비가측 집합(Vitali; 1권 2장의 선택 공리 note).
중심 정리: Carathéodory 확장, Lebesgue(–Stieltjes) 측도의 존재와 유일성, π–λ 정리.

**17. Measurable Functions and the Lebesgue Integral** (이름 유지)
가측 함수, 단순 함수 근사, 음 아닌 함수의 적분, 적분 가능 함수, 거의 어디서나, Riemann 적분 가능 ⇒ Lebesgue 적분 가능·같은 값(6장 Lebesgue 판정법 인용), Dirichlet 함수의 적분 0(7장 약속), 이상 Riemann 적분과의 관계(sin x/x는 Lebesgue 적분 가능하지 않음), 상(像) 측도와 추상 변수 변환 ∫g d(f_*μ) = ∫g∘f dμ(3권), Volterra 함수의 도함수가 Lebesgue 적분 가능(6장 약속; 18장과 함께).
중심 정리: 단순 함수 근사, Riemann 적분과 Lebesgue 적분의 일치.

**18. Convergence Theorems** (이름 유지)
단조 수렴, Fatou, 지배 수렴, 함수 급수의 항별 적분, 매개변수 적분의 연속·미분(1권 11장 정리들의 일반판), 수렴의 양식(거의 어디서나, 측도 수렴, L^1; Egorov, Riesz 부분수열), 고른 적분 가능성과 Vitali 수렴 정리(3권), 유계 도함수의 ∫F' = F(b) − F(a)(7장 약속; 지배 수렴), Riemann 적분의 한계 정리(1권 thm:parameter-improper 등)가 지배 수렴의 특수 경우임을 한 절로.
중심 정리: 단조 수렴, Fatou, 지배 수렴, Vitali.

**19. Product Measures and Change of Variables** (이름 바꿈: Product Measures; 변수 변환을 21장에서 옮김)
곱 σ-대수, 곱측도, **Tonelli·Fubini**(11장 약속; 1권 Riemann판 thm:fubini는 특수 경우), 완비화, R^{p+q}의 Lebesgue 측도 = 곱, **C^1 미분동형의 변수 변환 공식**(11·17장 약속; Lebesgue판을 완전히 증명하고 1권 thm:change-variables의 Jordan판을 따름 정리로), 극좌표와 구면 측도(1권 prop:sphere-area 인용), 합성곱(L^1, 연속성, 1권 24장 cor:convolution과 연결), 확률측도의 가산 곱(D8).
중심 정리: Tonelli·Fubini, 변수 변환 정리.

**20. L^p Spaces** (이름 유지, Hilbert 공간 추상론은 22장으로)
적분 꼴 Jensen(3권), Hölder·Minkowski(9·13장 약속), 거의 어디서나 같은 함수의 몫으로서 L^p(15장 약속), 완비성(Riesz–Fischer), L^∞, 단순·계단·C_c·C_c^∞ 함수의 조밀성(5장 bump로 mollifier), 합성곱과 근사 항등(Young 부등식; 11권 KDE 예고), 포함 관계, L²는 완비 내적 공간이라는 것까지(추상론은 22장).
중심 정리: Hölder·Minkowski, Riesz–Fischer, C_c^∞의 조밀성.

**21. Signed Measures and Differentiation** (이름 유지, 변수 변환 빼고 미분 정리 넣음)
부호 측도, Hahn·Jordan 분해, **Radon–Nikodym**(3권 조건부 기댓값), Lebesgue 분해, 쌍대 (L^p)* = L^q(1 ≤ p < ∞), Vitali 덮개 또는 최대 함수로 Lebesgue 미분 정리, 절대 연속 함수와 Lebesgue 적분의 미적분 기본 정리, BV 함수의 거의 어디서나 미분 가능성(6장과 연결), Cantor 함수, C[a,b]의 쌍대 = BV(RS)는 note(6장 연결).
중심 정리: Radon–Nikodym, Lebesgue 미분 정리, 절대 연속 함수의 FTC.

**22. Hilbert Spaces and Compact Operators** (새 장)
1권 18·20·21장이 유한 차원에서 증명한 것의 무한 차원판을 한 장에 모은다: Hilbert 공간, 닫힌 볼록 집합·닫힌 부분공간으로의 사영 정리(18장 약속), Riesz 표현(18장 약속; 1권 반례와 대조), 정규직교 기저·Parseval·가분 Hilbert 공간 ≅ ℓ², L²(−π, π)에서 삼각계의 완비성(9장을 닫음; 15·18장 약속), Legendre 다항식의 완비성(1권 26장 연결), 유계 연산자와 수반(16장 약속의 기초만), 경계 조건이 있는 미분 연산자의 수반과 대칭성(18장 약속; 비유계 연산자는 Green 함수의 역으로 다룸), 컴팩트 연산자, 연속·L² 핵의 적분 연산자, **컴팩트 자기 수반 연산자의 스펙트럼 정리**(20장 약속), Schmidt 전개와 Hilbert–Schmidt 연산자(21장 약속), I − K의 **Fredholm 대안**(15장 약속), 정칙 **Sturm–Liouville 문제**: Green 함수 → 컴팩트 자기 수반 역 → 고유함수 전개(20·24·26장 약속; Bessel 고유함수 전개는 진술). 다루지 않는 것을 장 머리에서 밝힘: Hahn–Banach, 열린 사상, 균등 유계성, 약위상, 비유계 자기 수반 연산자의 스펙트럼 정리, 초함수(note).
중심 정리: 사영 정리, Riesz 표현, 컴팩트 자기 수반 연산자의 스펙트럼 정리, Sturm–Liouville 전개 정리.

**23. The Fourier Transform** (새 장)
L^1의 Fourier 변환, Riemann–Lebesgue, 합성곱 정리, Gauss 함수(1권 prop:gaussian-fourier 인용), 반전 정리, **Plancherel**(L²), R^n의 변환, Poisson 합 공식, 표본화 정리와 에일리어싱(11권 DFT의 바탕), 불확정성 원리(note), Fourier 급수·DFT와의 관계, Laplace 변환(1권 24장 note)·Mellin 변환(11권 FFTLog)은 note. PROMISES.md의 명시 약속은 없으나 3권(특성함수의 반전·연속성 정리는 3권 몫, 그 도구는 여기)과 11권(11장 DFT "표본화와 에일리어싱", 14장 Hankel 변환; 1권 11장의 11권 약속 "Laplace·Fourier 변환")이 기댄다.
중심 정리: 반전 정리, Plancherel, Poisson 합 공식.

### 현재 목차와의 차이 (요약)
- 1장: 이름 유지, 공리 → 구성(D1).
- 2장: Metric Spaces → Metric and Normed Spaces. 완비화, Banach 고정점 정리(8장에서 옮김), 유계 선형 사상 추가. 노름 동치는 3장으로.
- 3장: Continuity and Compactness → Compactness and Connectedness. 노름 동치, Riesz 보조정리, Baire, GL_n 성분 추가.
- 현재 4장 Differentiation: **삭제**(1권 6·8장이 전부 증명; 볼록 함수의 남은 사실은 새 6장, 적분 꼴 Jensen은 20장).
- 현재 5장 The Riemann Integral: **대체** → 새 6장 Monotone Functions, Bounded Variation, and the Riemann–Stieltjes Integral(Lebesgue 판정법, Jordan 내용 포함).
- 현재 6장 Sequences and Series of Functions: **둘로 나눔** → 4장 Uniform Convergence, 5장 Approximation and Equicontinuity(Weierstrass 근사를 옮기고 Arzelà–Ascoli·Baire 응용·bump 추가).
- 현재 7장: Differentiation in R^n → Differential Calculus in Normed Spaces.
- 현재 8장: Fixed Points and Existence Theorems → Fixed Points and Differential Equations(축약 사상 정리는 2장으로, Peano·최대 해·미분 가능 의존 추가).
- 9장: 이름 유지, L² 완비성은 22장으로.
- 12장: Series Representations → Series Representations and Analytic Continuation(monodromy, 정칙 계수 ODE 추가).
- 13장: Rouché 응용(근의 연속성, Schur–Cohn), 행렬 해석 함수 연산 추가.
- 15장: zeta 함수 추가.
- 16장: Lebesgue–Stieltjes 측도, π–λ 추가.
- 19장: Product Measures → Product Measures and Change of Variables(21장의 "변수 변환"을 옮김).
- 21장: 변수 변환 빼고 Lebesgue 미분 정리·L^p 쌍대 추가.
- **새 22장** Hilbert Spaces and Compact Operators, **새 23장** The Fourier Transform.
- 장 수 21 → 23. 차례대로 Part I 9장(1–9), Part II 6장(10–15), Part III 8장(16–23).
- OUTLINE.md의 Part I 요약 줄("실수의 완비성, 거리 공간, …, Riemann 적분, …")과 "넣지 않은 것" 줄도 함께 고칠 것.

### 다른 권이 2권에서 기대하는 것 (확인용)
- 3권(OUTLINE 3권 목차): 확률 공간 = 측도 공간(16장), 분포함수 ↔ Lebesgue–Stieltjes 측도(16장), 측도 적분으로서의 기댓값(17장), 상 측도·밀도의 변수 변환(17·19장), 독립과 곱측도·가산 곱(19장, D8), 조건부 기댓값의 일반 정의(Radon–Nikodym, 21장)와 L² 사영(22장), 수렴 양식·고른 적분 가능성(18장), 적분 꼴 Jensen(20장), 특성함수의 반전·연속성의 도구(23장, Fubini·지배 수렴). 3권 3장의 "측도에 대한 적분으로서의 기댓값"은 2권 17장을 인용하면 됨.
- 10권: 복소 인자의 특수 함수, 해석적 연속·가지(10·12장), 감마의 복소 확장·반사·곱 공식(15장), zeta의 연속(15장), 최급강하법의 경로 변형(11·13장의 Cauchy 정리; 점근 계산 자체는 OUTLINE대로 10권), 특수 함수 ODE의 연결 공식·Frobenius의 복소 영역판(12장의 정칙 계수 ODE·monodromy), Bernoulli 수(15장).
- 11권: Weierstrass 근사의 정량판(5장), Fourier 급수의 수렴·계수 감소(9장), 해석 함수의 Chebyshev 계수 감소(Bernstein 타원; 12·13장의 Cauchy 부등식으로 한 절 또는 note), 보간 오차의 Hermite 경로 적분 공식(11장의 예로), Fourier 변환·Poisson 합·표본화(23장), 합성곱과 근사 항등(20장), Sturm–Liouville·Hilbert 공간(22장; 평활 스플라인).
- 12권: Fréchet 미분·음함수 정리·Newton–Kantorovich(7장). 8권: 고윳값의 매끄러운 의존(7장), 연속 의존(13장 Rouché).

---

## D. 결정이 필요한 것

1. **R을 구성하는가?** 추천: 한다. 1장에서 Cauchy 수열 구성을 완전히, Dedekind 절단을 줄거리로; thm:r-unique를 완전 증명으로. 이유: 1권 27장 맺음말이 이미 "proves the completeness of the real numbers"라고 약속했고, 2장의 완비화가 같은 구성이다.
2. **현재 4장 Differentiation·5장 The Riemann Integral을 대체하는가?** 추천: 대체(C의 6장). 1권 6·7·8장이 그 내용을 완전히 증명했으므로 다시 쓰면 GUIDE가 금하는 중복이다. 대안: 5장만 "The Riemann Integral Revisited"로 남기고 Lebesgue 판정법·RS·BV를 넣는 방법(그러면 장 수가 같다).
3. **함수해석 금지문과 충돌을 어떻게 푸는가?** 추천: OUTLINE.md의 금지문을 "함수해석 일반과 초함수"로 좁히고 22장 Hilbert Spaces and Compact Operators를 더한다. 대안: 22장 없이 L^p 장 끝에 Hilbert 절을 두고 컴팩트 연산자·Sturm–Liouville·Schmidt·Fredholm은 진술 + 줄거리로 처리(이 경우 PROMISES의 126·132·147·153행과 101행의 Fredholm을 "진술로 지킴"으로 고침).
4. **지키기 어려운 약속을 어떻게 고치는가?** 추천:
   - 작용소의 스펙트럼 이론과 해석 함수 연산의 작용소판(140행): 행렬판을 13장에서 완전히 증명하고, 유계 작용소판은 note(진술 + 참고)로. 약속 문구를 "Cauchy 적분으로 정의한 f(A)와 정의의 일치(행렬); 작용소판은 note"로 고친다.
   - 유계 연산자의 Banach·Hilbert 공간 이론(101행, 16장에서): 2장(연산자 노름, 완비성, Neumann)과 22장(수반, 컴팩트)의 기초로 한정한다고 문구를 고친다.
   - Fuchs 이론·Riemann P-기호(153행): monodromy 정리와 정칙 특이점 둘레의 해의 연속은 12장에서 증명, Fuchs·P-기호는 note + Whittaker–Watson. 1권 문장("Volume II provides the tools")과 맞으므로 약속은 그대로 두고 범위만 PROMISES에 적는다.
   - Gâteaux 미분: 정의·예·Fréchet와의 관계만.
5. **Hamel 기저와 불연속 가법 함수(101행 두 곳)는?** 추천: 3장의 note(선택 공리 필요, Hamel 1905 인용)로 지키고 약속을 "note로"라고 고친다. 1권 15장 note가 이미 설명을 했으므로 본문 정리로 만들 이유가 약하다.
6. **일반 Jordan 곡선 정리(101행)는?** 추천: 11장에서 진술 + 줄거리(Thomassen 1992 또는 Hales 2007 참고)로. 복소해석의 정리들은 감김 수(호몰로지) 판으로 세워 Jordan 곡선 정리에 기대지 않게 한다. 약속 문구를 "진술과 줄거리"로 고친다.
7. **Fourier 변환 장(23장)을 더하는가?** 추천: 더한다. 3권 특성함수와 11권 DFT·표본화·Hankel이 기대는데 현재 목차에는 Fourier 변환이 없다. 대안: 19장 합성곱 절에 넣기(장 수 22).
8. **확률측도의 가산 곱(Kolmogorov 확장)은 2권인가 3권인가?** 추천: 2권 19장에서 확률측도의 가산 곱의 존재를 증명하고, 일반 Kolmogorov 확장(일관된 유한 차원 분포)은 3권에 둔다. 3권 9장(강법칙)이 i.i.d. 수열을 쓰기 때문.
9. **15장에 zeta와 Bohr–Mollerup을 넣는가?** 추천: zeta는 넣는다(10권 13장, Excursions 24장 Riemann 가설의 바탕). Bohr–Mollerup은 PROMISES 107행이 10권에 "Artin 인용만"으로 남겼으므로 2권에서는 진술 + 증명(짧음)으로 넣고 10권 약속은 "2권 인용"으로 바꾼다 — 또는 그대로 10권에 둔다(사용자 판단).
10. **축약 사상 정리를 2장으로 옮기는가?** 추천: 옮긴다. 현재 목차는 7장(역함수 정리)이 8장(축약 사상)의 결과를 앞서 쓴다.
11. **변수 변환 공식의 완전한 증명을 어디서?** 추천: 19장에서 Lebesgue판을 증명하고 1권의 Jordan판(thm:change-variables)을 따름 정리로. 대안: 7장에서 Riemann판(Spivak 방식)을 직접 — 중복이 생기고 증명이 더 길다.
12. **1권 정리를 2권에서 어떻게 인용하는가(조판)?** 현재 sdbook-eng.cls와 원고에 권 사이 상호 참조 장치가 없다. 추천: `xr-hyper`로 vol1의 .aux를 읽어 `\vref{thm:evt}` 꼴로 "Volume I, Theorem 5.x"를 자동 생성하는 명령을 클래스에 더한다(꼬리표가 두 권에서 겹치지 않게 2권 꼬리표 규칙도 정함 — 예: 2권은 같은 이름을 쓰되 1권 것은 접두 없이 외부 참조로). 손으로 번호를 적으면 1권 수정 때 깨진다.
13. **PROMISES.md에 없는 본문 약속 14곳(B 끝)을 PROMISES.md에 더하는가?** 추천: 더한다(관리자). 특히 완비화, 연속 ⇔ 역상, Baire의 유리수 예, 27장 맺음말.
14. **리뷰 권의 증명 깊이**: 약속에 "Volume II proves"라고 적힌 것(Picard–Lindelöf, Peano, 역함수·음함수, 변수 변환, FTA, 반사 공식, Sturm–Liouville 전개 등)은 줄거리가 아니라 완전 증명으로 한다는 것을 원칙으로 정할지. 추천: 그렇게 한다(1권 본문이 이미 "proves"라고 독자에게 말했다).
15. **권을 넘는 기호 결정**: Fourier 변환 관례(추천 \hat f(ξ) = ∫f(x)e^{−iξx}dx — 1권 prop:gaussian-fourier, 3권 특성함수와 맞음; 11권 DFT 관례와도 함께 정함), Lebesgue 측도 기호(추천 λ^n, 1권 vol은 Jordan 내용으로 유지), 유계 연산자 공간(추천 𝓛(X, Y)), 정칙 함수 공간(H(U) 또는 𝒪(U)), 폐포 기호(추천 \overline{S}).

---

## E. 표기

### 1권에서 이어받을 관례 (vol1.tex Notation, GUIDE.md)
- ℕ은 0을 포함. 수열 (a_n)은 0부터(아니면 (a_n)_{n≥1}).
- 집합·함수: f(S) 상, f^{-1}(T) 역상(모든 f에), 𝟏_S 지시 함수, A ⊂ B는 진부분집합, ≔ 정의, sup·inf, f|_S, id_A.
- 구간 [a,b], (a,b); x → a^+, f(a^+).
- C(D), C^k(I), C^∞(I); f', f^{(n)}; O, o, ∼(극한점 명시).
- 복소수: z̄, |z|, Re, Im, arg z, Arg z ∈ (−π, π], Log(주로그, def:complex-log), e^z = Σz^k/k!, atan2(y, x)(y가 먼저).
- 벡터·행렬: 굵은 직립 소문자 열벡터, 대문자 이탤릭 행렬, A^T·A^H(\mathsf), ⟨x, y⟩ = x^H y는 **둘째 인자에 선형**(첫째에 켤레선형) — Hilbert 공간에서도 반드시 유지(수학 문헌 다수와 반대이므로 2권 Notation에 다시 적는다), ‖x‖ = Euclid 노름, ‖x‖_p, ‖A‖_p, ‖A‖_F, ρ(A), 𝓛(V, W), ker, im, S^⊥, T^*.
- 해석: B(a, r)·B̄(a, r) 열린·닫힌 공, D𝐟(a) Jacobian(m×n), ∇f 열벡터, ∇²f Hessian, Δf = ∇·∇f Laplacian(Hessian과 구별), ∫_S f(x) dx, vol(S), 구면 좌표 ISO 80000-2(θ 극각).
- Γ(s)(1권은 s > 0과 음의 비정수 실수), 베타 B(a, b)(\mathrm B).
- Δ = E − I(27장 차분)는 2권에서 거의 안 씀; 쓰면 Laplacian과 문맥으로 구별(1권과 같음).
- 조판 관례(GUIDE): 본문 분수는 사선, 행렬은 별행·대괄호, 줄거리 증명은 `\begin{proof}[sketch]`와 †, remark·example 환경 없음, 장 끝 Notes, abstract는 한 문단(120단어 안팎), 영국식 철자, 권 이름은 "Volume I"처럼 로마 숫자, Notation은 두세 쪽 이하이고 여러 장에 걸친 기호와 책마다 다른 관례만.

### 2권 Notation 후보 (세 쪽 이하)
여러 장에 걸치거나 책마다 다른 것만 골랐다. 한 장 안에서 정의하고 끝나는 기호(Dirichlet 핵 D_N, Fejér 핵, 연속도 ω, 임시 함수)는 싣지 않는다.
- **거리와 노름**: (X, d); B(x, r)·B̄(x, r)은 거리 공간에서도 같은 뜻; S̄(폐포) — 닫힌 공 B̄, 켤레 z̄와 겹치므로 폐포는 cl S 또는 \overline{S}로 할지 정해야 함(추천: cl S는 쓰지 않고 \overline{S}, 켤레와는 문맥, 닫힌 공은 1권대로 B̄); int S, ∂S, diam S, dist(x, S); ‖f‖_∞ = sup|f|(1권 16장 note와 같음); ℓ^p; C(K), C_b(X), C_c(U), C_c^∞(U).
- **연산자**: 𝓛(X, Y)를 유계(연속) 선형 사상으로(유한 차원에서는 1권과 같음; 추천 — 𝓑(X, Y)를 따로 두지 않음), ‖T‖ 연산자 노름, T^* 수반.
- **함수열**: 고른 수렴은 기호 없이 말로(⇉ 쓰지 않음 — 정할 것).
- **변동과 Stieltjes**: V_a^b(f) 전변동, ∫_a^b f dα.
- **복소해석**: 원판 D(a, r)와 단위 원판 𝔻 대 1권의 B(a, r) — 추천: 1권 기호 B(a, r)를 복소평면에서도 쓰고 𝔻만 더함; Ĉ Riemann 구; ∮_γ; 감김 수 w(γ, a)(1권 def:winding-number와 같은 글자); Res(f, a); H(U)(U 위 정칙 함수; 또는 𝒪(U) — 정할 것); ζ(s).
- **측도**: (X, 𝒜, μ); σ(𝓔); 𝓑(ℝ^n) Borel σ-대수; Lebesgue 측도 λ 또는 λ^n(추천; 1권 vol(S)는 Jordan 내용이자 λ^n(S)와 같은 값이라고 한 줄); μ^* 외측도; a.e.; ∫f dμ, ∫_X f(x) dμ(x); f^+, f^−; L^p(μ), ‖f‖_p, ‖f‖_∞(ess sup — sup 노름과 같은 글자이므로 주의 문구); μ ⊗ ν; f ∗ g; ν ≪ μ, ν ⊥ μ, dν/dμ, |ν|; δ_a; f_*μ 상 측도(3권이 이어 씀).
- **Fourier**: 계수 \hat f(k) = (1/2π)∫_{−π}^{π} f(x)e^{−ikx}dx와 1권 18장의 실수 꼴 a_k, b_k의 관계; S_N f(1권과 같음); 변환 \hat f(ξ) = ∫ f(x)e^{−iξx}dx(각진동수 꼴, 2π 없이) — 1권 prop:gaussian-fourier(∫e^{−x²}cos tx = √π e^{−t²/4})와 3권 특성함수 φ(t) = ∫e^{itx}dμ와 맞춘 추천안. 다른 관례(e^{−2πixξ})와의 환산을 Notation에 한 줄(D15).
- **Hilbert**: H, ⟨·,·⟩(둘째 인자 선형, 위와 같음), P_U 직교 사영(1권 def:orthogonal-projection과 같음), 정규직교 기저 (e_k).
