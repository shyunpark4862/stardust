# 앞 장이 뒤 장에 한 약속 (집필 에이전트가 갱신한다)

장을 쓰는 에이전트는 자기 장 앞으로 온 약속을 모두 지키고, 자기가 뒤 장에 한 약속을 여기에 더한다. 지킨 약속에는 [완료]를 붙인다.

## 1권
- [완료] 6장 (Differentiation): Newton 방법과 Heron 방법의 관계; sin의 도함수; 평균값 정리로 Lipschitz 상수 얻기 (5장) — 6장 sec:newton, thm:trig-derivatives, cor:lipschitz-derivative
- [완료] 7장 (Integration): 고른 연속에서 Riemann 적분 가능성으로 (5장); 미적분의 기본 정리의 유일성 절반은 6장 cor:zero-derivative를 인용; 기본 정리는 도함수의 적분 가능성을 가정(6장 Volterra 상자); arctan·arcsin을 1/(1+x²)·1/√(1−x²)의 적분으로 (6장) — 7장 thm:continuous-integrable, cor:antiderivative, thm:ftc2 뒤 'The Hypotheses of the Second Part', prop:arctan-integral
- 8장 (Sequences, Series, and Approximation): e = Σ 1/k!; H_n이 ln n처럼 자람; Stirling 공식; sin x / x의 편차 x²/6을 Taylor로 설명 (5장). 급수의 수렴·비교와 기하급수 (2장). 무한합의 극한 (3장 1223행 부근). 교란 순열 비율의 극한 1/e, 가운데 이항 계수의 Stirling 어림 (4장). Landau 기호 o·O를 본격적으로(무한대 극한 포함; 6장 def:landau는 h→0만); Taylor로 Newton 오차비 (x_{k+1}−c)/(x_k−c)² → f''(c)/(2f'(c)) 증명; sin·cos 부등식 사다리(6장 prop:sin-cos-ladder)를 Taylor 급수로 잇기; 차분몫 오차를 Taylor로 다시; arctan·arcsin의 거듭제곱 급수; 매끄러움 차수와 정확도 차수 (6장); (7장에서) 적분 판정법 일반(7장 prop:sum-integral은 유한 부등식과 Σ1/k^p만), H_n − ln n → Euler 상수(7장은 0과 1 사이까지), Taylor 정리의 적분 나머지(부분적분), Wallis 곱(7장 prop:wallis)으로 Stirling 상수 √(2π)
- 9장 (Elementary Transcendental Functions): 삼각함수의 엄밀한 정의와 Assumption(3장의 삼각 가정, 5장 asm:trig-geometry: 홀짝성, sin(π/2)=1, sin(kπ)=0, 주기, π>3, (0,π/2)에서 sin·cos>0, 부채꼴 넓이 x/2, 단조성)의 증명; 0<x<π/2에서 sin x < x < tan x를 정의에서 직접(5장 lem:sin-bounds가 삼각형 넓이·넓이의 단조성을 학교 기하로 썼으므로); 부채꼴 넓이를 적분으로; ln을 exp의 역함수로; e^{iθ}의 정의 (3·5장). 1 − t ≤ e^{−t} (4장). exp'=exp, ln'=1/x, 실수 지수 x^s의 도함수 s x^{s−1}과 실수 지수 Bernoulli 부등식(6장 prop:bernoulli-rational과 같은 증명); exp의 강볼록·ln의 강오목, e^x ≥ 1+x를 접선 부등식으로; ln의 오목성과 Jensen으로 AM–GM과 가중 꼴(6장 cor:power-means 뒤 문단); l'Hôpital로 ln x/x^p → 0, x^k/e^x → 0 (6장); (7장에서) 7장의 Λ(x)=∫_1^x dt/t (eq:log-integral, prop:log-additive)가 ln임을 보이기; 감마 함수(7장 def:gamma)에 쓴 e^{−t}의 성질과 실수 지수 t^s, 실수 p에 대한 p-적분(prop:p-integrals); 삼각함수가 복소 지수·로그로 표현되어 초등 함수에 든다는 것(7장 Liouville 절); ln 2를 급수로 계산
- 10장 (Differential Calculus in Several Variables): 여러 변수의 연쇄 법칙(자동 미분의 역방향 모드가 쓰는 꼴); 벡터 값 함수에서는 평균값 정리가 부등식 꼴로만 남음; 최선의 선형 근사로서의 미분 (6장)
- 11장 (Integral Calculus in Several Variables): Gauss 적분 ∫e^{−x²}=√π와 Γ(1/2)=√π, ∫e^{−x²/2}=√(2π); 회전체 부피·Cavalieri 원리의 엄밀한 정당화 (7장)
- 27장 (Difference Equations and Recurrences): k계 선형 점화식의 일반 풀이 (4장)

## 다른 권
- 2권: 대수학의 기본 정리의 증명 (1권 3장); Fourier 급수로 Σ1/n² = π²/6, 축약 사상, Baire, 열린 집합과 컴팩트 (1권 5장); 존재와 유일성(Picard–Lindelöf); Weierstrass 함수의 연속성(고른 수렴)과 Banach의 범주 정리(어디서도 미분 불가능한 함수가 전형적); 6장 prop:fixed-point-contraction을 축약 사상 정리의 1차원 경우로 인용; Volterra 함수와 Lebesgue 적분의 동기 (1권 6장); (1권 7장에서) Lebesgue 적분과 Dirichlet 함수의 Lebesgue 적분 0, 측도 0과 Lebesgue 판정법, 유계 도함수에 대한 ∫F'=F(b)−F(a), L² 공간, 유수 정리로 ∫_0^∞ sin x/x dx = π/2, Cauchy 주값, Riemann–Stieltjes를 측도 적분으로
- 4권: 위상 정렬, 정렬의 하한(이진 나무의 잎 수), 분할 정복 점화식 (1권 2·4장)
- 3권: Jensen 부등식의 확률 꼴 f(E X) ≤ E f(X); 중앙값은 절대 편차 합, 평균은 제곱 편차 합을 최소화 (1권 6장); (1권 7장에서) 확률 밀도와 분포함수, Riemann–Stieltjes 꼴의 기댓값, Cauchy 분포에 평균이 없음, 감마·카이제곱·t 분포의 정규화
- 5권: 문제의 조건과 알고리즘의 안정성 구분(√(x+1)−√x 예); 차분몫의 반올림 오차와 최적 h; Newton 역수 반복으로 나눗셈 (1권 6장); 구적 합의 반올림 오차와 합산 방법 (1권 7장)
- 11권: 고차 차분 공식과 외삽, sd.differentiate (1권 6장); (1권 7장에서) Gauss 구적, 적응형 구적, Romberg 적분과 Euler–Maclaurin 공식, 주기 함수의 사다리꼴 규칙, 조건 수렴 적분의 수치 계산, 특이점이 있는 적분(√x 예)
- 12권: 중근에서 Newton 수정, 이분법과 섞은 안전한 근 찾기와 할선법, 볼록 최적화 (1권 6장)
- 10권: 적분으로 정의되는 특수 함수 erf, Si, Li, 감마 함수, 타원 적분(타원의 길이) (1권 7장)
- 13권: Monte Carlo 적분 (1권 7장)
