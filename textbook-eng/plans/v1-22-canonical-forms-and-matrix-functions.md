# 1권 22장 Canonical Forms and Matrix Functions — 얼개

예비 권(리뷰 방식). Part III Linear Algebra의 마지막 장. Schur 분해, Jordan 표준형, 행렬 함수와 행렬 지수를 다룬다. plans/PROMISES.md에서 22장 앞으로 온 약속을 모두 지킨다(9·13·16·17·18·19·20·21장에서; "22·25장" 공동 약속은 행렬 지수의 정의·성질·계산은 여기서, x′ = Ax의 풀이와 안정성은 25장에 남긴다): Schur 삼각화와 대각화 가능 행렬의 조밀성, 정규 행렬을 Schur 꼴로 다시(20장 thm:normal-unitary가 이미 증명했으므로 둘째 증명으로), 실 Schur 꼴과 실 정규 행렬의 실 블록 꼴, Jordan 꼴의 존재와 유일성(16장의 핵 사슬·증분의 단조성·Fitting 분해를 T − λ id에 적용), 닮음의 완전 불변량, 핵 사슬의 마지막 차원 = 대수적 중복도, 최소다항식에서 λ의 지수 = 가장 큰 Jordan 블록의 크기, 거듭제곱의 이항 계수 꼴과 성분이 ΣP_i(k)μ_i^k 꼴임(19장이 4장을 인용해 진술한 것의 증명), A^k → 0 ⇔ ρ(A) < 1을 Jordan 꼴로 다시, 행렬 함수 f(A)(Hermite이면 Qf(Λ)Qᴴ; 20장), 행렬 지수 e^A의 정의·성질·det e^B = e^{tr B}(17장)·고윳값 e^λ(19장), f′ = Af의 유일성을 e^{tA}로(9장 thm:exp-ode의 확장; 완전한 풀이 이론은 25장), Weyl 부등식 |λ₁⋯λ_k| ≤ σ₁⋯σ_k의 증명과 비정규성(21장), Neumann 급수와 스펙트럼 반지름(19장에서 끝난 것은 인용). PROMISES.md를 직접 읽고 빠뜨리지 않는다. 스칼라는 ℂ를 기본으로, 실행렬의 실수 꼴을 따로.

## Abstract (한 문단, 120단어 이내, 동기부여)
모든 행렬이 대각화되지는 않는다. 그러면 닮음 변환으로 갈 수 있는 "가장 단순한 꼴"은 무엇인가. 답은 둘이다: 유니터리 변환만 쓰면 삼각 꼴(Schur)까지 — 이것이 수치 계산이 실제로 쓰는 꼴이고, 임의의 닮음을 쓰면 Jordan 꼴까지 — 이것이 이론이 쓰는 꼴이지만 수치적으로는 구할 수 없다. 이 두 꼴 위에서 행렬의 함수, 특히 미분방정식을 푸는 행렬 지수가 정의된다는 점.

## 절 구성
22.1 The Schur Decomposition — 정리: 모든 A ∈ ℂ^{n×n}에 A = QTQᴴ(Q 유니터리, T 위 삼각, 대각 = 고윳값을 원하는 순서로). 증명(귀납: 고유벡터 하나를 정규직교 기저로 확장; 18장). 따름: 정규 ⇔ T가 대각(20장 thm:normal-unitary의 둘째 증명; 삼각이면서 정규이면 대각임을 증명), Hermite·유니터리·반Hermite 행렬의 스펙트럼 정리들이 한꺼번에, tr = Σλ·det = ∏λ의 다른 증명, Cayley–Hamilton의 Schur 증명(한 문단), 스펙트럼 사상 정리의 다른 증명(19장 것과 비교), 교환하는 행렬들의 동시 삼각화(증명).
22.2 Consequences of Schur's Theorem — 대각화 가능 행렬의 조밀성(증명; 19장 note의 완결)과 그 쓰임(연속성 논증으로 항등식 옮기기: 예 Cayley–Hamilton, p_{AB} = p_{BA}), 비정규성의 척도: ‖A‖_F² = Σ|λ_i|² + ‖N‖_F²(N은 T의 순상삼각 부분; Schur 부등식 Σ|λ_i|² ≤ ‖A‖_F²와 등호 ⇔ 정규; 증명), Weyl 부등식 |λ₁⋯λ_k| ≤ σ₁⋯σ_k(21장 약속; 증명 — Schur 꼴의 선행 블록과 특이값 엇갈림/Cauchy–Binet), ρ(A) ≤ σ₁과 |det| = ∏σ의 사이를 메우는 것이라는 설명, 임의의 ε에 대해 ‖A‖ ≤ ρ(A) + ε인 유도 노름의 존재(증명; 19장 Gelfand·A^k → 0의 셋째 길).
22.3 The Real Schur Form — 실행렬은 실 직교 닮음으로 준삼각 꼴(1×1과 2×2 블록; 2×2 블록은 켤레 복소 고윳값 쌍)(증명; 19장 thm:real-block-diagonal과 연결), 실 정규 행렬의 실 표준 꼴(회전–확대 블록; 20장 약속), 따름: 직교 행렬의 표준 꼴(회전 블록과 ±1; 18장의 3×3 회전축 정리의 일반화), 반대칭 행렬의 표준 꼴.
22.4 Nilpotent Matrices and Jordan Chains — 멱영 연산자, 핵 사슬 ker N ⊂ ker N² ⊂ …(16장)과 증분 d_j = dim ker N^j − dim ker N^{j−1}의 단조 감소(16장 인용), Jordan 사슬과 사슬 기저의 존재(증명: 위에서부터 사슬을 고르는 표준 논증을 완전히), 멱영 행렬의 표준 꼴 = Jordan 블록 J_k(0)들의 직합, 블록 수와 크기가 d_j에서 결정됨(크기 ≥ j인 블록 수 = d_j; 증명) — Young 도형 그림. 그림: 사슬 기저의 점 도형(행 = 사슬, 열 = 핵 사슬의 층).
22.5 The Jordan Canonical Form — 일반 고유공간 G_λ = ker(A − λI)^n, dim G_λ = am(λ)(19장 약속; 증명), ℂⁿ = ⊕G_λ(증명: 16장 Fitting 분해를 A − λI에 적용하고 귀납, 또는 Cayley–Hamilton과 서로소 다항식 — 19장 lem:spectral-projections의 일반화; 한 길을 완전히), 각 G_λ 위에서 A − λI가 멱영 → Jordan 꼴의 존재(증명), 유일성(블록의 순서까지; 증명: rank(A − λI)^j가 닮음 불변), 닮음의 완전 불변량(16장 약속: 두 행렬이 닮음 ⇔ 모든 λ, j에서 rank(A − λI)^j가 같음), 손 계산 예(4×4 또는 5×5에서 Jordan 꼴과 사슬 기저를 끝까지), 최소다항식 = ∏(x − λ)^{가장 큰 블록 크기}(19장 약속; 증명)와 대각화 가능 ⇔ 블록이 모두 1×1(19장 정리의 재확인), 특성다항식 = 최소다항식 ⇔ 고윳값마다 블록 하나(nonderogatory)와 동반 행렬, A와 Aᵀ는 닮음(증명), 실 Jordan 꼴(진술 + 줄거리).
22.6 Powers, Again — J_k(λ)^m의 성분 = 이항 계수 C(m, j)λ^{m−j}(증명), A^m의 성분이 ΣP_i(m)λ_i^m 꼴(P_i의 차수 < 가장 큰 블록 크기; 19장이 인용만 한 것의 증명 — 4장 prop:linear-recurrence-k의 완전한 증명이 여기서 따라 나온다는 것을 밝히고 27장에 연결), A^m → 0 ⇔ ρ < 1의 Jordan 증명, 유계 ⇔ ρ ≤ 1이고 절댓값 1인 고윳값의 블록이 모두 1×1(증명; 25·27장 안정성 예고), ‖A^m‖의 성장률 m^{k−1}ρ^m, 13장 그림(‖A^k‖가 먼저 커지는 예)을 Jordan 블록으로 설명.
22.7 Why the Jordan Form Is Not Computed — Jordan 꼴은 행렬의 불연속 함수(J_n(0) + εE의 예: 19장 표), 블록 구조 판정이 계수 판정과 같은 불안정성을 가짐(15·21장), 닮음 변환 행렬의 조건수가 임의로 커짐(예를 직접 계산), 대조: Schur 꼴은 유니터리 변환이라 안정(18·21장), 의사스펙트럼은 note(Trefethen–Embree 인용), 공용 빌드로 작은 실험(API 이름 확인: sd.linalg.schur, eig 등): 결함 행렬의 schur 결과 T의 대각과 ‖QᴴQ − I‖, ‖A − QTQᴴ‖; 같은 행렬의 eig가 주는 고유벡터 행렬의 조건수. 수치는 직접 잰 것만, 원인 추측 없이.
22.8 Functions of Matrices — 세 정의와 그 일치: (i) 대각화 가능: f(A) = S f(Λ) S⁻¹(잘 정의됨 증명), Hermite이면 Qf(Λ)Qᴴ(20장), (ii) 거듭제곱 급수: Σc_kA^k가 ρ(A) < 수렴 반지름이면 수렴(증명; 8장 거듭제곱 급수와 19장 Gelfand), (iii) Jordan 꼴: f(J_k(λ))의 성분 = f^{(j)}(λ)/j!(증명: 다항식에서 Taylor로)과 일반 정의, (iv) Hermite 보간 다항식: f(A) = p(A), p는 스펙트럼에서 f와 필요한 차수까지 일치(16장 prop:hermite; 증명 — 정의들이 같음), 성질: f(A)는 A와 교환, f(SAS⁻¹) = Sf(A)S⁻¹, 고윳값 f(λ), (fg)(A) = f(A)g(A), 합성; 예: A^{1/2}(20장 thm:square-root와의 관계, 일반 행렬에서는 유일하지 않음·존재하지 않을 수 있음 — J₂(0)), A⁻¹, 부호 함수, 로그는 한 문단. Cauchy 적분 표현은 note(2권 예고).
22.9 The Matrix Exponential — e^A = ΣA^k/k!(모든 A에서 수렴; 증명), e^{A+B} = e^Ae^B(교환할 때; 증명, 교환하지 않으면 거짓인 예), (e^A)⁻¹ = e^{−A}, e^{SAS⁻¹} = Se^AS⁻¹, det e^A = e^{tr A}(17장 약속; 증명), 고윳값 e^λ(19장 약속), d/dt e^{tA} = Ae^{tA}(증명; 항별 미분 — 8장)과 x′ = Ax, x(0) = x₀의 해 x(t) = e^{tA}x₀의 존재와 유일성(9장 thm:exp-ode의 확장; 증명 — 9장 증명과 같은 꼴), e^{tJ_k(λ)}의 꼴(다항식 × e^{λt}; 증명)과 e^{tA}의 성분 = Σ p_i(t)e^{λ_it}, 반대칭 A에서 e^A가 직교(회전; 3×3 Rodrigues 공식 증명 — 12장 외적, 18장 회전축), Hermite A에서 e^{iA}가 유니터리, e^A = lim(I + A/m)^m(증명; 9장 thm:exp-limit의 행렬판), Lie 곱 공식·Baker–Campbell–Hausdorff는 note, 2×2·3×3 손 계산 예(대각화 가능·결함·회전).
22.10 Computing Matrix Functions: A First Look — e^A를 계산하는 "의심스러운" 방법들(Moler–Van Loan 인용 검증): Taylor 급수의 상쇄(‖A‖가 크고 고윳값이 음수일 때 — 8장의 e^{−20} 예의 행렬판; 직접 계산해 표), 고윳값 분해로 계산할 때 κ(S)가 오차를 키움(거의 결함인 행렬 예), 스케일링과 제곱 e^A = (e^{A/2^s})^{2^s}의 생각(알고리즘 유사코드: Taylor 또는 Padé 한 가지를 간단한 꼴로 — 실제 구현은 8권), 공용 빌드의 expm(API 이름은 API-INVENTORY.md에서 확인: sd.linalg.expm, matrix_power 등)과 비교하는 표(정확한 값은 python-flint), matrix_power의 반복 제곱(4권·8권 예고), 비정규 행렬에서 ‖e^{tA}‖의 일시적 증가 그림(ρ가 아니라 수치 범위/비정규성이 초기 행동을 정함 — 22.2와 연결; 25장 예고). 수치는 직접 잰 것만(시험 행렬을 만드는 법을 밝히고), 원인 추측 없이.

맺음: Part III를 닫는 문단 — 13–22장이 세운 것(계산의 말, 소거, 공간, 사상, 행렬식, 직교성, 고윳값, 대칭성, SVD, 표준형)과 Part IV(Differential Equations)로의 다리(25장에서 e^{tA}가 주인공이 된다는 것). OUTLINE.md의 23–27장이 계획하지 않는 것은 약속하지 않는다.

## note 상자 후보
- 의사스펙트럼(Trefethen–Embree)
- Cauchy 적분으로 정의한 f(A)와 해석 함수 연산(2권)
- Baker–Campbell–Hausdorff와 Lie 군·Lie 대수 한 문단
- 유리 표준형(Frobenius): 체를 넓히지 않고 얻는 표준형(동반 행렬 블록) 한 문단
- Jordan 꼴의 Weyr 특성과 Segre 특성(용어)
- 역사(Schur 1909, Jordan 1870 Traité, Weierstrass 1868 기본 인자, Sylvester 1883의 f(A) 공식, Moler–Van Loan 1978/2003, Higham의 Functions of Matrices)는 Notes에 — 출처로 뒷받침되는 것만

## 증명 방침
이 장의 정리는 모두 증명한다. 진술만: 실 Jordan 꼴(줄거리), Baker–Campbell–Hausdorff, 의사스펙트럼 관련 결과, 스케일링–제곱법의 오차 분석(8권).

## Notation
더할 것: e^A, J_k(λ) 정도(한 줄). Notation은 3쪽을 넘지 않는다(여유가 거의 없으면 한 장에서만 쓰는 기존 항목을 덜어낸다).

## 뒤 장에 하는 약속(PROMISES.md에 더할 것)
24장: 상수 계수 방정식의 중근과 t^je^{λt}(Jordan 블록의 그림자). 25장: x′ = Ax의 풀이·위상 평면·안정성(Re λ < 0 ⇔ e^{tA} → 0; Jordan 꼴로)·비동차계의 상수 변화법 e^{tA}∫e^{−sA}f(s)ds. 27장: 점화식의 중근과 k^jλ^k·안정성(유계 조건). 2권: 해석 함수 연산, 작용소의 스펙트럼. 8권: schur·hessenberg·expm·matrix_power·solve_sylvester의 알고리즘(QR 반복, 스케일링–제곱법과 Padé, Bartels–Stewart), 의사스펙트럼과 비정규성. 9권: 큰 행렬의 e^{A}v(Krylov). 

## 분량
제한 없음. 55쪽 안팎 예상.
