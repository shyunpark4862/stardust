# 1권 17장 Determinants — 얼개

예비 권(리뷰 방식). Part III의 다섯째 장. 행렬식을 정의하고 그 성질, 계산, 기하적 뜻을 모두 증명한다. plans/PROMISES.md에서 17장 앞으로 온 약속이 많다(10·12·13·14·15·16장에서): 치환 합 정의, det(AB) = det A det B, det Aᵀ = det A, 아무 행·열로의 여인수 전개, Cramer 규칙과 수반 행렬(A⁻¹의 성분이 다항식/det A), |det|가 넓이·부피(2×2, 3×3; 11장 thm:linear-volume과 연결), 3×3 행렬식의 순환 불변성과 삼중곱 ⟨x×y,z⟩ = det(x,y,z), Gram 행렬식 √det(AᵀA) = k차원 부피와 det(I + uuᵀ) = 1 + ‖u‖², 블록 삼각 행렬의 행렬식과 det M = det A · det S(Schur 여원), det A = ± 피벗의 곱·기본 행렬의 행렬식·det ≠ 0 ⇔ 가역(10장 thm:invertible-matrices (i)의 남은 부분), Wronskian 행렬식, 가장 큰 0 아닌 소행렬식 = 계수, 연산자의 행렬식이 잘 정의됨(닮음 불변). PROMISES.md를 직접 읽고 하나도 빠뜨리지 않는다. 고윳값은 쓰지 않는다(19장; 특성다항식은 예고만). 스칼라는 F(R 또는 C).

전개 방식: 행렬식을 "열에 대해 다중선형이고 교대이며 det I = 1인 함수"로 특징짓고(공리적 정의), 그런 함수가 꼭 하나 있음을 치환 합 공식으로 보인다. 10장이 치환 합으로 정의했으므로 그것과 같은 것임을 분명히 한다. 2·4장의 치환(부호, 호환)을 쓴다 — 치환의 부호가 잘 정의됨이 앞 장에 증명되어 있는지 확인하고, 없으면 여기서 증명한다(반전 수로).

## Abstract (한 문단, 120단어 이내, 동기부여)
행렬식은 선형 사상이 부피를 몇 배로 바꾸는지, 방향을 뒤집는지를 말하는 한 수이다. 이 수가 0인지 아닌지가 가역성을 가르고, 변수 변환 공식(11장)의 Jacobian 인자가 여기서 나온다. 한편 계산 도구로서는 공식(n!개 항)이 아니라 소거(n³)로 구하며, 행렬식 자체는 수치 계산에서 조심해서 써야 한다는 점.

## 절 구성
17.1 Area, Volume, and Orientation — 2×2: 평행사변형의 부호 있는 넓이로서 ad − bc(12장의 u∧v와 연결), 세 성질(각 열에 선형, 같은 열이면 0, 단위 정사각형은 1)을 그림으로 관찰; 3×3: 평행육면체의 부호 있는 부피와 삼중곱(12장 외적). 그림: 전단이 넓이를 보존하는 것, 한 변을 더하는 것(다중선형성), 방향(부호).
17.2 Permutations and Signs — 치환, 호환, 반전 수, 부호 sgn의 잘 정의됨과 sgn(στ) = sgn σ sgn τ(증명 또는 앞 장 인용), 짝·홀 치환이 반씩. 그림: 치환 도표의 교차 수.
17.3 The Determinant — 다중선형 교대 함수의 정의, 교대 ⇒ 열 교환에 부호 바뀜(증명; 표수 2 주의는 note), 유일성: 다중선형 교대 함수 D는 D(A) = D(I) · Σ sgn σ ∏ a_{σ(j) j}(증명), 존재: 치환 합이 실제로 다중선형·교대·det I = 1(증명), 따라서 정의. 2×2, 3×3 공식(Sarrus는 3×3에만 통한다는 경고), 삼각 행렬의 행렬식 = 대각 성분의 곱(증명).
17.4 Properties — det Aᵀ = det A(증명; 그래서 행에 대해서도 같은 성질), 행·열 연산의 효과(교환 −1, 배율 c, 전단 불변; 증명)와 기본 행렬의 행렬식, det(AB) = det A det B(유일성 정리로 증명), det(A⁻¹) = 1/det A, det(cA) = cⁿ det A, 가역 ⇔ det ≠ 0(증명; 14장 thm:invertible에 한 항목 더함), 닮음 불변과 연산자의 행렬식(16장 약속), det(Ā) = conj(det A)와 det Aᴴ, 직교 행렬은 ±1·유니터리는 |det| = 1(회전 대 반사), 덧셈에 대해서는 아무 법칙이 없다는 예.
17.5 Computing Determinants by Elimination — PA = LU에서 det A = ± ∏ u_ii(증명; 14장 약속), 비용 (2/3)n³ 대 치환 합 n·n!(표: n = 5, 10, 20에서 항의 수와 시간 어림), 알고리즘(유사코드: 부분 피벗과 부호 추적), 손 계산 예 4×4, 정수 행렬에서 분수를 피하는 Bareiss 소거는 note 또는 한 문단(인용).
17.6 Cofactor Expansion — 소행렬식과 여인수, 아무 행·열로의 Laplace 전개(증명), 다른 행의 여인수로 전개하면 0(증명), 수반 행렬 adj A와 A adj A = (det A) I(증명), A⁻¹ = adj A / det A(10장 thm:invertible-matrices (iii)과 cor:newton-n-local이 쓴 "성분이 다항식/det"), Cramer 규칙(증명; 실제 계산에 쓰지 말라는 것과 비용), 정수 행렬의 역이 정수 ⇔ det = ±1(증명), 예: Vandermonde 행렬식 ∏(x_j − x_i)(증명; 14장 thm:interpolation의 다른 증명), 삼중대각 T_n의 행렬식 점화식 det T_n = n + 1(13·14장 연결; 27장 예고).
17.7 Block Matrices and Low-Rank Updates — 블록 삼각의 행렬식 = 대각 블록 행렬식의 곱(증명), det M = det A · det(D − C A⁻¹ B)(13장 약속), 가역 ⇔ 대각 블록 가역(블록 삼각; 13장 약속), 행렬식 보조정리 det(A + u vᵀ) = det A (1 + vᵀA⁻¹u)(증명)와 det(I + uuᵀ) = 1 + ‖u‖²(12장 약속), Sylvester의 항등식 det(I_m + AB) = det(I_n + BA)(증명), 주의: det [A B; C D] ≠ det A det D − det B det C(반례; 교환할 때만 성립한다는 진술).
17.8 Volume — n차원 평행체 P(a_1, …, a_n)의 부피 = |det A|(증명: 기본 행렬 분해와 11장 thm:linear-volume — 11장이 무엇을 증명했고 무엇을 14장에 맡겼는지 읽고, 여기서 이야기를 완결한다; 순환하지 않게), 선형 사상의 부피 배율과 변수 변환 공식의 Jacobian(11장 연결), 방향: det > 0인 기저와 오른손 좌표계, GL_n(R)의 두 성분은 note, Gram 행렬식: k개 벡터가 펼치는 k차원 평행체의 부피 = √det(AᵀA)(증명: k = n의 경우와 직교 분해; 18장의 Gram–Schmidt 없이 할 수 있는 길 — 예: 귀납과 높이, 또는 A = QR을 쓰지 않는 Schur 여원 논증 — 을 택하고, 18장에서 QR로 다시 본다고 약속), det(AᵀA) ≥ 0과 = 0 ⇔ 열이 종속(13장 Gram 행렬), Cauchy–Binet 공식(진술 + 증명; k = 2, n = 3이 Lagrange 항등식 — 12·13장 연결), 12장 eq:graph-area의 근거 완결, Hadamard 부등식 |det A| ≤ ∏‖a_j‖(증명은 Gram 논증으로 가능하면 여기서, 아니면 18장 약속).
17.9 Minors and Rank; Determinants of Functions — 계수 = 가장 큰 0 아닌 소행렬식의 크기(증명; 15장의 "가장 큰 가역 부분행렬"에서), Wronskian W(f_1, …, f_k)와 독립성 판정: W ≠ 0인 점이 있으면 독립(증명), 역은 거짓(예: x²과 x|x|), 해들의 Wronskian에 대한 Abel 공식은 24장 약속, 행렬식의 미분: d/dt det A(t) = Σ(한 열씩 미분) 과 Jacobi 공식 (det A)' = tr(adj A · A')(증명; 10장 연결), det이 성분의 다항식이므로 연속·매끄러움 ⇒ 가역 행렬의 집합은 열린 집합이고 역행렬 사상이 매끄러움(10장 thm:invertible-matrices (iii) 완결).
17.10 Determinants in Computation: A First Look — 행렬식은 "거의 특이한가"의 척도가 아니다(예: 0.1·I_n의 det는 10^{−n}, 그러나 완벽한 조건; 반대로 det = 1이지만 나쁜 조건인 삼각 행렬 — 14장 Wilkinson형 예), 넘침·밑넘침과 slogdet(부호와 로그 절댓값; 공용 빌드에 sd.linalg.det·slogdet이 있는지 API-INVENTORY.md로 확인하고 작은 실험 표: n에 따라 det가 넘치는 지점과 slogdet 값), 소거로 구한 det의 상대 정확도 예(Hilbert 행렬의 정확한 행렬식(유리수)과 비교; 직접 계산), Cramer 규칙 대 소거의 시간(작은 n; 직접 재거나 연산 수 표만). 수치는 직접 잰 것만, 원인 추측 없이.

## note 상자 후보
- 외대수와 쐐기곱: 행렬식 = ⋀ⁿ 위의 작용 (한 문단; 12장의 미분 형식 note와 연결)
- 퍼머넌트: 부호만 뺐는데 계산이 어려워진다(Valiant; 인용 검증)
- 행렬식의 계산 복잡도 = 행렬 곱(13장 ω note 연결)
- 표수 2에서 교대와 반대칭의 차이
- 역사(Leibniz, Seki, Cramer 1750, Cauchy 1812, Jacobi)는 Notes에 — 출처로 뒷받침되는 것만

## 증명 방침
이 장의 정리는 모두 증명한다.

## Notation
det A는 이미 있을 것(확인). 더할 것: adj A, sgn σ 정도. Notation은 3쪽을 넘지 않는다.

## 뒤 장에 하는 약속(PROMISES.md에 더할 것)
18장: Gram 행렬식을 QR로 다시(det R의 대각), Hadamard 부등식(여기서 못 하면). 19장: 특성다항식 det(λI − A), det = 고윳값의 곱. 20장: 양정치 ⇔ 선행 주 소행렬식 > 0(Sylvester 판정), Gauss 적분 π^{n/2}/√det A. 24장: Abel 공식. 2권: 변수 변환 공식의 완전한 증명. 8권: det·slogdet의 구현과 정확도. 16권: 방향 판정 술어(12장 약속과 연결).

## 분량
제한 없음. 50쪽 안팎 예상.
