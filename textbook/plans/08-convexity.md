# 8장 Convexity and Optimality — 계획

GUIDE.md와 CHECKLIST.md를 다시 읽고 쓴다(2026-11-25, 7장 때와 바뀐 것 없음). 옛 42장(legacy/chapters/42-optimization-basics.tex, 1.5만 자)의 이론(최적성 조건, 볼록성의 특징, 강볼록·매끄러움, 하강 보조정리, 경사 하강의 수렴 속도, 사영 정리, 유한 정밀도의 한계)은 살리고, 빠진 이론(분리 정리, Farkas, 원뿔과 법선 원뿔, 부분기울기, 켤레 함수, KKT, Lagrange 쌍대와 강쌍대성)을 더한다. instardust 상자는 쓰지 않고 구현은 본문으로 옮긴다. 이 장은 1권의 수학 예비 장이므로 최적화 알고리즘 전체(선 탐색, 준뉴턴, 신뢰 영역, 최소제곱)는 9권이 맡는다. 여기서는 최적성 조건과 쌍대가 stardust 안에서 "멈춤 판정"과 "답의 증명서"로 구현되는 자리를 끝까지 따라간다.

## 이 장이 맡는 커널 (구현까지)
- optimize/lbfgsb.cpp `projgr`: 상자 제약의 사영 기울기 노름(KKT 잔차), `cauchy`(사영 경로의 일반화 Cauchy 점, 꺾이는 점들) — 이론은 여기, L-BFGS-B 전체는 9권
- optimize/linear.cpp `nnls_kernel`: Lawson–Hanson 활성 집합법(KKT로 유도), 들이기 판정 100 ulp, α = 2에서 시작; `lsq_linear`의 `optimality`·`active_mask`(KKT 잔차), 'bvls'
- regression/penalized.cpp: 라소·엘라스틱넷의 쌍대 간격(쌍대 점의 척도 s, 소거 없는 꼴), KKT 다듬기(부호 고정 선형계와 비활성 열의 |z^T Ω r| ≤ l1 검사), 좌표 하강의 한 걸음 = 연성 임계(soft thresholding) — 경로·교차검증은 12권
- optimize 결과의 `converged` 계약(BFGS ‖g‖∞ ≤ gtol, L-BFGS-B 사영 기울기 ≤ pgtol), minimize_scalar 'bounded'의 √ε|x| + xatol/3
- special.logsumexp(볼록성의 예, 최댓값 이동) — 짧게

## 절 배치 (세 축)
1. Minima and Optimality Conditions — T: 존재(Weierstrass, 강제적), 일계·이계 조건(증명) / A: 정류점 판정을 멈춤 규칙으로 / I: stardust의 `converged`는 판정의 뜻 그대로(BFGS ‖g‖∞, scale 문제), 그림: 안장점·최소점의 등고선
2. Convex Sets — T: 정의, 연산, 볼록 껍질(Carathéodory), 사영 정리, 분리·지지 초평면, Farkas, 원뿔·극원뿔·법선 원뿔 / A: 상자·반공간·공 사영 / 그림: 사영과 분리
3. Convex Functions — T: 정의, 에피그래프, Jensen, 일계·이계 특징(증명), 볼록성을 지키는 연산, 예(최소제곱, 로지스틱 음의 로그 우도, log-sum-exp, 노름), 로그 오목 / I: logsumexp의 최댓값 이동 / 그림: Jensen, 지지선
4. Subgradients and Conjugates — T: 부분미분, 0 ∈ ∂f, 합의 규칙(줄거리+참고), 노름의 부분미분, 근접 사상과 연성 임계, 켤레 함수·Fenchel 부등식 / A: 좌표 하강 한 걸음 / 그림: |x|의 부분미분, 연성 임계 함수
5. Constrained Optimality: KKT — T: 접원뿔, 볼록 문제의 KKT(Slater, 분리로 증명), 매끄러운 문제의 KKT(LICQ, 줄거리+참고), 상자 제약 / A: 사영 기울기 = KKT 잔차(projgr 유사코드) / I: L-BFGS-B pgtol, lsq_linear optimality, 그림: 상자 안의 법선 원뿔
6. Lagrange Duality — T: Lagrange 함수, 약쌍대, 강쌍대(Slater), 쌍대 간격 = 증명서, 라소의 쌍대 유도 / A: 쌍대 점 만들기(척도 s), 간격 계산 / I: penalized.cpp의 소거 없는 간격 식, 측정: 순진한 식 대 소거 없는 식 대 정확한 유리수
7. Active Sets: Nonnegative Least Squares — T: NNLS의 KKT, Lawson–Hanson의 유한 종료(줄거리와 증명) / A: 유사코드 / I: stardust nnls(dlarfgp, 100 ulp, α = 2), 측정: KKT 잔차, 그림: 활성 집합의 변화
8. Smoothness, Strong Convexity, and Gradient Methods — T: 하강 보조정리, 공동 강제성, 경사 하강 속도(증명), 사영 경사(KKT 고정점), 조건수 / A: 사영 경사 하강, 일반화 Cauchy 점 / I: lbfgsb `cauchy`의 꺾이는 점 / 그림: 지그재그, 걸음 수 대 κ(경사 하강·CG·stardust BFGS)
9. Limits of Finite Precision — T: √u 한계, 기울기로 u / I: minimize_scalar 'bounded' 허용오차, 측정(복제본) / 그림: f(1+h)의 계단
10. Testing Optimality Code, 11. APIs, 12. Concluding Remarks

## 그림
등고선(최소·안장), 사영·분리, Jensen·지지선, |x|의 부분미분·연성 임계, 상자의 법선 원뿔, 라소 간격(순진 대 소거 없음), NNLS 활성 집합, 경사 하강 지그재그, 걸음 수 대 κ, f(1+h) 계단

## 유사코드
상자 사영과 사영 기울기 노름(projgr), 좌표 하강 한 걸음(연성 임계), 쌍대 간격(척도 s), Lawson–Hanson NNLS, 사영 경사 하강, 일반화 Cauchy 점(꺾이는 점 정렬)

## 직접 얻을 수치(복제본; 참값은 Fraction·Decimal, mpmath 금지)
nnls KKT 잔차, lsq_linear optimality, 라소 dual_gap 대 정확한 간격(Fraction), 순진한 간격 식의 소거, BFGS·L-BFGS-B 걸음 수 대 κ, minimize_scalar의 정확도 대 기울기 방법의 정확도, f(1+h) 평평한 구간

## 진행 (2026-11-25)
- 초고 5.6만 자 → 검토 반영 뒤 약 7.5만 자, 절 12, 그림 10, 표 5, 알고리즘 10, 증명 20. 1권 338쪽 경고 0.
- 측정(복제본, 참값은 Fraction·Decimal): precision.py(평평한 구간 1.05e-8, e^x−2x에서 brent 1.2e-9·golden 2.1e-9·bounded 5.4e-7/1.4e-9·BFGS 1.6e-16), kappa.py(걸음 수 대 κ), zigzag.py, nnls_kkt.py·nnls_condP.py·nnls_traj.py(활성 집합 정답, 상대오차/κ(A_P)u 0.25–0.46), lasso_gap.py·gap_gram.py(순진한 식 바닥 4e-13 대 소거 없는 식 4e-15, Gram 경로 1.0e-15), box_kkt.py·box_ftol.py(기본 ftol은 사영 기울기 1.0e-3에서 멈춤), bvls_neg.py(optimality −1, trf 내부점), quad_exact.py, lse.py(lse(0,−40) → 0).
- 검토 반영: 윤문 57건(+2차), 수학 27항(둔각 방향, α ∈ (0,1], 강쌍대성의 행 독립, 로그 오목 단봉, 사영 기울기 내부 성분, Cauchy Δt 갱신, 이차식 정확 이유, 간격 소거 위치, 엘라스틱넷 증강 자료 유도 등), 깊이(멈춤 판정 표, ftol·stalled, logsumexp TwoSum·유사코드, Tseng 증명·참고, prox 비확대, Fenchel 등호, 지수족 공분산, 로그 오목 증명, 분리 정리 선분 원리, 감도 따름정리, 제약 이계 조건, 간격 바닥 uP·Gram 경로, KKT 다듬기·바깥 고리 유사코드, 분위 회귀의 쌍대 증명서, Frisch–Waugh 유도, 유한 종료 보강, nnls 구현 정정(scipy와 비트 동일 아님, w를 아래 행으로, 경계 처리, 반복 수), κ(A_P) 열, NNLS 궤적 그림, BVLS 유사코드, Kantorovich 증명, 사영 경사 되짚기 유사코드, Cauchy 비용·경계 처리, 4장 cref, 장 번호).
- LIBRARY-NOTES: L-BFGS-B 사영 기울기 미반환·기본 ftol, lsq_linear optimality 뜻·trf 내부점, logsumexp 상대 정확도, lasso dual_gap 바닥.
