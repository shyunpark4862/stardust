# 1권 11장 Integral Calculus in Several Variables — 얼개

예비 권(리뷰 방식). Part II Calculus. 7장(Riemann 적분, 이상 적분, 구적)과 10장(R^n의 극한·연속, 미분, Jacobian, 좌표 변환)을 쓴다. plans/PROMISES.md의 11장 약속을 모두 지킨다(Gauss 적분, Γ(1/2) = √π, 회전체 부피의 정당화, 구면 좌표의 Jacobian r² sin θ 사용 등). 엄밀한 측도론적 적분은 2권이므로 여기서는 Riemann 적분의 수준에서 정확하게 쓴다.

## Abstract (한 문단, 120단어 이내, 동기부여)
확률은 밀도의 적분이고 변수가 여럿이면 적분도 여러 겹이다. 정규 분포의 상수 √(2π)가 어디서 오는지, 변수를 바꾸면 왜 Jacobian이 붙는지가 뒤 권의 확률·통계 계산의 바탕이라는 점.

## 절 구성
11.1 Double Integrals over Rectangles — 분할과 상합·하합(7장을 2차원으로), 적분 가능성, 연속 함수는 적분 가능(진술; 증명은 7장과 같은 꼴임을 보이는 줄거리 또는 완전 증명). 그림: 직사각형 분할과 기둥.
11.2 Iterated Integrals and Fubini's Theorem — 반복 적분, Fubini 정리(연속 함수판; 증명), 적분 순서가 결과를 바꾸는 반례(절대 수렴하지 않는 경우; 2권 예고), 적분 기호 아래의 미분(Leibniz 규칙; 증명).
11.3 Integrals over General Regions — 넓이가 0인 집합(용량 0), Jordan 가측 영역, 기본 영역(x-단순, y-단순)과 그 위의 적분, 적분 순서 바꾸기(예와 그림), 넓이·부피·질량 중심.
11.4 Triple and Multiple Integrals — n겹 적분, n차원 부피, 단체(simplex)의 부피 1/n!(증명; 뒤 권의 순서통계량 암시), 회전체 부피의 정당화(7장 약속).
11.5 Change of Variables — 선형 변환과 행렬식이 부피를 늘리는 비율(2×2는 증명, 일반은 Part III 약속), 변수 변환 공식(진술; 증명은 줄거리+문헌, 2변수 C¹ 변환의 핵심 아이디어는 본문에), 극좌표·원기둥·구면 좌표(10장의 Jacobian 인용), 그림: 작은 사각형이 평행사변형으로.
11.6 The Gaussian Integral — ∫e^{−x²} dx = √π(극좌표 증명; 이상 중적분의 정당화 포함), Γ(1/2) = √π(7장 약속), 정규 밀도의 정규화 상수, n차원 Gauss 적분과 이차 형식(대각 행렬 꼴은 증명, 일반 양정치는 Part III 약속), n차원 공의 부피 공식(증명; Γ로).
11.7 Improper Multiple Integrals — 유계가 아닌 영역·함수, 소진열로 정의, 음이 아닌 함수에서는 잘 정의됨(증명), 절대 수렴, Beta 함수 B(a,b) = Γ(a)Γ(b)/Γ(a+b)(증명; 변수 변환으로), Dirichlet 적분은 상자.
11.8 Integrals Depending on a Parameter — 연속성과 미분 가능성(이상 적분판은 고른 수렴 조건과 함께 진술, 증명은 유계 구간판), 감마 함수의 미분 가능성, 응용: ∫ sin x / x dx = π/2(매개변수 미분으로; 증명 또는 줄거리).
11.9 Numerical Integration in Several Variables: A First Look — 곱 규칙(사다리꼴·Simpson의 텐서곱)과 그 오차, 차원의 저주(점 수 N^d; 표: 차원별 필요한 점 수), Monte Carlo 적분의 생각과 오차 1/√N(확률은 3권 예고; 여기서는 수치 실험 표를 직접 계산), 알고리즘: 곱 사다리꼴(유사코드). 순수 수학으로, 뒤 권(11권) 암시만.

## note 상자 후보
- Lebesgue 적분과 Fubini–Tonelli(2권 예고)
- 미분 형식은 넣지 않는다
- Dirichlet 적분(단체 위의 x^{a−1}y^{b−1}… 꼴; 뒤 권의 Dirichlet 분포 예고)

## Notation
GUIDE §2대로 꼭 필요한 것만: ∬_R f dA 또는 ∫_R f(x) dx, vol(S), B(a,b)(Beta 함수). 한 절에서만 쓰는 것은 싣지 않는다.

## 분량
제한 없음. 55쪽 안팎 예상.
