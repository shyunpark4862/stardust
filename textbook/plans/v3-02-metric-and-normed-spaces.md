# 3권 2장 Metric and Normed Spaces — 얼개

예비 권(리뷰 방식). 3권 Part I Real Analysis의 둘째 장. 파일 `chapters/v3-02-metric-and-normed-spaces.tex`, 장 라벨 `ch:metric-spaces`(1·2·3권 라벨과 충돌이 없는지 grep으로 확인하고 겹치면 바꾼다). `vol3.tex`에 `\input`을 더하고, 여러 장에 걸칠 기호만 Notation에 더한다(후보는 `plans/v3-00-preparation.md` E의 "거리와 노름"·"연산자").

먼저 읽을 것: `plans/v3-00-preparation.md`의 A.3, A.9, A.11, C의 2장 항목, E; `plans/PROMISES.md` '## 3권' 절의 2장 줄(받은 약속 전부 — 직접 읽고 빠뜨리지 않는다)과 3권 1장이 2장 앞으로 한 약속; 3권 1장(`chapters/v3-01-the-real-numbers.tex`)의 1.2–1.3절(완비화가 같은 구성임을 문장 수준에서 잇는다); 1권의 `def:open-closed` 주변, `prop:fixed-point-contraction`, 2권의 `prop:standard-norms`·note 'The p-norms'·note 'Linear operators in infinite dimensions'·`thm:neumann`·`thm:map-linearised`. 1·2권이 증명한 것은 `\vcref`로 인용하고 다시 증명하지 않는다. 일반 정리를 증명한 뒤 "1권의 어느 정리가 그 특수 경우"인지 한 줄로 밝힌다.

경계: 컴팩트·연결·Baire·노름 동치는 **다음 장**(쓰지 않는다; 필요한 예고는 "the next chapter"). 고른 수렴 이론은 4장 — 여기서는 C_b(X)의 sup 노름과 그 완비성(고른 극한의 연속성은 이 증명에 필요한 만큼 보조정리로)까지만. 수반·컴팩트 연산자·Hilbert 공간은 22장. L^p는 20장.

## Abstract (한 문단, 120단어 안팎, 동기부여)
1·2권의 해석학에서 쓰인 것은 수의 크기가 아니라 "거리"였다: ε-δ 논법은 |x − y|를 d(x, y)로 바꿔도 그대로 돈다. 거리 하나만 남기면 수열·함수·행렬·곡선이 모두 한 공간의 점이 되고, 미분방정식의 해나 반복법의 극한을 찾는 일이 "완비 공간에서 축약 사상의 고정점을 찾는 일" 하나로 모인다. 이 장은 그 언어(거리, 노름, 완비성)를 세우고 첫 수확으로 Banach 고정점 정리 — 오차 한계까지 주는 존재 정리 — 를 얻는다는 점. 뒤 권의 반복법 수렴 증명이 이 정리의 꼴이라는 점.

## 절 구성
2.1 Metrics and Norms — 거리 공간·노름 공간·내적 공간의 정의(내적 ⇒ 노름 ⇒ 거리; 2권 6장 인용), 부분공간, 곱 공간. 예: R^n의 p-노름, 이산 거리, ℓ^p(1 ≤ p ≤ ∞)와 c_0, 유계 함수 공간 B(S)의 sup 노름, C[a,b]의 ‖·‖_∞·‖·‖_1·‖·‖_2, 행렬 노름(2권 인용), Hamming 거리·그래프 거리(짧게), p진 거리(예로만; 초거리 부등식). 역삼각부등식, 노름의 볼록성과 단위 공의 모양(그림: p = 1, 2, ∞ 등).

2.2 The Inequalities of Hölder and Minkowski — Young 부등식은 1권 `cor:young` 인용. 유한 합과 급수의 Hölder·Minkowski(완전 증명, 등호 조건), 켤레 지수, p-노름의 단조성 ‖x‖_q ≤ ‖x‖_p(p ≤ q), lim_{p→∞}‖x‖_p = ‖x‖_∞, ℓ^p ⊂ ℓ^q. 적분 꼴(연속 함수의 Riemann 적분에 대해; C[a,b]의 ‖·‖_p가 노름임). 0 < p < 1에서는 노름이 아님. 2권 note 'The p-norms'의 약속을 완결한다고 명시.

2.3 Open and Closed Sets — 열린 공·닫힌 공(B(a, r), B̄(a, r); 1권 기호), 열린·닫힌 집합과 그 연산, 내부·폐포·경계·집적점·고립점·조밀·분리 가능(ℓ^p는 p < ∞에서 분리 가능, ℓ^∞는 아님 — 증명), 부분공간의 상대 위상, 수열 수렴과 폐포의 수열 특성화, 동치인 거리·동치인 노름의 정의(R^n의 표준 노름들은 2권 `prop:standard-norms`로 동치; 일반 정리는 다음 장), C[a,b]의 세 노름이 동치가 아님(예로 증명). 닫힌 공이 열린 공의 폐포가 아닐 수 있음(이산 거리).

2.4 Continuity — 점에서의 연속(ε-δ와 수열), **연속 ⇔ 열린 집합의 역상이 열림 ⇔ 닫힌 집합의 역상이 닫힘**(1권 2·5장의 약속), 합성, 고른 연속·Lipschitz·등거리·위상동형, 거리 함수 d와 dist(x, S)의 Lipschitz 연속성, 노름·덧셈·스칼라곱의 연속성, 곱 공간으로 가는 사상의 성분별 연속. 예와 반례(연속 전단사인데 역이 연속이 아님). Urysohn 꼴 보조정리(서로소 닫힌 집합을 가르는 연속 함수 dist 비)는 짧게.

2.5 Completeness — Cauchy 수열, 완비 거리 공간, Banach 공간. 기본 사실(수렴 ⇒ Cauchy, Cauchy + 수렴 부분열 ⇒ 수렴, 완비 공간의 닫힌 부분집합 ⇔ 완비 부분공간), R^n·C^n의 완비(1권 인용), **ℓ^p의 완비성**, **B(S)와 C_b(X)의 완비성**(고른 극한의 연속성 보조정리 포함), c_0, C[a,b]는 ‖·‖_∞에서 완비이나 ‖·‖_1에서 아님(구체적 Cauchy 수열로 증명), 노름 공간의 완비 ⇔ 절대수렴 급수가 수렴(증명), Cantor 교집합 정리(지름 → 0인 닫힌 집합의 축소열; 3권 1장의 축소 구간 꼴과 연결), 고른 연속 사상의 조밀 부분집합에서 완비 공간으로의 유일한 연속 확장(증명 — 2.6과 뒤 장들이 씀).

2.6 Completion — 임의의 거리 공간의 완비화의 존재(Cauchy 수열의 동치류; 3권 1장의 구성과 나란히, 단 거리 값이 R에 있으므로 이제 R을 쓸 수 있다는 차이를 밝힘)와 등거리 동형을 뺀 유일성(완전 증명). 노름 공간의 완비화는 Banach 공간, 내적 공간의 완비화는 Hilbert 공간(이름만 도입). 예: Q → R, (C[a,b], ‖·‖_1)의 완비화가 무엇인지는 Part III가 답한다는 예고(3권 1장의 약속 문구와 맞춤), p진수 Q_p는 Notes 또는 짧은 예.

2.7 The Contraction Mapping Theorem — **Banach 고정점 정리**(존재·유일, 선험·후험 오차 한계 d(x_n, x*) ≤ q^n/(1−q) d(x_1, x_0), d(x_n, x*) ≤ q/(1−q) d(x_n, x_{n−1}); 완전 증명), 변형: 어떤 거듭제곱 T^m이 축약이면 충분, 닫힌 공 위의 국소 꼴(T가 공을 자기 안으로 보내는 조건), 매개변수에 대한 고정점의 연속 의존. 가정이 빠지면 실패하는 예(d(Tx,Ty) < d(x,y)만으로는 부족; 완비가 아님). 1권 `prop:fixed-point-contraction`과 2권 `thm:map-linearised`가 특수 경우임을 한 줄씩. 응용 — 완전 증명 둘: (i) 제2종 Fredholm 적분방정식 f = g + λKf의 작은 |λ|에서의 유일 해와 Volterra 방정식(모든 λ; T^m 꼴), (ii) 선형 방정식의 Jacobi 반복(대각 지배 행렬; ∞-노름에서 축약) — 뒤 권의 반복법 예고. Lipschitz 조건이 존재 정리의 가정이 되는 이유(1권 5장 약속)를 여기서 말하고, 미분방정식의 Picard–Lindelöf는 8장 몫이라고만(장 번호는 글자로 쓰지 않고 "later in this part").

2.8 Bounded Linear Maps — 노름 공간 사이 선형 사상의 **유계 ⇔ 연속 ⇔ 0에서 연속**(증명), 연산자 노름과 그 성질(준승법성), 𝓛(X, Y) 표기(2권과 같은 글자, 여기서는 유계 선형 사상; preparation E의 추천), **Y가 완비이면 𝓛(X, Y)가 완비**, 쌍대 공간 X* 정의와 (ℓ^p)* ≅ ℓ^q (1 ≤ p < ∞; 완전 증명 — 길면 p = 1과 1 < p < ∞를 나누어) — 분량이 넘치면 진술과 p = 1만 증명하고 나머지는 Notes로 돌리며 그 판단을 보고에 적는다. 유한 차원에서 나오는 선형 사상은 (표준 노름에서) 항상 유계, 무한 차원의 **비유계 예: 미분 연산자 D**(C^1[0,1] ⊂ C[0,1], sup 노름; 2권 4장 note의 계산을 정리로)와 적분 연산자의 유계성, 값매김 범함수가 ‖·‖_∞에서는 유계이고 ‖·‖_1에서는 아님. **Neumann 급수**: Banach 공간에서 ‖T‖ < 1이면 I − T가 가역이고 역이 ΣT^k(2권 `thm:neumann`의 일반화), 가역 연산자 집합이 열려 있고 역 연산이 연속, 섭동 한계 ‖(A + E)^{-1} − A^{-1}‖(조건수와의 연결; 2권 인용). 2.7의 Fredholm 방정식을 Neumann 급수로 다시 봄.

2.9 Computation — 리뷰 방식의 계산 절(1·2권·3권 1장과 같은 꼴과 분량). 주제 후보: (i) 고정점 반복의 선험·후험 오차 한계를 실제 오차와 비교(cos x = x, 그리고 Jacobi 반복의 ∞-노름 축약; stardust의 선형대수 사용), (ii) p-노름이 p → ∞에서 ∞-노름으로 가는 것과 큰 p에서의 넘침(스케일링의 필요 — `sd.linalg.norm`의 동작 확인; 이상이 보이면 기록만), (iii) Fredholm 방정식의 Picard 반복을 구적으로 이산화해 Neumann 급수 부분합의 기하 수렴을 봄. stardust로 실행한 출력만 싣는다.

Notes — 역사와 문헌(검증된 것만): Fréchet 1906(거리 공간), Hausdorff 1914(이름 "metrischer Raum"과 완비화), Banach 1922 학위논문(노름 공간과 축약 사상 정리), Hölder 1889·Rogers 1888, Minkowski 1896, Riesz 1910(ℓ^p·L^p), Picard·Lindelöf의 축차 근사와의 관계, Caccioppoli 1930, p진수(Hensel)와 Ostrowski 정리(진술만), 다른 고정점 정리(Brouwer, Schauder — 이름과 문헌만), 문헌 안내(Rudin PMA, Kolmogorov–Fomin, Simmons, Kreyszig, Dieudonné 등 — references.bib에 이미 있는 키를 먼저 쓴다). 연습문제는 넣지 않는다(GUIDE).

## 분량과 주의
- 목표 50–60쪽.
- 증명 분류: "완전"이라 적은 것과 정리로 진술한 것은 모두 완전 증명이 원칙. sketch가 필요하면 `proof[sketch]`로 하고 보고에 적는다.
- 표기: 거리 공간 (X, d), 폐포는 \overline{S}(닫힌 공은 B̄(a, r)), int S, ∂S, diam S, dist(x, S), ‖f‖_∞ = sup|f|. 수열은 (x_n). 내적은 둘째 인자에 선형(2권 관례). ⊂는 진부분집합. 본문 분수는 사선, remark·example 환경 없음.
- 미래 장은 GUIDE의 규칙대로 가리킨다(3권 1장이 한 방식과 같게). 새 약속은 `plans/PROMISES.md`에 적는다.
- 빌드: `tools/build.sh vol3`, warnings 0, ?? 없음.
