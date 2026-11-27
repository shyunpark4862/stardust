# 3권 3장 Compactness and Connectedness — 얼개

예비 권(리뷰 방식). 3권 Part I Real Analysis의 셋째 장. 파일 `chapters/v3-03-compactness-and-connectedness.tex`, 장 라벨 `ch:compactness`(1·2·3권 라벨과 충돌이 없는지 grep으로 확인). `vol3.tex`에 `\input`을 더하고, 여러 장에 걸칠 기호만 Notation에 더한다.

먼저 읽을 것: `plans/v3-00-preparation.md`의 A.3, A.4, C의 3장 항목, D5, E; `plans/PROMISES.md` '## 3권' 절의 3장 줄 **전부**(1·2권에서 온 약속과 3권 1·2장이 3장 앞으로 한 약속 — 직접 읽고 하나도 빠뜨리지 않는다); 3권 2장(`chapters/v3-02-*.tex`) 전체(정의·기호·관례를 그대로 잇는다: 원소는 보통체, 폐포 \overline{S}, ℓ^∞(S), 𝓛(X, Y) 등)와 1장의 Cantor 집합(`def:cantor-set`, `prop:cantor-set`); 1권 `def:open-closed`·`thm:evt`·`thm:evt-rn`·`thm:heine-cantor`·`lem:heine-cantor-rn`·`thm:bolzano-weierstrass(-rn)`·`cor:interval-image`·`prop:intervals`·`def:connected-open`·`lem:exhaustion-compact`·note 'Functions with many discontinuities', 2권 `prop:standard-norms` 뒤 문단·5장 note 'The two components of the invertible matrices'·3장 note 'Bases of infinite-dimensional spaces'·8장 note(단위구가 컴팩트가 아님). 1·2권과 3권 1·2장이 증명한 것은 인용만 하고(`\vcref`, 같은 권은 `\cref`) 다시 증명하지 않는다. 일반 정리 뒤에 1권의 어느 정리가 특수 경우인지 한 줄로 밝힌다.

경계: 고른 수렴 일반론과 함수열은 4장, Arzelà–Ascoli(C(K)의 컴팩트 집합)·Weierstrass 근사·"어디서도 미분 불가능한 함수가 전형적"(Banach의 범주 논법)은 5장 — 여기서는 Baire 정리와 그 첫 응용까지. 함수해석의 큰 정리(열린 사상, 균등 유계성)는 이 책이 다루지 않는다(OUTLINE) — Baire의 응용으로 균등 유계성 원리를 **진술하고 증명해도 되는지**는 범위 밖이므로 Notes에서 이름과 문헌만. Hilbert 공간·컴팩트 연산자는 22장.

## Abstract (한 문단, 120단어 안팎, 동기부여)
1권에서 "닫힌 유계 구간 위의 연속 함수는 최댓값을 가진다", "고르게 연속이다"를 증명할 때마다 쓴 것은 한 가지 성질 — 유한한 것처럼 행동하는 무한 집합 — 이었다. 그 성질이 컴팩트성이고, 중간값 정리 뒤에 있던 것이 연결성이다. 두 개념에 이름을 주면 정리들이 R^n을 떠나 함수 공간에서도 성립하고, 왜 유한 차원에서는 모든 노름이 같은 수렴을 주는데(수치 계산이 노름 선택에 무심할 수 있는 이유) 무한 차원에서는 그렇지 않은지가 드러난다. 끝으로 Baire 정리: 완비 공간은 "얇은" 집합의 가산 합일 수 없다는 사실이 존재 증명의 새 도구가 된다는 점.

## 절 구성
3.1 Compact Sets — 열린 덮개와 컴팩트의 정의, 기본 성질(컴팩트 ⇒ 닫힘·유계, 컴팩트의 닫힌 부분집합, 유한 합집합, 유한 교차 성질과 축소 컴팩트 집합열의 교집합 — 3권 2장의 Cantor 교집합 정리와 비교), 예와 반례(R의 (0,1], 이산 공간, ℓ^2의 닫힌 단위 공은 닫히고 유계이나 컴팩트가 아님 — 단위 수열로 직접).

3.2 Sequential Compactness and Total Boundedness — 순차 컴팩트, 완전 유계(ε-그물), **거리 공간에서 컴팩트 ⇔ 순차 컴팩트 ⇔ 완비 + 완전 유계**(완전 증명; Lebesgue 수 보조정리 포함), 따름: 컴팩트 거리 공간은 분리 가능, 상대 컴팩트 ⇔ 완전 유계(완비 공간에서). **Heine–Borel**: R^n에서 컴팩트 ⇔ 닫힌 유계(1권 `thm:bolzano-weierstrass-rn` 인용으로; 1권 10장 약속), 1권 `lem:exhaustion-compact`가 덮개 논법의 특수형임을 한 줄. Cantor 집합이 컴팩트(1장 약속), 곱 공간의 컴팩트성(유한 곱은 완전 증명; 가산 곱은 대각 논법으로 완전 증명 — [0,1]^N과 Hilbert 입방체; Tychonoff 일반은 Notes).

3.3 Continuous Functions on Compact Sets — 컴팩트의 연속상은 컴팩트, **최대·최소 정리**(1권 `thm:evt`·`thm:evt-rn`이 특수 경우; 1권 5·10장 약속), 반연속 함수의 최솟값 존재(1장의 반연속 정의 인용; 1장이 13권 앞으로 한 예고와 맞춤), **Heine–Cantor**(컴팩트 위의 연속 ⇒ 고른 연속; 1권 두 정리가 특수 경우), **컴팩트에서 Hausdorff(거리) 공간으로의 연속 전단사는 위상동형**(2장 약속), 컴팩트 집합과 닫힌 집합 사이의 거리가 달성됨·양수임, 최근접점의 존재(유일성은 22장 몫이라고만), Dini 정리는 4장 몫.

3.4 Finite-Dimensional Normed Spaces — **유한 차원 노름 공간의 모든 노름은 동치**(완전 증명; 2권 1장·3권 2장 약속), 따름: 유한 차원 노름 공간은 완비, 유한 차원 부분공간은 닫힘, 유한 차원에서 나오는 선형 사상은 모두 유계(2장의 표준 노름 판을 일반화), 유한 차원에서 닫힌 유계 ⇔ 컴팩트, 행렬 노름 사이의 동치 상수(2권 `prop:norm-comparison` 인용)와 "수렴 여부는 노름과 무관, 속도 상수는 유관"이라는 계산적 의미. **Riesz 보조정리**와 **단위 공이 컴팩트 ⇔ 유한 차원**(완전 증명; 2권 8장 note를 정리로), 최선 근사의 존재(유한 차원 부분공간까지의 거리가 달성됨 — 12권 예고는 권 번호만).

3.5 Connectedness — 연결의 정의(열린 분리 없음; 2장이 약속한 "비연결"), R의 연결 부분집합은 구간(1권 `prop:intervals` 인용), **연결의 연속상은 연결**과 일반 중간값 정리(1권 `cor:interval-image`·`cor:ivt`가 특수 경우), 연결 집합의 합집합·폐포, 연결 성분(닫힘; 열린 집합의 성분은 R^n에서 열리고 가산 개 — R의 열린 집합은 서로소 열린 구간의 가산 합: 완전 증명, Part III가 씀), 경로 연결 ⇒ 연결, 역은 거짓(위상수학자의 사인 곡선: 완전 증명), **R^n(노름 공간)의 연결 열린 집합은 꺾은선 연결**이고 따라서 1권 `def:connected-open`과 일치(1권 10장 약속), 국소 상수 함수·`cor:zero-gradient` 재해석(인용), 완전 비연결(Q, Cantor 집합). R과 R^2가 위상동형이 아님(점 제거 논법).

3.6 Paths in Matrix Groups — **GL_n(R)의 두 경로 성분**(2권 5장 note의 줄거리를 완전 증명으로: det의 부호, 기본 행렬 분해 또는 극분해·QR로 경로 구성 — 2권의 어느 정리를 쓰는지 정확히 인용), GL_n(C)는 연결, SO(n)·O(n)의 성분(연결성 증명은 회전 표준형 2권 인용이 있으면 완전, 없으면 n = 2, 3만 완전하고 일반은 sketch로 — 판단을 보고에 적음), 가역 행렬이 M_n에서 열린 조밀 집합(2장 Neumann 인용 + 다항식의 영점), 대각화 가능 행렬의 조밀성(C 위; 2권 Schur 인용)과 그것으로 하는 "밀도 논법" 한 예(Cayley–Hamilton 재증명은 2권에 이미 있으면 인용만). 분량이 넘치면 이 절의 뒤 절반을 줄인다.

3.7 The Baire Category Theorem — 어디서도 조밀하지 않은 집합, 제1범주(meagre)·제2범주·잔여 집합, **Baire 정리**(완비 거리 공간에서 조밀 열린 집합의 가산 교집합은 조밀; 2장의 Cantor 교집합 정리로 완전 증명 — 2장 약속), 국소 컴팩트 판은 진술만. 응용(완전 증명): (i) R(완비 공간, 고립점 없음)은 비가산 — 3권 1장의 Cantor 논법과 비교, (ii) Q는 G_δ가 아님, (iii) **함수의 연속점 집합은 G_δ**(진동 함수)이고 따라서 **유리수에서만 연속인 함수는 없다**(1권 5장 약속; Thomae 함수는 1권 note 인용), (iv) 연속 함수열의 점별 극한은 조밀한 점들에서 연속(Baire 1급 함수; 따름: 도함수는 조밀한 집합에서 연속) — 증명이 길면 (iv)는 진술 + sketch, (v) 무한 차원 Banach 공간의 Hamel 기저는 비가산. 범주와 측도는 다른 크기 개념(잔여이면서 측도 0인 집합의 존재는 예고만, Part III).

3.8 Computation — 리뷰 방식의 계산 절(앞 장들과 같은 꼴·분량; 절 제목은 내용을 말하는 이름으로). 주제 후보: (i) 노름 동치 상수의 차원 의존 — R^n에서 ‖·‖_1/‖·‖_∞ 비가 n까지 벌어지는 것을 무작위·극단 벡터로, 반복법 정지 기준이 노름에 따라 달라지는 예(2장의 Jacobi 예와 이어도 좋다); (ii) 무한 차원의 비컴팩트성이 이산화에서 보이는 모습 — [0,1] 격자 위 "단위 공"의 ε-그물 크기(완전 유계의 ε-수)가 차원에 따라 지수적으로 크는 것(덮개 수의 간단한 상·하계와 수치), 또는 유한 차원 최선 근사의 존재와 비유일(‖·‖_∞·‖·‖_1); (iii) 컴팩트 위의 최대화에서 격자 탐색의 오차를 연속도로 보증(1권 `prop:grid-max`의 R^n판). stardust로 실행한 출력만 싣는다. 이상이 보이면 LIBRARY-NOTES.md에 기록만.

Notes — 역사와 문헌(검증된 것만): Heine 1872·Borel 1895·Lebesgue·Cousin의 덮개 정리, Fréchet 1906의 "compact", Alexandroff–Urysohn의 현대적 정의, Hausdorff 1914의 완전 유계, Tychonoff 정리(이름·문헌), F. Riesz 1918(단위 공 정리), 연결성 Jordan·Cantor·Hausdorff, Baire 1899 학위논문과 Osgood 1897, Banach–Steinhaus·열린 사상 정리가 Baire의 응용이라는 것(이름·문헌만), **Hamel 기저와 불연속 가법 함수**(D5: note 상자로 — 선택 공리로 R의 Q-기저를 잡아 f(x + y) = f(x) + f(y)인 불연속 f를 만든다는 구성, 그 그래프가 평면에서 조밀하다는 사실은 짧게 증명; 1권 `thm:cauchy-functional`과 2권 3장 note를 인용해 약속을 완결), 일반 위상(컴팩트 Hausdorff 공간)과의 관계, 문헌 안내. references.bib의 기존 키를 먼저 쓴다. 연습문제는 넣지 않는다(GUIDE).

## 분량과 주의
- 목표 50–60쪽.
- 증명 분류: 위에 "완전"이라 적은 것과 정리로 진술한 것은 완전 증명이 원칙. sketch는 `proof[sketch]`로 하고 보고에 적는다.
- 선택 공리: 순차 컴팩트 논법 등에서 가산 선택을 쓰는 곳은 3권 1·2장이 한 수준에 맞춘다(2장 검토에서 ℓ^∞ 비분리성은 선택 없이 증명하도록 고쳤다 — 같은 주의). Hamel 기저는 선택 공리를 쓴다고 명시.
- 미래 장은 GUIDE 규칙대로(3권 2장과 같게 "Chapter~N" 꼴) 가리키고, 새 약속은 `plans/PROMISES.md`에 적으며 OUTLINE.md의 그 장 범위와 맞는지 확인한다.
- 빌드: `tools/build.sh vol3`, warnings 0, ?? 없음.
