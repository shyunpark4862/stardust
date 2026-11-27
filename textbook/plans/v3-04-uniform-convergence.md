# 3권 4장 Uniform Convergence — 얼개

예비 권(리뷰 방식). 3권 Part I Real Analysis의 넷째 장. 파일 `chapters/v3-04-uniform-convergence.tex`, 장 라벨 `ch:uniform-convergence`(1·2·3권 라벨과 충돌이 없는지 grep으로 확인하고 겹치면 바꾼다). `vol3.tex`에 `\input`을 더하고, 여러 장에 걸칠 기호만 Notation에 더한다(고른 수렴은 기호 없이 말로 쓴다 — ⇉는 쓰지 않는다).

먼저 읽을 것: `plans/v3-00-preparation.md`의 A.6, A.7, C의 4장·5장 항목(경계를 알기 위해), E; `plans/PROMISES.md` '## 3권' 절의 4장 줄 **전부**(1·2권에서 온 약속과 3권 2·3장이 4장 앞으로 한 약속 — 직접 읽고 빠뜨리지 않는다); 3권 2장의 `lem:uniform-limit`·C_b(X)의 완비성·M-판정 문단과 3장의 note 'Uniform convergence on compact sets'·맺음말(문장 수준에서 잇는다); 1권 8장의 note 'Uniform convergence'·거듭제곱 급수 절(`thm:radius`, `thm:termwise-diff`, `cor:termwise-int`, `thm:taylor-convergence` 뒤 문단)·note 'Complex power series', 1권 6장의 Weierstrass 함수 언급, 1권 11장의 매개변수 적분(`thm:diff-under-integral`, `def:uniform-convergence-integral`, `thm:parameter-improper`), 2권 12장 `lem:vector-functions`. 이미 증명된 것은 인용만 하고(`\vcref`, 같은 권은 `\cref`) 다시 증명하지 않는다. 일반 정리 뒤에 앞 권의 어느 결과가 특수 경우인지 한 줄로 밝힌다.

경계: Weierstrass 근사·Stone–Weierstrass·등연속·Arzelà–Ascoli·"어디서도 미분 불가능한 함수가 전형적"(Banach의 범주 논법)·매끄러운 bump 함수는 **5장**(쓰지 않는다). Fourier 급수의 수렴은 9장. 복소 거듭제곱 급수의 해석성은 12장(여기서는 실변수 증명; 값은 복소여도 되는 곳은 그렇게 쓴다). Lebesgue의 수렴 정리는 18장 — 이 장의 적분은 1권의 Riemann(Darboux) 적분이다. Riemann–Stieltjes는 6장.

## Abstract (한 문단, 120단어 안팎, 동기부여)
해석학의 계산은 거의 모두 극한 두 개의 순서를 바꾸는 일이다: 급수를 항별로 미분하고, 적분 기호 안에서 극한을 취하고, 이중 합의 순서를 바꾼다. 점별 수렴만으로는 이 모든 것이 실패할 수 있고(연속 함수의 극한이 불연속, 적분의 극한이 극한의 적분과 다름), 19세기 해석학이 찾은 답이 고른 수렴이다. 3권 2장의 언어로는 sup 노름에서의 수렴일 뿐이라는 점. 수치 계산에서 "함수를 급수의 부분합으로 바꿔 쓴다"는 모든 일의 오차가 sup 노름으로 재어진다는 점. 장의 끝에서 고른 수렴이 만드는 낯선 대상 — 어디서도 미분할 수 없는 연속 함수 — 을 만든다는 점.

## 절 구성
4.1 Pointwise and Uniform Convergence — 점별 수렴과 고른 수렴의 정의(집합 S 위 함수열, 값은 R·C 또는 노름 공간), sup 노름과의 동치(3권 2장 ℓ^∞(S)), 고른 Cauchy 판정(2장 완비성 인용), 국소 고른 수렴·컴팩트 위 고른 수렴(3장 note와 연결), 기본 예와 그림(x^n, nx e^{−nx}, 움직이는 혹, sin(nx)/n), 고른 수렴의 연산(합, 유계 함수의 곱, 합성), 수열 판정(sup|f_n − f| ↛ 0을 보이는 법). 극한 교환이 실패하는 반례 모음을 표로(연속성, 적분, 미분, 유계성, 호의 길이).

4.2 Continuity of the Limit — 고른 극한의 연속성은 2장 `lem:uniform-limit` 인용, 일반화: 극한의 교환 lim_n lim_{x→a} = lim_{x→a} lim_n(Moore–Osgood 꼴; 이중 수열 판도 — 완전 증명), 고른 연속의 보존, **Dini 정리**(3장 약속; 컴팩트 + 단조 + 연속 극한 ⇒ 고른; 완전 증명과 가정마다의 반례), 응용: √x를 다항식으로 근사하는 단조 반복(5장의 Stone–Weierstrass가 쓸 수 있도록 결과로 남김 — 5장 예고는 "Chapter~5" 꼴).

4.3 Integration and Differentiation of the Limit — **고른 극한의 Riemann 적분 가능성과 ∫의 교환**(완전 증명; 1권 Darboux 판정 인용), 부정적분의 고른 수렴, 이상 적분에서는 고른 수렴만으로 부족하다는 반례와 지배 조건(1권 `thm:parameter-improper`와 맞춰 "고른 수렴 + 적분 가능한 지배 함수" 꼴의 정리; Lebesgue 판은 Part III라고만), **항별 미분 정리**(f_n'이 고르게 수렴하고 한 점에서 f_n이 수렴; 완전 증명 — 평균값 정리 판으로, f_n'의 연속성을 가정하지 않는 꼴), C^k 판과 C^1[a,b]가 노름 ‖f‖_∞ + ‖f'‖_∞에서 완비라는 따름(2장의 "D는 비유계"와 대비: 닫힌 연산자라는 말은 Notes), 유계 수렴 정리(Arzelà의 Riemann 판)는 진술과 Notes(증명은 Part III의 지배 수렴이 준다고 예고).

4.4 Series of Functions — 함수 급수의 고른 수렴, **Weierstrass M-판정**(일반 판; 2권 `lem:vector-functions`와 2장의 C_b 판이 특수 경우), 고른 수렴에 대한 Abel·Dirichlet 판정(부분합 공식으로 완전 증명; Σ sin(nx)/n이 [δ, 2π − δ]에서 고르게 수렴하지만 0 근처에서는 아님 — 톱니파, Gibbs는 9장), 항별 적분·미분 따름, 예: Σ x^n/n², ζ(s) = Σ n^{−s}가 s > 1에서 C^∞이고 항별 미분 가능, 급수로 정의한 exp·sin·cos의 재확인(1권 인용), 무한 곱의 고른 수렴은 짧게(로그로 환원; sin의 곱 공식은 뒤 장 몫).

4.5 Double Sequences and Double Series — 이중 수열의 반복 극한과 이중 극한, **이중 급수의 순서 교환**(절대 수렴 ⇒ 모든 재배열·반복 합이 같음; 완전 증명 — 1권 11장 note가 측도론으로 미룬 것을 초등적으로), 비음 항의 Tonelli 꼴(확장 실수 값; 1장 인용), Cauchy 곱(1권 `thm:cauchy-product` 인용)과의 관계, 반례(반복 합이 다른 이중 급수), 응용: ζ 값의 항등식 하나나 Lambert 급수 같은 고전적 재배열.

4.6 Power Series Revisited — 거듭제곱 급수는 수렴 구간의 컴팩트 부분집합에서 고르게 수렴(1권 `thm:radius` 인용 + M-판정)하고 1권의 항별 미분·적분 정리가 4.3의 따름으로 다시 나옴(한 문단), **중심 옮기기: 합은 수렴 구간의 모든 점에서 해석적**(이중 급수 교환으로 완전 증명; 1권 8장 약속; 새 수렴 반지름 ≥ R − |c − a|), 해석 함수의 영점은 고립·항등 정리(실변수 판; 3장의 연결성 사용), **Abel 정리**(끝점에서 수렴하면 합이 그 점까지 연속이고 [0, R]에서 고른 수렴; 완전 증명; 1권 8장 note 약속), 응용: ln 2 = Σ(−1)^{n+1}/n, π/4 = Leibniz 급수(1권이 다른 방법으로 얻었으면 인용해 비교), Cauchy 곱에 대한 Abel의 정리(세 급수가 모두 수렴하면 곱이 맞음), Abel 합과 Cesàro 합의 관계(Frobenius 정리: Cesàro ⇒ Abel; 증명), Tauber 정리(a_n = o(1/n) 꼴; 완전 증명 — 짧다)와 Hardy–Littlewood는 Notes.

4.7 Continuous Nowhere Differentiable Functions — **Takagi 함수(또는 van der Waerden 꼴)가 연속이고 어디서도 미분 불가능**(완전 증명; 그림), **Weierstrass 함수 Σ a^n cos(b^n πx)의 연속성**(M-판정; 1권 6장 약속)과 미분 불가능성: Weierstrass의 원래 조건(ab > 1 + 3π/2, b 홀수)에서의 증명을 완전하게 쓸 수 있으면 완전, 길면 `proof[sketch]`(판단을 보고에 적음), Hardy 1916의 최종 조건(0 < a < 1, ab ≥ 1)은 진술과 문헌. 고른 극한에서 미분 가능성이 보존되지 않는다는 4.3의 반례를 이 함수들이 극단으로 보여 준다는 연결. "전형적인 연속 함수가 이렇다"는 5장 예고. 공간을 채우는 곡선(Peano·Hilbert; 고른 극한으로 구성)은 구성과 연속성·전사성 증명을 넣을 분량이 되면 넣고(완전), 아니면 Notes.

4.8 Computation — 리뷰 방식의 계산 절(앞 장들과 같은 꼴·분량; 절 제목은 내용을 말하는 이름으로). 주제 후보: (i) 부분합의 sup 오차를 격자에서 재고 M-판정의 꼬리 한계와 비교(예: Σ sin(nx)/n²는 고르게, Σ sin(nx)/n은 0 근처에서 고르지 않게 — 최대 오차가 줄지 않는 모습), (ii) 항별 미분의 위험: 부분합의 도함수가 발산하는 급수를 수치 미분했을 때, (iii) 끝점 근처에서 거듭제곱 급수의 느린 수렴과 Abel 정리(ln 2를 x → 1^−로 접근) 및 간단한 가속(Euler 변환 — 1권 `prop:richardson` 등 기존 결과가 있으면 인용), (iv) Takagi 함수의 차분 몫이 수렴하지 않는 것을 표로. stardust로 실행한 출력만 싣는다(난수를 쓰면 seed를 본문에 적는다; 인쇄한 모든 자리는 반올림 뒤 원값과 다시 대조). 라이브러리 이상은 LIBRARY-NOTES.md에 기록만.

Notes — 역사와 문헌(검증된 것만): Cauchy 1821의 "연속 함수 급수의 합은 연속"과 Abel 1826의 반례 지적, Seidel 1847·Stokes 1847, Weierstrass의 강의에서의 고른 수렴(gleichmäßige Konvergenz)과 M-판정, Dini 1878, Arzelà의 유계 수렴 정리 1885, Moore–Osgood, Abel 1826(이항 급수 논문)의 극한 정리, Tauber 1897, Hardy–Littlewood, Weierstrass 1872 발표 함수와 Bolzano의 앞선 예, Takagi 1903·van der Waerden 1930, Hardy 1916, Peano 1890·Hilbert 1891. 문헌 안내(Rudin PMA 7장, Apostol, Knopp, Hardy *Divergent Series*, Titchmarsh, Krantz–Parks 등 — references.bib의 기존 키를 먼저 쓴다). 연습문제는 넣지 않는다(GUIDE).

## 분량과 주의
- 목표 50–60쪽.
- 증명 분류: "완전"이라 적은 것과 정리로 진술한 것은 완전 증명이 원칙. sketch는 `proof[sketch]`로 하고 보고에 적는다.
- 표기·관례는 3권 2·3장을 잇는다(원소는 보통체, ‖f‖_∞ = sup|f|, ℓ^∞(S), 수열 (f_n)은 0부터 또는 (f_n)_{n≥1}로 명시). 본문 분수는 사선, remark·example 환경 없음, 행렬은 bmatrix.
- 미래 장은 "Chapter~N" 꼴로 가리키고 OUTLINE.md의 그 장 범위와 맞는지 확인한다. 새 약속은 `plans/PROMISES.md`에 적는다.
- 빌드: `tools/build.sh vol3`, warnings 0, ?? 없음.
