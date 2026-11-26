# 3권 1장 The Real Numbers — 얼개

예비 권(리뷰 방식). 3권(Analysis) Part I Real Analysis의 첫 장이자 3권의 첫 장. 파일은 `chapters/v3-01-the-real-numbers.tex`, 장 라벨 `ch:real-numbers-construction`(1권 3장의 라벨과 겹치지 않게; 쓰기 전에 1·2권 라벨과 충돌이 없는지 grep으로 확인). `vol3.tex`에 Part I "Real Analysis"와 이 장을 넣고, Notation은 이 장에서 실제로 쓰는 것만 최소로 시작한다(뒤 장이 채운다).

1권은 완비성을 공리로 두고(`asm:reals`, 1권 3장) 그 결과를 모두 증명했다. 이 장은 **그 공리를 정리로 바꾼다**. 1·2권이 증명한 것은 `\vcref{1}{...}`로 인용하고 다시 증명하지 않는다. 먼저 `plans/v3-00-preparation.md`의 A.0–A.2, C의 1장 항목, E(표기)를 읽고, `plans/PROMISES.md`의 '## 3권' 절 1장 줄을 읽어 약속을 빠뜨리지 않는다. 1권 3장의 note 'Dedekind cuts and Cauchy sequences'와 5장의 note 'Equivalent forms of completeness', `thm:r-unique`(줄거리)를 직접 읽고 그 글과 이어지게 쓴다.

지킬 약속:
- 1권 3장·5장 본문이 알린 "Volume III가 Cauchy 수열로 완비성을 동치로 말한다" — 완비성의 동치 정리(완전 증명).
- 2권 15장 맺음말 "Its real analysis proves the completeness of the real numbers" — R의 구성과 완비성 증명(완전).
- `thm:r-unique`(1권은 줄거리) — 유일성의 완전한 증명.
- D1: Cauchy 수열 구성은 완전, Dedekind 절단은 줄거리(`proof[sketch]`).

논리의 순환을 피한다: 구성 절에서는 1권의 결과 가운데 **Q와 순서체 일반에 대해 성립하는 것만** 쓴다(1권 3장이 Q를 어떻게 다뤘는지 확인해 인용). R의 성질을 쓰는 1권 정리(단조 수렴, BW, Cauchy 판정법 등)는 구성이 끝나고 "구성한 체가 `asm:reals`를 만족하므로 1권의 모든 결과가 이 R에 대해 성립한다"고 밝힌 뒤에만 인용한다. 이 점을 장 첫머리에서 독자에게 분명히 말한다.

## Abstract (한 문단, 120단어 안팎, 동기부여)
1·2권의 해석학은 모두 한 공리 — 완비 순서체가 있다 — 위에 서 있었다. 그 공리가 정리임을 보이는 것이 해석학의 첫 일이다: 유리수만으로 실수를 만들고, 그렇게 만든 것이 하나뿐임을 보인다. 같은 구성(Cauchy 수열의 동치류)이 다음 장에서 임의의 거리 공간을 완비화하고, 뒤에서 L^p 공간을 만든다는 점. 완비성이 여러 얼굴(상한, 단조 수렴, 축소 구간, BW, Cauchy)을 가지며 어느 것이 어느 것과 같은지가 Archimedes 성질에 달려 있다는 점.

## 절 구성
1.1 Ordered Fields — 순서체의 정의는 1권 인용. 순서체 일반에서 성립하는 것: Q가 유일하게 들어감(소체, 순서를 보존하는 유일한 매장 Q → F; 증명), 절댓값·삼각부등식, 순서체에서의 수열 수렴·Cauchy 수열·유계의 정의(ε을 F의 양의 원소로), Archimedes 성질의 동치 꼴(N이 유계가 아님 ⇔ 1/n → 0 ⇔ Q가 조밀; 증명). 예: Q, 유리 함수체 Q(x)의 비Archimedes 순서(완전한 확인; 1.5의 반례로 다시 씀).

1.2 The Construction from Cauchy Sequences — Q의 Cauchy 수열 전체의 환 𝒞, 영수열의 이데알 𝒩, R ≔ 𝒞/𝒩. 잘 정의됨, 가환환, 체(0이 아닌 류의 대표가 결국 0에서 떨어져 있음 — 보조정리), 순서(양의 류: 결국 어떤 양의 유리수보다 큼; 삼분법과 연산과의 양립), Q의 매장과 조밀성, Archimedes. 모든 단계 완전 증명. 환·이데알이라는 낱말은 쓰되 대수의 사전 지식을 전제하지 않고 직접 확인한다.

1.3 Completeness — 구성한 R에서 (i) 모든 Cauchy 수열이 수렴(유리수 근사의 대각 논법; 완전), (ii) 상한 성질(이분법으로 Cauchy 수열을 만들어; 완전). 따라서 `asm:reals`가 정리가 된다는 것을 정리로 진술하고, 여기서부터 1권의 결과를 자유롭게 인용한다고 밝힌다. 소수 전개와의 관계는 1권 `thm:real-expansion` 인용으로 한 문단.

1.4 Dedekind Cuts — 절단의 정의, 순서(포함), 덧셈, 곱셈의 정의(부호 경우 나눔), 상한 = 합집합. 진술은 정확히, 증명은 `proof[sketch]`(상한 성질은 완전하게 써도 짧으므로 완전 증명; 체 공리 검증이 줄거리). 두 구성의 비교: 절단은 순서를 쓰고 Cauchy 수열은 거리를 쓴다 — 뒤의 것이 일반화된다(2장 완비화 예고, 미래 장은 장 번호를 글자로 쓰지 않고 "the next chapter").

1.5 Uniqueness — 완비 순서체는 순서체 동형을 빼고 하나이고 그 동형도 하나뿐(완전 증명: Q 위의 매장에서 출발해 상한으로 확장, 덧셈·곱셈·순서 보존, 전단사; 유일성은 순서 보존에서). 따름: R의 체 자기동형은 항등뿐(제곱이 양수 ⇒ 순서 보존; 완전). 1권 `thm:r-unique`의 줄거리를 완결한다고 명시.

1.6 The Forms of Completeness — 순서체 F에 대해 다음을 정리로: (a) 상한 성질, (b) 단조 수렴, (c) 축소 구간(닫힌 유계 구간의 임의 감소열의 교집합이 공집합이 아님) + Archimedes, (d) Bolzano–Weierstrass, (e) Cauchy 완비 + Archimedes 가 모두 동치(완전 증명; 1권 5장 note의 원을 닫음. 어느 함의가 1권의 증명을 그대로 읽은 것인지 밝히고 나머지를 증명). (a), (b), (d)는 Archimedes를 함의하지만 Cauchy 완비와 축소 구간(길이 → 0 조건 꼴)은 함의하지 않음 — 정확한 진술을 문헌으로 확인(Propp, "Real analysis in reverse", Amer. Math. Monthly 2013; Teismann 2013 등)하고, 반례는 형식 Laurent 급수체 Q((x)) 또는 R((x))가 Cauchy 완비이지만 비Archimedes임을 **완전히 증명**(짧다), 그 밖의 반례(초실수, 초현실수)는 Notes. 이 절에 쓰이는 연결성·IVT 꼴의 동치(IVT ⇔ 완비)는 한 명제로(증명 가능하면 완전, 아니면 Notes).

1.7 The Extended Real Line and Upper and Lower Limits — [−∞, ∞]의 순서와 연산 규약(0·∞는 여기서 정하지 않고 측도론 부에서 정한다고만), 모든 부분집합이 sup·inf를 가짐. limsup·liminf: 정의는 1권 `def:limsup` 인용, 성질을 체계적으로(부분열 극한 집합의 최대·최소, ε 특성화, 수렴 ⇔ 두 값 일치, 부등식 limsup(a_n + b_n) ≤ …와 등호 조건, 곱, 단조 함수와의 교환, Cesàro 평균과 Stolz–Cesàro 꼴: liminf a_n ≤ liminf σ_n ≤ limsup σ_n ≤ limsup a_n, 비·근 판정의 관계 liminf a_{n+1}/a_n ≤ liminf a_n^{1/n} ≤ …; 1권 8장에 이미 있는 것은 인용). 함수의 limsup·liminf와 상·하반연속은 정의와 기본 성질만(뒤 장이 씀).

1.8 Cardinality of the Continuum — 1권 2장(가산성, Cantor, R의 비가산, Schröder–Bernstein은 줄거리였음)을 인용. 여기서: Schröder–Bernstein의 완전한 증명(1권이 줄거리였는지 확인하고, 줄거리였으면 여기서 완결), |R| = |2^N| = |R^n| = |R^N|, 연속 함수 R → R의 집합의 기수는 연속체, 모든 함수의 집합은 그보다 큼. 완전 집합과 Cantor 집합(삼진 전개로 정의, 닫힘·내부 없음·비가산·길이 0; Cantor 함수는 6장 몫이므로 이름만). 대수적 수는 가산이므로 초월수가 존재(Cantor), Liouville 수의 초월성(Liouville 정리 완전 증명 — 짧다), e·π의 초월성은 Notes(인용).

1.9 Computation — 리뷰 방식의 계산 절(1·2권 장들의 계산 절과 같은 꼴과 분량, 브리프대로). 주제: 유리수 Cauchy 수열로서의 실수 — Heron 반복(1권 `prop:heron`)으로 √2의 유리 근사열과 Cauchy 계수(modulus), 부동소수점 수는 Q의 유한 부분집합이며 순서체도 아니라는 점을 작은 실험으로(결합 법칙 실패, 1 + u = 1; 자세한 것은 Volume VI), limsup의 수치 추정 한 예. stardust로 실행한 출력만 싣는다(공용 빌드, 브리프의 규칙). 정확한 유리수 산술은 python `fractions`를 써도 된다.

Notes — 역사와 문헌(검증된 것만): Cantor 1872·Méray 1869·Heine 1872의 Cauchy 수열 구성, Dedekind 1872 *Stetigkeit und irrationale Zahlen*, Hilbert 1900의 공리적 정의, Landau *Grundlagen*, 구성 대안(Eudoxus 실수 등은 이름만), 역해석(reverse) 관점 Propp, 비Archimedes 체·비표준 해석(Robinson)·초현실수(Conway), 구성적 해석(Bishop)의 실수 = modulus를 가진 Cauchy 수열, 계산 가능 실수(Turing 1936; 5권 예고는 권 번호만), 연속체 가설(Gödel, Cohen; Excursions는 언급하지 않는다), 초월수(Hermite 1873, Lindemann 1882, Gelfond–Schneider). 1권 references.bib에 이미 있는 키는 다시 만들지 않고 그대로 쓴다(키 충돌 금지; 새 키는 grep으로 확인).

Exercises — 브리프의 분량·꼴대로.

## 분량과 주의
- 목표 45–55쪽(1·2권 장들과 같은 밀도).
- 증명 분류: 위에 "완전"이라 한 것은 모두 완전 증명, Dedekind 체 공리만 sketch.
- 표기: ℕ은 0 포함, 수열 (a_n), ⊂는 진부분집합, 본문 분수는 사선, remark·example 환경 없음. 동치류는 [(a_n)] 꼴, 구성 중의 체는 일단 다른 글자(예 𝐑 대신 \mathcal R 등으로 하지 말고 "R"을 정의로 도입하되 1.3 끝에서 이것이 1·2권의 ℝ이라고 밝힘 — 글자는 처음부터 \mathbb R).
- 권 간 참조는 `\vcref{1}{label}`·`\Vcref`·`\veqref`; 같은 목록에 이 권 라벨과 섞지 않는다(GUIDE §3).
- 빌드: `tools/build.sh vol3`(vol1·vol2의 aux가 build/에 있어야 한다; 없으면 `tools/build.sh all`). warnings 0.
