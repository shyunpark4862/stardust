# 7장 Combinatorics and Generating Functions — 계획

GUIDE.md와 CHECKLIST.md를 다시 읽고 쓴다(2026-11-25, 바뀐 것 없음). 옛 7장(legacy/chapters/07-combinatorics.tex, 5.9만 자)은 이론과 알고리즘이 이미 충실하다(알고리즘 4, 증명 약 12). 그러나 instardust 상자 일곱 개가 구현을 맡고 있고 그림이 하나, 라이브러리 측정이 없다. 새 기준으로 바꾸는 일은 (1) 상자를 본문의 Implementation 단락과 유사코드로 옮기고, (2) 그림을 열 개 안팎으로 늘리고, (3) 복제본으로 p값과 계수를 정확한 정수와 견주고, (4) 확률 생성 함수의 모멘트와 정규 근사, 무작위 순열(Fisher–Yates)처럼 빠진 이론과 커널을 더하는 것이다. 116장(순위 검정), 117장(ks_2samp), 118장(kendalltau), 119장(fisher_exact)은 각 API의 인자·대립가설·근사·동점 정책을 맡고 이 장은 세는 수학과 그 커널을 맡는다.

## 이 장이 맡는 커널 (구현까지)
- htest/rank_exact.hpp: binom_u128(공약수 먼저), qbinom_step·mwu_lower(곱 공식, 꼬리에서 자름, 128비트), Ansari–Bradley 점수 반쪽, grouped_counts(동점의 두 변수 DP, double-double, dd_axpy NEON), signed_rank_counts·signed_rank_lower(부분집합 합, 대칭 반쪽, NEON, 스레드), page_row_counts(비트마스크 DP, 64비트), 정수 경로의 경계 2^124
- htest/kendall.hpp: Mahonian 수(정수 미끄러지는 창 n ≤ 33, double-double 블록 창), 2^±600 척도, 대칭 꼬리, 일의 상한
- htest/ks2.hpp: 흡수 경계 DP(정수 경계 g = iα − jβ, 띠), 동점 덩어리의 초기하 갈래(hypergeometric: 최빈값에서 비, double-double), equal_tail(반사 원리의 닫힌 꼴, 비의 곱)
- random: Fisher–Yates(첫 k개), 균등성 증명
- 정규 근사로 넘어가는 문턱(일의 양을 미리 셈, H13)

## 절 배치 (세 축)
1. Basic Counting — T: 곱의 법칙, 이항계수, Pascal·Vandermonde, 초기하와 단봉성(최빈값 증명) / A: 공약수 이항계수, 최빈값에서 비 / I: binom_u128, hypergeometric(ks2)
2. Generating Functions — T: 형식적 멱급수, 역원, 잘라내기 준동형, 분할, 확률 생성 함수와 모멘트(증명) / A: 인수 곱하기·나누기 / 그림: 분할 수
3. Gaussian Binomial Coefficients — T: q-Pascal, 조합적 해석(Young 그림), Mann–Whitney 분포, q-이항 정리, Ansari–Bradley / A: 곱 공식과 꼬리 자름 / I: qbinom_step, 128비트 경계, 표, 그림(Young 그림, U의 분포)
4. Subset Sums — T: 부호 순위 / A: 부분집합 합, 대칭 반쪽 / I: signed_rank_*, NEON·스레드·비트 / 그림
5. Inversions — T: Mahonian, Lehmer 부호 / A: 미끄러지는 창, 블록 창 / I: kendall.hpp, 측정(차를 쓰는 창 대 블록 창의 꼬리 오차) / 그림
6. Ties — T: 두 변수 생성 함수, 비대칭 / A: grouped_counts DP, 비트마스크 DP(Page) / I
7. Lattice Paths — T: KS 격자 경로, 반사 원리, 양쪽 포함–배제 / A: 흡수 경계 DP, 닫힌 꼴의 비 곱 / I: ks2.hpp, 띠 / 그림: 반사, 띠
8. Normal Approximation and Work Bounds — T: 모멘트, CLT(줄거리) / I: 일의 상한과 근사 / 그림: 정확 대 근사
9. Random Permutations — T: Fisher–Yates 균등성 / A·I: random의 첫 k개 셔플
10. Testing, 11. APIs, 12. Concluding Remarks

## 그림
Young 그림과 낱말, U 분포와 정규 곡선, T+ 분포, Mahonian 분포, 블록 창 대 차의 창 오차, 반사 원리, KS 띠, 정확 대 정규 근사 꼬리, 분할 수 성장, (일의 양)

## 직접 얻을 수치(복제본, 정확한 정수·유리수)
sd.htest.mannwhitneyu·wilcoxon·kendalltau·ks_2samp·page_trend_test·ansari의 정확 p값 대 정수 참값, 창 두 방식의 오차, 정규 근사 오차

## 진행 (2026-11-25)
- 초고 5.2만 자 → 검토 반영 뒤 약 7.3만 자, 절 12, 그림 12, 표 5, 알고리즘 15, 증명 21.
- 측정(복제본, 참값은 파이썬 정수·Fraction): library.py(동점 없는 정확 p값 4종, 2.2e-16 이하, 꼬리 8.8e-34까지), windows.py·windows2.py(Mahonian n = 200, 차의 창 2.3e-14 대 블록 창 2.8e-15 ≈ 25u), partitions.py, extras.py(초기하 pmf, 동점의 비대칭 1/33 대 1/55), kendall_dd.py(n = 200, 400 double-double 6.4e-17·5.2e-17), ties_check.py(동점 경로 전수 비교), worklim.py(일의 상한 안의 큰 크기), fisher_yates.py(카이제곱 3종).
- 검토 반영: 윤문 54건(+2차), 수학 11항(Berry–Esseen O(n^-1/2), 동점 부호 순위의 대칭, 거울 첨자 음수, 25u, 등꼬리 소거 조건, 두 비트 여유 = lgamma 판정, pgf 분산식, Ansari 홀수 N의 c−1, gcd는 옮긴 점수, i·C(n,i) ≤ k배, 1.2e-6에서 4.9배), 깊이(양수 누적 보조정리·add_pos 보조정리, Fisher 표의 다항 Vandermonde, U의 분산 계산, 분할 수 상한 증명·에타 변환, Ansari 유사코드, 정수 창 유사코드·2^600 척도, gcd 전처리, KS 묶음 일반화·띠 그림, CvM 두 표본 절·유사코드, 일의 양 표, 'auto' 문턱 정정, 순위 CLT 명제(Hoeffding 사영·min(c,d)→∞), Fisher–Yates 불변식·해시 경로·지수 경주 명제, ranksums·choice 표 정정, 6장 일계 점화와 연결).
- LIBRARY-NOTES: 큰 동점 없는 MWU의 Grouped 낙하, MWU 정수 경로의 칸 수 상한 없음.
