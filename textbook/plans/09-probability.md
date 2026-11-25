# 9장 Measure-Theoretic Probability — 계획

GUIDE.md와 CHECKLIST.md는 8장 시작 때 다시 읽은 판에서 바뀐 것이 없다(2026-11-25, 마지막 수정 11:01). 옛 49장(확률 공간, 분포, 분위수, 기댓값, 독립, 조건부 기댓값, 확률 계산, 2.1만 자)과 옛 51장(모멘트, 특성함수, 큰 수의 법칙, 중심극한정리, 수렴 속도, 표본 모멘트, 1.6만 자)의 이론을 합치고 수렴의 양식, 경험 분포 함수(Glivenko–Cantelli), 조건부 기댓값의 사영을 보탠다. instardust 상자(분포 객체, 표본 모멘트)는 본문으로 옮기되 표본 모멘트의 계산은 111장(기술통계)이 맡으므로 여기서는 연결만 한다. 분포족은 104장, 표본 생성 알고리즘 전체는 105장, KS 검정은 117장이 맡는다.

## 이 장이 맡는 커널 (구현까지)
- stats/discrete.hpp: 이산 분위수 `smallest`(추측에서 갤로핑, 이분, 2^53 너머, 2^1023 너머), `icdf`·`iccdf`·`ilogcdf`·`ilogccdf`의 작은 쪽 비교, 2^-1000 아래는 로그, 깊은 꼬리는 double-double 로그, `settle`(icdf(cdf(k)) == k, 1e-15 여유), 시작 추측(Cornish–Fisher), 후방 오차 계약(T25)
- stats/dist/normal.hpp: 연속 분위수(ndtri 뒤 작은 쪽 로그에 Newton 한 걸음), 분위수의 조건수
- stats/sample.hpp: 균등수 (m + 1/2)·2^-53, 역변환 `by_inverse`(u ≤ 1/2이면 icdf, 아니면 iccdf(1 − u)), 분포별 방법 표(sample_dist.hpp)
- stats 꼬리: ccdf ≠ 1 − cdf, 작은 쪽 규칙, log1p(−작은 쪽), 깊은 로그 꼬리(log_ndtr 등)
- htest: ks_1samp의 D 계산(경험 분포 함수), epps_singleton의 경험 특성함수, 재표집 p값 (b+1)/(n+1)

## 절 배치
1. Probability Spaces — Kolmogorov, π–λ, Borel–Cantelli 둘(둘째는 증명)
2. Distributions and Quantile Functions — 분포, CDF, Galois 보조정리, 역변환·확률 적분 변환, 분해 / 그림: 원자가 있는 CDF와 분위수 함수
3. Computing Quantiles — 후방 오차 계약(정의·이유), 갤로핑+이분 유사코드, 작은 쪽 비교, 정착, 연속 분위수의 조건수와 Newton / 측정: Poisson(3) 원자 근처
4. Sampling by Inversion — (m + 1/2)·2^-53의 성질(증명), by_inverse 유사코드, 분포별 방법 표
5. Expectation and Moments — 변수 바꾸기, 꼬리 적분, Markov·Chebyshev·Jensen, 적률생성함수, 누적량의 덧셈
6. Independence and Product Measures — π-체계 판정, Fubini, 합성곱, 난수열의 독립(카운터 기반 스트림)
7. Conditional Expectation — RN 정의, 존재·유일, 성질(증명), L² 사영(8장), 조건부 밀도 / 그림: 이변량 정규의 조건부 기댓값
8. Modes of Convergence — 넷의 정의, 함의(증명), 반례, Slutsky, 연속 사상, 델타 방법 / 그림: 함의 도식
9. Characteristic Functions — 성질, 반전(줄거리), Lévy 연속성(줄거리+참고), Taylor, Cauchy / 구현: epps_singleton의 경험 특성함수
10. Laws of Large Numbers — 약·강(4차), Etemadi, 몬테카를로 p값과 붓스트랩 / 그림: 누적 평균(정규·지수·Cauchy)
11. The Central Limit Theorem — 증명, Lindeberg, Berry–Esseen, Edgeworth / 그림: 지수 합의 오차 대 n
12. The Empirical Distribution Function — Glivenko–Cantelli(증명), DKW, Kolmogorov 극한 / 구현: D의 계산 / 그림: √n D_n
13. Numerical Computation of Probabilities — 작은 쪽 규칙, 로그 확률 / 측정 표
14. Testing, 15. APIs, 16. Concluding Remarks

## 직접 얻은 수치
clt_gamma.py(√n·오차 0.1330, BE 0.4748ρ), tails.py(Φ 소거, Poisson F(2) 반올림), tails_sd.py(stardust ccdf 4e-16, logccdf 1.7e-17, icdf(fl F(2)) = 2), lln.py(누적 평균, √n D_n)

## 진행 (2026-11-25)
- 초고 4.3만 자 → 검토 반영 뒤 약 7.3만 자, 절 16, 그림 13, 표 6, 알고리즘 13, 증명 18. 1권 378쪽 경고 0.
- 측정(복제본, 참값은 Decimal·Fraction): clt_gamma.py(√n·오차 → 0.1330), tails.py·tails_sd.py·tails_curve.py(ccdf 4.4e-16, 1 − cdf 소거), identities.py·contract_atoms.py(작은 쪽 항등식, 원자 근처 12386점 계약 위반 0), search_evals.py(술어 계산 CF 2.0 대 평균 시작 4.3–12.5), lln.py(누적 평균, √n D_n), extras.py(Borel–Cantelli, 꼬리 부등식, Irwin–Hall, 특성함수), condexp.py, mvn_cover.py(Genz 300/300 보고 오차 안), perm_pvalue.py(초균등).
- 검토 반영: 윤문 60건(+2차), 수학 15항(균등수 명제 정정과 라이브러리 u() 결함, λ-체계 정의, 독립 판정의 Ω, 델타 방법, 조건부 기댓값 (iii), 몬테카를로 p값의 동률, Berry–Esseen 줄거리, Cramér 조건, Poisson cdf(13) 예, 조건수 절댓값 등), 깊이(by_inverse를 쓰는 분포 없음과 표집 방법 표 확대, 정규 icdf의 needs_step, settle의 실제 역할과 2^-1000 단서, 술어 계산 수 표, KMemo, 갤로핑 세부, Philox 계수기 유사코드, log_ndtr 두 영역, KS p값 경로와 Birnbaum–Tingey, 재표집 p값 유사코드·그림, Epps–Singleton 유사코드, Edgeworth 조건 분리·유도, π–λ·Skorokhod·Lévy 조밀함·Etemadi·Lindeberg·DKW 줄거리, Kolmogorov 두 급수와 유사코드, Jacobian 변환, Chernoff·Hoeffding, 정규 벡터의 조건부 분포와 Genz 유사코드·그림, 조건부 분산 공식, 균등 적분 가능성, Cramér–Wold, 순서통계량, 이산 꼬리 합·중심 모멘트 유사코드, 꼬리 오차 곡선).
- LIBRARY-NOTES: u() = (m + ½)2^-53의 반올림(우선순위 높음), Cornish–Fisher 문턱 kWideSd, by_inverse 미사용, MVN cdf 3σ의 뜻.
