# G4 정확도: 강건 회귀

`bench/regression/g4/accuracy.py`로 쟀다(2026-11-18, MM의 S 뉴턴 걸음(G82)과 M-척도 기울기 수정 뒤 2026-11-19에 다시; 원래 출력은 `accuracy.log`). 테스트는 `tests/regression/test_robust.py`다.

- **Arb** (`tests/regression/ref/robust.json`, `tools/regref.py`의 `rlm`): 해의 ψ 조각과 척도의 행을 고정한 추정방정식을 고정밀로 풀고, 그 조각이 확실한 상자에서 Krawczyk 판정으로 증명한 해. 계수는 벡터의 상대오차(2-노름), 척도·표준오차는 최대 상대오차, 가중치는 최대 절대오차다. MM은 sd의 S 계수와 척도를 고정한 M 단계(와 avar1)의 참값이다.

| 문제 | 계수 | 척도 | 표준오차 | 가중치 | 반복 |
|---|---:|---:|---:|---:|---:|
| huber_mad_even | 4.9e-17 | 0 | 0 | 1.1e-16 | 15 |
| huber_mad_odd | 5.1e-17 | 1.7e-16 | 1.4e-16 | 2.2e-16 | 13 |
| huber_huber_even | 2.4e-17 | 1.3e-16 | 1.5e-16 | 1.1e-16 | 13 |
| huber_huber_odd | 8.0e-17 | 1.2e-16 | 1.5e-16 | 2.2e-16 | 15 |
| bisquare_mad_even | 7.0e-17 | 1.9e-16 | 1.9e-16 | 3.3e-16 | 12 |
| bisquare_mad_odd | 7.6e-17 | 0 | 1.9e-16 | 3.3e-16 | 13 |
| bisquare_huber_even | 8.4e-17 | 1.4e-16 | 2.0e-16 | 3.3e-16 | 9 |
| bisquare_huber_odd | 8.2e-17 | 1.2e-16 | 2.7e-16 | 3.3e-16 | 15 |
| hampel_mad_even | 4.9e-17 | 0 | 3.9e-16 | 3.5e-17 | 12 |
| hampel_mad_odd | 5.7e-17 | 0 | 1.9e-16 | 6.9e-17 | 18 |
| hampel_huber_even | 9.8e-17 | 1.4e-16 | 1.7e-16 | 2.2e-16 | 15 |
| hampel_huber_odd | 8.7e-17 | 1.2e-16 | 3.8e-16 | 1.1e-16 | 10 |
| huber_mad_noint | 5.7e-17 | 0 | 3.2e-16 | 2.8e-17 | 13 |
| bisquare_mad_400 | 3.5e-17 | 2.0e-16 | 5.2e-16 | 3.3e-16 | 12 |
| stackloss_huber | 9.6e-17 | 7.3e-16 | 1.0e-15 | 5.6e-16 | 19 |
| stackloss_bisquare | 7.2e-17 | 1.4e-15 | 1.1e-15 | 1.1e-15 | 11 |
| stackloss_hampel | 1.1e-16 | 4.3e-16 | 1.0e-15 | 3.3e-16 | 11 |
| mm_leverage | 5.0e-17 | 0 | 1.2e-15 | 3.3e-16 | 6 |
| mm_stackloss | 8.7e-17 | 2.3e-16 | 6.7e-16 | 4.4e-16 | 14 |

- 뉴턴 걸음(설계 §3.6) 없이 IRLS만이면 huber_mad_odd·hampel_mad_odd가 `max_iter` 50에 닿아 계수 오차가 6.7×10⁻¹⁰·8.7×10⁻¹¹이다. MM의 avar1을 XᵀΨ′X로 직접 만들면 mm_stackloss의 표준오차가 8×10⁻¹⁴다(얇은 Q 좌표에서 6.7×10⁻¹⁶).

- **statsmodels** (`RLM(...).fit(tol=1e-15, maxiter=1000)`, 제안 2는 `HuberScale(d=2.5, tol=1e-15, maxiter=1000)`; 10% 세로 이상점, n = 150, 157, 400, k = 3의 최대):

| ψ | 척도 | 계수 | 척도 | 표준오차 | 가중치 | sd 반복 | statsmodels 반복 |
|---|---|---:|---:|---:|---:|---:|---:|
| huber | mad | 2.3e-15 | 2.1e-15 | 1.2e-15 | 4.7e-15 | 12/13/12 | 20/47/22 |
| huber | huber | 2.1e-15 | 1.2e-15 | 9.5e-16 | 1.8e-15 | 11/11/11 | 1000/19/20 |
| bisquare | mad | 2.2e-15 | 9.9e-16 | 6.8e-16 | 2.4e-15 | 12/13/12 | 27/22/20 |
| bisquare | huber | 2.1e-15 | 1.4e-15 | 1.0e-15 | 1.3e-15 | 9/8/9 | 17/16/19 |
| hampel | mad | 1.6e-15 | 1.2e-15 | 8.5e-16 | 2.0e-15 | 9/16/9 | 14/30/17 |
| hampel | huber | 1.7e-15 | 1.1e-15 | 1.3e-15 | 1.4e-15 | 16/17/16 | 1000/1000/29 |

  - statsmodels의 1000번은 절대 변화 1e-15에 닿지 못하고 반복 상한에 닿은 것이다(값은 이미 수렴했다).

- **R** (4.6.2, MASS 7.3.66, robustbase 0.99.7; `tests/regression/ref/robust_r.json`, `tools/gen_robust_r.py`; 스택 손실, 세로 이상점 (200, 3), 지렛대 20% (300, 2)·30% (500, 2)·25% (5000, 3; 큰 n fast-S)):
  - `MASS::rlm(scale.est='proposal 2', k2=2.5, maxit=1000, acc=1e-15)`: 계수 4.1×10⁻¹⁵, 척도 1.7×10⁻¹⁴, 표준오차 1.0×10⁻¹⁴ 안(MASS의 `summary.rlm`이 var ψ′를 n − 1로 나누는 비를 곱해서; 곱하지 않으면 7.5×10⁻⁷–1.4×10⁻³ 다르다).
  - `MASS::rlm(scale.est='MAD')`: 0.6745로 나눠서 척도가 1.0–1.9×10⁻⁵, 계수가 3.2×10⁻⁷–2.7×10⁻⁶ 다르다.
  - robustbase의 M 단계(`lmrob..M..fit`, `rel.tol` 1e-15)와 `.vcov.avar1`을 sd의 S 시작에서: 계수 1.1×10⁻¹⁵, 표준오차 7.2×10⁻¹⁵ 안(큰 n 문제 4.0×10⁻¹⁵), 스택 손실만 1.5×10⁻¹³(robustbase가 XᵀΨ′X를 직접 역행렬로: sd가 Arb와 6.7×10⁻¹⁶).
  - `lmrob` 자체(자기 부분표본, S의 `refine.tol` 1e-7; R의 난수는 실행마다 다르다): 계수 1.1×10⁻⁹–1.2×10⁻⁷, 척도 1.3×10⁻⁸–1.1×10⁻⁵, 표준오차 9.2×10⁻⁹–1.9×10⁻⁶. lmrob의 `scale`은 마지막 고정점 걸음의 값이다.

- **scipy** (`theilslopes`, `siegelslopes`): n = 2–5,000의 무작위·동률·격자·평평한 자료, 두 방법, `alpha` 0.95·0.1·0.5·0.999, 배치, `nan_policy='omit'`에서 비트까지 같다(0의 부호 빼고). 쌍의 정확한 유리수 기울기의 중앙값과는 4ulp 안이고, 기울기가 double로 정확한 자료에서는 같다.
