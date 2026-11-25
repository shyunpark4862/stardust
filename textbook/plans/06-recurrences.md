# 6장 Differential Equations and Recurrences — 계획

GUIDE.md와 CHECKLIST.md를 다시 읽고 쓴다(2026-11-25, 5장 시작 때 읽은 판에서 바뀐 것 없음). 옛 6장(legacy/chapters/06-recurrences.tex, 1.6만 자)의 Theory(이계 선형 ODE와 Wronskian·Abel, 특수 함수의 방정식, Frobenius, 삼항 점화식의 최소해·지배해, 방향과 안정성, Miller, Pincherle와 연분수, 행렬 지수)는 살리고 instardust 상자(bessel.hpp)는 본문의 Algorithms·Implementation으로 옮긴다. 91–92장(Bessel·Airy)은 각 영역의 근사식과 정확도를, 57장은 expm 알고리즘을, 76–77장은 QUADPACK 전체를 맡는다. 이 장은 점화식과 ODE의 이론, 그리고 그 이론을 그대로 옮긴 커널(점화의 방향, Miller, CF1, 지배해의 위쪽 점화, Wronskian 정규화, 경계값 문제로 푸는 점화식, Clenshaw 합)을 맡는다.

## 이 장이 맡는 API·경로 (구현까지)
- `special.jv yv iv kv`의 점화 부분(special/ops/bessel.hpp): `miller`(N = top + 10 top^(1/3) + 8, Neumann 합 정규화, 2^600 척도), `upward`(계수 2v/x를 1/x의 double-double로 한 번 반올림), `i_ratio`(I의 역방향, 양수뿐), `jy_temme`의 CF1(앞 패스로 항 수만 세고 뒤 패스로 값, 음수 세어 부호), 아래쪽 점화(double-double, 지수 분리), Wronskian으로 척도, Y의 위쪽 점화, `fast_ik_scaled`(K 위쪽 + Wronskian으로 I)
- `integrate.quad(weight='cos'/'sin')`의 Chebyshev 모멘트(integrate/quadpack.hpp `qc25f`): |ωh| ≤ 24이면 삼중대각 경계값 문제(`dgtsl`, 끝값은 점근식), 넘으면 앞으로 점화
- Clenshaw 합: Cephes `chbevl`(special/ops/bessel01.hpp의 i0 i1 k0 k1), stats/mvcdf.hpp의 `Cheb::eval`
- 행렬 지수: 이론만(알고리즘은 57장)

## 절 배치 (세 축)
1. Linear Differential Equations — T: 존재·유일성(Picard, 증명), 해 공간, Wronskian·Abel, 연립 꼴과 행렬 지수 / 그림: Airy 방정식의 해 Ai, Bi(진동 대 지수)
2. Equations of Special Functions — T: Bessel·변형 Bessel·Airy·초기하, Frobenius, 점근(Liouville–Green 개요) / I: Wronskian 관계로 하는 검증(라이브러리 J·Y 잔차)
3. Linear Recurrences — T: 상수 계수(특성근, 복소평면), 최소해·지배해, Poincaré–Perron(줄거리), 방향과 안정성 / 그림: J_n(10), Y_n(10)의 크기(n < x 중립, n > x 분리)와 앞으로 점화의 오차
4. Backward Recurrence — T: Miller 정리, 시작 차수의 오차, Pincherle와 연분수 / A: Miller 유사코드, CF1 앞·뒤 패스 / I: bessel.hpp `miller`(N 선택과 그 측정), Neumann 합, 2^600 척도, `jy_temme`의 부호 세기·Wronskian 정규화, 그림: Miller 오차 대 N − top
5. Forward Recurrence for Dominant Solutions — A: 위쪽 점화(Y, K), Wronskian으로 다른 해 / I: double-double 계수, 지수 분리(m 2^e), 긴 점화의 반올림(86 ULP 이야기 재측정)
6. Recurrences as Boundary Value Problems — T: Olver의 방법(삼중대각) / A: 유사코드 / I: QUADPACK 모멘트(경계값 대 앞으로), 그림: 두 방법의 오차
7. Clenshaw's Algorithm — T: 삼항 점화로 하는 급수 합, 오차 해석 / A: 유사코드 / I: chbevl, mvcdf Cheb, 그림: 오차
8. Testing, 9. APIs, 10. Concluding Remarks

## 그림
Airy Ai·Bi, J_n·Y_n의 크기, 앞으로 점화 대 Miller 대 라이브러리, Miller 오차 대 N − top, 모멘트의 앞으로 점화 대 경계값, Clenshaw 오차, 라이브러리 jv 정확도·Wronskian 잔차

## 유사코드
Picard(설명용 아님, 증명만), Miller(Neumann 정규화), CF1 뒤 패스와 부호, 위쪽 점화(지수 분리), Olver 삼중대각, Clenshaw

## 직접 얻을 수치(복제본; 참값은 Decimal, mpmath 금지)
J_n(1) 앞으로 점화 오차, Miller 오차, 라이브러리 jv·yv 오차와 Wronskian 잔차, 모멘트 오차, Clenshaw 오차, Airy 값

## 진행 (2026-11-25)
- 초고 4.2만 자 → 검토 반영 뒤 약 7만 자, 절 11, 그림 12, 표 4, 알고리즘 11, 증명 14. 1권 252쪽 경고 0(5장의 \textsc 글꼴 경고 두 건도 \textrm으로 없앰).
- 측정(복제본, 참값 Decimal): bessel.py(앞으로 점화, Miller, 시작 차수, jv 포락선 9e-16, Wronskian 1.2e-15), threeway.py(400걸음: binary64 31–40u, 빠른 경로 식 32–43u, dd ≤1.3u), downrec.py, cf1terms.py, neumann.py(조건수, 전개 확인), moments.py·moments_scan.py(경계값 ≤1.4e-15), clenshaw.py, gram.py(차수 점화 붕괴, savgol 3.6e-17), wronsk.py(v ≤ 1000, 4e-15; 511.3/512.3 이진 지수 경계의 가짜 잔차), longrec.py(Airy).
- 검토 반영: 윤문 55건, 수학 18항(최대항 e^x/(πx), Neumann 부호, b_m, 반올림 누적, 2v 정확성, 모멘트 증가율, N 30.0/35.7, ε 부호, ψ(ν+1), (2ν/(ex))^ν, 비용, Clenshaw 한계, Poincaré–Perron, y_r, 잔차 식, chbevl), 깊이(빠른 경로·일반 경로와 경로 표, Temme 급수 명제·유사코드, CF2, fast_y·fast_ik 유사코드, K의 나눗셈 없는 CF2, 이산 상수 변화법, Debye 지수 유도, Frobenius, Neumann 증명(망원합), Pincherle 역 증명, CF1 보조정리·항 수, jy_upward, Olver 유사코드, QUADPACK 계 구성, 다른 매개변수의 점화 절(₁F₁, Gram, 일계, 이산 분포, Gauss), Airy 격자, Wronskian 그림).
- LIBRARY-NOTES: bessel.hpp "86 ULP" 주석의 기준.
