# 5장 Complex Analysis — 계획

GUIDE.md와 CHECKLIST.md를 처음부터 다시 읽고 쓴다(2026-11-25). 옛 5장(legacy/chapters/05-complex.tex, 1.9만 자)의 Theory(해석 함수와 Cauchy–Riemann, Cauchy 정리와 적분 공식, Laurent 급수와 유수, 가지와 가지 자름, 해석적 연속, 사다리꼴 규칙의 지수 수렴)는 살린다. instardust 상자 둘(loggamma의 가지, 비중심 분포의 sinh 사다리꼴)은 본문의 Algorithms·Implementation으로 옮긴다. 뒤의 장(27 원소별 복소 연산, 69 AAA, 78 tanh-sinh, 80–81 Fourier·FFT, 86 Gamma)은 각 API의 나머지를 맡는다. 이 장은 복소해석의 정리를 그대로 옮긴 커널을 맡는다: 1의 거듭제곱근(회전 인자), 원 위의 사다리꼴로 계수 얻기, 복소 단계 미분, 유리 함수의 극과 유수, 가지 자름과 부호 있는 0, 해석적 연속으로 정한 loggamma의 가지, 띠에서 해석적인 함수의 사다리꼴 규칙.

## 이 장이 맡는 API·경로 (구현까지)
- `sd.fft`의 회전 인자 `twiddle(k, n)`(fft/twiddle.hpp): k mod n의 정확한 정수 접기(8분원), 각도의 double-double, 사분원 교환. 계획 표 전체를 Arb로 잰 1.04u(테스트) → 복제본에서 다시 잰다.
- 복소 원소별 함수의 가지 규약: `sd.sqrt sd.log sd.log2 sd.log10 sd.log1p sd.expm1 sd.power`의 complex128(array/kernels/scalar_ops.hpp: libm의 C99 함수, numpy 식의 expm1·log1p·log2·log10)
- special 영역의 자체 복소 도우미(special/cplx.hpp): 0의 부호를 지키는 clog(척도 맞춤), clog1p, csqrt(Kahan), Smith 나눗셈
- `sd.special.loggamma`의 가지(special/ops/loggamma.hpp): Stirling·반사·점화식, 음의 축을 지나는 횟수 세기, 반사의 2πik
- `sd.interpolate.AAA`의 `poles residues roots`(interpolate/aaa.cpp): 화살촉 쌍의 일반화 고유값, N(p)/D'(p)
- 비중심 t의 반대쪽 꼬리(special/ops/noncentral.hpp `nct_opposite_s`): sinh 변수 변환 위의 사다리꼴, 띠 폭 π/4에서 정한 걸음

## 절 배치 (세 축)
1. Analytic Functions — T: 미분 가능성, Cauchy–Riemann, 거듭제곱 급수와 수렴 반지름, 1의 거듭제곱근과 직교성 / A: 회전 인자의 정확한 축소(유사코드) / I: twiddle.hpp, double-double 각, 측정(순진한 cos(2πk/n) 대 stardust, ULP)
2. Cauchy's Theorem and the Integral Formula — T: Cauchy 정리, 적분 공식, Cauchy 부등식, Liouville, Taylor 급수의 수렴 반지름 / A: 원 위 사다리꼴로 Taylor 계수(FFT), 반지름 고르기(절단 대 반올림); 복소 단계 미분 / I: sd.fft·복소 원소별 함수로 측정, 그림(반지름에 따른 오차, h에 따른 세 미분의 오차)
3. Laurent Series and Residues — T: Laurent 전개, 특이점 분류, 유수 정리, 편각 원리, 유리 함수의 부분 분수 / A: 무게 중심 유리 함수의 극(화살촉 쌍 행렬식 명제와 증명)과 유수 N/D' / I: aaa.cpp, ggev, 측정(tan의 극 ±π/2·유수 −1 등), 그림
4. Branches and Branch Cuts — T: arg, 주 로그, 주 제곱근, 거듭제곱, 가지 자름, 부호 있는 0(Kahan) / A: Kahan의 csqrt(상쇄 없음 증명), 척도 맞춘 clog, clog1p / I: libm(array) 대 cx(special) 두 층, 부호 표, 측정(단위원 근처 Re log, 작은 z의 log1p: numpy 식의 상쇄)
5. Analytic Continuation — T: 일치 정리, 경로를 따른 연속, 모노드로미, Γ의 반사, loggamma의 정의와 Hare의 명제 / A: 교차 횟수 세는 점화식(정확성 증명), 반사의 k / I: loggamma.hpp, 측정(loggamma(z+1)−loggamma(z)−log z의 잔차, log Γ와의 2π 차이), 그림(계단 Im loggamma(x±0i))
6. The Trapezoidal Rule — T: 주기 함수의 지수 수렴(앨리어싱 증명), 실수축 위(띠 폭 a: e^{−2πa/h}), 자르기, 변수 변환(sinh, tanh-sinh 이중 지수) / A: sinh 사다리꼴(블록, 멈춤 판정) / I: noncentral.hpp(걸음 c = min(σ, 0.4)의 근거 띠 π/4), 측정(e^{cos θ}, 띠 함수의 h에 따른 오차와 예측 기울기, 라이브러리 nct 꼬리)
7. Testing, 8. APIs in This Chapter, 9. Concluding Remarks

## 그림
회전 인자 오차(순진 대 정확한 축소), 원 위 사다리꼴 계수 오차 대 반지름, 세 미분 오차 대 h, AAA 극 위치와 유수 오차, 가지 자름(TikZ), 단위원 근처 log와 작은 z log1p의 상대오차, loggamma 계단, 주기 사다리꼴 오차 대 N, 띠 함수 사다리꼴 오차 대 h, sinh 사상의 절점

## 유사코드
twiddle, 원 위 사다리꼴 계수, 복소 단계 미분, 화살촉 쌍의 극과 유수, csqrt·clog·clog1p, loggamma 점화식(교차 세기)과 반사, sinh 사다리꼴

## 직접 얻을 수치(복제본 stardust로; 참값은 Decimal·Fraction, mpmath 금지)
twiddle 오차, 원 위 계수 오차, 복소 단계 미분, AAA 극·유수 오차, sd.log·sd.log1p·sd.sqrt의 부호와 정확도, loggamma 잔차, 사다리꼴 오차, nct 꼬리

## 진행 (2026-11-25)
- 초고 7.2만 자 → 검토 반영 뒤 약 8.6만 자, 절 9, 그림 10, 표 6, 알고리즘 9, 증명 26. 1권 214쪽, 경고 0.
- 측정(복제본): twiddle_lib.py(라이브러리 `_core.fft._twiddles`로 모든 k: n ≤ 2^20에서 1.31u, 순진 11.5u), cauchy.py(원 위 계수, 복소 단계: csqrt 때문에 h ≲ 1e-153에서 0.11), aaa_res.py, branch.py(부호 표, sqrt·log1p 작은 부분), loggamma_ref.py(가지를 세지 않는 참값 1.1e-15, 왼쪽 반평면 |y|>7 3.6e-16), trapezoid.py, extras.py(Smith, lambertw, AAA 영점, nct 절점).
- 검토 반영: 윤문 79건+C-11 1, 수학 14항(감은 수, Γ의 무한대, 줄 번호, 사다리꼴 수치, csqrt 문턱, clog1p 단서, 별 모양, 극=q의 근 단서, nct 지수 크기, e^{-49.3}, Euler 공식, 꼬리 급수 범위 등), 깊이(나눗셈 나머지 보조정리, 복소 Stirling 나머지 명제·왼쪽 반평면 줄거리, Blob 유사코드와 반올림 아래 교차 세기, loggamma 특수값 표, Smith 나눗셈 절과 부록 G, Bluestein 처프의 이유와 CisTable, 표 배치, NaN 멈춤 정정, Poisson 수렴, 베타 적분, 블록 행렬식, lambertw 가지, AAA 영점, sinh 절점 그림, 3장 \cref).
- LIBRARY-NOTES: 복소 sqrt의 csqrt 버림, 복소 log1p, cplx.hpp csqrt 넘침·clog, aaa.cpp cdiv의 fma 없음.
