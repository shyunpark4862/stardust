# 4장 Matrix Analysis — 계획

GUIDE.md(변경 없음)와 CHECKLIST.md(3장에서 복제본 측정 항목을 더함)를 다시 읽고 쓴다(2026-11-25). 옛 4장(legacy/chapters/04-matrix-analysis.tex, 3.4만 자)의 Theory(노름과 동치성, 유도 노름, SVD와 Eckart–Young, 수치적 랭크, 조건수, Courant–Fischer·Weyl·Bauer–Fike, Schur, Kronecker, 양정치)는 살리고 instardust 상자 넷(norms.cpp, linalg.md의 rtol, lapack.hpp의 Sylvester, kron/kronsum)을 본문의 Algorithms·Implementation으로 옮긴다. V권의 장들(47 노름·BLAS, 50 Cholesky, 52 lstsq·pinv·rank, 53 cond, 56 SVD, 57 Sylvester)은 각 분해의 알고리즘을 맡고, 이 장은 그 장들이 기대는 이론과 이론을 그대로 옮긴 커널(노름, 허용오차, Kronecker, 양정치 판정, Bartels–Stewart의 뼈대)을 맡는다. 47장은 노름과 kron의 구현을 이 장에 미룬다.

## 이 장이 맡는 API·경로 (구현까지)
- `linalg.vector_norm`: 빠른 한 번(척도 없이, NEON fma 8갈래와 최댓값을 같은 읽기에서) → 넘침이거나 모두 작으면(2^-969, count·2^-1000) 최댓값으로 척도 맞춘 둘째 읽기, 쌍별 합, 복소수는 두 항, NaN·inf, 빈 입력 0, 병렬에서도 같은 비트(vnorm.hpp, fastsum.hpp, norms.cpp)
- `linalg.matrix_norm`: 1·inf(선 합의 최대·최소, 메모리 배치에 맞춘 선 방향과 strips), fro(벡터 2-노름), 2·-2·nuc(특이값, NaN이면 ValueError)
- `linalg.matrix_rank`, `pinv`의 허용오차 rtol = max(m,n)·2^-52, τ = rtol·σ1 (spectral.cpp, args.cpp)
- `linalg.kron`, `kronsum`(basic.cpp): 열 우선 묶음 계산, 배치 브로드캐스트, 크기 넘침 검사, kronsum은 곱 없이 a, b, a+b만(희소와 비트 일치), 큰 결과의 열 병렬
- 양정치 판정과 이차 형식: Cholesky 시도, 다변량 정규 logpdf의 double-double Cholesky·전진 대입·log det(stats/dd_linalg.hpp)
- Bartels–Stewart의 삼각 Sylvester 단계(solve_sylvester는 57장, 여기서는 뼈대와 척도)

## 절 배치 (세 축)
1. Vector Norms — T: 정의, Hölder·Minkowski, 동치성 / A: 넘침 없는 2-노름(두 단계, 경계의 유도, Blue) / I: vector_norm, 측정 그림(척도에 따른 오차: 순진·numpy·stardust)
2. Matrix Norms — T: 유도 노름, 공식 증명, Frobenius, 유니터리 불변 / A: 선 합의 계산 방향 / I: matrix_norm(배치, strips, 병렬), 배치 순서별 시간 표
3. The Singular Value Decomposition — T: 존재, Eckart–Young, 핵 노름 / A: 수치적 랭크·pinv의 허용오차 / I: rtol 기본값과 근거, 측정 그림(정해진 특이값 행렬의 svdvals 오차와 uσ1/σk)
4. Condition Numbers — T: 연립방정식 섭동 정리, 특이까지의 거리 / I: Hilbert 행렬 풀이 오차 대 κu(정확한 유리수 해)
5. Perturbation of Eigenvalues and Singular Values — T: Courant–Fischer, Weyl, Bauer–Fike, 부분공간·최소제곱 / 그림: 섭동한 Jordan 블록의 고유값(원 반지름 ε^{1/n})
6. The Schur Decomposition — T: Schur, 실수 Schur / A: Bartels–Stewart 유사코드 / I: LAPACK trsyl의 척도, kron 풀이와 비교 표
7. The Kronecker Product — T: 성질 / A: kron 인덱스 사상(혼합 기수), kronsum 직접 구성 / I: 열 우선, 브로드캐스트, 넘침 검사, 희소와 비트 일치, 그림: kronsum 블록 구조
8. Positive Definite Matrices — T: 동치 조건, Schur 보수 / A: Cholesky 판정·이차 형식·log det / I: 다변량 정규 logpdf의 double-double Cholesky
9. Testing, 10. APIs, 11. Concluding Remarks

## 그림
척도별 2-노름 오차, 특이값 오차와 uσ1/σk, Hilbert 오차 대 κu, Jordan 블록 고유값, kronsum 블록 구조(TikZ)

## 유사코드
두 단계 2-노름, 1·inf 노름의 선 합, 수치적 랭크·pinv, Bartels–Stewart, kron, kronsum, Cholesky 판정과 이차 형식

## 직접 얻을 수치(복제본 stardust로)
vector_norm 오차(정확한 유리수 참값), matrix_norm 시간(C·F 배치), svdvals 오차, Hilbert 풀이 오차와 κ, Jordan 고유값, Sylvester 대 kron 풀이, kronsum 대 sparse.kronsum 비트 비교

## 진행 (2026-11-25)
- 초고 4.4만 자 → 검토 반영 뒤 5.4만 자, 절 11, 그림 6, 표 5, 알고리즘 9, 증명 16. 1권 168쪽 경고 0, 그림 6개와 새 알고리즘 쪽 확인(kronsum 그림의 A 글자 위치).
- 측정(복제본): measure.py(노름 척도, 행렬 노름 시간, svdvals, Hilbert, Jordan, Sylvester, kronsum), hilbert_kappa.py(정확한 역행렬로 κ: cond 대신), mvn_logpdf.py(binary64 Cholesky 1e-3 대 stardust 2e-16). matrix_norm ±1·±inf와 'fro'의 C·F 끝 비트 차이는 설계 LA3·LA4가 받아들인 것(17/40, 2/20).
- 검토 반영: 윤문 48건+쉼표 14, 수학 13항(norm2 판정을 a_max>0 && fl(a_max²)<2^-969로, 0 경계 2^-537.5, 순진한 계산의 실패 k, Hilbert κ와 1/κ, Cholesky 비용 d³/6, svd 선과의 거리, k_i 정수, pinv 최소 노름 조건, Bartels–Stewart 척도 누적, NaN이 모든 블록으로, 줄 번호), 깊이 검토(빠른 경로·조심스러운 경로 두 알고리즘과 ord 표, Walk, 레이아웃 비트, pinv 경계(2^-1022, 2^600)·비용, cond, MVN 빠른 경로 알고리즘과 한계, Wishart·행렬 정규, Kronecker 행렬식·Cholesky 명제, Frobenius·핵 노름·Eckart–Young F·정규 삼각·실수 Schur·Schur 보수 증명, sep와 Higham 역방향 오차, 라이브러리 시험 파일).
- LIBRARY-NOTES: 척도 맞춤의 2의 거듭제곱(우선순위 낮음).
