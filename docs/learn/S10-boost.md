# S10. Boost.Math 도입

> S9 뒤의 후속이다. 코드리뷰와 Boost.Math 비교를 한 뒤, 사용자가 "Boost로 갈아끼는 게 이득인 부분은 다 갈고, 손해인 부분만 우리가 작성한다"로 정했다. 설계는 docs/design/special.md §5.14, 측정은 `bench/special/boost/`(compare.md, regions.md, routing.md)와 `bench/special/s10/`이다.

## 1. 결과
| 항목 | 결과 |
|---|---|
| 테스트 | 3,597개 통과 (S9과 같은 수; 기준만 바꾼 것 하나: `betainc + betaincc`의 합 2 → 3 ULP). 골든 출력 세 영역 불일치 0. 컴파일 경고 없음 |
| Boost를 부르는 함수 | 20개(구간별): `zeta i0 i1 k0 k1 iv kv expi lambertw`(두 가지) `digamma gamma beta gammaincinv gammainccinv betainc betaincc betaincinv betainccinv hyp1f1` |
| 속도 (1스레드, 전 → 후 ns) | `zeta` 95 → 18, `i0` 16 → 4.3, `k0` 24 → 11, `kv` 1625 → 873, `expi` 26 → 12, `lambertw` 21 → 11, 불완전 베타 553 → 403 등 (routing.md §2) |
| 정확도 | 모두 계약 안. `hyp1f1` 느슨한 영역이 상대 3.9e-3에서 2.8e-13으로 좋아졌다. 최대 ULP가 늘어난 곳도 있다(`zeta` 24 → 45, `hyp1f1` 보장 영역 6.9 → 79) |
| 의존성 | Boost.Math 헤더 125개(2.7 MB)를 `third_party/boost_math/`에 (BSL-1.0, boostorg/math 4682f44). 빌드 시간은 special_bind.cpp 한 파일이 늘었다 |

사용자 결정 (2026-11-14)
- 계수 가져오기가 아니라 Boost.Math를 직접 부른다. 이미 우리가 쓴 함수라도 Boost가 손해가 아니면 바꾼다.
- 헤더는 쓰는 것만 저장소에 둔다(서브모듈이나 시스템 Boost가 아니라).
- 손해의 기준: 정확도(계약)와 속도 둘 다. Boost가 더 정확해도 느리면 우리 것이다. 시간 차이 10% 안은 같은 것으로 본다.
- Boost가 일부 점에서만 틀리면 구간을 잘게 나눠 지키는 구간만 Boost로 한다.

## 2. 어떻게 골랐나
- **함수 전체로 먼저**(compare.md): Boost에 대응하는 43개를 같은 참값 파일과 같은 10⁶개 표본으로 쟀다. 기하평균으로 Boost가 1스레드에서 9% 빨랐지만, 우리 참값 파일의 어려운 점(서브노멀 꼬리, 분기점, 큰 인자)에서 50개 중 상당수가 1e-14를 넘었다.
- **구간으로**(fails.py, rules.py): Boost가 틀리는 점은 대부분 한쪽에 몰려 있었다. `zeta`는 x < −63, `gamma`는 x < −171(서브노멀), `expi`는 x ≈ −709, `lambertw`는 분기점에서 5e-6 안, `digamma`는 음의 극점, `iv`는 결과가 1e-166보다 작은 큰 차수였다. 입력으로 가르는 규칙과 "Boost의 결과가 정상 범위가 아니면 우리 것"을 함께 쓴다.
- **그 구간의 속도**(routing.md): 규칙이 Boost로 보내는 점만 모아 다시 쟀다. 전체 평균이 아니라 구간 안에서 느리지 않아야 바꾼다. `gamma`는 9% 느렸지만 10% 안이라 바꿨다.
- Airy는 바꾸지 않았다. 우리 `airy`는 네 값을 한 번에 계산해서(277 ns) Boost를 넷 따로 부르는 것(약 720 ns)보다 빠르고, Boost는 x < −15와 큰 양의 x에서 틀린다.

## 3. 바꾸면서 생긴 일
- **같은 모양의 코드에 잘못 넣음**: `beta`에 넣을 Boost 분기를 문자열 치환으로 넣었는데, `betaln`에 같은 줄이 먼저 있어서 그쪽에 들어갔다. `betaln(1, 1)`이 log를 빼먹고 1을 냈고, `betaln`을 초기값에 쓰는 베타 역함수까지 틀렸다(테스트 6개 실패). 넣을 곳을 함수 안에서만 찾게 고쳤다.
- **내부 호출**: 우리 식 안에서 부르는 `Gamma::f`, `Digamma::f`(`hyp1f1`, `hyp2f1`)도 바뀐 구간에서는 Boost다. `hyp2f1`의 느슨한 영역 최대가 3.5e3에서 3.1e3 ULP로 조금 바뀌었다.
- **짝의 합**: `betainc + betaincc = 1`은 전에는 같은 BRATIO 호출이라 2 ULP 안이었다. Boost는 둘을 따로 계산해서 3 ULP가 됐다. 두 함수의 Boost 구간을 a, b ≤ 6으로 맞추고 테스트를 3 ULP로 바꿨다(계약 밖의 성질이다).
- **복소 `lambertw`의 실수축**: 실수 함수가 Boost로 바뀌자 `lambertw(-0.1+0j)`가 실수 `lambertw(-0.1)`과 비트가 달라졌다. 복소 경로가 실수 함수와 같은 길(`w0_routed`)을 쓰게 했다.

## 4. 확인 부탁
- **최대 ULP가 늘어난 함수**: `zeta`(24 → 45), `gammaincinv`(6.7 → 50), `hyp1f1` 보장 영역(6.9 → 79), `beta`(9 → 53), `kv`·`iv`(8–16 → 22–39). 모두 계약(상대 1e-14) 안이고 대신 빨라졌다. 결정의 기준(계약과 속도)대로다.
- **코드리뷰에서 나온 10건**은 이번에 고치지 않았다. Boost로 가는 구간과 겹치지 않아 그대로 남았다(`hyp1f1(-1e12, 2, 0.5)`는 Boost도 값을 못 내서 우리 식으로 돌아와 여전히 inf).

## 5. 승인
- (대기)
