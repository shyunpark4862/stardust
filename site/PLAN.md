# 사이트 작업 계획

작성 지침은 `WRITING.md`, 기준 빌드는 `README.md`. 마일스톤은 한 번에 하나만 진행한다(여러 마일스톤을 동시에 돌리지 않음, 2026-11-27 사용자). 한 마일스톤 안에서는 페이지 묶음을 나눠 작성 에이전트 여럿을 병렬로 돌려도 되고(맡은 파일이 겹치지 않게, 영역 개요 파일은 한 에이전트만), 검증도 같은 방식. 에이전트는 Opus 5.5, effort medium. 작성 → 검증 → 메인 확인 → 커밋·push → Project Logger1 알림. 영역별 얼개는 `plans/<area>.md`.

## 순서
common(Install·Conventions 첫판은 템플릿 작업과 함께) → array → linalg → frame → special → stats → random → regression → htest → optimize → integrate → differentiate → sparse → spatial → graph → interpolate → fft → smooth → plot → common 마무리. `sd.cluster`는 기준 빌드에 일부만 있어 뺀다.

## 마일스톤
| 그룹 | 마일스톤 | 내용 | 상태 |
|---|---|---|---|
| common | C1 Install and Conventions | 설치, 공통 규칙 페이지 | 진행 |
| array | A1 Array model and creation | 개요, ndarray, dtype·상수·인덱싱·브로드캐스팅·뷰, 만들기, 모양, 잇기, dtype 도구 | 예정 |
| array | A2 Element-wise functions | 산술, 지수·로그, 삼각·쌍곡, 반올림, 비교·논리·비트, 판별, where, 복소수 | 예정 |
| array | A3 Reductions and statistics | 집계, 누적, NaN 건너뛰는 집계, 분위수·중앙값, 공분산, bincount·histogram | 예정 |
| array | A4 Indexing, sorting and products | take·nonzero·searchsorted·interp·diff, 정렬·고유값, 행렬곱, 스레드 | 예정 |
| common | C2 A short tour | 여러 영역을 잇는 짧은 둘러보기 | 예정 |
| common | C3 Home and index polish | 첫 페이지 다듬기, 색인·검색 점검 | 예정 |

다른 영역의 마일스톤은 그 영역을 시작할 때 얼개와 함께 더한다.
