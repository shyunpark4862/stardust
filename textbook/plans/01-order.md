# 1장 Notation and Order — 계획

GUIDE.md 전체와 CHECKLIST.md를 다시 읽고 쓴다(2026-11-25). 옛 1장(legacy/chapters/01-order.tex, 5.4만 자)의 Theory는 탄탄하므로 살리고, Algorithms·Implementation을 같은 무게로 새로 쓴다. 정렬 알고리즘 자체(introsort, 기수 정렬, 정렬 네트워크)는 18장과 31장의 몫이고, 이 장은 "순서와 동점을 계약으로 정하고 그것을 지키는 구현"을 맡는다.

## 이 장이 맡는 API·규칙 (구현까지)
- 비교의 계약: 배열 `sort`/`argsort`의 NaN 위치와 안정성, `descending=True`, ±0의 순서 복원 — 정렬 키 변환(float64 → uint64 전순서 키)
- 동치의 계약: `unique_*`(NaN 하나, ±0 하나, 대표는 처음 나온 원소), frame의 키 규칙(NA, NaN)
- 선택의 동점 규칙: `argmin`/`argmax`(NaN 먼저, 동점은 첫 위치)와 병렬 축약의 결정성(결합적 선택 연산)
- 순위: `sd.stats.rankdata`(average/min/max/dense/ordinal, NaN 정책)
- 여러 키: frame `sort_values`(LSD 안정 정렬, 키별 ascending·na_position), 문자열 8바이트 접두 키
- 표기 절은 짧게 유지

## 절 배치 (세 축)
1. Notation Used in This Book (짧게)
2. Relations and Orders — Theory: 동치·분할, 부분·전순서, 약순서와 키 표현, 사전식 순서
3. Comparators in Practice — Algorithms/Implementation: 비교 함수 계약(엄격한 약순서), 위반 시 UB(범위 밖 읽기) 유사코드로 보이기, NaN이 깨는 것, stardust의 비교자 설계
4. Floating-Point Orders — Theory+Algorithm+Impl: IEEE 비교, totalOrder, 부호 비트 뒤집기로 uint64 키 만들기(유사코드·증명), NaN 마지막 규칙, ±0
5. Sorting and Stability — Theory(정렬 정의, 유일성, 안정성 정리) + Algorithm(키에 색인 붙이기, LSD 다중 키 정리) + Impl(stardust: 정렬 네트워크의 ±0 복원 알고리즘, 내림차순은 비교자 뒤집기, frame 다중 키)
6. Selection and Deterministic Ties — argmin/argmax의 결합적 선택 연산과 병렬 축약(증명, 유사코드), stardust의 argmax 구현(NaN 먼저, FMAX 사슬 뒤 위치 찾기)
7. Equivalence in Practice: unique and Group Keys — 동치관계의 구현(NaN 정규화, ±0 정규화 키, 첫 등장 대표), 해시와 정렬 두 길(자세한 해시는 19장)
8. Ranks — 정의, 합 보존 명제, 순위 알고리즘(유사코드), rankdata 구현과 NaN 정책
9. Strings and Lexicographic Keys — 8바이트 접두 키(증명: 접두 키 순서 = 사전식 순서의 첫 성분), 동점 구간 재정렬 유사코드
10. Testing and Verification — 이 계약들을 무엇으로 확인하는지(골든, 레이아웃·스레드 불변, numpy와의 차이)
11. APIs in This Chapter (본문 + 표)
12. Concluding Remarks

## 그림
- Hasse 그림(약수), 약순서 = 키로 묶인 층(동치류의 사슬) 그림
- 비교자 위반 시 무경계 안쪽 루프가 범위를 넘는 그림(포인터 진행)
- float64 비트 패턴 → uint64 키 사상 그림(수직선 위 순서: −NaN, −inf, …, −0, +0, …, +inf, +NaN)
- 안정 정렬/LSD 다중 키 단계 그림(표 예시)
- 병렬 argmax 축약 트리 그림
- 순위 표(옛 표) + 동점 구간 그림
- 문자열 접두 키 정렬 단계 그림

## 유사코드
- 비교자 위반으로 범위 밖을 읽는 unguarded insertion
- float64 → uint64 키
- ±0 부호 복원
- 결합적 argmax 축약(병렬)
- unique의 정렬 길(정규화 키, 첫 등장 대표)
- 순위(옛 것)
- LSD 다중 키(옛 것), 문자열 접두 키 정렬

## 직접 얻을 수치
- 키 변환의 순서 확인(모든 특수값), NaN 페이로드
- 병렬 argmax 결정성 실험(직접 C로, 스레드 수 바꿔)
- 정렬 네트워크 ±0 흐트러짐 예(직접 C로 작은 네트워크)

## 진행 (2026-11-25)
- 초고 5만 자, 절 12, 그림 8, 표 3, 알고리즘 9, 증명 12. audit 통과, 1권 38쪽 경고 0. 그림 8개 100dpi로 확인(1.2 설명 글자 겹침, 1.8 문자열 그림 다시 그림).
- 측정: figures/01/keymap.py(키 순서), argmax_chunks.py(동점 규칙 없는 합치기의 답이 조각 수마다 바뀜). numpy unique([0.0,-0.0])의 대표가 -0임을 확인.
- 남은 것: 수학·깊이 검토, 윤문.
