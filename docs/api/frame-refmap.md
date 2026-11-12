# 데이터프레임 문서 참조 대응표 (임시)

> 코드 주석과 테스트의 옛 참조("API.md §14", "API.md §11 C2" 등)를 새 문서로 바꾸는 데 쓴다. 참조를 모두 바꾸면 이 파일을 지운다.

- 옛 문서: `stardust-pandas/API.md` (M5까지 `legacy/pandas/API.md`) (절 번호 `§1`–`§22`, 규칙 ID `S1`–`S8`, `C1`–`C7`, `A6`)
- 새 문서: `docs/api/frame.md` (아래 표의 `§`는 따로 적지 않으면 이 문서의 절), 루트 `API.md`
- "인용 수"는 옛 데이터프레임 코드와 테스트에서 센 값이다.
  - Python과 테스트는 옮기기 전의 `legacy/pandas/src`, `legacy/pandas/tests`에서 셌다. 옮긴 뒤의 `src/stardust/frame/`, `tests/frame/`에서 다시 세어도 같았다.
  - C++는 이미 옮겨진 뒤라서 `csrc/frame/`, `csrc/bind/frame_bind.*`에서 셌다. `frame_bind.*`에는 인용이 없다.
  - 코드는 옛 API.md를 모두 `API.md §N` 형태로 인용한다. `API §N` 형태와 `§N.M` 형태는 없다.
  - `API.md §5, §18`, `API.md §15 and §16`처럼 여러 절을 한 번에 인용하면 절마다 하나로 셌다. 주석이 줄을 넘어가도(`(API.md` / `// §19`) 하나로 셌다.
  - `API.md §11 C2`처럼 규칙 ID가 붙은 인용은 §1의 `§11` 행이 아니라 §2의 그 ID 행에서 셌다.

## 1. 옛 API.md의 절

| 옛 참조 | 내용 | 새 절 | 인용 수 |
|---|---|---|---|
| 머리말 (`import stardust_pandas as sdpd`) | 결정 이력, `[pandas와 다름]` 표시 | `sd.frame` (머리말). 이력은 옮기지 않는다 | 0 |
| §1 (S1) | NaN은 값, 리덕션·정렬의 NaN, 키 비교(NaN끼리, `-0.0`과 `0.0`, 키 정규화) | NaN §1.3, 키 비교 §1.6 | 4 (C++ 2: `ops/hash.hpp`; Python 1: `isna`; 테스트 1) |
| §2 (S2) | NA: 산술·비교, Kleene 논리, 필터에서 False, `bool(NA)`, 모두 NA인 리덕션, `ddof=1` | §1.4 (모두 NA인 리덕션은 §6.4도) | 6 |
| §3 (S3) | 정수 연산: `int / int`, wrap, `// 0`·`% 0`·`** 음수`는 NA, float 0 나누기 | §1.5 | 6 |
| §4 (S4) | groupby의 `sort`, `dropna`, 키의 위치 | §7.2 | 1 (`groupby.py`, 키 컬럼) |
| §5 (S5) | merge의 결과 행 순서, NA 키, 접미사 | §9.1 | 7 |
| §6 (S6) | 라벨: Index dtype, Index의 NA, 중복 라벨, 컬럼 이름, 정렬(alignment)의 라벨 순서 | §1.7 (`index_col`은 §10.1, 헤더의 중복 이름은 §10.2도) | 10 |
| §7 (S7) | `[]`: `df["a"]`, `df[[...]]`, `df[mask]`, `s[x]`는 `.loc` | §5.3 | 2 |
| §8 (S8) | 리덕션은 Python 스칼라, `NA`, 경고 없음, 대입으로 dtype이 바뀌면 에러 | §1.8 | 0 |
| §9 | repr | §11 | 4 |
| §10 | 생성과 변환: list의 dtype 추론, 생성자 `dtype=`(손실 없는 변환), numpy 입력, DataFrame 입력, `Series.name`, `to_numpy` | 추론 §2.2, 손실 없는 변환 §2.3, numpy §2.4, DataFrame 입력·`columns=` §2.5, `name` §2.6, `to_numpy` §2.7, dtype 표기 §1.2 | 19 |
| §11 | 시그니처 일관성 (C1–C7) | §1.1. 각 ID는 §2의 표 | 1 (`API.md §10, §11`, `tests/test_construction.py`). ID가 붙은 12회는 §2 |
| §12 | 원소별 연산: 피연산자, 산술, 비교, 논리, `astype`, `isna`/`notna`, `fillna`, `dropna`, 마스크 필터, M2의 Index | 피연산자·결과 이름 §3.1, 산술 §3.2, 비교 §3.3, 논리 §3.4, `astype`과 엄격한 파싱 §3.5, 결측 §3.6, 마스크 필터 §5.5, Index §5.1 | 22 |
| §13 | 중복과 고유값: 키가 같다는 것, 대표값, 시그니처, `value_counts`, factorize | 키·대표값 §1.6, 함수 §4.1, 결과 §4.2, `value_counts` §4.3 | 15 |
| §14 | Index, 생성자의 `index=`, `reindex`, `columns`/`dtypes`/`nunique`의 반환 타입, 선택, 정렬(alignment), 대입, 반복과 `in` | Index §5.1, `index=` §2.1, `reindex` §5.2, `columns`·`dtypes` §2.6, `nunique` §4.2, `[]` §5.3, `.loc`·`.iloc` §5.4, 마스크 §5.5, 정렬 §5.6, 대입 §5.7, 반복과 `in` §5.8 | 36 |
| §15 | 리덕션: 시그니처, 반환값, dtype별 규칙, NA, NaN, `-0.0`, float 합의 순서, `var` | 시그니처 §6.1, 반환값 §6.2, dtype 표 §6.3, NA·NaN·`-0.0` §6.4, 합의 순서·`var` §6.5 | 15 |
| §16 | groupby: `by`, 결과, 그룹 순서, `dropna`, 메서드, `agg`, float 합의 순서 | `by`·값 컬럼 §7.1, 결과·그룹 순서·`dropna` §7.2, 메서드 §7.3, `agg` §7.4, 합의 순서 §7.5 | 11 |
| §17 | 정렬(sort): 시그니처, `ascending`/`na_position`, 순서, 안정성, 알고리즘 | 시그니처·인자 §8.1, 순서·안정성·결과 §8.2, 알고리즘 §8.3 | 7 |
| §18 | merge와 concat | merge §9.1, concat §9.2 | 13 |
| §19 | read_csv | 인자 §10.1, 파일 형식 §10.2, NA §10.3, 타입 추론 §10.4 | 12 |
| §20 | 스레드: API, 결정성, 에러, GIL 임계값, 동시 대입 | §12 (공통 규칙은 API.md §3) | 9 |
| §21 | 공개 API 목록과 API 표면 테스트의 규칙 | 목록은 각 절의 시그니처 블록, 테스트 규칙은 머리말 | 2 (`tests/test_api_surface.py`) |
| §22 | 미확정 ("없음") | §13 (새로 쓴 미확정 항목) | 0 |

## 2. 규칙 ID

| ID | 출처 | 내용 | 새 절 | 인용 수 |
|---|---|---|---|---|
| S1–S8 | 옛 API.md §1–§8의 제목 | 위 표의 §1–§8 행과 같다 | 위 표 | 0 |
| C1 | 옛 API.md §11 | 별칭을 두지 않는다. `merge`는 함수와 메서드, `concat`은 함수만 | §1.1 | 0 |
| C2 | 옛 API.md §11 | 주 인자만 위치 인자, 나머지는 keyword-only | §1.1 | 8 (`API.md §11 C2` 7, `C2`만 1: `tests/test_unique.py`) |
| C3 | 옛 API.md §11 | 리덕션 인자는 Series, DataFrame, GroupBy에서 같다. `count`는 인자가 없다 | §6.1 (§1.1이 가리킨다), GroupBy는 §7.3 | 3 (`API.md §11 C3` 2, `C3`만 1: `tests/test_reduce.py`) |
| C4 | 옛 API.md §11 | 의미가 없는 인자를 두지 않는다 (`copy`, `inplace`, `level`, `kind`, `errors`, `engine`, Series의 `axis`) | §1.1, `sort_*`의 없는 인자는 §8.1 | 2 (`C4`만, `tests/test_sort.py`) |
| C5 | 옛 API.md §11 | `keep`은 `"first"`, `"last"`, `"none"` | §1.1, §4.1 | 1 |
| C6 | 옛 API.md §11 | `Series.unique()`는 Series, `DataFrame.value_counts()`는 DataFrame | `unique` §4.2, `value_counts` §4.3 | 1 (`series.py`, `unique`) |
| C7 | 옛 API.md §11 | 연산은 연산자로만. `add`, `abs()` 같은 메서드 없음 | §1.1 | 1 (`_ops.py`) |
| A6 | 옛 API.md §6 (4라운드 결정) | 정렬(alignment) 결과의 라벨 순서: 왼쪽 순서, 그 다음 오른쪽에만 있는 라벨 | §1.7, 정렬이 일어나는 곳은 §5.6 | 2 (`A6`만: `_select.py`, `tests/test_indexing.py`) |

## 3. 대응하지 못한 것

| 참조 | 이유 | 제안 |
|---|---|---|
| `tests/test_api_surface.py`의 `## 21. `, `## 22. ` (문자열), `§21`만 쓴 2회, 파일 경로 `API.md` | 옛 문서의 절 제목을 잘라 읽는 테스트다. 새 문서에는 §21이 없다 | `tests/_api_doc.py`의 `signature_blocks`로 `docs/api/frame.md` 전체의 시그니처 블록을 읽게 바꾼다. 모듈 이름은 접두어 없이(`merge(...)`, `NA`), 멤버는 `Owner.name` 형태다 (머리말) |
| `src/stardust_pandas/__init__.py`의 "specified in API.md" (절 없음, 1회) | 문서 전체를 가리킨다 | `docs/api/frame.md` (옮긴 `__init__.py`는 이미 그렇게 바뀌었다) |
| 옛 §20의 `set_num_threads`, `get_num_threads` | `sd.frame`에서 빠졌다 | API.md §3 (`sd.set_num_threads`, `sd.get_num_threads`) |
| 옛 §12의 "M2에서는 두 Index가 같아야 한다", "M2의 `Index`" | M4에서 정렬(alignment)과 Index로 바뀐 임시 규칙이다 | 지금 규칙: §3.1, §5.1, §5.5, §5.6 |
| 옛 §13의 "`DataFrame.nunique`는 M4까지 dict" | M4에서 int64 Series로 바뀌었다 | §4.2 |
| 옛 §6의 Index dtype `datetime64` | datetime64 dtype이 없다 | §13 (미확정) |
| `DESIGN.md`, `ARCHITECTURE.md`, `sdnp DESIGN`, M1–M12 마일스톤 이름 | 이 표의 범위가 아니다 | 따로 다룬다 |
