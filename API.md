# stardust API

`import stardust as sd`. 영역마다 API 문서가 하나씩 있다.

| 영역 | 이름공간 | 문서 |
|---|---|---|
| 배열 | `sd` | [docs/api/array.md](docs/api/array.md) |
| 선형대수 | `sd.linalg` | M2에서 옮긴다 |
| 데이터프레임 | `sd.frame` | M5에서 옮긴다 |

## 공통 규칙
- 라이브러리 전체의 공통 규칙(dtype 객체, 에러 종류, 스레드 수와 결정성)은 지금 [docs/api/array.md §0](docs/api/array.md)에 있다.
  - 두 번째 영역이 들어오는 M2에서 이 문서로 옮긴다. 그때 배열에만 해당하는 규칙과 라이브러리 전체의 규칙을 나눈다.
- 영역 사이의 일관성이 가장 앞선 원칙이다 (DESIGN.md §1.4). 같은 개념은 어느 영역에서든 같은 이름, 같은 시그니처, 같은 동작이다.
  - 기준은 Python Array API 관례다 (DESIGN.md §8.1).
