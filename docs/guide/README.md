# stardust 안내서

stardust를 처음 보는 사람을 위한 책이다. 두 가지를 함께 다룬다.
- **쓰는 법**: 배열 `sd`, 선형대수 `sd.linalg`, 데이터프레임 `sd.frame`을 함께 쓰는 법
- **만든 법**: 라이브러리가 안에서 어떻게 되어 있고, 왜 그렇게 정했는지

세 교육용 프로젝트(stardust-numpy, stardust-linalg, stardust-pandas)를 하나로 합친 뒤에 새로 썼다. 옛 프로젝트의 안내서(각 13장)는 영역마다 따로였다. 이 책은 영역을 넘나든다. 같은 부품(Buffer, 스레드풀, pairwise 합)과 같은 규칙(dtype, 에러, 결정성)이 세 영역에 어떻게 함께 쓰이는지가 중심이다.

## 누구를 위한 글인가
- Python을 쓸 줄 안다. numpy나 pandas를 조금 써 봤으면 더 좋다.
- C++은 읽을 수 있으면 좋지만 없어도 된다. 코드는 위치(파일과 함수 이름)만 적고, 본문은 그림과 표로 설명한다.
- 캐시, SIMD(NEON), 부동소수점 반올림, 스레드풀, GIL, LAPACK은 몰라도 된다. 처음 나올 때 설명한다.

## 다른 문서와 무엇이 다른가
| 문서 | 성격 | 언제 읽나 |
|---|---|---|
| **이 안내서** (`docs/guide/`) | 이야기. 규칙과 구조를 이어서 설명하고 "왜"를 말한다 | 처음 볼 때, 전체를 이해하고 싶을 때 |
| [API.md](../../API.md), `docs/api/<영역>.md` | 공개 API의 기준. 함수 목록, 시그니처, 동작 규칙 | 함수 하나의 정확한 규칙을 확인할 때 |
| [DESIGN.md](../../DESIGN.md), [ARCHITECTURE.md](../../ARCHITECTURE.md), `docs/design/<영역>.md` | 설계의 기준. 확정한 결정, 구조, 측정 근거 | 어떤 결정의 근거와 수치를 볼 때 |
| `docs/learn/M<n>-*.md` | 통합 과정의 해설. 마일스톤마다 무엇을 바꾸고 무엇을 쟀는지 | 결정이 나오기까지의 과정이 궁금할 때 |

- 안내서는 세부를 되풀이하지 않는다. 장마다 끝에 "더 읽을 곳"이 있고, 본문도 `docs/design/array.md §6.2`처럼 절을 가리킨다.
- 안내서와 기준 문서가 다르면 기준 문서와 코드가 맞다.
- 수치는 Apple M4 Pro 한 대에서 잰 값이다. 출처는 그 수치를 적은 설계 문서다.

## 차례
| 장 | 제목 | 핵심 질문 |
|---|---|---|
| [1장](01-overview.md) | 전체 그림 | 무엇이 있고, 세 영역은 어떻게 한 라이브러리가 되었나? `x + y` 한 줄은 어디를 지나가나? |
| [2장](02-ndarray-and-views.md) | 배열의 정체: Buffer, strides, 뷰 | 배열은 메모리 한 덩어리와 그것을 읽는 방법이다. 복사 없이 전치와 슬라이스를 하는 방법은? |
| [3장](03-iterator-and-dtypes.md) | 브로드캐스팅, 이터레이터, dtype | N차원 연산을 1차원 루프로 바꾸는 방법은? 결과 dtype과 변환 규칙은 어떻게 정했나? |
| [4장](04-elementwise.md) | 원소별 연산 | `sin(x)`, `x * y`는 어떻게 빠르고, numpy와 비트까지 같으려면 무엇이 필요한가? |
| [5장](05-reductions.md) | 리덕션과 pairwise 합 | 합 하나에 왜 여러 전략이 필요하고, 덧셈 순서는 왜 중요한가? |
| [6장](06-threads-and-determinism.md) | 스레드풀과 결정성 | 풀 하나를 세 영역이 어떻게 나눠 쓰고, 스레드 수와 무관하게 같은 비트를 내려면? |
| [7장](07-indexing-and-sorting.md) | 인덱싱, 정렬, unique | `x[mask]`와 `sort`는 어떻게 동작하고, NaN은 어디에 놓이나? |
| [8장](08-linalg-rules.md) | `sd.linalg` 쓰기 | 37개 함수가 함께 따르는 규칙(배치, dtype, 실패, 결과 레이아웃)은 무엇인가? |
| [9장](09-linalg-inside.md) | LAPACK 위의 층 | 행렬을 LAPACK에 넘기기까지 무엇을 하고, 배치는 언제 나누나? |
| [10장](10-frame-model.md) | `sd.frame`의 데이터 모델 | Column, NA, Copy-on-Write는 무엇이고 왜 배열 위에 올리지 않았나? |
| [11장](11-frame-engines.md) | 해시, groupby, 정렬, merge, CSV | "같은 키"와 "순서"를 한 곳에서 정하면 무엇이 쉬워지나? |
| [12장](12-boundaries.md) | 영역의 경계 | 확장 모듈 하나, linalg의 경계 없애기, frame과 배열을 잇는 파일 하나 |
| [13장](13-consistency-and-safety-nets.md) | 일관성과 안전망 | 세 영역의 규칙을 어떻게 하나로 맞췄고, 합치는 동안 무엇이 결과를 지켰나? |

## 읽는 순서
- **처음이라면 1장 → 2장 → 3장.** 1장은 지도이고, 2장과 3장은 뒤의 모든 장이 기대는 기초다.
- **배열의 계산 엔진은 4장 → 5장 → 6장.** 6장(스레드)은 5장 뒤에 읽는다. 합의 순서가 왜 어려운지 알아야 결정성 규칙이 자연스럽다.
- **7장**은 개별 연산이다. 관심이 있으면 읽는다.
- **linalg만 쓴다면** 1장, 2장 다음 8장 → 9장. 9장은 6장의 풀을 안다고 가정한다.
- **frame만 쓴다면** 1장 다음 10장 → 11장. 11장의 병렬화는 6장을 함께 보면 좋다.
- **12장과 13장**은 세 영역을 다 본 뒤에 읽으면 가장 많이 얻는다. 합친 라이브러리만의 이야기다.

## 예제를 돌리려면
저장소 루트에서 빌드한다 (루트 [README.md](../../README.md)).
```bash
python3.15 -m venv .venv && .venv/bin/pip install nanobind scikit-build-core pytest hypothesis numpy scipy pandas polars
.venv/bin/pip install --no-build-isolation -e .
```
- 이 책의 ```` ```python ```` 블록은 **블록 하나가 그대로 돌아가는 프로그램**이다. 블록마다 `import stardust as sd`로 시작한다.
- `print(...)  # 값`의 주석은 실제 출력이다. 이 책의 예제는 모두 지금의 라이브러리로 돌려 출력을 맞춰 두었다.
- `x._flags()`처럼 밑줄로 시작하는 이름은 공개 API가 아니다. 테스트와 해설을 위한 것이라 이 책에서는 쓴다.

```python
import stardust as sd

x = sd.arange(6.0).reshape((2, 3))
print(x.strides)  # (24, 8)
```

코드를 함께 보려면 본문의 파일과 함수 이름으로 찾는다. 줄 번호는 적지 않는다.
```bash
grep -n "NdIter::plan" csrc/array/iter.cpp
```
