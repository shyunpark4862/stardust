# stardust-pandas

pandas의 일부를 처음부터 다시 만드는 교육용 프로젝트다 ("rebuilding the wheel").
- 계산 코어는 C++20, Python 바인딩은 nanobind로 만든다. `Series`와 `DataFrame`은 Python 클래스다.
- 대상 기기는 Apple M4 Pro 한 대다.
- 설계를 먼저 확정하고, 마일스톤마다 구현하고 측정하고 해설을 쓴다.

```python
import stardust_pandas as sdpd
```

## 무엇이 다른가
동작 규칙은 [API.md](API.md)에 있다. pandas와 다른 점에는 **[pandas와 다름]** 표시를 달았다.
- **결측값은 `NA` 하나다.** 모든 dtype이 validity 비트맵을 가진다. int 컬럼에 결측이 생겨도 float으로 바뀌지 않는다.
- **NaN은 결측이 아니라 평범한 float 값이다.**
- **Copy-on-Write를 쓴다.** `inplace=`와 chained assignment는 없다.
- **MultiIndex가 없다.**
- **경고를 내지 않는다.** 문제가 있으면 에러를 낸다.
- **결과는 스레드 수와 무관하게 비트까지 같다.**

## 빌드와 테스트
macOS (Apple Silicon), Python 3.15에서 확인했다.
```bash
python3.15 -m venv .venv && .venv/bin/pip install nanobind scikit-build-core pytest hypothesis numpy pandas polars
```
```bash
.venv/bin/pip install --no-build-isolation -e .
```
```bash
.venv/bin/python -m pytest -q
```

## 진행 상태
- M0 (환경과 뼈대) 완료
- M1 (Column 코어: `Series`, `DataFrame`, `RangeIndex`, 생성, `to_numpy`, `head`/`tail`, repr) 완료
- M2 (원소별 연산: 산술, 비교, Kleene 논리, `astype`, `isna`/`fillna`/`dropna`, 마스크 필터) 완료
- M3 (해싱: factorize, `unique`, `nunique`, `value_counts`, `duplicated`, `drop_duplicates`) 완료
- M4 (Index와 조회 엔진, `.loc`/`.iloc`/`[]`, 정렬, `reindex`, 대입 + Copy-on-Write) 완료
- M5 (리덕션: `sum`, `mean`, `min`, `max`, `count`, `std`, `var`) 완료
- M6 (groupby: 단일/다중 키, 리덕션, `first`/`last`/`size`, `agg`) 완료
