# sd.spatial API: 계산 기하 (초안)

> 상태: **초안** (2026-11-23, SPT8 마일스톤). 아래 이름은 아직 구현되어 있지 않다: `Delaunay`는 SPT9(d = 2, 구현됨, 2026-11-23)·SPT10(d = 3, 구현됨, 2026-11-23), `ConvexHull`은 SPT11(구현됨, 2026-11-23), `Voronoi`는 SPT12(구현됨, 2026-11-23) 마일스톤에서 생긴다. SPT14 마일스톤에서 이 문서를 [spatial.md](spatial.md)에 합치고 이 파일을 지운다(SPT118). 그때까지 `tests/spatial/test_api_surface.py`는 구현된 이름의 시그니처만 이 파일과 견준다(SPT137).

- 설계와 결정의 근거는 [docs/design/spatial_geometry.md](../design/spatial_geometry.md)(§1.1 SPT90–SPT183), 공통 규칙은 [API.md](../../API.md), 이 영역의 이미 있는 이름(거리, 이웃 찾기, `KDTree`)은 [spatial.md](spatial.md)다.
- 비교 대상은 scipy 1.18.1의 `scipy.spatial.Delaunay`, `ConvexHull`, `Voronoi`(안은 Qhull)다.
- 언어 표시가 없는 코드 블록의 줄은 spatial.md와 같은 꼴이다: 모듈의 이름(`name(...)`, 클래스는 생성자), 메서드(`Class.method(...)`), 속성(`Class.attr`).

## §1. 공통 규칙

### §1.1 이름과 호출 모양
- 세 클래스는 `sd.spatial`에 바로 있다(SPT5, SPT6): `sd.spatial.Delaunay`, `sd.spatial.ConvexHull`, `sd.spatial.Voronoi`. 이름과 속성 이름은 scipy의 것이다.
- 생성자는 점만 받는다(위치로만). Qhull의 옵션(`qhull_options`, `furthest_site`, `incremental`)과 `add_points`, `close`는 없다(SPT99): 주면 모르는 키워드의 `TypeError`다(`Delaunay(): unexpected keyword argument 'qhull_options'`).
- 차원은 2와 3뿐이다(SPT101).
- 스레드 수는 `sd.set_num_threads` 하나다. 같은 점이면 스레드 수와 상관없이 모든 배열이 같은 바이트다.

### §1.2 입력 (SPT98)
- `points`는 (n, d), d = 2 또는 3이다. API.md §5의 상위 함수 입력(`sd.ndarray`, 수의 시퀀스, numpy 배열, `DataFrame`)을 받고 float64로 계산한다. bool·int64는 float64로 바꾼다.
- complex128은 `TypeError`다(`[scipy와 다름]`: scipy는 허수부를 버리고 `ComplexWarning`을 낸다).
- NaN·inf는 `ValueError`다(`Delaunay(): 'points' must be finite`). scipy는 NaN을 `ValueError`로, inf를 `QhullError`(QH6019)로 막는다.
- 2차원 배열이 아니거나 열이 2·3개가 아니면 `ValueError`다(`Delaunay(): 'points' must have 2 or 3 columns, not 4`). `[scipy와 다름]`: scipy는 4차원 이상도 받는다.
- 서로 다른 점이 d + 1개보다 적거나 모두 한 직선(d = 3이면 한 평면) 위면 `ValueError`다(SPT97): `Delaunay(): needs at least 3 distinct points, got 2`, `Delaunay(): 'points' are collinear`, `Delaunay(): 'points' are coplanar`. 판정은 정확하다(유리수로 읽은 좌표의 답). scipy는 같은 자리에서 `QhullError`(QH6154, QH6214)다.
- 점은 언제나 복사한다. 결과의 배열은 쓰기 금지다. `[scipy와 다름]`: scipy의 `Delaunay(P).points`는 `P`와 메모리를 함께 쓴다.
- 좌표의 크기에 제한이 없다: 1e300과 1e-300이 섞인 점, 비정상 수, `DBL_MAX` 둘레의 점에서도 조합 구조가 정확하다(SPT100).
- 같은 좌표의 점(−0.0과 0.0은 같다)은 합친다(SPT96): 가장 작은 색인이 대표이고 삼각분할·껍질·보로노이의 꼭짓점은 대표뿐이다.

### §1.3 결과의 꼴 (SPT93, SPT94)
- 색인은 int64다(`[scipy와 다름]`: scipy는 int32). 없음은 −1이다.
- **표준 꼴**: 단체(삼각형, 사면체)는 양의 방향이다(d = 2는 반시계, d = 3은 det[b − a, c − a, d − a] > 0). 단체 안은 가장 작은 색인이 먼저이고(d = 3은 그다음도 가장 작은 것), 단체의 차례는 행의 사전 순서다. 경계 면은 바깥을 향하고(d = 2는 내부가 왼쪽인 변, d = 3은 밖에서 볼 때 반시계) 가장 작은 색인이 먼저이고 행의 사전 순서다. 색인 목록은 오름차순이다.
- **동점**(SPT95): 한 원(구) 위에 d + 2개 이상의 점이 있으면 삼각분할이 여럿인데, 좌표의 사전 순서로 정한 규칙 하나로 고른다. 그래서 삼각분할은 점 집합의 함수다: 입력의 행을 섞으면 색인만 바뀐다. `[scipy와 다름]`: scipy의 차례와 방향은 Qhull이 만든 순서라 약속이 없다(3차원 단체의 절반이 음의 방향이다).
- **정확성**: 어느 단체가 있는지, 어느 점이 껍질의 꼭짓점인지, 어느 점이 이웃인지, 평가점이 어느 단체 안인지는 입력 float64를 유리수로 읽은 정확한 답이다. 허용오차 인자가 없다. `[scipy와 다름]`: scipy는 가까운 점을 버리고(10⁸ + 0.1 × 격자의 900점에서 896점) 거의 퇴화인 입력에서 Delaunay가 아닌 삼각형과 뒤집힌 삼각형을 낸다(`tests/spatial/data/scipy_geometry.json`의 `exact/…`).

## §2. scipy.spatial → sd.spatial

| scipy 1.18.1 | sd | 마일스톤 |
|---|---|---|
| `Delaunay(points, furthest_site=False, incremental=False, qhull_options=None)` | `Delaunay(points, /)` | SPT9, SPT10 |
| `Delaunay.points`, `ndim`, `npoints`, `nsimplex`, `min_bound`, `max_bound` | 같은 이름 | SPT9 |
| `Delaunay.simplices`, `neighbors`, `convex_hull`, `vertex_to_simplex`, `coplanar` (int32) | 같은 이름, int64, 표준 꼴 | SPT9 |
| `Delaunay.vertex_neighbor_vertices` | 같은 이름, `(indptr, indices)`, 행 안은 오름차순 | SPT9 |
| `Delaunay.transform` | 같은 이름 | SPT9 |
| `Delaunay.find_simplex(xi, bruteforce=False, tol=None)` | `Delaunay.find_simplex(xi, /)` | SPT9, SPT10 |
| `Delaunay.equations`, `paraboloid_scale`, `paraboloid_shift`, `good`, `furthest_site`, `plane_distance`, `lift_points`, `add_points`, `close` | 없음 | — |
| `ConvexHull(points, incremental=False, qhull_options=None)` | `ConvexHull(points, /)` | SPT11 |
| `ConvexHull.points`, `ndim`, `npoints`, `nsimplex`, `min_bound`, `max_bound`, `vertices`, `simplices`, `neighbors`, `equations`, `area`, `volume` | 같은 이름 | SPT11 |
| `ConvexHull.coplanar`, `good`, `add_points`, `close` | 없음 | — |
| `Voronoi(points, furthest_site=False, incremental=False, qhull_options=None)` | `Voronoi(points, /)` | SPT12 |
| `Voronoi.points`, `ndim`, `npoints`, `min_bound`, `max_bound`, `vertices`, `ridge_points`, `point_region` | 같은 이름 | SPT12 |
| `Voronoi.regions`, `ridge_vertices` (list의 list) | 같은 이름, `(indptr, indices)` 짝 | SPT12 |
| `Voronoi.ridge_dict`, `furthest_site`, `add_points`, `close` | 없음 | — |
| `SphericalVoronoi`, `HalfspaceIntersection`, `tsearch`, `delaunay_plot_2d`, `convex_hull_plot_2d`, `voronoi_plot_2d` | 없음(나중 후보; 그림은 `sd.plot`) | — |

## §3. `Delaunay`

```
Delaunay(points, /)
Delaunay.find_simplex(xi, /)
Delaunay.points
Delaunay.simplices
Delaunay.neighbors
Delaunay.convex_hull
Delaunay.vertex_to_simplex
Delaunay.vertex_neighbor_vertices
Delaunay.coplanar
Delaunay.transform
Delaunay.ndim
Delaunay.npoints
Delaunay.nsimplex
Delaunay.min_bound
Delaunay.max_bound
```

- `points` (n, d) float64: 입력의 쓰기 금지 사본. `ndim`은 d, `npoints`는 n, `nsimplex`는 t(Python `int`), `min_bound`·`max_bound`는 (d,) float64다.
- `simplices` (t, d + 1) int64: 표준 꼴의 단체. d = 2에서 서로 다른 점이 m개이고 경계 위의 점이 b개면 t = 2m − 2 − b다. 한 직선 위의 경계점도 꼭짓점이다.
- `neighbors` (t, d + 1) int64: `neighbors[i, k]`는 단체 i에서 꼭짓점 k의 맞은편 면을 함께 가진 단체, 없으면 −1이다(scipy와 같다).
- `convex_hull` (면 수, d) int64: 삼각분할의 경계 면(표준 꼴). 변·면 위의 경계점에서 나뉘므로 `ConvexHull.simplices`와 다를 수 있다.
- `vertex_to_simplex` (n,) int64: 그 점을 가진 가장 작은 번호의 단체. 같은 점은 대표의 값이다.
- `vertex_neighbor_vertices`: `(indptr, indices)` int64 짝. 점 i의 이웃 꼭짓점은 `indices[indptr[i]:indptr[i + 1]]`(오름차순)다. 대표가 아닌 같은 점은 빈 행이다. 두 배열은 CSR의 것이라 `sd.sparse.csr_array`에 그대로 들어간다.
- `coplanar` (k, 3) int64: 삼각분할에 들지 않은 점의 행 (점, 단체, 꼭짓점). sd에서는 대표가 아닌 같은 점뿐이다(꼭짓점은 대표, 단체는 `vertex_to_simplex[대표]`; 점의 오름차순).
- `transform` (t, d + 1, d) float64: scipy의 뜻이다. `transform[i, :d, :]`는 T⁻¹, `transform[i, d, :]`는 꼭짓점 d다. 처음 읽을 때 만든다. 평평한 단체가 없으므로 NaN이 없다.
- `find_simplex(xi)`: `xi`는 (…, d)이고 결과는 그 앞 모양의 int64 배열이다(점 하나 (d,)는 Python `int`). 평가점을 품은 **닫힌** 단체의 번호이고, 여러 단체가 품으면 가장 작은 번호다. 볼록 껍질 밖(경계 위는 안)과 NaN·inf 좌표는 −1이다. 마지막 축이 d가 아니면 `ValueError`다. `[scipy와 다름]`: `tol`·`bruteforce`가 없다. scipy는 허용오차 때문에 단위 정사각형 밖 10⁻¹⁵의 점을 안으로 치고(10⁻¹⁴은 밖), 걸친 점에서 고르는 단체를 정해 두지 않는다.
- `repr`는 `Delaunay(n=1000, d=2, nsimplex=1975)`다. 피클과 `copy`는 점으로 다시 만든다.
- d = 3의 사면체 수는 입력에 따라 O(n)에서 O(n²)이다(균등 입력에서 약 6.7n, 모멘트 곡선 (t, t², t³) 위의 점은 n²에 비례). 만드는 동안 사면체를 세어 이 기기의 물리 메모리에 들지 않으면 `MemoryError`다(`Delaunay(): the triangulation is larger than this machine's physical memory`).

## §4. `ConvexHull`

```
ConvexHull(points, /)
ConvexHull.points
ConvexHull.vertices
ConvexHull.simplices
ConvexHull.neighbors
ConvexHull.equations
ConvexHull.area
ConvexHull.volume
ConvexHull.ndim
ConvexHull.npoints
ConvexHull.nsimplex
ConvexHull.min_bound
ConvexHull.max_bound
```

- `vertices` (h,) int64: 극점만이다. 변이나 면의 안쪽에 놓인 경계점과 같은 점의 대표 아닌 색인은 들지 않는다(scipy의 기본과 같다). d = 2는 반시계이고 가장 작은 색인에서 시작한다. d = 3은 오름차순이다.
- `simplices` (면 수, d) int64: d = 2는 이웃한 극점의 변, d = 3은 삼각형이다(표준 꼴). 평평한 면(꼭짓점이 넷 이상인 다각형)은 그 면의 가장 작은 색인에서 부채꼴로 나눈다. `[scipy와 다름]`: scipy의 면은 방향이 섞여 있다(무작위 1,000점: 2차원 변 16개 중 7개, 3차원 삼각형 142개 중 71개만 바깥을 향한다).
- `neighbors` (면 수, d) int64: `neighbors[i, k]`는 면 i에서 꼭짓점 k의 맞은편 이웃 면이다. −1이 없다.
- `equations` (면 수, d + 1) float64: 행 (법선, 절편). 법선은 바깥을 향하는 단위 벡터이고 내부의 점 x에서 법선·x + 절편 < 0이다(scipy와 같다). 한 평평한 면에서 나온 삼각형들은 같은 행이다.
- `area`, `volume`(Python `float`): scipy의 뜻이다. d = 3은 겉넓이와 부피, d = 2는 **둘레**와 **넓이**다.
- `repr`는 `ConvexHull(n=1000, d=3, nvertices=87)`이다. 퇴화 입력의 에러와 피클은 `Delaunay`와 같다.

## §5. `Voronoi`

```
Voronoi(points, /)
Voronoi.points
Voronoi.vertices
Voronoi.ridge_points
Voronoi.ridge_vertices
Voronoi.regions
Voronoi.point_region
Voronoi.ndim
Voronoi.npoints
Voronoi.min_bound
Voronoi.max_bound
```

- 입력 점 집합의 참 보로노이 그림이다. 한 원(구) 위의 d + 2개 이상의 점은 꼭짓점 하나를 함께 갖고, 길이(넓이)가 0인 능선은 없다(SPT111). 3 × 3 격자의 꼭짓점은 4개, 능선은 12개다(scipy와 같다).
- `vertices` (꼭짓점 수, d) float64: Delaunay 단체의 외심. 같은 외접원(구)의 단체들은 꼭짓점 하나다.
- `vertices`의 번호는 그 외접원(구)을 가진 단체 가운데 가장 작은 번호(`Delaunay(points).simplices`의 행 번호)의 차례다. 좌표의 한계는 §6이다. 외심이 float64를 넘으면 ±inf다(아주 큰 좌표의 거의 한 직선인 점).
- `regions`: `(indptr, indices)` int64 짝이고 영역 i는 점 i의 것이다(SPT110). `indices[indptr[i]:indptr[i + 1]]`가 그 영역의 꼭짓점 번호이고 끝없는 영역(껍질 위의 점의 것)은 −1을 하나 갖는다. d = 2는 점 둘레의 반시계 차례(−1이 있으면 −1이, 없으면 가장 작은 번호가 먼저), d = 3은 오름차순이다. 대표가 아닌 같은 점의 행은 비어 있다.
- `point_region` (n,) int64: 점의 영역 번호. 대표인 점은 자기 색인, 같은 점은 대표의 색인이다. scipy 코드의 `vor.regions[vor.point_region[i]]`는 `indices[indptr[r]:indptr[r + 1]]`(`r = point_region[i]`)이다.
- `ridge_points` (능선 수, 2) int64: 능선을 함께 가진 점의 짝 (i, j), i < j, 사전 순서. 대표가 아닌 같은 점은 나오지 않는다.
- `ridge_vertices`: `(indptr, indices)` int64 짝. 능선 k의 꼭짓점 번호이고 끝없는 능선은 −1을 하나 가지며 그것이 먼저다. d = 2는 꼭짓점 둘(오름차순)이거나 −1과 하나다. d = 3은 `ridge_points[k, 0]`에서 `ridge_points[k, 1]`을 볼 때 반시계의 차례이고 −1이 없으면 가장 작은 번호가 먼저다.
- 능선은 두 영역이 d − 1차원의 면을 함께 가질 때만 있다: 길이(넓이)가 0인 것, 예를 들어 한 원 위의 네 점의 대각선이나 평평한 껍질 면의 대각선(3 × 3 × 3 격자의 면 대각선)은 없다.
- `[scipy와 다름]`: scipy의 `regions`·`ridge_vertices`는 list의 list이고 `regions`에 빈 영역 하나가 끼어 있다. scipy는 3차원에서 유한한 꼭짓점이 하나뿐인 끝없는 능선(사면체 하나에만 든 Delaunay 변)을 빼놓는다(무작위 1,000점에서 7,440개 중 23개). sd는 두 영역이 2차원 면을 함께 가지면 모두 낸다. `ridge_dict`는 없다.
- `repr`는 `Voronoi(n=1000, d=2, nvertices=1975)`다. 퇴화 입력의 에러(`Voronoi(): 'points' are collinear` 등), 3차원 삼각분할이 물리 메모리에 들지 않을 때의 `MemoryError`, 피클은 `Delaunay`와 같다.

## §6. 정확도

조합 구조는 정확하다(§1.3). 실수 출력의 계약은 [docs/design/spatial_geometry.md](../design/spatial_geometry.md) §4의 표다(u = 2⁻⁵³): 넓이·부피·둘레, `equations`, 보로노이 꼭짓점, `transform`의 한계가 단체의 모양(가늘거나 납작함)에 기대지 않는다.
