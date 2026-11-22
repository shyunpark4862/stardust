"""The test graphs of sd.graph (docs/design/graph.md §7.1): generated from formulas and fixed seeds,
nothing downloaded. The tests, the reference tool (tools/graph_ref.py), the probe of scipy and
networkx (tools/graph_probe.py) and the measurements take their graphs from here, by id.

A graph is an adjacency matrix in canonical CSR (int64 indptr and indices, the column indices of a
row ascending and distinct) with float64, int64 or bool values: the entry (i, j) is the edge i -> j
and its weight, a stored 0 an edge of weight 0. A few graphs exist only as dense matrices (`dense`,
where 0, NaN and +-inf mean no edge) and the assignment problems are dense cost matrices (`cost`,
where every entry is a pair and +inf one that cannot be chosen). Kinds, by tag:

    shape       empty (n = 0, 1), isolated nodes, path, cycle, star, complete, 2-D and 3-D grids (ties by
                the thousand), ladder (2^(n/2) shortest paths), DAG, Erdos-Renyi directed and undirected,
                a long-tailed degree distribution, several components, self loops, one-way edges and
                two-way edges of different weights (the undirected rule)
    int         small integer weights (ties); `bigint` integers whose sums stay below 2^53 and past it
    float       random weights over [0.01, 1.01) and over 1e-8 .. 1e8, dyadic rationals, weights that
                are absorbed (fl(d + w) = d)
    zero        stored zeros: zero-weight edges and a zero-weight cycle
    negative    negative weights with no cycle, a negative cycle reachable from node 0, one that is not,
                a cycle whose exact sum is 0
    bool        bool values
    nonfinite   stored NaN, +inf, -inf (errors for the weighted functions) and the dense 0, NaN, inf rules
    bipartite   rectangular biadjacency matrices: matchings, no full matching, tied and float costs
    cost        dense cost matrices of linear_sum_assignment (ties, rectangular, +inf, infeasible)
    flow        integer capacities: small, near the int64 limits, past int32 (scipy's truncation)
    dangling    nodes with no out-edge (PageRank)

    import tests.graph.graphs as G
    g = G.get("grid2d/6x6")        # cached; G.ids(tag="int") lists ids
    g.indptr, g.indices, g.data    # numpy arrays; g.dense() the dense matrix, g.edges() the edge list
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib

import numpy as np

SEED = 20261123


@dataclasses.dataclass(frozen=True)
class Graph:
    id: str
    shape: tuple
    indptr: np.ndarray  # int64 (rows + 1,)
    indices: np.ndarray  # int64, ascending within a row
    data: np.ndarray  # float64, int64 or bool
    tags: frozenset
    matrix: np.ndarray | None = None  # the dense-only graphs and the cost matrices: the matrix as given
    note: str = ""

    @property
    def n(self) -> int:
        return int(self.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    @property
    def square(self) -> bool:
        return self.shape[0] == self.shape[1]

    def edges(self) -> list:
        """[(i, j, w)] in CSR order (w a Python float, int or bool)."""
        out = []
        for i in range(self.shape[0]):
            for k in range(int(self.indptr[i]), int(self.indptr[i + 1])):
                out.append((i, int(self.indices[k]), self.data[k].item()))
        return out

    def dense(self, null=0.0) -> np.ndarray:
        """The matrix with `null` where no entry is stored (the dense-only graphs: the matrix itself)."""
        if self.matrix is not None:
            return self.matrix.copy()
        out = np.full(self.shape, null, dtype=np.float64)
        for i, j, w in self.edges():
            out[i, j] = w
        return out

    def scipy(self):
        """A scipy.sparse.csr_array of the stored entries (for the probes)."""
        import scipy.sparse as S

        return S.csr_array((self.data, self.indices, self.indptr), shape=self.shape)


def hash_int(k) -> int:
    return int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:8], "little")


def rng(*key) -> np.random.Generator:
    """The generator of a key: the same numbers on every machine (PCG64 of SEED and the key)."""
    return np.random.Generator(np.random.PCG64([SEED, *[hash_int(k) for k in key]]))


def digest(*arrays) -> str:
    """SHA-256 of arrays as given (dtype, shape and bytes): the records and references name the
    graphs they were made from."""
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(a)
        h.update(repr((str(a.dtype), a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def graph_digest(g: Graph) -> str:
    return digest(np.asarray(g.shape, np.int64), g.indptr, g.indices, g.data, g.matrix)


# ---------------------------------------------------------------- building


def csr(shape, entries, dtype=np.float64):
    """(indptr, indices, data) of {(i, j): w} (or an [(i, j, w)] list; a later duplicate replaces)."""
    if not isinstance(entries, dict):
        entries = {(int(i), int(j)): w for i, j, w in entries}
    rows = shape[0]
    keys = sorted(entries)
    indptr = np.zeros(rows + 1, np.int64)
    for i, _ in keys:
        indptr[i + 1] += 1
    np.cumsum(indptr, out=indptr)
    indices = np.array([j for _, j in keys], np.int64)
    data = np.array([entries[k] for k in keys], dtype=dtype)
    return indptr, indices, data


def _both(entries):
    """Every edge in both directions with the same weight."""
    out = dict(entries)
    for (i, j), w in entries.items():
        out[(j, i)] = w
    return out


def _int_w(g, lo=1, hi=9):
    return lambda: int(g.integers(lo, hi + 1))


def _path(n, w):
    return {(i, i + 1): w() for i in range(n - 1)}


def _cycle(n, w):
    return {(i, (i + 1) % n): w() for i in range(n)}


def _grid(dims):
    """Unit-weight grid, edges both ways (node = row-major index)."""
    idx = np.arange(int(np.prod(dims))).reshape(dims)
    out = {}
    for axis in range(len(dims)):
        a = np.moveaxis(idx, axis, 0)
        for u, v in zip(a[:-1].ravel(), a[1:].ravel()):
            out[(int(u), int(v))] = 1.0
            out[(int(v), int(u))] = 1.0
    return out


def _ladder(rungs):
    """2 x rungs ladder, unit weights both ways: 2^(rungs-1) shortest paths corner to corner."""
    out = {}
    for k in range(rungs):
        a, b = 2 * k, 2 * k + 1
        out[(a, b)] = out[(b, a)] = 1.0
        if k + 1 < rungs:
            for u, v in ((a, a + 2), (b, b + 2)):
                out[(u, v)] = out[(v, u)] = 1.0
    return out


def _er(n, p, key, weight, symmetric=False, loops=False):
    g = rng("er", key)
    out = {}
    for i in range(n):
        for j in range(n):
            if (i == j and not loops) or (symmetric and j < i):
                continue
            if g.random() < p:
                out[(i, j)] = weight(g)
    return _both(out) if symmetric else out


def _dag(n, p, key):
    g = rng("dag", key)
    order = g.permutation(n)
    out = {}
    for a in range(n):
        for b in range(a + 1, n):
            if g.random() < p:
                out[(int(order[a]), int(order[b]))] = float(g.integers(1, 10))
    return out


def _heavytail(n, key):
    """Each new node links to m = 3 earlier nodes chosen by in-degree + 1, and they link back with
    probability 1/2: a long-tailed degree distribution."""
    g = rng("heavy", key)
    deg = np.ones(n)
    out = {}
    for v in range(3, n):
        p = deg[:v] / deg[:v].sum()
        for u in g.choice(v, size=3, replace=False, p=p):
            out[(v, int(u))] = float(g.uniform(0.01, 1.01))
            deg[u] += 1
            if g.random() < 0.5:
                out[(int(u), v)] = float(g.uniform(0.01, 1.01))
    return out


def _components(sizes, key):
    g = rng("comp", key)
    out, base = {}, 0
    for s in sizes:
        for i in range(s - 1):  # a path keeps each component connected
            out[(base + i, base + i + 1)] = float(g.integers(1, 5))
        for _ in range(s):
            i, j = g.integers(0, s, 2)
            if i != j:
                out[(base + int(i), base + int(j))] = float(g.integers(1, 5))
        base += s
    return out


def _bipartite(r, c, p, key, weight):
    g = rng("bip", key)
    return {(i, j): weight(g) for i in range(r) for j in range(c) if g.random() < p}


_BUILDERS = {}


def _register(id_, tags, build, note=""):
    _BUILDERS[id_] = (frozenset(tags), build, note)


def _sq(n, entries, dtype=np.float64):
    return (n, n), entries, dtype


# shapes with small integer weights
_register("empty/0", {"shape"}, lambda: _sq(0, {}), "no node")
_register("empty/1", {"shape"}, lambda: _sq(1, {}), "one node, no edge")
_register("isolated/5", {"shape"}, lambda: _sq(5, {}), "five nodes, no edge")
_register("path/10", {"shape", "int"}, lambda: _sq(10, _path(10, _int_w(rng("path"), 1, 3))))
_register("path/both10", {"shape", "int", "undirected"}, lambda: _sq(10, _both(_path(10, _int_w(rng("pathb"), 1, 3)))))
_register("cycle/12", {"shape", "int"}, lambda: _sq(12, _cycle(12, _int_w(rng("cycle"), 1, 4))))
_register("star/9", {"shape", "int", "undirected"}, lambda: _sq(9, _both({(0, k): float(k) for k in range(1, 9)})))
_register(
    "star/out4",
    {"shape", "int"},
    lambda: _sq(4, {(2, 0): 1.0, (2, 1): 1.0, (2, 3): 1.0}),
    "scipy's reverse_cuthill_mckee example of a star",
)
_register(
    "complete/8",
    {"shape", "int", "ties"},
    lambda: _sq(8, {(i, j): float(1 + (3 * i + 5 * j) % 4) for i in range(8) for j in range(8) if i != j}),
)
_register(
    "diamond/4",
    {"shape", "int", "ties"},
    lambda: _sq(4, {(0, 1): 1.0, (0, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}),
    "two shortest paths 0-1-3 and 0-2-3: the predecessor tie",
)
_register("grid2d/6x6", {"shape", "int", "ties", "undirected"}, lambda: _sq(36, _grid((6, 6))))
_register("grid2d/5x5", {"shape", "int", "ties", "undirected"}, lambda: _sq(25, _grid((5, 5))))
_register("grid2d/3x3", {"shape", "int", "ties", "undirected"}, lambda: _sq(9, _grid((3, 3))), "yen's ties")
_register("grid3d/4x4x4", {"shape", "int", "ties", "undirected"}, lambda: _sq(64, _grid((4, 4, 4))))
_register("ladder/20", {"shape", "int", "ties", "undirected"}, lambda: _sq(40, _ladder(20)), "2^19 shortest paths")
_register("dag/30", {"shape", "int", "dag"}, lambda: _sq(30, _dag(30, 0.15, "a")))
_register("er/int60", {"shape", "int"}, lambda: _sq(60, _er(60, 0.08, "int60", lambda g: float(g.integers(1, 10)))))
_register(
    "er/sym60",
    {"shape", "int", "undirected"},
    lambda: _sq(60, _er(60, 0.06, "sym60", lambda g: float(g.integers(1, 10)), symmetric=True)),
)
_register(
    "er/float100", {"shape", "float"}, lambda: _sq(100, _er(100, 0.06, "f100", lambda g: float(g.uniform(0.01, 1.01))))
)
_register(
    "er/wide80",
    {"shape", "float"},
    lambda: _sq(80, _er(80, 0.08, "wide80", lambda g: float(10 ** g.uniform(-8, 8)))),
    "weights 1e-8 .. 1e8",
)
_register(
    "er/symfloat50",
    {"shape", "float", "undirected"},
    lambda: _sq(50, _er(50, 0.1, "sf50", lambda g: float(g.uniform(0.01, 1.01)), symmetric=True)),
)
_register("heavytail/120", {"shape", "float"}, lambda: _sq(120, _heavytail(120, "a")))
_register("components/40", {"shape", "int"}, lambda: _sq(40, _components((12, 1, 9, 18), "a")), "four components")
_register(
    "selfloop/10",
    {"shape", "int"},
    lambda: _sq(10, {**_path(10, lambda: 2.0), **{(i, i): 1.0 for i in range(0, 10, 3)}, (9, 0): 1.0}),
)
_register(
    "oneway/12",
    {"shape", "float"},
    lambda: _sq(
        12, {(i, j): 1.0 + ((i * 7 + j * 3) % 5) / 4 for i in range(12) for j in range(i + 1, 12) if (i + j) % 3 == 0}
    ),
    "every edge one way only (the undirected rule reads it both ways)",
)
_register(
    "twoway/12",
    {"shape", "float"},
    lambda: _sq(
        12,
        {
            **{
                (i, j): 1.0 + ((i * 7 + j * 3) % 5) / 4 for i in range(12) for j in range(i + 1, 12) if (i + j) % 3 == 0
            },
            **{(j, i): 0.5 + ((i * 5 + j) % 7) / 4 for i in range(12) for j in range(i + 1, 12) if (i + j) % 3 == 0},
        },
    ),
    "both ways with different weights (the undirected rule takes the smaller)",
)
# weights
_register(
    "dyadic/50", {"float", "exact"}, lambda: _sq(50, _er(50, 0.1, "dy50", lambda g: float(g.integers(1, 65)) / 8))
)
_register(
    "bigint/exact40",
    {"int", "bigint"},
    lambda: _sq(40, _er(40, 0.12, "bx40", lambda g: float(g.integers(2**44, 2**46)))),
    "integers below 2^46: every path sum stays below 2^53",
)
_register(
    "bigint/over40",
    {"int", "bigint"},
    lambda: _sq(
        40, {**_path(40, lambda: float(2**51 + 1)), **_er(40, 0.05, "bo40", lambda g: float(g.integers(2**51, 2**52)))}
    ),
    "integers near 2^51: long path sums pass 2^53 and round",
)
_register(
    "absorb/20",
    {"float", "absorb"},
    lambda: _sq(
        20,
        {
            **_path(20, lambda: 1.0),
            **{(i, i + 2): 2.0 for i in range(0, 18, 2)},
            **{(i, i + 1): 1e-17 for i in range(1, 19, 4)},
        },
    ),
    "weights 1e-17 that fl(d + w) absorbs: equal distances along different edge counts",
)
_register(
    "zero/15",
    {"float", "zero"},
    lambda: _sq(
        15,
        {
            **_path(15, lambda: 1.0),
            (3, 4): 0.0,
            (4, 5): 0.0,
            (5, 3): 0.0,
            (7, 8): 0.0,
            (8, 7): 0.0,
            (2, 9): 0.0,
            (0, 14): 5.0,
        },
    ),
    "stored zeros: zero-weight edges and zero-weight cycles 3-4-5 and 7-8",
)
_register(
    "negative/dag30",
    {"negative", "int", "dag"},
    lambda: _sq(30, {k: v - 4.0 for k, v in _dag(30, 0.15, "neg").items()}),
    "negative weights, no cycle",
)
_register(
    "negative/reach12",
    {"negative", "negcycle"},
    lambda: _sq(12, {**_path(12, lambda: 1.0), (6, 4): -3.0}),
    "the cycle 4-5-6 of sum -1, reachable from 0",
)
_register(
    "negative/unreach12",
    {"negative", "negcycle"},
    lambda: _sq(12, {**_path(8, lambda: 1.0), (8, 9): 1.0, (9, 10): 1.0, (10, 8): -3.0, (10, 11): 2.0}),
    "the cycle 8-9-10 of sum -1, not reachable from 0..7",
)
_register(
    "negative/zerocycle10",
    {"negative"},
    lambda: _sq(10, {**_path(10, lambda: 1.0), (4, 2): -2.0}),
    "the cycle 2-3-4 of exact sum 0",
)
_register(
    "negative/undirected6",
    {"negative"},
    lambda: _sq(6, {**_path(6, lambda: 2.0), (2, 3): -1.0}),
    "a negative edge: a negative cycle when read undirected",
)
_register("bool/30", {"bool", "shape"}, lambda: _sq(30, _er(30, 0.12, "bool30", lambda g: True), np.bool_))
_register(
    "int64/30", {"int", "shape"}, lambda: _sq(30, _er(30, 0.12, "i64", lambda g: int(g.integers(1, 10))), np.int64)
)
# non-finite stored values (errors for the weighted functions; patterns for the structural ones)
_register("nonfinite/nan", {"nonfinite"}, lambda: _sq(4, {(0, 1): 1.0, (1, 2): np.nan, (2, 3): 1.0, (0, 3): 5.0}))
_register("nonfinite/inf", {"nonfinite"}, lambda: _sq(4, {(0, 1): 1.0, (1, 2): np.inf, (2, 3): 1.0, (0, 3): 5.0}))
_register("nonfinite/neginf", {"nonfinite"}, lambda: _sq(4, {(0, 1): 1.0, (1, 2): -np.inf, (2, 3): 1.0, (0, 3): 5.0}))
# PageRank
_register(
    "dangling/50",
    {"float", "dangling"},
    lambda: _sq(
        50, {k: v for k, v in _er(50, 0.08, "dang", lambda g: float(g.uniform(0.01, 1.01))).items() if k[0] % 7 != 3}
    ),
    "nodes 3, 10, 17, ... have no out-edge",
)
# bipartite (rectangular) and assignment
_register(
    "bipartite/5x7",
    {"bipartite", "int", "ties"},
    lambda: ((5, 7), _bipartite(5, 7, 0.6, "a", lambda g: float(g.integers(1, 4))), np.float64),
)
_register(
    "bipartite/8x8",
    {"bipartite", "int"},
    lambda: ((8, 8), _bipartite(8, 8, 0.5, "b", lambda g: float(g.integers(1, 20))), np.float64),
)
_register(
    "bipartite/7x4",
    {"bipartite", "int"},
    lambda: ((7, 4), _bipartite(7, 4, 0.6, "c", lambda g: float(g.integers(1, 9))), np.float64),
)
_register(
    "bipartite/float8x8",
    {"bipartite", "float"},
    lambda: ((8, 8), _bipartite(8, 8, 0.6, "d", lambda g: float(g.uniform(0.01, 1.01))), np.float64),
)
_register(
    "bipartite/ties6x6",
    {"bipartite", "int", "ties"},
    lambda: ((6, 6), {(i, j): 1.0 for i in range(6) for j in range(6) if (i + j) % 2 == 0 or i == j}, np.float64),
)
_register(
    "bipartite/zero5x5",
    {"bipartite", "zero"},
    lambda: ((5, 5), {**{(i, i): 0.0 for i in range(5)}, **{(i, (i + 1) % 5): 1.0 for i in range(5)}}, np.float64),
    "a perfect matching of stored zeros",
)
_register(
    "bipartite/nofull5x5",
    {"bipartite", "int"},
    lambda: (
        (5, 5),
        {(i, j): float(i + j + 1) for i in range(5) for j in range(5) if j != 4 and i != 2 or (i, j) == (2, 1)},
        np.float64,
    ),
    "column 4 has no edge: no full matching",
)
_register(
    "cost/int8x8", {"cost", "int"}, lambda: ((8, 8), None, rng("cost", "a").integers(0, 20, (8, 8)).astype(np.float64))
)
_register(
    "cost/rect5x8", {"cost", "int"}, lambda: ((5, 8), None, rng("cost", "b").integers(0, 10, (5, 8)).astype(np.float64))
)
_register(
    "cost/rect8x5", {"cost", "int"}, lambda: ((8, 5), None, rng("cost", "c").integers(0, 10, (8, 5)).astype(np.float64))
)
_register(
    "cost/ties6x6", {"cost", "int", "ties"}, lambda: ((6, 6), None, np.ones((6, 6))), "every assignment is optimal"
)
_register("cost/float7x7", {"cost", "float"}, lambda: ((7, 7), None, rng("cost", "d").uniform(-1, 1, (7, 7))))
_register(
    "cost/inf5x5",
    {"cost", "int"},
    lambda: (
        (5, 5),
        None,
        np.where(
            np.eye(5, k=1, dtype=bool) | np.eye(5, k=-4, dtype=bool), 1.0, np.where(np.eye(5, dtype=bool), 3.0, np.inf)
        ),
    ),
    "+inf except a 5-cycle and the diagonal",
)
_register(
    "cost/infeasible4x4",
    {"cost"},
    lambda: (
        (4, 4),
        None,
        np.array(
            [[1.0, np.inf, np.inf, np.inf], [2.0, np.inf, np.inf, np.inf], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
        ),
    ),
    "rows 0 and 1 both need column 0",
)
# dense-only graphs: 0, NaN, +-inf mean no edge
_register(
    "dense/special6",
    {"dense", "nonfinite"},
    lambda: (
        (6, 6),
        None,
        np.array(
            [
                [0, 1, np.nan, np.inf, 0, 4],
                [1, 0, 2, -np.inf, 0, 0],
                [0, 0, 0, 1, np.nan, 0],
                [3, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 2],
                [0, 1e-9, 0, 0, 0, 0],
            ],
            np.float64,
        ),
    ),
    "dense entries 0, NaN, +-inf (no edge) and 1e-9 (an edge scipy's minimum_spanning_tree drops)",
)
# flow (integer capacities)
_register(
    "flow/int20", {"flow", "int"}, lambda: _sq(20, _er(20, 0.25, "flow20", lambda g: int(g.integers(1, 100))), np.int64)
)
_register(
    "flow/big6",
    {"flow", "int"},
    lambda: _sq(
        6, {(0, 1): 2**61, (0, 2): 2**61, (1, 3): 2**62, (2, 4): 2**62, (3, 5): 2**62, (4, 5): 2**62}, np.int64
    ),
    "capacities near 2^62: the flow 2^62 fits int64",
)
_register(
    "flow/overflow3",
    {"flow", "int"},
    lambda: _sq(3, {(0, 1): 2**62, (0, 2): 2**62, (1, 2): 2**62}, np.int64),
    "the source's capacities sum past int64",
)
_register(
    "flow/past_int32",
    {"flow", "int"},
    lambda: _sq(4, {(0, 1): 2**40, (1, 3): 2**40, (0, 2): 2**32 + 5, (2, 3): 2**32 + 5}, np.int64),
    "capacities 2^40 and 2^32 + 5: scipy reads them as int32",
)

IDS = tuple(_BUILDERS)


@functools.cache
def get(id_: str) -> Graph:
    if id_ not in _BUILDERS:
        raise KeyError(f"unknown graph {id_!r}")
    tags, build, note = _BUILDERS[id_]
    shape, entries, third = build()
    matrix = None
    if entries is None:  # a dense matrix: the cost matrices and the dense-only graphs
        matrix = np.ascontiguousarray(third, dtype=np.float64)
        if "cost" in tags:
            entries = {(i, j): matrix[i, j] for i in range(shape[0]) for j in range(shape[1])}
        else:  # the dense edge rule: not 0, not NaN, not +-inf
            entries = {
                (i, j): matrix[i, j]
                for i in range(shape[0])
                for j in range(shape[1])
                if matrix[i, j] != 0 and np.isfinite(matrix[i, j])
            }
        dtype = np.float64
    else:
        dtype = third
    indptr, indices, data = csr(shape, entries, dtype)
    return Graph(id_, tuple(shape), indptr, indices, data, tags, matrix, note)


def ids(tag: str | None = None) -> list[str]:
    return [i for i in IDS if tag is None or tag in _BUILDERS[i][0]]


def undirected(g: Graph) -> dict:
    """{(i, j): w} of the undirected reading (docs/design/graph.md §2.1): the edge i - j is whichever
    of i -> j and j -> i is stored, the smaller weight when both are; both directions listed."""
    out = {}
    for i, j, w in g.edges():
        for a, b in ((i, j), (j, i)):
            out[(a, b)] = w if (a, b) not in out else min(out[(a, b)], w)
    return out


def random_graph(n: int, d: int, seed=SEED, symmetric=False):
    """The timing graphs of docs/design/graph.md §5: each node has d out-edges to distinct random
    nodes (no self loop), weights in [0.01, 1.01); `symmetric` adds the transpose (the larger weight
    where both exist). Returns (indptr, indices, data) in canonical CSR, built with numpy only."""
    g = np.random.Generator(np.random.PCG64([seed, n, d]))
    rows = np.repeat(np.arange(n, dtype=np.int64), d)
    cols = (rows + 1 + g.integers(0, n - 1, size=n * d)) % n
    w = g.uniform(0.01, 1.01, size=n * d)
    if symmetric:
        rows, cols, w = np.concatenate([rows, cols]), np.concatenate([cols, rows]), np.concatenate([w, w])
    key = rows * n + cols
    order = np.lexsort((-w, key))
    key, w = key[order], w[order]
    keep = np.ones(key.size, bool)
    keep[1:] = key[1:] != key[:-1]  # the first of a duplicate pair is the largest weight
    key, w = key[keep], w[keep]
    r, c = key // n, key % n
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, r + 1, 1)
    return np.cumsum(indptr), c.astype(np.int64), w
