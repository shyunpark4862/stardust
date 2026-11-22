"""sd.graph's references and their contract (docs/design/graph.md §4, §7), as the tests and the
accuracy measurement (bench/accuracy) will both judge them.

tests/graph/ref/<family>.npz (tools/graph_ref.py; layout: tests/reffile.py) holds one point per graph
and options: the inputs "graph" (the id in tests/graph/graphs.py), "digest" (graphs.graph_digest of
the graph) and the family's options, and per method the reference hi + lo (a double-double; per-node
values are vectors):

    paths        source; methods directed, unweighted, undirected, undirected_unweighted (fields pred,
                 edges, abs_sum, exact_pred) and negcycle, negcycle_undirected (1 or 0)
    yen          source, sink; method lengths (fields edges, paths, total)
    assignment   method feasible (1 or 0), min and max (fields count, col_ind)
    pagerank     alpha, personalization, dangling (empty: the default), directed, unweighted; method x
    betweenness  directed, unweighted, normalized, endpoints; method bc
    laplacian    normed, use_out_degree, symmetrized; method L (row-major n x n)

tests/graph/data/scipy.json and networkx.json hold the comparison libraries' records
(tools/graph_probe.py), each with its own "_provenance".

The contract (draft until the milestones measure it; docs/design/graph.md §4), in units of its bound
(1 or less is within it): a shortest distance d of a path of at most H edges and absolute weight sum S
is within gamma_{H-1} S of the true distance; integer weights are exact. An assignment of the smaller side r
costs at most gamma_{2r} sum(|c|) (over the returned and an optimal assignment) above the optimum; integer
costs are exact. A laplacian's degree of m terms is within gamma_m s (s the sum of |A| over its terms), its
normalized entries within the relative bound of laplacian_bound (decision GR55); integer inputs are exact.
"""

from __future__ import annotations

import collections
import json
import math
import pathlib

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("paths", "yen", "assignment", "pagerank", "betweenness", "laplacian")
U = 2.0**-53
NONE = -9999  # the predecessor marker
YEN_K = 20  # the paths per yen point (tools/graph_ref.py)


def names() -> list[str]:
    return sorted(p.stem for p in REF.glob("*.npz"))


def ref(family: str) -> reffile.Ref:
    return reffile.read(REF / f"{family}.npz")


def data(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def gamma(k: int) -> float:
    return k * U / (1 - k * U) if k > 0 else 0.0


def bound(edges: int, abs_sum: float) -> float:
    """The contract's absolute bound for a shortest distance: gamma_{H-1} S (docs/design/graph.md §4)."""
    return gamma(edges - 1) * abs_sum


def units(got: float, hi: float, lo: float, edges: int, abs_sum: float) -> float:
    """|got - ref| in units of the contract's bound (0 when both are exactly equal)."""
    if got == hi and lo == 0:
        return 0.0
    if math.isinf(hi) or math.isinf(got):
        return 0.0 if got == hi else math.inf
    err = abs((got - hi) - lo)
    b = bound(edges, abs_sum)
    return math.inf if b == 0 else err / b


def assignment_bound(r: int, abs_sum: float) -> float:
    """The contract's bound for an assignment's exact cost above the optimum: gamma_{2r} times the sum of
    |c| over the returned and an optimal assignment (docs/design/graph.md §4; r the smaller side)."""
    return gamma(2 * r) * abs_sum


def assignment_units(excess: float, r: int, abs_sum: float) -> float:
    """An assignment's exact excess cost over the optimum (>= 0 for a correct reference) in units of the
    contract's bound (0 when it is optimal)."""
    if excess <= 0:
        return 0.0
    b = assignment_bound(r, abs_sum)
    return math.inf if b == 0 else excess / b


def laplacian_terms(n: int, edges, symmetrized: bool, use_out_degree: bool):
    """(degree, m, s) per node of a laplacian's degree sums (docs/design/graph.md §3.6, §4): the exact
    degree (a Fraction; the diagonal left out), the number of terms m the sum adds and s, the sum of |A|
    over the entries that make those terms. edges: [(i, j, w)] the stored entries (w a Python number)."""
    from fractions import Fraction

    a = {}
    for i, j, w in edges:
        if i != j:
            a[(i, j)] = Fraction(w)
    terms = collections.defaultdict(list)  # node -> [(exact term, sum of |parts|)]
    if symmetrized:
        for i, j in set(a) | {(j, i) for i, j in a}:
            x, y = a.get((i, j), Fraction(0)), a.get((j, i), Fraction(0))
            terms[i].append((x + y, float(abs(x) + abs(y))))
    else:
        for (i, j), x in a.items():
            terms[i if use_out_degree else j].append((x, float(abs(x))))
    degree = [sum((t for t, _ in terms[k]), Fraction(0)) for k in range(n)]
    m = [len(terms[k]) for k in range(n)]
    s = [math.fsum(p for _, p in terms[k]) for k in range(n)]
    return degree, m, s


def laplacian_bound(normed: bool, symmetrized: bool, i: int, j: int, terms, exact: float, a_abs: float = 0.0):
    """The contract's bound for entry (i, j) of a laplacian (docs/design/graph.md §4; decision GR55), terms
    = laplacian_terms(...), exact the reference entry, a_abs = |A(i, j)| + |A(j, i)| (symmetrized
    off-diagonal entries). A degree of m terms is within gamma_k s of the exact one, k = m (m + 1
    symmetrized: the terms are rounded sums); an unnormalized off-diagonal entry is exact (gamma_1 a_abs
    symmetrized); a normalized one is within |L(i, j)| B, B = (1 + gamma_c) / sqrt((1 - rho_i)(1 - rho_j))
    - 1 with rho_k = gamma_k s_k / |degree_k| and c = 4 (the square roots and the two divisions; 5
    symmetrized); the normalized diagonal (1, or 0 for degree 0) is exact. inf where rho >= 1 (a degree
    the rounding can move across 0: no contract)."""
    degree, m, s = terms
    extra = 1 if symmetrized else 0
    if not normed:
        if i == j:
            return gamma(m[i] + extra) * s[i]
        return gamma(1) * a_abs if symmetrized else 0.0
    if i == j:
        return 0.0

    def rho(k):
        if degree[k] == 0:
            return 0.0 if s[k] == 0 else math.inf
        return gamma(m[k] + extra) * s[k] / abs(float(degree[k]))

    ri, rj = rho(i), rho(j)
    if ri >= 1 or rj >= 1:
        return math.inf
    return ((1 + gamma(4 + extra)) / math.sqrt((1 - ri) * (1 - rj)) - 1) * abs(exact)


def laplacian_units(got: float, hi: float, lo: float, b: float) -> float:
    """|got - ref| of a laplacian entry in units of its bound b (0 when equal, NaN against NaN too)."""
    if (got == hi and lo == 0) or (math.isnan(got) and math.isnan(hi)):
        return 0.0
    if math.isinf(b):
        return 0.0
    err = abs((got - hi) - lo)
    return math.inf if b == 0 else err / b


def pairwise_depth(m: int) -> int:
    """The most roundings a term goes through in the shared pairwise tree over m terms (csrc/base/pairwise.hpp;
    csrc/graph/centrality.cpp pairwise_depth): a leaf of fewer than 16 terms is one chain from 0.0, a longer
    leaf (at most 512) 16 chains, their 4-level fold and a tail, a longer sum split in two at a multiple of 16."""
    if m <= 512:
        return max(m - 1, 0) if m < 16 else (m // 16 - 1) + 4 + (m % 16)
    half = m // 2
    a = half - half % 16
    return 1 + max(pairwise_depth(a), pairwise_depth(m - a))


def pagerank_k(n: int, max_in: int, max_out: int, n_dangling: int, weighted: bool) -> int:
    """The roundings of one PageRank step relative to each entry (docs/design/graph.md §4, decision GR57): a
    row term (weighted: w / out(j), out(j) a pairwise sum of max_out terms, times x(j); else x(j) / degree)
    through its row's pairwise tree of max_in terms, the dangling term added, times alpha, plus (1 - alpha)
    v(i); the dangling term (a pairwise sum over the dangling nodes times d(i) = q(i) / pairwise sum of n)
    the same; the teleport term from v(i) = p(i) / pairwise sum of n."""
    hn = pairwise_depth(n)
    term = pairwise_depth(max_out) + 2 if weighted else 1
    return max(term + pairwise_depth(max_in) + 3, pairwise_depth(n_dangling) + hn + 5, hn + 4)


def pagerank_bound(tol: float, alpha: float, k: int) -> float:
    """The contract's l1 bound for PageRank: tol + F, F = gamma_K / (1 - alpha) the rounding of one step over
    the contraction's gap (the norm of a step's result is 1 within a few units of rounding: the factor
    1 + 1e-9 covers it)."""
    return tol + gamma(k) * (1 + 1e-9) / (1 - alpha)


def betweenness_k(L: int, dout: int, din: int, sigma: int, n: int) -> int:
    """The roundings bounding betweenness's relative error (docs/design/graph.md §4, decision GR58), every
    term non-negative: a path count of a DAG of longest path L and in-degree din is within gamma_A, A = 0
    when every count is below 2^53 (exact) and L (din - 1) otherwise; a dependency within gamma_B, B = L (2A
    + dout + 2) (per DAG edge the ratio, 1 + delta, the product, and dout - 1 additions per node); the sum
    over the sources within n more, and the endpoints and the scaling 3."""
    a = 0 if sigma < 2**53 else L * (din - 1)
    return L * (2 * a + dout + 2) + n + 3
