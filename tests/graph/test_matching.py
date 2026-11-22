"""maximum_bipartite_matching, min_weight_full_bipartite_matching and linear_sum_assignment (docs/api/graph.md
§4, docs/design/graph.md §2.4, §3.5, §4; decision GR16; milestone GR3): the largest matching's size (the
definition), the exact optimum of integer costs (the enumerated references of tests/graph/ref/assignment.npz,
and the exact dual certificate on large problems), float costs within the contract, the shared solver
(the dense and the sparse assignment agree), the meaning of +-inf, 0 and maximize, and the errors."""

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as S
import scipy.sparse.csgraph as CS
from scipy.optimize import linear_sum_assignment as scipy_lsa

import stardust as sd

from . import definitions as D
from . import graphs as G
from . import refdata
from .conftest import LIVE
from .forms import csr

GR = sd.graph
CORE = sd._core.graph


def pairs_of(g):
    return {(i, j) for i, j, _ in g.edges()}


def check_matching(rows, cols, pairs, col_of_row):
    used = [c for c in col_of_row if c >= 0]
    assert len(used) == len(set(used))
    for i, c in enumerate(col_of_row):
        assert c == -1 or (i, c) in pairs
    return len(used)


@pytest.mark.parametrize("gid", G.ids("bipartite") + ["er/int60", "grid2d/6x6", "components/40", "empty/0"])
def test_maximum_bipartite_matching(gid):
    g = G.get(gid)
    rows, cols = g.shape
    pairs = pairs_of(g)
    want = D.matching_size(rows, cols, pairs)
    for A in (csr(g), csr(g).tocsc(), csr(g).tocoo()):
        by_row = np.asarray(GR.maximum_bipartite_matching(A, perm_type="column")).tolist()
        by_col = np.asarray(GR.maximum_bipartite_matching(A)).tolist()
        assert len(by_row) == rows and len(by_col) == cols
        assert check_matching(rows, cols, pairs, by_row) == want
        assert by_col == [by_row.index(j) if j in by_row else -1 for j in range(cols)]


def test_matching_dense_rule_stored_zeros_and_dtype():
    M = np.array([[0, 2.0, 0], [0, 0, np.nan], [np.inf, 1, 0]])
    assert np.asarray(GR.maximum_bipartite_matching(M, perm_type="column")).tolist() == [1, -1, -1]
    Z = sd.sparse.csr_array((np.array([0.0, 0.0]), np.array([0, 1]), np.array([0, 1, 2])), shape=(2, 2))
    r = GR.maximum_bipartite_matching(Z)
    assert r.dtype == sd.int64 and np.asarray(r).tolist() == [0, 1]
    assert np.asarray(GR.maximum_bipartite_matching(np.zeros((2, 0)), perm_type="column")).tolist() == [-1, -1]
    with pytest.raises(ValueError, match="'perm_type' must be"):
        GR.maximum_bipartite_matching(M, perm_type="both")


def test_random_matchings_against_the_definition():
    rng = np.random.Generator(np.random.PCG64(31))
    for t in range(20):
        r, c = int(rng.integers(1, 40)), int(rng.integers(1, 40))
        M = rng.random((r, c)) < rng.uniform(0.02, 0.3)
        pairs = {(int(i), int(j)) for i, j in zip(*np.nonzero(M))}
        got = np.asarray(GR.maximum_bipartite_matching(M, perm_type="column")).tolist()
        assert check_matching(r, c, pairs, got) == D.matching_size(r, c, pairs), t


# ---- the assignment: the references


def exact_cost(cost, ri, ci):
    return sum((Fraction(float(cost[(i, j)])) for i, j in zip(ri, ci)), Fraction(0))


def solve(g, maximize):
    if "cost" in g.tags:
        C = g.matrix
        if maximize:  # the reference's +inf is a pair that cannot be chosen: -inf when maximizing
            C = np.where(np.isposinf(C), -np.inf, C)
        return GR.linear_sum_assignment(C, maximize=maximize)
    return GR.min_weight_full_bipartite_matching(csr(g), maximize=maximize)


def test_assignment_references():
    ref = refdata.ref("assignment")
    feasible = ref.methods["feasible"]
    for k in range(ref.points):
        gid = str(ref.inputs["graph"][k])
        g = G.get(gid)
        cost = {(i, j): w for i, j, w in g.edges()}
        r = min(g.shape)
        if not feasible.hi[k]:
            for maximize in (False, True):
                with pytest.raises(ValueError):
                    solve(g, maximize)
            continue
        for kind in ("min", "max"):
            m = ref.methods[kind]
            j = int(np.nonzero(m.index == k)[0][0])
            res = solve(g, kind == "max")
            ri, ci = np.asarray(res.row_ind).tolist(), np.asarray(res.col_ind).tolist()
            assert ri == sorted(ri) and len(ri) == r and len(set(ci)) == r
            got = exact_cost(cost, ri, ci)
            best = Fraction(float(m.hi[j])) + Fraction(float(m.lo[j]))
            if "float" in g.tags:  # within the contract
                ref_ci = list(m.fields["col_ind"][j])
                ref_ri = ri if g.shape[0] <= g.shape[1] else sorted(ri)
                abs_sum = sum(abs(float(cost[p])) for p in zip(ri, ci))
                abs_sum += sum(abs(float(cost[p])) for p in zip(ref_ri, ref_ci))
                excess = float(got - best) if kind == "min" else float(best - got)
                assert refdata.assignment_units(excess, r, abs_sum) <= 1, (gid, kind)
            else:  # integer costs: the exact optimum
                assert got == best, (gid, kind)
            if m.fields["count"][j] == 1:  # a unique optimum is the reference's
                assert ci == list(m.fields["col_ind"][j]), (gid, kind)


def certificate(C, maximize=False):
    """The exact dual certificate of linear_sum_assignment's solver on an integer matrix (rows <= cols):
    u(i) + v(j) <= c(i, j) for every finite pair, equality on the chosen ones, v <= 0 on the free columns."""
    status, ri, ci, u, v = CORE._assign_dense(sd.asarray(C), maximize)
    assert status == 0
    c = -C if maximize else C
    u, v = np.asarray(u), np.asarray(v)
    assert np.all(u == np.round(u)) and np.all(v == np.round(v))  # exact integers
    ui, vi = u.astype(np.int64), v.astype(np.int64)
    fin = np.isfinite(c)
    red = np.where(fin, c, 0).astype(np.int64) - ui[:, None] - vi[None, :]
    assert np.all(red[fin] >= 0)
    ri, ci = np.asarray(ri), np.asarray(ci)
    assert np.all(red[ri, ci] == 0)
    free = np.setdiff1d(np.arange(C.shape[1]), ci)
    assert np.all(vi[free] <= 0)
    return ri, ci


@pytest.mark.parametrize("shape", [(200, 200), (150, 400), (300, 300), (64, 64)])
@pytest.mark.parametrize("hi", [5, 1000, 2**40])
def test_integer_costs_are_exact_by_the_dual_certificate(shape, hi):
    rng = np.random.Generator(np.random.PCG64([shape[0], shape[1], hi]))
    C = rng.integers(0, hi, shape).astype(np.float64)
    C[rng.random(shape) < 0.05] = np.inf  # pairs that cannot be chosen
    ri, ci = certificate(C)
    res = GR.linear_sum_assignment(C)
    assert np.asarray(res.row_ind).tolist() == ri.tolist() and np.asarray(res.col_ind).tolist() == ci.tolist()
    certificate(np.where(np.isinf(C), -np.inf, -C), maximize=True)  # maximize: -inf cannot be chosen


@LIVE
def test_integer_costs_match_scipys_optimum():
    rng = np.random.Generator(np.random.PCG64(55))
    for shape in ((300, 300), (120, 350), (350, 120), (1, 9), (9, 1)):
        for hi in (3, 100, 10**9):
            C = rng.integers(-hi, hi, shape).astype(np.float64)
            for maximize in (False, True):
                mine = GR.linear_sum_assignment(C, maximize=maximize)
                theirs = scipy_lsa(C, maximize=maximize)
                assert C[np.asarray(mine.row_ind), np.asarray(mine.col_ind)].sum() == C[theirs].sum()


def test_dense_and_sparse_solver_agree():
    # the same pairs give the same assignment (ties included): the dense scan and the sparse heap take the
    # columns in one total order
    rng = np.random.Generator(np.random.PCG64(12))
    for shape in ((40, 40), (25, 60), (60, 25), (90, 90)):
        for C in (rng.integers(1, 4, shape).astype(np.float64), rng.uniform(0.5, 2, shape)):
            for maximize in (False, True):
                a = GR.linear_sum_assignment(C, maximize=maximize)
                b = GR.min_weight_full_bipartite_matching(sd.sparse.csr_array(C), maximize=maximize)
                assert np.asarray(a.col_ind).tolist() == np.asarray(b.col_ind).tolist()
                assert np.asarray(a.row_ind).tolist() == np.asarray(b.row_ind).tolist()


def test_ties_give_a_fixed_assignment():
    r = GR.linear_sum_assignment(np.ones((6, 6)))
    assert np.asarray(r.col_ind).tolist() == np.asarray(GR.linear_sum_assignment(np.ones((6, 6))).col_ind).tolist()
    assert sorted(np.asarray(r.col_ind).tolist()) == list(range(6))


# ---- the assignment: the meaning of the entries and the errors


def test_linear_sum_assignment_entries():
    # every entry is a pair (0 too), +inf none; with maximize -inf is none
    C = np.array([[0.0, 5], [np.inf, 1]])
    r = GR.linear_sum_assignment(C)
    assert np.asarray(r.row_ind).tolist() == [0, 1] and np.asarray(r.col_ind).tolist() == [0, 1]
    r = GR.linear_sum_assignment(np.array([[-np.inf, 5.0], [2.0, 1.0]]), maximize=True)
    assert np.asarray(r.col_ind).tolist() == [1, 0]
    with pytest.raises(ValueError, match=r"linear_sum_assignment\(\): cost matrix is infeasible"):
        GR.linear_sum_assignment(np.array([[1.0, np.inf], [2.0, np.inf]]))
    with pytest.raises(ValueError, match="must not hold NaN or -inf"):
        GR.linear_sum_assignment(np.array([[1.0, -np.inf], [2.0, 3.0]]))
    with pytest.raises(ValueError, match="must not hold NaN or -inf"):
        GR.linear_sum_assignment(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError, match=r"must not hold NaN or \+inf"):
        GR.linear_sum_assignment(np.array([[1.0, np.inf], [2.0, 3.0]]), maximize=True)
    with pytest.raises(TypeError, match="min_weight_full_bipartite_matching"):
        GR.linear_sum_assignment(sd.sparse.csr_array(np.eye(2)))
    with pytest.raises(TypeError, match="'maximize' must be True or False"):
        GR.linear_sum_assignment(np.eye(2), maximize=1)
    with pytest.raises(ValueError, match="2 dimensions"):
        GR.linear_sum_assignment(np.ones(3))
    # int64 and bool costs, rectangular and empty shapes
    r = GR.linear_sum_assignment(np.array([[3, 1, 2]], np.int64))
    assert np.asarray(r.col_ind).tolist() == [1]
    r = GR.linear_sum_assignment(np.array([[3], [1], [2]]))
    assert np.asarray(r.row_ind).tolist() == [1] and np.asarray(r.col_ind).tolist() == [0]
    for shape in ((0, 0), (0, 3), (3, 0)):
        r = GR.linear_sum_assignment(np.zeros(shape))
        assert r.row_ind.dtype == sd.int64 and r.row_ind.shape == (0,) and r.col_ind.shape == (0,)
    r = GR.linear_sum_assignment(np.eye(3, dtype=bool), maximize=True)
    assert np.asarray(r.col_ind).tolist() == [0, 1, 2]


def test_min_weight_full_bipartite_matching_entries():
    # a stored 0 is a pair of weight 0 (scipy reads it as none); dense: 0, NaN, inf are no pair
    Z = sd.sparse.coo_array((np.array([0.0, 5.0, 4.0]), (np.array([0, 0, 1]), np.array([0, 1, 0]))), shape=(2, 2))
    r = GR.min_weight_full_bipartite_matching(Z)
    assert np.asarray(r.col_ind).tolist() == [1, 0]
    r = GR.min_weight_full_bipartite_matching(np.array([[0, 5.0], [4.0, np.inf]]))
    assert np.asarray(r.col_ind).tolist() == [1, 0]
    with pytest.raises(ValueError, match=r"min_weight_full_bipartite_matching\(\): no full matching exists"):
        GR.min_weight_full_bipartite_matching(np.array([[0, 5.0], [0, 4.0]]))
    with pytest.raises(ValueError, match="stored edge weights must be finite"):
        GR.min_weight_full_bipartite_matching(
            sd.sparse.csr_array((np.array([np.nan]), np.array([0]), np.array([0, 1])), shape=(1, 1))
        )
    with pytest.raises(TypeError, match="sd.sparse.csr_array"):
        GR.min_weight_full_bipartite_matching(S.eye_array(2))
    r = GR.min_weight_full_bipartite_matching(np.zeros((0, 4)))
    assert r.row_ind.shape == (0,) and r.col_ind.dtype == sd.int64
    r = GR.min_weight_full_bipartite_matching(
        sd.sparse.csr_array(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]])), maximize=True
    )
    assert np.asarray(r.col_ind).tolist() == [1, 2]
    with pytest.raises(ValueError):  # the result arrays are read-only
        r.row_ind[0] = 5


@LIVE
def test_scipy_reads_stored_zeros_as_no_pair():
    # [scipy differs]: scipy's min_weight_full_bipartite_matching drops a stored 0
    z = S.csr_array((np.array([0.0, 5.0, 4.0]), (np.array([0, 0, 1]), np.array([0, 1, 0]))), shape=(2, 2))
    with pytest.warns(UserWarning, match="explicit zero weights are removed"):
        assert CS.min_weight_full_bipartite_matching(z)[1].tolist() == [1, 0]
    w = S.csr_array((np.array([0.0, 1.0, 1.0]), (np.array([0, 0, 1]), np.array([0, 1, 1]))), shape=(2, 2))
    with pytest.raises(ValueError), pytest.warns(UserWarning):
        CS.min_weight_full_bipartite_matching(w)
    r = GR.min_weight_full_bipartite_matching(sd.sparse.csr_array(w))
    assert np.asarray(r.col_ind).tolist() == [0, 1]


@LIVE
def test_scipy_matching_sizes():
    for gid in G.ids("bipartite"):
        g = G.get(gid)
        mine = np.asarray(GR.maximum_bipartite_matching(csr(g)))
        theirs = CS.maximum_bipartite_matching(g.scipy())
        assert int((mine >= 0).sum()) == int((theirs >= 0).sum()), gid


def float_costs(shape, seed):
    """Float costs over a wide range of magnitudes and both signs (rows <= cols: the exact solver's side)."""
    rng = np.random.Generator(np.random.PCG64([seed, *shape]))
    return rng.uniform(-1, 1, shape) * 10.0 ** rng.integers(-4, 5, shape)


@pytest.mark.parametrize("shape", [(12, 12), (25, 25), (10, 30), (40, 40)])
@pytest.mark.parametrize("seed", range(3))
def test_float_costs_within_the_contract(shape, seed):
    # the exact optimum of the exact-rational solver (definitions.assignment_exact); the bound
    # gamma_{2r} (sum |c| over both assignments) of docs/design/graph.md §4
    C = float_costs(shape, seed)
    for maximize in (False, True):
        c = -C if maximize else C
        best, opt = D.assignment_exact(c.tolist())
        res = GR.linear_sum_assignment(C, maximize=maximize)
        ri, ci = np.asarray(res.row_ind), np.asarray(res.col_ind)
        got = sum((Fraction(float(c[i, j])) for i, j in zip(ri, ci)), Fraction(0))
        abs_sum = float(np.abs(c[ri, ci]).sum() + np.abs(c[np.arange(shape[0]), opt]).sum())
        assert refdata.assignment_units(float(got - best), shape[0], abs_sum) <= 1
        sparse = GR.min_weight_full_bipartite_matching(sd.sparse.csr_array(C), maximize=maximize)
        assert np.asarray(sparse.col_ind).tolist() == ci.tolist()
