"""The matrix functions and norms of sd.sparse (milestone SP7; docs/api/sparse.md §8,
docs/design/sparse.md §3.9, §4): expm, expm_multiply, onenormest, norm.

What the tests hold them to:
  - the contract (docs/design/sparse.md §4) against Arb: expm on expm.npz and expm_multiply on
    expmv.npz (t = 1 and the interval t = 0 .. 1) within C_EXPM eps max(1, ||s A||_1) scale; the
    norms on norms.npz (sums within gamma_m, the 2-norm within C_EIG eps ||A||_2); onenormest a lower
    bound of the exact 1-norm, exact when t >= n;
  - scipy 1.18.1: expm_multiply the same values (live; bits where the step choice does not depend on
    scipy's random estimate), its recorded results (tools/gen_sparse_scipy.py), every path of the
    interval (q <= s, q a multiple of s, the rest); expm, onenormest (t = 1 and t >= n are not
    random in scipy) and norm the same values; the dense sd.linalg.expm and vector_norm /
    matrix_norm the same values;
  - the decisions: onenormest's fixed columns (SP98) and max_iter (SP100), expm's result in A's
    format (SP99), norm's 'nuc' and -2 refused (SP101), the documented choices of §8 (a dense B,
    traceA, onenormest's result object);
  - the bits: the same for 1, 3 and 10 threads and for every input format;
  - the inputs and errors, and sd.linalg's refusals of a sparse array naming these functions.
"""

import math

import numpy as np
import pytest
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import stardust as sd

from .._bits import bits_equal
from . import matrices as M
from . import refdata as R
from .conftest import LIVE, coo

S = sd.sparse
EPS = R.EPS
INF = math.inf


def npy(x):
    return np.asarray(x.tolist()) if x is not None else None


def dense_of(X):
    return npy(X.toarray())


def _rand(n, density, seed, cplx=False, shift=0.0):
    g = np.random.default_rng(seed)
    a = sp.random_array((n, n), density=density, rng=g, format="csr")
    if cplx:
        a = a + 1j * sp.random_array((n, n), density=density, rng=g, format="csr")
    return (a + shift * sp.eye_array(n)).tocsr()


# ================================================================== expm


def test_expm_within_the_contract_against_arb():
    worst = 0.0
    for p in R.points("expm"):
        m = M.get(p["matrix"])
        s = float(p["scale"])
        ref = p["expm_hi"] + p["expm_lo"]
        E = dense_of(S.expm(coo(m).tocsr() * s)).ravel()
        err = R.expm_error(E, ref, np.abs(m.dense() * s).sum(axis=0).max(), np.abs(ref).max())
        assert err <= 1.0, (p["matrix"], s, err)
        worst = max(worst, err)
    assert worst > 0.05  # the measure is not empty


@LIVE
def test_expm_agrees_with_scipy_and_the_dense_expm():
    for p in R.points("expm")[::2]:
        m = M.get(p["matrix"])
        s = float(p["scale"])
        E = dense_of(S.expm(coo(m).tocsr() * s))
        Es = spla.expm(m.scipy("csc") * s).toarray()
        Ed = npy(sd.linalg.expm(sd.asarray(m.dense() * s)))
        top = np.abs(Es).max()
        assert np.abs(E - Es).max() <= 1e-13 * top * max(1.0, np.abs(m.dense() * s).sum(axis=0).max()), p["matrix"]
        assert np.abs(E - Ed).max() <= 1e-13 * top * max(1.0, np.abs(m.dense() * s).sum(axis=0).max()), p["matrix"]


def test_expm_recorded_scipy_results():
    data = R.data("scipy")
    for k, rec in data.items():
        if not k.startswith("matfuncs/expm/"):
            continue
        _, _, *mid, scale = k.split("/")
        m = M.get("/".join(mid))
        E = S.expm(coo(m).tocsc() * float(scale))
        assert list(E.shape) == rec["shape"] and str(E.dtype) == rec["dtype"]
        assert rec["format"] == "csc" and E.format == "csc"  # A's format (SP99); scipy's CSC here


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "dok"])
def test_expm_returns_the_input_format_with_the_same_bits(fmt):
    # SP99: the input's format; the computation runs on CSR, so the values are the same bits
    m = M.get("convdiff2d/10")
    A = coo(m).tocsr() * 0.125
    E = S.expm(A.asformat(fmt))
    assert E.format == fmt and E.dtype == sd.float64
    assert bits_equal(dense_of(E), dense_of(S.expm(A)))


def test_expm_diagonal_and_triangular_shortcuts():
    d = np.array([1.0, -2.0, 0.0, -800.0, 3.5])
    E = S.expm(S.diags_array(sd.asarray(d), format="csr"))
    assert bits_equal(dense_of(E), np.diag(npy(sd.exp(sd.asarray(d)))))
    assert E.nnz == 4  # exp(-800) underflows to an exact zero, dropped
    g = np.random.default_rng(5)
    for T in (np.triu(g.standard_normal((8, 8))) * 3, np.tril(g.standard_normal((8, 8))) * 0.01):
        E = dense_of(S.expm(S.csr_array(sd.asarray(T))))
        ref = sl.expm(T)
        assert np.abs(E - ref).max() <= 1e-13 * np.abs(ref).max()
        assert np.all(E[np.tril_indices(8, -1) if T[1, 0] == 0 else np.triu_indices(8, 1)] == 0)
        assert np.array_equal(np.diag(E), npy(sd.exp(sd.asarray(np.diag(T).copy()))))  # the fix-up's exact diagonal
    # close eigenvalues: the fix-up's sinch form keeps the off-diagonal entry (scipy loses digits)
    T = np.array([[10.0, 1.0], [0.0, 10.0 + 1e-12]])
    E = dense_of(S.expm(S.csr_array(sd.asarray(T))))
    ref = 22026.46579481773  # Arb: (e^l2 - e^l1) / (l2 - l1); scipy's is 22026.0036
    assert abs(E[0, 1] - ref) <= 1e-14 * ref  # (the rounding of l1 + d / 2 alone is 4 eps)


def test_expm_dtypes_and_small_shapes():
    assert S.expm(S.csr_array(sd.asarray([[1, 2], [0, 1]]))).dtype == sd.float64
    assert S.expm(S.csr_array(sd.asarray([[True, False], [True, False]]))).dtype == sd.float64
    Z = S.expm(S.csr_array(sd.asarray([[1j, 2.0], [0.5, -1j]])))
    assert Z.dtype == sd.complex128
    ref = sl.expm(np.array([[1j, 2.0], [0.5, -1j]]))
    assert np.abs(dense_of(Z) - ref).max() <= 1e-14 * np.abs(ref).max()
    assert dense_of(S.expm(S.csr_array(sd.asarray([[2.0]])))).tolist() == [[math.exp(2.0)]]
    E0 = S.expm(S.csc_array((0, 0)))
    assert E0.shape == (0, 0) and E0.format == "csc" and E0.dtype == sd.float64
    E = S.expm(S.csr_array((3, 3)))
    assert dense_of(E).tolist() == np.eye(3).tolist() and E.nnz == 3


def test_expm_extreme_norms():
    # ||A|| > 2^100: scaled first; eigenvalues near -1e40 give exp ~ 0 (scipy gives I)
    A = S.csr_array(sd.asarray([[-1e40, 1.0], [0.0, -2e40]]))
    assert np.abs(dense_of(S.expm(A))).max() == 0.0
    B = S.csr_array(sd.asarray([[-1e40, 1e39], [1e39, -2e40]]))
    assert np.abs(dense_of(S.expm(B))).max() == 0.0
    # finite entries whose column sum overflows: no endless squaring; the value overflows
    H = S.csr_array(sd.asarray([[1e308, 1e308], [1e308, 1e308]]))
    assert not np.isfinite(dense_of(S.expm(H))).all()
    # a subnormal norm: I + A
    T = S.csr_array(sd.asarray([[1e-310, 1e-310], [1e-310, 1e-310]]))
    E = dense_of(S.expm(T))
    assert np.abs(E - (np.eye(2) + 1e-310)).max() <= 1e-323


def test_expm_errors():
    with pytest.raises(TypeError, match=r"^expm\(\): 'A' must be a sparse array, not ndarray; for a dense array use "
                                        r"sd.linalg.expm$"):  # fmt: skip
        S.expm(sd.eye(3))
    with pytest.raises(TypeError, match=r"^expm\(\): 'A' must be a sparse array, not list$"):
        S.expm([[1.0]])
    with pytest.raises(ValueError, match=r"^expm\(\): 'A' must be square, got 2x3$"):
        S.expm(S.csr_array((2, 3)))
    with pytest.raises(ValueError, match=r"^expm\(\): 'A' must have 2 dimensions, got 1$"):
        S.expm(S.csr_array(sd.asarray([1.0, 2.0])))
    with pytest.raises(ValueError, match=r"^expm\(\): 'A' contains NaN or infinity$"):
        S.expm(S.csr_array(sd.asarray([[1.0, math.nan], [0.0, 1.0]])))
    with pytest.raises(TypeError, match=r"^expm\(\): takes 1 positional argument but 2 were given$"):
        S.expm(S.csr_array((2, 2)), 1)


# ================================================================== expm_multiply


def _points_with_scale():
    for p, q in zip(R.points("expm"), R.points("expmv")):
        assert (p["matrix"], p["scale"]) == (q["matrix"], q["scale"])
        yield p, q


def test_expm_multiply_within_the_contract_against_arb():
    worst = {"expmv": 0.0, "interval": 0.0}
    for p, q in _points_with_scale():
        m = M.get(p["matrix"])
        s = float(p["scale"])
        d = m.dense() * s
        norm1 = np.abs(d).sum(axis=0).max()
        A, b = coo(m).tocsr() * s, sd.asarray(q["b"])
        Eref = (p["expm_hi"] + p["expm_lo"]).reshape(d.shape)
        y = npy(S.expm_multiply(A, b))
        err = R.expm_error(y, q["expmv_hi"] + q["expmv_lo"], norm1, (np.abs(Eref) @ np.abs(q["b"])).max())
        assert err <= 1.0, (p["matrix"], s, err)
        worst["expmv"] = max(worst["expmv"], err)
        Y = npy(S.expm_multiply(A, b, start=0.0, stop=1.0, num=5))
        refs = (q["interval_hi"] + q["interval_lo"]).reshape(len(R.TIMES), -1)
        for k, t in enumerate(R.TIMES):
            scale = (np.abs(sl.expm(t * d)) @ np.abs(q["b"])).max()
            err = R.expm_error(Y[k], refs[k], t * norm1, scale)
            assert err <= 1.0, (p["matrix"], s, t, err)
            worst["interval"] = max(worst["interval"], err)
    assert min(worst.values()) > 0.05


@LIVE
@pytest.mark.parametrize(
    "kw",
    [{}, {"start": 0, "stop": 1, "num": 5}, {"start": 0.0, "stop": 2.0, "num": 3, "endpoint": False},
     {"start": 0.5, "stop": 3.0, "num": 40}, {"start": -1.0, "stop": 1.0, "num": 7}, {"start": 1, "stop": 1, "num": 3},
     {"start": 0.0, "stop": 20.0, "num": 4}],
)  # fmt: skip
def test_expm_multiply_agrees_with_scipy(kw):
    """Every path of the interval (q <= s, q a multiple of s, the rest) and of the simple call,
    vector and matrix B, real and complex: scipy's values (its random estimate can choose other
    steps: 1e-13 relative)."""
    for a in (_rand(40, 0.15, 1, shift=-2.0), _rand(30, 0.2, 2, cplx=True)):
        g = np.random.default_rng(3)
        Bm = g.standard_normal((a.shape[0], 3))
        for B in (Bm[:, 0].copy(), Bm):
            y = npy(S.expm_multiply(S.csr_array(a), sd.asarray(B), **kw))
            ys = spla.expm_multiply(a, B, **kw)
            assert y.shape == ys.shape and y.dtype == ys.dtype
            assert np.abs(y - ys).max() <= 1e-13 * np.abs(ys).max(), kw


@LIVE
def test_expm_multiply_bits_as_scipy_where_the_steps_need_no_estimate():
    # ||A||_1 small enough that (m*, s) comes from ||A||_1 alone (scipy's condition 3.13): the same
    # products (SpMV is scipy's), sums and stopping tests give scipy's bits
    a = _rand(200, 0.02, 7, shift=-1.0) * 0.05
    b = np.sin(np.arange(200.0))
    assert bits_equal(
        npy(S.expm_multiply(S.csr_array(a), sd.asarray(b), traceA=0.0)), spla.expm_multiply(a, b, traceA=0.0)
    )


def test_expm_multiply_recorded_scipy_results():
    import hashlib

    same = 0
    for mid in ("lap2d/30", "convdiff2d/40", "random_complex/800", "random_unsym/1000x8"):
        rec = R.data("scipy")[f"matfuncs/{mid}"]
        m = M.get(mid)
        y = npy(S.expm_multiply(coo(m).tocsc() * -0.1, sd.asarray(M.rhs(m))))
        same += hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()[:32] == rec["expm_multiply_sha"]
        ys = spla.expm_multiply(m.scipy("csc") * -0.1, M.rhs(m))
        assert np.abs(y - ys).max() <= 1e-14 * np.abs(ys).max(), mid
    assert same >= 2  # convdiff2d/40 and random_unsym/1000x8: scipy's bits (the others: the trace's sum)


def test_expm_multiply_operands():
    a = _rand(50, 0.1, 4, shift=-1.0)
    A = S.csr_array(a)
    b = sd.asarray(np.cos(np.arange(50.0)))
    y = npy(S.expm_multiply(A, b))
    ref = sl.expm(a.toarray()) @ npy(b)
    assert np.abs(y - ref).max() <= 1e-13 * np.abs(ref).max()
    # every format and the dense matrix: the same bits; an operator with traceA: the same values
    for X in (A.tocsc(), A.tocoo(), A.todok(), S.aslinearoperator(A)):
        assert bits_equal(npy(S.expm_multiply(X, b)), y)
    yd = npy(S.expm_multiply(A.toarray(), b))
    assert np.abs(yd - y).max() <= 1e-14 * np.abs(y).max()
    op = S.LinearOperator(A.shape, lambda v: A @ v, rmatvec=lambda v: A.T @ v, dtype=sd.float64)
    yo = npy(S.expm_multiply(op, b, traceA=float(a.trace())))
    assert np.abs(yo - y).max() <= 1e-14 * np.abs(y).max()
    # an operator without traceA: scipy's estimate on fixed draws: the same result every call
    y1, y2 = npy(S.expm_multiply(op, b)), npy(S.expm_multiply(op, b))
    assert bits_equal(y1, y2) and np.abs(y1 - y).max() <= 1e-13 * np.abs(y).max()
    # traceA only shifts: any value gives exp(A) b
    for tr in (0.0, -7.5, 3 + 2j):
        yt = npy(S.expm_multiply(A, b, traceA=tr))
        assert np.abs(yt - y).max() <= 1e-13 * np.abs(y).max()
    assert S.expm_multiply(A, b, traceA=1j).dtype == sd.complex128
    # int and bool B become float64; a complex A or B gives complex
    assert S.expm_multiply(A, sd.ones(50, dtype=sd.int64)).dtype == sd.float64
    assert S.expm_multiply(A * (1 + 0j), b).dtype == sd.complex128
    assert S.expm_multiply(A, b * 1j).dtype == sd.complex128


def test_expm_multiply_zero_and_empty():
    Z = S.csr_array((4, 4))
    b = sd.asarray([1.0, 2.0, 3.0, 4.0])
    assert npy(S.expm_multiply(Z, b)).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert npy(S.expm_multiply(Z, b, start=0, stop=1, num=3)).tolist() == [[1.0, 2.0, 3.0, 4.0]] * 3
    assert S.expm_multiply(S.csr_array((0, 0)), sd.zeros(0)).shape == (0,)
    assert S.expm_multiply(S.csr_array((0, 0)), sd.zeros((0, 2)), start=0, stop=1).shape == (50, 0, 2)


def test_expm_multiply_errors():
    A = S.csr_array(sd.asarray([[1.0, 2.0], [0.0, 1.0]]))
    b = sd.asarray([1.0, 1.0])
    f = S.expm_multiply
    with pytest.raises(TypeError, match=r"^expm_multiply\(\): 'B' must be an sd.ndarray, not csr_array; for a sparse "
                                        r"B use B.toarray\(\)$"):  # fmt: skip
        f(A, A)
    with pytest.raises(TypeError, match=r"^expm_multiply\(\): 'B' must be an sd.ndarray, not list$"):
        f(A, [1.0, 1.0])
    with pytest.raises(ValueError, match=r"^expm_multiply\(\): 'A' is 2x2 but 'B' has 3 rows$"):
        f(A, sd.ones(3))
    with pytest.raises(ValueError, match=r"^expm_multiply\(\): 'B' must have 1 or 2 dimensions, got 3$"):
        f(A, sd.ones((2, 1, 1)))
    with pytest.raises(ValueError, match=r"^expm_multiply\(\): 'B' contains NaN or infinity$"):
        f(A, sd.asarray([1.0, math.inf]))
    with pytest.raises(ValueError, match=r"^expm_multiply\(\): 'A' contains NaN or infinity$"):
        f(A * math.nan, b)
    with pytest.raises(ValueError, match=r"^expm_multiply\(\): 'A' must be square, got 2x3$"):
        f(S.csr_array((2, 3)), b)
    with pytest.raises(TypeError, match=r"^expm_multiply\(\): 'A' must be a sparse array, an sd.ndarray or a "
                                        r"LinearOperator, not list$"):  # fmt: skip
        f([[1.0]], b)
    with pytest.raises(ValueError, match=r"^expm_multiply\(\): a time interval needs both 'start' and 'stop'$"):
        f(A, b, num=5)
    with pytest.raises(ValueError, match=r"^expm_multiply\(\): 'num' must be >= 2$"):
        f(A, b, start=0, stop=1, num=1)
    with pytest.raises(TypeError, match=r"^expm_multiply\(\): 'num' must be an int, not float$"):
        f(A, b, start=0, stop=1, num=5.0)
    with pytest.raises(TypeError, match=r"^expm_multiply\(\): 'endpoint' must be True or False, not int$"):
        f(A, b, start=0, stop=1, endpoint=1)
    with pytest.raises(TypeError, match=r"^expm_multiply\(\): 'start' must be an int or a float, not complex$"):
        f(A, b, start=1j, stop=1)
    with pytest.raises(ValueError, match=r"^expm_multiply\(\): 'stop' must be finite$"):
        f(A, b, start=0, stop=math.inf)
    with pytest.raises(TypeError, match=r"^expm_multiply\(\): 'traceA' must be a number or None, not bool$"):
        f(A, b, traceA=True)
    with pytest.raises(ValueError, match=r"^expm_multiply\(\): 'traceA' must be finite$"):
        f(A, b, traceA=math.nan)


# ================================================================== onenormest


def test_onenormest_is_a_lower_bound_and_often_exact():
    """Every small square real or complex test matrix: est <= ||A||_1 (exact sums of the stored
    values); measured in SP7 (not a contract): est = ||A||_1 on 26 of 42, the smallest ratio 0.76
    (struct_singular/40)."""
    exact = 0
    count = 0
    ratios = []
    for mid in M.ids(small=True):
        m = M.get(mid)
        if m.shape[0] != m.shape[1] or m.shape[0] < 3 or m.data.dtype.kind in "bi" or "nan" in mid:
            continue
        A = coo(m).tocsr()
        est = S.onenormest(A).est
        true = float(np.abs(m.dense()).sum(axis=0).max())
        assert est <= true * (1 + 4 * EPS), mid
        exact += math.isclose(est, true, rel_tol=1e-14)
        count += 1
        ratios.append(est / true)
    assert count >= 30 and exact >= count // 2 and min(ratios) >= 0.5


def test_onenormest_exact_when_t_reaches_n():
    m = M.get("random_unsym/50x5")
    A = coo(m).tocsr()
    true = float(np.abs(m.dense()).sum(axis=0).max())
    est, v, w = S.onenormest(A, t=50, compute_v=True, compute_w=True)
    assert math.isclose(est, true, rel_tol=1e-14)
    assert spla.onenormest(m.scipy("csr"), t=50) == pytest.approx(est, rel=1e-14)
    j = int(np.argmax(npy(v)))
    assert npy(v).tolist() == np.eye(50)[j].tolist() and np.allclose(npy(w), m.dense()[:, j], rtol=0, atol=0)


@LIVE
def test_onenormest_t1_as_scipy():
    # t = 1 has no random column in scipy either: the same estimates
    for mid in ("random_unsym/50x5", "random_complex/40", "convdiff2d/10", "graded/100x8", "random_spd/60"):
        m = M.get(mid)
        assert S.onenormest(coo(m).tocsr(), t=1).est == pytest.approx(spla.onenormest(m.scipy("csr"), t=1), rel=1e-14)


def test_onenormest_fixed_columns_and_result_forms():
    # SP98: the random columns are fixed: the same estimate on every call
    A = S.csr_array(_rand(300, 0.02, 9))
    r = S.onenormest(A)
    first = r.est
    assert all(S.onenormest(A).est == first for _ in range(3))
    assert isinstance(first, float)
    # SP102: one result object whatever compute_v / compute_w ask (scipy: a bare float by default)
    assert tuple(r) == (first, None, None) and len(r) == 3 and r[0] == first
    assert r.v is None and r.w is None and r.n_iter >= 2 and r.n_op in (2 * r.n_iter - 1, 2 * r.n_iter)
    est, v, w = S.onenormest(A, compute_v=True, compute_w=True)
    assert est == first
    assert npy(v).sum() == 1.0 and np.count_nonzero(npy(v)) == 1
    assert np.allclose(npy(A @ v), npy(w), rtol=1e-15, atol=0)  # w = A v
    est2, v2, w2 = S.onenormest(A, compute_v=True)
    assert w2 is None and bits_equal(npy(v2), npy(v))
    est3, v3, w3 = S.onenormest(A, compute_w=True)
    assert v3 is None and bits_equal(npy(w3), npy(w))
    # more columns, more iterations: never above the norm
    true = float(np.abs(_rand(300, 0.02, 9).toarray()).sum(axis=0).max())
    for t, it in ((1, 2), (3, 5), (5, 10)):
        assert S.onenormest(A, t=t, max_iter=it).est <= true * (1 + 4 * EPS)


def test_onenormest_operands():
    a = _rand(60, 0.1, 11, cplx=True)
    A = S.csr_array(a)
    est = S.onenormest(A).est
    op = S.LinearOperator(A.shape, lambda v: A @ v, rmatvec=lambda v: A.T.conj() @ v, dtype=sd.complex128)
    for X in (A.tocsc(), A.tocoo(), A.todok(), A.toarray(), S.aslinearoperator(A), op):
        assert S.onenormest(X).est == est
    e = S.onenormest(S.csr_array((0, 0)), compute_v=True, compute_w=True)
    assert e.est == 0.0 and e.v.shape == (0,) and (e.n_iter, e.n_op) == (0, 0)
    assert S.onenormest(S.csr_array(sd.asarray([[-3.0]]))).est == 3.0
    op = S.LinearOperator((5, 5), lambda v: v * 2.0, dtype=sd.float64)  # no rmatvec
    with pytest.raises(ValueError, match="rmatvec"):
        S.onenormest(op)
    x = S.onenormest(op, t=5)  # t >= n: the explicit matrix, no rmatvec needed
    assert x.est == 2.0 and (x.n_iter, x.n_op) == (0, 1)


def test_onenormest_errors():
    A = S.csr_array(sd.eye(4))
    f = S.onenormest
    with pytest.raises(ValueError, match=r"^onenormest\(\): 'max_iter' must be >= 2$"):
        f(A, max_iter=1)
    with pytest.raises(TypeError, match=r"^onenormest\(\): unexpected keyword argument 'itmax'$"):
        f(A, itmax=3)
    with pytest.raises(ValueError, match=r"^onenormest\(\): 't' must be >= 1$"):
        f(A, t=0)
    with pytest.raises(TypeError, match=r"^onenormest\(\): 't' must be an int, not float$"):
        f(A, t=2.0)
    with pytest.raises(TypeError, match=r"^onenormest\(\): 'compute_v' must be True or False, not int$"):
        f(A, compute_v=1)
    with pytest.raises(ValueError, match=r"^onenormest\(\): 'A' must be square, got 2x3$"):
        f(S.csr_array((2, 3)))
    with pytest.raises(ValueError, match=r"^onenormest\(\): 'A' contains NaN or infinity$"):
        f(A * math.inf)


# ================================================================== norm


def test_norms_within_the_contract_against_arb():
    worst = {}
    for p in R.points("norms"):
        m = M.get(p["matrix"])
        A = coo(m).tocsr()
        rows, cols = m.shape
        r, c, _ = m.canonical()
        longest = {
            "1": np.bincount(c, minlength=cols).max(initial=0), "-1": np.bincount(c, minlength=cols).max(initial=0),
            "inf": np.bincount(r, minlength=rows).max(initial=0), "-inf": np.bincount(r, minlength=rows).max(initial=0),
            "fro": len(r),
        }  # fmt: skip
        for key, o in (("1", 1), ("-1", -1), ("inf", INF), ("-inf", -INF), ("fro", "fro"), ("2", 2)):
            if f"{key}_hi" not in p:
                continue
            ref = float(p[f"{key}_hi"][0] + p[f"{key}_lo"][0])
            v = S.norm(A, ord=o)
            assert isinstance(v, float)
            if key == "2":
                err = abs(v - ref) / (R.C_EIG * EPS * ref) if ref else abs(v)
            else:
                err = R.norm_error(v, ref, longest[key])
            assert err <= 1.0, (p["matrix"], key, v, ref, err)
            worst[key] = max(worst.get(key, 0.0), err)
    assert set(worst) == {"1", "-1", "inf", "-inf", "fro", "2"}


def test_norm_recorded_scipy_results():
    data = R.data("scipy")
    for mid in ("lap2d/30", "convdiff2d/40", "random_complex/800", "random_unsym/1000x8"):
        rec = data[f"matfuncs/{mid}"]
        A = coo(M.get(mid)).tocsr()
        for o, key in (("fro", "fro"), (1, "1"), (-1, "-1"), (INF, "inf"), (-INF, "-inf")):
            assert S.norm(A, ord=o) == pytest.approx(rec[f"norm_{key}"], rel=1e-14), (mid, o)
        assert S.norm(A, ord=2) == pytest.approx(rec["norm_2"]["value"], rel=1e-13), mid


def _dense_case():
    g = np.random.default_rng(3)
    D = g.standard_normal((7, 5))
    D[D < 0.3] = 0
    D[2, 1] = 1e200  # a sum of squares that overflows: the rescaled second pass
    D[4, 3] = 1e-200  # a column whose every term is tiny
    D[4, 0] = D[4, 2] = D[4, 4] = 0.0
    D[3] = 0
    return D


@pytest.mark.parametrize("o", [None, 2, 1, 0, -1, INF, -INF, 3, 0.5, -2.5])
def test_vector_norms_as_vector_norm(o):
    D = _dense_case()
    A = S.csr_array(sd.asarray(D))
    for ax in (0, 1, -1, (1,), [0]):
        v = npy(S.norm(A, ord=o, axis=ax))
        w = npy(sd.linalg.vector_norm(sd.asarray(D), ord=2 if o is None else o, axis=ax))
        assert v.dtype == np.float64 and np.allclose(v, w, rtol=1e-14, atol=0, equal_nan=True), (o, ax)
    x = D[:, 1].copy()
    v = S.norm(S.csr_array(sd.asarray(x)), ord=o)
    assert isinstance(v, float)
    assert v == pytest.approx(sd.linalg.vector_norm(sd.asarray(x), ord=2 if o is None else o), rel=1e-14)


@LIVE
def test_norms_as_scipy():
    a = _rand(60, 0.1, 13, cplx=True)
    A = S.csr_array(a)
    for o in (None, "fro", 1, -1, 2, INF, -INF):
        assert S.norm(A, ord=o) == pytest.approx(spla.norm(a, o), rel=1e-13), o
    for o in (None, 1, INF, -INF, 3):
        for ax in (0, 1):
            assert np.allclose(npy(S.norm(A, ord=o, axis=ax)), spla.norm(a, o, axis=ax), rtol=1e-14, atol=0), (o, ax)
    assert np.array_equal(npy(S.norm(A, ord=0, axis=1)), spla.norm(a, 0, axis=1).astype(float))


def test_matrix_norms_as_matrix_norm():
    D = _dense_case()
    A = S.csr_array(sd.asarray(D))
    for o in (None, "fro", 1, -1, 2, INF, -INF, 1.0, np.int64(-1), np.float64(INF)):
        w = sd.linalg.matrix_norm(sd.asarray(D), ord="fro" if o is None else o)
        assert S.norm(A, ord=o) == pytest.approx(w, rel=1e-13), o
        assert S.norm(A, ord=o, axis=(1, 0)) == pytest.approx(
            sd.linalg.matrix_norm(sd.asarray(D.T.copy()), ord="fro" if o is None else o), rel=1e-13
        )
        assert S.norm(A, ord=o, axis=(-2, -1)) == S.norm(A, ord=o)
    # every format: the same bits
    for fmt in ("csc", "coo", "dok"):
        for o in (None, 1, -1, 2, INF, -INF):
            assert S.norm(A.asformat(fmt), ord=o) == S.norm(A, ord=o)


def test_norm_special_values_and_shapes():
    E = S.csr_array((0, 3))
    for o in (None, 1, -1, 2, INF, -INF):
        assert S.norm(E, ord=o) == 0.0
    for o in (None, 1, 0, -1, INF, -INF, 3):  # empty lines: 0, as vector_norm
        assert S.norm(S.csr_array(sd.zeros(0)), ord=o) == 0.0
        assert npy(S.norm(E, ord=o, axis=0)).tolist() == [0.0, 0.0, 0.0]
        assert npy(S.norm(E, ord=o, axis=1)).tolist() == []
    Z = S.csr_array((3, 4))
    assert S.norm(Z, ord=2) == 0.0 and S.norm(Z) == 0.0
    row = S.csr_array(sd.asarray([[3.0, 0.0, -4.0]]))
    assert S.norm(row, ord=2) == 5.0 and S.norm(row.T, ord=2) == 5.0
    N = S.csr_array(sd.asarray([[1.0, math.nan], [0.0, 2.0]]))
    assert math.isnan(S.norm(N)) and math.isnan(S.norm(N, ord=1))
    assert np.isnan(npy(S.norm(N, axis=1))[0]) and npy(S.norm(N, axis=1))[1] == 2.0
    with pytest.raises(ValueError, match=r"^norm\(\): 'x' contains NaN or infinity$"):
        S.norm(N, ord=2)
    # int and bool arrays: float64 results
    I = S.csr_array(sd.asarray([[1, -2], [3, 0]]))
    assert S.norm(I, ord=1) == 4.0 and npy(S.norm(I, axis=0)).dtype == np.float64
    assert S.norm(S.csr_array(sd.asarray([[True, True]])), ord=INF) == 2.0
    # duplicates of a COO are summed first (scipy's norm reads the stored values as they are)
    C = S.coo_array((sd.asarray([1.0, 2.0]), (sd.asarray([0, 0]), sd.asarray([0, 0]))), shape=(1, 1))
    assert S.norm(C) == 3.0


def test_norm_errors():
    A = S.csr_array(sd.eye(3))
    msg = r"^norm\(\): 'ord' must be 'fro', 1, -1, 2, inf or -inf, not "
    for o, shown in (("nuc", "'nuc'"), (-2, "-2"), (3, "3"), ("f", "'f'"), (1.5, "1.5"), ("FRO", "'FRO'")):
        with pytest.raises(ValueError, match=msg + shown + "$"):  # SP101: 'nuc' and -2 refused
            S.norm(A, ord=o)
    with pytest.raises(TypeError, match=r"^norm\(\): 'ord' must be a str, an int or a float, not bool$"):
        S.norm(A, ord=True)
    with pytest.raises(TypeError, match=r"^norm\(\): 'ord' must be an int or a float, not str$"):
        S.norm(A, ord="fro", axis=0)
    with pytest.raises(ValueError, match=r"^norm\(\): 'ord' must not be NaN$"):
        S.norm(A, ord=math.nan, axis=0)
    with pytest.raises(ValueError, match=r"^norm\(\): axis 2 is out of bounds for an array of dimension 2$"):
        S.norm(A, axis=2)
    with pytest.raises(ValueError, match=r"^norm\(\): repeated axis$"):
        S.norm(A, axis=(1, 1))
    with pytest.raises(ValueError, match=r"^norm\(\): 'axis' must name 1 or 2 axes, got 0$"):
        S.norm(A, axis=())
    with pytest.raises(TypeError, match=r"^norm\(\): 'axis' must be an int or a tuple or list of ints, not float$"):
        S.norm(A, axis=1.0)
    with pytest.raises(TypeError, match=r"^norm\(\): 'x' must be a sparse array, not ndarray; for a dense array use "
                                        r"sd.linalg.matrix_norm or sd.linalg.vector_norm$"):  # fmt: skip
        S.norm(sd.eye(3))
    with pytest.raises(TypeError, match=r"^norm\(\): takes 1 positional argument but 2 were given$"):
        S.norm(A, 1)  # ord is a keyword (scipy's is positional)


# ================================================================== bits, threads, linalg


def _band(n):
    t = np.arange(n, dtype=np.float64)
    return S.diags_array([sd.asarray(-1.0 - t / n), sd.full(n - 1, 0.3), sd.full(n - 1, 0.2)], offsets=[0, 1, -1],
                         format="csr")  # fmt: skip


@pytest.mark.usefixtures("restore_threads")
def test_same_bits_on_every_thread_count():
    """n = 4e4: the pool splits the products and the vector operations of expm_multiply and
    onenormest; expm on a 300 x 300 random matrix (SpGEMM rows, the LU solve)."""
    A = _band(40000)
    b = sd.asarray(np.sin(np.arange(40000.0)))
    R0 = S.csr_array(_rand(300, 0.01, 21, shift=-0.5))
    first = None
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        got = [npy(S.expm_multiply(A, b)), npy(S.expm_multiply(A, b, start=0, stop=2, num=4)),
               S.onenormest(A).est, npy(S.onenormest(R0, compute_w=True)[2]), dense_of(S.expm(R0)),
               S.norm(A, ord=2), npy(S.norm(A, axis=0)), S.norm(A)]  # fmt: skip
        if first is None:
            first = got
        else:
            assert all(bits_equal(g, f) for g, f in zip(got, first))


def test_linalg_names_these_functions_for_a_sparse_array():
    A = S.csr_array(sd.eye(3))
    with pytest.raises(TypeError, match=r"^expm\(\): 'x' must be an sd.ndarray, not csr_array; for a sparse array use "
                                        r"sd.sparse.expm$"):  # fmt: skip
        sd.linalg.expm(A)
    for f in (sd.linalg.matrix_norm, sd.linalg.vector_norm):
        with pytest.raises(TypeError, match=r"for a sparse array use sd.sparse.norm$"):
            f(A)


def test_onenormest_result_object():
    """SP102: a read-only OnenormestResult (not exported), fields est, v, w, n_iter, n_op; repr
    and summary() as API.md §2.3."""
    A = S.csr_array(_rand(80, 0.05, 17))
    r = S.onenormest(A, compute_v=True)
    assert type(r).__name__ == "OnenormestResult" and not hasattr(S, "OnenormestResult")
    with pytest.raises(AttributeError):
        r.est = 1.0
    assert r.w is None and r.v.shape == (80,)
    j = int(np.argmax(npy(r.v)))
    assert repr(r) == (f"OnenormestResult(est={r.est:.6g}, v=<array of shape (80,)>, w=None, n_iter={r.n_iter}, "
                       f"n_op={r.n_op})")  # fmt: skip
    text = r.summary()
    assert text.splitlines()[0].strip() == "1-Norm Estimate Results"
    assert "Estimate:" in text and "Column:" in text and str(j) in text and "No. Iterations:" in text
    assert "Column:" not in S.onenormest(A).summary()
