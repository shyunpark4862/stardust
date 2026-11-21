"""The direct solves of sd.sparse (milestone SP4; docs/api/sparse.md §5, docs/design/sparse.md §3.6,
§4): spsolve, splu (SparseLU), factorized, inv, spsolve_triangular, is_sptriangular, spbandwidth.

What the tests hold them to:
  - the contract: the normwise backward error of spsolve is at most C_BACKWARD eps on every
    nonsingular test matrix, and against the verified references (tests/sparse/ref/solve.npz) the
    forward error is within C_SOLVE kappa eps (refdata);
  - singularity (SP62): structurally singular, an exactly zero pivot and the exactly singular
    matrices whose rounded pivots are not zero (kappa_1 eps >= 1) are all LinAlgError naming the
    function; ill-conditioned nonsingular ones (kappa up to 7.5e14) are solved;
  - the condition estimate is a lower bound of kappa_1 and close to it;
  - the format does not change the bits (every input becomes CSC), the thread count does not, nor
    do concurrent Python threads; several right-hand sides are solved together (SP64);
  - the non-rounding behavior is scipy 1.18.1's (shapes, dtypes, result formats, zeros dropped)
    except where docs/api/sparse.md says otherwise;
  - the triangular solve reads the chosen triangle, sums each row as every sparse line sum, and is
    backward stable entry by entry (exact residuals).
"""

import fractions
import threading

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import matrices as M
from . import refdata as R
from .conftest import LIVE, coo

S = sd.sparse
LinAlgError = sd.linalg.LinAlgError
EPS = R.EPS


def sparse(m, fmt="csc"):
    return coo(m).asformat(fmt)


def nonsingular_ids():
    out = []
    for mid in M.ids(heavy=False):
        m = M.get(mid)
        if m.shape[0] != m.shape[1] or m.shape[0] == 0 or m.tags & {"coo", "singular", "struct_singular", "cancel"}:
            continue
        if mid == "empty/rows":  # empty rows: structurally singular
            continue
        out.append(mid)
    return out


NONSINGULAR = nonsingular_ids()
SINGULAR = ["lap1d/10n", "lap2d/10n", "lap2d/60n", "dupcol/40", "dupcol/1000", "struct_singular/40",
            "struct_singular/1000", "cancel/50", "cancel/400", "empty/rows"]  # fmt: skip


# ---------------------------------------------------------------- the contract


@pytest.mark.parametrize("mid", NONSINGULAR)
def test_backward_error_within_the_contract(mid):
    m = M.get(mid)
    b = M.rhs(m)
    x = np.asarray(S.spsolve(sparse(m), sd.asarray(b)))
    assert x.dtype == (np.complex128 if m.dtype == np.complex128 else np.float64)
    assert R.backward_error(m.scipy("csr"), x, b) <= R.C_BACKWARD


@pytest.mark.parametrize("p", R.points("solve"), ids=lambda p: p["matrix"])
def test_forward_error_against_the_verified_references(p):
    m = M.get(p["matrix"])
    assert M.digest(m) == p["digest"]
    x = np.asarray(S.spsolve(sparse(m), sd.asarray(p["b"])))
    assert R.normwise_error(x, p["solve_hi"]) <= R.C_SOLVE * p["solve_cond"] * EPS


@pytest.mark.parametrize("mid", ["saddle/600x150", "random_unsym/1000x8", "lap3d/10", "blockdiag/200x8"])
def test_several_right_hand_sides_and_transposes_within_the_contract(mid):
    m = M.get(mid)
    B = M.rhs(m, 4)
    lu = S.splu(sparse(m))
    d = m.dense()
    for trans, op in (("N", d), ("T", d.T), ("H", d.conj().T)):
        X = np.asarray(lu.solve(sd.asarray(B), trans=trans))
        assert X.shape == B.shape
        assert R.backward_error(op, X, B) <= R.C_BACKWARD, trans


def test_refinement_fixes_threshold_pivoting():
    # saddle/600x150: Accelerate's threshold pivoting alone leaves 404 eps (SP4's measurement); the
    # one refinement step (SP63) brings it under 1 eps.
    m = M.get("saddle/600x150")
    b = M.rhs(m)
    A = sparse(m)
    raw = np.asarray(S._core._lu_solve(A.shape, A.indptr, A.indices, A.data, sd.asarray(b)))
    x = np.asarray(S.spsolve(A, sd.asarray(b)))
    assert R.backward_error(m.dense(), raw, b) > 100
    assert R.backward_error(m.dense(), x, b) < 1


# ---------------------------------------------------------------- singularity (SP62)


@pytest.mark.parametrize("mid", SINGULAR)
@pytest.mark.parametrize("fn", ["spsolve", "splu", "factorized", "inv"])
def test_singular_matrices_are_linalg_error(mid, fn):
    m = M.get(mid)
    A = sparse(m)
    call = {
        "spsolve": lambda: S.spsolve(A, sd.asarray(M.rhs(m))),
        "splu": lambda: S.splu(A),
        "factorized": lambda: S.factorized(A),
        "inv": lambda: S.inv(A),
    }[fn]
    with pytest.raises(LinAlgError, match=rf"^{fn}\(\): singular matrix$"):
        call()


def test_exact_zero_pivot_is_linalg_error():
    A = S.csr_array(sd.asarray([[1.0, 2.0], [2.0, 4.0]]))
    with pytest.raises(LinAlgError, match="singular matrix"):
        S.spsolve(A, sd.ones(2))


def test_zero_matrix_with_a_full_pattern_is_singular():
    A = S.csr_array((sd.zeros(4), sd.asarray([0, 1, 0, 1]), sd.asarray([0, 2, 4])), shape=(2, 2))
    with pytest.raises(LinAlgError, match="singular matrix"):
        S.splu(A)


@pytest.mark.parametrize("mid", ["near_singular/20", "graded/400x12", "graded/100x8", "lap1d/1000"])
def test_ill_conditioned_nonsingular_matrices_are_solved(mid):
    m = M.get(mid)
    lu = S.splu(sparse(m))
    assert lu._factor.core.cond * EPS < 1


@pytest.mark.parametrize("mid", ["lap2d/10", "random_unsym/50x5", "convdiff2d/10", "graded/100x8", "near_singular/20",
                                 "random_complex/40", "blockdiag/20x8", "saddle/40x10"])  # fmt: skip
def test_condition_estimate_is_a_close_lower_bound(mid):
    m = M.get(mid)
    est = S.splu(sparse(m))._factor.core.cond
    true = np.linalg.cond(m.dense(), 1)
    assert true / 10 <= est <= true * (1 + 1e-6)


# ---------------------------------------------------------------- bits


def test_the_format_does_not_change_the_bits():
    m = M.get("random_unsym/1000x8")
    b = sd.asarray(M.rhs(m))
    first = S.spsolve(sparse(m, "csc"), b)
    for fmt in ("csr", "coo", "dok"):
        assert bits_equal(S.spsolve(sparse(m, fmt), b), first), fmt
    assert bits_equal(S.splu(sparse(m, "csr")).solve(b), first)
    assert bits_equal(S.factorized(sparse(m, "coo"))(b), first)


def test_same_bits_on_every_thread_count(restore_threads):
    m = M.get("lap2d/30")
    A, b, B = sparse(m), sd.asarray(M.rhs(m)), sd.asarray(M.rhs(m, 5))
    first = (S.spsolve(A, b), S.spsolve(A, B), S.splu(A).solve(B, trans="T"), S.inv(sparse(M.get("lap2d/10"))))
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        again = (S.spsolve(A, b), S.spsolve(A, B), S.splu(A).solve(B, trans="T"), S.inv(sparse(M.get("lap2d/10"))))
        dense = [np.asarray(v.toarray() if S.issparse(v) else v) for v in again + first]
        assert all(bits_equal(g, f) for g, f in zip(dense[:4], dense[4:]))


def test_concurrent_solves_with_one_factor():
    m = M.get("lap2d/30")
    lu = S.splu(sparse(m))
    b = sd.asarray(M.rhs(m, 3))
    first = lu.solve(b)
    out = [None] * 8

    def run(k):
        out[k] = lu.solve(b, trans="T" if k % 2 else "N")

    threads = [threading.Thread(target=run, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    second = lu.solve(b, trans="T")
    assert all(bits_equal(x, first if k % 2 == 0 else second) for k, x in enumerate(out))


def test_several_columns_are_solved_together():
    # SP64: deterministic for a given k; a column agrees with its own solve to rounding only
    m = M.get("convdiff2d/40")
    B = M.rhs(m, 6)
    A = sparse(m)
    X = np.asarray(S.spsolve(A, sd.asarray(B)))
    assert bits_equal(np.asarray(S.spsolve(A, sd.asarray(B))), X)
    for j in range(6):
        xj = np.asarray(S.spsolve(A, sd.asarray(np.ascontiguousarray(B[:, j]))))
        assert np.abs(X[:, j] - xj).max() <= 4 * EPS * np.abs(xj).max()


# ---------------------------------------------------------------- shapes, dtypes, formats


def test_shapes_and_dtypes():
    m = M.get("lap2d/10")
    A = sparse(m)
    assert S.spsolve(A, sd.ones(100)).shape == (100,)
    assert S.spsolve(A, sd.ones((100, 1))).shape == (100, 1)  # [scipy: (100,)]
    assert S.spsolve(A, sd.ones((100, 3))).shape == (100, 3)
    assert S.spsolve(A, sd.ones(100, dtype=sd.int64)).dtype == sd.float64
    assert S.spsolve(A.astype(sd.int64), sd.ones(100, dtype=sd.bool)).dtype == sd.float64
    assert S.spsolve(A, sd.ones(100) * 1j).dtype == sd.complex128
    assert S.spsolve(A.astype(sd.complex128), sd.ones(100)).dtype == sd.complex128
    lu = S.splu(A)
    assert lu.shape == (100, 100) and repr(lu) == "SparseLU(shape=(100, 100), dtype=float64)"
    assert lu.solve(sd.ones((100, 2))).shape == (100, 2)


def test_real_factor_and_complex_right_hand_side():
    m = M.get("random_unsym/1000x8")
    b = M.rhs(m) + 1j * M.rhs(m, seed=1)
    lu = S.splu(sparse(m))
    for trans, op in (("N", m.dense()), ("T", m.dense().T), ("H", m.dense().T)):
        x = np.asarray(lu.solve(sd.asarray(b), trans=trans))
        assert x.dtype == np.complex128
        assert R.backward_error(op, x, b) <= R.C_BACKWARD


def test_empty_matrix():
    for fmt in ("csr", "csc", "coo"):
        A = S.coo_array((0, 0)).asformat(fmt)
        assert S.spsolve(A, sd.ones(0)).shape == (0,)
        assert S.spsolve(A, sd.ones((0, 3))).shape == (0, 3)
        assert S.splu(A).solve(sd.ones(0)).shape == (0,)
        assert S.inv(A).shape == (0, 0)


QUIET = pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")


@LIVE
@QUIET
def test_sparse_right_hand_side_matches_scipy():
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    m = M.get("lap2d/10")
    rng = np.random.default_rng(3)
    Bd = np.where(rng.random((100, 70)) < 0.05, rng.standard_normal((100, 70)), 0.0)
    for fmt in ("csc", "csr", "coo"):
        X = S.spsolve(sparse(m, fmt), S.csc_array(sd.asarray(Bd)))
        ref = spla.spsolve(m.scipy(fmt if fmt != "coo" else "csc"), sp.csc_array(Bd))
        assert X.format == ref.format == ("csr" if fmt == "csr" else "csc")
        assert X.shape == ref.shape and X.nnz == ref.nnz
        assert np.abs(np.asarray(X.toarray()) - ref.toarray()).max() <= 1e-13
    x1 = S.spsolve(sparse(m), S.csr_array(sd.asarray(Bd[:, 0])))
    assert x1.format == "csr" and x1.shape == (100,)  # [scipy: dense (100,)]


@LIVE
@QUIET
@pytest.mark.parametrize("fmt", ["csc", "csr", "coo"])
def test_inv_matches_scipy(fmt):
    import scipy.sparse.linalg as spla

    m = M.get("convdiff2d/10")
    X = S.inv(sparse(m, fmt))
    ref = spla.inv(m.scipy(fmt))
    assert X.format == ref.format
    assert X.nnz == ref.nnz
    d = np.asarray(X.toarray())
    assert np.abs(d - ref.toarray()).max() <= 1e-14 * np.abs(ref.toarray()).max() * 100
    assert np.abs(d @ m.dense() - np.eye(100)).max() <= 1e-13


def test_inv_drops_exact_zeros():
    A = S.csr_array(sd.asarray([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [1.0, 0.0, 8.0]]))
    X = S.inv(A)
    assert X.format == "csr" and X.nnz == 4
    assert np.array_equal(np.asarray(X.toarray()), np.array([[0.5, 0, 0], [0, 0.25, 0], [-0.0625, 0, 0.125]]))


def test_scipy_records():
    # the non-rounding facts of tests/sparse/data/scipy.json (tools/gen_sparse_scipy.py)
    rec = R.data("scipy")
    for mid in ("lap2d/30", "convdiff2d/40", "random_complex/800", "saddle/600x150", "graded/400x12",
                "blockdiag/200x8", "near_singular/20"):  # fmt: skip
        m = M.get(mid)
        r = rec[f"direct/{mid}/spsolve"]
        assert r["digest"] == M.digest(m) and r["finite"]
        A = sparse(m)
        x = np.asarray(S.spsolve(A, sd.asarray(M.rhs(m))))
        assert np.abs(x).max() == pytest.approx(r["max_abs_x"], rel=100 * EPS * S.splu(A)._factor.core.cond)
    for mid in ("struct_singular/40",):
        assert "error" in rec[f"direct/{mid}/splu"] and "MatrixRankWarning" in rec[f"direct/{mid}/spsolve"]["warnings"]
    # scipy solves these exactly singular matrices to a finite 1/eps-sized x without a word; sd
    # says singular (SP62)
    for mid in ("dupcol/40", "lap2d/10n"):
        assert rec[f"direct/{mid}/spsolve"]["finite"] and rec[f"direct/{mid}/spsolve"]["max_abs_x"] > 1e14
    assert rec["direct/spsolve(b sparse)"]["format"] == "csc"
    assert rec["direct/spsolve(b (n,1))"]["shape"] == [100]


# ---------------------------------------------------------------- argument errors


def test_argument_errors():
    m = M.get("lap2d/4")
    A = sparse(m)
    cases = [
        (lambda: S.spsolve(sd.eye(16), sd.ones(16)), TypeError,
         "spsolve(): 'A' must be a sparse array, not ndarray; for a dense array use sd.linalg.solve"),
        (lambda: S.splu([[1.0]]), TypeError, "splu(): 'A' must be a sparse array, not list"),
        (lambda: S.spsolve(A, [1.0] * 16), TypeError,
         "spsolve(): 'b' must be an sd.ndarray or a sparse array, not list"),
        (lambda: S.spsolve(A, np.ones(16)), TypeError,
         "spsolve(): 'b' must be an sd.ndarray or a sparse array, not numpy.ndarray"),
        (lambda: S.spsolve(A, sd.ones(3)), ValueError, "spsolve(): 'A' is 16x16 but 'b' has 3 rows"),
        (lambda: S.spsolve(A, sd.ones((16, 2, 2))), ValueError, "spsolve(): 'b' must have 1 or 2 dimensions, got 3"),
        (lambda: S.spsolve(S.csr_array((3, 4)), sd.ones(3)), ValueError, "spsolve(): 'A' must be square, got 3x4"),
        (lambda: S.splu(S.csr_array(sd.ones(3))), ValueError, "splu(): 'A' must have 2 dimensions, got 1"),
        (lambda: S.spsolve(S.csr_array(sd.asarray([[1.0, float("nan")], [0, 1]])), sd.ones(2)), ValueError,
         "spsolve(): 'A' contains NaN or infinity"),
        (lambda: S.spsolve(A, sd.ones(16) * float("inf")), ValueError, "spsolve(): 'b' contains NaN or infinity"),
        (lambda: S.splu(A).solve(sd.ones(3)), ValueError, "solve(): the factor is 16x16 but 'b' has 3 rows"),
        (lambda: S.splu(A).solve(sd.ones(16), trans="C"), ValueError,
         "solve(): 'trans' must be 'N', 'T' or 'H', not 'C'"),
        (lambda: S.splu(A).solve(sd.ones(16), trans=None), TypeError,
         "solve(): 'trans' must be 'N', 'T' or 'H', not NoneType"),
        (lambda: S.splu(A).solve(S.csr_array(sd.ones((16, 1)))), TypeError,
         "solve(): 'b' must be an sd.ndarray, not csr_array"),
        (lambda: S.spsolve(A, sd.ones(16), permc_spec="NATURAL"), TypeError,
         "spsolve(): unexpected keyword argument 'permc_spec'"),
        (lambda: S.splu(A, diag_pivot_thresh=1.0), TypeError,
         "splu(): unexpected keyword argument 'diag_pivot_thresh'"),
        (lambda: S.spsolve(A, sd.ones(16), use_umfpack=False), TypeError,
         "spsolve(): unexpected keyword argument 'use_umfpack'"),
        (lambda: S.splu(A, ordering="metis"), TypeError, "splu(): unexpected keyword argument 'ordering'"),
        (lambda: S.inv(sd.eye(2)), TypeError,
         "inv(): 'A' must be a sparse array, not ndarray; for a dense array use sd.linalg.inv"),
        (lambda: S.spsolve(A, b=sd.ones(16)), TypeError, None),
        (lambda: S._direct.SparseLU(), TypeError, "SparseLU(): use sd.sparse.splu(A)"),
    ]  # fmt: skip
    for call, kind, msg in cases:
        with pytest.raises(kind) as e:
            call()
        if msg is not None:
            assert str(e.value) == msg


# ---------------------------------------------------------------- triangular solves


def _exact_residual_bound(T, x, b):
    """max_i |b - T x|_i / (gamma_n (|T| |x|)_i), the residual in exact rational arithmetic (real
    and imaginary parts apart for complex; then the componentwise bound uses |re| + |im|)."""
    n = T.shape[0]
    worst = 0.0
    g = R.gamma(n) * 2  # the complex product's extra rounding
    for i in range(n):
        cols = np.flatnonzero(T[i])
        for part in ("real", "imag"):
            fx = lambda v: fractions.Fraction(float(getattr(v, part)))  # noqa: E731
            if part == "imag" and not np.iscomplexobj(T) and not np.iscomplexobj(x):
                continue
            if np.iscomplexobj(T) or np.iscomplexobj(x):
                acc = fx(b[i])
                for j in cols:
                    t, v = complex(T[i, j]), complex(x[j])
                    re = fractions.Fraction(t.real) * fractions.Fraction(v.real) - fractions.Fraction(
                        t.imag
                    ) * fractions.Fraction(v.imag)
                    im = fractions.Fraction(t.real) * fractions.Fraction(v.imag) + fractions.Fraction(
                        t.imag
                    ) * fractions.Fraction(v.real)
                    acc -= re if part == "real" else im
            else:
                acc = fractions.Fraction(float(b[i])) - sum(
                    fractions.Fraction(float(T[i, j])) * fractions.Fraction(float(x[j])) for j in cols
                )
            scale = float(
                sum((abs(T[i, j].real) + abs(T[i, j].imag)) * (abs(x[j].real) + abs(x[j].imag)) for j in cols)
            )
            if scale:
                worst = max(worst, abs(float(acc)) / (g * scale))
    return worst


@pytest.mark.parametrize("mid", ["lap2d/10", "random_unsym/50x5", "random_complex/40", "graded/100x8", "long_rows/300"])
@pytest.mark.parametrize("lower", [True, False])
def test_triangular_solve_is_backward_stable_entrywise(mid, lower):
    m = M.get(mid)
    d = m.dense()
    T = np.tril(d) if lower else np.triu(d)
    b = M.rhs(m)
    x = np.asarray(S.spsolve_triangular(sparse(m, "csr"), sd.asarray(b), upper=not lower))
    assert _exact_residual_bound(T, x, b) <= 1


def _substitution(T, b, lower, unit):
    """The defined order in Python: s_i from zero in ascending column order with fma, then
    (b_i - s_i) / t_ii (real matrices of rows shorter than L)."""
    import math

    n = T.shape[0]
    x = np.zeros(n)
    order = range(n) if lower else range(n - 1, -1, -1)
    for i in order:
        cols = [j for j in np.flatnonzero(T[i]) if (j < i if lower else j > i)]
        s = 0.0
        for j in cols:
            s = math.fma(float(T[i, j]), float(x[j]), s)
        x[i] = b[i] - s if unit else (b[i] - s) / T[i, i]
    return x


@pytest.mark.parametrize("mid", ["lap2d/10", "random_unsym/50x5", "long_rows/300", "banded/200x2x3"])
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("unit", [False, True])
def test_triangular_solve_follows_the_defined_order_bit_for_bit(mid, lower, unit):
    m = M.get(mid)
    d = m.dense()
    b = M.rhs(m)
    x = np.asarray(S.spsolve_triangular(sparse(m, "csr"), sd.asarray(b), upper=not lower, unit_diagonal=unit))
    assert bits_equal(x, _substitution(d, b, lower, unit))


def test_triangular_solve_columns_formats_and_threads(restore_threads):
    m = M.get("long_rows/9000")  # rows of 9000 entries: the line sum's pieces of L
    d = m.dense()
    B = M.rhs(m, 20)
    first = np.asarray(S.spsolve_triangular(sparse(m, "csr"), sd.asarray(B), upper=False))
    for j in (0, 7, 19):  # a column's bits do not depend on the others
        xj = np.asarray(S.spsolve_triangular(sparse(m, "csr"), sd.asarray(np.ascontiguousarray(B[:, j])), upper=False))
        assert bits_equal(first[:, j].copy(), xj)
    for fmt in ("csc", "coo", "dok"):
        assert bits_equal(np.asarray(S.spsolve_triangular(sparse(m, fmt), sd.asarray(B), upper=False)), first)
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        assert bits_equal(np.asarray(S.spsolve_triangular(sparse(m, "csr"), sd.asarray(B), upper=False)), first)
    assert R.backward_error(np.tril(d), first, B) <= R.C_BACKWARD


def test_triangular_solve_ignores_the_other_triangle_and_the_unit_diagonal():
    m = M.get("random_unsym/50x5")
    b = sd.asarray(M.rhs(m))
    A = sparse(m, "csr")
    assert bits_equal(
        S.spsolve_triangular(A, b, upper=False), S.spsolve_triangular(S.tril(A, format="csr"), b, upper=False)
    )
    assert bits_equal(
        S.spsolve_triangular(A, b, upper=True), S.spsolve_triangular(S.triu(A, format="csr"), b, upper=True)
    )
    off = S.tril(A, k=-1, format="csr")  # no diagonal at all: read as ones
    assert bits_equal(
        S.spsolve_triangular(off, b, upper=False, unit_diagonal=True),
        S.spsolve_triangular(A, b, upper=False, unit_diagonal=True),
    )


def test_triangular_solve_dtypes_and_shapes():
    A = S.csr_array(sd.asarray([[2, 0], [1, 4]]))
    assert S.spsolve_triangular(A, sd.ones(2, dtype=sd.int64), upper=False).dtype == sd.float64
    assert S.spsolve_triangular(A, sd.ones((2, 1)), upper=False).shape == (2, 1)
    x = S.spsolve_triangular(A, sd.ones(2) * 1j, upper=False)
    assert x.dtype == sd.complex128 and np.allclose(np.asarray(x), [0.5j, 0.125j])
    assert S.spsolve_triangular(S.csr_array((0, 0)), sd.ones(0), upper=False).shape == (0,)
    assert S.spsolve_triangular(A, sd.ones(2)[::-1], upper=False).shape == (2,)  # a strided b


def test_triangular_solve_errors():
    A = S.csr_array(sd.asarray([[2.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(LinAlgError, match=r"^spsolve_triangular\(\): singular matrix$"):
        S.spsolve_triangular(A, sd.ones(2), upper=False)  # a stored zero on the diagonal
    with pytest.raises(LinAlgError, match="singular matrix"):
        S.spsolve_triangular(
            S.csr_array(sd.asarray([[2.0, 0.0], [1.0, 0.0]])).tocoo().tocsr()[:, :], sd.ones(2), upper=False
        )
    with pytest.raises(LinAlgError, match="singular matrix"):
        S.spsolve_triangular(S.eye_array(3, k=-1, format="csr"), sd.ones(3), upper=False)  # no diagonal
    assert np.asarray(
        S.spsolve_triangular(S.eye_array(3, k=-1, format="csr"), sd.ones(3), upper=False, unit_diagonal=True)
    ).tolist() == [1, 0, 1]
    with pytest.raises(TypeError, match=r"^spsolve_triangular\(\): 'upper' must be True or False, not int$"):
        S.spsolve_triangular(A, sd.ones(2), upper=1)
    with pytest.raises(TypeError, match=r"'unit_diagonal' must be True or False, not NoneType"):
        S.spsolve_triangular(A, sd.ones(2), upper=False, unit_diagonal=None)
    with pytest.raises(TypeError, match="for a dense array use sd.linalg.solve_triangular"):
        S.spsolve_triangular(sd.eye(2), sd.ones(2), upper=False)
    with pytest.raises(ValueError, match="contains NaN or infinity"):
        S.spsolve_triangular(A, sd.asarray([1.0, float("nan")]), upper=False)
    # upper is a required keyword, as in sd.linalg.solve_triangular (SP68); scipy's lower is not a keyword
    with pytest.raises(TypeError, match=r"^spsolve_triangular\(\): missing 1 required keyword-only argument: 'upper'$"):
        S.spsolve_triangular(A, sd.ones(2))
    with pytest.raises(TypeError, match=r"^spsolve_triangular\(\): unexpected keyword argument 'lower'$"):
        S.spsolve_triangular(A, sd.ones(2), upper=False, lower=True)
    with pytest.raises(TypeError, match="unexpected keyword argument 'overwrite_b'"):
        S.spsolve_triangular(A, sd.ones(2), upper=False, overwrite_b=True)


# ---------------------------------------------------------------- patterns


@LIVE
@pytest.mark.parametrize("mid", ["lap2d/10", "random_unsym/50x5", "banded/200x2x3", "rect/60x20", "rect/20x60",
                                 "coo/dups", "coo/cancel", "empty/4x6", "long_rows/300"])  # fmt: skip
@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "dok"])
def test_bandwidth_and_triangularity_match_scipy(mid, fmt):
    import scipy.sparse.linalg as spla

    m = M.get(mid)
    A = coo(m).asformat(fmt)
    s = m.scipy("csr")
    if s.nnz:
        assert S.spbandwidth(A) == spla.spbandwidth(s)
    else:
        assert S.spbandwidth(A) == (0, 0)  # scipy 1.18.1 fails on an empty pattern
    assert S.is_sptriangular(A) == tuple(bool(v) for v in spla.is_sptriangular(s))
    L = S.tril(A)
    assert S.is_sptriangular(L)[0] and S.spbandwidth(L)[1] == 0


def test_patterns_count_stored_zeros_and_empty():
    A = S.csr_array((sd.asarray([1.0, 0.0]), sd.asarray([0, 2]), sd.asarray([0, 2, 2, 2])), shape=(3, 3))
    assert S.spbandwidth(A) == (0, 2) and S.is_sptriangular(A) == (False, True)
    assert S.spbandwidth(S.csr_array((0, 0))) == (0, 0) and S.is_sptriangular(S.csr_array((3, 3))) == (True, True)
    with pytest.raises(ValueError, match=r"^spbandwidth\(\): 'A' must have 2 dimensions, got 1$"):
        S.spbandwidth(S.csr_array(sd.ones(3)))
    msg = r"must be a sparse array, not ndarray; for a dense array use sd\.linalg\.bandwidth$"
    with pytest.raises(TypeError, match=r"^is_sptriangular\(\): 'A' " + msg):
        S.is_sptriangular(sd.eye(2))
    with pytest.raises(TypeError, match=r"^spbandwidth\(\): 'A' " + msg):
        S.spbandwidth(sd.eye(2))


def test_spbandwidth_is_the_dense_bandwidth():
    # the same result as sd.linalg.bandwidth (BandwidthResult(lower, upper), a tuple as scipy's)
    for D in (np.triu(np.ones((4, 6)), 2), np.tril(np.ones((5, 3)), -1), np.eye(3), np.zeros((2, 2))):
        A = S.csr_array(sd.asarray(D))
        for X in (A, A.tocsc(), A.tocoo()):
            r = S.spbandwidth(X)
            assert type(r) is type(sd.linalg.bandwidth(sd.asarray(D))) and r == sd.linalg.bandwidth(sd.asarray(D))
            assert (r.lower, r.upper) == tuple(r)
