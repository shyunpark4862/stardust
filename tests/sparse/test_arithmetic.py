"""The operators of SP2 (docs/api/sparse.md §3.1, docs/design/sparse.md §2.4, §3.2): sparse with
sparse, with a scalar and with a dense sd.ndarray, and the comparisons.

Every stored value is the array's own operation on the stored values, so the tests compare bit for
bit with the dense sd operation on A.toarray() (where the rule says the implicit zero takes part)
or on the stored values (where it does not: an implicit zero is a strong zero, SP12). The
non-rounding behavior -- the result's format, its stored entries, the dtypes -- is compared with
scipy 1.18.1 live. Where sd decided otherwise the test says so: A / 0 has no ZeroDivisionError
(SP12), A + c and A ** 0 are a ValueError (SP39; scipy: NotImplementedError), A * dense keeps A's
format (SP38; scipy: COO), a numpy operand is a TypeError (SP35), sparse * sparse keeps only the
positions both store (scipy computes nan * 0 = nan where one side stores a NaN), comparisons
follow IEEE for NaN (scipy's <=, >= are the negation of >, <), a dok_array + sparse is CSR (scipy:
DOK)."""

import operator
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import stardust as sd

from .._bits import bits_equal, value_equal
from . import matrices as M
from .conftest import LIVE, coo, npy

S = sd.sparse
FORMATS = ("coo", "csr", "csc", "dok")
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
CMP = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
REAL = ["random_unsym/50x5", "coo/dups", "coo/int", "coo/bool", "coo/nan_inf", "empty/rows", "cancel/50"]


def pair(mid, fa="csr", fb="csr", seed=1):
    """Matrix mid and a second matrix of the same shape (the transpose when square, else a
    shuffled copy), as sd arrays of formats fa and fb."""
    m = M.get(mid)
    A = coo(m).asformat(fa)
    g = np.random.default_rng(seed)
    perm = g.permutation(m.nnz)
    data = m.data[perm]
    if m.data.dtype.kind in "fc":
        data = data * g.choice([-1.0, 1.0, 0.5], m.nnz)
    row, col = m.row, m.col[perm] if m.shape[1] else m.col
    B = S.coo_array((sd.asarray(data), (sd.asarray(row), sd.asarray(col))), shape=m.shape).asformat(fb)
    return A, B


def dense(A):
    return npy(A.toarray())


def no_stored_zeros(R):
    return R.count_nonzero() == R.nnz


def result_format(fa, fb):
    """sd's format of sparse (op) sparse: CSC when the left is CSC, else CSR."""
    return "csc" if fa == "csc" else "csr"


# ---- sparse with sparse ----


@pytest.mark.parametrize("mid", REAL + ["random_complex/40"])
@pytest.mark.parametrize("sym", ["+", "-"])
def test_add_subtract_is_the_dense_operation_on_the_union(mid, sym):
    A, B = pair(mid)
    R = OPS[sym](A, B)
    assert R.format == "csr" and no_stored_zeros(R)
    want = OPS[sym](A.toarray(), B.toarray())
    assert bits_equal(dense(R), npy(want + 0))  # +0: the dense result's -0.0 is a dropped zero


def stored(A):
    """Where A stores a value (a bool mask of A's shape)."""
    mask = np.zeros(A.shape, bool)
    mask[tuple(npy(c) for c in A.tocoo().coords)] = True
    return mask


@pytest.mark.parametrize("mid", REAL + ["random_complex/40"])
def test_multiply_keeps_the_positions_both_store(mid):
    A, B = pair(mid)
    R = A * B
    assert R.format == "csr" and no_stored_zeros(R)
    prod = npy(sd.multiply(A.toarray(), B.toarray()))
    want = np.where(stored(A) & stored(B), prod, np.zeros_like(prod))
    assert bits_equal(dense(R), want + np.zeros_like(want))


def test_strong_zero_differs_from_scipy_where_a_nan_meets_an_implicit_zero():
    d1 = np.array([[np.nan, 1.0, np.inf, 0.0], [2.0, 0.0, 0.0, 0.0]])
    d2 = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]])
    A, B = S.csr_array(sd.asarray(d1)), S.csr_array(sd.asarray(d2))
    R = A * B
    assert R.nnz == 1 and R.toarray().tolist() == [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    s = sp.csr_array(d1) * sp.csr_array(d2)  # scipy: nan * 0 and inf * 0 are stored NaN
    assert s.nnz == 3 and np.isnan(s.toarray()[0, [0, 2]]).all()


@pytest.mark.parametrize("fa", FORMATS)
@pytest.mark.parametrize("fb", FORMATS)
@LIVE
def test_formats_and_entries_match_scipy(fa, fb):
    A, B = pair("random_unsym/50x5", fa, fb)
    a, b = A.to_scipy(copy=True), B.to_scipy(copy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sym, f in {**OPS, **CMP}.items():
            R, s = f(A, B), f(a, b)
            want = result_format(fa, fb)
            assert R.format == want, (sym, R.format)
            if fa == "dok" and sym == "+":  # scipy: dok + sparse is a DOK keeping the zeros it makes
                assert s.format == "dok"
            else:
                assert s.format == want and R.nnz == s.nnz, sym
            assert value_equal(dense(R), s.toarray()), sym
        for name in ("maximum", "minimum"):
            R, s = getattr(A, name)(B), getattr(a, name)(b)
            assert R.format == s.format == result_format(fa, fb) and R.nnz == s.nnz
            assert value_equal(dense(R), s.toarray())


@pytest.mark.parametrize("mid", REAL)
@pytest.mark.parametrize("sym", list(CMP))
def test_comparisons_are_ieee_everywhere(mid, sym):
    A, B = pair(mid)
    R = CMP[sym](A, B)
    assert R.dtype == sd.bool and R.format == "csr"
    assert bits_equal(dense(R), npy(CMP[sym](A.toarray(), B.toarray())))
    assert R.nnz == R.count_nonzero()  # only True is stored


def test_comparisons_with_nan_follow_ieee_not_scipy():
    d1 = np.array([[np.nan, 1.0], [0.0, 0.0]])
    d2 = np.array([[0.0, 1.0], [0.0, 0.0]])
    A, B = S.csr_array(sd.asarray(d1)), S.csr_array(sd.asarray(d2))
    assert (A <= B).toarray().tolist() == [[False, True], [True, True]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sp.csr_array(d1).__le__(sp.csr_array(d2)).toarray()[0, 0]  # scipy: not (nan > 0)


@pytest.mark.parametrize("name", ["maximum", "minimum"])
@pytest.mark.parametrize("mid", REAL)
def test_maximum_minimum_are_the_array_functions_on_the_union(name, mid):
    A, B = pair(mid)
    if A.dtype == sd.complex128:
        return
    R = getattr(A, name)(B)
    assert no_stored_zeros(R)
    want = npy(getattr(sd, name)(A.toarray(), B.toarray()))
    if want.dtype.kind == "f":
        want = want + 0.0  # a -0.0 of the dense result is a dropped zero
    assert bits_equal(dense(R), want)


@pytest.mark.parametrize(
    "da, db, want",
    [("bool", "bool", "int64"), ("bool", "int64", "int64"), ("int64", "float64", "float64"),
     ("float64", "complex128", "complex128"), ("bool", "float64", "float64")],
)  # fmt: skip
def test_dtypes_promote_as_arrays(da, db, want):
    A = S.csr_array(sd.asarray([[1, 0], [0, 1]], dtype=getattr(sd, da)))
    B = S.csr_array(sd.asarray([[1, 1], [0, 0]], dtype=getattr(sd, db)))
    for f in (operator.add, operator.sub, operator.mul):
        assert f(A, B).dtype == getattr(sd, want)
        assert f(A, B).dtype == f(A.toarray(), B.toarray()).dtype
    assert (A == B).dtype == sd.bool


def test_bool_maximum_stays_bool_and_complex_has_no_order():
    A = S.csr_array(sd.asarray([[True, False], [False, True]]))
    assert A.maximum(A).dtype == sd.bool
    C = S.csr_array(sd.asarray([[1j, 0], [0, 1]]))
    with pytest.raises(TypeError, match=r"unsupported operand dtypes for <: complex128 and complex128"):
        _ = C < C
    with pytest.raises(TypeError, match=r"^maximum\(\): not supported for complex128"):
        C.maximum(C)


def test_shapes_must_match():
    A = S.csr_array((2, 3))
    with pytest.raises(ValueError, match=r"^shapes could not be broadcast together: \(2, 3\) \(3, 2\)$"):
        A + A.T
    with pytest.raises(ValueError, match=r"^multiply\(\): shapes could not be broadcast together"):
        A.multiply(S.csr_array((1, 3)))
    with pytest.raises(ValueError):
        _ = A == S.csr_array((2, 2))  # scipy: False


def test_one_dimensional_arrays():
    a = S.coo_array(sd.asarray([1.0, 0.0, 2.0, 0.0]))
    b = S.csr_array(sd.asarray([0.0, 3.0, -2.0, 0.0]))
    assert (a + b).format == "csr" and (a + b).toarray().tolist() == [1.0, 3.0, 0.0, 0.0]
    assert (a + b).nnz == 2 and (a * b).toarray().tolist() == [0.0, 0.0, -4.0, 0.0]
    assert (a == b).toarray().tolist() == [False, False, False, True]


# ---- sparse with a scalar ----


def values(A):
    """The stored values in the order with_values sees them (a DOK's: row-major)."""
    return A.tocoo().data if A.format == "dok" else A.data


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("mid", REAL + ["random_complex/40", "coo/complex"])
def test_scalar_multiply_divide_keep_the_pattern_and_format(fmt, mid):
    A = coo(M.get(mid)).asformat(fmt)
    v = values(A)
    for c in (2, 0.5, 0, -1.5, float("nan"), float("inf"), 1j):
        if c == 1j and A.dtype in (sd.bool, sd.int64):
            continue
        for R, want in ((A * c, sd.multiply(v, c)), (c * A, sd.multiply(c, v)), (A / c, sd.divide(v, c))):
            assert R.format == fmt and R.nnz == A.nnz  # A * 0 keeps the pattern (scipy)
            assert bits_equal(npy(values(R)), npy(want))


def test_divide_by_zero_is_inf_and_nan_at_the_stored_values_only():
    B = S.csr_array((sd.asarray([1.0, 0.0, -3.0]), sd.asarray([0, 1, 0]), sd.asarray([0, 2, 3])), shape=(2, 2))
    R = B / 0
    assert R.nnz == 3 and npy(R.toarray()).tolist()[1] == [-np.inf, 0.0]
    assert np.isnan(npy(R.toarray())[0, 1]) and npy(R.toarray())[0, 0] == np.inf
    with pytest.raises(ZeroDivisionError):
        sp.csr_array(np.eye(2)) / 0  # [scipy와 다름]


def test_negate_and_abs():
    m = M.get("coo/dups")
    A = coo(m)
    N = -A
    assert N.format == "coo" and N.nnz == A.nnz  # linear: the repeats stay (scipy)
    assert bits_equal(npy(N.data), npy(sd.negative(A.data)))
    P = abs(A)
    assert P.format == "coo" and P.has_canonical_format  # nonlinear: the repeats summed first (scipy)
    assert bits_equal(npy(P.toarray()), np.abs(npy(A.toarray())))
    D = A.todok()
    assert (-D).format == "dok" and abs(D).format == "csr"  # scipy's formats
    assert (-S.csr_array(sd.asarray([True, False]))).dtype == sd.int64  # the array's rule (scipy: an error)


def test_adding_a_scalar():
    A = S.csr_array(sd.asarray([[1.0, 0.0], [0.0, -0.0]]))
    for R in (A + 0, 0 + A, A - 0, A + 0.0, A + False):
        assert R.format == "csr" and R.nnz == A.nnz
    assert bits_equal(npy((0 - A).data), npy(sd.subtract(0, A.data)))
    with pytest.raises(ValueError, match=r"^adding a nonzero scalar to a sparse array is not supported; "
                                         r"use A.toarray\(\) \+ c$"):  # fmt: skip
        A + 1
    with pytest.raises(ValueError, match="adding a nonzero scalar"):
        1.5 + A
    with pytest.raises(ValueError, match=r"^subtracting a nonzero scalar from a sparse array"):
        A - 1
    with pytest.raises(ValueError, match=r"^subtracting a sparse array from a nonzero scalar"):
        1 - A
    with pytest.raises(ValueError, match="nonzero scalar"):
        A + float("nan")


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("sym", list(CMP))
@pytest.mark.parametrize("c", [0, 1, -1, 0.5, float("nan")])
@LIVE
def test_scalar_comparisons(fmt, sym, c):
    A = coo(M.get("random_unsym/50x5")).asformat(fmt)
    R = CMP[sym](A, c)
    assert R.format == ("csc" if fmt == "csc" else "csr") and R.dtype == sd.bool
    assert bits_equal(dense(R), npy(CMP[sym](A.toarray(), c)))
    assert R.nnz == R.count_nonzero()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = CMP[sym](A.to_scipy(copy=True) if fmt != "dok" else A.tocsr().to_scipy(), c)
    assert R.nnz == s.nnz


@pytest.mark.parametrize("name", ["maximum", "minimum"])
@pytest.mark.parametrize("c", [0, -1, 2, 0.5, -0.5])
@LIVE
def test_scalar_maximum_minimum(name, c):
    A = coo(M.get("random_unsym/50x5"))
    R = getattr(A, name)(c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy's SparseEfficiencyWarning where it densifies
        s = getattr(A.to_scipy(copy=True), name)(c)
    zero_kept = getattr(np, name)(0, c) != 0
    assert R.format == s.format == ("coo" if zero_kept else "csr")  # scipy densifies into A's class
    assert R.nnz == s.nnz
    assert bits_equal(dense(R), npy(getattr(sd, name)(A.toarray(), c)) + 0.0)


def test_power_and_pow():
    A = coo(M.get("coo/dups"))
    R = A**2
    assert R.format == "coo" and R.has_canonical_format
    assert bits_equal(npy(R.toarray()), npy(sd.pow(A.toarray(), 2)))
    assert A.power(0.5).format == "coo"
    assert S.dok_array(A).power(2).format == "csr"  # scipy
    with pytest.raises(ValueError, match=r"^the power 0 is not supported: the result would not be sparse"):
        A**0
    with pytest.raises(ValueError, match=r"^power\(\): the power 0 is not supported"):
        A.power(0)
    with pytest.raises(ValueError, match=r"^power\(\): 'n' must be positive"):
        A.power(-1)
    with pytest.raises(TypeError, match=r"^power\(\): 'n' must be a real number, not complex"):
        A.power(1j)
    with pytest.raises(TypeError):
        A ** sd.asarray([1.0])


# ---- sparse with a dense array ----


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dshape", [(50, 50), (50,), (1, 50), (50, 1), (1, 1)])
def test_multiply_divide_dense_keep_the_pattern(fmt, dshape):
    A = coo(M.get("random_unsym/50x5")).asformat(fmt)
    g = np.random.default_rng(3)
    D = g.standard_normal(dshape)
    D[0] = np.nan  # an implicit zero times NaN stays 0 (SP12, scipy)
    Ds = sd.asarray(D)
    for R in (A * Ds, Ds * A, A.multiply(Ds), A / Ds):
        assert R.format == fmt and R.nnz == A.nnz  # [scipy와 다름]: scipy gives COO
    full = np.broadcast_to(D, A.shape)
    c = A.tocoo()
    r, k = npy(c.coords[0]), npy(c.coords[1])
    vals = npy(c.data)
    got = (A * Ds).tocoo()
    assert bits_equal(npy(got.data), vals * full[r, k])
    assert bits_equal(npy((A / Ds).tocoo().data), vals / full[r, k])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = A.tocsr().to_scipy().multiply(D)
    assert s.format == ("csr" if dshape == (1, 1) else "coo") and s.nnz == A.nnz  # scipy: (1, 1) is a scalar


@pytest.mark.parametrize("fmt", FORMATS)
def test_add_subtract_compare_dense_are_dense(fmt):
    A = coo(M.get("coo/dups")).asformat(fmt)
    D = sd.asarray(np.random.default_rng(4).standard_normal(A.shape))
    Ad = A.toarray()
    for got, want in ((A + D, Ad + D), (D + A, D + Ad), (A - D, Ad - D), (D - A, D - Ad), (A == D, Ad == D),
                      (A < D, Ad < D), (D >= A, D >= Ad), (A.maximum(D), sd.maximum(Ad, D)),
                      (A.minimum(D), sd.minimum(Ad, D))):  # fmt: skip
        assert isinstance(got, sd.ndarray) and bits_equal(npy(got), npy(want))
    row = sd.asarray(np.arange(A.shape[1], dtype=float))
    assert bits_equal(npy(A + row), npy(Ad + row))


def test_a_dense_operand_must_broadcast_to_the_sparse_shape():
    A = S.csr_array(sd.asarray([[1.0, 0.0, 2.0]]))
    with pytest.raises(ValueError, match=r"^shapes could not be broadcast together: \(1, 3\) \(4, 3\)$"):
        A * sd.ones((4, 3))
    with pytest.raises(ValueError):
        A + sd.ones((2,))
    with pytest.raises(ValueError, match=r"^multiply\(\): shapes could not be broadcast"):
        A.multiply(sd.ones((4, 3)))


def test_sparse_divided_by_sparse_is_dense():
    A, B = pair("random_unsym/50x5")
    R = A / B
    assert isinstance(R, sd.ndarray) and bits_equal(npy(R), npy(A.toarray() / B.toarray()))
    with pytest.raises(TypeError):
        sd.ones(A.shape) / A  # scipy: NotImplemented as well
    with pytest.raises(TypeError):
        2 / A


# ---- operands that are not ----


@pytest.mark.parametrize("other", [np.ones((2, 2)), [[1.0, 0.0], [0.0, 1.0]], None, "x"])
def test_numpy_arrays_and_lists_are_not_operands(other):
    A = S.csr_array(sd.asarray([[1.0, 0.0], [0.0, 2.0]]))
    for f in (operator.add, operator.sub, operator.mul, operator.truediv, operator.matmul, operator.lt,
              operator.le, operator.gt, operator.ge):  # fmt: skip
        with pytest.raises(TypeError):
            f(A, other)
        with pytest.raises(TypeError):
            f(other, A)
    for f in (operator.eq, operator.ne):  # never Python's quiet False
        with pytest.raises(TypeError, match="not supported between instances of"):
            f(A, other)
    for name in ("multiply", "maximum", "minimum"):
        with pytest.raises(TypeError, match=rf"^{name}\(\): 'other' must be a sparse array, an sd.ndarray or a number"):
            getattr(A, name)(other)


def test_numpy_scalars_are_operands():
    A = S.csr_array(sd.asarray([[1.0, 0.0], [0.0, 2.0]]))
    for c in (np.float64(2.0), np.int64(2), np.bool_(True), np.float32(2.0)):
        assert bits_equal(npy((A * c).data), npy(A.data * c))


def test_array_functions_point_at_the_methods():
    A = S.csr_array(sd.asarray([[1.0, 0.0], [0.0, 2.0]]))
    with pytest.raises(TypeError, match=r"^sin\(\): 'x' must be an sd.ndarray .* not csr_array; for a sparse array "
                                        r"use A.sin\(\)$"):  # fmt: skip
        sd.sin(A)
    with pytest.raises(TypeError, match=r"use A.asin\(\)$"):
        sd.asin(A)
    with pytest.raises(TypeError, match=r"^cos\(\): .*; a sparse array has no cos: use cos\(A.toarray\(\)\)$"):
        sd.cos(A)
    with pytest.raises(TypeError, match=r"^add\(\): 'x2' .* not coo_array; for a sparse array use A \+ B$"):
        sd.add(1.0, A.tocoo())
    with pytest.raises(TypeError, match=r"use A.maximum\(B\)$"):
        sd.maximum(A, A)
