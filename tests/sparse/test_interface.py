"""LinearOperator and aslinearoperator (milestone SP4; docs/api/sparse.md §5): scipy 1.18.1's
interface on sd arrays. shape and matvec positional, the rest keywords (SP65); dtype=None asks
matvec once on zeros; the products, the combinations (+, -, scalar *, @, **, .T, .H) and their
shapes and dtypes are scipy's; a subclass may define _matvec / _matmat; aslinearoperator wraps a
sparse array, a 2-D sd.ndarray or an operator. Also: sd.linalg refuses a sparse array with a
message that names the sd.sparse function to use (docs/design/sparse.md §3.10)."""

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import matrices as M
from .conftest import LIVE, coo

S = sd.sparse

D = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]])
DC = np.array([[1.0 + 1j, 2.0], [0.5j, 3.0 - 1j]])


def custom(d=D, rmat=True):
    a = sd.asarray(d)
    ah = sd.asarray(d.conj().T.copy())
    return S.LinearOperator(d.shape, lambda x: a @ x, rmatvec=(lambda y: ah @ y) if rmat else None)


def npy(x):
    return np.asarray(x)


def test_products_and_shapes():
    L = custom()
    x, y = np.array([1.0, -2.0, 0.5]), np.array([2.0, 1.0])
    assert L.shape == (2, 3) and L.dtype == sd.float64
    assert np.allclose(npy(L.matvec(sd.asarray(x))), D @ x)
    assert L.matvec(sd.asarray(x[:, None])).shape == (2, 1)
    assert np.allclose(npy(L.rmatvec(sd.asarray(y))), D.T @ y)
    assert np.allclose(npy(L.matmat(sd.asarray(np.eye(3)))), D)
    assert np.allclose(npy(L.rmatmat(sd.asarray(np.eye(2)))), D.T)
    assert np.allclose(npy(L @ sd.asarray(x)), D @ x)
    assert np.allclose(npy(L @ sd.asarray(np.eye(3))), D)
    assert np.allclose(npy(sd.asarray(y) @ L), y @ D)
    assert np.allclose(npy(sd.asarray(np.eye(2)) @ L), D)
    assert L.matmat(sd.zeros((3, 0))).shape == (2, 0)


def test_dtype_is_asked_of_matvec_once():
    calls = []

    def mv(x):
        calls.append(x.dtype)
        return sd.asarray(DC) @ x[:2]

    L = S.LinearOperator((2, 3), mv)
    assert L.dtype == sd.complex128 and calls == [sd.float64]
    assert S.LinearOperator((2, 3), mv, dtype="float64").dtype == sd.float64 and len(calls) == 1


def test_combinations():
    A, B = custom(), custom(2 * D)
    x = sd.asarray(np.array([1.0, 2.0, 3.0]))
    y = sd.asarray(np.array([1.0, -1.0]))
    assert np.allclose(npy((A + B) @ x), 3 * D @ npy(x))
    assert np.allclose(npy((A - B) @ x), -D @ npy(x))
    assert np.allclose(npy((-A) @ x), -D @ npy(x))
    assert np.allclose(npy((2.5 * A) @ x), 2.5 * D @ npy(x)) and np.allclose(npy((A * 2) @ x), 2 * D @ npy(x))
    assert np.allclose(npy(A.T @ y), D.T @ npy(y)) and A.T.shape == (3, 2)
    assert np.allclose(npy(A.H @ y), D.T @ npy(y)) and np.allclose(npy(A.adjoint() @ y), D.T @ npy(y))
    assert np.allclose(npy((A.T @ A) @ x), D.T @ D @ npy(x))
    assert np.allclose(npy(((A.T @ A) ** 3) @ x), np.linalg.matrix_power(D.T @ D, 3) @ npy(x))
    assert np.allclose(npy(((A.T @ A) ** 0) @ x), npy(x))
    assert ((A + B) * 1j).dtype == sd.complex128 and (A + B).dtype == sd.float64


def test_complex_transpose_and_adjoint():
    L = S.aslinearoperator(S.csr_array(sd.asarray(DC)))
    v = np.array([1.0 + 2j, -1j])
    assert np.allclose(npy(L.T @ sd.asarray(v)), DC.T @ v)
    assert np.allclose(npy(L.H @ sd.asarray(v)), DC.conj().T @ v)
    assert np.allclose(npy(L.T.T @ sd.asarray(v)), DC @ v)
    assert np.allclose(npy((1j * L).rmatvec(sd.asarray(v))), (1j * DC).conj().T @ v)


def test_subclass_form():
    class Twice(S.LinearOperator):
        def __init__(self, n):
            super().__init__((n, n), None, dtype=sd.float64)

        def _matvec(self, x):
            return 2 * x

    class Block(S.LinearOperator):  # only _matmat: matvec goes through it
        def __init__(self):
            super().__init__((3, 3), None)

        def _matmat(self, X):
            return 3 * X

    t = Twice(4)
    assert repr(t) == "Twice(shape=(4, 4), dtype=float64)"
    assert npy(t @ sd.ones(4)).tolist() == [2, 2, 2, 2]
    assert npy(t.matmat(sd.eye(4))).tolist() == (2 * np.eye(4)).tolist()
    b = Block()
    assert b.dtype == sd.float64 and npy(b.matvec(sd.ones(3))).tolist() == [3, 3, 3]


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_aslinearoperator_of_sparse_and_dense(fmt):
    m = M.get("random_complex/40")
    A = coo(m).asformat(fmt)
    L = S.aslinearoperator(A)
    assert L.shape == A.shape and L.dtype == sd.complex128 and L._matrix is A
    x = sd.asarray(M.rhs(m))
    assert bits_equal(L @ x, A @ x)
    assert np.allclose(npy(L.rmatvec(x)), m.dense().conj().T @ npy(x))
    Ld = S.aslinearoperator(sd.asarray(m.dense()))
    assert np.allclose(npy(Ld @ x), m.dense() @ npy(x)) and np.allclose(npy(Ld.H @ x), m.dense().conj().T @ npy(x))
    assert S.aslinearoperator(L) is L


@LIVE
def test_matches_scipy_on_the_combinations():
    import scipy.sparse.linalg as spla

    m = M.get("random_unsym/50x5")
    A = coo(m).tocsr()
    s = spla.aslinearoperator(m.scipy("csr"))
    L = S.aslinearoperator(A)
    x = M.rhs(m)
    for mine, ref in ((L, s), (L.T, s.T), (L.H, s.H), (2 * L - L, 2 * s - s), (L @ L, s @ s), (L**2, s**2)):
        assert mine.shape == ref.shape and str(mine.dtype) == ref.dtype.name
        assert np.allclose(npy(mine @ sd.asarray(x)), ref @ x, rtol=1e-13, atol=1e-13)


def test_errors():
    L = custom(rmat=False)
    cases = [
        (lambda: S.LinearOperator((2, 3)), TypeError,
         "LinearOperator(): missing 1 required positional argument: 'matvec'"),
        (lambda: S.LinearOperator((2, 3), None), TypeError,
         "LinearOperator(): 'matvec' must be callable, not NoneType"),
        (lambda: S.LinearOperator((2, 3), 3), TypeError,
         "LinearOperator(): 'matvec' must be callable or None, not int"),
        (lambda: S.LinearOperator((2, 3), lambda x: x, 2), TypeError, None),  # rmatvec is a keyword (SP65)
        (lambda: S.LinearOperator(3, lambda x: x), ValueError,
         "LinearOperator(): 'shape' must have 2 dimensions, got 1"),
        (lambda: S.LinearOperator((2, 3), lambda x: x, dtype="float32"), TypeError, None),
        (lambda: L.matvec(sd.ones(4)), ValueError, "matvec(): 'x' must have shape (3,) or (3, 1), got (4,)"),
        (lambda: L.matvec([1, 2, 3]), TypeError, "matvec(): 'x' must be an sd.ndarray, not list"),
        (lambda: L.matmat(sd.ones(3)), ValueError, "matmat(): 'X' must have 2 dimensions, got 1"),
        (lambda: L.matmat(sd.ones((2, 2))), ValueError, "matmat(): the operator has 3 columns but 'X' has 2 rows"),
        (lambda: L.rmatvec(sd.ones(2)), ValueError, "rmatvec(): the operator was made without 'rmatvec'"),
        (lambda: S.LinearOperator((2, 3), lambda x: sd.ones(5)).matvec(sd.ones(3)), ValueError,
         "matvec(): the operator's function returned shape (5,), expected (2,)"),
        (lambda: L @ [1, 2, 3], TypeError, None),
        (lambda: L * L, TypeError, None),
        (lambda: L + 1, TypeError, None),
        (lambda: L**2, ValueError, "a power needs a square operator, got shape (2, 3)"),
        (lambda: L + custom(np.ones((3, 3))), ValueError, "cannot add operators of shapes (2, 3) and (3, 3)"),
        (lambda: L @ L, ValueError, "cannot multiply operators of shapes (2, 3) and (2, 3)"),
        (lambda: S.aslinearoperator([1]), TypeError,
         "aslinearoperator(): 'A' must be a sparse array, an sd.ndarray or a LinearOperator, not list"),
        (lambda: S.aslinearoperator(sd.ones(3)), ValueError, "aslinearoperator(): 'A' must have 2 dimensions, got 1"),
    ]  # fmt: skip
    for call, kind, msg in cases:
        with pytest.raises(kind) as e:
            call()
        if msg is not None:
            assert str(e.value) == msg


# ---------------------------------------------------------------- sd.linalg refuses sparse arrays


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda A: sd.linalg.solve(A, sd.ones(3)),
         "solve(): 'a' must be an sd.ndarray, not csr_array; "
         "for a sparse array use sd.sparse.spsolve"),
        (lambda A: sd.linalg.solve_triangular(A, sd.ones(3), upper=False),
         "solve_triangular(): 'a' must be an sd.ndarray, not csr_array; "
         "for a sparse array use sd.sparse.spsolve_triangular"),
        (lambda A: sd.linalg.lu(A),
         "lu(): 'x' must be an sd.ndarray, not csr_array; "
         "for a sparse array use sd.sparse.splu"),
        (lambda A: sd.linalg.inv(A),
         "inv(): 'x' must be an sd.ndarray, not csr_array; "
         "for a sparse array use sd.sparse.inv"),
        (lambda A: sd.linalg.kron(A, A),
         "kron(): 'x1' must be an sd.ndarray, not csr_array; "
         "for a sparse array use sd.sparse.kron"),
        (lambda A: sd.linalg.det(A),
         "det(): 'x' must be an sd.ndarray, not csr_array; "
         "a sparse array has no det: use det(A.toarray())"),
        # (SP6's eigensolvers and SP7's norm and expm exist: the hints name them)
        (lambda A: sd.linalg.eigh(A),
         "eigh(): 'a' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.eigsh"),
        (lambda A: sd.linalg.svdvals(A),
         "svdvals(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.svds"),
        (lambda A: sd.linalg.eigvals(A),
         "eigvals(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.eigs"),
        (lambda A: sd.linalg.matrix_norm(A),
         "matrix_norm(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.norm"),
        (lambda A: sd.linalg.vector_norm(A),
         "vector_norm(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.norm"),
        (lambda A: sd.linalg.expm(A),
         "expm(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.expm"),
        (lambda A: sd.linalg.lstsq(A, sd.ones(3)),  # (SP5's solvers exist: the hint names them)
         "lstsq(): 'a' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.lsqr or sd.sparse.lsmr"),
    ],
)  # fmt: skip
def test_linalg_names_the_sparse_function(call, msg):
    A = S.csr_array(sd.eye(3))
    with pytest.raises(TypeError) as e:
        call(A)
    assert str(e.value) == msg


def test_linalg_names_every_argument_and_other_inputs_are_unchanged():
    A = S.csc_array(sd.eye(3))
    with pytest.raises(
        TypeError,
        match=r"^solve\(\): 'b' must be an sd.ndarray, not csc_array; for a sparse array use sd.sparse.spsolve$",
    ):
        sd.linalg.solve(sd.eye(3), A)
    with pytest.raises(TypeError, match=r"^solve\(\): 'a' must be an sd.ndarray, not list$"):
        sd.linalg.solve([[1.0]], sd.ones(1))


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda A: sd.sum(A), "sum(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use A.sum()"),
        (lambda A: sd.matmul(A, A),
         "matmul(): 'x1' must be an sd.ndarray, not csr_array; for a sparse array use A @ B"),
        (lambda A: sd.matmul(sd.eye(3), A),
         "matmul(): 'x2' must be an sd.ndarray, not csr_array; for a sparse array use A @ B"),
        (lambda A: sd.max(A), "max(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use A.max()"),
        (lambda A: sd.tril(A),
         "tril(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use sd.sparse.tril"),
        (lambda A: sd.sort(A),
         "sort(): 'x' must be an sd.ndarray, not csr_array; a sparse array has no sort: use sort(A.toarray())"),
        (lambda A: sd.sin(A), "sin(): 'x' must be an sd.ndarray or a bool, int, float or complex, not csr_array; "
                              "for a sparse array use A.sin()"),
    ],
)  # fmt: skip
def test_array_functions_name_the_sparse_counterpart(call, msg):
    # SP67: the array's one check (bind/pyutil's require_ndarray), on the error path only
    A = S.csr_array(sd.eye(3))
    with pytest.raises(TypeError) as e:
        call(A)
    assert str(e.value) == msg
    with pytest.raises(TypeError, match=r"^sum\(\): 'x' must be an sd.ndarray, not list$"):
        sd.sum([1.0])


# ---- numpy operands, the results of the user's function, 3-D operands ----


def test_numpy_arrays_are_not_operands():
    """numpy defers (__array_ufunc__ = None, SPR-4), and Python's TypeError follows (not an
    object array of operators)."""
    L = S.aslinearoperator(sd.eye(3))
    x = np.ones(3)
    cases = [
        (lambda: x * L, r"^unsupported operand type\(s\) for \*: 'numpy.ndarray' and 'LinearOperator'$"),
        (lambda: L * x, r"^unsupported operand type\(s\) for \*: 'LinearOperator' and 'numpy.ndarray'$"),
        (lambda: L @ x, r"^unsupported operand type\(s\) for @: 'LinearOperator' and 'numpy.ndarray'$"),
        (lambda: x @ L, r"^unsupported operand type\(s\) for @: 'numpy.ndarray' and 'LinearOperator'$"),
        (lambda: L + np.ones((3, 3)), r"^unsupported operand type\(s\) for \+: 'LinearOperator' and 'numpy.ndarray'$"),
        (lambda: np.ones((3, 3)) - L, r"^unsupported operand type\(s\) for -: 'numpy.ndarray' and 'LinearOperator'$"),
    ]
    for f, msg in cases:
        with pytest.raises(TypeError, match=msg):
            f()
    assert npy((np.float64(2.0) * L) @ sd.ones(3)).tolist() == [2.0, 2.0, 2.0]  # numpy scalars still scale
    assert npy((L * np.int64(3)) @ sd.ones(3)).tolist() == [3.0, 3.0, 3.0]
    with pytest.raises(TypeError, match=r"^unsupported operand type\(s\) for \+: 'LinearOperator' and 'int'$"):
        L + 1


def test_a_function_result_that_is_not_an_array():
    """matvec's conversion of the user's result names the method and says what it must return
    (SPR-12; not asarray's 'obj')."""
    L = S.LinearOperator((2, 3), lambda v: None, dtype=sd.float64)
    with pytest.raises(TypeError, match=r"^matvec\(\): the operator's function must return an sd.ndarray or a "
                                        r"sequence of numbers, not NoneType$"):  # fmt: skip
        L @ sd.ones(3)
    L = S.LinearOperator((2, 3), lambda v: v[:2], rmatvec=lambda v: "ab", dtype=sd.float64)
    with pytest.raises(TypeError, match=r"^rmatvec\(\): the operator's function must return .* not str$"):
        L.rmatvec(sd.ones(2))


def test_operands_of_three_dimensions():
    """L @ X and X @ L of a 3-D X: an operator's message, no function name (R225)."""
    L = S.LinearOperator((5, 5), lambda v: v)
    for f in (lambda: L @ sd.ones((5, 5, 5)), lambda: sd.ones((5, 5, 5)) @ L):
        with pytest.raises(ValueError, match=r"^the operand must have 1 or 2 dimensions, got 3$"):
            f()
