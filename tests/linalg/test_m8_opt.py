"""M8: the optimizations keep the results.

- sums are pairwise (user decision, DESIGN.md §2.17): the bits of numpy's and
  stardust-numpy's sum of the same terms, whatever the input layout;
- workspace sizes are asked once per batch, and every LAPACK array is 16-byte
  aligned: a batch gives each matrix's own result, bit for bit;
- read-only triangles are used in place when column-major: the same bits as packed.
(experiments/m8_bits compares every function's output with the M7 build.)
"""

import math
import os

import numpy as np
import pytest
import stardust as sd
from .conftest import DTYPES, LAYOUTS, npy, random_values, as_sd, well_conditioned, with_layout
from hypothesis import given, settings
from hypothesis import strategies as st


SETTINGS = settings(max_examples=int(os.environ.get("SDLA_EXAMPLES", 150)), deadline=None)
FLOATS = ["float64", "complex128"]


def bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


# ------------------------------------------------------------ pairwise sums --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    rows=st.integers(1, 3),
    n=st.integers(0, 1100),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(["C", "F", "strided", "reversed"]),
)
def test_sums_are_numpys_pairwise(seed, rows, n, dtype, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (rows, n), dtype), layout)
    y = with_layout(random_values(rng, (rows, n), dtype), layout)
    xc, yc = np.ascontiguousarray(x), np.ascontiguousarray(y)
    calc = np.complex128 if dtype == "complex128" else np.float64
    xf, yf = xc.astype(calc), yc.astype(calc)
    # vecdot: sum(conj(x1) * x2) along the axis, numpy's and sd's bits
    v = npy(sd.linalg.vecdot(as_sd(x), as_sd(y)))
    want = np.sum(np.conj(xf) * yf, axis=-1)
    assert bits_equal(v, want)
    assert bits_equal(v, npy(sd.sum(sd.conj(as_sd(xf)) * as_sd(yf), axis=-1)))
    # vector_norm ord=1 and matrix_norm ord=inf: sums of |x| along the rows (real
    # values: a complex modulus can differ from np.abs in the last bit, which is not
    # about the sum)
    if dtype != "complex128":
        a = np.abs(xf)
        assert bits_equal(npy(sd.linalg.vector_norm(as_sd(x), ord=1, axis=-1)), np.sum(a, axis=-1))
        if n:
            assert sd.linalg.matrix_norm(as_sd(x), ord=math.inf) == np.max(np.sum(a, axis=-1))
    # 2-norm and fro of real values (no rescaling at these magnitudes): sqrt(sum(x*x)) over
    # the flattened C-order sequence
    if dtype != "complex128":
        assert sd.linalg.vector_norm(as_sd(x)) == np.sqrt(np.sum(xf * xf))
        assert sd.linalg.matrix_norm(as_sd(x)) == np.sqrt(np.sum(xf * xf))
    else:  # the parts are separate terms: re0, im0, re1, im1, ...
        parts = xf.view(np.float64)
        assert sd.linalg.vector_norm(as_sd(x)) == np.sqrt(np.sum(parts * parts))


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(0, 600),
    k=st.integers(-3, 3),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_trace_is_sum_of_diagonal(seed, n, k, dtype, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (2, n, n), dtype), layout)
    t = npy(sd.linalg.trace(as_sd(x), k=k))
    d = npy(sd.linalg.diagonal(as_sd(x), k=k))
    if dtype in ("bool", "int64"):
        d = d.astype(np.float64)
    assert bits_equal(t, np.sum(d, axis=-1))
    assert bits_equal(t, npy(sd.sum(as_sd(d), axis=-1)))


def test_pairwise_is_more_accurate():
    # 1 + many tiny terms: a sequential sum loses them one by one, the tree keeps them
    x = np.full(1 << 20, 1e-16)
    x[0] = 1.0
    exact = 1.0 + (len(x) - 1) * 1e-16
    got = sd.linalg.vecdot(as_sd(x), as_sd(np.ones_like(x)))
    assert abs(got - exact) < 1e-14  # a sequential sum would be off by 1e-10 (it stays at 1.0)
    seq = 0.0
    for v in x[:4096]:  # the sequential sum already lost every term after the first
        seq += v
    assert seq == 1.0


def test_norm_layouts_agree():
    rng = np.random.default_rng(5)
    for dtype in FLOATS:
        x = random_values(rng, (37, 29), dtype)
        ref = [sd.linalg.matrix_norm(as_sd(x), ord=o) for o in ("fro", 1, math.inf, -1, -math.inf)]
        for layout in ("F", "strided", "reversed"):
            y = with_layout(x, layout)
            assert [sd.linalg.matrix_norm(as_sd(y), ord=o) for o in ("fro", 1, math.inf, -1, -math.inf)] == ref
        # cond's norm(x) * norm(inv(x)) uses the same sums as matrix_norm
        sq = x[:29, :29] + 40 * np.eye(29)
        for o in ("fro", 1, math.inf):
            inv = npy(sd.linalg.inv(as_sd(sq)))
            assert sd.linalg.cond(as_sd(sq), ord=o) == sd.linalg.matrix_norm(as_sd(sq), ord=o) * sd.linalg.matrix_norm(as_sd(inv), ord=o)


# ------------------------------------------------- workspace once per batch --


@pytest.mark.parametrize(
    "call",
    [
        lambda a: sd.linalg.qr(a),
        lambda a: sd.linalg.qr_pivoted(a),
        lambda a: sd.linalg.lstsq(a, a),
        lambda a: sd.linalg.svd(a),
        lambda a: sd.linalg.svd(a, mode="complete"),
        lambda a: sd.linalg.svdvals(a),
        lambda a: sd.linalg.pinv(a),
        lambda a: sd.linalg.eig(a),
        lambda a: sd.linalg.eigvals(a),
        lambda a: sd.linalg.schur(a),
        lambda a: sd.linalg.hessenberg(a),
        lambda a: sd.linalg.eigh(a),
        lambda a: sd.linalg.ldl(a),
        lambda a: sd.linalg.solve_sylvester(a, a, a),
        lambda a: sd.linalg.solve(a, a),
        lambda a: sd.linalg.inv(a),
        lambda a: sd.linalg.cholesky(a),
        lambda a: sd.linalg.cholesky(a, upper=True),
        lambda a: sd.linalg.eigh(a, a),
        lambda a: sd.linalg.eigvalsh(a),
        lambda a: sd.linalg.lu(a),
        lambda a: sd.linalg.matrix_power(a, 3),
    ],
)
@pytest.mark.parametrize("dtype", FLOATS)
def test_batch_equals_one_by_one(call, dtype):
    rng = np.random.default_rng(7)
    # odd n: a float64 batch element's slot sits at an odd multiple of 8 bytes,
    # where Accelerate rounded differently before M8 (DESIGN.md §2.17)
    for n in (3, 5, 7, 40):
        x = random_values(rng, (5, n, n), dtype)
        x = x + np.conj(np.swapaxes(x, -1, -2)) + 4 * n * np.eye(n)  # Hermitian, safely nonsingular
        whole = call(as_sd(x))
        whole = whole if isinstance(whole, tuple) else (whole,)
        for i in range(5):
            one = call(as_sd(x[i]))
            one = one if isinstance(one, tuple) else (one,)
            for w, o in zip(whole, one):
                assert bits_equal(np.asarray(w)[i], o)


# ------------------------------------------------------- in-place operands --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(0, 40),
    dtype=st.sampled_from(DTYPES),
    upper=st.booleans(),
)
def test_triangle_in_place_or_packed(seed, n, dtype, upper):
    rng = np.random.default_rng(seed)
    a = well_conditioned(rng, (2,), n, dtype)
    b = random_values(rng, (2, n, 2), "float64")
    f = with_layout(a, "F")  # read in place when float64/complex128
    results = []
    for layout in ("F", "C", "strided"):
        y = with_layout(a, layout)
        results.append(npy(sd.linalg.solve_triangular(as_sd(y), as_sd(b), upper=upper)))
    assert all(bits_equal(r, results[0]) for r in results)
    # lu_solve on lu's own column-major factors equals solve (two trtrs == getrs)
    L, U, P = sd.linalg.lu(as_sd(f))
    assert bits_equal(npy(sd.linalg.lu_solve(L, U, P, as_sd(b))), npy(sd.linalg.solve(as_sd(f), as_sd(b))))
    Lc, Uc = (as_sd(np.ascontiguousarray(npy(t))) for t in (L, U))
    assert bits_equal(npy(sd.linalg.lu_solve(Lc, Uc, P, as_sd(b))), npy(sd.linalg.solve(as_sd(f), as_sd(b))))


def test_in_place_operand_checks():
    # NaN in the triangle read in place: still found; NaN outside: still ignored
    a = np.asfortranarray(np.triu(np.ones((4, 4))) + 3 * np.eye(4))
    a[3, 0] = np.nan  # below the diagonal: not read for upper=True
    sd.linalg.solve_triangular(as_sd(a), as_sd(np.ones(4)), upper=True)
    a[0, 3] = np.nan
    with pytest.raises(ValueError, match="'a' contains NaN"):
        sd.linalg.solve_triangular(as_sd(a), as_sd(np.ones(4)), upper=True)
    z = np.asfortranarray(np.triu(np.ones((4, 4))))
    z[2, 2] = 0
    with pytest.raises(sd.linalg.LinAlgError, match="singular"):
        sd.linalg.solve_triangular(as_sd(z), as_sd(np.ones(4)), upper=True)
