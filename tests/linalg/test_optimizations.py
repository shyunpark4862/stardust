"""The optimizations keep the results.

- sums (docs/design/linalg.md §8.2): float64 and complex128 data read in place go through
  the fast sums (csrc/linalg/fastsum.hpp: multi-accumulator chunks of a fixed size, a
  pairwise tree over the chunks), any other layout or dtype through buffered pairwise
  leaves: every path is within a plain multi-accumulator sum's error bound of numpy's and
  of sd's sum, whatever the input layout (the last bits may depend on the layout);
- workspace sizes are asked once per batch, and every LAPACK array is 16-byte
  aligned: a batch gives each matrix's own result, bit for bit;
- read-only triangles are used in place when column-major (the same bits as packed) or
  row-major (equal within rounding).
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd
from tests._bits import bits_equal

from .._tier import examples
from .conftest import DTYPES, LAYOUTS, as_sd, npy, random_values, well_conditioned, with_layout

SETTINGS = settings(max_examples=examples(150), deadline=None)
FLOATS = ["float64", "complex128"]


def within_pairwise_bound(got, want, terms_abs, n):
    """|got - want| within two pairwise sums' error bounds of each other: 2 (log2 n + 2) eps
    times the sum of the terms' magnitudes (numpy-bit identity is no promise, D-33)."""
    got, want, terms_abs = np.asarray(got), np.asarray(want), np.asarray(terms_abs)
    bound = 2 * (math.log2(max(n, 1)) + 2) * 2.0**-52 * terms_abs
    return got.shape == want.shape and bool(np.all(np.abs(got - want) <= bound))


# A lane of a fast sum adds at most 64 terms in order (fastsum.hpp), the chunks pairwise.
FAST_LANE = 64


def within_sum_bound(got, want, terms_abs, n):
    """|got - want| within two error bounds of the fast sums (or a pairwise sum and a fast
    sum) of each other: 2 (64 + log2 n + 2) eps times the sum of the terms' magnitudes."""
    got, want, terms_abs = np.asarray(got), np.asarray(want), np.asarray(terms_abs)
    bound = 2 * (FAST_LANE + math.log2(max(n, 1)) + 2) * 2.0**-52 * terms_abs
    return got.shape == want.shape and bool(np.all(np.abs(got - want) <= bound))


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
    # vecdot: sum(conj(x1) * x2) along the axis: within the bound of numpy's sum, and of
    # the contiguous layout's
    v = npy(sd.linalg.vecdot(as_sd(x), as_sd(y)))
    want = np.sum(np.conj(xf) * yf, axis=-1)
    mags = np.sum(np.abs(xf) * np.abs(yf), axis=-1)
    assert within_sum_bound(v, want, mags, n)
    assert within_sum_bound(v, npy(sd.linalg.vecdot(as_sd(xc), as_sd(yc))), mags, n)
    # vector_norm ord=1 and matrix_norm ord=inf: sums of |x| along the rows, within the
    # bound of sd.sum's and numpy's (real values: a complex modulus can differ from np.abs
    # in the last bit, which is not about the sum)
    if dtype != "complex128":
        a = np.abs(xf)
        r1 = npy(sd.linalg.vector_norm(as_sd(x), ord=1, axis=-1))
        assert within_sum_bound(r1, npy(sd.sum(as_sd(a), axis=-1)), np.sum(a, axis=-1), n)
        assert within_sum_bound(r1, np.sum(a, axis=-1), np.sum(a, axis=-1), n)
        if n:
            assert within_sum_bound(sd.linalg.matrix_norm(as_sd(x), ord=math.inf), np.max(r1), np.max(r1), n)
    # 2-norm and fro (no rescaling at these magnitudes): the squares of the elements
    # (complex: the parts re0, im0, re1, ... as separate terms), fused
    parts = xf.view(np.float64) if dtype == "complex128" else xf
    sq = np.sum(parts * parts)
    for got in (sd.linalg.vector_norm(as_sd(x)), sd.linalg.matrix_norm(as_sd(x))):
        assert within_sum_bound(got * got, sq, sq, parts.size) or abs(got - math.sqrt(sq)) <= 4e-16 * got
    for f in (sd.linalg.vector_norm, sd.linalg.matrix_norm):
        g, c = f(as_sd(x)), f(as_sd(xc))
        assert within_sum_bound(g * g, c * c, sq, parts.size) or abs(g - c) <= 4e-16 * c


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
    # the bits of sd.sum over the diagonal; numpy's sum agrees within rounding (DESIGN.md §1.6)
    assert bits_equal(t, npy(sd.sum(as_sd(d), axis=-1)))
    assert np.allclose(t, np.sum(d, axis=-1), rtol=1e-13, atol=1e-13 * np.abs(d).sum())


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
            got = [sd.linalg.matrix_norm(as_sd(y), ord=o) for o in ("fro", 1, math.inf, -1, -math.inf)]
            assert got == pytest.approx(ref, rel=1e-14)  # the sums' order may depend on the layout
        # cond's norm(x) * norm(inv(x)) uses the same sums as matrix_norm, within rounding
        sq = x[:29, :29] + 40 * np.eye(29)
        for o in ("fro", 1, math.inf):
            inv = npy(sd.linalg.inv(as_sd(sq)))
            assert sd.linalg.cond(as_sd(sq), ord=o) == pytest.approx(
                sd.linalg.matrix_norm(as_sd(sq), ord=o) * sd.linalg.matrix_norm(as_sd(inv), ord=o), rel=1e-14
            )


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
    # where Accelerate rounds differently unless arrays are 16-byte aligned
    # (docs/design/linalg.md §6.4)
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
    results = {}
    for layout in ("F", "C", "strided"):
        y = with_layout(a, layout)
        results[layout] = npy(sd.linalg.solve_triangular(as_sd(y), as_sd(b), upper=upper))
    # F in place and strided packed: the same column-major solve, the same bits; C read in
    # place as row-major: equal within rounding (docs/design/linalg.md §5.5)
    assert bits_equal(results["strided"], results["F"])

    def close(got, want):
        return np.allclose(got, want, rtol=1e-12, atol=1e-12 * max(1.0, np.abs(want).max(initial=0)))

    assert close(results["C"], results["F"])
    # lu_solve on lu's own column-major factors equals solve (the same two triangular solves)
    L, U, P = sd.linalg.lu(as_sd(f))
    assert bits_equal(npy(sd.linalg.lu_solve(L, U, P, as_sd(b))), npy(sd.linalg.solve(as_sd(f), as_sd(b))))
    Lc, Uc = (as_sd(np.ascontiguousarray(npy(t))) for t in (L, U))
    assert close(npy(sd.linalg.lu_solve(Lc, Uc, P, as_sd(b))), npy(sd.linalg.solve(as_sd(f), as_sd(b))))


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


@pytest.mark.parametrize("order", ["C", "F"])
def test_matrix_norm_sums_each_line_pairwise(order):
    # Each column (1, -1) or row (inf, -inf) is summed on its own: within rounding of sd.sum
    # over that line as a contiguous vector, whatever the memory layout (a contiguous line is
    # one fast sum, lines side by side are summed in strips). 'fro' fuses the squares into
    # the sum (fma), so it agrees with sqrt(sum(x * x)) within rounding.
    rng = np.random.default_rng(7)
    m = rng.standard_normal((300, 500))
    x = sd.asarray(np.asarray(m, order=order))

    def line_sum(v):
        return float(sd.sum(sd.asarray(np.abs(np.ascontiguousarray(v)))))

    cols = np.array([line_sum(m[:, j]) for j in range(m.shape[1])])
    rows = np.array([line_sum(m[i, :]) for i in range(m.shape[0])])
    assert float(sd.linalg.matrix_norm(x, ord=1)) == pytest.approx(cols.max(), rel=1e-14)
    assert float(sd.linalg.matrix_norm(x, ord=-1)) == pytest.approx(cols.min(), rel=1e-14)
    assert float(sd.linalg.matrix_norm(x, ord=float("inf"))) == pytest.approx(rows.max(), rel=1e-14)
    assert float(sd.linalg.matrix_norm(x, ord=float("-inf"))) == pytest.approx(rows.min(), rel=1e-14)
    fro = float(sd.linalg.matrix_norm(x, ord="fro"))
    assert fro == pytest.approx(float(np.sqrt((m * m).sum())), rel=1e-14)
    assert fro == pytest.approx(float(sd.linalg.matrix_norm(sd.asarray(np.ascontiguousarray(m)), ord="fro")), rel=1e-14)
