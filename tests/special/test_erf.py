"""erf erfc erfcx erfinv erfcinv ndtr log_ndtr ndtri (docs/api/special.md §2).

Accuracy is tests/special/test_reference.py's. Here: the special values, bit for bit
against scipy except where docs/design/special.md §2.4 says otherwise, and the layouts of
the chunk kernels."""

import math

import numpy as np
import pytest
import scipy.special as sp

import stardust as sd
from tests._bits import bits_equal, bits_equal_any_nan

S = sd.special

NAMES = ["erf", "erfc", "erfcx", "erfinv", "erfcinv", "ndtr", "log_ndtr", "ndtri"]
EDGES = [
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.0,
    -1.0,
    2.0,
    -2.0,
    8.0,
    -8.0,
    np.inf,
    -np.inf,
    np.nan,
    5e-324,
    -5e-324,
    1e-300,
    -1e-300,
    1e200,
    -1e200,
    1 - 2**-53,
    -1 + 2**-53,
    2 - 2**-52,
    0.075,
    0.925,
]

# Where sd.special differs from scipy on purpose (docs/design/special.md §2.4): name -> {input: value}
DIFFERENT = {
    "erfinv": {-0.0: -0.0},  # odd function: the sign of zero is kept (scipy: +0.0)
    "erfcinv": {5e-324: None},  # finite (scipy: inf); the value is test_reference's
}


@pytest.mark.parametrize("name", NAMES)
def test_special_values(name):
    f, g = getattr(S, name), getattr(sp, name)
    for x in EDGES:
        got, want = f(x), g(x)
        special = [
            v for k, v in DIFFERENT.get(name, {}).items() if bits_equal_any_nan(k, x)
        ]  # matched by bits: -0.0 is not 0.0
        if special:
            if special[0] is not None:
                assert bits_equal_any_nan(got, special[0]), (name, x, got)
            continue
        if np.isfinite(want) and want != 0 and np.isfinite(got) and got != 0:
            continue  # an ordinary value: within ULPs, test_reference's business
        assert bits_equal_any_nan(got, want), (name, x, got, want)


def test_documented_values():
    assert S.erf(-0.0) == 0.0 and np.signbit(S.erf(-0.0))
    assert S.erfc(np.inf) == 0.0 and S.erfc(-np.inf) == 2.0 and S.erfcx(-np.inf) == np.inf
    assert S.erfcx(np.inf) == 0.0 and S.erfcx(-27.0) == np.inf
    assert S.ndtr(np.inf) == 1.0 and S.ndtr(-np.inf) == 0.0
    assert S.log_ndtr(np.inf) == 0.0 and np.signbit(S.log_ndtr(np.inf)) and S.log_ndtr(-np.inf) == -np.inf
    assert S.ndtri(0.0) == -np.inf and S.ndtri(1.0) == np.inf and S.ndtri(0.5) == 0.0
    assert S.erfinv(1.0) == np.inf and S.erfinv(-1.0) == -np.inf and np.signbit(S.erfinv(-0.0))
    assert S.erfcinv(0.0) == np.inf and S.erfcinv(2.0) == -np.inf and np.signbit(S.erfcinv(1.0))
    for name, bad in [
        ("ndtri", [-0.5, 1.5, np.inf]),
        ("erfinv", [1.5, -1.5, np.inf]),
        ("erfcinv", [-0.5, 2.5, np.inf]),
    ]:
        for x in bad:
            assert np.isnan(getattr(S, name)(x)), (name, x)


def test_subnormal_tails_differ_from_scipy():
    # docs/design/special.md §2.4: values where scipy flushes to 0 (or -0.0)
    for name, x in [("erfc", 26.8), ("erfc", 27.2), ("ndtr", -38.0), ("ndtr", -38.4), ("log_ndtr", 38.0)]:
        got = getattr(S, name)(x)
        assert got != 0 and abs(got) < 2.3e-308 and getattr(sp, name)(x) == 0, (name, x)
    assert S.erfc(27.3) == 0.0 and S.ndtr(-38.5) == 0.0  # below the smallest subnormal


def test_erfcinv_smallest_subnormal_is_finite():
    assert S.erfcinv(5e-324) == pytest.approx(27.2132932108129, rel=1e-14)
    assert sp.erfcinv(5e-324) == np.inf


def test_erfinv_subnormal_inputs():
    assert S.erfinv(5e-324) == 5e-324 and S.erfinv(-5e-324) == -5e-324


CHUNKED = ["erf", "erfc", "ndtr", "log_ndtr", "ndtri", "erfinv", "erfcinv"]


def sample(name, rng, n):
    if name == "ndtri":
        return rng.uniform(0, 1, n)
    if name == "erfinv":
        return rng.uniform(-1, 1, n)
    if name == "erfcinv":
        return rng.uniform(0, 2, n)
    return rng.normal(scale=12, size=n)


def fn(name):
    return getattr(S, name)


@pytest.mark.parametrize("name", CHUNKED)
@pytest.mark.parametrize("n", [1, 2, 3, 1023, 1025, 2049])
def test_chunk_kernels_every_layout_same_bits(name, n):
    rng = np.random.default_rng(n)
    x = sample(name, rng, 2 * n)
    f = fn(name)
    contiguous = np.asarray(f(sd.asarray(x[:n].copy())))
    strided = np.asarray(f(sd.asarray(x)[::2]))
    out = sd.zeros(2 * n)
    f(sd.asarray(x[:n].copy()), out=out[::2])
    scalars = np.array([f(float(v)) for v in x[:n]])
    np.testing.assert_array_equal(contiguous.view(np.uint64), scalars.view(np.uint64))
    np.testing.assert_array_equal(strided.view(np.uint64), np.array([f(float(v)) for v in x[::2]]).view(np.uint64))
    np.testing.assert_array_equal(np.asarray(out)[::2].view(np.uint64), scalars.view(np.uint64))


@pytest.mark.parametrize("name", CHUNKED)
def test_chunk_kernels_in_place(name):
    x = sample(name, np.random.default_rng(1), 3000)
    a = sd.asarray(x.copy())
    want = np.asarray(fn(name)(a))
    fn(name)(a, out=a)
    np.testing.assert_array_equal(np.asarray(a).view(np.uint64), want.view(np.uint64))


def test_same_bits_for_any_thread_count():
    before = sd.get_num_threads()
    rng = np.random.default_rng(5)
    x = sd.asarray(rng.normal(scale=10, size=300_000))
    p = sd.asarray(rng.uniform(0, 1, 300_000))
    try:
        runs = []
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            runs.append(
                [
                    np.asarray(r).view(np.uint64)
                    for r in (
                        S.erf(x),
                        S.erfc(x),
                        S.erfcx(x),
                        S.ndtr(x),
                        S.log_ndtr(x),
                        S.ndtri(p),
                        S.erfinv(p),
                        S.erfcinv(p),
                    )
                ]
            )
    finally:
        sd.set_num_threads(before)
    for r in runs[1:]:
        for a, b in zip(r, runs[0]):
            np.testing.assert_array_equal(a, b)


def test_messages_name_the_documented_arguments():
    for name, arg in [("ndtri", "p"), ("erfinv", "y"), ("erfcinv", "z"), ("erf", "x")]:
        with pytest.raises(TypeError, match=rf"^{name}\(\): '{arg}' must be"):
            getattr(S, name)("a")
    # erf takes complex (docs/api/special.md §12); erfinv does not
    with pytest.raises(TypeError, match=r"^erfinv\(\): not supported for complex128$"):
        S.erfinv(1j)


INF, NAN = math.inf, math.nan


# ---- erf's NaN ----


def test_erf_nan_is_the_input():
    for f in (S.erf, S.erfc):
        assert bits_equal(f(NAN), NAN)
        for n in (1, 2, 3, 1025):
            assert bits_equal(f(sd.asarray([NAN] * n)), np.full(n, NAN))
