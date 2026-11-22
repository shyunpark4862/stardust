"""expit logit log_expit xlogy xlog1py entr rel_entr (docs/api/special.md §5).

Accuracy is tests/special/test_reference.py's. Here: the special values, bit for bit
against scipy except where docs/design/special.md §2.4 says otherwise, and the layouts of
the chunked NEON + vForce kernels (expit, logit)."""

import itertools

import numpy as np
import pytest
import scipy.special as sp

import stardust as sd
from tests._bits import bits_equal_any_nan

S = sd.special

EDGES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    0.25,
    0.75,
    2.0,
    -2.0,
    1e-300,
    -1e-300,
    5e-324,
    -5e-324,
    np.inf,
    -np.inf,
    np.nan,
    709.78,
    -709.78,
    -745.2,
    36.7,
    -36.7,
    1 - 2**-53,
    2**-53,
    1e300,
    -1e300,
]
UNARY = ["expit", "logit", "log_expit", "entr"]
BINARY = ["xlogy", "xlog1py", "rel_entr"]


@pytest.mark.parametrize("name", UNARY)
def test_special_values_unary(name):
    for x in EDGES:
        if name == "logit" and x in (0.25, 0.75):
            continue  # not a special value: 1 ULP apart (2 atanh(+-0.5) vs scipy's log), see test_reference
        got, want = getattr(S, name)(x), getattr(sp, name)(x)
        if name == "log_expit" and np.isfinite(want) and want != 0.0:
            # vForce's exp and log1p (the chunked kernel): within 2 ULP of scipy's libm values
            # (log_expit(1) is 1 ULP below scipy's, and the correctly rounded one)
            assert abs(got - want) <= 2 * np.spacing(abs(want)), (name, x, got, want)
            continue
        assert bits_equal_any_nan(got, want), (name, x)


@pytest.mark.parametrize("name", BINARY)
def test_special_values_binary(name):
    for x, y in itertools.product(EDGES, EDGES):
        assert bits_equal_any_nan(getattr(S, name)(x, y), getattr(sp, name)(x, y)), (name, x, y)


def test_documented_conventions():
    assert S.xlogy(0, -1.0) == 0.0 and S.xlogy(0, np.inf) == 0.0 and np.isnan(S.xlogy(0, np.nan))
    assert S.xlog1py(0, -2.0) == 0.0 and S.xlog1py(1, -1.0) == -np.inf
    assert S.entr(0) == 0.0 and S.entr(-1.0) == -np.inf
    assert S.rel_entr(0, 0) == 0.0 and S.rel_entr(0, -1.0) == np.inf and S.rel_entr(-1.0, 1.0) == np.inf
    assert S.logit(0) == -np.inf and S.logit(1) == np.inf and np.isnan(S.logit(1.5))


def test_expit_subnormal_tail_differs_from_scipy():
    # docs/design/special.md §2.4: scipy returns 0 from about -709.78 down; we return e^x / (1 + e^x)
    x = np.array([-709.8, -710.0, -720.0, -740.0, -745.0])
    got = np.asarray(S.expit(sd.asarray(x)))
    np.testing.assert_array_equal(got, np.exp(x))  # e^x / (1 + e^x) == e^x there
    assert (got > 0).all() and (sp.expit(x) == 0).all()
    assert S.expit(-745.2) == 0.0  # below the smallest subnormal


def test_results_are_python_floats_for_scalars():
    for name in UNARY:
        assert type(getattr(S, name)(0.3)) is float
    for name in BINARY:
        assert type(getattr(S, name)(0.3, 0.7)) is float


def test_int_and_bool_inputs():
    np.testing.assert_array_equal(np.asarray(S.expit(sd.asarray(np.array([0, 1, -1])))), sp.expit([0.0, 1.0, -1.0]))
    assert S.xlogy(True, 2) == sp.xlogy(1.0, 2.0)
    assert S.logit(0) == -np.inf


@pytest.mark.parametrize("name", ["expit", "logit"])
@pytest.mark.parametrize("n", [1, 2, 3, 1023, 1024, 1025, 2049, 5000])
def test_chunked_kernels_every_layout_same_bits(name, n):
    # f(x)[i] == f(x[i]) through every path of kernels/chunked.hpp: contiguous, gathered
    # (strided input), scattered (strided output), broadcast scalar, chunk boundaries
    rng = np.random.default_rng(n)
    x = rng.uniform(0, 1, 2 * n) if name == "logit" else rng.normal(scale=20, size=2 * n)
    f = getattr(S, name)
    contiguous = np.asarray(f(sd.asarray(x[:n].copy())))
    strided = np.asarray(f(sd.asarray(x)[::2]))
    out = sd.zeros(2 * n)
    f(sd.asarray(x[:n].copy()), out=out[::2])
    scalars = np.array([f(float(v)) for v in x[:n]])
    np.testing.assert_array_equal(contiguous.view(np.uint64), scalars.view(np.uint64))
    np.testing.assert_array_equal(strided.view(np.uint64), np.array([f(float(v)) for v in x[::2]]).view(np.uint64))
    np.testing.assert_array_equal(np.asarray(out)[::2].view(np.uint64), scalars.view(np.uint64))
    b = np.asarray(f(sd.asarray(np.full(3, x[0]))))
    assert all(bits_equal_any_nan(v, f(float(x[0]))) for v in b)


@pytest.mark.parametrize("name", ["expit", "logit"])
def test_chunked_in_place(name):
    x = np.random.default_rng(0).uniform(0.01, 0.99, 3000)
    a = sd.asarray(x.copy())
    want = np.asarray(getattr(S, name)(a))
    getattr(S, name)(a, out=a)
    np.testing.assert_array_equal(np.asarray(a), want)


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
                        S.expit(x),
                        S.logit(p),
                        S.log_expit(x),
                        S.xlogy(x, p),
                        S.xlog1py(x, p),
                        S.entr(p),
                        S.rel_entr(p, x * x),
                    )
                ]
            )
    finally:
        sd.set_num_threads(before)
    for r in runs[1:]:
        for a, b in zip(r, runs[0]):
            np.testing.assert_array_equal(a, b)


def test_complex_is_a_type_error():
    with pytest.raises(TypeError, match=r"^expit\(\): not supported for complex128$"):
        S.expit(1j)
    with pytest.raises(TypeError, match=r"^xlogy\(\): not supported for float64 and complex128$"):
        S.xlogy(1.0, 1j)


def test_messages_name_the_documented_arguments():
    with pytest.raises(TypeError, match=r"^logit\(\): 'p' must be"):
        S.logit("a")
    with pytest.raises(TypeError, match=r"^rel_entr\(\): 'y' must be"):
        S.rel_entr(1.0, None)
