"""FFTLog (milestone FT5; docs/api/fft.md §6; docs/design/fft.md §3.9, §4.2, §7.3): fht, ifht and fhtoffset.

- The contract against Arb (refdata.arb_fht, bound_fht): every point of refdata.fht_points on the lengths of
  refdata.FHT_LENGTHS (1-4096: even, odd and Bluestein inner lengths) and the parameter sets of
  refdata.FHT_PARAMS (unbiased, biased, negative spacing, the poles of Gamma, a large order, a small spacing),
  both directions; physical, subnormal and huge values on a few lengths. The plan's coefficients themselves
  within u of Arb's (refdata.fht_coefficients), fhtoffset within half an ulp of Arb's (refdata.arb_fhtoffset).
- scipy: the values of scipy's own tests (the Fortran FFTLog output, fht then ifht as the identity, the
  singular cases repaired as scipy repairs them, the exact transform of a power law), and every point within
  1e-11 of scipy's result relative to the norm (1e-9 at dln = 1e-4: scipy's own phase errors reach 29 times
  this contract at n = 4096, so the comparison is two-sided only that far); the dtypes and boundaries as
  scipy's, the differences marked.
- Layouts and batches: a line's bits are those of the line alone (the last axis of 2-d and 3-d arrays, C, F
  and strided, int64 and bool input, the empty batch); the plan cache (its limits, the same bits after a
  clear); n = 1 and 2.
- The wrong calls and their messages.
"""

import math
import warnings

import numpy as np
import pytest
import scipy.fft as sf
from scipy.special import poch

import stardust as sd

from .._bits import bits_equal
from . import refdata
from .conftest import LIVE

F = sd.fft
C = sd._core.fft
U = 2.0**-53


def _sd(a, dln, mu, offset, bias, inverse):
    f = F.ifht if inverse else F.fht
    return np.asarray(f(sd.asarray(a), dln, mu, offset=offset, bias=bias))


def _scipy(a, dln, mu, offset, bias, inverse):
    f = sf.ifht if inverse else sf.fht
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy warns on the singular sets
        return f(a, dln, mu, offset=offset, bias=bias)


SMALL = tuple(n for n in refdata.FHT_LENGTHS if n <= 1009)
POINTS = list(refdata.fht_points(SMALL)) + list(refdata.fht_points((2018, 4096), ("normal",)))
POINTS_VALUES = list(refdata.fht_points((16, 63, 64, 1000), ("gauss", "tiny", "huge")))


def _ids(v):
    return v if isinstance(v, str) else ""


@pytest.mark.parametrize("inverse", [False, True], ids=["fht", "ifht"])
@pytest.mark.parametrize("pid, a, dln, mu, offset, bias", POINTS + POINTS_VALUES, ids=_ids)
def test_contract_against_arb(pid, a, dln, mu, offset, bias, inverse):
    hi, lo = refdata.arb_fht(a, dln, mu, offset, bias, inverse)
    got = _sd(a, dln, mu, offset, bias, inverse)
    b = refdata.bound_fht(a, dln, mu, offset, bias, inverse)
    assert refdata.units(refdata.error(got, hi, lo), b) <= 1.0, pid


@pytest.mark.parametrize("inverse", [False, True], ids=["fht", "ifht"])
@pytest.mark.parametrize("n", [1, 2, 5, 64, 127, 1009, 4096])
@pytest.mark.parametrize("name, dln, mu, offset, bias", refdata.FHT_PARAMS, ids=[p[0] for p in refdata.FHT_PARAMS])
def test_coefficients_within_u(name, dln, mu, offset, bias, n, inverse):
    if dln is None and n < 2:
        return
    dln = refdata.fht_dln(dln, n)
    hi, lo = refdata.fht_coefficients(n, dln, mu, offset, bias, inverse)
    got = np.asarray(C._fht_plan(n, dln, mu, offset, bias, inverse)["coef"])
    ref = hi + lo
    err = np.abs((got - hi) - lo)
    assert np.all(err <= U * np.abs(ref) + 1e-300), (name, n, float(np.max(err / np.maximum(np.abs(ref), 1e-300))))
    if name in ("pole-plus", "pole-minus"):
        assert got[0] == 0 and C._fht_plan(n, dln, mu, offset, bias, inverse)["singular"]


@pytest.mark.parametrize("inverse", [False, True], ids=["fht", "ifht"])
@pytest.mark.parametrize(
    "dln, mu, offset, bias",
    [(0.05, -30.3, 0.4, 0.7), (0.003, 5.5, -2.0, -2.2), (2.0, 0.0, 0.0, 0.0), (0.7, -7.25, 3.0, 0.5),
     (1e-3, 120.0, 0.0, 1.5), (-0.02, -1.0, -0.5, 0.0)],
)  # fmt: skip
def test_coefficients_within_u_far_out(dln, mu, offset, bias, inverse):
    # the reflection with a far negative real part, a large order, large and negative spacings
    for n in (8, 333, 1024):
        hi, lo = refdata.fht_coefficients(n, dln, mu, offset, bias, inverse)
        got = np.asarray(C._fht_plan(n, dln, mu, offset, bias, inverse)["coef"])
        err = np.abs((got - hi) - lo)
        assert np.all(err <= U * np.abs(hi + lo) + 1e-300), (n, float(np.max(err / np.abs(hi + lo))))


@pytest.mark.parametrize(
    "dln, mu, initial, bias",
    [(0.1, 2.0, 0.5, 0.0), (0.01, 0.0, -6 * math.log(10), 0.0), (-0.3, 1.5, 0.2, -0.6), (1e-4, 0.5, 3.0, 0.8),
     (math.log(1e8) / 15, 0.3, 0.0, 0.8), (2.5, -3.5, 0.0, 0.5), (0.05, 50.5, -1.0, 0.0)],
)  # fmt: skip
def test_fhtoffset_against_arb(dln, mu, initial, bias):
    ref = refdata.arb_fhtoffset(dln, mu, initial, bias)
    got = F.fhtoffset(dln, mu, initial=initial, bias=bias)
    assert type(got) is float
    assert abs(got - ref) <= 0.5 * math.ulp(ref) + 1e-300, (got, ref)
    assert abs(got - sf.fhtoffset(dln, mu, initial=initial, bias=bias)) <= 1e-12 * max(1.0, abs(ref))


# ---------------------------------------------------------------- scipy's own tests


def _gauss(r, mu):
    return r ** (mu + 1) * np.exp(-(r**2) / 2)


def test_agrees_with_fortran_fftlog():
    # scipy's test_fht_agrees_with_fftlog: the output of Hamilton's fftlogtest (its four settings)
    r = np.logspace(-4, 4, 16)
    dln = math.log(r[1] / r[0])
    mu = 0.3
    a = _gauss(r, mu)
    theirs = {
        (0.0, False): [-0.1159922613593045e-02, +0.1625822618458832e-02, -0.1949518286432330e-02,
                       +0.3789220182554077e-02, +0.5093959119952945e-03, +0.2785387803618774e-01,
                       +0.9944952700848897e-01, +0.4599202164586588e00, +0.3157462160881342e00,
                       -0.8201236844404755e-03, -0.7834031308271878e-03, +0.3931444945110708e-03,
                       -0.2697710625194777e-03, +0.3568398050238820e-03, -0.5554454827797206e-03,
                       +0.8286331026468585e-03],
        (0.0, True): [+0.4353768523152057e-04, -0.9197045663594285e-05, +0.3150140927838524e-03,
                      +0.9149121960963704e-03, +0.5808089753959363e-02, +0.2548065256377240e-01,
                      +0.1339477692089897e00, +0.4821530509479356e00, +0.2659899781579785e00,
                      -0.1116475278448113e-01, +0.1791441617592385e-02, -0.4181810476548056e-03,
                      +0.1314963536765343e-03, -0.5422057743066297e-04, +0.3208681804170443e-04,
                      -0.2696849476008234e-04],
        (0.8, True): [-7.3436673558316850e00, +0.1710271207817100e00, +0.1065374386206564e00,
                      -0.5121739602708132e-01, +0.2636649319269470e-01, +0.1697209218849693e-01,
                      +0.1250215614723183e00, +0.4739583261486729e00, +0.2841149874912028e00,
                      -0.8312764741645729e-02, +0.1024233505508988e-02, -0.1644902767389120e-03,
                      +0.3305775476926270e-04, -0.7786993194882709e-05, +0.1962258449520547e-05,
                      -0.8977895734909250e-06],
        (-0.8, True): [+0.8985777068568745e-05, +0.4074898209936099e-04, +0.2123969254700955e-03,
                       +0.1009558244834628e-02, +0.5131386375222176e-02, +0.2461678673516286e-01,
                       +0.1235812845384476e00, +0.4719570096404403e00, +0.2893487490631317e00,
                       -0.1686570611318716e-01, +0.2231398155172505e-01, -0.1480742256379873e-01,
                       +0.1692387813500801e00, +0.3097490354365797e00, +2.7593607182401860e00,
                       10.5251075070045800e00],
    }  # fmt: skip
    for (bias, optimal), want in theirs.items():
        offset = F.fhtoffset(dln, mu, bias=bias) if optimal else 0.0
        got = np.asarray(F.fht(sd.asarray(a), dln, mu, offset=offset, bias=bias))
        np.testing.assert_allclose(got, want, rtol=1e-7)


@pytest.mark.parametrize("optimal", [True, False])
@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("bias", [0, 0.1, -0.1])
@pytest.mark.parametrize("n", [64, 63])
def test_identity(n, bias, offset, optimal):
    # scipy's test_fht_identity: ifht(fht(a)) = a
    rng = np.random.RandomState(3491349965)
    a = rng.standard_normal(n)
    dln = rng.uniform(-1, 1)
    mu = rng.uniform(-2, 2)
    if optimal:
        offset = F.fhtoffset(dln, mu, initial=offset, bias=bias)
    A = F.fht(sd.asarray(a), dln, mu, offset=offset, bias=bias)
    np.testing.assert_allclose(np.asarray(F.ifht(A, dln, mu, offset=offset, bias=bias)), a, rtol=1.5e-7)


def test_singular_cases_are_scipys_repair_without_a_warning():
    # scipy's test_fht_special_cases: x = (mu + 1 + q) / 2, y = (mu + 1 - q) / 2 at the poles M = {0, -1, ...}
    rng = np.random.RandomState(3491349965)
    a = rng.standard_normal(64)
    dln = rng.uniform(-1, 1)
    for mu, bias, inverse, singular in ((-4.0, 1.0, False, False), (-2.5, 0.5, False, True), (-3.5, 0.5, False, True),
                                        (-2.5, 0.5, True, True), (-4.0, 1.0, True, False)):  # fmt: skip
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # sd warns about nothing (docs/api/fft.md §1.4)
            got = _sd(a, dln, mu, 0.0, bias, inverse)
        want = _scipy(a, dln, mu, 0.0, bias, inverse)
        assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want)), (mu, bias, inverse)
        assert C._fht_plan(64, dln, mu, 0.0, bias, inverse)["singular"] is (singular and (mu, bias) != (-4.0, 1.0))


@pytest.mark.parametrize("n", [64, 63])
def test_exact_power_law(n):
    # scipy's test_fht_exact: a power law r^gamma biased with q = gamma transforms exactly
    rng = np.random.RandomState(3491349965)
    mu = rng.uniform(0, 3)
    gamma = rng.uniform(-1 - mu, 1 / 2)
    r = np.logspace(-2, 2, n)
    dln = math.log(r[1] / r[0])
    offset = F.fhtoffset(dln, mu, initial=0.0, bias=gamma)
    A = np.asarray(F.fht(sd.asarray(r**gamma), dln, mu, offset=offset, bias=gamma))
    k = np.exp(offset) / r[::-1]
    np.testing.assert_allclose(A, (2 / k) ** gamma * poch((mu + 1 - gamma) / 2, gamma), rtol=1e-12)


@LIVE
@pytest.mark.parametrize("inverse", [False, True], ids=["fht", "ifht"])
def test_scipy_agrees(inverse):
    for pid, a, dln, mu, offset, bias in POINTS:
        got = _sd(a, dln, mu, offset, bias, inverse)
        want = _scipy(a, dln, mu, offset, bias, inverse)
        scale = refdata.norm2(want)
        tol = 1e-9 if pid.startswith("small-dln") else 1e-11  # y up to 15708: scipy's phases lose more
        assert refdata.norm2(got - want) <= tol * scale + 1e-300, pid


@LIVE
def test_scipy_misses_this_contract_where_its_phases_are_large():
    # the reason for decision FT58: scipy's double log-gamma phases at n = 4096, dln = ln(1e8) / 4095
    n = 4096
    dln, offset = math.log(1e8) / (n - 1), -6 * math.log(10)
    a = np.random.default_rng(4096).standard_normal(n)
    hi, lo = refdata.arb_fht(a, dln, 0.3, offset)
    b = refdata.bound_fht(a, dln, 0.3, offset)
    assert refdata.units(refdata.error(_scipy(a, dln, 0.3, offset, 0.0, False), hi, lo), b) > 1.0
    assert refdata.units(refdata.error(_sd(a, dln, 0.3, offset, 0.0, False), hi, lo), b) < 0.05


# ---------------------------------------------------------------- dtypes, layouts, batches, the cache


def test_dtypes_and_shapes():
    a = np.array([3, -1, 4, 1, -5, 9, 2, 6])
    want = np.asarray(F.fht(sd.asarray(a.astype(np.float64)), 0.3, 0.5))
    got = F.fht(sd.asarray(a), 0.3, 0.5)
    assert str(got.dtype) == "float64" and got.shape == (8,) and bits_equal(got, want)
    b = np.array([True, False, True, True])
    assert bits_equal(F.ifht(sd.asarray(b), 0.2, 1.0), F.ifht(sd.asarray(b.astype(np.float64)), 0.2, 1.0))
    assert F.fht(sd.asarray(np.zeros((0, 8))), 0.3, 0.5).shape == (0, 8)
    assert F.fht(sd.asarray(np.zeros((3, 0, 8))), 0.3, 0.5).shape == (3, 0, 8)
    # scipy: real input of any shape, the last axis transformed
    x = np.random.default_rng(5).standard_normal((3, 4, 10))
    np.testing.assert_allclose(np.asarray(F.fht(sd.asarray(x), 0.1, 0.2)), sf.fht(x, 0.1, 0.2), rtol=0, atol=1e-13)


def test_short_lengths():
    for n in (1, 2, 3):
        a = np.arange(1.0, n + 1)
        for inverse in (False, True):
            for bias in (0.0, 0.5):
                got = _sd(a, 0.7, 0.4, 0.1, bias, inverse)
                np.testing.assert_allclose(got, _scipy(a, 0.7, 0.4, 0.1, bias, inverse), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("n", [64, 63, 1009])
@pytest.mark.parametrize("bias", [0.0, 0.3])
def test_a_line_is_the_line_alone(n, bias):
    g = np.random.default_rng(n)
    x = g.standard_normal((6, 5, n))
    for inverse in (False, True):
        f = F.ifht if inverse else F.fht
        alone = [f(sd.asarray(np.ascontiguousarray(x[i, j])), 0.05, 0.5, offset=0.2, bias=bias)
                 for i in range(6) for j in range(5)]  # fmt: skip
        for view in (x, np.asfortranarray(x), x[:, :, ::-1][:, :, ::-1], np.ascontiguousarray(x[::-1])[::-1]):
            got = np.asarray(f(sd.asarray(view), 0.05, 0.5, offset=0.2, bias=bias))
            for i in range(6):
                for j in range(5):
                    assert bits_equal(got[i, j], alone[5 * i + j]), (n, bias, inverse, i, j)
        sx = sd.asarray(np.concatenate([x, x], axis=2))[:, :, ::2]  # a strided last axis
        got = np.asarray(f(sx, 0.05, 0.5, offset=0.2, bias=bias))
        want = np.asarray(f(sd.asarray(np.ascontiguousarray(np.concatenate([x, x], axis=2)[:, :, ::2])), 0.05, 0.5,
                            offset=0.2, bias=bias))  # fmt: skip
        assert bits_equal(got, want)


def test_plan_cache():
    C._fht_cache_clear()
    a = sd.asarray(np.random.default_rng(1).standard_normal(256))
    first = F.fht(a, 0.1, 0.5, bias=0.2)
    s = C._fht_cache_state()
    assert s["plans"] == 1 and s["lengths"] == [256]
    assert s["bytes"] == C._fht_plan(256, 0.1, 0.5, 0.0, 0.2, False)["bytes"]
    for k in range(s["max_plans"] + 4):  # more plans than the cache keeps: the oldest go
        F.fht(a, 0.1 + k, 0.5)
    s = C._fht_cache_state()
    assert s["plans"] == s["max_plans"] and s["bytes"] <= s["max_bytes"]
    assert bits_equal(F.fht(a, 0.1, 0.5, bias=0.2), first)  # made again: the same bits
    C._fht_cache_clear()
    assert C._fht_cache_state()["plans"] == 0
    assert bits_equal(F.fht(a, 0.1, 0.5, bias=0.2), first)


def test_ifht_inverts_fht():
    g = np.random.default_rng(9)
    for n, biases in ((16, (0.0, 0.4)), (127, (0.0, 0.4)), (1000, (0.0,))):  # a bias makes it ill-conditioned
        a = g.standard_normal(n)
        for bias in biases:
            A = F.fht(sd.asarray(a), 0.08, 0.7, offset=0.3, bias=bias)
            back = np.asarray(F.ifht(A, 0.08, 0.7, offset=0.3, bias=bias))
            assert refdata.norm2(back - a) <= 1e-13 * refdata.norm2(a), (n, bias)


# ---------------------------------------------------------------- wrong calls


def test_errors():
    a = sd.asarray(np.arange(8.0))
    cases = [
        (lambda: F.fht([1.0, 2.0], 0.1, 0.5), TypeError, "fht(): 'a' must be an sd.ndarray, not list"),
        (lambda: F.ifht(np.arange(4.0), 0.1, 0.5), TypeError, "ifht(): 'A' must be an sd.ndarray, not numpy.ndarray"),
        (lambda: F.fht(sd.asarray(np.ones(4) + 1j), 0.1, 0.5), TypeError, "fht(): 'a' must be real, not complex128"),
        (lambda: F.fht(a, 0.0, 0.5), ValueError, "fht(): 'dln' must be finite and nonzero, not 0.0"),
        (lambda: F.fht(a, math.inf, 0.5), ValueError, "fht(): 'dln' must be finite and nonzero, not inf"),
        (lambda: F.fht(a, "0.1", 0.5), TypeError, "fht(): 'dln' must be an int or a float, not str"),
        (lambda: F.fht(a, 0.1, math.nan), ValueError, "fht(): 'mu' must be finite, not nan"),
        (lambda: F.fht(a, 0.1, None), TypeError, "fht(): 'mu' must be an int or a float, not NoneType"),
        (lambda: F.fht(a, 0.1, 0.5, offset=math.inf), ValueError, "fht(): 'offset' must be finite, not inf"),
        (lambda: F.ifht(a, 0.1, 0.5, bias=True), TypeError, "ifht(): 'bias' must be an int or a float, not bool"),
        (lambda: F.fht(sd.asarray(np.zeros((3, 0))), 0.1, 0.5), ValueError,
         "fht(): cannot transform an axis of length 0"),
        (lambda: F.fht(a, 0.1, 0.5, 0.0), TypeError, "fht(): takes 3 positional arguments but 4 were given"),
        (lambda: F.fht(a, 0.1), TypeError, "fht(): missing 1 required positional argument: 'mu'"),
        (lambda: F.fht(a, 0.1, 0.5, workers=2), TypeError, "fht(): unexpected keyword argument 'workers'"),
        (lambda: F.fhtoffset(0.0, 0.5), ValueError, "fhtoffset(): 'dln' must be finite and nonzero, not 0.0"),
        (lambda: F.fhtoffset(0.1, 0.5, initial=math.nan), ValueError, "fhtoffset(): 'initial' must be finite, not nan"),
        (lambda: F.fhtoffset(0.1, 0.5, 0.0), TypeError, "fhtoffset(): takes 2 positional arguments but 3 were given"),
    ]  # fmt: skip
    for call, kind, message in cases:
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message


def test_numpy_scalars_are_parameters():
    a = sd.asarray(np.arange(8.0))
    assert bits_equal(F.fht(a, np.float64(0.1), np.int64(1)), F.fht(a, 0.1, 1.0))
    assert F.fhtoffset(np.float64(0.1), 2) == F.fhtoffset(0.1, 2.0)
