"""sd.interp keeps numpy 2.5.4's interp where the design says so (docs/api/array.md §8.5; docs/design/interpolate.md
§2.2): left, right and period, the step at a repeated node, non-finite values, complex values, one node, the result's
shape, dtype and scalar type; and departs where it says so: xp is checked, NaN points are NaN, both -0 ends give -0.
Values are numpy's up to the linear contract (test_reference.py holds them to the exact references); here they are
compared with numpy live, and with the recorded numpy and R values."""

import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal, bits_equal_any_nan
from . import refdata
from . import samples as S
from .conftest import LIVE

INF, NAN = math.inf, math.nan
NUMPY = refdata.data("numpy")
R = refdata.data("r")


def interp(x, xp, fp, **kw):
    xs = x if np.isscalar(x) else sd.asarray(np.asarray(x, dtype=float))
    got = sd.interp(xs, sd.asarray(np.asarray(xp)), sd.asarray(np.asarray(fp)), **kw)
    return got if np.isscalar(got) else np.asarray(got)


def close(got, want, a_scale):
    """got and want within the linear contract's scale at every point (both NaN counts as equal)."""
    got, want = np.asarray(got), np.asarray(want)
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    return bool(np.all(same | (np.abs(got - want) <= 2 * refdata.gamma(6) * a_scale)))


@LIVE
def test_values_are_numpys_up_to_the_contract():
    g = np.random.default_rng(5)
    for n in (1, 2, 3, 17, 1000):
        xp = np.sort(g.uniform(-5, 5, n))
        fp = g.standard_normal(n) * 10.0 ** g.integers(-3, 4, n)
        x = np.concatenate([g.uniform(-6, 6, 5000), xp, [xp[0] - 1, xp[-1] + 1]])
        got = interp(x, xp, fp)
        assert close(got, np.interp(x, xp, fp), 2 * np.max(np.abs(fp))), n
        assert np.array_equal(got[5000 : 5000 + n], fp) or n > 1 and np.any(np.diff(xp) == 0)  # nodes: the data


@LIVE
def test_left_right_and_their_defaults():
    x, xp, fp = [-1.0, 0.0, 1.0, 2.0], [0.0, 1.0], [3.0, 4.0]
    assert interp(x, xp, fp).tolist() == np.interp(x, xp, fp).tolist() == [3.0, 3.0, 4.0, 4.0]
    assert interp(x, xp, fp, left=-7.0, right=9.0).tolist() == [-7.0, 3.0, 4.0, 9.0]
    assert interp(x, xp, fp, left=-7, right=NAN)[3] != interp(x, xp, fp, left=-7, right=NAN)[3]
    assert interp(x, xp, fp, left=2).tolist() == np.interp(x, xp, fp, left=2).tolist()


def test_repeated_nodes_are_a_step():
    # numpy: the value on the right at the node (tests/interpolate/data/numpy.json repeated, repeated_end)
    rep = NUMPY["repeated"]["value"]["array"]
    assert interp([0.5, 1.0, 1.5, 2.0], [0, 1, 1, 2], [0, 1, 5, 6]).tolist() == rep
    assert interp([1.0, 1.5], [0, 1, 1], [0, 1, 5]).tolist() == NUMPY["repeated_end"]["value"]["array"]
    assert interp([0.0, 0.5], [0, 0, 1], [1, 2, 3]).tolist() == [2.0, 2.5]
    assert interp(1.0, [0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]) == 3.0


def test_non_finite_values_are_numpys():
    for name, fp in (("inf_1", [INF, 1.0]), ("1_inf", [1.0, INF]), ("inf_inf", [INF, INF]),
                     ("-inf_inf", [-INF, INF]), ("nan_1", [NAN, 1.0]), ("1_nan", [1.0, NAN]),
                     ("-inf_1", [-INF, 1.0])):  # fmt: skip
        want = NUMPY[f"nonfinite/{name}"]["value"]["array"]
        assert np.array_equal(interp([0.0, 0.25, 0.5, 1.0], [0.0, 1.0], fp), want, equal_nan=True), name


def test_nan_points():
    assert np.array_equal(interp([NAN, 0.5], [0.0, 1.0], [0.0, 1.0]), [NAN, 0.5], equal_nan=True)  # numpy: the same
    # one node: NaN (numpy gives fp[0] there; decision IP39)
    assert math.isnan(interp(NAN, [0.0], [5.0]))
    got = interp([NAN, 0.25], [0.0, 1.0], [1 + 2j, 3 - 1j])  # numpy: NaN + 0j
    assert math.isnan(got[0].real) and got[0].imag == 0.0
    assert math.isnan(interp(NAN, [0.0, 1.0], [0.0, 1.0], period=2.0))


def test_one_node():
    assert interp([-1.0, 0.0, 1.0], [0.0], [5.0]).tolist() == NUMPY["one_point"]["value"]["array"]
    assert interp([-1.0, 0.0, 1.0], [0.0], [5.0], left=1.0, right=2.0).tolist() == [1.0, 5.0, 2.0]


def test_complex_values():
    rec = NUMPY["complex"]["value"]
    got = interp([0.25, 2.0], [0.0, 1.0], [1 + 2j, 3 - 1j])
    assert got.real.tolist() == rec["real"] and got.imag.tolist() == rec["imag"]
    assert interp(-1.0, [0.0, 1.0], [1j, 2.0], left=2 + 3j) == 2 + 3j
    assert interp(-1.0, [0.0, 1.0], [1j, 2.0], left=2) == 2 + 0j
    # the parts are computed as real interpolations with the same t
    g = np.random.default_rng(6)
    xp, fp, x = np.sort(g.random(30)), g.standard_normal(30) + 1j * g.standard_normal(30), g.random(500)
    got = interp(x, xp, fp)
    assert bits_equal(got.real, interp(x, xp, fp.real)) and bits_equal(got.imag, interp(x, xp, fp.imag))


def test_period():
    x, xp, fp = [-180.0, -10.0, 5.0, 190.0, 370.0], [-170.0, 10.0, 170.0, 350.0], [0.0, 1.0, 2.0, 3.0]
    assert interp(x, xp, fp, period=-360.0).tolist() == NUMPY["period_neg"]["value"]["array"]
    # left and right are ignored with a period (numpy)
    assert interp(x, xp, fp, period=360.0, left=-1.0).tolist() == NUMPY["period_left"]["value"]["array"]
    assert interp(x, xp, fp, period=360.0, left="ignored").tolist() == NUMPY["period_left"]["value"]["array"]
    # an unsorted xp is sorted after wrapping
    got = interp([5.0, 175.0], [190.0, -10.0, 10.0], [1.0, 2.0, 3.0], period=360)
    assert close(got, NUMPY["period_unsorted"]["value"]["array"], 4.0)
    assert got[1] == 7 / 6  # numpy: 1.1666666666666665, one ulp below the exact value


@LIVE
def test_period_against_numpy():
    g = np.random.default_rng(7)
    for p in (1.0, 2 * math.pi, 360.0, 1e-3):
        xp = g.uniform(-3 * p, 3 * p, 40)
        fp = g.standard_normal(40)
        x = np.concatenate([g.uniform(-10 * p, 10 * p, 3000), xp, [0.0, p, -p, 2 * p]])
        assert close(interp(x, xp, fp, period=p), np.interp(x, xp, fp, period=p), 2 * np.max(np.abs(fp))), p
        assert close(interp(x, xp, fp + 1j * fp[::-1], period=p), np.interp(x, xp, fp + 1j * fp[::-1], period=p),
                     4 * np.max(np.abs(fp)))  # fmt: skip


def test_signed_zero_ends():
    # numpy gives +0 inside when both ends are -0 (numpy.json signed_zeros); sd keeps -0 (§3.2)
    assert NUMPY["signed_zeros"]["signs"][0] == [-1.0, 1.0, -1.0]
    got = interp([0.0, 0.5, 1.0], [0.0, 1.0], [-0.0, -0.0])
    assert [math.copysign(1, v) for v in got] == [-1.0, -1.0, -1.0]
    for (a, b), want in zip(NUMPY["signed_zeros"]["ab"][1:], NUMPY["signed_zeros"]["signs"][1:]):
        assert [math.copysign(1, v) for v in interp([0.0, 0.5, 1.0], [0.0, 1.0], [a, b])] == want


def test_monotone_and_within_range_where_numpy_is_not():
    # the node on the right (numpy.json right_node): numpy's formula leaves [a, b] just left of x1; sd does not
    g = np.random.default_rng(8)
    x0 = g.uniform(0, 1, 20000)
    x1 = x0 + g.uniform(0.5, 2.0, 20000)
    a = g.standard_normal(20000)
    b = a + g.exponential(1.0, 20000) * np.where(g.random(20000) < 0.5, 1.0, -1.0)
    left = np.nextafter(x1, -np.inf)
    v = np.array([interp(q, [p0, p1], [fa, fb]) for q, p0, p1, fa, fb in zip(left, x0, x1, a, b)])
    assert np.all((v >= np.minimum(a, b)) & (v <= np.maximum(a, b)))
    assert np.all(np.array([interp(p1, [p0, p1], [fa, fb]) for p0, p1, fa, fb in zip(x0, x1, a, b)]) == b)


def test_recorded_numpy_and_r_values_within_the_contract():
    # the records of the real curves: sd agrees with numpy's interp and R's approx(rule = 2) up to the contract
    for cid in S.real_curve_ids():
        c = S.curve(cid)
        got = interp(c.xe, c.x, c.y)
        scale = 2 * np.max(np.abs(c.y))
        if cid != "subnormal/x":  # numpy's slope overflows there (test_reference_data.py)
            assert close(got, NUMPY[f"interp/{cid}"]["values"], scale), cid
        assert close(got, R[f"approx2/{cid}"]["values"], scale), cid
    c = S.curve("subnormal/x")
    assert np.all(np.isfinite(interp(c.xe, c.x, c.y)))  # finite where numpy's interp is not


def test_inputs_are_not_changed():
    xp, fp, x = np.array([2.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]), np.array([0.5, 7.0])
    sxp, sfp, sx = sd.asarray(xp), sd.asarray(fp), sd.asarray(x)
    sd.interp(sx, sxp, sfp, period=3.0)
    assert np.asarray(sxp).tolist() == xp.tolist() and np.asarray(sfp).tolist() == fp.tolist()
    assert np.asarray(sx).tolist() == x.tolist()


@pytest.mark.parametrize("n", [2, 5, 1000, 10**5])
def test_sorted_and_random_points_give_the_same_bits(n):
    # the search gallops through sorted blocks and searches others in lanes: the same interval either way
    g = np.random.default_rng(n)
    xp = np.sort(g.standard_normal(n))
    fp = g.standard_normal(n)
    x = np.concatenate([g.standard_normal(70000), xp, [NAN, INF, -INF]])
    xs = np.sort(x)
    order = np.argsort(x, kind="stable")
    a = interp(x, xp, fp)
    b = interp(xs, xp, fp)
    assert bits_equal_any_nan(a[order], b)  # np.sort changes the NaN's payload; sd keeps each point's NaN
    assert bits_equal(interp(xs[::-1].copy(), xp, fp), b[::-1].copy())
