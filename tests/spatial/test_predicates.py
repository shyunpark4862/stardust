"""The exact predicates of csrc/spatial/predicates.hpp (docs/design/spatial_geometry.md §3.1, §3.2, §7.3;
SPT8): every stage gives the sign of the rational determinant, the tie rule equals the polynomial in eps,
the value kernels keep their contract, and the filter constants satisfy their inequality."""

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from . import exactgeom as eg
from . import geompoints as gp

C = sd._core.spatial
U = Fraction(1, 2**53)
DEGREE = {"orient2d": 2, "orient3d": 3, "incircle": 4, "insphere": 5}
ROWS = {"uniform": 1500, "grid": 1500, "zero": 600, "near": 1500, "far": 1500, "dyadic": 1500, "extreme": 400}


def call(name, rows, mode="auto"):
    return np.asarray(C._predicate(name, sd.asarray(np.ascontiguousarray(rows, dtype=np.float64)), mode))


def exact(name, rows):
    return np.array([eg.predicate_sign(name, r) for r in rows], dtype=np.int64)


def safe(rows, q):
    """Range::safe(q) of each row, recomputed here from the definition."""
    out = []
    for r in rows:
        v = [abs(float(x)) for x in np.ravel(r) if x != 0.0]
        if not v:
            out.append(True)
            continue
        hi = max(math.frexp(x)[1] - 1 for x in v)  # floor(log2 x)
        fr = [Fraction(x) for x in v]
        lo = min((f.numerator & -f.numerator).bit_length() - f.denominator.bit_length() for f in fr)
        out.append(q * (hi + 2) + 8 <= 1023 and (lo >= 0 or q * lo >= -1074))
    return np.array(out)


def test_filter_constants_satisfy_their_inequality():
    """K >= theta_(c-1) / (1 - u)^(c_P + 2) with the operation counts of predicates.hpp's header, in rational
    arithmetic against the compiled constants; each is the smallest (k + 16 j u) u that does."""
    k = C._predicate_constants()
    counts = {"orient2d": (4, 4, 3), "orient3d": (8, 8, 7), "incircle": (11, 11, 10), "insphere": (16, 17, 15)}
    for name, (c, cp, lead) in counts.items():
        need = ((1 + U) ** (c - 1) - 1) / (1 - U) ** (cp + 2)
        have = Fraction(k[name])
        assert have >= need, name
        j = (have / U - lead) / U
        assert j.denominator == 1 and j % 16 == 0 and (lead + (j - 16) * U) * U < need, name
    assert k["value_accept"] == 2.0**-36


@pytest.mark.parametrize("name", list(gp.PREDICATES))
@pytest.mark.parametrize("kind", gp.ROW_KINDS)
def test_every_stage_gives_the_rational_sign(name, kind):
    rows = gp.predicate_rows(name, kind, ROWS[kind])
    want = exact(name, rows)
    assert np.array_equal(call(name, rows, "auto"), want)
    assert np.array_equal(call(name, rows, "wide"), want)
    ok = safe(rows, DEGREE[name])
    if kind != "extreme":
        assert ok.all()
    inside, w = rows[ok], want[ok]
    if len(inside):
        assert np.array_equal(call(name, inside, "fast"), w)
        f = call(name, inside, "filter")
        assert np.array_equal(f[f != 2], w[f != 2]) and not (f == 0).any()
        e = call(name, inside, "expansion")
        assert np.array_equal(e[e != 2], w[e != 2])
    stages = C._predicate_stages(name, sd.asarray(rows))
    assert sum(stages) == len(rows) and stages[3] == int((~ok).sum())
    if kind in ("zero", "grid", "far"):  # exact differences: the wide path is never needed
        assert stages[2] == 0
    if kind == "zero":
        assert (want == 0).all() and stages[1] == len(rows)


@pytest.mark.parametrize("id_", [i for i in gp.IDS if "large" not in gp.get(i).tags])
def test_points_of_every_test_set(id_):
    """Tuples drawn from each point set of geompoints (neighbours in lexicographic order, where the
    degenerate configurations are, and random ones)."""
    s = gp.get(id_)
    x, d = s.x, s.x.shape[1]
    g = np.random.default_rng(len(id_))
    order = np.lexsort(x.T[::-1])
    for name in ("orient2d", "incircle") if d == 2 else ("orient3d", "insphere"):
        k = gp.PREDICATES[name][0]
        idx = [order[(i + np.arange(k)) % len(x)] for i in range(0, len(x), max(1, len(x) // 150))]
        idx += [g.integers(0, len(x), k) for _ in range(150)]
        rows = x[np.array(idx)]
        want = exact(name, rows)
        assert np.array_equal(call(name, rows, "auto"), want)
        assert np.array_equal(call(name, rows, "wide"), want)
        if safe(x[None], DEGREE[name])[0]:
            assert np.array_equal(call(name, rows, "fast"), want)


finite = st.floats(allow_nan=False, allow_infinity=False, width=64)
small_int = st.integers(-4, 4).map(float)
dyadic = st.builds(lambda s, k: s * 2.0**k, st.sampled_from([-1.0, 1.0, 3.0, -5.0]), st.integers(-30, 30))


def rows_of(name, element):
    k, d = gp.PREDICATES[name]
    return st.lists(st.lists(element, min_size=d, max_size=d), min_size=k, max_size=k)


@pytest.mark.parametrize("name", list(gp.PREDICATES))
@settings(max_examples=300, deadline=None)
@given(data=st.data())
def test_hypothesis(name, data):
    element = data.draw(st.sampled_from([finite, small_int, dyadic, st.one_of(finite, small_int, dyadic)]))
    row = np.array([data.draw(rows_of(name, element))], dtype=np.float64)
    want = exact(name, row)
    assert np.array_equal(call(name, row, "auto"), want)
    assert np.array_equal(call(name, row, "wide"), want)
    if safe(row, DEGREE[name])[0]:
        assert np.array_equal(call(name, row, "fast"), want)


@pytest.mark.parametrize("name", list(gp.PREDICATES))
def test_antisymmetry_and_translation(name):
    rows = gp.predicate_rows(name, "near", 300)
    base = call(name, rows)
    swapped = rows.copy()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    assert np.array_equal(call(name, swapped), -base)
    ints = gp.predicate_rows(name, "grid", 300)
    assert np.array_equal(call(name, ints + 2.0**30), call(name, ints))  # an exact translation


@pytest.mark.parametrize("name", ["incircle", "insphere"])
@pytest.mark.parametrize("kind", ["zero", "grid", "near", "extreme"])
def test_tie_rule_is_the_eps_polynomial(name, kind):
    rows = gp.predicate_rows(name, kind, 2500 if kind == "grid" else 250)
    rows = np.array([r for r in rows if len({tuple(p) for p in r.tolist()}) == len(r)][:250])
    g = np.random.default_rng(11)
    rows = np.concatenate([rows, np.array([r[g.permutation(len(r))] for r in rows])])
    want = np.array([eg.in_sphere_eps(*eg.scaled_ints(r)) for r in rows])
    plain = exact(name, rows)
    assert np.array_equal(want[plain != 0], plain[plain != 0])
    for mode in ("auto", "wide") + (() if kind == "extreme" else ("fast",)):
        assert np.array_equal(call(name + "_sos", rows, mode), want), mode
    simplex = "orient2d" if name == "incircle" else "orient3d"
    nondegenerate = exact(simplex, rows[:, :-1]) != 0
    assert (want[nondegenerate] != 0).all()
    if kind == "zero":
        assert (plain == 0).all() and nondegenerate.sum() > 100


def test_tie_rule_makes_each_cocircular_quadruple_one_diagonal():
    """Four cocircular points in convex position: exactly one of the two diagonals passes (each of its two
    triangles has the fourth point outside), whatever the argument order."""
    pts = np.array(gp._on_circle_int(1105), dtype=np.float64)
    g = np.random.default_rng(3)
    for _ in range(200):
        q = pts[g.choice(len(pts), 4, replace=False)]
        a = np.arctan2(q[:, 1], q[:, 0])
        q = q[np.argsort(a)]  # counterclockwise a, b, c, d
        rows = np.array(
            [[q[0], q[1], q[2], q[3]], [q[0], q[2], q[3], q[1]], [q[1], q[2], q[3], q[0]], [q[0], q[1], q[3], q[2]]]
        )
        s = call("incircle_sos", rows)
        assert s[0] == s[1] and s[2] == s[3] and s[0] == -s[2] and s[0] != 0


def det_exact(name, row):
    p = eg.fpoints(row)
    if name == "det2":
        return eg.orient_value(p[2], p[0], p[1])  # det[a - c; b - c]
    return eg.orient_value(*p)


def to_float(v: Fraction) -> float:
    try:
        return float(v)
    except OverflowError:
        return math.inf if v > 0 else -math.inf


@pytest.mark.parametrize("name", ["det2", "det3"])
@pytest.mark.parametrize("kind", gp.ROW_KINDS)
def test_value_kernels(name, kind):
    """|r - x| <= 1.002 u |x| (+ 2^-1075 for subnormal results; 0 exactly; +-inf on overflow). The wide path
    is the correctly rounded value; the fast path answers NaN or a certified value."""
    pred = "orient2d" if name == "det2" else "orient3d"
    rows = gp.predicate_rows(pred, kind, 500 if kind != "extreme" else 300)
    P = sd.asarray(rows)
    auto, wide = np.asarray(C._det(name, P, "auto")), np.asarray(C._det(name, P, "wide"))
    ok = safe(rows, DEGREE[pred])
    fast = np.asarray(C._det(name, sd.asarray(rows[ok]), "fast")) if ok.any() else np.empty(0)
    exacts = [det_exact(name, r) for r in rows]
    for i, x in enumerate(exacts):
        assert wide[i] == to_float(x), (i, wide[i], x)
        if x == 0:
            assert auto[i] == 0.0
        elif math.isinf(to_float(x)):
            assert auto[i] == to_float(x)
        else:
            assert abs(Fraction(float(auto[i])) - x) <= Fraction(1002, 1000) * U * abs(x) + Fraction(1, 2**1075)
    for r, x in zip(fast, [x for x, k in zip(exacts, ok) if k]):
        if not math.isnan(r):
            assert abs(Fraction(float(r)) - x) <= Fraction(1002, 1000) * U * abs(x)
    if kind == "uniform":
        assert not np.isnan(fast).any()
    if kind == "zero":
        assert (auto == 0.0).all()


def test_value_kernel_fast_path_declines_ill_conditioned_determinants():
    a = np.array([[[1.0, 1.0], [1.0 + 2.0**-30, 1.0 + 2.0**-30 + 2.0**-52], [0.1, 0.1]]])
    r = np.asarray(C._det("det2", sd.asarray(a), "fast"))
    assert np.isnan(r).all()
    x = det_exact("det2", a[0])
    assert np.asarray(C._det("det2", sd.asarray(a), "auto"))[0] == float(x)


@settings(max_examples=300, deadline=None)
@given(x=st.lists(finite, min_size=6, max_size=6))
def test_wide_integers(x):
    fr = [Fraction(v) for v in x]
    if all(v == 0 for v in fr):
        assert C._wide_eval(sd.asarray(x)) == "0x0"
        return
    lo = min((v.numerator & -v.numerator).bit_length() - v.denominator.bit_length() for v in fr if v != 0)
    X = [int(v / Fraction(2) ** lo) for v in fr]  # the coordinates over their lowest set bit
    assert int(C._wide_eval(sd.asarray(x)), 16) == (X[0] * X[1] - X[2] * X[3]) * X[4] + X[5] ** 3


@settings(max_examples=500, deadline=None)
@given(x=st.lists(finite, min_size=3, max_size=3), shift=st.integers(-1200, 1200))
def test_wide_rounding_is_correct(x, shift):
    v = Fraction(x[0]) * Fraction(x[1]) * Fraction(x[2]) * Fraction(2) ** shift
    assert C._wide_round(sd.asarray(x), shift) == to_float(v)


def test_wide_rounding_at_the_subnormal_boundary():
    tiny = 5e-324
    for num, want in [
        (1, 0.0),
        (2, 0.0),
        (3, tiny),
        (5, tiny),
        (6, 2 * tiny),
        (7, 2 * tiny),
        (10, 2 * tiny),
        (11, 3 * tiny),
    ]:
        # num / 4 of the smallest subnormal: ties go to the even neighbour
        assert C._wide_round(sd.asarray([float(num), 1.0, 1.0]), -1076) == to_float(Fraction(num, 2**1076)) == want
    assert C._wide_round(sd.asarray([1.0, 1.0, 1.0]), 1024) == math.inf
    assert C._wide_round(sd.asarray([-(2.0 - 2.0**-53), 1.0, 1.0]), 1023) == -math.inf  # rounds up to 2^1024
    assert C._wide_round(sd.asarray([2.0 - 2.0**-52, 1.0, 1.0]), 1023) == np.finfo(np.float64).max


@settings(max_examples=300, deadline=None)
@given(
    x=st.lists(
        st.floats(-1e60, 1e60, allow_subnormal=False).filter(lambda v: v == 0 or abs(v) > 1e-60), min_size=8, max_size=8
    )
)
def test_expansions_are_exact(x):
    comps, s = C._expansion_eval(sd.asarray(x))
    f = [Fraction(v) for v in x]
    want = (f[0] * f[1] - f[2] * f[3]) * (f[4] * f[5] + f[6] * f[7])
    assert sum(Fraction(c) for c in comps) == want
    assert s == eg.sign(want)
    assert 0.0 not in comps and len(comps) <= 32


def test_range():
    assert C._range(sd.asarray([0.0, -0.0])) is None
    assert C._range(sd.asarray([1.0])) == (0, 0)
    assert C._range(sd.asarray([0.75, -6.0, 0.0])) == (2, -2)
    assert C._range(sd.asarray([5e-324, 1.5])) == (0, -1074)
    assert C._range(sd.asarray([np.finfo(np.float64).max])) == (1023, 971)
    assert C._range(sd.asarray([[0.1, 2.0**-1022]])) == (-4, -1022)


def test_wrong_calls():
    rows = sd.asarray(np.zeros((2, 3, 2)))
    with pytest.raises(ValueError):
        C._predicate("orient2d", rows, "nope")
    with pytest.raises(ValueError):
        C._predicate("incircle", rows, "auto")
    with pytest.raises(ValueError):
        C._predicate("incircle_sos", sd.asarray(np.zeros((1, 4, 2))), "filter")
    with pytest.raises(TypeError):
        C._predicate("orient2d", np.zeros((2, 3, 2)), "auto")
