"""The errors of sd.interp (docs/api/array.md §8.5; API.md §2, §2.1): kind and message, named by the function."""

import math

import numpy as np
import pytest

import stardust as sd

A = sd.asarray
NOT_X = "interp(): 'x' must be an sd.ndarray or a bool, int or float, not "
XP, FP = A([0.0, 1.0]), A([0.0, 1.0])

CASES = [
    # (call, exception, message)
    (lambda: sd.interp(A([0.5]), A([0.0, 2.0, 1.0]), A([0.0, 1.0, 2.0])), ValueError,
     "interp(): 'xp' must be nondecreasing"),
    (lambda: sd.interp(A([0.5]), A([0.0, math.nan]), FP), ValueError, "interp(): 'xp' must be finite"),
    (lambda: sd.interp(A([0.5]), A([0.0, math.inf]), FP), ValueError, "interp(): 'xp' must be finite"),
    (lambda: sd.interp(A([0.5]), A([math.nan, 0.0]), FP, period=1.0), ValueError, "interp(): 'xp' must be finite"),
    (lambda: sd.interp(A([0.5]), A(np.zeros(0)), A(np.zeros(0))), ValueError, "interp(): 'xp' must not be empty"),
    (lambda: sd.interp(A([0.5]), XP, A([1.0])), ValueError, "interp(): 'fp' must be as long as 'xp'"),
    (lambda: sd.interp(A([0.5]), A([[0.0, 1.0]]), A([[0.0, 1.0]])), ValueError, "interp(): 'xp' must be 1-D"),
    (lambda: sd.interp(A([0.5]), XP, A([[0.0], [1.0]])), ValueError, "interp(): 'fp' must be 1-D"),
    (lambda: sd.interp(A([0.5j]), XP, FP), TypeError, "interp(): 'x' must be real, not complex128"),
    (lambda: sd.interp(0.5j, XP, FP), TypeError, "interp(): 'x' must be real, not complex"),
    (lambda: sd.interp(0.5, A([0j, 1.0]), FP), TypeError, "interp(): 'xp' must be real, not complex128"),
    (lambda: sd.interp([0.5], XP, FP), TypeError, NOT_X + "list"),
    (lambda: sd.interp(np.array([0.5]), XP, FP), TypeError, NOT_X + "numpy.ndarray"),
    (lambda: sd.interp(None, XP, FP), TypeError, NOT_X + "NoneType"),
    (lambda: sd.interp(0.5, [0.0, 1.0], FP), TypeError, "interp(): 'xp' must be an sd.ndarray, not list"),
    (lambda: sd.interp(0.5, XP, (0.0, 1.0)), TypeError, "interp(): 'fp' must be an sd.ndarray, not tuple"),
    (lambda: sd.interp(0.5, XP, FP, period=0), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period=-0.0), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period=math.nan), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period=math.inf), ValueError, "interp(): 'period' must be finite and nonzero"),
    (lambda: sd.interp(0.5, XP, FP, period="1"), TypeError, "interp(): 'period' must be an int or a float, not str"),
    (lambda: sd.interp(0.5, XP, FP, period=True), TypeError, "interp(): 'period' must be an int or a float, not bool"),
    (lambda: sd.interp(0.5, XP, FP, left=1j), TypeError, "interp(): 'left' must be an int or a float, not complex"),
    (lambda: sd.interp(0.5, XP, FP, right="a"), TypeError, "interp(): 'right' must be an int or a float, not str"),
    (lambda: sd.interp(0.5, XP, A([0j, 1.0]), left="a"), TypeError, "interp(): 'left' must be a number, not str"),
    (lambda: sd.interp(0.5, XP, FP, period=10**400), OverflowError, "interp(): 'period' is out of range"),
]  # fmt: skip


@pytest.mark.parametrize("k", range(len(CASES)))
def test_error(k):
    call, exc, msg = CASES[k]
    with pytest.raises(exc) as e:
        call()
    assert str(e.value) == msg


def test_call_shape():
    # numpy takes left and right by position; sd only by keyword (docs/api/array.md §8.5)
    with pytest.raises(TypeError, match=r"^interp\(\): incompatible function arguments"):
        sd.interp(A([0.5]), XP, FP, 0.0, 1.0)
    with pytest.raises(TypeError, match=r"^interp\(\): incompatible function arguments"):
        sd.interp(x=A([0.5]), xp=XP, fp=FP)


def test_unsorted_xp_with_a_period_is_sorted_not_refused():
    assert sd.interp(0.5, A([1.0, 0.0]), A([1.0, 0.0]), period=2.0) == 0.5


# ---- the cubic family and PPoly (docs/api/interpolate.md §1, §3; milestone IP2)

IP = sd.interpolate
X, Y = [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 2.0]
P = IP.PPoly([[1.0, 2.0, 3.0]], X)
SP = IP.CubicSpline(X, Y)

CASES_IP2 = [
    (lambda: IP.CubicSpline([0.0, 2.0, 1.0], Y[:3]), ValueError, "CubicSpline(): 'x' must be strictly increasing"),
    (lambda: IP.CubicSpline([0.0, 1.0, 1.0], Y[:3]), ValueError, "CubicSpline(): 'x' must be strictly increasing"),
    (lambda: IP.CubicSpline([0.0, math.nan, 1.0], Y[:3]), ValueError, "CubicSpline(): 'x' must be finite"),
    (lambda: IP.CubicSpline([0.0], [1.0]), ValueError, "CubicSpline(): 'x' must have at least 2 points, not 1"),
    (lambda: IP.CubicSpline([[0.0, 1.0]], [1.0, 2.0]), ValueError, "CubicSpline(): 'x' must be 1-D, not 2-D"),
    (lambda: IP.CubicSpline([0.0, 1j], [1.0, 2.0]), TypeError, "CubicSpline(): 'x' must be real, not complex128"),
    (lambda: IP.CubicSpline(X, [0.0, 1.0, math.inf, 2.0]), ValueError, "CubicSpline(): 'y' must be finite"),
    (lambda: IP.CubicSpline(X, Y[:3]), ValueError,
     "CubicSpline(): 'y' must have as many points as 'x' (4) along axis 0, not 3"),
    (lambda: IP.CubicSpline(X, Y, axis=1), ValueError,
     "CubicSpline(): axis 1 is out of bounds for an array of dimension 1"),
    (lambda: IP.CubicSpline(X, Y, axis=0.0), TypeError, "CubicSpline(): 'axis' must be an int, not float"),
    (lambda: IP.CubicSpline(X, Y, bc_type="foo"), ValueError,
     "CubicSpline(): 'bc_type' must be 'not-a-knot', 'clamped', 'natural', 'periodic' or (order, value) pairs, "
     "not 'foo'"),
    (lambda: IP.CubicSpline(X, Y, bc_type="periodic"), ValueError,
     "CubicSpline(): 'y' must have equal first and last values along axis 0 for bc_type='periodic' "
     "(within rtol = atol = 1e-15)"),
    (lambda: IP.CubicSpline(X, Y, bc_type=("periodic", "natural")), ValueError,
     "CubicSpline(): bc_type='periodic' applies to both ends at once"),
    (lambda: IP.CubicSpline(X, Y, bc_type=((3, 0.0), "natural")), ValueError,
     "CubicSpline(): the derivative order of a boundary condition must be 1 or 2, not 3"),
    # a pair is always (left end, right end): (1, 0.0) is not one condition for both ends (scipy)
    (lambda: IP.CubicSpline(X, Y, bc_type=(1, 0.0)), ValueError,
     "CubicSpline(): a pair 'bc_type' is read as (left end, right end), and each end must be a condition name "
     "or an (order, value) pair, not 1"),
    (lambda: IP.CubicSpline(X, Y, bc_type=("natural", None)), ValueError,
     "CubicSpline(): a pair 'bc_type' is read as (left end, right end), and each end must be a condition name "
     "or an (order, value) pair, not NoneType"),
    (lambda: IP.CubicSpline(X, Y, bc_type=((1, [0.0, 1.0]), "natural")), ValueError,
     "CubicSpline(): a boundary value must have the shape (), not (2,)"),
    (lambda: IP.CubicSpline(X, Y, bc_type=("natural",) * 3), ValueError,
     "CubicSpline(): 'bc_type' must be one condition or a pair of them, not 3"),
    (lambda: IP.CubicSpline(X, Y, bc_type=None), TypeError,
     "CubicSpline(): 'bc_type' must be a str or a pair of conditions, not NoneType"),
    (lambda: IP.CubicSpline(X, Y, extrapolate="foo"), ValueError,
     "CubicSpline(): 'extrapolate' must be True, False, 'periodic' or None, not 'foo'"),
    (lambda: IP.CubicSpline(X, Y, extrapolate=1), TypeError,
     "CubicSpline(): 'extrapolate' must be True, False, 'periodic' or None, not int"),
    (lambda: IP.PchipInterpolator(X, [0, 1j, 0, 1]), ValueError,
     "PchipInterpolator(): 'y' must be real, not complex128"),
    (lambda: IP.Akima1DInterpolator(X, [0, 1j, 0, 1]), ValueError,
     "Akima1DInterpolator(): 'y' must be real, not complex128"),
    (lambda: IP.Akima1DInterpolator(X, Y, method="foo"), ValueError,
     "Akima1DInterpolator(): 'method' must be 'akima' or 'makima', not 'foo'"),
    (lambda: IP.CubicHermiteSpline(X, Y, Y[:3]), ValueError,
     "CubicHermiteSpline(): 'dydx' must have the shape of 'y' (4,), not (3,)"),
    (lambda: IP.CubicHermiteSpline(X, Y, [0.0, math.nan, 0.0, 0.0]), ValueError,
     "CubicHermiteSpline(): 'dydx' must be finite"),
    (lambda: IP.pchip_interpolate(X, Y, 0.5, der=-1), ValueError,
     "pchip_interpolate(): 'der' must be >= 0, not -1"),
    (lambda: IP.pchip_interpolate([0.0, 0.0], [1.0, 2.0], 0.5), ValueError,
     "pchip_interpolate(): 'x' must be strictly increasing"),
    (lambda: IP.PPoly([1.0, 2.0], X[:2]), ValueError, "PPoly(): 'c' must have at least 2 dimensions, not 1"),
    (lambda: IP.PPoly([[1.0, 2.0]], X), ValueError,
     "PPoly(): 'c' must have one piece per interval of 'x' (3) along axis 1, not 2"),
    (lambda: IP.PPoly([[1.0, 2.0]], [0.0, 1.0, 1.0]), ValueError,
     "PPoly(): 'x' must be strictly increasing or strictly decreasing"),
    (lambda: IP.PPoly([[1.0, 2.0]], [0.0, 1.0, math.inf]), ValueError, "PPoly(): 'x' must be finite"),
    (lambda: IP.PPoly(np.zeros((0, 3)), X), ValueError, "PPoly(): 'c' must have at least one coefficient per piece"),
    (lambda: IP.PPoly([[1.0, 2.0, 3.0]], X, axis=1), ValueError,
     "PPoly(): axis 1 is out of bounds for an array of dimension 1"),
    (lambda: P(0.5j), TypeError, "PPoly(): 'x' must be real, not complex"),
    (lambda: P([0.5j]), TypeError, "PPoly(): 'x' must be real, not complex128"),
    (lambda: P("a"), TypeError,
     "PPoly(): 'x' must be an sd.ndarray, a sequence of numbers, a numpy array or a frame Series or DataFrame, "
     "not str"),
    (lambda: P(["a"]), TypeError,
     "PPoly(): 'x' must be an sd.ndarray, a sequence of numbers, a numpy array or a frame Series or DataFrame, "
     "not list"),
    (lambda: P(0.5, nu=-1), ValueError, "PPoly(): 'nu' must be >= 0, not -1"),
    (lambda: P(0.5, nu=1.0), TypeError, "PPoly(): 'nu' must be an int, not float"),
    (lambda: P(0.5, nu=True), TypeError, "PPoly(): 'nu' must be an int, not bool"),
    (lambda: SP(0.5, extrapolate="no"), ValueError,
     "CubicSpline(): 'extrapolate' must be True, False, 'periodic' or None, not 'no'"),
    (lambda: P.derivative(nu=0.5), TypeError, "derivative(): 'nu' must be an int, not float"),
    (lambda: P.antiderivative(nu="1"), TypeError, "antiderivative(): 'nu' must be an int, not str"),
    (lambda: P.integrate("0", 1.0), TypeError, "integrate(): 'a' must be an int or a float, not str"),
    (lambda: P.integrate(0.0, 1j), TypeError, "integrate(): 'b' must be an int or a float, not complex"),
    (lambda: P.integrate(0.0, 1.0, extrapolate=3), TypeError,
     "integrate(): 'extrapolate' must be True, False, 'periodic' or None, not int"),
    (lambda: P.solve(math.inf), ValueError, "solve(): 'y' must be finite"),
    (lambda: P.roots(discontinuity=1), TypeError, "roots(): 'discontinuity' must be True or False, not int"),
    (lambda: P.extend([[1.0]], [2.5]), ValueError,
     "extend(): 'x' must lie beyond either end of self.x, without touching it"),
    (lambda: P.extend([[1.0, 2.0]], [4.0]), ValueError,
     "extend(): 'x' must have one point per piece of 'c' (2), not 1"),
    (lambda: P.extend([[1.0, 2.0]], [5.0, 4.0]), ValueError, "extend(): 'x' must be strictly increasing, as self.x"),
]  # fmt: skip


@pytest.mark.parametrize("k", range(len(CASES_IP2)))
def test_error_ip2(k):
    call, exc, msg = CASES_IP2[k]
    with pytest.raises(exc) as e:
        call()
    assert str(e.value) == msg


def test_call_shape_ip2():
    with pytest.raises(TypeError, match=r"^CubicSpline\(\): takes 2 positional arguments but 3 were given"):
        IP.CubicSpline(X, Y, 0)
    with pytest.raises(TypeError, match=r"^CubicSpline\(\): unexpected keyword argument 'check_finite'"):
        IP.CubicSpline(X, Y, check_finite=False)
    with pytest.raises(TypeError, match=r"^pchip_interpolate\(\): unexpected keyword argument 'foo'"):
        IP.pchip_interpolate(X, Y, 0.5, foo=1)
    with pytest.raises(TypeError):
        IP.PPoly([[1.0]], [0.0, 1.0], True)


# ---- the B-splines (docs/api/interpolate.md §4; milestone IP3)

T8 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
B = IP.BSpline(T8, [1.0, 2.0, 0.0, 1.0], 3)
X6 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
Y6 = [0.0, 1.0, 0.0, 2.0, 1.0, 0.0]

CASES_IP3 = [
    (lambda: IP.BSpline(T8, [1.0, 2.0, 3.0], 3), ValueError,
     "BSpline(): 'c' must have at least n = len(t) - k - 1 = 4 coefficients along axis 0, not 3"),
    (lambda: IP.BSpline(T8, [1.0] * 4, -1), ValueError, "BSpline(): 'k' must be >= 0, not -1"),
    (lambda: IP.BSpline(T8, [1.0] * 4, 3.0), TypeError, "BSpline(): 'k' must be an int, not float"),
    (lambda: IP.BSpline(T8[:5], [1.0], 3), ValueError,
     "BSpline(): 't' must have at least 2k + 2 = 8 knots for degree 3, not 5"),
    (lambda: IP.BSpline([0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0] * 4, 3), ValueError,
     "BSpline(): 't' must be nondecreasing"),
    (lambda: IP.BSpline([0.0] * 5 + [1.0] * 3, [1.0] * 4, 3), ValueError,
     "BSpline(): the base interval [t[k], t[n]] = [t[3], t[4]] must not be empty"),
    (lambda: IP.BSpline(T8[:-1] + [math.inf], [1.0] * 4, 3), ValueError, "BSpline(): 't' must be finite"),
    (lambda: IP.BSpline(T8, 1.0, 3), ValueError, "BSpline(): 'c' must have at least 1 dimension"),
    (lambda: B(0.5, nu=-1), ValueError, "BSpline(): 'nu' must be >= 0, not -1"),
    (lambda: B(0.5j), TypeError, "BSpline(): 'x' must be real, not complex"),
    (lambda: B(3.5, extrapolate=2), TypeError,
     "BSpline(): 'extrapolate' must be True, False, 'periodic' or None, not int"),
    (lambda: B.derivative(nu=4), ValueError, "derivative(): 'nu' must be at most the degree k = 3, not 4"),
    (lambda: B.insert_knot(9.0), ValueError,
     "insert_knot(): 'x' must lie in the base interval [t[k], t[n]] = [3.0, 4.0], not 9.0"),
    (lambda: B.insert_knot(3.5, m=0), ValueError, "insert_knot(): 'm' must be positive, not 0"),
    (lambda: IP.BSpline.basis_element([0.0]), ValueError, "basis_element(): 't' must have at least 2 knots, not 1"),
    (lambda: IP.BSpline.design_matrix([0.0, 9.0], T8, 3), ValueError,
     "design_matrix(): 'x' must lie in the base interval [t[k], t[n]] = [3.0, 4.0] when extrapolate=False"),
    (lambda: IP.BSpline.design_matrix([math.nan], T8, 3), ValueError, "design_matrix(): 'x' must be finite"),
    (lambda: IP.BSpline.design_matrix([[3.5]], T8, 3), ValueError, "design_matrix(): 'x' must be 1-D, not 2-D"),
    (lambda: IP.BSpline.design_matrix([3.5], T8, 3, extrapolate=None), TypeError,
     "design_matrix(): 'extrapolate' must be True, False or 'periodic', not NoneType"),
    (lambda: IP.BSpline.from_power_basis(IP.PchipInterpolator(X, Y)), TypeError,
     "from_power_basis(): 'pp' must be a CubicSpline, not PchipInterpolator"),
    (lambda: IP.BSpline.from_power_basis(SP, bc_type="foo"), ValueError,
     "from_power_basis(): 'bc_type' must be 'not-a-knot', 'natural', 'clamped' or 'periodic', not 'foo'"),
    (lambda: IP.PPoly.from_spline((T8, [1.0] * 3, 3)), ValueError,
     "from_spline(): 'c' must have at least n = len(t) - k - 1 = 4 coefficients along axis 0, not 3"),
    (lambda: IP.PPoly.from_spline(5), TypeError,
     "from_spline(): 'tck' must be a BSpline or a (t, c, k) tuple, not int"),
    (lambda: IP.make_interp_spline(X6, Y6, k=-1), ValueError, "make_interp_spline(): 'k' must be >= 0, not -1"),
    (lambda: IP.make_interp_spline(X6, Y6, k=5, bc_type="natural"), ValueError,
     "make_interp_spline(): the knots need 4 derivative conditions at the ends, not 1 + 1"),
    (lambda: IP.make_interp_spline(X6[:3], Y6[:3], k=3), ValueError,
     "make_interp_spline(): the knots need 1 derivative condition at the ends, not 0 + 0"),
    (lambda: IP.make_interp_spline(X6, Y6, k=1, bc_type="natural"), ValueError,
     "make_interp_spline(): k=1 takes no end conditions in 'bc_type' (unless 't' is given)"),
    (lambda: IP.make_interp_spline(X6, Y6, k=0, t=X6 + [5.0]), ValueError,
     "make_interp_spline(): k=0 takes neither 't' nor end conditions in 'bc_type'"),
    (lambda: IP.make_interp_spline(X6, Y6, bc_type="foo"), ValueError,
     "make_interp_spline(): 'bc_type' must be None, 'not-a-knot', 'natural', 'clamped', 'periodic' or a pair of "
     "end conditions, not 'foo'"),
    (lambda: IP.make_interp_spline(X6, Y6, bc_type=((1, 0.0), (1, 0.0))), ValueError,
     "make_interp_spline(): an end condition must be a list of (order, value) pairs"),
    (lambda: IP.make_interp_spline(X6, Y6, bc_type=([(4, 0.0)], [(1, 0.0)])), ValueError,
     "make_interp_spline(): a derivative order at x[0] must be in [0, k] = [0, 3], not 4"),
    (lambda: IP.make_interp_spline(X6, Y6[:5] + [3.0], bc_type="periodic"), ValueError,
     "make_interp_spline(): 'y' must have equal first and last values along axis 0 for bc_type='periodic' "
     "(within rtol = atol = 1e-15)"),
    (lambda: IP.make_interp_spline(X6, Y6, bc_type="periodic", t=[0.0] * 4 + [2.0, 3.0] + [5.0] * 4), ValueError,
     "make_interp_spline(): bc_type='periodic' makes its own knots: 't' must be None"),
    (lambda: IP.make_interp_spline(X6, Y6, t=[0.0] * 4 + [5.0] * 4), ValueError,
     "make_interp_spline(): 't' must have at least len(x) + k + 1 = 10 knots, not 8"),
    (lambda: IP.make_interp_spline(X6, Y6, t=[1.0] * 4 + [2.0, 3.0] + [5.0] * 4), ValueError,
     "make_interp_spline(): 'x' must lie within the knots: t[k] <= x[0] and x[-1] <= t[-k]"),
    (lambda: IP.make_interp_spline(X6, Y6[:5]), ValueError,
     "make_interp_spline(): 'y' must have as many points as 'x' (6) along axis 0, not 5"),
    (lambda: IP.make_interp_spline(X6, Y6[:5] + [math.nan]), ValueError, "make_interp_spline(): 'y' must be finite"),
    (lambda: IP.make_interp_spline([0.0, 2.0, 1.0], Y6[:3], k=1), ValueError,
     "make_interp_spline(): 'x' must be strictly increasing"),
    (lambda: IP.make_interp_spline(X6, Y6, bc_type=3), TypeError,
     "make_interp_spline(): 'bc_type' must be None, a str or a pair of end conditions, not int"),
]  # fmt: skip


@pytest.mark.parametrize("k", range(len(CASES_IP3)))
def test_error_ip3(k):
    call, exc, msg = CASES_IP3[k]
    with pytest.raises(exc) as e:
        call()
    assert str(e.value) == msg


def test_singular_collocation_is_linalg_error():
    with pytest.raises(sd.linalg.LinAlgError, match=r"^make_interp_spline\(\): the collocation matrix is singular"):
        IP.make_interp_spline(X6, Y6, t=[0.0] * 4 + [0.1, 0.2] + [5.0] * 4)  # Schoenberg-Whitney fails
    with pytest.raises(sd.linalg.LinAlgError, match="singular"):
        IP.make_interp_spline(X6, Y6, bc_type=([(0, 0.0)], [(1, 0.0)]))  # the value at x[0] twice
    with pytest.raises(TypeError, match=r"^make_interp_spline\(\): takes 2 positional arguments but 3 were given"):
        IP.make_interp_spline(X6, Y6, 3)
    with pytest.raises(TypeError, match=r"^BSpline\(\): unexpected keyword argument 'check_finite'"):
        IP.BSpline(T8, [1.0] * 4, 3, check_finite=True)


# ---- the polynomial and rational interpolants (docs/api/interpolate.md §5; milestone IP4)

BI = IP.BarycentricInterpolator([0.0, 1.0, 2.0], [1.0, 0.0, 3.0])
KI = IP.KroghInterpolator([0.0, 1.0, 2.0], [1.0, 0.0, 3.0])
AA = IP.AAA([0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 1.25, 1.5, 2.0, 2.5])
FH = IP.FloaterHormannInterpolator([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 0.0, 1.0, 3.0])
NOT_DATA = "must be an sd.ndarray, a sequence of numbers, a numpy array or a frame Series or DataFrame, not str"

CASES_IP4 = [
    (lambda: IP.BarycentricInterpolator([0.0, 1.0, 1.0], Y[:3]), ValueError,
     "BarycentricInterpolator(): 'xi' must be distinct: xi[2] = 1.0 repeats an earlier node"),
    (lambda: IP.BarycentricInterpolator([0.0, math.nan], [1.0, 2.0]), ValueError,
     "BarycentricInterpolator(): 'xi' must be finite"),
    (lambda: IP.BarycentricInterpolator([], []), ValueError,
     "BarycentricInterpolator(): 'xi' must have at least 1 point"),
    (lambda: IP.BarycentricInterpolator([0.0, 1j], [1.0, 2.0]), TypeError,
     "BarycentricInterpolator(): 'xi' must be real, not complex128"),
    (lambda: IP.BarycentricInterpolator([0.0, 1.0], Y[:3]), ValueError,
     "BarycentricInterpolator(): 'yi' must have as many points as 'xi' (2) along axis 0, not 3"),
    (lambda: IP.BarycentricInterpolator([0.0, 1.0], [1.0, 2.0], wi=[1.0]), ValueError,
     "BarycentricInterpolator(): 'wi' must have one weight per node (2,), not (1,)"),
    (lambda: IP.BarycentricInterpolator([0.0, 1.0], [1.0, 2.0], wi=[1.0, -1.0], axis=1), ValueError,
     "BarycentricInterpolator(): axis 1 is out of bounds for an array of dimension 1"),
    (lambda: IP.BarycentricInterpolator([0.0, 1.0], [1.0, 2.0], rng=0), TypeError,
     "BarycentricInterpolator(): unexpected keyword argument 'rng'"),
    (lambda: IP.BarycentricInterpolator([0.0, 1.0])(0.5), ValueError,
     "BarycentricInterpolator(): no values: give them with set_yi()"),
    (lambda: BI(0.5j), TypeError, "BarycentricInterpolator(): 'x' must be real, not complex"),
    (lambda: BI.derivative(0.5, der=-1), ValueError, "derivative(): 'der' must be >= 0, not -1"),
    (lambda: BI.derivative(0.5, der=1.0), TypeError, "derivative(): 'der' must be an int, not float"),
    (lambda: BI.derivatives(0.5, der="a"), TypeError, "derivatives(): 'der' must be an int, not str"),
    (lambda: BI.add_xi([1.0], [2.0]), ValueError,
     "add_xi(): 'xi' must be distinct: xi[3] = 1.0 repeats an earlier node"),
    (lambda: BI.add_xi([5.0]), ValueError, "add_xi(): 'yi' is needed: the object has values"),
    (lambda: BI.add_xi([5.0], [[1.0]]), ValueError, "add_xi(): 'yi' must have the shape (1,), not (1, 1)"),
    (lambda: BI.set_yi([1.0, 2.0]), ValueError,
     "set_yi(): 'yi' must have as many points as 'xi' (3) along axis 0, not 2"),
    (lambda: IP.KroghInterpolator([0.0, 1.0, 0.0], Y[:3]), ValueError,
     "KroghInterpolator(): a repeated value of 'xi' must follow its equals: xi[2] = 0.0 repeats an earlier node that "
     "is not next to it"),
    (lambda: KI(0.5j), TypeError, "KroghInterpolator(): 'x' must be real, not complex"),
    (lambda: KI.derivative(0.5, der=-2), ValueError, "derivative(): 'der' must be >= 0, not -2"),
    (lambda: IP.krogh_interpolate([0.0, 1.0], [1.0, 2.0], 0.5, der=-1), ValueError,
     "krogh_interpolate(): 'der' must be >= 0, not -1"),
    (lambda: IP.barycentric_interpolate([0.0, 0.0], [1.0, 2.0], 0.5), ValueError,
     "barycentric_interpolate(): 'xi' must be distinct: xi[1] = 0.0 repeats an earlier node"),
    (lambda: IP.barycentric_interpolate([0.0, 1.0], [1.0, 2.0], 0.5j), TypeError,
     "barycentric_interpolate(): 'x' must be real, not complex"),
    (lambda: IP.AAA([0.0, 1.0], [1.0]), ValueError, "AAA(): 'y' must have as many points as 'x' (2), not 1"),
    (lambda: IP.AAA([0.0, 1.0], [[1.0, 2.0]]), ValueError, "AAA(): 'y' must be 1-D, not 2-D"),
    (lambda: IP.AAA([0.0, math.inf], [1.0, 2.0]), ValueError, "AAA(): 'x' must be finite"),
    (lambda: IP.AAA([0.0, 1.0], [1.0, 2.0], rtol=-1.0), ValueError,
     "AAA(): 'rtol' must be finite and >= 0, not -1.0"),
    (lambda: IP.AAA([0.0, 1.0], [1.0, 2.0], rtol="a"), TypeError, "AAA(): 'rtol' must be an int or a float, not str"),
    (lambda: IP.AAA([0.0, 1.0], [1.0, 2.0], max_terms=0), ValueError, "AAA(): 'max_terms' must be 1 or more, not 0"),
    (lambda: IP.AAA([0.0, 1.0], [1.0, 2.0], max_terms=1.5), TypeError, "AAA(): 'max_terms' must be an int, not float"),
    (lambda: IP.AAA([0.0, 1.0], [1.0, 2.0], clean_up=1), TypeError, "AAA(): 'clean_up' must be True or False, not int"),
    (lambda: IP.AAA([0.0, 1.0], [1.0, 2.0], clean_up_tol=-1e-13), ValueError,
     "AAA(): 'clean_up_tol' must be finite and >= 0, not -1e-13"),
    (lambda: IP.AAA([0.0, 1.0], [math.nan, math.inf]), ValueError, "AAA(): no sample has a finite value"),
    (lambda: AA.clean_up(clean_up_tol="x"), TypeError, "clean_up(): 'clean_up_tol' must be an int or a float, not str"),
    (lambda: AA("a"), TypeError, "AAA(): 'z' " + NOT_DATA),
    (lambda: IP.FloaterHormannInterpolator([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], d=3), ValueError,
     "FloaterHormannInterpolator(): 'd' must satisfy 0 <= d < n, the number of finite samples (3), not 3"),
    (lambda: IP.FloaterHormannInterpolator([0.0, 1.0, 1.0], [1.0, 2.0, 3.0], d=1), ValueError,
     "FloaterHormannInterpolator(): 'points' must be distinct"),
    (lambda: IP.FloaterHormannInterpolator([0.0, 1j, 2.0], [1.0, 2.0, 3.0]), TypeError,
     "FloaterHormannInterpolator(): 'points' must be real, not complex128"),
    (lambda: IP.FloaterHormannInterpolator([0.0, 1.0, 2.0], [1.0, 2.0], d=1), ValueError,
     "FloaterHormannInterpolator(): 'values' must have as many points as 'points' (3) along axis 0, not 2"),
    (lambda: IP.FloaterHormannInterpolator([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], d=1.0), TypeError,
     "FloaterHormannInterpolator(): 'd' must be an int, not float"),
    (lambda: IP.FloaterHormannInterpolator([0.0, 1.0, 2.0], [[1.0, 2.0]] * 3, d=1).roots(),
     ValueError, "roots(): the values must be 1-D, not a batch of shape (2,)"),
    (lambda: FH("a"), TypeError, "FloaterHormannInterpolator(): 'z' " + NOT_DATA),
]  # fmt: skip


@pytest.mark.parametrize("k", range(len(CASES_IP4)))
def test_error_ip4(k):
    call, exc, msg = CASES_IP4[k]
    with pytest.raises(exc) as e:
        call()
    assert str(e.value) == msg


def test_call_shape_ip4():
    with pytest.raises(TypeError, match=r"^BarycentricInterpolator\(\): takes from 1 to 2 positional arguments but 3"):
        IP.BarycentricInterpolator([0.0, 1.0], [1.0, 2.0], 0)
    with pytest.raises(TypeError, match=r"^krogh_interpolate\(\): unexpected keyword argument 'foo'"):
        IP.krogh_interpolate([0.0, 1.0], [1.0, 2.0], 0.5, foo=1)
    with pytest.raises(TypeError, match=r"^AAA\(\): unexpected keyword argument 'cleanup_tol'"):
        IP.AAA([0.0, 1.0], [1.0, 2.0], cleanup_tol=1e-13)  # one spelling: clean_up_tol (IP32)


# ---------------------------------------------------------------- IP6: the scattered-data interpolants

Y2 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]]
D5 = [1.0, 2.0, 3.0, 4.0, 5.0]
RBF = IP.RBFInterpolator(Y2, D5)
KINDS = "'linear', 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric', 'inverse_multiquadric', " \
        "'inverse_quadratic' or 'gaussian'"  # fmt: skip

CASES_IP6 = [
    (lambda: IP.RBFInterpolator([0.0, 1.0], [1.0, 2.0]), ValueError,
     "RBFInterpolator(): 'y' must have 2 dimensions (points, coordinates), not 1"),
    (lambda: IP.RBFInterpolator(np.zeros((0, 2)), np.zeros(0)), ValueError,
     "RBFInterpolator(): 'y' must have at least 1 point"),
    (lambda: IP.RBFInterpolator([[0.0, math.nan]], [1.0]), ValueError, "RBFInterpolator(): 'y' must be finite"),
    (lambda: IP.RBFInterpolator([[0.0, 1j]], [1.0]), TypeError, "RBFInterpolator(): 'y' must be real, not complex128"),
    (lambda: IP.RBFInterpolator(Y2, D5[:4]), ValueError,
     "RBFInterpolator(): 'd' must have one row per data point (5), not 4"),
    (lambda: IP.RBFInterpolator(Y2, [1.0, 2.0, math.inf, 4.0, 5.0]), ValueError,
     "RBFInterpolator(): 'd' must be finite"),
    (lambda: IP.RBFInterpolator(Y2, D5, kernel="Gaussian", epsilon=1.0), ValueError,
     f"RBFInterpolator(): 'kernel' must be {KINDS}, not 'Gaussian'"),
    (lambda: IP.RBFInterpolator(Y2, D5, kernel=1), TypeError, f"RBFInterpolator(): 'kernel' must be {KINDS}, not int"),
    (lambda: IP.RBFInterpolator(Y2, D5, kernel="gaussian"), ValueError,
     "RBFInterpolator(): 'epsilon' must be given for kernel 'gaussian' (the default 1 is only for the scale-invariant "
     "linear, thin_plate_spline, cubic and quintic)"),
    (lambda: IP.RBFInterpolator(Y2, D5, epsilon=0.0), ValueError,
     "RBFInterpolator(): 'epsilon' must be finite and positive, not 0.0"),
    (lambda: IP.RBFInterpolator(Y2, D5, epsilon=math.inf), ValueError,
     "RBFInterpolator(): 'epsilon' must be finite and positive, not inf"),
    (lambda: IP.RBFInterpolator(Y2, D5, epsilon="1"), TypeError,
     "RBFInterpolator(): 'epsilon' must be an int or a float, not str"),
    (lambda: IP.RBFInterpolator(Y2, D5, smoothing=-1e-300), ValueError,
     "RBFInterpolator(): 'smoothing' must be finite and >= 0, not -1e-300"),
    (lambda: IP.RBFInterpolator(Y2, D5, smoothing=[0.0, 0.1, math.nan, 0.0, 0.0]), ValueError,
     "RBFInterpolator(): 'smoothing' must be finite and >= 0"),
    (lambda: IP.RBFInterpolator(Y2, D5, smoothing=[0.0, 0.1]), ValueError,
     "RBFInterpolator(): 'smoothing' must be a number or an array of shape (5,), not (2,)"),
    (lambda: IP.RBFInterpolator(Y2, D5, smoothing=True), TypeError,
     "RBFInterpolator(): 'smoothing' must be a number or an array, not bool"),
    (lambda: IP.RBFInterpolator(Y2, D5, degree=0), ValueError,
     "RBFInterpolator(): 'degree' must be at least 1 for kernel 'thin_plate_spline' (the system may not be solvable "
     "below it), not 0"),
    (lambda: IP.RBFInterpolator(Y2, D5, kernel="quintic", degree=-1), ValueError,
     "RBFInterpolator(): 'degree' must be at least 2 for kernel 'quintic' (the system may not be solvable below it), "
     "not -1"),
    (lambda: IP.RBFInterpolator(Y2, D5, kernel="gaussian", epsilon=1.0, degree=-2), ValueError,
     "RBFInterpolator(): 'degree' must be at least -1, not -2"),
    (lambda: IP.RBFInterpolator(Y2, D5, degree=1.0), TypeError,
     "RBFInterpolator(): 'degree' must be an int, not float"),
    (lambda: IP.RBFInterpolator(Y2, D5, degree=2), ValueError,
     "RBFInterpolator(): at least 6 data points are required for 'degree' 2 in 2 dimensions, not 5"),
    (lambda: IP.RBFInterpolator(Y2, D5, neighbors=2), ValueError,
     "RBFInterpolator(): at least 3 neighbors are required for 'degree' 1 in 2 dimensions, not 2"),
    (lambda: IP.RBFInterpolator(Y2, D5, neighbors=0), ValueError,
     "RBFInterpolator(): 'neighbors' must be at least 1, not 0"),
    (lambda: IP.RBFInterpolator(Y2, D5, neighbors=3.0), TypeError,
     "RBFInterpolator(): 'neighbors' must be an int, not float"),
    (lambda: IP.RBFInterpolator(Y2[:4] + [[1.0, 0.0]], D5), sd.linalg.LinAlgError,
     "RBFInterpolator(): data points 1 and 4 are equal and their smoothing is 0: the system is singular"),
    (lambda: IP.RBFInterpolator([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [1.0, 2.0, 3.0]), sd.linalg.LinAlgError,
     "RBFInterpolator(): the monomials of degree 1 at the data points do not have full column rank (2/3): the system "
     "is singular"),
    (lambda: IP.RBFInterpolator([[0.0], [1.0], [2.0], [3.0], [4.0]], D5, kernel="multiquadric", epsilon=1e-30),
     sd.linalg.LinAlgError, "RBFInterpolator(): the system of the data points is singular"),
    (lambda: RBF([0.5, 0.5, 0.5]), ValueError,
     "RBFInterpolator(): 'x' must have 2 entries on its last axis (the data points' dimension), not 3"),
    (lambda: RBF(0.5), ValueError, "RBFInterpolator(): 'x' must have at least 1 dimension (points of 2 coordinates)"),
    (lambda: RBF([0.5j, 0.5]), TypeError, "RBFInterpolator(): 'x' must be real, not complex128"),
    (lambda: IP.NearestNDInterpolator(Y2, D5[:3]), ValueError,
     "NearestNDInterpolator(): 'y' must have one row per data point (5), not 3"),
    (lambda: IP.NearestNDInterpolator([1.0, 2.0], [1.0, 2.0]), ValueError,
     "NearestNDInterpolator(): 'x' must have 2 dimensions (points, coordinates), not 1"),
    (lambda: IP.NearestNDInterpolator(Y2, D5, rescale=1), TypeError,
     "NearestNDInterpolator(): 'rescale' must be True or False, not int"),
    (lambda: IP.NearestNDInterpolator(Y2, D5)([0.5]), ValueError,
     "NearestNDInterpolator(): 'xi' must have 2 entries on its last axis (the data points' dimension), not 1"),
    (lambda: IP.NearestNDInterpolator(Y2, D5)(([0.5], [0.5], [0.5])), ValueError,
     "NearestNDInterpolator(): a tuple 'xi' must have 2 coordinate arrays, not 3"),
    (lambda: IP.NearestNDInterpolator(Y2, D5)(([0.5, 0.1], [0.5, 0.2, 0.3])), ValueError,
     "NearestNDInterpolator(): the arrays of a tuple 'xi' must broadcast together"),
]  # fmt: skip


@pytest.mark.parametrize("k", range(len(CASES_IP6)))
def test_error_ip6(k):
    call, exc, msg = CASES_IP6[k]
    with pytest.raises(exc) as e:
        call()
    assert str(e.value) == msg


def test_call_shape_ip6():
    with pytest.raises(TypeError, match=r"^RBFInterpolator\(\): takes 2 positional arguments but 3"):
        IP.RBFInterpolator(Y2, D5, 3)
    with pytest.raises(TypeError, match=r"^RBFInterpolator\(\): unexpected keyword argument 'workers'"):
        IP.RBFInterpolator(Y2, D5, workers=1)
    with pytest.raises(TypeError, match=r"^NearestNDInterpolator\(\): unexpected keyword argument 'tree_options'"):
        IP.NearestNDInterpolator(Y2, D5, tree_options={})  # IP31


# ---- the methods' call shapes (docs/api/interpolate.md §1.1): the method's name, and an object's call its class


def _shape_error(call):
    with pytest.raises(TypeError) as e:
        call()
    return str(e.value)


def test_method_call_shapes_name_the_method():
    cs = IP.CubicSpline([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    x6 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y6 = [v * v for v in x6]
    bs = IP.make_interp_spline(x6, y6)
    aaa, fh = IP.AAA(x6, y6), IP.FloaterHormannInterpolator(x6, y6)
    bary, krogh = IP.BarycentricInterpolator(x6, y6), IP.KroghInterpolator(x6, y6)
    for call, msg in (
        (lambda: cs.integrate(0, 1, 2), "integrate(): takes 2 positional arguments but 3 were given"),
        (lambda: cs.integrate(0, 1, foo=1), "integrate(): unexpected keyword argument 'foo'"),
        (lambda: cs.derivative(2), "derivative(): takes 0 positional arguments but 1 was given"),
        (lambda: cs.antiderivative(2), "antiderivative(): takes 0 positional arguments but 1 was given"),
        (lambda: cs.roots(foo=1), "roots(): unexpected keyword argument 'foo'"),
        (lambda: cs.solve(0.0, 1), "solve(): takes from 0 to 1 positional arguments but 2 were given"),
        (lambda: cs.extend(), "extend(): missing 2 required positional arguments: 'c' and 'x'"),
        (lambda: IP.PPoly.from_spline(bs, True), "from_spline(): takes 1 positional argument but 2 were given"),
        (lambda: IP.CubicSpline.from_spline(bs, True), "from_spline(): takes 1 positional argument but 2 were given"),
        (lambda: cs(0.5, foo=1), "CubicSpline(): unexpected keyword argument 'foo'"),
        (lambda: IP.PPoly(cs.c, cs.x)(0.5, 1), "PPoly(): takes 1 positional argument but 2 were given"),
        (lambda: bs(0.5, foo=1), "BSpline(): unexpected keyword argument 'foo'"),
        (lambda: bs.integrate(0, 1, 2), "integrate(): takes 2 positional arguments but 3 were given"),
        (lambda: bs.insert_knot(), "insert_knot(): missing 1 required positional argument: 'x'"),
        (lambda: IP.BSpline.basis_element(), "basis_element(): missing 1 required positional argument: 't'"),
        (lambda: IP.BSpline.design_matrix(x6, x6), "design_matrix(): missing 1 required positional argument: 'k'"),
        (lambda: IP.BSpline.from_power_basis(cs, 1),
         "from_power_basis(): takes 1 positional argument but 2 were given"),
        (lambda: aaa.roots(foo=1), "roots(): unexpected keyword argument 'foo'"),
        (lambda: aaa.poles(1), "poles(): takes 0 positional arguments but 1 was given"),
        (lambda: aaa.clean_up(1e-13), "clean_up(): takes 0 positional arguments but 1 was given"),
        (lambda: aaa(0.5, 1), "AAA(): takes 1 positional argument but 2 were given"),
        (lambda: fh.residues(1), "residues(): takes 0 positional arguments but 1 was given"),
        (lambda: fh(0.5, foo=1), "FloaterHormannInterpolator(): unexpected keyword argument 'foo'"),
        (lambda: bary.derivative(), "derivative(): missing 1 required positional argument: 'x'"),
        (lambda: bary.set_yi(), "set_yi(): missing 1 required positional argument: 'yi'"),
        (lambda: bary(0.5, foo=1), "BarycentricInterpolator(): unexpected keyword argument 'foo'"),
        (lambda: krogh.derivatives(0.5, 2), "derivatives(): takes 1 positional argument but 2 were given"),
        (lambda: krogh(), "KroghInterpolator(): missing 1 required positional argument: 'x'"),
        (lambda: RBF(0.5, foo=1), "RBFInterpolator(): unexpected keyword argument 'foo'"),
        (lambda: IP.NearestNDInterpolator(Y2, D5)(),
         "NearestNDInterpolator(): missing 1 required positional argument: 'xi'"),
        (lambda: IP.RegularGridInterpolator(([0.0, 1.0],), [1.0, 2.0])(0.5, foo=1),
         "RegularGridInterpolator(): unexpected keyword argument 'foo'"),
    ):  # fmt: skip
        assert _shape_error(call) == msg
    # an error raised inside a method keeps its own message
    with pytest.raises(TypeError, match=r"^integrate\(\): 'a' must be"):
        cs.integrate("a", 1)


# ---- frame inputs (docs/api/interpolate.md §1.2): a DataFrame is its columns side by side


def test_dataframe_points():
    F = sd.frame
    df = F.DataFrame({"x": [0.0, 1.0, 0.0, 1.0, 0.5], "y": [0.0, 0.0, 1.0, 1.0, 0.3]})
    pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.3]]
    d = [0.0, 1.0, 1.0, 2.0, 0.8]
    q = F.DataFrame({"x": [0.25, 0.75], "y": [0.5, 0.125]})
    qa = [[0.25, 0.5], [0.75, 0.125]]
    for cls in (IP.RBFInterpolator, IP.NearestNDInterpolator):
        want = cls(pts, d)(qa)
        assert cls(df, d)(q).tolist() == want.tolist()
        assert cls(df, F.Series(d))(qa).tolist() == want.tolist()
    g = IP.RegularGridInterpolator(([0.0, 1.0], [0.0, 1.0]), [[0.0, 1.0], [1.0, 2.0]])
    assert g(q).tolist() == g(qa).tolist()
    with pytest.raises(ValueError, match=r"^CubicSpline\(\): 'x' must be 1-D, not 2-D$"):
        IP.CubicSpline(df, [0.0, 1.0])
    with pytest.raises(TypeError, match=r"^RBFInterpolator\(\): 'y' must have numeric columns$"):
        IP.RBFInterpolator(F.DataFrame({"s": ["a", "b"]}), [0.0, 1.0])
