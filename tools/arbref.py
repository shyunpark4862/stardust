"""Rigorous reference values with Arb (python-flint), shared by tools/special_ref.py and
tools/stats_ref.py (docs/design/special.md §6, docs/design/stats.md §9).

Every reference is an Arb ball [mid +- rad] that contains the true value. A value is
settled when its radius is below 2^-TARGET_BITS of its magnitude: then its double-double
hi + lo (to_dd) is the true value rounded, with no heuristic in between (a rule such as
"two precisions agree to 30 digits" accepts wrong values that both precisions share: a
1 - p that loses p = 3e-123 at both). A value that does not settle by MAX_PREC raises
NotSettled and the point is dropped.

What Arb cannot certify is a truncation we choose ourselves (a sum cut after N terms):
such sums add a bound on their tail to the radius (add_error).

The ball arithmetic is C (FLINT), with no Python-side caches, so a long run's memory
stays flat. The project uses Arb alone for true values: no mpmath (slow, and its
quadrature caches grow without bound).
"""

import contextlib
import math
import struct
import sys
from contextlib import contextmanager

from flint import acb, arb, ctx

TARGET_BITS = 110  # hi + lo is exact to rounding (2^-106 relative) with a margin
START_PREC = 128
MAX_PREC = 1 << 13  # 8192 bits, ~2500 digits: past it a point is dropped (and recorded as dropped)
DOUBLE_TINY = 2.0**-1022  # the smallest normal double: settled's floor for double-double references


class NotSettled(ArithmeticError):
    """The ball did not shrink below the target by MAX_PREC (or holds a NaN)."""


@contextmanager
def precision(bits):
    old = ctx.prec
    ctx.prec = bits
    try:
        yield
    finally:
        ctx.prec = old


def settled(v, target=TARGET_BITS, floor=0, modulus_bits=None):
    """The ball v pins its value to 2^-target relative (exact balls always; a ball around
    zero only when it is exactly zero).

    floor (>= 0): the radius may instead reach 2^-target floor, an absolute bound for values
    near zero -- with floor = DOUBLE_TINY the ball fixes the value's rounding to a double-double
    even where the value is subnormal or underflows to 0 (special_ref passes it; the default
    keeps the purely relative rule).
    modulus_bits (acb only): each part may instead reach 2^-modulus_bits |v|, the accuracy a
    complex value is measured to (|error| / |f|), so a part far below the modulus need not
    settle on its own."""
    if isinstance(v, acb):
        if modulus_bits is not None:
            m = abs(v)
            if m.is_finite() and not m.is_nan():
                f = arb(m.mid()) * arb(2) ** -(modulus_bits - target)
                floor = max(floor, f)
        parts = (v.real, v.imag) if v.imag != 0 or not v.imag.is_exact() else (v.real,)
        return all(settled(p, target, floor) for p in parts)
    if v.is_nan():
        return False
    if not v.is_finite():
        return False
    if v.is_exact():
        return True
    rad = v.rad()
    mid = abs(v.mid())
    if mid == 0 and floor == 0:
        return False
    big = mid if mid > floor else arb(floor)
    return bool(rad * arb(2) ** target <= big)


def rigorous(fn, target=TARGET_BITS, start=START_PREC, max_prec=MAX_PREC, floor=0, modulus_bits=None):
    """fn() (an arb or acb, or a Python float for an exact special value) at doubling
    precision until settled (floor, modulus_bits: as settled). A NotSettled raised by fn
    below max_prec (a part of it that cannot finish at this precision) moves on to the next."""
    p = start
    while p <= max_prec:
        with precision(p):
            try:
                v = fn()
            except NotSettled:  # a part of fn gave up at this precision (a sum that cancels to nothing)
                p *= 2
                continue
        if isinstance(v, float):
            return v
        if settled(v, target, floor, modulus_bits):
            return v
        p *= 2
    raise NotSettled(fn)


def _even(c: float) -> bool:
    """The double's last bit is 0 (0 itself is even): the tie-breaking rule of rounding."""
    return struct.unpack("<q", struct.pack("<d", c))[0] & 1 == 0


def to_dd(v):
    """(hi, lo) of a settled arb: hi the double nearest the midpoint (ties to even, by the last
    bit of the double, subnormals and 0 included), lo the double nearest the rest. inf or 0
    outside the double range, as rounding gives."""
    if isinstance(v, float):
        return v, 0.0
    m = v.mid()
    with precision(2600):  # every double is exact here, and so are the differences below
        h0 = float(m)
        if math.isinf(h0):
            return h0, 0.0
        best, err = h0, abs(m - arb(h0))
        for c in (math.nextafter(h0, math.inf), math.nextafter(h0, -math.inf)):
            e = abs(m - arb(c))
            if e < err or (e == err and _even(c) and not _even(best)):
                best, err = c, e
        lo = float(m - arb(best))
    return best, lo


def add_error(v, err):
    """The ball v widened by |err| (the bound on a truncated tail)."""
    return v + arb(0, abs(arb(err).upper()))


def certify_root(g, x, rel=2.0**-TARGET_BITS, increasing=True):
    """x (a float or arb point) is the root of g to rel: g(x (1 - rel)) and g(x (1 + rel))
    have the signs of an increasing (or decreasing) g across its root. Raises NotSettled
    when the signs are not certain."""
    x = arb(x)
    lo, hi = (x * (1 - arb(rel)), x * (1 + arb(rel))) if x > 0 else (x * (1 + arb(rel)), x * (1 - arb(rel)))
    if x == 0:
        lo, hi = arb(-rel), arb(rel)
    glo, ghi = g(lo), g(hi)
    ok = (glo < 0 and ghi > 0) if increasing else (glo > 0 and ghi < 0)
    if not ok:
        raise NotSettled("root not bracketed")
    return x


def newton(g, dg, x0, iters=200):
    """A root of g by Newton from x0 on the balls' midpoints (the caller certifies it)."""
    x = arb(x0)
    for _ in range(iters):
        gx = g(x).mid()
        d = dg(x).mid()
        if d == 0:
            break
        step = gx / d
        x = (x - step).mid()
        if abs(step) <= abs(x) * arb(2) ** (-ctx.prec + 8):
            break
    return x


def solve_settled(g, dg, x0, increasing=True, rel_bits=TARGET_BITS, start=START_PREC, max_prec=MAX_PREC):
    """The root of g (a function of an arb x at the current precision), as an arb point
    certified to 2^-rel_bits relative by a sign change: Newton from x0, then the bracket;
    the precision doubles (Newton restarting from the last point) until both hold."""
    p = start
    x = arb(x0)
    while p <= max_prec:
        with precision(p):
            try:
                x = newton(g, dg, x)
                return certify_root(g, x, 2.0**-rel_bits, increasing)
            except NotSettled, ZeroDivisionError, ValueError:
                pass
        p *= 2
    raise NotSettled("root")


def newton_log(g, xdg, x0, iters=200, hi=None):
    """A positive root of g by Newton's method in s = log x from x0 > 0, on the balls'
    midpoints (the caller certifies it): x <- x exp(-g(x) / xdg(x)), xdg(x) = x g'(x). With
    g a log-space residual (log F(x) - log t) deep tails converge as fast as the middle.
    hi: an upper end the iterate stays below (halving the distance instead)."""
    x, prev = arb(x0), None
    noise = arb(2) ** (-ctx.prec // 2)
    for _ in range(iters):
        gx, d = g(x).mid(), xdg(x).mid()
        if d == 0 or not (gx.is_finite() and d.is_finite()):
            raise NotSettled("newton_log: no slope")
        step = abs(gx / d)  # the step in log x
        xn = (x * (-(gx / d)).exp()).mid()
        if hi is not None and xn >= hi:
            xn = ((x + hi) / 2).mid()
        if abs(xn - x) <= abs(xn) * arb(2) ** (-ctx.prec + 8):
            return xn
        if prev is not None and step < noise and step > prev / 2:  # no longer converging: g's rounding
            return xn
        x, prev = xn, step
    return x


def solve_log_settled(g, xdg, x0, increasing=True, hi=None, rel_bits=TARGET_BITS, start=START_PREC, max_prec=MAX_PREC):
    """solve_settled for a positive root, with Newton in log x (newton_log): the root of g as
    an arb point certified to 2^-rel_bits relative by a sign change."""
    p = start
    x = arb(x0)
    while p <= max_prec:
        with precision(p):
            try:
                x = newton_log(g, xdg, x, hi=hi)
                return certify_root(g, x, 2.0**-rel_bits, increasing)
            except NotSettled, ZeroDivisionError, ValueError:
                pass
        p *= 2
    raise NotSettled("root")


def geometric_tail(term, r):
    """A bound on sum_{i >= 1} term r^i (0 <= r < 1): the tail of a sum whose terms beyond
    the last one kept shrink at least by the ratio r each (add_error takes it)."""
    r = arb(r)
    if not r < 1:
        raise NotSettled("geometric_tail: ratio not below 1")
    return abs(arb(term)).upper() * (r / (1 - r)).upper()


def sign(fn, exact=None, exact_after=1024, start=START_PREC, max_prec=MAX_PREC):
    """The sign (-1, 0 or 1) of the value fn() computes (an arb at the current precision),
    at doubling precision until the ball leaves 0; 0 only for an exact zero ball. A value
    that may be exactly 0 (a discrete cdf equal to its target) keeps straddling 0: exact()
    (a Fraction, or None when it has none) decides it from exact_after bits on."""
    p = start
    while p <= max_prec:
        with precision(p):
            v = fn()
        if isinstance(v, float):
            if math.isnan(v):
                raise NotSettled("sign: NaN")
            return (v > 0) - (v < 0)
        if v.is_exact() and v.is_zero():
            return 0
        if v > 0:
            return 1
        if v < 0:
            return -1
        if exact is not None and p >= exact_after:
            e = exact()
            if e is not None:
                return (e > 0) - (e < 0)
        p *= 2
    raise NotSettled("sign")


def intersect(*vs):
    """The intersection of balls that all contain the same true value (several ways to
    compute it): the narrowest result; non-finite balls are skipped. Disjoint balls are a
    bug in one of the ways and raise."""
    out = None
    for v in vs:
        if v is None or not v.is_finite():
            continue
        if out is None:
            out = v
        elif not out.overlaps(v):
            raise ArithmeticError(f"intersect: disjoint balls {out} and {v}")
        else:
            out = out.intersection(v)
    if out is None:
        raise NotSettled("intersect: no finite ball")
    return out


def nstr(v, n):
    """v to n significant digits in the generators' coefficient format: fixed notation for a
    decimal exponent e with min(-(n // 3), -5) < e < n, else d.ddd...e+x / e-x; trailing zeros
    kept. v is an arb (its midpoint, taken exactly) or an exact number (fmpq, int), rounded
    at 4n + 128 bits first. The digits are the exact binary value m 2^p rounded half up at
    n digits (the next digit of the exact expansion 5 or more: round up)."""
    if not isinstance(v, arb):
        with precision(4 * n + 128):
            v = arb(v)
    man, p = (int(t) for t in v.mid().man_exp())
    if man == 0:
        return "0.0"
    sign = "-" if man < 0 else ""
    num, den = (abs(man) << p, 1) if p >= 0 else (abs(man), 1 << -p)

    def below(e):  # num / den < 10^e
        return num * 10**-e < den if e < 0 else num < den * 10**e

    e = len(str(num)) - len(str(den))  # floor(log10(num / den)) is e or e - 1
    while below(e):
        e -= 1
    while not below(e + 1):
        e += 1
    k = n - 1 - e  # q = floor(num / den 10^k) has n digits
    a, b = (num * 10**k, den) if k >= 0 else (num, den * 10**-k)
    q, r = divmod(a, b)
    if 2 * r >= b:
        q += 1
    digits = str(q)
    if len(digits) > n:  # 9.99...95 rounded up to 10.00...0
        digits, e = digits[:n], e + 1
    if min(-(n // 3), -5) < e < n:
        if e < 0:
            digits, split = "0" * -e + digits, 1
        else:
            split = e + 1
        e = 0
    else:
        split = 1
    out = sign + digits[:split] + "." + digits[split:]
    return out if e == 0 else out + ("e+" if e > 0 else "e") + str(e)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def logged(tool: str):
    """The generator's console output also into build/logs/<tool>/<stamp>.log (bench/cli.py)."""
    from bench import cli

    with open(cli.log_file(tool), "w") as fh, contextlib.redirect_stdout(_Tee(sys.stdout, fh)):
        yield


def provenance(target_bits=TARGET_BITS, extra=None) -> dict:
    """The provenance fields of a reference file (tests/reffile.py): the Arb that computed it
    (python-flint's version, and any other method after "; ") and the target."""
    import flint

    arb_ = f"python-flint {flint.__version__}" + (f"; {extra}" if extra else "")
    return {"arb": arb_, "target_bits": int(target_bits)}
