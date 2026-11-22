"""Sampling a curve t -> (x(t), y(t)) for drawing (docs/design/plot.md §4).

The curve is evaluated in batches: a first grid, then rounds that halve every interval where
the polyline still bends on the screen, where a value stops being finite (the edge of the
domain is closed in on), or where a jump has not been resolved. What is left after the last
round is read off: an interval that is as narrow as allowed and still jumps is a break (a pole
when the values run away on both sides, else a jump), and the line is cut there.

Nothing is guaranteed: a feature narrower than the first grid can be missed (PL14). The grid is
a little uneven on purpose, so that a periodic function is not sampled in step with itself.

Screen means a reference plot of W0 x H0 units (4:3); a unit is about a third of a pixel at
300 dpi on a one-column figure, so the tolerances below are well under a pixel.
"""

import math

from .. import _core as sd

W0, H0 = 1000.0, 750.0
N0 = 256  # intervals of the first grid
DEPTH = 10  # halvings of a first-grid interval at most
CAP = 40000  # samples at most
BEND = math.tan(math.radians(4.0))  # a turn sharper than this asks for a midpoint
MIN_LEN = 1.5  # ... when the interval is longer than this on the screen
JUMP = 0.02 * H0  # a narrowest interval rising more than this is a break
PHI = 0.6180339887498949


def first_grid(a, b, n=N0):
    """n + 1 points of [a, b]: uniform, the inner ones shifted by up to 0.15 of a step by a fixed
    low-discrepancy sequence."""
    i = sd.arange(0.0, n + 1.0)
    shift = (i * PHI - sd.floor(i * PHI) - 0.5) * 0.3
    shift = sd.where((i > 0.0) & (i < float(n)), shift, 0.0)
    t = a + (b - a) * (i + shift) / n
    return sd.where(i == float(n), b, sd.where(i == 0.0, a, t))  # the ends exactly


def robust_range(v):
    """(lo, hi, clipped_lo, clipped_hi) of the values of the first grid: their full range, unless
    a few of them run far away from the bulk (a pole nearby): then the bulk, widened by its own
    spread."""
    ok = sd.isfinite(v)
    if not bool(sd.any(ok)):
        return 0.0, 1.0, False, False
    w = v[ok]
    lo, hi = float(sd.min(w)), float(sd.max(w))
    if w.shape[0] < 8 or lo == hi:
        return lo, hi, False, False
    q10, q50, q90 = (float(q) for q in sd.quantile(w, sd.asarray([0.1, 0.5, 0.9])))
    spread = q90 - q10
    if spread <= 0.0:
        spread = max(abs(q50), 1.0) * 1e-9
    clip_hi = hi - q90 > 4.0 * spread
    clip_lo = q10 - lo > 4.0 * spread
    return (q10 - spread if clip_lo else lo), (q90 + spread if clip_hi else hi), clip_lo, clip_hi


class Curve:
    """A sampled curve: t, x, y (sd arrays, t increasing), the indices i of the intervals
    [t[i], t[i+1]] that are breaks, and what was found."""

    __slots__ = ("t", "x", "y", "breaks", "poles", "jumps", "undefined", "capped")


def sample(g, a, b, xr, yr):
    """Samples g(t) -> (x, y) on [a, b] for a view of x in xr = (lo, hi) and y in yr. g takes an
    sd array and returns two of its shape."""
    t = first_grid(a, b)
    x, y = g(t)
    return refine(g, t, x, y, xr, yr)


def _screen(x, y, xr, yr):
    sx = W0 / (xr[1] - xr[0]) if xr[1] > xr[0] else 1.0
    sy = H0 / (yr[1] - yr[0]) if yr[1] > yr[0] else 1.0
    return (x - xr[0]) * sx, (y - yr[0]) * sy


def refine(g, t, x, y, xr, yr):
    hmin = (float(t[-1]) - float(t[0])) / N0 / 2.0**DEPTH
    for _ in range(DEPTH + 2):
        n = t.shape[0]
        if n >= CAP:
            break
        u, v = _screen(x, y, xr, yr)
        fin = sd.isfinite(u) & sd.isfinite(v)
        du, dv = u[1:] - u[:-1], v[1:] - v[:-1]
        dt = t[1:] - t[:-1]
        both = fin[1:] & fin[:-1]
        # a bend at an inner vertex: the turn between its two segments
        cross = du[:-1] * dv[1:] - dv[:-1] * du[1:]
        dot = du[:-1] * du[1:] + dv[:-1] * dv[1:]
        bend = (sd.abs(cross) > BEND * dot) & both[:-1] & both[1:]
        false1 = sd.zeros(1, dtype=sd.bool)
        at = sd.concat([false1, bend]) | sd.concat([bend, false1])  # either end of the interval bends
        long_ = (du * du + dv * dv) > MIN_LEN * MIN_LEN
        # out of sight: both ends far above or far below the view
        hidden = ((v[1:] > 2.0 * H0) & (v[:-1] > 2.0 * H0)) | ((v[1:] < -H0) & (v[:-1] < -H0))
        want = at & long_ & both & ~hidden
        want = want | (fin[1:] ^ fin[:-1])  # the edge of the domain
        want = want & (dt > hmin)
        if not bool(sd.any(want)):
            break
        tm = (t[:-1] + t[1:])[want] * 0.5
        if n + tm.shape[0] > CAP:
            tm = tm[: CAP - n]
        xm, ym = g(tm)
        t = sd.concat([t, tm])
        order = sd.argsort(t, stable=True)
        t, x, y = sd.take(t, order), sd.take(sd.concat([x, xm]), order), sd.take(sd.concat([y, ym]), order)
    c = Curve()
    c.t, c.x, c.y = t, x, y
    c.capped = t.shape[0] >= CAP
    u, v = _screen(x, y, xr, yr)
    fin = sd.isfinite(u) & sd.isfinite(v)
    both = fin[1:] & fin[:-1]
    dt = t[1:] - t[:-1]
    dv = sd.abs(v[1:] - v[:-1])
    du = sd.abs(u[1:] - u[:-1])
    narrow = dt <= 2.0 * hmin
    # a narrowest interval that still jumps on the screen
    brk = both & narrow & ((dv > JUMP) | (du > JUMP))
    idx = [int(i) for i in sd.nonzero(brk)[0].tolist()]
    c.breaks, c.poles, c.jumps = idx, [], []
    last_t, last_kind = None, None
    for i in idx:
        v0, v1 = float(v[i]), float(v[i + 1])
        far = (v0 > 3.0 * H0 or v0 < -2.0 * H0) and (v1 > 3.0 * H0 or v1 < -2.0 * H0)
        at = 0.5 * (float(t[i]) + float(t[i + 1]))
        kind = c.poles if far else c.jumps
        # neighbouring narrow intervals around one pole or jump are one finding
        if last_t is not None and at - last_t <= 64.0 * hmin:
            if far and last_kind is c.jumps:
                c.poles.append(c.jumps.pop())
                last_kind = c.poles
        else:
            kind.append(at)
            last_kind = kind
        last_t = at
    # runs of t where the curve is not defined
    c.undefined = []
    mask = bytes(memoryview(fin))
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        c.undefined.append((float(t[max(i - 1, 0)]), float(t[min(j, n - 1)])))
        i = j
    return c


def with_breaks(c):
    """(x, y) of a Curve with a NaN point put into every break, so that a line stops there."""
    return tuple(cut(c, [c.x, c.y]))


def cut(c, arrays):
    """The arrays (one value per sample of the Curve c) with a NaN put into every break."""
    if not c.breaks:
        return list(arrays)
    out, last = [[] for _ in arrays], 0
    nan = sd.full((1,), math.nan)
    # neighbouring break intervals (the narrow ones around one pole) are one cut: the samples
    # between them belong to neither side
    groups = [[c.breaks[0], c.breaks[0]]]
    for i in c.breaks[1:]:
        if i <= groups[-1][1] + 1:
            groups[-1][1] = i
        else:
            groups.append([i, i])
    for first, end in groups:
        for k, a in enumerate(arrays):
            out[k] += [a[last : first + 1], nan]
        last = end + 1
    for k, a in enumerate(arrays):
        out[k].append(a[last:])
    return [sd.concat(parts) for parts in out]


def oscillation(c, xr, yr, cols=2000):
    """Runs of screen columns in which the curve reverses several times: [(x array, lower y,
    upper y)] envelopes to draw as bands, and the mask of the samples inside them. A column is
    1/`cols` of the view's width (finer than a pixel of most outputs)."""
    x, y = c.x, c.y
    n = x.shape[0]
    if n < 8:
        return [], None
    xs = [float(q) for q in x.tolist()]
    ys = [float(q) for q in y.tolist()]
    w = (xr[1] - xr[0]) / cols
    if not w > 0.0:
        return [], None
    counts, lo, hi = {}, {}, {}
    prev = 0.0
    for i in range(1, n):
        a, b = ys[i - 1], ys[i]
        if not (math.isfinite(a) and math.isfinite(b)):
            prev = 0.0
            continue
        d = b - a
        k = int((xs[i] - xr[0]) / w)
        if d * prev < 0.0:
            counts[k] = counts.get(k, 0) + 1
        if d != 0.0:
            prev = d
        lo[k] = min(lo.get(k, b), a, b)
        hi[k] = max(hi.get(k, b), a, b)
    busy = sorted(k for k, m in counts.items() if m >= 3 and (hi[k] - lo[k]) * H0 / max(yr[1] - yr[0], 1e-300) > 2.0)
    if not busy:
        return [], None
    runs, start, last = [], busy[0], busy[0]
    for k in busy[1:]:
        if k > last + 2:
            runs.append((start, last))
            start = k
        last = k
    runs.append((start, last))
    bands, inside = [], [False] * n
    for k0, k1 in runs:
        if k1 - k0 < 2:
            continue  # a column or two: the line itself shows it
        ks = [k for k in range(k0, k1 + 1) if k in lo]
        bx = [xr[0] + (k + 0.5) * w for k in ks]
        bands.append((sd.asarray(bx), sd.asarray([lo[k] for k in ks]), sd.asarray([hi[k] for k in ks])))
        x0, x1 = xr[0] + k0 * w, xr[0] + (k1 + 1) * w
        for i in range(n):
            if x0 <= xs[i] <= x1:
                inside[i] = True
    return bands, inside
