"""Tick positions and their labels (docs/design/plot.md §3.4).

Linear axes use the extended Wilkinson algorithm of Talbot, Lin and Hanrahan (2010): among the
labelings step * q * 10^z, q from (1, 5, 2, 2.5, 4, 3), the one scoring best on simplicity (a
round step that includes zero), coverage of the range, the wanted density and nothing else is
chosen, here among those that lie inside the range. Log axes take the decades.

Labels are numbers in math mode ("$-1.5$"), all of one axis with the same number of decimals;
very large or small values share a power of ten shown once beside the axis.
"""

import functools
import math

_Q = (1.0, 5.0, 2.0, 2.5, 4.0, 3.0)
_W = (0.25, 0.2, 0.5, 0.05)


def _simplicity(i, start, j, step):
    eps = 1e-10
    rem = math.fmod(start, step)
    zero = 1.0 if (abs(rem) < eps * step or abs(step - abs(rem)) < eps * step) and start <= 0.0 else 0.0
    return 1.0 - i / (len(_Q) - 1.0) - j + zero


def _coverage(lo, hi, a, b):
    r = 0.1 * (hi - lo)
    return 1.0 - 0.5 * ((hi - b) ** 2 + (lo - a) ** 2) / (r * r)


def _density(k, m, lo, hi, a, b):
    r = (k - 1.0) / (b - a)
    rt = (m - 1.0) / (max(b, hi) - min(lo, a))
    return 2.0 - max(r / rt, rt / r)


def linear(lo, hi, m):
    """Tick values inside [lo, hi] (lo < hi, finite), about m of them (a new list each call)."""
    try:
        return list(_linear(float(lo), float(hi), int(m)))
    except ArithmeticError, ValueError:  # a range at the edge of the doubles: its ends
        return [float(lo), float(hi)]


@functools.lru_cache(maxsize=1024)
def _linear(lo, hi, m):
    best, best_score = None, -1e30
    span = hi - lo
    for j in (1, 2):
        for i, q in enumerate(_Q):
            for k in range(max(2, m // 2), min(12, 2 * m) + 1):
                # the largest step of this (j, q) with k ticks inside the range, and the next finer
                zmax = math.floor(math.log10(span / ((k - 1) * j * q)) + 1e-9)
                for z in (zmax, zmax - 1):
                    step = j * q * 10.0**z
                    s0 = math.ceil(lo / step - 1e-9)
                    s1 = math.floor(hi / step + 1e-9) - (k - 1)
                    for s in range(s0, min(s1, s0 + 20) + 1):
                        a, b = s * step, (s + k - 1) * step
                        score = (
                            _W[0] * _simplicity(i, a, j, step)
                            + _W[1] * _coverage(lo, hi, a, b)
                            + _W[2] * _density(k, m, lo, hi, a, b)
                            + _W[3]
                        )
                        if score > best_score + 1e-12:
                            best_score, best = score, (a, step, k)
    if best is None:  # a range too narrow for two ticks of any step: its ends
        return (lo, hi)
    a, step, k = best
    decimals = max(0, -int(math.floor(math.log10(step) + 1e-9)) + 2)
    return tuple(round(a + n * step, decimals) + 0.0 for n in range(k))


def log(lo, hi, m):
    """(major, minor) tick values of a log axis over [lo, hi] (0 < lo < hi): the decades (every
    n-th when there are many), or the linear choice when the range holds fewer than two."""
    e0, e1 = math.ceil(math.log10(lo) - 1e-9), math.floor(math.log10(hi) + 1e-9)
    if e1 - e0 < 1:
        return linear(lo, hi, m), []
    n = max(1, math.ceil((e1 - e0 + 1) / max(m, 2)))
    major = [10.0**e for e in range(e0, e1 + 1) if (e - e0) % n == 0]
    minor = []
    if n == 1:
        for e in range(e0 - 1, e1 + 1):
            minor += [d * 10.0**e for d in range(2, 10) if lo <= d * 10.0**e <= hi]
    return major, minor


def _decimals(values, scale):
    for d in range(0, 12):
        if all(abs(round(v / scale, d) - v / scale) <= 1e-9 * max(1.0, abs(v / scale)) for v in values):
            return d
    return 12


def labels(values):
    """(labels, multiplier): the tick values as math-mode numbers with common decimals. When the
    values are very large or small, the labels are mantissas and multiplier is the shared
    "$\\times10^{e}$" (else None)."""
    big = max((abs(v) for v in values), default=0.0)
    e = 0
    if big != 0.0:
        ex = math.floor(math.log10(big) + 1e-12)
        if ex >= 5 or ex <= -4:
            e = ex
    scale = 10.0**e
    d = _decimals(values, scale)
    out = []
    for v in values:
        s = f"{v / scale:.{d}f}"
        if float(s) == 0.0:
            s = s.lstrip("-")
        out.append(f"${s}$")
    return out, (rf"$\times10^{{{e}}}$" if e else None)


def log_labels(values):
    """Powers of ten as "$10^{k}$"; other values as plain numbers."""
    out = []
    for v in values:
        k = round(math.log10(v))
        if abs(v - 10.0**k) <= 1e-9 * v:
            out.append(f"$10^{{{k}}}$")
        else:
            out.append(labels([v])[0][0])
    return out
