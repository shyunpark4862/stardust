"""sd.random against numpy's Generator: PCG64 (numpy's default) and Philox (the bit
generator sd.random uses). The 10^6-element calls of each method, with several argument
kinds (array bounds, sampling from an array, without replacement with
weights, a large sample without replacement, rows of a 2-D array); the small calls (one
element to 10^5, around the split threshold of 4096 elements) show the cost of a call.

numpy's generators run on one thread; each call draws from the same generator object
(the stream moves on between calls, the work per call does not change). No input repeats, so nothing is rotated: every call draws new numbers. One
element and 1000 elements have nothing to split and run at one thread only.

The case name is the call with its size as n (the id's size part holds it). The streams
differ from numpy's, so the results are not compared value by value (the case's Check)."""

from __future__ import annotations

from ..registry import Case, Check, Ref, count

N = 1_000_000
BIG_GROUP = "generator"
SMALL_GROUP = "small"


def _setup():
    import numpy as np

    import stardust as sd

    rng = np.random.default_rng(0)
    p = rng.random(1000)
    p /= p.sum()
    p5 = rng.random(100_000)
    p5 /= p5.sum()
    x = np.arange(N, dtype=np.float64)
    lo = rng.integers(-1000, 0, N)
    hi = lo + rng.integers(1, 1_000_000, N)
    a = rng.standard_normal(1000)
    rows = rng.standard_normal((100_000, 10))
    out = {
        "sd": sd.random.default_rng(1),
        "pcg": np.random.default_rng(1),
        "philox": np.random.Generator(np.random.Philox(1)),
    }
    for k, v in {"p": p, "p5": p5, "x": x, "lo": lo, "hi": hi, "a": a, "rows": rows}.items():
        out[k] = v
        out["s" + k] = sd.asarray(v)
    return out


# (group, name, n, sd call, numpy call on a Generator g, quick, note)
CALLS = [
    (BIG_GROUP, "random(n)", N, lambda d: d["sd"].random(N), lambda d, g: g.random(N), True, ""),
    (
        BIG_GROUP,
        "integers(0, 1000, shape=n)",
        N,
        lambda d: d["sd"].integers(0, 1000, shape=N),
        lambda d, g: g.integers(0, 1000, N),
        False,
        "",
    ),
    (
        BIG_GROUP,
        "integers(-2^62, 2^62 + 1, shape=n)",
        N,
        lambda d: d["sd"].integers(-(2**62), 2**62 + 1, shape=N),
        lambda d, g: g.integers(-(2**62), 2**62 + 1, N),
        False,
        "범위 2^63 + 1: 절반쯤 다시 뽑는다",
    ),
    (
        BIG_GROUP,
        "integers(lo, hi)",
        N,
        lambda d: d["sd"].integers(d["slo"], d["shi"]),
        lambda d, g: g.integers(d["lo"], d["hi"]),
        False,
        "원소마다 다른 범위, 10^6개",
    ),
    (
        BIG_GROUP,
        "choice(1000, shape=n)",
        N,
        lambda d: d["sd"].choice(1000, shape=N),
        lambda d, g: g.choice(1000, N),
        False,
        "",
    ),
    (
        BIG_GROUP,
        "choice(array, shape=n)",
        N,
        lambda d: d["sd"].choice(d["sa"], shape=N),
        lambda d, g: g.choice(d["a"], N),
        False,
        "float64 배열 1000개에서",
    ),
    (
        BIG_GROUP,
        "choice(1000, shape=n, p=...)",
        N,
        lambda d: d["sd"].choice(1000, shape=N, p=d["sp"]),
        lambda d, g: g.choice(1000, N, p=d["p"]),
        True,
        "",
    ),
    (
        BIG_GROUP,
        "choice(10^6, shape=n, replace=False)",
        1000,
        lambda d: d["sd"].choice(N, shape=1000, replace=False),
        lambda d, g: g.choice(N, 1000, replace=False),
        False,
        "",
    ),
    (
        BIG_GROUP,
        "choice(10^6, shape=n, replace=False)",
        500_000,
        lambda d: d["sd"].choice(N, shape=500_000, replace=False),
        lambda d, g: g.choice(N, 500_000, replace=False),
        True,
        "",
    ),
    (
        BIG_GROUP,
        "choice(10^5, shape=n, replace=False, p=...)",
        1000,
        lambda d: d["sd"].choice(100_000, shape=1000, replace=False, p=d["sp5"]),
        lambda d, g: g.choice(100_000, 1000, replace=False, p=d["p5"]),
        False,
        "",
    ),
    (BIG_GROUP, "permutation(n)", N, lambda d: d["sd"].permutation(N), lambda d, g: g.permutation(N), False, ""),
    (
        BIG_GROUP,
        "permutation(x)",
        N,
        lambda d: d["sd"].permutation(d["sx"]),
        lambda d, g: g.permutation(d["x"]),
        False,
        "float64 배열",
    ),
    (
        BIG_GROUP,
        "permutation(rows)",
        N,
        lambda d: d["sd"].permutation(d["srows"]),
        lambda d, g: g.permutation(d["rows"]),
        False,
        "행 10^5개를 섞는다",
    ),
    (SMALL_GROUP, "random()", 1, lambda d: d["sd"].random(), lambda d, g: g.random(), False, "float 하나"),
    (SMALL_GROUP, "random(n)", 1000, lambda d: d["sd"].random(1000), lambda d, g: g.random(1000), False, ""),
    (SMALL_GROUP, "random(n)", 10_000, lambda d: d["sd"].random(10_000), lambda d, g: g.random(10_000), True, ""),
    (
        SMALL_GROUP,
        "random(n)",
        100_000,
        lambda d: d["sd"].random(100_000),
        lambda d, g: g.random(100_000),
        False,
        "",
    ),
    (
        SMALL_GROUP,
        "integers(0, 1000, shape=n)",
        10_000,
        lambda d: d["sd"].integers(0, 1000, shape=10_000),
        lambda d, g: g.integers(0, 1000, 10_000),
        False,
        "",
    ),
]


def _bind(f, d):
    return lambda: f(d)


def _bind_g(f, key):
    return lambda d: lambda: f(d, d[key])


# The section title of each group in bench/random/speed.md (the group is the English key in
# the case ids).
TITLES = {
    "generator": "Generator 메서드",
    "small": "작은 크기",
}


SIZES = {"permutation(rows)": "(100000, 10)", "random()": ""}


def cases() -> list[Case]:
    out = []
    for group, name, n, sdf, npf, quick, note in CALLS:
        out.append(
            Case(
                area="random",
                group=group,
                name=name,
                size=SIZES.get(name, count(n)),
                n=n,
                setup=_setup,
                sd=lambda d, f=sdf: _bind(f, d),
                refs=(
                    Ref("numpy PCG64", _bind_g(npf, "pcg"), lib="numpy"),
                    Ref("numpy Philox", _bind_g(npf, "philox"), lib="numpy"),
                ),
                threads=(1,) if n <= 1000 and group == SMALL_GROUP else (1, 10),
                quick=quick,
                note=note,
                check=Check(skip="sd.random's streams differ from numpy's"),
            )
        )
    return out
