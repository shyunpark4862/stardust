"""sd.random against numpy's Generator: PCG64 (numpy's default) and Philox (the bit
generator sd.random uses), 10^6 elements. From bench/random/t3_speed.py.

numpy's generators run on one thread; each call draws from the same generator object,
as the old script did (the stream moves on between calls, the work per call does not
change)."""

from __future__ import annotations

from ..registry import Case, Ref

N = 1_000_000


def _setup():
    import numpy as np

    import stardust as sd

    p = np.random.default_rng(0).random(1000)
    p /= p.sum()
    x = np.arange(N, dtype=np.float64)
    return {
        "sd": sd.random.default_rng(1),
        "pcg": np.random.default_rng(1),
        "philox": np.random.Generator(np.random.Philox(1)),
        "p": p,
        "sp": sd.asarray(p),
        "x": x,
        "sx": sd.asarray(x),
    }


# (name, sd call, numpy call on a Generator g, quick)
CALLS = [
    ("random(10^6)", lambda d: d["sd"].random(N), lambda d, g: g.random(N), True),
    ("integers(0, 1000, shape=10^6)", lambda d: d["sd"].integers(0, 1000, shape=N),
     lambda d, g: g.integers(0, 1000, N), False),
    ("integers(-2^62, 2^62, shape=10^6)", lambda d: d["sd"].integers(-2**62, 2**62, shape=N),
     lambda d, g: g.integers(-2**62, 2**62, N), False),
    ("choice(1000, shape=10^6)", lambda d: d["sd"].choice(1000, shape=N), lambda d, g: g.choice(1000, N), False),
    ("choice(1000, shape=10^6, p=...)", lambda d: d["sd"].choice(1000, shape=N, p=d["sp"]),
     lambda d, g: g.choice(1000, N, p=d["p"]), True),
    ("choice(10^6, shape=1000, replace=False)", lambda d: d["sd"].choice(N, shape=1000, replace=False),
     lambda d, g: g.choice(N, 1000, replace=False), False),
    ("permutation(10^6)", lambda d: d["sd"].permutation(N), lambda d, g: g.permutation(N), False),
    ("permutation(float64 array 10^6)", lambda d: d["sd"].permutation(d["sx"]), lambda d, g: g.permutation(d["x"]),
     False),
]


def _bind(f, d):
    return lambda: f(d)


def _bind_g(f, key):
    return lambda d: (lambda: f(d, d[key]))


def cases() -> list[Case]:
    out = []
    for name, sdf, npf, quick in CALLS:
        out.append(Case(
            area="random", group="Generator 메서드", name=name, n=N,
            setup=_setup,
            sd=lambda d, f=sdf: _bind(f, d),
            refs=(Ref("numpy PCG64", _bind_g(npf, "pcg"), lib="numpy"),
                  Ref("numpy Philox", _bind_g(npf, "philox"), lib="numpy")),
            quick=quick,
        ))
    return out
