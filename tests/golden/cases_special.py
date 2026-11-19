"""Golden cases for sd.special (tools/golden.py has the contract).

Every function of sd.special.__all__. The positional parameters come from the signature
blocks of docs/api/special.md, and each parameter's name gives the kind of values it gets
(KIND, with per-function overrides in OVERRIDE): real x, probability p, positive shape s,
integer-valued count i, real order o, and a few ranges (u, u2, lp, zu).

Per elementwise function:
  grid      every combination of the kinds' special values (infinities, NaN, signed zeros,
            subnormals, domain ends), broadcast over one axis per argument: at most 1024 results
  random    1000 points spread over the typical domain (log-uniform shapes, all magnitudes)
  large     2**15 + 5 points (4101 for HEAVY; sha): the multithreaded loop over many tasks
  scalar, ints, pyint, bool, list, strided, broadcast, out, empty, complex, and the call errors
The reductions (logsumexp, softmax, log_softmax), airy, lambertw's branches, zeta's two forms
and the complex functions get their own cases.
"""

import math
import pathlib
import re
import zlib

import numpy as np

from tests._api_doc import signature_blocks

NAN, INF = math.nan, math.inf
API = pathlib.Path(__file__).resolve().parents[2] / "docs" / "api" / "special.md"

KIND = {
    "x": "x",
    "z": "x",
    "t": "x",
    "y": "x",
    "f": "x",
    "p": "p",
    "q": "p",
    "a": "s",
    "b": "s",
    "c": "s",
    "df": "s",
    "dfn": "s",
    "dfd": "s",
    "nc": "s",
    "v": "s",
    "m": "s",
    "n": "i",
    "k": "i",
}
OVERRIDE = {
    "erfinv": "u",
    "erfcinv": "u2",
    "ndtri_exp": "lp",
    "ellipk": "x",
    "ellipe": "x",
    "binom": "oo",
    "beta": "oo",
    "betaln": "oo",
    "zeta": "xx",
    "xlogy": "xx",
    "xlog1py": "xx",
    "rel_entr": "xx",
    "jv": "ox",
    "yv": "ox",
    "iv": "ox",
    "kv": "ox",
    "jve": "ox",
    "yve": "ox",
    "ive": "ox",
    "kve": "ox",
    "hyp1f1": "oox",
    "hyp2f1": "ooozu",
    "nctdtr": "sox",
    "nctdtrc": "sox",
    "nctdtrit": "sop",
    "pdtr": "is",
    "pdtrc": "is",
}
REDUCTIONS = ("logsumexp", "softmax", "log_softmax")
COMPLEX = ("erf", "erfc", "erfcx", "lambertw", "wofz", "loggamma")

# Special values per kind, NaN last (so every subset keeps it).
GRID = {
    "x": [
        -INF,
        -1e300,
        -1e10,
        -745.2,
        -710.0,
        -100.0,
        -27.5,
        -10.0,
        -6.0,
        -3.5,
        -2.0,
        -1.0,
        -0.75,
        -0.5,
        -1e-5,
        -1e-300,
        -5e-324,
        -0.0,
        0.0,
        5e-324,
        1e-300,
        1e-200,
        1e-16,
        1e-8,
        1e-5,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9999,
        1.0,
        1.0000000000000002,
        1.5,
        2.0,
        2.5,
        3.0,
        5.0,
        7.5,
        10.0,
        20.0,
        26.5,
        50.0,
        100.0,
        171.5,
        300.0,
        709.7,
        710.0,
        1e5,
        1e10,
        1e300,
        INF,
        NAN,
    ],
    "p": [
        -0.5,
        0.0,
        5e-324,
        1e-300,
        1e-100,
        1e-20,
        1e-10,
        1e-5,
        0.001,
        0.01,
        0.1,
        0.25,
        0.3,
        0.5,
        0.7,
        0.75,
        0.9,
        0.99,
        0.999,
        1 - 1e-10,
        1 - 2**-53,
        1.0,
        1.5,
        NAN,
    ],
    "s": [
        -1.5,
        -1.0,
        -0.0,
        0.0,
        5e-324,
        1e-300,
        1e-10,
        1e-3,
        0.5,
        1.0,
        2.0,
        3.5,
        10.0,
        100.0,
        1e5,
        1e10,
        1e300,
        INF,
        NAN,
    ],
    "i": [-1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 170.0, 1e4, 1e9, 2.5, INF, NAN],
    "o": [-INF, -100.5, -10.5, -2.0, -1.0, -0.5, -0.0, 0.0, 1e-10, 0.5, 1.0, 2.0, 10.5, 100.0, 1e4, INF, NAN],
    "u": [-1.5, -1.0, -0.999999, -0.5, -1e-10, -0.0, 0.0, 1e-300, 0.3, 0.9, 1 - 1e-16, 1.0, NAN],
    "u2": [-0.5, 0.0, 5e-324, 1e-300, 1e-10, 0.3, 1.0, 1.5, 2 - 1e-10, 2.0, 2.5, NAN],
    "lp": [-INF, -1e10, -745.0, -100.0, -10.0, -1.0, -0.5, -1e-10, -5e-324, -0.0, 0.0, 0.5, NAN],
    "zu": [-INF, -10.0, -1.0, -0.999, -0.5, -0.0, 0.0, 0.5, 0.9, 0.999999, 1.0, 1.5, 10.0, NAN],
}
SCALAR = {"x": 0.7, "p": 0.3, "s": 2.5, "i": 3.0, "o": 1.5, "u": 0.4, "u2": 0.6, "lp": -0.7, "zu": 0.4}
SUB = {1: None, 2: 15, 3: 8, 4: 5}  # grid values per argument by arity (None: all)
# The loop goes parallel at 2**15 adds of work (csrc/special/cost.hpp): 2**15 + 5 elements for
# a cheap function, a few thousand for these heavy ones (weight 256: parallel from 128 elements).
HEAVY = {
    "nctdtr",
    "nctdtrc",
    "nctdtrit",
    "ncfdtr",
    "ncfdtrc",
    "ncfdtri",
    "chndtr",
    "chndtrc",
    "chndtrix",
    "fdtri",
    "stdtrit",
    "betaincinv",
    "betainccinv",
    "hyp2f1",
}


def documented():
    """{name: [positional parameter names]} from the signature blocks of docs/api/special.md."""
    out = {}
    for block in signature_blocks(API.read_text()):
        for line in block:
            m = re.match(r"([A-Za-z_0-9]+)\((.*?)\)\s*(#.*)?$", line.strip())
            if m:
                pos = m.group(2).split("/")[0]
                out[m.group(1)] = [p.split("=")[0].strip() for p in pos.split(",") if p.strip()]
    return out


def kinds_of(name, params):
    ov = OVERRIDE.get(name)
    if ov is None:
        return [KIND[p] for p in params]
    if len(params) == 1:
        return [ov]
    out, i = [], 0
    for _ in params:  # kinds are one letter except u2, lp, zu
        for k in ("u2", "lp", "zu"):
            if ov.startswith(k, i):
                out.append(k)
                i += len(k)
                break
        else:
            out.append(ov[i])
            i += 1
    return out


def subset(g, k):
    if k is None or len(g) <= k:
        return g
    return [g[round(i * (len(g) - 1) / (k - 1))] for i in range(k)]


def seed_of(name):
    return zlib.crc32(name.encode())


def domain(r, kind, n):
    if kind == "x":
        return r.standard_normal(n) * 10.0 ** r.uniform(-3, 2.5, n)
    if kind == "p":
        return r.uniform(0, 1, n)
    if kind == "s":
        return 10.0 ** r.uniform(-3, 3, n)
    if kind == "i":
        return r.integers(0, 60, n).astype(np.float64)
    if kind == "o":
        return r.uniform(-10, 30, n)
    if kind == "u":
        return r.uniform(-1, 1, n)
    if kind == "u2":
        return r.uniform(0, 2, n)
    if kind == "lp":
        return -(10.0 ** r.uniform(-5, 3, n))
    if kind == "zu":
        return r.uniform(-1, 1, n)
    raise ValueError(kind)


def cases(L):
    S = L.sd.special
    sig = documented()
    names = [n for n in S.__all__ if n not in REDUCTIONS]
    for name in names:
        f = getattr(S, name)
        params = sig[name]
        kinds = kinds_of(name, params)
        k = len(kinds)
        yield from elementwise(L, name, f, kinds, k)
    yield from reductions(L)
    yield from special_forms(L)


def elementwise(L, name, f, kinds, k):
    sd = L.sd
    seed = seed_of(name)
    has_out = name != "airy"

    def grid():
        sub = SUB[k]
        arrs = []
        for i, kind in enumerate(kinds):
            g = np.array(subset(GRID[kind], sub))
            shape = [1] * k
            shape[i] = g.size
            arrs.append(L.arr(g.reshape(shape)))
        return arrs

    def rand(n, s=0):
        r = L.rng(seed + s)
        return [L.arr(domain(r, kind, n)) for kind in kinds]

    yield f"{name} grid", lambda: f(*grid())
    yield f"{name} random", lambda: f(*rand(1000))
    yield f"{name} large", lambda: f(*rand(4101 if name in HEAVY else 2**15 + 5, 1))
    yield f"{name} scalar", lambda: f(*[SCALAR[kd] for kd in kinds])
    yield f"{name} scalar nan", lambda: f(*[NAN] * k)
    yield f"{name} ints", lambda: f(*[L.arr(np.array([0, 1, 2, 3, 5, -1, 40], dtype=np.int64)) for _ in kinds])
    yield f"{name} pyint", lambda: f(*[2] * k)
    yield f"{name} pyint big", lambda: f(*[2**63] * k)
    yield f"{name} bool", lambda: f(*[L.arr(np.array([True, False, True])) for _ in kinds])
    yield f"{name} list", lambda: f(*[[0.5, 1.0]] * k)
    yield f"{name} str", lambda: f(*["a"] * k)
    yield f"{name} err no args", lambda: f()
    yield f"{name} err extra arg", lambda: f(*[0.5] * (k + 1))

    def strided():
        r = L.rng(seed + 2)
        xs = []
        for kind in kinds:
            a = L.arr(domain(r, kind, 30 * 40).reshape(30, 40))
            xs.append(a[::2, ::-3].T)
        return f(*xs)

    yield f"{name} strided", strided

    def broadcast():
        r = L.rng(seed + 3)
        xs = []
        for i, kind in enumerate(kinds):
            shape = [1] * k
            shape[i] = 3 + i
            xs.append(L.arr(domain(r, kind, 3 + i).reshape(shape)))
        return f(*xs)

    yield f"{name} broadcast", broadcast
    yield f"{name} empty", lambda: f(*[sd.zeros((0, 3))] * k)
    yield f"{name} complex", lambda: f(*[sd.asarray(np.array([0.5 + 0.25j, -1.0 + 0j]))] * k)
    yield f"{name} pycomplex", lambda: f(*[0.5 + 0.5j] * k)
    if has_out:

        def with_out(dtype):
            o = sd.full((7,), -1.0, dtype=dtype)
            args = rand(7, 4)
            r = f(*args, out=o)
            return {"same": r is o, "r": r, "out": o}

        yield f"{name} out", lambda: with_out(sd.float64)
        yield f"{name} out int", lambda: with_out(sd.int64)
        yield f"{name} out bad shape", lambda: f(*rand(7, 4), out=sd.zeros((6,)))
        yield f"{name} out not array", lambda: f(*rand(7, 4), out=[0.0] * 7)


def reductions(L):
    sd = L.sd
    S = sd.special
    r = L.rng(11)
    base = r.standard_normal((6, 7)) * 30
    special = np.array(
        [
            [0.0, -INF, INF, NAN, 1e308, -1e308, 5e-324],
            [-INF, -INF, -INF, -INF, -INF, -INF, -INF],
            [INF, 1.0, 2.0, 3.0, -0.0, 0.0, 700.0],
            [745.0, 746.0, -745.0, 1e-300, 2.0, 2.0, 2.0],
        ]
    )
    for name in ("logsumexp", "softmax", "log_softmax"):
        f = getattr(S, name)
        for label, src in (("base", base), ("special", special)):
            for axis in (None, 0, 1, -1, (0, 1), (1, 0)):
                yield f"{name} {label} axis={axis}", lambda f=f, src=src, axis=axis: f(L.arr(src), axis=axis)
        yield f"{name} 1d", lambda f=f: f(L.arr(base[0]))
        yield f"{name} 3d axis=1", lambda f=f: f(L.arr(r.standard_normal((3, 4, 5))), axis=1)
        yield f"{name} strided", lambda f=f: f(L.arr(base)[::2, ::-1].T, axis=0)
        yield f"{name} large", lambda f=f: f(L.arr(L.rng(12).standard_normal((2**17 + 3,)) * 50))
        yield f"{name} large axis=0", lambda f=f: f(L.arr(L.rng(13).standard_normal((40000, 5))), axis=0)
        yield f"{name} empty", lambda f=f: f(sd.zeros((0,)))
        yield f"{name} empty axis=0", lambda f=f: f(sd.zeros((0, 3)), axis=0)
        yield f"{name} 0d", lambda f=f: f(sd.asarray(2.5))
        yield f"{name} scalar", lambda f=f: f(2.5)
        yield f"{name} ints", lambda f=f: f(sd.asarray([1, 2, 3]))
        yield f"{name} bool", lambda f=f: f(sd.asarray([True, False]))
        yield f"{name} list", lambda f=f: f([1.0, 2.0])
        yield f"{name} complex", lambda f=f: f(sd.asarray([1 + 1j]))
        yield f"{name} err axis out of range", lambda f=f: f(L.arr(base), axis=2)
        yield f"{name} err axis repeated", lambda f=f: f(L.arr(base), axis=(0, 0))
        yield f"{name} err axis float", lambda f=f: f(L.arr(base), axis=1.0)
        yield f"{name} err unknown keyword", lambda f=f: f(L.arr(base), b=1.0)
    for keepdims in (False, True):
        for axis in (None, 0, 1, (0, 1)):
            yield (
                f"logsumexp keepdims={keepdims} axis={axis}",
                lambda axis=axis, keepdims=keepdims: S.logsumexp(L.arr(base), axis=axis, keepdims=keepdims),
            )
    yield "logsumexp err keepdims int", lambda: S.logsumexp(L.arr(base), keepdims=1)


def special_forms(L):
    sd = L.sd
    S = sd.special
    re_ = [-INF, -10.0, -1.0, -0.5, -1e-300, -0.0, 0.0, 1e-300, 0.25, 1.0, 3.0, 30.0, 1e300, INF, NAN]
    zgrid = np.array([complex(a, b) for a in re_ for b in re_])
    r = L.rng(21)
    zrand = r.standard_normal(1000) * 10.0 ** r.uniform(-3, 2, 1000) + 1j * r.standard_normal(1000) * 10.0 ** r.uniform(
        -3, 2, 1000
    )
    zlarge = L.rng(22).standard_normal(2**17 + 1) * 5 + 1j * L.rng(23).standard_normal(2**17 + 1) * 5
    for name in COMPLEX:
        f = getattr(S, name)
        yield f"{name} complex grid", lambda f=f: f(L.arr(zgrid))
        yield f"{name} complex random", lambda f=f: f(L.arr(zrand))
        yield f"{name} complex large", lambda f=f: f(L.arr(zlarge))
        yield f"{name} complex scalar", lambda f=f: f(0.5 - 1.5j)
        yield (
            f"{name} complex out",
            lambda f=f: (lambda o: (f(L.arr(zrand[:5]), out=o) is o, o))(sd.zeros((5,), dtype=sd.complex128)),
        )
        yield f"{name} complex out float", lambda f=f: f(L.arr(zrand[:5]), out=sd.zeros((5,)))
    # lambertw's branches
    xs = L.arr(np.array(GRID["x"]))
    for kk in (0, -1, 1, 2, -3, 100):
        yield f"lambertw real k={kk}", lambda kk=kk: S.lambertw(xs, k=kk)
        yield f"lambertw complex k={kk}", lambda kk=kk: S.lambertw(L.arr(zgrid), k=kk)
    yield "lambertw k bool", lambda: S.lambertw(xs, k=True)
    yield "lambertw k float", lambda: S.lambertw(xs, k=0.0)
    yield "lambertw k big", lambda: S.lambertw(L.arr(zgrid[:5]), k=2**63)
    yield "lambertw large k=-1", lambda: S.lambertw(L.arr(-L.rng(24).uniform(0, 0.36787944117144233, 2**17 + 1)), k=-1)
    # zeta with one and two arguments
    yield "zeta one arg grid", lambda: S.zeta(L.arr(np.array(GRID["x"])))
    yield "zeta one arg random", lambda: S.zeta(L.arr(L.rng(25).uniform(-60, 60, 1000)))
    yield "zeta q=None", lambda: S.zeta(2.0, None)
    # airy's tuple
    yield "airy result fields", lambda: S.airy(L.arr(np.linspace(-30, 30, 61)))
    yield "airy scalar negative", lambda: S.airy(-2.5)
    yield "airy err out", lambda: S.airy(1.0, out=sd.zeros((1,)))
