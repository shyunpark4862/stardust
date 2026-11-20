"""sd.special against scipy.special: every function with a reference file at 10^6
elements (sd on 1 and 10 threads; scipy.special runs on one) and one Python-float (or
complex) call. The functions, their counterparts and the reference-mix inputs (drawn from
the guaranteed-region reference inputs) come from bench/speed/cases/_special_common.py.

The 1-thread time per element of the reference-mix rows is the cost table of the parallel
split (csrc/special/cost_table.hpp, read by csrc/special/cost.hpp): each such case carries
cost=("special", <reference name>). The noncentral distribution functions take 10^5
elements there instead of 10^6 (seconds per call at 10^6; the per-element cost is the
same). The table's unit, sd.add of two contiguous float64 arrays, the engine's test kernels
(_fma3, _sum4, _powers4, on normal inputs; EXTRA below) and _cf_tan (on its reference
inputs) are cost-unit cases.

The reference mix leans on the hard corners of each function (huge and tiny arguments,
branch edges), so the other groups time what a user usually passes: the "typical input"
rows (fixed ranges per function, TYPICAL; not in the cost table), the mid sizes 10^2-10^4
(rotating fresh inputs, harness.rotate, so the branch predictor cannot learn one input),
the reductions logsumexp / softmax / log_softmax, and the scalar calls at an everyday
point per function (SCALAR_POINT).
"""

from __future__ import annotations

import re

from ..harness import fresh_count, rotate
from ..registry import Case, Ref, RRef

N = 10**6
N_NONCENTRAL = 10**5
GROUP = "array-1e6"
TYPICAL_GROUP = "array-1e6-typical"
MID_GROUP = "array-small"
REDUCE_GROUP = "logsumexp-softmax"
SCALAR_GROUP = "scalar"
COST_GROUP = "cost-units"

# Argument names (scipy's) of the functions that take more than one argument.
ARGS = {
    "bdtr": "k, n, p",
    "bdtrc": "k, n, p",
    "nbdtr": "k, n, p",
    "nbdtrc": "k, n, p",
    "beta": "a, b",
    "betaln": "a, b",
    "betainc": "a, b, x",
    "betaincc": "a, b, x",
    "betaincinv": "a, b, y",
    "betainccinv": "a, b, y",
    "binom": "n, k",
    "chdtr": "v, x",
    "chdtrc": "v, x",
    "chdtri": "v, p",
    "chndtr": "df, nc, x",
    "chndtrc": "df, nc, x",
    "chndtrix": "df, nc, p",
    "expn": "n, x",
    "fdtr": "dfn, dfd, x",
    "fdtrc": "dfn, dfd, x",
    "fdtri": "dfn, dfd, p",
    "gammainc": "a, x",
    "gammaincc": "a, x",
    "gammaincinv": "a, y",
    "gammainccinv": "a, y",
    "hyp1f1": "a, b, x",
    "hyp2f1": "a, b, c, x",
    "iv": "v, x",
    "ive": "v, x",
    "jv": "v, x",
    "kv": "v, x",
    "kve": "v, x",
    "yv": "v, x",
    "ncfdtr": "dfn, dfd, nc, f",
    "ncfdtrc": "dfn, dfd, nc, f",
    "ncfdtri": "dfn, dfd, nc, p",
    "nctdtr": "df, nc, t",
    "nctdtrc": "df, nc, t",
    "nctdtrit": "df, nc, p",
    "pdtr": "k, m",
    "pdtrc": "k, m",
    "polygamma": "n, x",
    "rel_entr": "x, y",
    "stdtr": "df, t",
    "stdtrit": "df, p",
    "xlog1py": "x, y",
    "xlogy": "x, y",
    "zeta_hurwitz": "x, q",
}

QUICK = {
    "erf",
    "erf_complex",
    "gamma",
    "gammaln",
    "gammainc",
    "betainc",
    "jv",
    "kv",
    "hyp2f1",
    "ndtri",
    "lambertw",
    "zeta_hurwitz",
}
TYPICAL_QUICK = {"gammainc"}
REDUCE_QUICK = {("logsumexp", None)}
SCALAR_QUICK = {"erf", "jv"}

# The same call as airy_ai (airy(x) computes all four): timed once, as airy(x)[0].
SKIP = {"airy_aip", "airy_bi", "airy_bip"}
# Seconds per call at 10^6 elements: timed at N_NONCENTRAL.
NONCENTRAL = {"chndtr", "chndtrc", "chndtrix", "nctdtr", "nctdtrc", "nctdtrit", "ncfdtr", "ncfdtrc", "ncfdtri"}
# Reference files timed only as cost-unit cases.
COST_ONLY = {"_cf_tan"}

# Engine test kernels without a reference file: number of normal inputs.
EXTRA = {"_fma3": 3, "_sum4": 4, "_powers4": 1}
EXTRA_ARGS = {"_fma3": "x, y, z", "_sum4": "a, b, c, d", "_powers4": "x"}
# _add is the unit every weight of the table is divided by: more batches than the default.
ADD_TIMING = {"batches": 25, "budget_s": 5.0}

MID_SIZES = ((100, "10^2"), (1000, "10^3"), (10_000, "10^4"))
MID_NAMES = ("expit", "erf", "gammaln", "gammainc", "jv", "erf_complex")
REDUCE_SHAPES = (((N,), None, "10^6"), ((1000, 1000), 1, "(1000, 1000)"), ((1000, 1000), 0, "(1000, 1000)"))


# ---------------------------------------------------------------- typical inputs

P_LO, P_HI = 1e-6, 1 - 1e-6


def _u(rng, n, lo, hi):
    return rng.uniform(lo, hi, n)


def _gamma_x(rng, n):
    a = _u(rng, n, 0.5, 20.0)
    return a, rng.gamma(a)


def _gamma_y(rng, n):
    return _u(rng, n, 0.5, 20.0), _u(rng, n, P_LO, P_HI)


def _beta_x(rng, n):
    a, b = _u(rng, n, 0.5, 20.0), _u(rng, n, 0.5, 20.0)
    return a, b, rng.beta(a, b)


def _beta_y(rng, n):
    return _u(rng, n, 0.5, 20.0), _u(rng, n, 0.5, 20.0), _u(rng, n, P_LO, P_HI)


def _bessel(rng, n):
    return _u(rng, n, 0.0, 20.0), _u(rng, n, 0.0, 50.0)


GAMMA_X_NOTE = "a는 0.5–20에서 균등하게, x는 Gamma(a)의 표본으로 뽑는다(분포의 몸통)"
GAMMA_Y_NOTE = "a는 0.5–20, y는 (1e-6, 1 − 1e-6)에서 균등하게 뽑는다"
BETA_X_NOTE = "a, b는 0.5–20에서 균등하게, x는 Beta(a, b)의 표본으로 뽑는다"
BETA_Y_NOTE = "a, b는 0.5–20, y는 (1e-6, 1 − 1e-6)에서 균등하게 뽑는다"
BESSEL_NOTE = "v는 0–20, x는 0–50에서 균등하게 뽑는다"

# name -> (rng, n -> argument arrays, the ranges as a note)
TYPICAL = {
    "gammainc": (_gamma_x, GAMMA_X_NOTE),
    "gammaincc": (_gamma_x, GAMMA_X_NOTE),
    "gammaincinv": (_gamma_y, GAMMA_Y_NOTE),
    "betainc": (_beta_x, BETA_X_NOTE),
    "betaincinv": (_beta_y, BETA_Y_NOTE),
    "jv": (_bessel, BESSEL_NOTE),
    "yv": (_bessel, BESSEL_NOTE),
    "iv": (_bessel, BESSEL_NOTE),
    "kv": (_bessel, BESSEL_NOTE),
    "hyp1f1": (
        lambda rng, n: (_u(rng, n, 0.1, 10.0), _u(rng, n, 0.5, 20.0), _u(rng, n, -10.0, 10.0)),
        "a는 0.1–10, b는 0.5–20, x는 −10–10에서 균등하게 뽑는다",
    ),
    "hyp2f1": (
        lambda rng, n: (_u(rng, n, 0.1, 5.0), _u(rng, n, 0.1, 5.0), _u(rng, n, 0.5, 10.0), _u(rng, n, -0.9, 0.9)),
        "a, b는 0.1–5, c는 0.5–10, x는 −0.9–0.9에서 균등하게 뽑는다",
    ),
    "lambertw": (lambda rng, n: (_u(rng, n, -0.3, 50.0),), "x는 −0.3–50에서 균등하게 뽑는다(가지 k=0)"),
    "ndtri": (lambda rng, n: (_u(rng, n, P_LO, P_HI),), "p는 (1e-6, 1 − 1e-6)에서 균등하게 뽑는다"),
    "zeta_hurwitz": (
        lambda rng, n: (_u(rng, n, 1.5, 10.0), _u(rng, n, 0.5, 10.0)),
        "x는 1.5–10, q는 0.5–10에서 균등하게 뽑는다",
    ),
}

# One everyday point per function for the scalar calls (the reference mix's first draw is
# often an extreme, e.g. x = -2.2e9 for airy or a = 7e-40 for gammainc).
_X1 = (0.7,)
_Z1 = (0.7 + 0.4j,)
SCALAR_POINT = {
    "airy_ai": _X1,
    "bdtr": (8.0, 20.0, 0.4),
    "bdtrc": (8.0, 20.0, 0.4),
    "beta": (2.5, 3.5),
    "betaln": (2.5, 3.5),
    "betainc": (2.5, 3.5, 0.4),
    "betaincc": (2.5, 3.5, 0.4),
    "betaincinv": (2.5, 3.5, 0.3),
    "betainccinv": (2.5, 3.5, 0.3),
    "binom": (20.0, 7.0),
    "chdtr": (4.5, 3.2),
    "chdtrc": (4.5, 3.2),
    "chdtri": (4.5, 0.3),
    "chndtr": (3.5, 2.0, 5.0),
    "chndtrc": (3.5, 2.0, 5.0),
    "chndtrix": (3.5, 2.0, 0.3),
    "digamma": (2.5,),
    "ellipe": (0.4,),
    "ellipk": (0.4,),
    "entr": _X1,
    "erf": _X1,
    "erfc": _X1,
    "erfcinv": (0.6,),
    "erfcx": (1.5,),
    "erfinv": (0.4,),
    "erf_complex": _Z1,
    "erfc_complex": _Z1,
    "erfcx_complex": _Z1,
    "expi": (1.5,),
    "expit": _X1,
    "expn": (3.0, 1.5),
    "fdtr": (5.0, 12.0, 1.3),
    "fdtrc": (5.0, 12.0, 1.3),
    "fdtri": (5.0, 12.0, 0.3),
    "gamma": (2.5,),
    "gammaln": (2.5,),
    "rgamma": (2.5,),
    "gammainc": (2.5, 2.0),
    "gammaincc": (2.5, 2.0),
    "gammaincinv": (2.5, 0.3),
    "gammainccinv": (2.5, 0.3),
    "hyp1f1": (1.5, 3.5, 0.7),
    "hyp2f1": (1.5, 2.5, 3.5, 0.4),
    **{f: (2.5,) for f in ("i0", "i0e", "i1", "i1e", "j0", "j1", "y0", "y1", "k0", "k0e", "k1", "k1e")},
    **{f: (2.5, 4.0) for f in ("iv", "ive", "jv", "yv", "kv", "kve")},
    "lambertw": (1.5,),
    "lambertw_m1": (-0.2,),
    **{f: (1.5 + 0.5j,) for f in ("lambertw_c0", "lambertw_c1", "lambertw_c4", "lambertw_cm1")},
    "log_expit": _X1,
    "log_ndtr": _X1,
    "loggamma_complex": (2.5 + 1.0j,),
    "logit": (0.3,),
    "nbdtr": (8.0, 5.0, 0.4),
    "nbdtrc": (8.0, 5.0, 0.4),
    "ncfdtr": (4.0, 13.0, 3.0, 1.5),
    "ncfdtrc": (4.0, 13.0, 3.0, 1.5),
    "ncfdtri": (4.0, 13.0, 3.0, 0.3),
    "nctdtr": (9.0, 1.5, 2.0),
    "nctdtrc": (9.0, 1.5, 2.0),
    "nctdtrit": (9.0, 1.5, 0.3),
    "ndtr": _X1,
    "ndtri": (0.3,),
    "ndtri_exp": (-1.2,),
    "pdtr": (4.0, 4.5),
    "pdtrc": (4.0, 4.5),
    "polygamma": (1.0, 2.5),
    "rel_entr": (0.4, 0.6),
    "stdtr": (3.5, 0.7),
    "stdtrit": (3.5, 0.3),
    "wofz": _Z1,
    "xlog1py": (0.7, 0.4),
    "xlogy": (0.7, 1.5),
    "zeta": (2.5,),
    "zeta_hurwitz": (2.5, 1.5),
}


def _common():
    from . import _special_common

    return _special_common


def _complex(name):
    return name.endswith("_complex") or name.startswith("lambertw_c") or name == "wofz"


def call_name(name):
    """The call as code, e.g. 'betainc(a, b, x)', 'airy(x)[1]', 'lambertw(z, k=4)'."""
    common = _common()
    args = ARGS.get(name, "z" if _complex(name) else "x")
    if name in common.PART:
        base, i = common.PART[name]
        return f"{base}({args})[{i}]"
    if name in common.KW:
        base, kw = common.KW[name]
        extra = "".join(f", {k}={v}" for k, v in kw.items() if v != 0)
        return f"{base}({args}{extra})"
    return f"{common.ALIAS.get(name, name)}({args})"


def _note(name):
    common = _common()
    c = common.counterpart(name)
    if (
        c is None
        or c == f"scipy.special.{name}"
        or name.endswith("_complex")
        or name in common.PART
        or name.startswith("lambertw_c")
    ):  # the call name already says which branch
        return ""
    return f"비교 대상 {c}"


def _join(*notes):
    return ", ".join(n for n in notes if n)


def _lib(name):
    c = _common().counterpart(name)
    return c.split(".")[0] if c else None


def _refs(name, key):
    lib = _lib(name)
    return (Ref(lib, _call("ref", key), differs=DIFFERS.get((name, lib), "")),) if lib else ()


# (function, comparator) -> the elements it is timed on, where one call over the case's inputs
# takes seconds (the smoke run's times, about 1-2 s a call at these sizes); the tables scale the
# time to the case's size and mark it
COMPARATOR_SAMPLE = {
    ("ive", "R"): 30_000,
    ("binom", "R"): 200_000,
    ("i0", "R"): 200_000,
    ("i0e", "R"): 200_000,
    ("i1", "R"): 200_000,
    ("i1e", "R"): 200_000,
    ("stdtrit", "R"): 500_000,
    ("chndtrix", "R"): 10_000,
    ("ncfdtri", "R"): 30_000,
    ("nctdtrit", "R"): 3_000,
    ("expn", "Boost"): 100_000,
    ("jv", "Boost"): 300_000,
    ("yv", "Boost"): 300_000,
    ("nctdtrit", "Boost"): 50,
}


# Comparators that compute something else than sd on these inputs, by design: not checked,
# and the reason goes into the table's note.
DIFFERS = {
    ("binom", "R"): "R의 choose는 정수가 아닌 k를 반올림한다",
    ("nbdtr", "scipy"): "scipy는 정수가 아닌 n을 버림한다",
    ("nbdtrc", "scipy"): "scipy는 정수가 아닌 n을 버림한다",
    ("i0e", "R"): "R의 besselI는 x < 0에서 NaN, 아주 큰 x에서 0이다",
    ("i1e", "R"): "R의 besselI는 x < 0에서 NaN, 아주 큰 x에서 0이다",
}


def _r_boost(name, rotated=False):
    """R and Boost.Math for an array case of `name`, where the accuracy measurement compares
    with them (bench/accuracy/compare.py: R_SPECIAL, BOOST_SPECIAL): R's vectorized function in
    the run's R session on the case's inputs (every input set of a rotated case, in turn),
    Boost in a C++ loop over them (bench/speed/boost.py)."""
    from bench.accuracy.compare import BOOST_SPECIAL, R_SPECIAL

    from .. import boost

    out = []
    if name in R_SPECIAL:
        expr = R_SPECIAL[name]
        used = [v for v in "abcd" if re.search(rf"\b{v}\b", expr)]
        k = COMPARATOR_SAMPLE.get((name, "R"))
        setup = "; ".join(f"{v} <- {v}[seq_len({k})]" for v in used) if k else ""
        data = {f"arg{'abcd'.index(v)}": v for v in used}
        differs = DIFFERS.get((name, "R"), "")
        out.append(RRef("R", expr, data=data, setup=setup, n=k, sets="rsets" if rotated else "", differs=differs))
    if name in BOOST_SPECIAL and boost.available():
        k = COMPARATOR_SAMPLE.get((name, "Boost"))
        if rotated:
            make = lambda d: rotate([boost.special(name, xs) for xs in d["xs"]])  # noqa: E731
        else:
            make = lambda d, k=k: boost.special(name, [x[:k] for x in d["xs"]] if k else d["xs"])  # noqa: E731
        out.append(Ref("Boost", make, lib="Boost", n=k))
    return tuple(out)


def _funcs(name):
    common = _common()
    return {"f": common.sd_function(name), "ref": common.scipy_function(name)}


def _with_sd(d, xs):
    import numpy as np

    import stardust as sd

    d["xs"] = xs
    d["sxs"] = tuple(sd.asarray(x) for x in xs)
    for i, x in enumerate(xs):  # R's inputs (float64 arrays only travel)
        if not np.iscomplexobj(x):
            d[f"arg{i}"] = np.asarray(x, dtype=np.float64)
    return d


def _setup(name, n=N):
    return _with_sd(_funcs(name), _common().sample_inputs(name, n))


def _typical_inputs(name, n, seed=0):
    import numpy as np

    return tuple(np.ascontiguousarray(x) for x in TYPICAL[name][0](np.random.default_rng(seed), n))


def _typical_setup(name):
    return _with_sd(_funcs(name), _typical_inputs(name, N))


def _mid_setup(name, n):
    """fresh_count(n) inputs of n elements, each called in turn."""
    import numpy as np

    import stardust as sd

    k = fresh_count(n)
    if name in TYPICAL:
        sets = [_typical_inputs(name, n, seed=i) for i in range(k)]
    else:
        xs, _, _, guaranteed = _common().ref_load(name)
        pool = np.flatnonzero(guaranteed) if guaranteed.any() else np.arange(guaranteed.size)
        idx = np.random.default_rng(0).choice(pool, (k, n))
        sets = [tuple(np.ascontiguousarray(x[i]) for x in xs) for i in idx]
    d = _funcs(name)
    d["xs"] = sets
    d["sxs"] = [tuple(sd.asarray(x) for x in s) for s in sets]
    # R's input sets (float64 arrays only travel), called in turn as the others are
    d["rsets"] = [
        {f"arg{i}": np.asarray(x, dtype=np.float64) for i, x in enumerate(s) if not np.iscomplexobj(x)} for s in sets
    ]
    return d


def _rotated(fkey, xkey):
    return lambda d: rotate([lambda f=d[fkey], a=a: f(*a) for a in d[xkey]])


def _scalar_point(name):
    """(the point as Python numbers, its note) for the scalar call."""
    import numpy as np

    if name in SCALAR_POINT:
        one = SCALAR_POINT[name]
        names = ARGS.get(name, "z" if _complex(name) else "x").split(", ")
        return one, ", ".join(f"{a}={v:g}" for a, v in zip(names, one))
    xs = _common().sample_inputs(name, 1)  # a function without an everyday point yet: the first draw
    return tuple(complex(x[0]) if np.iscomplexobj(x) else float(x[0]) for x in xs), "기준 입력의 첫 점"


def _scalar_setup(name):
    d = _funcs(name)
    d["one"] = _scalar_point(name)[0]
    return d


def _extra_setup(name):
    import numpy as np

    import stardust as sd

    if name in COST_ONLY:
        return _setup(name)
    rng = np.random.default_rng(0)
    if name == "_add":
        f, k = sd.add, 2
    else:
        f, k = _common().sd_function(name), EXTRA[name]
    return {"f": f, "sxs": tuple(sd.asarray(rng.normal(size=N)) for _ in range(k))}


def _reduce_setup(fname, shape):
    import numpy as np
    import scipy.special

    import stardust as sd

    x = np.random.default_rng(0).normal(0, 10, shape)
    return {"f": getattr(sd.special, fname), "g": getattr(scipy.special, fname), "x": x, "X": sd.asarray(x)}


def _call(fkey, xkey):
    return lambda d: lambda f=d[fkey], a=d[xkey]: f(*a)


def _reduce_cases():
    out = []
    for fname in ("logsumexp", "softmax", "log_softmax"):
        for shape, axis, size in REDUCE_SHAPES:
            out.append(
                Case(
                    area="special",
                    group=REDUCE_GROUP,
                    name=f"{fname}(x)" if axis is None else f"{fname}(x, axis={axis})",
                    size=size,
                    n=N,
                    setup=lambda fname=fname, shape=shape: _reduce_setup(fname, shape),
                    sd=lambda d, axis=axis: lambda f=d["f"], X=d["X"]: f(X, axis=axis),
                    refs=(Ref("scipy", lambda d, axis=axis: lambda g=d["g"], x=d["x"]: g(x, axis=axis)),),
                    quick=(fname, axis) in REDUCE_QUICK,
                    note="정규분포 N(0, 10²) 입력",
                )
            )
    return out


# The section title of each group in bench/special/speed.md (the group is the English key in
# the case ids).
TITLES = {
    "array-1e6": "배열 10^6개",
    "array-1e6-typical": "배열 10^6개(흔한 입력)",
    "array-small": "배열 10^2–10^4개",
    "logsumexp-softmax": "로그합·소프트맥스",
    "cost-units": "비용표 단위",
    "scalar": "스칼라 한 번",
}


def cases() -> list[Case]:
    common = _common()
    names = [nm for nm in common.ref_names() if nm not in SKIP]
    array_names = [nm for nm in names if nm not in COST_ONLY]
    out = []
    for name in array_names:
        n = N_NONCENTRAL if name in NONCENTRAL else N
        out.append(
            Case(
                area="special",
                group=GROUP,
                name=call_name(name),
                size="10^5" if n == N_NONCENTRAL else "10^6",
                n=n,
                setup=lambda name=name, n=n: _setup(name, n),
                sd=_call("f", "sxs"),
                refs=_refs(name, "xs") + _r_boost(name),
                threads=(1, 10),
                quick=name in QUICK,
                cost=("special", name),
                note=_note(name),
            )
        )
    for name in TYPICAL:
        out.append(
            Case(
                area="special",
                group=TYPICAL_GROUP,
                name=call_name(name),
                size="10^6",
                n=N,
                setup=lambda name=name: _typical_setup(name),
                sd=_call("f", "sxs"),
                refs=_refs(name, "xs") + _r_boost(name),
                threads=(1, 10),
                quick=name in TYPICAL_QUICK,
                note=_join(TYPICAL[name][1], _note(name)),
            )
        )
    for name in MID_NAMES:
        for n, size in MID_SIZES:
            out.append(
                Case(
                    area="special",
                    group=MID_GROUP,
                    name=call_name(name),
                    size=size,
                    n=n,
                    setup=lambda name=name, n=n: _mid_setup(name, n),
                    sd=_rotated("f", "sxs"),
                    refs=(Ref(_lib(name), _rotated("ref", "xs"), differs=DIFFERS.get((name, _lib(name)), "")),)
                    + _r_boost(name, rotated=True),
                    threads=(1, 10),
                    note=_join(
                        f"입력 {fresh_count(n)}벌을 돌려 쓴다",
                        "흔한 입력(10^6개 묶음과 같은 범위)" if name in TYPICAL else "기준 입력",
                    ),
                )
            )
    out += _reduce_cases()
    for name in ("_add", *EXTRA, *sorted(COST_ONLY)):
        if name == "_add":
            label, note = "add(x, y)", "비용표의 단위(float64 두 배열의 덧셈), 배치를 늘려 잰다"
        elif name in COST_ONLY:
            label, note = call_name(name), "연분수 tan 시험용 커널, 기준 입력"
        else:
            label, note = f"{name}({EXTRA_ARGS[name]})", "엔진 시험용 커널, 정규분포 입력"
        out.append(
            Case(
                area="special",
                group=COST_GROUP,
                name=label,
                size="10^6",
                n=N,
                setup=lambda name=name: _extra_setup(name),
                sd=_call("f", "sxs"),
                threads=(1,),
                quick=name == "_add",
                cost=("special", name),
                note=note,
                timing=ADD_TIMING if name == "_add" else {},
            )
        )
    for name in array_names:
        out.append(
            Case(
                area="special",
                group=SCALAR_GROUP,
                name=call_name(name),
                setup=lambda name=name: _scalar_setup(name),
                sd=_call("f", "one"),
                refs=_refs(name, "one"),
                threads=(1,),
                quick=name in SCALAR_QUICK,
                note=_join(_scalar_point(name)[1], _note(name)),
            )
        )
    return out
