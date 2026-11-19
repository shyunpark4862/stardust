"""sd.special against scipy.special: every function with a reference file at 10^6
elements (sd on 1 and 10 threads; scipy.special runs on one) and one Python-float (or
complex) call. From bench/special/speed.py; the functions, their counterparts and the
inputs (drawn from the guaranteed-region reference inputs) come from
bench/special/common.py, imported as it is.

The 1-thread time per element of the 10^6 rows is the cost table of the parallel split
(csrc/special/cost_table.hpp, read by csrc/special/cost.hpp; bench/special/cost.py
printed it before): each such case carries cost=("special", <reference name>). The
table's unit, sd.add of two contiguous float64 arrays, and the engine's test kernels
(_fma3, _sum4, _powers4, on normal inputs) are cost-only cases, as cost.py's EXTRA."""

from __future__ import annotations

from ..registry import Case, Ref

N = 10**6
GROUP = "배열 10^6개"
SCALAR_GROUP = "스칼라 한 번"
COST_GROUP = "비용표 단위"

# Argument names (scipy's) of the functions that take more than one argument.
ARGS = {
    "bdtr": "k, n, p", "bdtrc": "k, n, p", "nbdtr": "k, n, p", "nbdtrc": "k, n, p",
    "beta": "a, b", "betaln": "a, b",
    "betainc": "a, b, x", "betaincc": "a, b, x", "betaincinv": "a, b, y", "betainccinv": "a, b, y",
    "binom": "n, k", "chdtr": "v, x", "chdtrc": "v, x", "chdtri": "v, p",
    "chndtr": "df, nc, x", "chndtrc": "df, nc, x", "chndtrix": "df, nc, p",
    "expn": "n, x", "fdtr": "dfn, dfd, x", "fdtrc": "dfn, dfd, x", "fdtri": "dfn, dfd, p",
    "gammainc": "a, x", "gammaincc": "a, x", "gammaincinv": "a, y", "gammainccinv": "a, y",
    "hyp1f1": "a, b, x", "hyp2f1": "a, b, c, x",
    "iv": "v, x", "ive": "v, x", "jv": "v, x", "kv": "v, x", "kve": "v, x", "yv": "v, x",
    "ncfdtr": "dfn, dfd, nc, f", "ncfdtrc": "dfn, dfd, nc, f", "ncfdtri": "dfn, dfd, nc, p",
    "nctdtr": "df, nc, t", "nctdtrc": "df, nc, t", "nctdtrit": "df, nc, p",
    "pdtr": "k, m", "pdtrc": "k, m", "polygamma": "n, x", "rel_entr": "x, y", "stdtr": "df, t",
    "stdtrit": "df, p", "xlogy": "x, y", "xlog1py": "x, y", "zeta_hurwitz": "x, q",
}

QUICK = {"erf", "erf_complex", "gamma", "gammaln", "gammainc", "betainc", "jv", "kv", "hyp2f1", "ndtri",
         "lambertw", "zeta_hurwitz"}
SCALAR_QUICK = {"erf", "jv"}

# Engine test kernels without a reference file (cost.py EXTRA): number of normal inputs.
EXTRA = {"_fma3": 3, "_sum4": 4, "_powers4": 1}
EXTRA_ARGS = {"_fma3": "x, y, z", "_sum4": "a, b, c, d", "_powers4": "x"}


def _common():
    from bench.special import common

    return common


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
    if (c is None or c == f"scipy.special.{name}" or name.endswith("_complex") or name in common.PART
            or name.startswith("lambertw_c")):  # the call name already says which branch
        return ""
    return f"비교 대상 {c}"


def _lib(name):
    c = _common().counterpart(name)
    return c.split(".")[0] if c else None


def _setup(name):
    import stardust as sd

    common = _common()
    xs = common.sample_inputs(name, N)
    return {"f": common.sd_function(name), "ref": common.scipy_function(name),
            "xs": xs, "sxs": tuple(sd.asarray(x) for x in xs)}


def _scalar_setup(name):
    """One Python float (or complex) per argument: the first of the 10^6 inputs, as speed.py
    (the first draw does not depend on how many are drawn)."""
    import numpy as np

    common = _common()
    xs = common.sample_inputs(name, 1)
    return {"f": common.sd_function(name), "ref": common.scipy_function(name),
            "one": tuple(complex(x[0]) if np.iscomplexobj(x) else float(x[0]) for x in xs)}


def _extra_setup(name):
    import numpy as np

    import stardust as sd

    rng = np.random.default_rng(0)
    if name == "_add":
        f, k = sd.add, 2
    else:
        f, k = _common().sd_function(name), EXTRA[name]
    return {"f": f, "sxs": tuple(sd.asarray(rng.normal(size=N)) for _ in range(k))}


def _call(fkey, xkey):
    return lambda d: (lambda f=d[fkey], a=d[xkey]: f(*a))


def cases() -> list[Case]:
    common = _common()
    names = common.ref_names()
    out = []
    for name in names:
        lib = _lib(name)
        refs = (Ref(lib, _call("ref", "xs")),) if lib else ()
        out.append(Case(
            area="special", group=GROUP, name=call_name(name), size="10^6", n=N,
            setup=lambda name=name: _setup(name),
            sd=_call("f", "sxs"), refs=refs, threads=(1, 10),
            quick=name in QUICK, cost=("special", name), note=_note(name),
        ))
    for name in ("_add", *EXTRA):
        out.append(Case(
            area="special", group=COST_GROUP,
            name="add(x, y)" if name == "_add" else f"{name}({EXTRA_ARGS[name]})", size="10^6", n=N,
            setup=lambda name=name: _extra_setup(name),
            sd=_call("f", "sxs"), threads=(1,),
            quick=name == "_add", cost=("special", name),
            note="비용표의 단위(float64 두 배열의 덧셈)" if name == "_add" else "엔진 시험용 커널, 정규분포 입력",
        ))
    for name in names:
        lib = _lib(name)
        refs = (Ref(lib, _call("ref", "one")),) if lib else ()
        out.append(Case(
            area="special", group=SCALAR_GROUP, name=call_name(name),
            setup=lambda name=name: _scalar_setup(name),
            sd=_call("f", "one"), refs=refs, threads=(1,),
            quick=name in SCALAR_QUICK, note=_note(name),
        ))
    return out
