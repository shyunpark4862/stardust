"""Golden cases for sd.integrate (tools/golden.py has the contract).

sd.integrate's only public name so far is its result class (milestone C0 prepared the area:
docs/design/integrate.md §8): the class's fields, read-only rules, unpacking (the quad family's
results, C10) and repr. Each milestone adds the cases of the functions it makes public, on the
problems of tests/integrate/problems.py.
"""


def encode(x, L, enc):
    """An IntegrateResult: the fields it gives, the names of those it does not, and whether it
    unpacks (and into what)."""
    if type(x) is not L.sd.integrate.IntegrateResult:
        return None
    attrs, absent = {}, []
    for name in sorted(n for n in dir(x) if not n.startswith("_")):
        try:
            attrs[name] = enc(getattr(x, name))
        except AttributeError:
            absent.append(name)
    try:
        pair = enc(tuple(x))
    except TypeError as e:
        pair = {"error": "TypeError", "msg": str(e)}
    return {"obj": f"{type(x).__module__}.{type(x).__qualname__}", "attrs": attrs, "absent": absent,
            "unpack": pair, "repr": repr(x)}  # fmt: skip


def cases(L):
    sd = L.sd
    R = sd.integrate.IntegrateResult
    yield "IntegrateResult fields", lambda: R(integral=0.5, error=1e-15, converged=True, status=0, nfev=21)
    yield (
        "IntegrateResult quad-like",
        lambda: R._make(integral=0.5, error=1e-15, converged=True, status=0, message="ok", nfev=21, unpack=True),
    )
    yield "IntegrateResult unpack", lambda: tuple(R._make(integral=2.0, error=3.0, unpack=True))
    yield "IntegrateResult getitem", lambda: R._make(integral=2.0, error=3.0, unpack=True)[1]
    yield "err IntegrateResult no unpack", lambda: tuple(R(integral=2.0, error=3.0))
    yield (
        "IntegrateResult batch",
        lambda: R(batch=(3,), integral=sd.asarray([1.0, 2.0, 3.0]), error=sd.zeros(3),
                  converged=sd.asarray([True, False, True]), nfev=sd.asarray([67, 16387, 131])),
    )  # fmt: skip
    yield "IntegrateResult series", lambda: R(sum=1.6449340668482264, error=7.4e-9, converged=True, nfev=8561)
    yield "IntegrateResult repr", lambda: repr(R(integral=float("nan"), converged=False))
    yield "err IntegrateResult positional", lambda: R(1)
    yield "err IntegrateResult unknown field", lambda: R(estimate=1.0)
    yield "err IntegrateResult absent field", lambda: R(integral=1.0).error
    yield "err IntegrateResult read-only", lambda: setattr(R(integral=1.0), "integral", 2.0)
    yield (
        "err IntegrateResult array read-only",
        lambda: R(batch=(2,), integral=sd.zeros(2)).integral.__setitem__(0, 1.0),
    )
