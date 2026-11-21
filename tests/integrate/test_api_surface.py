"""docs/api/integrate.md and the module agree: the signature blocks name the first version's 15
functions (decision C2) and the result class; the result class (C0) and the functions of the
milestones done (C3) are exported with the documented signatures, and the documented signatures
follow the area's naming decisions (C4, C11-C14, C40)."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "integrate.md"

# C2: the first version's functions, all documented and none public yet (milestones C3-C7).
PLANNED = {
    "quad", "quad_vec", "dblquad", "tplquad", "nquad", "tanhsinh", "nsum", "cubature", "fixed_quad",
    "newton_cotes", "trapezoid", "simpson", "romb", "cumulative_trapezoid", "cumulative_simpson",
}  # fmt: skip
RESULTS = {"IntegrateResult"}
PUBLIC = {
    "trapezoid",
    "simpson",
    "romb",
    "cumulative_trapezoid",
    "cumulative_simpson",
    "fixed_quad",
    "newton_cotes",
}  # C3
ELEMENTWISE = {"tanhsinh", "nsum"}  # C4: the elementwise-iterative family, the only ones with args


def documented():
    """({name: signature text} of every `name(...)` line, the bare names) of the signature blocks."""
    names, bare = {}, set()
    for block in signature_blocks(API.read_text()):
        for line in block:
            line = line.split("#")[0].strip()
            m = re.fullmatch(r"([A-Za-z_0-9]+)\((.*)\)", line)
            if m:
                names[m.group(1)] = m.group(2)
            elif re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", line):
                bare.add(line)
    return names, bare


def test_exports_match_the_document():
    funcs, bare = documented()
    assert set(funcs) == PLANNED
    assert bare == RESULTS
    assert set(sd.integrate.__all__) == RESULTS | PUBLIC
    assert all(not hasattr(sd.integrate, n) for n in PLANNED - PUBLIC)  # milestones C4-C7


def test_signatures_match_the_document():
    funcs, _ = documented()
    for name in PUBLIC:
        got = str(inspect.signature(getattr(sd.integrate, name)))[1:-1]
        assert got == funcs[name], (name, got, funcs[name])


def test_not_in_the_first_version():
    # C2: the ODE solvers come later; ode, complex_ode, lebedev_rule, qmc_quad never; no old names
    funcs, bare = documented()
    for name in ("solve_ivp", "odeint", "solve_bvp", "ode", "complex_ode", "lebedev_rule", "qmc_quad", "romberg",
                 "quadrature", "simps", "trapz", "cumtrapz", "IntegrationWarning", "LowLevelCallable"):  # fmt: skip
        assert name not in funcs and name not in bare and not hasattr(sd.integrate, name)


def test_signatures_follow_the_naming_decisions():
    funcs, _ = documented()
    for name, sig in funcs.items():
        params = [p.strip().split("=")[0].strip() for p in sig.split(",")]
        assert ("args" in params) == (name in ELEMENTWISE), name  # C4, O39
        for gone in ("kwargs", "callback", "preserve_shape", "full_output", "epsabs", "epsrel", "maxp1", "maxlevel",
                     "minlevel", "maxterms", "tolerances", "show", "wopts", "workers", "cache_size"):  # fmt: skip
            assert gone not in params, (name, gone)  # C11, C12, C14
        if "atol" in params or "rtol" in params:
            assert "atol" in params and "rtol" in params, name
        assert params[0] in ("func", "f", "y", "rn"), name  # the integrand or the data first, positional
        assert "/" in params, name  # positional-only inputs, keyword-only options (§1.1)
