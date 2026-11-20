"""What the sd.special speed cases need of each function under test: the sd function, its
scipy (or numpy) counterpart, and inputs drawn from the reference files
(tests/special/refdata.py). Moved here from bench/special/common.py (at commit 4151d2a).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


ALIAS = {"zeta_hurwitz": "zeta"}  # reference files named apart from their function


# one output of a several-output function (tests/special/refdata.py PART)
PART = {"airy_ai": ("airy", 0), "airy_aip": ("airy", 1), "airy_bi": ("airy", 2), "airy_bip": ("airy", 3)}
# a keyword option, or a complex argument (tests/special/refdata.py KW); scipy's lambertw
# is complex: its real part for the real files
KW = {
    "lambertw_m1": ("lambertw", {"k": -1}),
    "erf_complex": ("erf", {}),
    "erfc_complex": ("erfc", {}),
    "erfcx_complex": ("erfcx", {}),
    "loggamma_complex": ("loggamma", {}),
    "lambertw_c0": ("lambertw", {"k": 0}),
    "lambertw_cm1": ("lambertw", {"k": -1}),
    "lambertw_c1": ("lambertw", {"k": 1}),
    "lambertw_c4": ("lambertw", {"k": 4}),
}


def sd_function(name):
    import stardust as sd

    if name in PART:
        base, i = PART[name]
        f = getattr(sd.special, base)
        return lambda *a: f(*a)[i]
    if name in KW:
        base, kw = KW[name]
        f = getattr(sd.special, base)
        return lambda *a: f(*a, **kw)
    name = ALIAS.get(name, name)
    return getattr(sd.special, name, None) or getattr(sd.special._core, name)


# The scipy (or numpy) function each sd.special function is compared with; by default
# scipy.special.<name>. None: nothing to compare with.
COUNTERPART = {
    "_cf_tan": "numpy.tan",
    "airy_ai": "scipy.special.airy[0]",
    "airy_aip": "scipy.special.airy[1]",
    "airy_bi": "scipy.special.airy[2]",
    "airy_bip": "scipy.special.airy[3]",
    "lambertw_m1": "scipy.special.lambertw(k=-1).real",
    "lambertw": "scipy.special.lambertw.real",
    "zeta_hurwitz": "scipy.special.zeta",
    "erf_complex": "scipy.special.erf",
    "erfc_complex": "scipy.special.erfc",
    "erfcx_complex": "scipy.special.erfcx",
    "loggamma_complex": "scipy.special.loggamma",
    "lambertw_c0": "scipy.special.lambertw(k=0)",
    "lambertw_cm1": "scipy.special.lambertw(k=-1)",
    "lambertw_c1": "scipy.special.lambertw(k=1)",
    "lambertw_c4": "scipy.special.lambertw(k=4)",
    # T2: scipy's chndtr takes x first; the upper tails exist only as scipy.stats sf
    "chndtr": "scipy.special.chndtr(x, df, nc)",
    "chndtrix": "scipy.special.chndtrix(p, df, nc)",
    "chndtrc": "scipy.stats.ncx2.sf",
    "nctdtrc": "scipy.stats.nct.sf",
    "ncfdtrc": "scipy.stats.ncf.sf",
}


def _t2_counterpart(name):
    import scipy.special as sc
    import scipy.stats as st

    return {
        "chndtr": lambda df, nc, x: sc.chndtr(x, df, nc),
        "chndtrix": lambda df, nc, p: sc.chndtrix(p, df, nc),
        "chndtrc": lambda df, nc, x: st.ncx2.sf(x, df, nc),
        "nctdtrc": lambda df, nc, t: st.nct.sf(t, df, nc),
        "ncfdtrc": lambda dfn, dfd, nc, f: st.ncf.sf(f, dfn, dfd, nc),
    }.get(name)


def counterpart(name):
    """The dotted name of what `name` is compared with, or None."""
    return COUNTERPART.get(name, f"scipy.special.{name}")


def scipy_function(name):
    import importlib

    if name in PART:
        import scipy.special

        base, i = PART[name]
        f = getattr(scipy.special, base)
        return lambda *a: f(*a)[i]
    if name.startswith("lambertw_c"):
        import scipy.special

        k = KW[name][1]["k"]
        return lambda z: scipy.special.lambertw(z, k)
    if name in ("lambertw", "lambertw_m1"):
        import numpy
        import scipy.special

        k = -1 if name == "lambertw_m1" else 0
        return lambda x: numpy.real(scipy.special.lambertw(x, k))
    if _t2_counterpart(name) is not None:
        return _t2_counterpart(name)
    path = counterpart(name)
    if path is None:
        return None
    if name.endswith("_complex"):
        path = f"scipy.special.{KW[name][0]}"
    module, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


def ref_names():
    sys.path.insert(0, str(ROOT))
    from tests.special import refdata

    return refdata.names()


def ref_load(name):
    sys.path.insert(0, str(ROOT))
    from tests.special import refdata

    return refdata.load(name)


def sample_inputs(name, n, seed=0):
    """n inputs drawn (with replacement) from the guaranteed-region reference inputs:
    the distribution the function is measured on."""
    xs, _, _, guaranteed = ref_load(name)
    pool = (
        np.flatnonzero(guaranteed) if guaranteed.any() else np.arange(guaranteed.size)
    )  # betaln: no guaranteed region
    idx = np.random.default_rng(seed).choice(pool, n)
    return tuple(np.ascontiguousarray(x[idx]) for x in xs)
