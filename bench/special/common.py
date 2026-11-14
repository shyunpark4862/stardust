"""What the sd.special benchmarks share: the functions under test and their scipy
counterparts, inputs drawn from the reference files, and timing.

Run the scripts from the repository root (.venv/bin/python bench/special/<script>.py).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def use_build(site):
    """Import stardust from an unpacked wheel at `site` instead of the editable build
    (bench/special/s0_operands.py compares a baseline commit this way). Call before
    importing stardust."""
    sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in type(f).__module__.lower()]
    sys.path.insert(0, str(site))


ALIAS = {"zeta_hurwitz": "zeta"}  # reference files named apart from their function


# one output of a several-output function (tests/special/refdata.py PART)
PART = {"airy_ai": ("airy", 0), "airy_aip": ("airy", 1), "airy_bi": ("airy", 2), "airy_bip": ("airy", 3)}


def sd_function(name):
    import stardust as sd

    if name in PART:
        base, i = PART[name]
        f = getattr(sd.special, base)
        return lambda *a: f(*a)[i]
    name = ALIAS.get(name, name)
    return getattr(sd.special, name, None) or getattr(sd.special._core, name)


# The scipy (or numpy) function each sd.special function is compared with; by default
# scipy.special.<name>. None: nothing to compare with.
COUNTERPART = {
    "_cf_tan": "numpy.tan",
    "airy_ai": "scipy.special.airy[0]", "airy_aip": "scipy.special.airy[1]",
    "airy_bi": "scipy.special.airy[2]", "airy_bip": "scipy.special.airy[3]",
    "zeta_hurwitz": "scipy.special.zeta",
}


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
    path = counterpart(name)
    if path is None:
        return None
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
    pool = np.flatnonzero(guaranteed) if guaranteed.any() else np.arange(guaranteed.size)  # betaln: no guaranteed region
    idx = np.random.default_rng(seed).choice(pool, n)
    return tuple(np.ascontiguousarray(x[idx]) for x in xs)


def best_ns(f, repeat=7, number=1):
    """The fastest of `repeat` timings of `number` calls of f, per call, in ns."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        for _ in range(number):
            f()
        best = min(best, (time.perf_counter_ns() - t0) / number)
    return best


def machine():
    import platform

    return f"{platform.machine()} {platform.mac_ver()[0]}, python {platform.python_version()}, {os.cpu_count()} cpus"
