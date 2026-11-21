"""sd.differentiate against scipy.differentiate 1.18.1 (docs/design/integrate.md §5, §7.5).

The plan of the area's cases: every spec names the sd.differentiate names it needs and its
milestone, and cases() keeps a case only once the milestone is in DONE and the names are public
(C2 brought all three functions). The comparator is scipy at its defaults (one
thread); the functions are numpy's for scipy and sd's for sd, elementwise on the batch. A case
returns the derivative (df, ddf), which the speed check compares with scipy's (the Jacobians and
Hessians with an absolute floor: their zero entries are both rounding noise of 1e-10).
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Callable

import numpy as np

from ..registry import Case, Check, Ref

DONE: tuple[str, ...] = ("C2",)  # the milestones whose cases run
BATCH_SIZES = ((10, "10"), (1000, "10^3"), (10**5, "10^5"))


@dataclasses.dataclass(frozen=True)
class Spec:
    milestone: str
    group: str
    name: str
    needs: tuple[str, ...]  # sd.differentiate names the sd side calls
    sd: Callable  # data -> zero-argument callable
    scipy: Callable  # data -> zero-argument callable
    setup: Callable = dict
    size: str = ""
    n: int | None = None
    quick: bool = False
    note: str = ""


def _quiet(f):
    def g():
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return f()

    return g


def _points(n):
    def setup():
        import stardust as sd

        x = np.linspace(0.5, 2.0, n)
        return {"x": x, "sx": sd.asarray(x)}

    return setup


def _rosen(x, xp):
    return xp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2, axis=0)


def _vec_np(x):
    return np.sin(x) * np.cumsum(x, axis=0)


def _vec_sd(x):
    import stardust as sd

    return sd.sin(x) * sd.cumulative_sum(x, axis=0)


def specs() -> list[Spec]:
    """Every planned case, public or not (the C0 scipy timings use them all)."""
    import scipy.differentiate as sdf

    import stardust as sd

    D = sd.differentiate
    out = []
    for n, size in BATCH_SIZES:
        out.append(Spec("C2", "derivative", "derivative exp", ("derivative",), lambda d: lambda: D.derivative(sd.exp, d["sx"]).df,
                        lambda d: _quiet(lambda: sdf.derivative(np.exp, d["x"]).df), setup=_points(n), size=size, n=n,
                        quick=n == 1000))  # fmt: skip
    out.append(Spec("C2", "derivative", "derivative exp one-sided", ("derivative",),
                    lambda d: lambda: D.derivative(sd.exp, d["sx"], step_direction=1).df,
                    lambda d: _quiet(lambda: sdf.derivative(np.exp, d["x"], step_direction=1).df), setup=_points(1000),
                    size="10^3", n=1000))  # fmt: skip
    for m in (3, 10, 30):
        x0 = list(np.linspace(0.8, 1.2, m)) if m > 3 else [1.2, 0.8, 1.1]
        out.append(Spec("C2", "multivariate", "jacobian Rosenbrock", ("jacobian",),
                        lambda d, x0=x0: lambda: D.jacobian(lambda x: _rosen(x, sd), sd.asarray(x0)).df,
                        lambda d, x0=x0: _quiet(lambda: sdf.jacobian(lambda x: _rosen(x, np), np.array(x0)).df),
                        size=f"n={m}"))  # fmt: skip
    for m in (3, 10):
        x0 = list(np.linspace(0.8, 1.2, m)) if m > 3 else [1.2, 0.8, 1.1]
        out.append(Spec("C2", "multivariate", "hessian Rosenbrock", ("hessian",),
                        lambda d, x0=x0: lambda: D.hessian(lambda x: _rosen(x, sd), sd.asarray(x0)).ddf,
                        lambda d, x0=x0: _quiet(lambda: sdf.hessian(lambda x: _rosen(x, np), np.array(x0)).ddf),
                        size=f"n={m}", note="33 calls" if m == 3 else ""))  # fmt: skip
    out.append(Spec("C2", "multivariate", "jacobian vector", ("jacobian",),
                    lambda d: lambda: D.jacobian(_vec_sd, sd.asarray(np.linspace(0.5, 1.5, 10))).df,
                    lambda d: _quiet(lambda: sdf.jacobian(_vec_np, np.linspace(0.5, 1.5, 10)).df), size="n=10",
                    note="f: R^10 → R^10"))  # fmt: skip
    return out


def _public(spec) -> bool:
    import stardust as sd

    return spec.milestone in DONE and all(n in sd.differentiate.__all__ for n in spec.needs)


def cases() -> list[Case]:
    return [
        Case(area="differentiate", group=s.group, name=s.name, size=s.size, n=s.n, setup=s.setup, sd=s.sd,
             refs=(Ref("scipy", s.scipy),), quick=s.quick, note=s.note,
             check=Check(atol=1e-8) if s.group == "multivariate" else Check())
        for s in specs()
        if _public(s)
    ]  # fmt: skip


TITLES = {"derivative": "배치 도함수", "multivariate": "야코비안과 헤시안"}
