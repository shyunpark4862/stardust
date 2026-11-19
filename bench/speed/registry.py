"""The case model and the registry: every area lists its cases in bench/speed/cases/<area>.py.

A case is one call at one size: stardust's call and the calls it is compared with, all
built from the same data. The data is built once per case and round (setup), and every
call is a factory: given the data, it returns the zero-argument callable to time, so
conversions (numpy to sd, adding a constant column, ...) happen outside the timing.

    Case(
        area="stats", group="continuous", name="Normal.cdf", size="10^5", n=100_000,
        setup=lambda: {"x": ...},
        sd=lambda d: lambda: X.cdf(d["sx"]),
        refs=(Ref("scipy", lambda d: lambda: norm.cdf(d["x"])),),
        quick=True, cost=("stats", "normal.cdf"),
    )

A comparator (Ref) is timed once per round at its libraries' default threads
(threads="fixed": numpy's elementwise loops and scipy.special run on one thread, their
BLAS on its default pool), or at each of the case's thread counts with its BLAS pool
limited to that count (threads="follow", as the regression benches did with
threadpoolctl). An RRef is an R expression timed in the run's one R session.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import importlib
import os
import re
from collections.abc import Callable, Iterable, Sequence

AREAS = ("array", "linalg", "frame", "random", "special", "stats", "regression", "optimize")

Factory = Callable[[dict], Callable[[], object]]


@dataclasses.dataclass(frozen=True)
class Ref:
    """A Python comparator. label names the column (e.g. "scipy", "sklearn lbfgs"); lib
    is the library recorded in the results (defaults to the label's first word)."""

    label: str
    make: Factory
    threads: str = "fixed"  # "fixed" or "follow"
    lib: str | None = None
    n: int | None = None  # elements its call handles, when not the case's n (a sample of a slow input)

    def __post_init__(self):
        if self.threads not in ("fixed", "follow"):
            raise ValueError(f"Ref {self.label!r}: threads must be 'fixed' or 'follow'")

    @property
    def library(self) -> str:
        return self.lib or self.label.split()[0]


@dataclasses.dataclass(frozen=True)
class RRef:
    """An R comparator: `expr` evaluated each call, after the case's arrays named in
    `data` (case data key -> R variable name) are loaded and `setup` has run. R runs on
    one thread (its reference BLAS)."""

    label: str
    expr: str
    data: dict[str, str] = dataclasses.field(default_factory=dict)  # case data key -> R variable
    setup: str = ""
    packages: tuple[str, ...] = ()
    threads: str = "fixed"
    n: int | None = None

    @property
    def library(self) -> str:
        return "R"


@dataclasses.dataclass
class Case:
    area: str
    group: str  # a section of the area's table
    name: str  # what is called, e.g. "Normal.cdf", "solve", "ols fit"
    sd: Factory
    setup: Callable[[], dict] = dict
    size: str = ""  # shown in the table, e.g. "10^6", "(100,000, 10)"
    n: int | None = None  # elements per call, for per-element costs
    refs: tuple = ()
    threads: tuple[int, ...] = (1, 10)
    quick: bool = False  # part of the quick subset (--quick)
    cost: tuple[str, str] | None = None  # (cost table, key) this case's 1-thread time feeds
    note: str = ""  # one short remark for the table
    timing: dict = dataclasses.field(default_factory=dict)  # harness.measure overrides
    teardown: Callable[[dict], None] | None = None  # undo global settings the setup made

    def __post_init__(self):
        if self.area not in AREAS:
            raise ValueError(f"unknown area {self.area!r}")

    @property
    def id(self) -> str:
        base = f"{self.area}/{self.group}/{self.name}"
        return f"{base} [{self.size}]" if self.size else base


def load(areas: Iterable[str] | None = None) -> list[Case]:
    """The cases of the given areas (all areas that have a case module by default).

    SDBENCH_CASES_MODULE=<module> replaces the registry by that module's cases() (the
    harness tests use it to run fake cases through a real worker)."""
    override = os.environ.get("SDBENCH_CASES_MODULE")
    if override:
        wanted = set(areas or AREAS)
        return [c for c in importlib.import_module(override).cases() if c.area in wanted]
    out: list[Case] = []
    for area in areas or AREAS:
        try:
            mod = importlib.import_module(f"bench.speed.cases.{area}")
        except ModuleNotFoundError as e:
            if e.name == f"bench.speed.cases.{area}":
                continue
            raise
        cases = mod.cases()
        for c in cases:
            if c.area != area:
                raise ValueError(f"{c.id}: listed in cases/{area}.py")
        out.extend(cases)
    seen: set[str] = set()
    for c in out:
        if c.id in seen:
            raise ValueError(f"duplicate case id {c.id!r}")
        seen.add(c.id)
    return out


def select(
    cases: Sequence[Case], *, match: Sequence[str] = (), quick: bool = False, groups: Sequence[str] = ()
) -> list[Case]:
    """Filter cases: `match` patterns (a glob if it contains * or ?, else a regular
    expression searched in the case id; any pattern may match), `groups` (exact group
    names), `quick` (the quick subset only)."""
    out = []
    for c in cases:
        if quick and not c.quick:
            continue
        if groups and c.group not in groups:
            continue
        if match and not any(_matches(p, c.id) for p in match):
            continue
        out.append(c)
    return out


def _matches(pattern: str, text: str) -> bool:
    if any(ch in pattern for ch in "*?"):
        return fnmatch.fnmatchcase(text, pattern) or fnmatch.fnmatchcase(text, f"*{pattern}*")
    return re.search(pattern, text) is not None
