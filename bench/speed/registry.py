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
(threads="fixed": numpy's elementwise loops and scipy.special run on one thread), or at
each of the case's thread counts with its BLAS pool limited to that count
(threads="follow": every comparator whose work is mostly BLAS or LAPACK, so a 1-thread sd
row meets a 1-thread BLAS). An RRef is an R expression timed in the run's one R session,
which runs on one thread (OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1).

Results check (python -m bench.speed run --smoke): every case's sd result is compared
with its first Python comparator's and with every R and Boost comparator's (check.py).
A comparator that computes something else by design carries `differs` (why; it is not
compared and the reason goes into the table's note); a case whose results cannot be
compared value by value carries a Check (a normaliser, or `skip` with the reason).
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import re
from collections.abc import Callable, Iterable, Sequence

from bench import areas as _areas
from bench import cli

AREAS = _areas.AREAS

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
    differs: str = ""  # it computes something else by design: why (not value-checked; shown as a note)

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
    one thread (rsession.R_ENV).

    `sets`: the case data key of a list of input sets (dicts with the same keys as `data`)
    for a rotated case: R calls `expr` on each set in turn, as harness.rotate does."""

    label: str
    expr: str
    data: dict[str, str] = dataclasses.field(default_factory=dict)  # case data key -> R variable
    setup: str = ""
    packages: tuple[str, ...] = ()
    threads: str = "fixed"
    n: int | None = None
    differs: str = ""
    sets: str = ""
    value: str = ""  # for the results check: an R expression of the call's result `.v` (default .v)

    @property
    def library(self) -> str:
        return "R"


@dataclasses.dataclass(frozen=True)
class Check:
    """How the smoke run compares a case's results (check.py).

    sd, ref: turn sd's result, or a comparator's (given its label), into what is compared
    (default check.normalise: arrays, scalars, frames, fit and optimizer results).
    rtol, atol: the relative and absolute tolerance of one value (atol None: check.ATOL);
    agree: the share of the values both sides give (not NaN) that must agree. skip: no
    value comparison, and why (e.g. two random streams)."""

    sd: Callable[[object], object] | None = None
    ref: Callable[[str, object], object] | None = None
    rtol: float = 1e-6
    atol: float | None = None
    agree: float = 0.5
    skip: str = ""


@dataclasses.dataclass
class Case:
    area: str
    group: str  # a section of the area's table: an English key (part of the id)
    name: str  # what is called, e.g. "Normal.cdf", "solve", "ols fit"
    sd: Factory
    setup: Callable[[], dict] = dict
    size: str = ""  # shown in the table, e.g. "10^6", "(100000, 10)" (bench/README.md: case ids)
    n: int | None = None  # elements per call, for per-element costs
    refs: tuple = ()
    threads: tuple[int, ...] = (1, 10)
    quick: bool = False  # part of the quick subset (--quick)
    cost: tuple[str, str] | None = None  # (cost table, key) this case's 1-thread time feeds
    note: str = ""  # one short remark for the table
    timing: dict = dataclasses.field(default_factory=dict)  # harness.measure overrides
    teardown: Callable[[dict], None] | None = None  # undo global settings the setup made
    title: str = ""  # the group's section title in bench/<area>/speed.md (set by load from TITLES)
    check: Check = Check()  # how --smoke compares the results

    @property
    def notes(self) -> str:
        """The note with the comparators' `differs` reasons appended."""
        parts = [self.note] if self.note else []
        parts += [f"{r.label}: {r.differs}" for r in self.refs if r.differs]
        return "; ".join(parts)

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
        titles = getattr(mod, "TITLES", {})
        for c in cases:
            if c.area != area:
                raise ValueError(f"{c.id}: listed in cases/{area}.py")
            c.title = c.title or titles.get(c.group, c.group)
        out.extend(cases)
    seen: set[str] = set()
    for c in out:
        if c.id in seen:
            raise ValueError(f"duplicate case id {c.id!r}")
        seen.add(c.id)
    return out


def select(cases: Sequence[Case], *, match: Sequence[str] = (), quick: bool = False) -> list[Case]:
    """Filter cases: `match` patterns on the case id (bench.cli.matches: a glob, or plain
    text matched as a substring; any pattern may match), `quick` (the quick subset only)."""
    return [c for c in cases if (c.quick or not quick) and cli.matches(match, c.id)]


# ---------------------------------------------------------------- case ids
# One notation for the size part of an id (bench/README.md, "사례 id"):
#   COUNT [UNIT]   an element count: 10^k, 2^k or a plain integer (a power of ten from 10^3
#                  up is written 10^k); UNIT one of rows, points, matrices
#   SHAPE          (m, n) / (m,) / (b, m, n): plain integers, ", " between them
#   COUNT x SHAPE  a batch of B problems of one shape
#   n=COUNT        the dimension of an optimization problem
# No thousands separators, ASCII only; a named test problem goes into the name.
_COUNT = r"(?:10\^\d+|2\^\d+|\d+)"
_DIM = r"\d+"
_SHAPE = rf"\((?:{_DIM},|{_DIM}(?:, {_DIM})+)\)"
SIZE_RE = re.compile(rf"(?:{_COUNT}(?: (?:rows|points|matrices))?|{_SHAPE}|{_COUNT} x {_SHAPE}|n={_COUNT})")


def count(n: int) -> str:
    """An element count in the id notation: 10^k for a power of ten from 10^3 up."""
    k = len(str(n)) - 1
    return f"10^{k}" if k >= 3 and n == 10**k else str(n)


def size_problem(size: str) -> str | None:
    """Why `size` breaks the id notation, or None."""
    if not size:
        return None
    if not size.isascii():
        return "not ASCII"
    if not SIZE_RE.fullmatch(size):
        return "not COUNT [UNIT], (m, n), B x (m, n) or n=COUNT"
    for tok in re.findall(r"(?<![\^\d])\d+(?![\d)])", size.split("(")[0]):
        if len(tok) >= 4 and re.fullmatch(r"10*", tok):
            return f"write {tok} as 10^{len(tok) - 1}"
    return None
