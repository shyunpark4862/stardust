"""The test data of sd.smooth (docs/design/smooth.md §7.1): generated from formulas and fixed seeds,
nothing downloaded. The tests, the reference tool (tools/smooth_ref.py), the probe of scipy, statsmodels,
scikit-learn and R (tools/smooth_probe.py) and the measurements take their data from here, by id.

Kinds, each with its own getter:

    scatters   1-D regression data (x, y, w) with evaluation points xe: a uniform grid, uniform random x,
               clusters (gaps that differ by 1e8), repeated x (all equal x too), unsorted x, a very narrow
               and a very wide range, a large shift, subnormal spacing, 1 to 5 points; y a smooth function
               with noise, outliers, a step, a constant, an exact line and an exact quadratic, more than
               half the residuals zero, scales 1e300 and 1e-300; weights (zeros among them)
    multis     regression data (X (n, p), y, w) with points Xe: p = 2, 3, 5 uniform, columns whose scales
               differ by 1e6, points on a line (rank deficient), a lattice (many equal distances)
    densities  samples (x (n,) or (n, p), w) with points xe: normal, two modes, a heavy tail, a bounded
               sample (exponential), a discrete one, weights, degenerate ones (all equal, 2-D points on a
               line), correlated 2-D and 3-D normals; xe reaches far into the tails
    signals    arrays for Savitzky-Golay: a cubic at integers (reproduced exactly), a signal with heavy
               cancellation, noise, a batch (two axes), complex values, a line as short as the window
    SAVGOL     (window_length, polyorder, deriv, pos) of the coefficient references, the high orders where
               a Vandermonde least squares goes wrong among them

    import tests.smooth.samples as S
    s = S.scatter("random/50")        # cached; S.scatter_ids(tag="ties") lists ids
    s.x, s.y, s.w, s.xe               # numpy arrays (w None when there are no weights)

Every record and reference names the data it was made from by `digest`.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math

import numpy as np

SEED = 20261123
NAN, INF = math.nan, math.inf


def hash_int(k) -> int:
    return int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:8], "little")


def rng(*key) -> np.random.Generator:
    """The generator of a key: the same numbers on every machine (PCG64 of SEED and the key)."""
    return np.random.Generator(np.random.PCG64([SEED, *[hash_int(k) for k in key]]))


def digest(*arrays) -> str:
    """SHA-256 of arrays as given (dtype, shape and bytes, one after another; None counts)."""
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(a)
        h.update(repr((str(a.dtype), a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _frozen(*arrays):
    for a in arrays:
        if a is not None:
            a.setflags(write=False)


# ---------------------------------------------------------------- scatters (1-D regression data)


@dataclasses.dataclass(frozen=True)
class Scatter:
    id: str
    x: np.ndarray  # (n,), in the order given (not sorted unless tagged "sorted")
    y: np.ndarray  # (n,)
    w: np.ndarray | None  # (n,) weights >= 0, or None
    xe: np.ndarray  # evaluation points: inside, at data points and outside the range
    tags: frozenset
    note: str = ""

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def digest(self) -> str:
        return digest(self.x, self.y, self.w, self.xe)


def _signal(x):
    return np.sin(x)


def _xe(x, g, m=9):
    """Evaluation points of data x: random inside the range, three data points, one outside each end."""
    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo if hi > lo else max(abs(lo), 1.0)
    inside = lo + (hi - lo) * g.random(m) if hi > lo else np.full(1, lo)
    picks = x[np.unique(np.linspace(0, x.size - 1, 3).astype(int))]
    return np.sort(np.concatenate([inside, picks, [lo - 0.125 * span, hi + 0.125 * span]]))


def _base(n, key, grid=False):
    g = rng("scatter", key, n)
    x = np.linspace(0.0, 10.0, n) if grid else np.sort(10.0 * g.random(n))
    y = _signal(x) + 0.3 * g.standard_normal(n)
    return g, x, y


def _scatter_table():
    t = {}

    def add(sid, x, y, tags, w=None, note="", g=None):
        g = g or rng("xe", sid)
        x, y = np.asarray(x, float), np.asarray(y, float)
        xe = _xe(x, g)
        _frozen(x, y, w, xe)
        t[sid] = Scatter(sid, x, y, w, xe, frozenset(tags), note)

    for n in (40,):
        g, x, y = _base(n, "uniform", grid=True)
        add(f"uniform/{n}", x, y, {"sorted", "distinct", "plain"})
    for n in (30, 50, 200):
        g, x, y = _base(n, "random")
        add(
            f"random/{n}",
            x,
            y,
            {"sorted", "distinct", "plain"} | ({"probe"} if n == 200 else set()),
            note="x uniform on [0, 10], y = sin x + N(0, 0.3^2): the data of the design draft's probes",
        )
    g, x, y = _base(40, "clustered")
    x = np.sort(np.concatenate([np.arange(4.0)[:, None] * 2.5 + 1e-8 * g.random((4, 10))]).ravel())
    add(
        "clustered/40",
        x,
        _signal(x) + 0.3 * g.standard_normal(40),
        {"sorted", "distinct", "clustered"},
        note="four clusters 2.5 apart, 1e-8 wide: spacings that differ by 1e8",
    )
    g, x, y = _base(30, "ties")
    x = np.round(x)
    add("ties/30", x, y, {"sorted", "ties"}, note="x rounded to integers: every x repeated")
    g = rng("scatter", "allsame")
    add("allsame/8", np.full(8, 2.5), g.standard_normal(8), {"sorted", "ties", "degenerate"}, note="one x")
    g, x, y = _base(30, "unsorted")
    p = g.permutation(30)
    add("unsorted/30", x[p], y[p], {"distinct", "unsorted"})
    g, x, y = _base(30, "scale")
    add("narrow/30", x * 1e-8, y, {"sorted", "distinct", "scale"}, note="x of random/30's kind times 1e-8")
    add("wide/30", x * 1e8, y, {"sorted", "distinct", "scale"}, note="the same x times 1e8")
    add("shifted/30", x + 1e8, y, {"sorted", "distinct", "scale"}, note="the same x plus 1e8 (27 bits lost)")
    g = rng("scatter", "subnormal")
    add(
        "subnormal/12",
        np.arange(1.0, 13.0) * 2.0**-1070,
        g.standard_normal(12),
        {"sorted", "distinct", "subnormal"},
        note="spacing 2^-1070",
    )
    for n in (1, 2, 3, 5):
        g = rng("scatter", "tiny", n)
        add(f"tiny/{n}", np.arange(n) * 1.5 + 0.25, g.standard_normal(n), {"sorted", "distinct", "tiny"})
    # the kinds of y, on dyadic x so the exact polynomials are exact doubles
    xd = np.arange(20) / 8.0
    add("line/20", xd, 2.0 * xd + 1.0, {"sorted", "distinct", "exact", "line"}, note="y = 2x + 1 exactly")
    add(
        "quadratic/20",
        xd,
        xd * xd - 3.0 * xd + 2.0,
        {"sorted", "distinct", "exact", "quadratic"},
        note="y = x^2 - 3x + 2 exactly",
    )
    add("constant/20", xd, np.full(20, 0.75), {"sorted", "distinct", "exact", "constant"})
    g, x, y = _base(40, "outliers")
    y[[5, 17, 31]] += (25.0, -40.0, 30.0)
    add("outliers/40", x, y, {"sorted", "distinct", "outliers"}, note="three points moved by 25 to 40")
    g, x, y = _base(40, "step")
    add("step/40", x, np.where(x < 5.0, 0.0, 1.0) + 0.05 * g.standard_normal(40), {"sorted", "distinct", "step"})
    g = rng("scatter", "halfzero")
    xh = np.arange(21) / 4.0
    yh = 0.5 * xh - 1.0
    yh[::3] += g.standard_normal(7)
    add(
        "halfzero/21",
        xh,
        yh,
        {"sorted", "distinct", "halfzero"},
        note="14 of 21 points exactly on a line: the median absolute residual of a robust step is 0",
    )
    g, x, y = _base(20, "magnitude")
    add("big/20", x, y * 1e300, {"sorted", "distinct", "magnitude"}, note="y near 1e300")
    add("small/20", x, y * 1e-300, {"sorted", "distinct", "magnitude"}, note="y near 1e-300")
    g, x, y = _base(30, "weights")
    w = 0.5 + 1.5 * g.random(30)
    w[[3, 4, 20]] = 0.0
    _frozen(w)
    add("weights/30", x, y, {"sorted", "distinct", "weights"}, w=w, note="weights in [0.5, 2], three of them 0")
    return t


@functools.cache
def _scatters():
    return _scatter_table()


def scatter(sid: str) -> Scatter:
    return _scatters()[sid]


def scatter_ids(tag=None, exclude=()) -> list[str]:
    return [k for k, s in _scatters().items() if (tag is None or tag in s.tags) and not (set(exclude) & s.tags)]


# ---------------------------------------------------------------- multis ((n, p) regression data)


@dataclasses.dataclass(frozen=True)
class Multi:
    id: str
    X: np.ndarray  # (n, p)
    y: np.ndarray  # (n,)
    w: np.ndarray | None
    Xe: np.ndarray  # (m, p)
    tags: frozenset
    note: str = ""

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def p(self) -> int:
        return int(self.X.shape[1])

    @property
    def digest(self) -> str:
        return digest(self.X, self.y, self.w, self.Xe)


def _multi_table():
    t = {}

    def add(mid, X, tags, note="", scale=None):
        g = rng("multi", mid)
        y = np.sin(X[:, 0] * (1.0 if scale is None else 1.0 / scale[0])) + 0.5 * X[:, -1] ** 2 / (
            1.0 if scale is None else scale[-1] ** 2
        )
        y = y + 0.2 * g.standard_normal(X.shape[0])
        lo, hi = X.min(axis=0), X.max(axis=0)
        Xe = np.concatenate([lo + (hi - lo) * g.random((6, X.shape[1])), X[[0, X.shape[0] // 2]]])
        _frozen(X, y, Xe)
        t[mid] = Multi(mid, X, y, None, Xe, frozenset(tags), note)

    for p, n in ((2, 60), (3, 60), (5, 80)):
        add(f"uniform{p}/{n}", 2.0 * rng("multi", "X", p, n).random((n, p)), {"plain"})
    sc = np.array([1.0, 1e6])
    add(
        "scales2/50",
        2.0 * rng("multi", "X", "scales").random((50, 2)) * sc,
        {"scales"},
        scale=sc,
        note="the second column is 1e6 times the first's scale",
    )
    u = 2.0 * rng("multi", "X", "lowrank").random(40)
    add("lowrank2/40", np.stack([u, 0.5 * u + 0.25], axis=1), {"rank_deficient"}, note="every point on one line")
    a = np.arange(7.0) / 4.0
    add(
        "lattice2/49",
        np.stack(np.meshgrid(a, a, indexing="ij"), axis=-1).reshape(-1, 2),
        {"lattice"},
        note="a 7 x 7 lattice: many neighbours at equal distances",
    )
    return t


@functools.cache
def _multis():
    return _multi_table()


def multi(mid: str) -> Multi:
    return _multis()[mid]


def multi_ids(tag=None) -> list[str]:
    return [k for k, s in _multis().items() if tag is None or tag in s.tags]


# ---------------------------------------------------------------- densities


@dataclasses.dataclass(frozen=True)
class Density:
    id: str
    x: np.ndarray  # (n,) or (n, p): points are rows
    w: np.ndarray | None
    xe: np.ndarray  # (m,) or (m, p), far tail points included
    bounds: tuple | None  # the support (lo, hi) of a bounded sample
    tags: frozenset
    note: str = ""

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def p(self) -> int:
        return 1 if self.x.ndim == 1 else int(self.x.shape[1])

    @property
    def digest(self) -> str:
        return digest(self.x, self.w, self.xe)


def _density_table():
    t = {}

    def add(did, x, tags, w=None, bounds=None, note=""):
        g = rng("density", "xe", did)
        x = np.asarray(x, float)
        if x.ndim == 1:
            lo, hi = float(x.min()), float(x.max())
            s = (hi - lo) or 1.0
            xe = np.sort(
                np.concatenate([lo + s * g.random(8), x[:2], [lo - 0.5 * s, hi + 0.5 * s, lo - 30 * s, hi + 30 * s]])
            )
        else:
            lo, hi = x.min(axis=0), x.max(axis=0)
            s = np.where(hi > lo, hi - lo, 1.0)
            xe = np.concatenate([lo + s * g.random((6, x.shape[1])), x[:2], (lo - 0.5 * s)[None], (hi + 30 * s)[None]])
        _frozen(x, w, xe)
        t[did] = Density(did, x, w, xe, bounds, frozenset(tags), note)

    g = rng("density", "normal")
    add("normal/50", g.standard_normal(50), {"1d", "plain"})
    add(
        "normal/200",
        1.25 * rng("density", "normal", 200).standard_normal(200),
        {"1d", "plain", "probe"},
        note="the bandwidth probes' sample",
    )
    g = rng("density", "bimodal")
    add("bimodal/80", np.concatenate([g.standard_normal(50) - 2.0, 0.5 * g.standard_normal(30) + 2.0]), {"1d", "plain"})
    g = rng("density", "heavy")
    add("heavytail/60", g.standard_normal(60) / np.sqrt(g.chisquare(3, 60) / 3.0), {"1d", "plain"}, note="t with 3 df")
    g = rng("density", "exponential")
    add("exponential/60", g.exponential(1.0, 60), {"1d", "bounded"}, bounds=(0.0, INF), note="support [0, inf)")
    g = rng("density", "discrete")
    add("discrete/60", g.poisson(3.0, 60).astype(float), {"1d", "discrete"}, note="integers: many equal values")
    g = rng("density", "weighted")
    w = g.random(40) + 0.1
    _frozen(w)
    add("weighted/40", g.standard_normal(40), {"1d", "weights"}, w=w)
    add("allsame/10", np.full(10, 1.5), {"1d", "degenerate"}, note="one value: a rule's bandwidth is 0")
    for p, n in ((2, 60), (3, 50)):
        g = rng("density", "mv", p)
        a = np.tril(0.6 * g.standard_normal((p, p))) + np.eye(p)
        add(f"normal{p}/{n}", g.standard_normal((n, p)) @ a.T, {"nd", "plain"}, note="correlated normal")
    u = rng("density", "line").standard_normal(20)
    add("line2/20", np.stack([u, 2.0 * u - 1.0], axis=1), {"nd", "degenerate"}, note="2-D points on one line")
    return t


@functools.cache
def _densities():
    return _density_table()


def density(did: str) -> Density:
    return _densities()[did]


def density_ids(tag=None, exclude=()) -> list[str]:
    return [k for k, s in _densities().items() if (tag is None or tag in s.tags) and not (set(exclude) & s.tags)]


# ---------------------------------------------------------------- signals (Savitzky-Golay)


@dataclasses.dataclass(frozen=True)
class Signal:
    id: str
    x: np.ndarray  # float64 or complex128; the filter runs along the last axis
    tags: frozenset
    note: str = ""

    @property
    def digest(self) -> str:
        return digest(self.x)


def _signal_table():
    t = {}

    def add(sid, x, tags, note=""):
        x = np.asarray(x)
        _frozen(x)
        t[sid] = Signal(sid, x, frozenset(tags), note)

    k = np.arange(40.0)
    add(
        "cubic/40",
        0.125 * k**3 - 2.0 * k * k + 3.0 * k - 7.0,
        {"real", "exact"},
        note="a cubic at the integers: every filter of polyorder >= 3 returns it (and its derivatives) exactly",
    )
    g = rng("signal", "cancel")
    add("cancel/50", 1e15 + np.cumsum(g.standard_normal(50)), {"real"}, note="1e15 plus a random walk")
    g = rng("signal", "noisy")
    add("noisy/100", np.sin(np.arange(100) / 7.0) + 0.2 * g.standard_normal(100), {"real", "plain"})
    add("batch/6x30", rng("signal", "batch").standard_normal((6, 30)), {"real", "batch"}, note="two axes")
    g = rng("signal", "complex")
    add("complex/40", g.standard_normal(40) + 1j * g.standard_normal(40), {"complex"})
    add("short/7", rng("signal", "short").standard_normal(7), {"real", "short"}, note="as long as a window of 7")
    return t


@functools.cache
def _signals():
    return _signal_table()


def signal(sid: str) -> Signal:
    return _signals()[sid]


def signal_ids(tag=None) -> list[str]:
    return [k for k, s in _signals().items() if tag is None or tag in s.tags]


# (window_length, polyorder, deriv, pos): pos None is the centre (scipy's default, half a step off for an even window)
SAVGOL = (
    (5, 2, 0, None),
    (7, 3, 0, None),
    (7, 3, 1, None),
    (7, 3, 2, None),
    (7, 3, 3, None),
    (7, 3, 4, None),
    (11, 3, 0, None),
    (11, 3, 0, 0),
    (11, 3, 0, 10),
    (11, 3, 1, 2),
    (10, 3, 0, None),
    (10, 3, 1, None),
    (4, 3, 0, None),
    (1, 0, 0, None),
    (3, 2, 0, None),
    (25, 5, 0, None),
    (25, 5, 2, None),
    (51, 45, 0, None),
    (101, 3, 0, None),
    (101, 6, 0, None),
    (101, 8, 0, None),
    (101, 10, 0, None),
    (101, 10, 3, None),
    (301, 30, 0, None),
    (1001, 5, 0, None),
    (1001, 12, 1, None),
)
# the filters of the exact filtering references (window_length, polyorder, deriv), every mode on every real signal
# long enough
SAVGOL_FILTERS = ((5, 2, 0), (7, 3, 0), (7, 3, 1), (10, 3, 0), (11, 5, 2))
SAVGOL_MODES = ("interp", "mirror", "nearest", "constant", "wrap")
SAVGOL_CVAL = 0.5  # the cval of the 'constant' references


def savgol_id(case) -> str:
    n, d, deriv, pos = case
    return f"{n},{d},{deriv},{'c' if pos is None else pos}"
