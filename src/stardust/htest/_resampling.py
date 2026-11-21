"""Resampling and Monte Carlo (milestone H9; docs/api/htest.md §10, docs/design/htest.md §3.10).

permutation_test, monte_carlo_test and bootstrap with a user statistic; the configuration
objects PermutationMethod, MonteCarloMethod and BootstrapMethod that the tests of H1-H8 take as
their method= (scipy's); bws_test, multiscale_graphcorr and goodness_of_fit (decisions H117-H125
in docs/design/htest.md §1.1).

The rules (scipy's names, this library's conventions):
- data is a sequence of samples (positional); axis=None flattens each sample (H123), an int
  reads the samples along that axis and broadcasts the other axes as a batch. The statistic is
  called with axis=-1 on sd arrays: the observed one on (*batch, n_s), permutation_test's and
  monte_carlo_test's resamples on (B, *batch, n_s) (B resamples a call), bootstrap's on
  (*batch, B, n_s), and must give the leading shape. vectorized=None reads whether the
  statistic takes axis; a statistic without it is called on each 1-D lane.
- Random resamples need rng= (an sd.random.Generator; no unseeded default, H3, H117). Resample
  j is a function of the generator's key, its position and j only: the same bits on any number
  of threads and for any batch (htest/resampling.cpp). monte_carlo_test's rvs(size, *, rng)
  (H118) draws each sample from its own child of rng (rng.spawn), so the batches do not change
  the draws of a generator that draws element by element (sd.stats' sample()).
- batch=None bounds the memory: a call's resampled data holds at most 2^22 values.
- p-values are scipy's: the share of the null distribution at least as extreme as the observed
  statistic within a relative 100 eps, (b + 1)/(n + 1) for random resamples and b/n for the
  exact enumeration (permutation_test when every permutation fits in n_resamples), the
  two-sided one twice the smaller one-sided p-value, at most 1. A NaN statistic gives NaN.
- No warnings: a degenerate bootstrap interval (BCa of a constant statistic) is NaN.
"""

import inspect
import math
import numbers

from .. import _calls
from .. import _core as _sd
from .. import _summary as _S
from .. import special as _special
from ..random import _check_alloc
from ..stats import _distribution as _dist
from ..stats._descriptive import _as_array, _scalar_value, _type_name
from . import _common as C
from ._result import _TestResult

_k = _sd.htest
_ALT = {"two-sided": 0, "less": 1, "greater": 2}
_PTYPES = ("independent", "samples", "pairings")
_PCODE = {"independent": 0, "pairings": 1, "samples": 2}
_BOOT = ("percentile", "basic", "BCa")
_BATCH_VALUES = 1 << 22  # the resampled values of one statistic call when batch=None
_EXACT_CAP = 10**7  # permutation_test(n_resamples=inf): the permutations it enumerates at most
_BWS_CAP = 10**6  # bws_test(method=None): the permutations it enumerates at most (H122)


# ---- the configuration objects (H117) -----------------------------------------------------


def _n_resamples(v, name, *, inf=False, minimum=1):
    if inf and isinstance(v, float) and math.isinf(v) and v > 0:
        return math.inf
    return C.integer(v, name, "n_resamples", minimum=minimum)


def _batch(v, name):
    return None if v is None else C.integer(v, name, "batch", minimum=1)


def _rvs(v, name):
    """rvs: None, a callable rvs(size, *, rng), or a list or tuple of them (one per sample)."""
    if v is None or callable(v):
        return v
    if isinstance(v, (list, tuple)) and v and all(callable(f) for f in v):
        return tuple(v)
    raise TypeError(f"{name}(): 'rvs' must be a callable or a list or tuple of callables, not {_type_name(v)}")


class _ResamplingMethod:
    __slots__ = ("_n_resamples", "_batch", "_rng")

    @property
    def n_resamples(self):
        """The number of resamples (read-only)."""
        return self._n_resamples

    @property
    def batch(self):
        """The resamples a statistic call takes (None: bounded by the memory; read-only)."""
        return self._batch

    @property
    def rng(self):
        """The sd.random.Generator the resamples draw from (read-only)."""
        return self._rng

    def _fields(self):
        return [("n_resamples", self._n_resamples), ("batch", self._batch), ("rng", self._rng)]

    def __repr__(self):
        parts = ", ".join(f"{k}={v!r}" for k, v in self._fields())
        return f"{type(self).__name__}({parts})"


class PermutationMethod(_ResamplingMethod):
    """The permutation method of a test's method= (scipy's PermutationMethod; H117): the test's
    statistic under every permutation when there are at most n_resamples of them (no rng
    needed), else under n_resamples random ones drawn from rng."""

    __slots__ = ()

    def __init__(self, n_resamples=9999, batch=None, *, rng=None):
        name = "PermutationMethod"
        self._n_resamples = _n_resamples(n_resamples, name, inf=True)
        self._batch = _batch(batch, name)
        self._rng = None if rng is None else C.rng(rng, name)


class MonteCarloMethod(_ResamplingMethod):
    """The Monte Carlo method of a test's method= (scipy's MonteCarloMethod; H117, H118): the
    test's statistic of n_resamples samples drawn by rvs(size, *, rng) (one callable, or one per
    sample; None: the standard normal), each sample from its own child of rng."""

    __slots__ = ("_rvs",)

    def __init__(self, n_resamples=9999, batch=None, rvs=None, *, rng):
        name = "MonteCarloMethod"
        self._n_resamples = _n_resamples(n_resamples, name)
        self._batch = _batch(batch, name)
        self._rvs = _rvs(rvs, name)
        self._rng = C.rng(rng, name)

    @property
    def rvs(self):
        """The samplers (None: the standard normal; read-only)."""
        return self._rvs

    def _fields(self):
        f = super()._fields()
        return f[:2] + [("rvs", self._rvs)] + f[2:]


class BootstrapMethod(_ResamplingMethod):
    """The bootstrap of a confidence interval's method= (scipy's BootstrapMethod; H117):
    n_resamples resamples drawn from rng and the 'percentile', 'basic' or 'BCa' interval."""

    __slots__ = ("_method",)

    def __init__(self, n_resamples=9999, batch=None, method="BCa", *, rng):
        name = "BootstrapMethod"
        self._n_resamples = _n_resamples(n_resamples, name)
        self._batch = _batch(batch, name)
        self._method = C.choice(method, name, "method", _BOOT)
        self._rng = C.rng(rng, name)

    @property
    def method(self):
        """The interval: 'percentile', 'basic' or 'BCa' (read-only)."""
        return self._method

    def _fields(self):
        f = super()._fields()
        return f[:2] + [("method", self._method)] + f[2:]


PermutationMethod.__init__ = _calls.method("PermutationMethod")(PermutationMethod.__init__)
MonteCarloMethod.__init__ = _calls.method("MonteCarloMethod")(MonteCarloMethod.__init__)
BootstrapMethod.__init__ = _calls.method("BootstrapMethod")(BootstrapMethod.__init__)


# ---- the result classes (H6: not exported) ------------------------------------------------


class PermutationTestResult(_TestResult):
    __slots__ = ()
    _TITLE = "Permutation test"
    _FIELDS = ("statistic", "pvalue", "null_distribution")
    _HIDDEN = ("null_distribution",)


class MonteCarloTestResult(PermutationTestResult):
    __slots__ = ()
    _TITLE = "Monte Carlo test"


class BootstrapResult(_TestResult):
    __slots__ = ()
    _TITLE = "Bootstrap"
    _FIELDS = ("bootstrap_distribution", "standard_error")
    _HIDDEN = ("bootstrap_distribution",)
    _CI = True
    _CI_LABEL = "statistic"


class BwsTestResult(PermutationTestResult):
    __slots__ = ()
    _TITLE = "Baumgartner-Weiss-Schindler test"


class MultiscaleGraphcorrResult(_TestResult):
    __slots__ = ()
    _TITLE = "Multiscale graph correlation"
    _FIELDS = ("statistic", "pvalue", "mgc_dict")
    _UNPACK = ("statistic", "pvalue", "mgc_dict")


class GoodnessOfFitResult(_TestResult):
    __slots__ = ()
    _TITLE = "Goodness of fit (Monte Carlo)"
    _FIELDS = ("fit_result", "statistic", "pvalue", "null_distribution")
    _HIDDEN = ("null_distribution",)

    def __repr__(self):
        """fit_result always shown (H158): the fitted distribution as <Name> (an array's
        <array of shape ...> form), None where the fit failed."""
        fr = self._values.get("fit_result")
        text = "None" if fr is None else f"<{type(fr).__name__}>"
        rest = _S.one_line("", [(k, self._values[k]) for k in self._FIELDS[1:] if k in self._values])
        return f"{type(self).__name__}(fit_result={text}" + (", " + rest[1:] if rest != "()" else ")")


# ---- the statistic --------------------------------------------------------------------------


def _vectorized(statistic, vectorized, name):
    if not callable(statistic):
        raise TypeError(f"{name}(): 'statistic' must be callable, not {_type_name(statistic)}")
    if vectorized is None:
        try:
            return "axis" in inspect.signature(statistic).parameters
        except TypeError, ValueError:
            raise TypeError(f"{name}(): the signature of 'statistic' cannot be read; pass vectorized=") from None
    return C.flag(vectorized, name, "vectorized")


def _value(v, shape, name):
    """A statistic's result as a float64 array of `shape` (a Python float for ())."""
    if isinstance(v, _sd.ndarray):
        a = v if v.dtype == _sd.float64 else v.astype(_sd.float64)
    elif isinstance(v, numbers.Real) and not isinstance(v, bool):
        a = None
        if shape == ():
            return float(v)
    else:
        try:
            a = _as_array(v, name, "statistic")
        except TypeError:
            raise TypeError(f"{name}(): 'statistic' must return numbers, not {_type_name(v)}") from None
    if a is None or tuple(a.shape) != tuple(shape):
        got = "a number" if a is None else f"shape {tuple(a.shape)}"
        raise ValueError(
            f"{name}(): 'statistic' must return {'a number' if not shape else f'shape {shape}'}, not {got}"
        )
    return float(a[0]) if shape == () and a.ndim == 1 and a.shape[0] == 1 else a


class _Statistic:
    """The user's statistic on arrays whose last axis is the sample's: called with axis=-1, or
    lane by lane on 1-D arrays when it is not vectorized."""

    def __init__(self, fn, vectorized, name):
        self.fn = fn
        self.vec = vectorized
        self.name = name

    def __call__(self, arrays, lead):
        """arrays: sd arrays (*lead, n_s); the statistic's values as a (prod(lead),) array."""
        size = math.prod(lead)
        if self.vec:
            v = self.fn(*arrays, axis=-1)
            v = _value(v, tuple(lead), self.name)
            return _sd.asarray([v]) if isinstance(v, float) else _sd.reshape(v, (size,))
        flat = [_sd.reshape(a, (size, a.shape[-1])) for a in arrays]
        out = [None] * size
        for i in range(size):
            out[i] = _value(self.fn(*(f[i] for f in flat)), (), self.name)
        return _sd.asarray(out) if size else _sd.empty((0,))


def _shape(lead, v):
    """A (prod(lead),) array in the shape lead (a Python float for ())."""
    if not lead:
        return float(v[0])
    return _sd.reshape(v, tuple(lead))


# ---- the samples ----------------------------------------------------------------------------


def _samples(name, data, axis, *, paired=False, least=1, arg="data"):
    """data: a list or tuple of samples -> C.Samples (no NaN policy: NaNs reach the statistic)."""
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"{name}(): '{arg}' must be a list or tuple of samples, not {_type_name(data)}")
    if not data:
        raise ValueError(f"{name}(): '{arg}' must hold at least one sample")
    if all(_scalar_value(e) is not None for e in data):  # one sample given bare: its numbers
        raise TypeError(
            f"{name}(): '{arg}' must be a sequence of samples (one sample is given as (x,)), not a sequence of numbers"
        )
    args = tuple(f"{arg}[{i}]" for i in range(len(data)))
    s = C.Samples(name, list(data), args, axis, "propagate", paired=paired, scan=False)
    for i, n in enumerate(s.n):
        if n < least:
            raise ValueError(f"{name}(): each sample must hold at least {least} observations, not {n} ({args[i]})")
    return s


def _lead_rows(rows, batch):
    """The samples' rows (lanes, n_s) as arrays (*batch, n_s)."""
    return [_sd.reshape(r, tuple(batch) + (r.shape[1],)) for r in rows]


def _per_call(batch, lanes, n):
    """Resamples per statistic call: the user's batch, else at most _BATCH_VALUES values."""
    if batch is not None:
        return batch
    return max(1, _BATCH_VALUES // max(1, lanes * n))


def _keys(rng, n):
    """n elements of rng's stream: (key word 0, key word 1, position)."""
    key, pos = rng._reserve(n)
    return key[0], key[1], pos


def _alloc(n, lanes, name):
    _check_alloc(8 * n * max(1, lanes), name)


# ---- the permutations -----------------------------------------------------------------------


def _count(ptype, ns):
    """The number of distinct permutations (scipy's n_max)."""
    if ptype == "independent":
        total, c = 0, 1
        for n in ns:
            total += n
            c *= math.comb(total, n)
        return c
    if ptype == "pairings":
        return math.factorial(ns[0]) ** len(ns)
    return math.factorial(len(ns)) ** ns[0]


def _draws(ptype, ns):
    """The stream elements one random permutation draws."""
    k, N = len(ns), sum(ns)
    if ptype == "independent":
        return max(0, N - 1)
    if ptype == "pairings":
        return k * max(0, ns[0] - 1)
    return ns[0] * max(0, k - 1)


def _permutation_null(name, rows, batch, stat, ptype, n_resamples, per, rng, *, what=None):
    """The null distribution (R, lanes) and whether it is the exact enumeration."""
    if ptype == "samples" and len(rows) == 1:  # a single sample: its signs (scipy's [x, -x])
        rows = [rows[0], -rows[0]]
        stat = _OneOf(stat)
    ns = [r.shape[1] for r in rows]
    lanes = rows[0].shape[0]
    total = _count(ptype, ns)
    exact = total <= n_resamples
    if exact:
        if total > _EXACT_CAP:
            raise ValueError(
                f"{name}(): enumerating all {total:.3g} permutations is more than the cap of {_EXACT_CAP:.3g}; "
                "use a finite n_resamples"
            )
        R = total
        k0 = k1 = pos = 0
    else:
        if rng is None:
            C.rng(rng, what or name)  # random permutations: rng is required
        R = n_resamples
        _alloc(R, lanes, name)
        k0, k1, pos = _keys(rng, R * _draws(ptype, ns))
    _alloc(R, lanes, name)
    B = _per_call(per, lanes, sum(ns))
    chunks = []
    for start in range(0, R, B):
        count = min(B, R - start)
        res = _k.perm_resamples(rows, _PCODE[ptype], exact, k0, k1, pos, start, count)
        arrays = [_sd.reshape(a, (count,) + tuple(batch) + (a.shape[2],)) for a in res]
        v = stat(arrays, (count,) + tuple(batch))
        chunks.append(_sd.reshape(v, (count, lanes)))
    null = chunks[0] if len(chunks) == 1 else _sd.concat(chunks, axis=0)
    return null, exact


def _pvalue(null, obs, alt, exact):
    """scipy's p-value of each lane: null (R, lanes) or (R, 1), obs (lanes,)."""
    return _k.resample_pvalue(null, obs, _ALT[alt], 0 if exact else 1)


def _observed(stat, rows, batch):
    return stat(_lead_rows(rows, batch), tuple(batch))


def permutation_test(
    data, statistic, /, *, permutation_type="independent", vectorized=None, n_resamples=9999, batch=None,
    alternative="two-sided", axis=None, rng=None,
):  # fmt: skip
    """A permutation test of the samples in data (a list or tuple) by the statistic: statistic
    (the observed value), pvalue and null_distribution (the statistic under the permutations,
    (R, *batch)). permutation_type 'independent' (the pooled observations dealt to the samples
    again), 'pairings' (each sample's order permuted: tests of association) or 'samples' (each
    observation's values permuted between the samples: paired tests; one sample flips signs).
    Every permutation when there are at most n_resamples of them (math.inf: always, up to 1e7), in
    scipy's order (rng not needed), else n_resamples random ones from rng. Does not unpack."""
    name = "permutation_test"
    ptype = C.choice(permutation_type, name, "permutation_type", _PTYPES)
    vec = _vectorized(statistic, vectorized, name)
    nres = _n_resamples(n_resamples, name, inf=True)
    per = _batch(batch, name)
    alt = C.alternative(alternative, name)
    if rng is not None:
        C.rng(rng, name)
    if ptype == "independent" and isinstance(data, (list, tuple)) and len(data) < 2:
        raise ValueError(f"{name}(): permutation_type='independent' needs at least two samples, not {len(data)}")
    s = _samples(name, data, axis, paired=ptype != "independent", least=2)
    stat = _Statistic(statistic, vec, name)
    rows = s.rows
    obs = _observed(stat, rows, s.batch)
    null, exact = _permutation_null(name, rows, s.batch, stat, ptype, nres, per, rng)
    p = _pvalue(null, obs, alt, exact)
    R = null.shape[0]
    info = {
        "alternative": alt,
        "nobs": s.sizes(),
        "pairs": [("Permutation Type:", ptype), ("Resamples:", f"{R} (all)" if exact else str(R))],
    }
    return PermutationTestResult._make(
        batch=s.batch,
        info=info,
        statistic=_shape(s.batch, obs),
        pvalue=_shape(s.batch, p),
        null_distribution=_sd.reshape(null, (R,) + tuple(s.batch)),
    )


class _OneOf:
    """'samples' of one sample: the statistic of the first of [x, -x]."""

    def __init__(self, stat):
        self.stat = stat

    def __call__(self, arrays, lead):
        return self.stat(arrays[:1], lead)


# ---- Monte Carlo ----------------------------------------------------------------------------


_NORMAL = None


def _normal():
    global _NORMAL
    if _NORMAL is None:
        from ..stats import Normal

        _NORMAL = Normal()
    return _NORMAL


def _draw(f, size, child, name):
    v = f(size, rng=child)
    a = v if isinstance(v, _sd.ndarray) else _as_array(v, name, "rvs")
    if a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
    if tuple(a.shape) != tuple(size):
        raise ValueError(f"{name}(): 'rvs' must return an array of shape {tuple(size)}, not {tuple(a.shape)}")
    return a


def _monte_carlo_null(name, rvs, ns, stat, n_resamples, per, rng):
    """The null distribution (R, 1): the statistic of samples drawn by rvs[i]((B, n_i), rng=child_i)."""
    children = rng.spawn(len(ns))
    _alloc(n_resamples, 1, name)
    B = _per_call(per, 1, sum(ns))
    chunks = []
    for start in range(0, n_resamples, B):
        count = min(B, n_resamples - start)
        arrays = [_draw(f, (count, n), c, name) for f, n, c in zip(rvs, ns, children)]
        chunks.append(stat(arrays, (count,)))
    null = chunks[0] if len(chunks) == 1 else _sd.concat(chunks, axis=0)
    return _sd.reshape(null, (n_resamples, 1))


def monte_carlo_test(
    data, rvs, statistic, /, *, vectorized=None, n_resamples=9999, batch=None, alternative="two-sided", axis=None,
    rng,
):  # fmt: skip
    """A Monte Carlo test that data comes from the distribution rvs samples: rvs(size, *, rng)
    (H118; sd.stats' X.sample(size, rng=rng) is one) and data one sample, or a list or tuple of
    them and data as many samples. statistic, pvalue and null_distribution (the statistic of
    n_resamples samples of data's sizes, (n_resamples,): one for every lane of a batch). Each
    sample draws from its own child of rng (rng.spawn). Does not unpack."""
    name = "monte_carlo_test"
    if callable(rvs) and not isinstance(rvs, (list, tuple)):
        fns, data = (rvs,), [data]
    elif isinstance(rvs, (list, tuple)) and rvs and all(callable(f) for f in rvs):
        fns = tuple(rvs)
        if not isinstance(data, (list, tuple)) or len(data) != len(fns):
            got = len(data) if isinstance(data, (list, tuple)) else _type_name(data)
            raise ValueError(f"{name}(): 'data' must hold one sample per 'rvs' ({len(fns)}), not {got}")
    else:
        raise TypeError(f"{name}(): 'rvs' must be a callable or a list or tuple of callables, not {_type_name(rvs)}")
    vec = _vectorized(statistic, vectorized, name)
    nres = _n_resamples(n_resamples, name)
    per = _batch(batch, name)
    alt = C.alternative(alternative, name)
    C.rng(rng, name)
    s = _samples(name, data, axis)
    stat = _Statistic(statistic, vec, name)
    obs = _observed(stat, s.rows, s.batch)
    null = _monte_carlo_null(name, fns, s.n, stat, nres, per, rng)
    p = _pvalue(null, obs, alt, False)
    info = {"alternative": alt, "nobs": s.sizes(), "pairs": [("Resamples:", str(nres))]}
    return MonteCarloTestResult._make(
        batch=s.batch,
        info=info,
        statistic=_shape(s.batch, obs),
        pvalue=_shape(s.batch, p),
        null_distribution=_sd.reshape(null, (nres,)),
    )


# ---- the bootstrap --------------------------------------------------------------------------


def _bootstrap_distribution(name, rows, batch, stat, paired, n_resamples, per, rng):
    lanes = rows[0].shape[0]
    ns = [r.shape[1] for r in rows]
    _alloc(n_resamples, lanes, name)
    k0, k1, pos = _keys(rng, n_resamples * (ns[0] if paired else sum(ns)))
    B = _per_call(per, lanes, sum(ns))
    chunks = []
    for start in range(0, n_resamples, B):
        count = min(B, n_resamples - start)
        res = _k.boot_resamples(rows, paired, k0, k1, pos, start, count)
        arrays = [_sd.reshape(a, tuple(batch) + (count, a.shape[2])) for a in res]
        v = stat(arrays, tuple(batch) + (count,))
        chunks.append(_sd.reshape(v, (lanes, count)))
    return chunks[0] if len(chunks) == 1 else _sd.concat(chunks, axis=1)


def _acceleration(rows, batch, stat, paired, per):
    """BCa's acceleration of each lane (scipy's _bca_interval: the jackknife of every sample)."""
    lanes = rows[0].shape[0]
    groups = [list(range(len(rows)))] if paired else [[j] for j in range(len(rows))]
    nums, dens = None, None
    for g in groups:
        n = rows[g[0]].shape[1]
        B = per or n
        vals = []
        for start in range(0, n, B):
            count = min(B, n - start)
            arrays = []
            for i, r in enumerate(rows):
                if i in g:
                    a = _k.jackknife(r, start, count)
                    arrays.append(_sd.reshape(a, tuple(batch) + (count, n - 1)))
                else:
                    a = _sd.reshape(r, tuple(batch) + (1, r.shape[1]))
                    arrays.append(_sd.broadcast_to(a, tuple(batch) + (count, r.shape[1])))
            v = stat(arrays, tuple(batch) + (count,))
            vals.append(_sd.reshape(v, (lanes, count)))
        th = vals[0] if len(vals) == 1 else _sd.concat(vals, axis=1)
        nf = float(n)
        u = (nf - 1.0) * (_sd.mean(th, axis=-1, keepdims=True) - th)
        num = _sd.sum(u**3, axis=-1) / nf**3
        den = _sd.sum(u**2, axis=-1) / nf**2
        nums = num if nums is None else nums + num
        dens = den if dens is None else dens + den
    return nums / dens**1.5 / 6.0


def _bootstrap_interval(dist, theta, z0, accel, method, alt, s):
    """conf_int(alpha) of a bootstrap result (scipy's bootstrap): the quantiles of the bootstrap
    distribution at alpha/2 and 1 - alpha/2 (alpha one-sided), BCa's corrected ones, 'basic'
    reflected about the observed statistic; the one-sided interval's other end infinite."""

    def ci(alpha):
        a = alpha / 2.0 if alt == "two-sided" else alpha
        lanes = dist.shape[0]
        if method == "BCa":
            za = float(_special.ndtri(a))
            n1 = z0 + za
            n2 = z0 - za
            a1 = _special.ndtr(z0 + n1 / (1.0 - accel * n1))
            a2 = _special.ndtr(z0 + n2 / (1.0 - accel * n2))
            ps = _sd.stack([a1, a2], axis=-1)
        else:
            ps = _sd.reshape(_sd.asarray([a, 1.0 - a] * lanes), (lanes, 2))
        q = _k.lane_quantiles(dist, ps)
        lo, hi = q[:, 0], q[:, 1]
        if method == "basic":
            lo, hi = 2.0 * theta - hi, 2.0 * theta - lo
        if alt == "less":
            lo = _sd.full((lanes,), -math.inf)
        elif alt == "greater":
            hi = _sd.full((lanes,), math.inf)
        return _shape(s.batch, lo), _shape(s.batch, hi)

    return ci


def _bootstrap(name, rows, s, stat, *, n_resamples, per, paired, alt, method, rng):
    """The bootstrap of the statistic: its distribution (lanes, R), standard error and interval."""
    dist = _bootstrap_distribution(name, rows, s.batch, stat, paired, n_resamples, per, rng)
    theta = _observed(stat, rows, s.batch)
    se = _sd.std(dist, axis=-1, correction=1)
    z0 = accel = None
    if method == "BCa":
        z0 = _special.ndtri(_k.score_percentile(dist, theta))
        accel = _acceleration(rows, s.batch, stat, paired, per)
    return dist, se, _bootstrap_interval(dist, theta, z0, accel, method, alt, s)


def bootstrap(
    data, statistic, /, *, n_resamples=9999, batch=None, vectorized=None, paired=False, axis=None,
    alternative="two-sided", method="BCa", rng,
):  # fmt: skip
    """The bootstrap of the statistic of the samples in data (a list or tuple): bootstrap_distribution
    ((*batch, n_resamples)), standard_error (its standard deviation, n - 1) and conf_int(alpha=0.05)
    (H16: the interval at any level from the one distribution): 'percentile', 'basic' or 'BCa'
    (bias-corrected and accelerated: the jackknife's acceleration). paired resamples the
    observations' indices together. alternative 'less' or 'greater' gives a one-sided interval
    (the other end infinite). Each sample needs 2 observations. Does not unpack."""
    name = "bootstrap"
    nres = _n_resamples(n_resamples, name)
    per = _batch(batch, name)
    vec = _vectorized(statistic, vectorized, name)
    pair = C.flag(paired, name, "paired")
    alt = C.alternative(alternative, name)
    meth = C.choice(method, name, "method", _BOOT)
    C.rng(rng, name)
    s = _samples(name, data, axis, paired=pair, least=2)
    stat = _Statistic(statistic, vec, name)
    dist, se, ci = _bootstrap(name, s.rows, s, stat, n_resamples=nres, per=per, paired=pair, alt=alt, method=meth,
                              rng=rng)  # fmt: skip
    info = {
        "alternative": alt,
        "nobs": s.sizes(),
        "title": f"Bootstrap ({meth})",
        "pairs": [("Resamples:", str(nres))] + ([("Paired:", "yes")] if pair else []),
    }
    lanes = dist.shape[0]
    return BootstrapResult._make(
        batch=s.batch,
        ci=ci,
        info=info,
        bootstrap_distribution=_sd.reshape(dist, tuple(s.batch) + (nres,)) if s.batch else _sd.reshape(dist, (nres,)),
        standard_error=_shape(s.batch, se) if lanes else se,
    )


# ---- the tests' method= (H117) -----------------------------------------------------------------


def _method_name(method):
    return {PermutationMethod: "permutation", MonteCarloMethod: "monte carlo"}.get(type(method))


def resampling_pairs(method, R=None, exact=False):
    """What summary() shows of a test's resampling method (H75)."""
    if isinstance(method, PermutationMethod):
        n = R if R is not None else method.n_resamples
        return [("P-value:", "permutation"), ("Resamples:", f"{n} (all)" if exact else str(n))]
    return [("P-value:", "Monte Carlo"), ("Resamples:", str(method.n_resamples))]


def nested_names(v, shape):
    """A batch's per-lane method field (H48): the name v in nested lists of the batch's shape."""
    from ._gof import _nest

    return _nest([v] * math.prod(shape), shape)


def _nan_lanes(rows):
    """The lanes of the rows (lanes, n_s) holding a NaN, or None."""
    has = None
    for r in rows:
        if r.size:
            m = _sd.any(_sd.isnan(r), axis=1)
            has = m if has is None else (has | m)
    return has if has is not None and bool(_sd.any(has)) else None


def _lane_without_nan(rows, i, paired):
    xs = [r[i] for r in rows]
    if paired:
        keep = None
        for x in xs:
            k = ~_sd.isnan(x)
            keep = k if keep is None else (keep & k)
        xs = [x[keep] for x in xs]
    else:
        xs = [x[~_sd.isnan(x)] for x in xs]
    return [_sd.reshape(x, (1, x.shape[0])) for x in xs]


class _Held:
    """A statistic of held rows and permuted ones: the held rows (*batch, n) broadcast against the
    resamples (pearsonr's x while y's order is permuted)."""

    def __init__(self, stat, held):
        self.stat, self.held = stat, held

    def __call__(self, arrays, lead):
        return self.stat(self.held + list(arrays), lead)


def kernel_rows(arrays):
    """Samples (*lead, n_i) whose leads broadcast (a held row against the resamples) as the
    kernels' rows (L, n_i), and the lead: a test's statistic of the resamples straight from its
    kernel (H119), the bits of calling the test along axis=-1."""
    leads = [tuple(a.shape[:-1]) for a in arrays]
    lead = leads[0]
    if any(ld != lead for ld in leads):  # numpy's rule: aligned at the right, 1 stretches
        k = max(len(ld) for ld in leads)
        lead = tuple(max(d) for d in zip(*[(1,) * (k - len(ld)) + ld for ld in leads]))
    rows = []
    for a, ld in zip(arrays, leads):
        if ld != lead:
            a = _sd.broadcast_to(a, lead + (a.shape[-1],))
        rows.append(_sd.reshape(a, (math.prod(lead), a.shape[-1])))
    return rows, lead


def from_lanes(v, lead):
    """A kernel's per-lane values (L,) in the lead's shape: a float for one 1-D test."""
    return float(v[0]) if lead == () else _sd.reshape(v, lead)


def method_pvalue(
    name, method, s, statistic, alternative, *, ptype="independent", data=None, rvs_default=True, held=0
):  # fmt: skip
    """A test's p-value by its PermutationMethod or MonteCarloMethod (H117): the statistic
    `statistic` (a function of sd arrays (*lead, n_s) along the last axis giving the lead shape: a
    number for 1-D samples) of permutations of the rows (data, default the samples' s.rows), or of
    samples drawn by the method's rvs (the standard normal when rvs is None) of the samples' sizes.
    Under nan_policy='omit' a lane holding a NaN is tested alone without it (paired rows drop the
    pair). held: the leading rows a permutation keeps in place (pearsonr's x: scipy permutes y
    alone); a Monte Carlo draws every row. Returns (p (lanes,), the pairs summary() shows)."""
    stat = _Statistic(lambda *a, axis: statistic(*a), True, name)
    rows = data if data is not None else s.rows
    paired = s.paired or ptype != "independent" or held > 0
    nan = _nan_lanes(rows) if s.policy == "omit" else None
    state = {"exact": True, "R": None}

    def one(rows, batch):
        if isinstance(method, PermutationMethod):
            ps = stat if not held else _Held(stat, _lead_rows(rows[:held], batch))
            null, exact = _permutation_null(name, rows[held:], batch, ps, ptype, method.n_resamples, method.batch,
                                            method.rng, what=name)  # fmt: skip
            state["exact"] &= exact
            state["R"] = null.shape[0]
        else:
            rvs = method.rvs
            if rvs is None:
                rvs = (_normal_rvs,) * len(rows)
            elif not rvs_default:
                raise ValueError(f"{name}(): a MonteCarloMethod's 'rvs' must be None for this test (it draws tables)")
            elif callable(rvs):
                rvs = (rvs,) * len(rows)
            elif len(rvs) != len(rows):
                raise ValueError(
                    f"{name}(): a MonteCarloMethod's 'rvs' must hold {len(rows)} callables, not {len(rvs)}"
                )
            null = _monte_carlo_null(name, rvs, [r.shape[1] for r in rows], stat, method.n_resamples, method.batch,
                                     method.rng)  # fmt: skip
            exact = False
        obs = _observed(stat, rows, batch)
        return _pvalue(null, obs, alternative, exact)

    if nan is None:
        p = one(rows, s.batch)
    else:
        p = _sd.asarray([float(one(_lane_without_nan(rows, i, paired), ())[0]) for i in range(s.lanes)])
    exact = state["exact"] and isinstance(method, PermutationMethod)
    return p, resampling_pairs(method, state["R"], exact)


def method_interval(name, method, s, statistic, alternative, alpha, *, paired=True, clip=None):
    """A test's interval by its BootstrapMethod (pearsonr's conf_int(method=), scipy's): the
    bootstrap of the statistic of the samples (paired), its 'percentile', 'basic' or 'BCa'
    interval at 1 - alpha, clipped to `clip`; lanes holding a NaN under nan_policy='omit' one by
    one without it. (low, high) of the lanes as (lanes,) arrays."""
    stat = _Statistic(lambda *a, axis: statistic(*a), True, name)
    nan = _nan_lanes(s.rows) if s.policy == "omit" else None
    kw = dict(n_resamples=method.n_resamples, per=method.batch, paired=paired, alt=alternative, method=method.method,
              rng=method.rng)  # fmt: skip

    class _One:
        batch = ()

    if nan is None:
        flat = _One()
        flat.batch = (s.lanes,)
        _, _, ci = _bootstrap(name, s.rows, flat, stat, **kw)
        lo, hi = ci(alpha)
    else:
        los, his = [], []
        for i in range(s.lanes):
            rows = _lane_without_nan(s.rows, i, paired)
            _, _, ci = _bootstrap(name, rows, _One(), stat, **kw)
            lo, hi = ci(alpha)
            los.append(float(lo)), his.append(float(hi))
        lo, hi = _sd.asarray(los), _sd.asarray(his)
    if clip is not None:
        lo, hi = _sd.clip(lo, min=clip[0], max=clip[1]), _sd.clip(hi, min=clip[0], max=clip[1])
    return lo, hi


def table_pvalues(name, method, tables, kind, alternative):
    """chi2_contingency's (kind 0: Pearson's statistic) and fisher_exact's (kind 1: the table's
    probability) p-values by a PermutationMethod or a MonteCarloMethod (scipy's): the statistic of
    tables with the observed margins, every permutation of the untabulated labels when N! fits in
    n_resamples, else tables drawn under independence (cell by cell from the hypergeometric given
    what is left of its row and column; a PermutationMethod's random permutations give the same
    tables in distribution). tables: (lanes, r, c) integer counts. (statistic, p) of each table as
    lists, and the pairs summary() shows."""
    if isinstance(method, MonteCarloMethod) and method.rvs is not None:
        raise ValueError(f"{name}(): a MonteCarloMethod's 'rvs' must be None (the tables are drawn by their margins)")
    lanes, r, c = tables.shape
    stats, ps = [], []
    R, exact_all = None, True
    for i in range(lanes):
        t = tables[i]
        obs = _k.table_stat(t, kind)
        N = int(round(float(_sd.sum(t))))
        exact = isinstance(method, PermutationMethod) and math.factorial(N) <= method.n_resamples
        if exact:
            if math.factorial(N) > _EXACT_CAP:
                raise ValueError(
                    f"{name}(): enumerating all {math.factorial(N):.3g} permutations is more than the cap of "
                    f"{_EXACT_CAP:.3g}; use a finite n_resamples"
                )
            count, k0, k1, pos = math.factorial(N), 0, 0, 0
        else:
            if method.rng is None:
                C.rng(None, name)
            count = method.n_resamples
            _alloc(count, 1, name)
            k0, k1, pos = _keys(method.rng, count * (r - 1) * (c - 1))
        null = _k.table_null(t, kind, exact, k0, k1, pos, count)
        p = _k.resample_pvalue(_sd.reshape(null, (count, 1)), _sd.asarray([obs]), _ALT[alternative], 0 if exact else 1)
        stats.append(obs)
        ps.append(float(p[0]))
        R, exact_all = count, exact_all and exact
    return stats, ps, resampling_pairs(method, R, exact_all and isinstance(method, PermutationMethod))


def _normal_rvs(size, *, rng):
    return _normal().sample(size, rng=rng)


def check_method(method, name, allowed, strings=()):
    """A test's method= of the resampling kind: one of the `allowed` classes; else a TypeError
    naming them (and the strings the test takes)."""
    if isinstance(method, allowed):
        return method
    names = [v if v == "None" else repr(v) for v in strings]
    names += [f"a {c.__name__}" for c in (allowed if isinstance(allowed, tuple) else (allowed,))]
    lst = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
    raise TypeError(f"{name}(): 'method' must be {lst}, not {_type_name(method)}")


# ---- bws_test -------------------------------------------------------------------------------


def bws_test(x, y, /, *, axis=None, alternative="two-sided", method=None):
    """The Baumgartner-Weiss-Schindler test of two samples: statistic B of the pooled midranks
    (one-sided: the signed form, scipy's), pvalue from the permutations of the ranks (large B is
    extreme; 'less' small B) and null_distribution. method=None enumerates all C(n + m, n)
    permutations up to 1e6 of them (H122; past it a PermutationMethod with rng is required);
    method=PermutationMethod(...) as permutation_test's. Does not unpack."""
    name = "bws_test"
    alt = C.alternative(alternative, name)
    if method is not None:
        check_method(method, name, PermutationMethod, ("None",))
    s = C.Samples(name, [x, y], ("x", "y"), axis, "propagate", scan=False)
    for n, a in zip(s.n, ("x", "y")):
        if n < 2:
            raise ValueError(f"{name}(): '{a}' must hold at least 2 observations, not {n}")
    pooled = _sd.concat(s.rows, axis=1)
    from ..stats import rankdata

    ranks = rankdata(pooled, axis=-1)
    n1 = s.n[0]
    rows = [ranks[:, :n1], ranks[:, n1:]]
    code = _ALT[alt]

    def statistic(arrays, lead):
        size = math.prod(lead)
        a = _sd.reshape(arrays[0], (size, arrays[0].shape[-1]))
        b = _sd.reshape(arrays[1], (size, arrays[1].shape[-1]))
        return _k.bws(a, b, code)

    palt = "less" if alt == "less" else "greater"
    if method is None:
        total = math.comb(s.n[0] + s.n[1], s.n[0])
        if total > _BWS_CAP:
            raise TypeError(
                f"{name}(): the {total:.3g} permutations are more than the {_BWS_CAP:.3g} it enumerates; pass "
                "method=PermutationMethod(rng=sd.random.default_rng(seed))"
            )
        nres, per, rng = total, None, None
    else:
        nres, per, rng = method.n_resamples, method.batch, method.rng
    obs = statistic(_lead_rows(rows, s.batch), tuple(s.batch))
    null, exact = _permutation_null(name, rows, s.batch, statistic, "independent", nres, per, rng)
    p = _pvalue(null, obs, palt, exact)
    R = null.shape[0]
    info = {"alternative": alt, "nobs": s.sizes(), "pairs": [("Resamples:", f"{R} (all)" if exact else str(R))]}
    return BwsTestResult._make(
        batch=s.batch,
        info=info,
        statistic=_shape(s.batch, obs),
        pvalue=_shape(s.batch, p),
        null_distribution=_sd.reshape(null, (R,) + tuple(s.batch)),
    )


# ---- multiscale_graphcorr ------------------------------------------------------------------------


def _points(v, name, arg):
    a = _as_array(v, name, arg)
    if a.ndim == 1:
        a = _sd.reshape(a, (a.shape[0], 1))
    elif a.ndim != 2:
        raise ValueError(f"{name}(): '{arg}' must be a 1-D or 2-D array, not {a.ndim}-D")
    if a.size and not bool(_sd.all(_sd.isfinite(a))):
        raise ValueError(f"{name}(): '{arg}' must hold finite values (no NaN or inf)")
    return a


def multiscale_graphcorr(x, y, /, *, compute_distance="euclidean", reps=1000, is_twosamp=False, rng):
    """Multiscale graph correlation (Vogelstein et al. 2019; scipy's): the test of independence of
    x (n, p) and y (n, q) (1-D: one column), or with is_twosamp (or when the row counts differ and
    the columns agree) of two samples of the same space. compute_distance 'euclidean' (default),
    a callable giving the (n, n) distance matrix of its argument, or None (x and y are distance
    matrices). statistic, pvalue ((1 + #{null >= stat}) / (1 + reps), random permutations of y
    from rng) and mgc_dict ('mgc_map' (n, n), 'opt_scale' [k, l], 'null_dist' (reps,)). Memory
    O(n^2) per thread. Unpacks as (statistic, pvalue, mgc_dict)."""
    name = "multiscale_graphcorr"
    two = C.flag(is_twosamp, name, "is_twosamp")
    nreps = C.integer(reps, name, "reps", minimum=0)
    C.rng(rng, name)
    if not (compute_distance is None or compute_distance == "euclidean" or callable(compute_distance)):
        raise TypeError(
            f"{name}(): 'compute_distance' must be 'euclidean', a callable or None, not {_type_name(compute_distance)}"
        )
    a = _points(x, name, "x")
    b = _points(y, name, "y")
    nx, px = a.shape
    ny, py = b.shape
    if nx != ny:
        if px != py:
            raise ValueError(
                f"{name}(): 'x' and 'y' must have shapes (n, p) and (n, q), or (n, p) and (m, p), not {tuple(a.shape)} "
                f"and {tuple(b.shape)}"
            )
        two = True
    if nx < 5 or ny < 5:
        raise ValueError(f"{name}(): at least 5 samples are required, not {min(nx, ny)}")
    if two:
        if compute_distance is None:
            raise ValueError(f"{name}(): a two-sample test needs compute_distance (the samples are not distances)")
        a = _sd.concat([a, b], axis=0)
        b = _sd.reshape(_sd.concat([_sd.zeros(nx), _sd.ones(ny)]), (nx + ny, 1))

    def dist(v, arg):
        if compute_distance is None:
            if v.shape[0] != v.shape[1]:
                raise ValueError(f"{name}(): '{arg}' must be a square distance matrix, not {tuple(v.shape)}")
            return v
        if compute_distance == "euclidean":
            return _k.euclidean(v)
        d = _as_array(compute_distance(v), name, "compute_distance")
        if tuple(d.shape) != (v.shape[0], v.shape[0]):
            raise ValueError(
                f"{name}(): 'compute_distance' must return an ({v.shape[0]}, {v.shape[0]}) matrix, not {tuple(d.shape)}"
            )
        return d

    da, db = dist(a, "x"), dist(b, "y")
    if da.shape[0] != db.shape[0]:
        raise ValueError(
            f"{name}(): the distance matrices must have the same size, not {da.shape[0]} and {db.shape[0]}"
        )
    n = da.shape[0]
    _alloc(nreps, 1, name)
    k0, k1, pos = _keys(rng, nreps * max(0, n - 1))
    stat, mmap, k, l, null = _k.mgc(da, db, nreps, k0, k1, pos)
    p = (1.0 + float(_sd.sum(null >= stat))) / (1.0 + nreps) if nreps else 1.0
    opt = int(k) if l < 0 else [int(k), int(l)]
    info = {"nobs": n, "pairs": [("Permutations:", str(nreps))] + ([("Two Samples:", "yes")] if two else [])}
    return MultiscaleGraphcorrResult._make(
        info=info, statistic=float(stat), pvalue=p, mgc_dict={"mgc_map": mmap, "opt_scale": opt, "null_dist": null}
    )


# ---- goodness_of_fit (H121) -------------------------------------------------------------------

_STATS = {"ad": 0, "ks": 1, "cvm": 2, "filliben": 3}


def _families():
    from .. import stats as st

    # sd.stats class -> (htest/mle.hpp's code, the parameters in its order)
    return {
        st.Normal: (0, ("mu", "sigma")),
        st.Exponential: (1, ("scale",)),
        st.Logistic: (2, ("mu", "sigma")),
        st.Gumbel: (3, ("mu", "sigma")),
        st.Weibull: (4, ("c", "scale")),
        st.LogNormal: (5, ("mu", "sigma")),
        st.Laplace: (6, ("mu", "sigma")),
        st.Rayleigh: (7, ("scale",)),
    }


_FAMILY_NAMES = "sd.stats.Normal, Exponential, Logistic, Gumbel, Weibull, LogNormal, Laplace or Rayleigh"


def _params(v, names, name, arg):
    """known_params, fit_params: a dict of the family's parameter names and real values."""
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise TypeError(f"{name}(): '{arg}' must be a dict, not {_type_name(v)}")
    out = {}
    for key, val in v.items():
        if key not in names:
            raise ValueError(f"{name}(): '{arg}' has {key!r}, not a parameter of the family ({', '.join(names)})")
        out[key] = C.real(val, name, f"{arg}[{key!r}]")
    return out


def _fitted(cls, names, th):
    return cls(**{k: float(v) for k, v in zip(names, th)})


def _fit_lanes(rows, code, fixed, stat):
    """(statistic (lanes,), params (lanes, k)) of the family fitted to each row."""
    return _k.gof_fit(rows, code, _sd.asarray(fixed), stat)


class _Shifted:
    """A distribution moved by loc (anderson's three-parameter Weibull: sd.stats' Weibull has no
    location): its samples."""

    def __init__(self, dist, loc):
        self.dist, self.loc = dist, loc

    def sample(self, shape, *, rng):
        return self.dist.sample(shape, rng=rng) + self.loc


def weibull3(th):
    from ..stats import Weibull

    return _Shifted(Weibull(c=th[0], scale=th[2]), th[1])


def gof_pvalue(name, rows, code, names, cls, known, fit, statistic, n_mc, per, rng, make=None):
    """goodness_of_fit's computation on one sample (rows (1, n)): (statistic, pvalue, null
    (n_mc,), the fitted parameters of the null distribution). statistic a name or a callable
    statistic(dist, data, axis); make(th) the null distribution (default cls of the parameters)."""
    n = rows.shape[1]
    null_fixed = [known.get(k, fit.get(k, math.nan)) for k in names]
    rfd_fixed = [known.get(k, math.nan) for k in names]
    _, nhd = _fit_lanes(rows, code, null_fixed, -1)
    th = [float(v) for v in nhd[0].tolist()]
    if not all(math.isfinite(v) for v in th):
        return math.nan, math.nan, _sd.full((n_mc,), math.nan), th
    null_dist = make(th) if make is not None else _fitted(cls, names, th)
    custom = callable(statistic)
    alt = "less" if (not custom and statistic == "filliben") else "greater"
    if custom:
        alt = getattr(statistic, "alternative", "greater")
        alt = C.alternative(alt, name)

    def stat_of(data):  # data (B, n)
        if not custom:
            v, _ = _fit_lanes(data, code, rfd_fixed, _STATS[statistic])
            return v
        _, ps = _fit_lanes(data, code, rfd_fixed, -1)
        B = data.shape[0]
        cols = {k: _sd.reshape(ps[:, i], (B, 1)) for i, k in enumerate(names)}
        try:
            d = cls(**cols)
        except ValueError:  # a resample whose fit failed: its statistic is NaN
            return _sd.full((B,), math.nan)
        v = statistic(d, data, axis=-1)
        return _sd.reshape(_value(v, (B,), name), (B,)) if B != 1 else _sd.asarray([_value(v, (1,), name)[0]])

    obs = stat_of(rows)
    _alloc(n_mc, 1, name)
    B = _per_call(per, 1, n)
    chunks = []
    for start in range(0, n_mc, B):
        count = min(B, n_mc - start)
        draws = null_dist.sample((count, n), rng=rng)
        chunks.append(stat_of(draws))
    null = chunks[0] if len(chunks) == 1 else _sd.concat(chunks, axis=0)
    p = _k.resample_pvalue(_sd.reshape(null, (n_mc, 1)), obs, _ALT[alt], 1)
    return float(obs[0]), float(p[0]), null, th


def goodness_of_fit(
    dist, data, /, *, known_params=None, fit_params=None, statistic="ad", n_mc_samples=9999, rng
):  # fmt: skip
    """The Monte Carlo goodness-of-fit test of scipy (H121): the family dist (an sd.stats class:
    Normal, Exponential, Logistic, Gumbel, Weibull, LogNormal, Laplace, Rayleigh) is fitted to the
    one-dimensional data by maximum likelihood (Normal as scipy: the standard deviation (n - 1)),
    known_params held everywhere and fit_params held for the null distribution (fitted in the
    statistic); statistic 'ad' (Anderson-Darling), 'ks' (Kolmogorov-Smirnov), 'cvm' (Cramér-von
    Mises), 'filliben' (the probability plot correlation; small is extreme) or a callable
    statistic(dist, data, axis) of the fitted distribution (its 'alternative' attribute, default
    'greater'), compared with n_mc_samples samples of the null distribution refitted alike.
    fit_result (the fitted null distribution: an sd.stats object), statistic, pvalue and
    null_distribution. Does not unpack."""
    name = "goodness_of_fit"
    fams = _families()
    if not (isinstance(dist, type) and dist in fams):
        what = f"sd.stats.{dist.__name__}" if isinstance(dist, type) else _type_name(dist)
        kind = ValueError if isinstance(dist, type) and issubclass(dist, _dist.ContinuousDistribution) else TypeError
        raise kind(f"{name}(): 'dist' must be one of the classes {_FAMILY_NAMES}, not {what}")
    code, names = fams[dist]
    known = _params(known_params, names, name, "known_params")
    fit = _params(fit_params, names, name, "fit_params")
    both = sorted(set(known) & set(fit))
    if both:
        raise ValueError(f"{name}(): {both[0]!r} is in both known_params and fit_params")
    if not callable(statistic):
        C.choice(statistic, name, "statistic", tuple(_STATS))
    nmc = C.integer(n_mc_samples, name, "n_mc_samples", minimum=1)
    C.rng(rng, name)
    a = _as_array(data, name, "data")
    if a.ndim != 1:
        raise ValueError(f"{name}(): 'data' must be one-dimensional, not {a.ndim}-D")
    if a.shape[0] < 2:
        raise ValueError(f"{name}(): 'data' must hold at least 2 observations, not {a.shape[0]}")
    if not bool(_sd.all(_sd.isfinite(a))):
        raise ValueError(f"{name}(): 'data' must hold finite values (no NaN or inf)")
    rows = _sd.reshape(a, (1, a.shape[0]))
    stat, p, null, th = gof_pvalue(name, rows, code, names, dist, known, fit, statistic, nmc, None, rng)
    if all(math.isfinite(v) for v in th):
        fitted = _fitted(dist, names, th)
    else:
        fitted = None
    label = statistic if isinstance(statistic, str) else getattr(statistic, "__name__", "callable")
    info = {
        "nobs": a.shape[0],
        "pairs": [("Distribution:", dist.__name__), ("Statistic type:", label), ("Samples:", str(nmc))],
    }
    return GoodnessOfFitResult._make(info=info, fit_result=fitted, statistic=stat, pvalue=p, null_distribution=null)


permutation_test = _calls.function(permutation_test)
monte_carlo_test = _calls.function(monte_carlo_test)
bootstrap = _calls.function(bootstrap)
bws_test = _calls.function(bws_test)
multiscale_graphcorr = _calls.function(multiscale_graphcorr)
goodness_of_fit = _calls.function(goodness_of_fit)
