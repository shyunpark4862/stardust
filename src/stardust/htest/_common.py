"""The input layer every sd.htest function shares (docs/api/htest.md §1, docs/design/htest.md
§2.1; decisions H3, H9, H11-H13).

- Data (samples, tables, counts): sd.stats' descriptive input (an sd.ndarray, a sequence of
  numbers, a real scalar, a frame Series or DataFrame; float64, complex is a TypeError) through
  the same function, so the messages are those of sd.stats.
- axis (H9): None (each sample flattened: one test), an int, or a tuple or list of ints, as
  sd.stats' descriptive functions. The sample axes are moved last; the other axes are the batch,
  broadcast between the samples (scipy's rule), while the sample lengths may differ. Paired
  samples must also agree in length. keepdims keeps the sample axes as length 1.
- Tables (contingency tables, page_trend_test's data, cochrans_q's x): the last axes are the
  table and the leading axes the batch (H9).
- nan_policy: 'propagate' (a lane holding a NaN gives NaN results), 'omit' (each lane drops its
  NaNs; paired samples drop the pair), 'raise' (ValueError), with sd.stats' messages.
- alternative: 'two-sided', 'less' or 'greater'; rng: an sd.random.Generator, required wherever
  a test draws (sd.random has no global state); method: a string option whose 'auto' resolves
  to the method used, which the result records (H12); method='exact' beyond its documented
  work cap raises ValueError (H13), so every call ends in bounded time.
- Degenerate lanes (too few observations, zero variance) give NaN results without warnings
  (H11); the tests decide that per lane, so a batch's other lanes still compute.
"""

import math
import operator

from .. import _core as _sd
from .. import random as _random
from ..stats._descriptive import _INT64_MAX, _INT64_MIN, _as_array, _axes, _bool, _choice, _type_name
from ..stats._distribution import _real_option

ALTERNATIVES = ("two-sided", "less", "greater")
POLICIES = ("propagate", "omit", "raise")


# ---- options ------------------------------------------------------------------------------


def alternative(v, name):
    """The alternative hypothesis: 'two-sided', 'less' or 'greater' (statsmodels' 'larger' and
    'smaller' are not taken: scipy's names, H3)."""
    return _choice(v, name, "alternative", ALTERNATIVES)


def nan_policy(v, name):
    return _choice(v, name, "nan_policy", POLICIES)


def flag(v, name, arg):
    """A bool option (True or False only, API.md §2)."""
    return _bool(v, name, arg)


def choice(v, name, arg, choices):
    """A string option (API.md §2.1's message form)."""
    return _choice(v, name, arg, tuple(choices))


def real(v, name, arg):
    """A real number option (int or float; numpy's too; bool is not) as a float."""
    r = _real_option(v, name, arg)
    if r is None:
        raise TypeError(f"{name}(): '{arg}' must be an int or a float, not {_type_name(v)}")
    return r


def integer(v, name, arg, *, minimum=None):
    """An int argument by operator.index (API.md §2: numpy integers are ints, bool is not),
    within int64 and at least `minimum`."""
    if isinstance(v, bool):
        raise TypeError(f"{name}(): '{arg}' must be an int, not bool")
    try:
        i = operator.index(v)
    except TypeError:
        raise TypeError(f"{name}(): '{arg}' must be an int, not {_type_name(v)}") from None
    if not _INT64_MIN <= i <= _INT64_MAX:
        raise OverflowError(f"{name}(): '{arg}' is out of range")
    if minimum is not None and i < minimum:
        raise ValueError(f"{name}(): '{arg}' must be >= {minimum}")
    return i


def alpha(v, name="conf_int"):
    """A significance level: a real number strictly between 0 and 1 (the interval is 1 - alpha).
    regression's conf_int(*, alpha=0.05) and its messages (G99, H16)."""
    if isinstance(v, bool):
        raise TypeError(f"{name}(): 'alpha' must be an int or a float, not bool")
    a = _real_option(v, name, "alpha")
    if a is None:
        raise TypeError(f"{name}(): 'alpha' must be an int or a float, not {_type_name(v)}")
    if not 0.0 < a < 1.0:
        raise ValueError(f"{name}(): 'alpha' must be between 0 and 1 (exclusive)")
    return a


def rng(v, name):
    """The random generator of a test that draws: an sd.random.Generator, always given (there is
    no unseeded default, sd.random's R3; optimize's O31 message)."""
    if not isinstance(v, _random.Generator):
        raise TypeError(
            f"{name}(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not {_type_name(v)}"
        )
    return v


def method(v, name, choices, *, auto=True):
    """A method string: 'auto' (when the test has it) or one of `choices`."""
    return _choice(v, name, "method", (("auto",) if auto else ()) + tuple(choices))


def resolved(requested, picked):
    """The method used: what 'auto' picked, else the one requested (the result's method field,
    H12)."""
    return picked if requested == "auto" else requested


def check_exact_work(work, cap, name, *, instead="method='asymptotic'"):
    """method='exact' within its documented cap (H13): `work` the call's operation count (or
    memory, in the same unit as `cap`). Beyond it a ValueError that names the way out."""
    if work > cap:
        if not math.isfinite(work):  # (a count past the doubles: only that it passes the cap is known)
            raise ValueError(
                f"{name}(): method='exact' needs more than {cap:.3g} operations for this input; use {instead}"
            )
        raise ValueError(
            f"{name}(): method='exact' needs about {work:.3g} operations for this input, more than its cap of "
            f"{cap:.3g}; use {instead}"
        )


# ---- samples ------------------------------------------------------------------------------


class Samples:
    """k samples read along `axis` (H9), the axes beyond it a batch of independent lanes.

    rows[s] is sample s as a (lanes, n_s) float64 array (C order); batch the batch shape (() for
    axis=None); lanes its size; n[s] the sample lengths along the axis. nan is a (lanes,) bool
    array, a NaN in any of the lane's samples (None when nothing is NaN). lane(i) gives the lane's
    samples as 1-D arrays, NaNs dropped under 'omit' (pairs, when paired). out() puts a per-lane
    result back in the batch's shape.

    scan=False leaves the NaNs to a native kernel that handles nan_policy itself: the samples are
    then looked at only for 'raise', and nan stays None (lane() and propagates() know of no NaN)."""

    def __init__(self, name, xs, args, axis, policy, *, paired=False, keepdims=False, scan=True):
        policy = nan_policy(policy, name)
        flag(keepdims, name, "keepdims")
        arrays = [_as_array(x, name, a) for x, a in zip(xs, args)]
        nd = max(a.ndim for a in arrays) if arrays else 1
        # equal ndims by leading axes of length 1 (numpy's broadcasting rule, scipy's too)
        arrays = [a if a.ndim == nd else _sd.reshape(a, (1,) * (nd - a.ndim) + tuple(a.shape)) for a in arrays]
        red = _axes(axis, nd, name)
        kept = tuple(d for d in range(nd) if d not in red)
        batch = []
        for d in kept:
            sizes = {a.shape[d] for a in arrays} - {1}
            if len(sizes) > 1:
                shapes = ", ".join(str(tuple(a.shape)) for a in arrays)
                raise ValueError(f"{name}(): shapes could not be broadcast together outside the sample axes: {shapes}")
            batch.append(sizes.pop() if sizes else 1)
        self.batch = tuple(batch)
        self.lanes = math.prod(self.batch)
        self.n = [math.prod(a.shape[d] for d in red) for a in arrays]
        if paired and len(set(self.n)) > 1:
            listed = " and ".join(str(k) for k in self.n)
            what = "in size" if axis is None else "in length along the sample axes"
            raise ValueError(f"{name}(): the paired samples must agree {what}, not {listed}")
        perm = kept + red
        rows = []
        for a, n in zip(arrays, self.n):
            t = _sd.permute_dims(a, perm) if perm != tuple(range(nd)) else a
            full = self.batch + tuple(a.shape[d] for d in red)
            if tuple(t.shape) != full:
                t = _sd.broadcast_to(t, full)
            rows.append(_sd.reshape(t, (self.lanes, n)))
        self.rows = rows
        self.keep_shape = tuple(1 if d in red else self.batch[kept.index(d)] for d in range(nd))
        self.keepdims = keepdims
        self.paired = paired
        self.policy = policy
        self.nan = None
        self.scanned = scan or policy == "raise"
        if not scan and policy != "raise":
            return
        has = None
        for r in rows:
            if r.size:
                m = _sd.any(_sd.isnan(r), axis=1)
                has = m if has is None else (has | m)
        if has is not None and bool(_sd.any(has)):
            if policy == "raise":
                raise ValueError(f"{name}(): the input contains NaN (nan_policy='raise')")
            self.nan = has

    def lane(self, i):
        """Lane i's samples as 1-D float64 arrays; under 'omit' without their NaNs (paired
        samples without the pairs that hold one)."""
        xs = [r[i] for r in self.rows]
        if self.policy != "omit" or self.nan is None or not bool(self.nan[i]):
            return xs
        if self.paired:
            keep = None
            for x in xs:
                k = ~_sd.isnan(x)
                keep = k if keep is None else (keep & k)
            return [x[keep] for x in xs]
        return [x[~_sd.isnan(x)] for x in xs]

    def sizes(self):
        """The sample sizes summary() shows (H75): an int for one sample (or the pairs of paired
        samples), a tuple for several; None where nan_policy='omit' may have dropped values the
        call did not count."""
        if self.policy == "omit" and (not self.scanned or self.nan is not None):
            return None
        if len(self.n) == 1 or self.paired:
            return self.n[0]
        return tuple(self.n)

    def info(self, alternative=None, **extra):
        """What summary() shows of the call that is not a field (H75): the alternative, the
        sample sizes where known, and extra keys (title, pairs, notes) that are not None."""
        d = {} if alternative is None else {"alternative": alternative}
        n = self.sizes()
        if n is not None:
            d["nobs"] = n
        d.update((k, v) for k, v in extra.items() if v is not None)
        return d

    def propagates(self, i):
        """Lane i's results are NaN by nan_policy='propagate'."""
        return self.policy == "propagate" and self.nan is not None and bool(self.nan[i])

    def out(self, values):
        """A per-lane result (a sequence or a (lanes,) array) in the batch's shape: a Python number
        for one lane without keepdims, else a new float64 array (int64 for ints)."""
        shape = self.keep_shape if self.keepdims else self.batch
        if not shape and not self.keepdims:
            v = values[0]
            return int(v) if isinstance(v, int) and not isinstance(v, bool) else float(v)
        a = values if isinstance(values, _sd.ndarray) else _sd.asarray(list(values))
        if a.dtype == _sd.bool:
            a = a.astype(_sd.float64)
        return _sd.reshape(a, shape)


# ---- tables -------------------------------------------------------------------------------


class Tables:
    """Tables of counts: the last `ndim` axes of x are one table, the leading axes a batch (H9).
    rows is a (lanes, *table) float64 array, batch the batch shape."""

    def __init__(self, x, name, arg, *, ndim=2, integer=True, minimum=None):
        a = _as_array(x, name, arg)
        if a.ndim < ndim:
            raise ValueError(f"{name}(): '{arg}' must have at least {ndim} dimensions (the table's), not {a.ndim}")
        table = tuple(a.shape[a.ndim - ndim :])
        if minimum is not None and any(s < m for s, m in zip(table, minimum)):
            want = " x ".join(str(m) for m in minimum)
            raise ValueError(f"{name}(): '{arg}' must be at least a {want} table, not {' x '.join(map(str, table))}")
        if a.size:
            ok = _sd.isfinite(a) & (a >= 0.0)
            if integer:
                ok = ok & (_sd.floor(a) == a)
            if not bool(_sd.all(ok)):
                kind = "integer counts >= 0" if integer else "finite counts >= 0"
                raise ValueError(f"{name}(): '{arg}' must hold {kind}")
        self.batch = tuple(a.shape[: a.ndim - ndim])
        self.table = table
        self.lanes = math.prod(self.batch)
        self.rows = _sd.reshape(a, (self.lanes,) + table)

    def out(self, values):
        """A per-table result in the batch's shape (a Python number for one table)."""
        if not self.batch:
            v = values[0]
            return int(v) if isinstance(v, int) and not isinstance(v, bool) else float(v)
        a = values if isinstance(values, _sd.ndarray) else _sd.asarray(list(values))
        return _sd.reshape(a, self.batch)


def count(v, name, arg):
    """A non-negative count given as a scalar (binomtest's k and n): an int argument (API.md §2;
    scipy also takes only integers here)."""
    return integer(v, name, arg, minimum=0)
