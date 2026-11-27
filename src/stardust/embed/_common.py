"""The argument checks and messages sd.embed's functions share (docs/api/embed.md §1.1, §1.2, §1.4; decisions
EM8-EM12, EM44 in docs/design/embed.md §1.1).

The area-neutral checks (data, nan_policy, metric, rng and its key, X_new, precomputed distances, the allocation
check) are stardust._learn's, the same functions and messages as sd.cluster's (decision EM44: the learning areas
agree on every argument check); this module re-exports them and adds the area's own:

- embedded(Z, func, d): the Z of inverse_transform, (m, d) like the embedding (docs/api/embed.md §1.2:
  "'Z' must have 2 columns like the embedding, not 3"), NaN and infinity ValueErrors.
- rows_back(Z, omitted): the embedding of the kept rows at their places, NaN rows for those nan_policy='omit' left
  out (decision EM9).
- one_of_rng(rng, func, needed, where): the rule of decision EM11 for a function whose options decide whether it
  draws: TypeError when it draws and rng is None, ValueError when it does not and rng is given.
"""

import math

from .. import _core as _sd
from .._learn import (  # noqa: F401  (re-exported: the area's modules import them from here)
    METRICS,
    POLICIES,
    START_BITS,
    _array,
    check_alloc,
    check_rng,
    choice,
    condensed_size,
    contiguous,
    distances,
    integer,
    key,
    metric,
    metric_columns,
    new_points,
    no_rng,
    points,
    policy,
    positive_int,
    real,
    rows,
    type_name,
)


def embedded(Z, func, d):
    """The Z of inverse_transform: (m, d) float64 like the embedding; NaN and infinity are ValueErrors."""
    a = points(Z, func, "Z")
    if a.shape[1] != d:
        cols = f"{d} column" + ("" if d == 1 else "s")
        raise ValueError(f"{func}(): 'Z' must have {cols} like the embedding, not {a.shape[1]}")
    if bool(_sd.any(_sd.isnan(a))):
        raise ValueError(f"{func}(): 'Z' contains NaN")
    if bool(_sd.any(_sd.isinf(a))):
        raise ValueError(f"{func}(): 'Z' contains infinity")
    return a


def rows_back(Z, omitted):
    """The (nobs, d) embedding of the kept rows at their places in an (n, d) array of NaN (nan_policy='omit')."""
    if omitted is None:
        return Z
    out = _sd.full((omitted.shape[0], Z.shape[1]), math.nan)
    out[_sd.logical_not(omitted)] = Z
    return out


def one_of_rng(rng, func, needed, where):
    """Decision EM11: the options alone decide whether a call draws. needed: they do (rng must be a Generator;
    where names the option, "init='random'"); not needed: rng must be None (where names the options that draw)."""
    if needed:
        if rng is None:
            raise TypeError(f"{func}(): {where} needs 'rng' (sd.random.default_rng(seed))")
        return check_rng(rng, func)
    no_rng(rng, func, where)
    return None
