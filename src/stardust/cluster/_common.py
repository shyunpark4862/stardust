"""The argument checks and messages sd.cluster's functions share (docs/api/cluster.md §1.1, §1.2, §1.4; decisions
CL5, CL8, CL11, CL12, CL41 in docs/design/cluster.md §1.1).

The area-neutral checks (data, nan_policy, weights, metric, rng and its key, X_new, precomputed distances, the
allocation check) live in stardust/_learn.py, shared with sd.embed (docs/design/embed.md EM44); this module
re-exports them under their old names and adds what only the cluster area uses (labels_back).
"""

from .. import _core as _sd
from .._learn import (  # noqa: F401  (re-exported: the area's modules import them from here)
    _TAKES,
    METRICS,
    POLICIES,
    START_BITS,
    _array,
    _is_numpy_scalar,
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
    weights,
)


def labels_back(labels, omitted):
    """The labels of the kept rows at their places in an (n,) int64 array of -1 (nan_policy='omit')."""
    if omitted is None:
        return labels
    out = _sd.full((omitted.shape[0],), -1, dtype=_sd.int64)
    out[_sd.logical_not(omitted)] = labels
    return out
