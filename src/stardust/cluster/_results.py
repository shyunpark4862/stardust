"""The common base of sd.cluster's results (docs/api/cluster.md §1.3, docs/design/cluster.md §2.1, §2.7;
decisions CL1, CL6, CL44): KMeansResult, KMedoidsResult, FuzzyCMeansResult, HierarchyResult, DBSCANResult,
HDBSCANResult, OPTICSResult, GaussianMixtureResult, AffinityPropagationResult and GapResult derive from it.

A result holds its fields as given by the fitting function: arrays become read-only (API.md §2.2), a field
given as a callable is computed when first read and kept (a lazy field), and no field can be set afterwards.
Every result has `labels` ((n,) int64 in the input's order: 0 .. K - 1 numbered by first appearance, -1 for
noise and for the rows nan_policy='omit' left out), `n_clusters` (K; noise not counted), `nobs` (the points
fitted) and `omitted` ((n,) bool). A result does not hold a copy of the data (§1.3 of the API document).

summary() is the regression box of stardust._summary (API.md §2.3): a header of the method, nobs, the
clusters, the noise and omitted rows when there are any, the objective, the iterations and convergence on the
left and the subclass's tuning on the right, then one table row per cluster: its size and the subclass's
per-cluster columns (a table of more than 20 clusters shows 5 at each end). repr is one line of the
subclass's main fields.

A subclass sets (private class constants, like regression's _I_NOBS and smooth's SmootherResult)
    _TITLE      the box's title ("K-Means Results")
    _METHOD     the header's method name ("k-means (Lloyd)"); a field name in braces reads that field
    _OBJECTIVE  (label, field) of the objective's header row, or None ("Inertia:", "inertia")
    _TUNING     (label, field) rows of the header's right column: the tuning parameters
    _COLUMNS    (column, field) of the per-cluster table after Size: fields of shape (K,)
    _REPR       the fields of the one-line repr

Every public method of a result (summary, predict, cut, ...) reports a wrong call shape under its own name, as
the other areas' results do (API.md §2.1): "predict(): missing 1 required positional argument: 'X_new'".
"""

from .. import _calls, _summary
from .. import _core as _sd

_MISSING = object()


class ClusterResult:
    """Base class: ClusterResult(**fields). Not a public constructor (docs/api/cluster.md §1.3)."""

    _TITLE = "Clustering Results"
    _METHOD = ""
    _OBJECTIVE = None
    _TUNING = ()
    _COLUMNS = ()
    _REPR = ("nobs", "n_clusters")

    def __init__(self, **fields):
        values = {}
        for k, v in fields.items():
            if isinstance(v, _sd.ndarray):
                v._set_readonly()
            values[k] = v
        object.__setattr__(self, "_fields", values)

    def __getattr__(self, name):
        fields = object.__getattribute__(self, "_fields")
        v = fields.get(name, _MISSING)
        if v is _MISSING:
            raise AttributeError(f"{type(self).__name__} has no field {name!r}")
        if callable(v) and not isinstance(v, _sd.ndarray):
            v = v()
            if isinstance(v, _sd.ndarray):
                v._set_readonly()
            fields[name] = v
        return v

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__}: the fields are read-only")

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__}: the fields are read-only")

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(object.__getattribute__(self, "_fields")))

    def _has(self, name):
        return name in object.__getattribute__(self, "_fields")

    def _cell(self, field):
        v = getattr(self, field)
        if v is None:
            return "None"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, str):
            return v
        if isinstance(v, int):
            return str(v)
        if isinstance(v, _sd.ndarray):
            return _summary.compact(v)
        return _summary.stat(v)

    def _sizes(self):
        """(cluster sizes (K,), noise points, omitted rows) from labels and omitted."""
        k = int(self.n_clusters)
        sizes = [0] * k
        noise = 0
        for v in self.labels.tolist():
            if v >= 0:
                sizes[v] += 1
            else:
                noise += 1
        omitted = int(_sd.count_nonzero(self.omitted)) if self._has("omitted") else 0
        return sizes, noise - omitted, omitted

    def summary(self):
        """The regression box: the method, the sample, the clusters, the objective and the iterations on the
        left, the tuning on the right; one table row per cluster (its size and the per-cluster columns)."""
        sizes, noise, omitted = self._sizes()
        method = self._METHOD.format(**{k: self._cell(k) for k in _fields_in(self._METHOD)})
        left = [
            ("Method:", method),
            ("No. Observations:", _summary.count(self.nobs)),
            ("No. Clusters:", str(int(self.n_clusters))),
        ]
        if noise:
            left.append(("Noise points:", str(noise)))
        if omitted:
            left.append(("Omitted rows (NaN):", str(omitted)))
        if self._OBJECTIVE is not None:
            label, field = self._OBJECTIVE
            left.append((label, _summary.stat(getattr(self, field))))
        if self._has("n_iter"):
            left.append(("Iterations:", str(int(self.n_iter))))
        if self._has("converged"):
            left.append(("Converged:", _summary.flag(self.converged)))
        right = [(label, self._cell(field)) for label, field in self._TUNING]
        columns = ["Size"] + [c for c, _ in self._COLUMNS]
        values = [getattr(self, f).tolist() for _, f in self._COLUMNS]
        rows = []
        for i in _summary.shown(len(sizes)):
            if i is None:
                rows.append(None)
                continue
            rows.append((str(i), [str(sizes[i])] + [_summary.forg(v[i], 4) for v in values]))
        table = _summary.Table(columns, rows, stub_head="cluster")
        return _summary.render(self._TITLE, header=(left, right), tables=(table,) if rows else ())

    def __repr__(self):
        return _summary.one_line(type(self).__name__, [(k, getattr(self, k)) for k in self._REPR])


def _fields_in(text):
    """The field names a format string reads ("k-means ({init})" -> ["init"])."""
    import string

    return [name for _, name, _, _ in string.Formatter().parse(text) if name]


_calls.methods(ClusterResult)
