"""The common base of sd.embed's results (docs/api/embed.md §1.3, docs/design/embed.md §2.1, §2.7; decisions EM7,
EM44, EM45): PCAResult, TruncatedSVDResult, RandomProjectionResult, FactorAnalysisResult, FastICAResult, NMFResult,
KernelPCAResult, ClassicalMDSResult, MDSResult, IsomapResult, LLEResult, SpectralEmbeddingResult, TSNEResult and
UMAPResult derive from it.

The field mechanics are stardust._learn.LearnResult's, the same as sd.cluster's results (decision EM44): arrays
become read-only (API.md §2.2), a field given as a callable is computed when first read and kept (a lazy field),
and no field can be set or deleted afterwards. Every result has `embedding` ((n, d) float64 in the input's order;
NaN rows for the rows nan_policy='omit' left out), `n_components` (d), `nobs` (the points fitted) and `omitted`
((n,) bool). A result does not hold a copy of the data (§1.3 of the API document).

summary() is the regression box of stardust._summary (API.md §2.3; decision EM45): a header of the method, nobs,
the components, the omitted rows when there are any, the objective, the iterations and convergence on the left
and the subclass's tuning on the right; a result with components then has one table row per component (its
columns, and the cumulative ratio when the subclass names a ratio field; a table of more than 20 components shows
5 at each end); a subclass may add tables of its own (factor analysis: the uniquenesses). Nonlinear results have
no table. repr is one line of the subclass's main fields.

A subclass sets (private class constants, like ClusterResult's)
    _TITLE       the box's title ("PCA Results")
    _METHOD      the header's method name ("PCA ({svd_solver})"); a field name in braces reads that field
    _OBJECTIVE   (label, field) of the objective's header row, or None ("KL divergence:", "kl_divergence")
    _TUNING      (label, field) rows of the header's right column: the tuning parameters
    _COLUMNS     (column, field) of the per-component table: fields of shape (d,); empty for no table
    _CUMULATIVE  the field of shape (d,) whose running sum is the table's last column ("Cumulative"), or None
    _REPR        the fields of the one-line repr
and may override _tables() (more tables after the components').

Every public method of a result (summary, transform, inverse_transform, ...) reports a wrong call shape under its
own name, as the other areas' results do (API.md §2.1): "transform(): missing 1 required positional argument:
'X_new'".
"""

from .. import _calls, _summary
from .._learn import LearnResult
from .._learn import fields_in as _fields_in


class EmbeddingResult(LearnResult):
    """Base class: EmbeddingResult(**fields). Not a public constructor (docs/api/embed.md §1.3)."""

    _TITLE = "Embedding Results"
    _METHOD = ""
    _OBJECTIVE = None
    _TUNING = ()
    _COLUMNS = ()
    _CUMULATIVE = None
    _REPR = ("nobs", "n_components")

    def _tables(self):
        """Tables after the components' (none here)."""
        return ()

    def _component_table(self):
        if not self._COLUMNS:
            return None
        values = [getattr(self, f).tolist() for _, f in self._COLUMNS]
        d = len(values[0])
        cum = None
        if self._CUMULATIVE is not None:
            cum, s = [], 0.0
            for v in getattr(self, self._CUMULATIVE).tolist():
                s += v
                cum.append(s)
        columns = [c for c, _ in self._COLUMNS] + (["Cumulative"] if cum is not None else [])
        rows = []
        for i in _summary.shown(d):
            if i is None:
                rows.append(None)
                continue
            cells = [_summary.forg(v[i], 4) for v in values]
            if cum is not None:
                cells.append(_summary.forg(cum[i], 4))
            rows.append((str(i), cells))
        return _summary.Table(columns, rows, stub_head="component")

    def summary(self):
        """The regression box: the method, the sample, the components, the objective and the iterations on the
        left, the tuning on the right; one table row per component when the result has components."""
        method = self._METHOD.format(**{k: self._cell(k) for k in _fields_in(self._METHOD)})
        left = [
            ("Method:", method),
            ("No. Observations:", _summary.count(self.nobs)),
            ("No. Components:", str(int(self.n_components))),
        ]
        omitted = self._omitted_count()
        if omitted:
            left.append(("Omitted rows (NaN):", str(omitted)))
        if self._OBJECTIVE is not None:
            label, field = self._OBJECTIVE
            left.append((label, _summary.stat(getattr(self, field))))
        if self._has("n_iter") and self.n_iter is not None:
            left.append(("Iterations:", str(int(self.n_iter))))
        if self._has("converged") and self.converged is not None:
            left.append(("Converged:", _summary.flag(self.converged)))
        right = [(label, self._cell(field)) for label, field in self._TUNING]
        table = self._component_table()
        tables = ((table,) if table is not None else ()) + tuple(self._tables())
        return _summary.render(self._TITLE, header=(left, right), tables=tables)


_calls.methods(EmbeddingResult)
