"""Named result tuples of every area, in one module (DESIGN.md §9.7). Each is
named after its function: <FunctionName>Result in CamelCase, with the Array
API's spellings (UniqueAllResult, EighResult, QRResult, SVDResult, ...).

Field names follow the Array API where it has the function
(docs/api/array.md §8.3, docs/api/linalg.md §1.6)."""

from collections import namedtuple

from . import _calls

# array: unique_* (built by csrc/array/ops/sort.cpp, which looks them up by name)
UniqueAllResult = namedtuple("UniqueAllResult", ["values", "indices", "inverse_indices", "counts"])
UniqueCountsResult = namedtuple("UniqueCountsResult", ["values", "counts"])
UniqueInverseResult = namedtuple("UniqueInverseResult", ["values", "inverse_indices"])

# array: finfo / iinfo (built by csrc/array/ops/dtype_info.cpp, by name); the Array API's attributes
FinfoResult = namedtuple("FinfoResult", ["bits", "eps", "max", "min", "smallest_normal", "dtype"])
IinfoResult = namedtuple("IinfoResult", ["bits", "max", "min", "dtype"])
# array: histogram (csrc/array/ops/histogram.cpp, by name); numpy's (hist, bin_edges) order
HistogramResult = namedtuple("HistogramResult", ["hist", "bin_edges"])

# linalg (built by the C++ core, which receives them in LINALG)
SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
LUResult = namedtuple("LUResult", ["L", "U", "perm"])
QRResult = namedtuple("QRResult", ["Q", "R"])
QRPivotedResult = namedtuple("QRPivotedResult", ["Q", "R", "perm"])
LstsqResult = namedtuple("LstsqResult", ["solution", "rank", "singular_values"])
EighResult = namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
SVDResult = namedtuple("SVDResult", ["U", "S", "Vh"])
EigResult = namedtuple("EigResult", ["eigenvalues", "eigenvectors"])
SchurResult = namedtuple("SchurResult", ["T", "Z"])
HessenbergResult = namedtuple("HessenbergResult", ["H", "Q"])
LDLResult = namedtuple("LDLResult", ["L", "D", "perm"])

# special: airy (built by csrc/bind/special_bind.cpp, which looks it up by name)
AiryResult = namedtuple("AiryResult", ["ai", "aip", "bi", "bip"])

# regression: the linear models' diagnostics (docs/api/regression.md §2; statsmodels' het_breuschpagan
# fields and order). Jarque-Bera's result is sd.htest's class (G165, htest H18, H39). The three
# tuples below have summary() (G191, G192): what it shows beyond the fields is in _info, set by
# the function that makes the tuple (no slots, so the tuple has a __dict__ for it).


class BreuschPaganResult(namedtuple("BreuschPaganResult", ["lm", "lm_pvalue", "fvalue", "f_pvalue"])):
    @_calls.method("summary")
    def summary(self):
        """The test as a text table (G191): LM and F statistics with their p-values."""
        from . import _summary

        info = getattr(self, "_info", {})
        batch = tuple(self.lm.shape) if hasattr(self.lm, "shape") else ()
        title = info.get("title", "Breusch-Pagan test for heteroscedasticity")
        return _summary.test_box(title, list(zip(self._fields, self)), batch, info)


# regression: theilslopes, siegelslopes (scipy.stats' fields and order, docs/api/regression.md §5)
class TheilslopesResult(namedtuple("TheilslopesResult", ["slope", "intercept", "low_slope", "high_slope"])):
    @_calls.method("summary")
    def summary(self):
        """The estimate as a text table (G192): slope and intercept, the slope's interval."""
        from . import _summary

        return _summary.slopes_box(self, "Theil-Sen Regression Results")


class SiegelslopesResult(namedtuple("SiegelslopesResult", ["slope", "intercept"])):
    @_calls.method("summary")
    def summary(self):
        """The estimate as a text table (G192): slope and intercept."""
        from . import _summary

        return _summary.slopes_box(self, "Siegel Repeated Medians Regression Results")


# The order the C++ core expects in _core.linalg._setup (csrc/linalg/pyrefs.hpp, enum Result).
LINALG = (
    SlogdetResult,
    LUResult,
    QRResult,
    QRPivotedResult,
    LstsqResult,
    EighResult,
    SVDResult,
    EigResult,
    SchurResult,
    HessenbergResult,
    LDLResult,
)

# stats: describe, mode (scipy.stats' fields and order, docs/api/stats.md §4)
DescribeResult = namedtuple("DescribeResult", ["nobs", "minmax", "mean", "variance", "skewness", "kurtosis"])
ModeResult = namedtuple("ModeResult", ["mode", "count"])
# stats: MultivariateNormal.cdf and MultivariateT.cdf with return_error=True (docs/api/stats.md §3)
CdfResult = namedtuple("CdfResult", ["value", "error"])
