"""Named result tuples of every area, in one module (DESIGN.md §9.7). Each is
named after its function: <FunctionName>Result in CamelCase, with the Array
API's spellings (UniqueAllResult, EighResult, QRResult, SVDResult, ...).

Field names follow the Array API where it has the function
(docs/api/array.md §8.3, docs/api/linalg.md §1.6)."""

from collections import namedtuple

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

# regression: linregress (scipy.stats' fields and order, docs/api/regression.md §2)
LinregressResult = namedtuple(
    "LinregressResult", ["slope", "intercept", "rvalue", "pvalue", "stderr", "intercept_stderr"]
)
# regression: the linear models' diagnostics (docs/api/regression.md §2; statsmodels' het_breuschpagan
# fields and order)
JarqueBeraResult = namedtuple("JarqueBeraResult", ["statistic", "pvalue"])
BreuschPaganResult = namedtuple("BreuschPaganResult", ["lm", "lm_pvalue", "fvalue", "f_pvalue"])
# regression: theilslopes, siegelslopes (scipy.stats' fields and order, docs/api/regression.md §5)
TheilslopesResult = namedtuple("TheilslopesResult", ["slope", "intercept", "low_slope", "high_slope"])
SiegelslopesResult = namedtuple("SiegelslopesResult", ["slope", "intercept"])

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
