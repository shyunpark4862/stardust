"""Named result tuples (docs/api/linalg.md §1.6). Field names follow the Array API where it
has the function; matrices are single capitals, vectors lowercase words."""

from collections import namedtuple

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

# The order the C++ core expects in _core._setup (csrc/linalg/pyrefs.hpp, enum Result).
ALL = (SlogdetResult, LUResult, QRResult, QRPivotedResult, LstsqResult, EighResult, SVDResult,
       EigResult, SchurResult, HessenbergResult, LDLResult)
