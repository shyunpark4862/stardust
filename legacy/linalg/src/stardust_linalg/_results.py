"""Named result tuples (API.md §0.3). Field names follow the Array API where it
has the function; matrices are single capitals, vectors lowercase words."""

from collections import namedtuple

SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
LUResult = namedtuple("LUResult", ["L", "U", "perm"])
QRResult = namedtuple("QRResult", ["Q", "R"])
QRPivotedResult = namedtuple("QRPivotedResult", ["Q", "R", "perm"])
LstsqResult = namedtuple("LstsqResult", ["solution", "rank", "singular_values"])
EighResult = namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
SVDResult = namedtuple("SVDResult", ["U", "S", "Vh"])

# The order the C++ core expects in _core._setup (csrc/core/pyrefs.hpp, enum Result).
ALL = (SlogdetResult, LUResult, QRResult, QRPivotedResult, LstsqResult, EighResult, SVDResult)
