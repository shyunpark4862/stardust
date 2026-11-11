"""Named result tuples (API.md §0.3). Field names follow the Array API where it
has the function; matrices are single capitals, vectors lowercase words."""

from collections import namedtuple

SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
LUResult = namedtuple("LUResult", ["L", "U", "perm"])

# The order the C++ core expects in _core._setup (csrc/core/pyrefs.hpp, enum Result).
ALL = (SlogdetResult, LUResult)
