"""Golden cases for sd.interpolate (tools/golden.py has the contract; docs/design/interpolate.md §7.5).

sd.interpolate has no public function yet (milestone IP0 prepared the area: docs/design/interpolate.md
§8); each milestone adds the cases of the names it makes public: every public name on the edge data of
tests/interpolate/samples.py (rebuilt from fixed literals and L.rng(seed)), the three values of
extrapolate, and the wrong calls. sd.interp's cases go to the array area's cases_array.py (IP1).
"""


def cases(L):
    return iter(())
