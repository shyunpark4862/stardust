"""Golden cases for sd.spatial (tools/golden.py has the contract).

sd.spatial has no public function yet (milestone SPT0 prepared the area: docs/design/spatial.md §8);
each milestone adds the cases of the names it makes public, on the points of
tests/spatial/points.py rebuilt from fixed literals and L.rng(seed).
"""


def cases(L):
    return iter(())
