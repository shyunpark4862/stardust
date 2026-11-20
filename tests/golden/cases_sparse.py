"""Golden cases for sd.sparse (tools/golden.py has the contract).

sd.sparse has no public function yet (milestone SP0 prepared the area: docs/design/sparse.md §8);
each milestone adds the cases of the names it makes public, on the matrices of
tests/sparse/matrices.py rebuilt from fixed literals and L.rng(seed).
"""


def cases(L):
    return iter(())
