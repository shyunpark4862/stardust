"""sd.spatial's accuracy sweeps: one per reference family of tests/spatial/ref (tools/spatial_ref.py;
distances, counting, haversine, neighbors) once the functions it measures are public.

SPT0 made the references and the comparators' records (scipy 1.18.1: tests/spatial/data/scipy.json;
scikit-learn 1.9.1: sklearn.json; R 4.6.2 stats::dist: r.json; all from tools/spatial_probe.py) but no
public function, so the area has no sweep yet. The plan (docs/design/spatial.md §7.5): `distances`,
`counting` and `haversine` join with SPT1 (a region per metric, judged by the contract of
docs/design/spatial.md §4 through tests/spatial/refdata.py, the recorded scipy, scikit-learn and R
values as the comparator columns), `neighbors` with SPT2 and SPT3 (the share of queries whose
neighbours equal Arb's: 100% is sd's contract on the lattice sets).
"""

from __future__ import annotations


def sweeps():
    return []
