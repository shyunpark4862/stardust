"""Accuracy measurement against Arb and exact references (bench/README.md, which also lists the
reference files; their layout: tests/reffile.py).

The areas are bench/areas.py's that have references: array, linalg, special, stats (with its
multivariate group, sweeps stats/mv/...), regression, optimize, htest, sparse, integrate,
differentiate, spatial (its distances since SPT1; the neighbour searches come with SPT2 to SPT4),
graph (no sweep yet: its functions come with milestones GR1 to GR5), interpolate (sd.interp's linear
sweep since IP1; the others with IP2 to IP6) and fft (no sweep yet: milestones FT1 to FT5).
Each area's sweeps are its reference files, read through tests/<area>/refdata.py, which also
holds the contract the tests judge with (special: tests/special/test_reference.py), so a run
computes only the stardust side and judges it as the tests do. A region of a sweep gets its
error in ULP (ulp_distance, bench/accuracy/common.py) and the area's own measures (relative,
mixed, residual, ...), and its error against the contract (1 is the bound). A run is a JSON file
in build/accuracy/; a baseline per area is bench/accuracy/results/<area>.json;
bench/<area>/accuracy.md (and the stats area's bench/stats/accuracy_multivariate.md) is
generated from it.
"""
