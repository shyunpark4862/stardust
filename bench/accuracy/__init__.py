"""Accuracy measurement against Arb references (bench/README.md, tests/README.md).

The sweeps are the reference files the tests already hold (tests/special/ref, tests/stats/ref:
inputs, the reference hi + lo, the guaranteed region and the bound's scale, made by
tools/special_ref.py and tools/stats_ref.py), so a run computes only the stardust side. Each
(function, region) gets its error in ULP, its relative error, ULP / scale for every function
whose bound has a scale above 1 somewhere (max(1, kappa) for the inverses, the envelope near
the zeros of the Bessel, Airy and complex error functions), and its error against the contract
(tests/<area>/test_reference.py's measure: 1 is the bound). A run is a JSON file in build/accuracy/; a baseline per area is
bench/accuracy/results/<area>.json; bench/<area>/accuracy.md is generated from it.
"""
