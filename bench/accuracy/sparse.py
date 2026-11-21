"""sd.sparse's accuracy sweeps: one per reference family of tests/sparse/ref (tools/sparse_ref.py;
spmv, sums, norms, solve, lstsq, eig, expm) once the functions it measures are public.

SP0 made the references and the comparators' records (scipy 1.18.1: tests/sparse/data/scipy.json,
tools/gen_sparse_scipy.py; R 4.6.2 Matrix 1.7.6: tests/sparse/data/r.json, tools/gen_sparse_r.py).
The public functions so far are SP1's formats and conversions: exact operations (repeats summed
in input order by definition), which the tests check bit for bit, so the area has no sweep yet.
The plan (docs/design/sparse.md §7): a family joins when the milestone that makes its functions
public lands (spmv and sums with SP2, solve with SP4, lstsq with SP5, eig with SP6, norms and expm
with SP7), judged by the contract of tests/sparse/refdata.py (docs/design/sparse.md §4), with the
recorded scipy and R values as the comparator columns.
"""

from __future__ import annotations


def sweeps():
    return []
