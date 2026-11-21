"""sd.integrate's accuracy sweeps: one per reference family of tests/integrate/ref
(tools/integrate_ref.py; integrals, series, multiple, vector) once the functions it measures are
public.

C0 made the references and scipy 1.18.1's records (tests/integrate/data/scipy_*.json,
tools/gen_integrate_scipy.py) and the QUADPACK Fortran originals' (fortran_quadpack.json,
tools/gen_integrate_fortran.py) but no public function, so the area has no sweep yet. The plan
(docs/design/integrate.md §4, §7.5): a sweep per function joins with the milestone that makes it
public (quad with C4 and C5, tanhsinh and nsum with C6, quad_vec and cubature with C7), measuring
the relative error against the reference and the honesty of the error estimate (estimate over
the true error; a converged result below 1 is counted), with scipy's recorded values as the
comparator column.
"""

from __future__ import annotations


def sweeps():
    return []
