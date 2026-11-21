"""sd.differentiate's accuracy sweeps: one per reference family of tests/differentiate/ref
(tools/differentiate_ref.py; the elementary functions' derivatives, the MGH Jacobians and
Hessians) once the functions it measures are public.

C0 made the references and scipy 1.18.1's records (tests/differentiate/data/scipy_*.json,
tools/gen_differentiate_scipy.py) but no public function, so the area has no sweep yet. The plan
(docs/design/integrate.md §4, §7.5): derivative, jacobian and hessian join with milestone C2,
measuring the relative error against the analytic derivative and whether a converged result
meets error < atol + rtol |df|, with scipy's recorded values as the comparator column.
"""

from __future__ import annotations


def sweeps():
    return []
