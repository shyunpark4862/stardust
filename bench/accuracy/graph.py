"""sd.graph's accuracy sweeps: one per reference family of tests/graph/ref (tools/graph_ref.py) once
the functions it measures are public.

GR0 made the references and the comparator's record (scipy 1.18.1: tests/graph/data/scipy.json, from
tools/graph_probe.py) but no public function, so the area has no sweep yet. The plan
(docs/design/graph.md §7.5): `paths` joins with GR2 (the general-weight distance contract in units
of tests/graph/refdata.bound, scipy's D and FW as the comparator columns), `assignment` with GR3
(the general-cost contract; scipy's linear_sum_assignment), `laplacian` with GR4, `pagerank` and
`betweenness` with GR5.
"""

from __future__ import annotations


def sweeps():
    return []
