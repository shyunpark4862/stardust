"""The input forms of a test graph (docs/api/graph.md §1.2): the four sd.sparse formats and, when the dense
rule reads the same edges (no stored 0, NaN or +-inf), the dense forms."""

from __future__ import annotations

import numpy as np

import stardust as sd

from . import graphs as G


def square_ids() -> list[str]:
    """The square test graphs (the rectangular biadjacency matrices and the cost matrices left out)."""
    return [i for i in G.ids() if G.get(i).square and "cost" not in G.get(i).tags]


def csr(g: G.Graph):
    if g.matrix is not None:
        return sd.graph.csgraph_from_dense(g.matrix)
    return sd.sparse.csr_array((g.data, g.indices, g.indptr), shape=g.shape)


def dense_reads_the_same(g: G.Graph) -> bool:
    if g.matrix is not None:
        return True
    d = g.data.astype(np.float64)
    return bool(np.all((d != 0) & np.isfinite(d)))


def dense(g: G.Graph) -> np.ndarray:
    """The dense matrix (the graph's own for the dense-only graphs; the values otherwise, in their dtype)."""
    if g.matrix is not None:
        return g.matrix.copy()
    out = np.zeros(g.shape, dtype=g.data.dtype)
    for i, j, w in g.edges():
        out[i, j] = w
    return out


def forms(g: G.Graph) -> dict:
    """{form name: input} of a square test graph: csr, csc, coo, dok, and dense (sd, numpy, list) when it
    reads the same edges."""
    A = csr(g)
    out = {"csr": A, "csc": A.tocsc(), "coo": A.tocoo(), "dok": A.todok()}
    if dense_reads_the_same(g):
        D = dense(g)
        out |= {"sd": sd.asarray(D), "numpy": D}
        if g.n:  # [] is a 1-D sequence
            out["list"] = D.tolist()
    return out
