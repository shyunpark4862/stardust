// The graph area's shared pieces (docs/design/graph.md §3.1, §6): the CSR a graph is given as, the
// undirected reading of it, the transpose and the sentinel.
//
// Every function of stardust._core.graph takes a graph as canonical CSR arrays: int64 indptr (n + 1) and
// indices (nnz), ascending and distinct within a row, and float64 values or none (the functions that read
// only the pattern). The Python layer (src/stardust/graph/_common.py) makes them from any accepted input
// (an sd.sparse array as it is, a dense matrix through _dense_csr) and checks them, so the native side
// only checks that the arrays are the ones it expects (a TypeError names the private function otherwise).
#pragma once

#include "array/ndarray.hpp"
#include "sparse/format.hpp"

#include <nanobind/nanobind.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace sd::graph {

// The predecessor of the start node and of a node no search reached (scipy's NULL_IDX).
constexpr std::int64_t kNoNode = -9999;
// docs/design/array.md §10.4: from this much work (a scan of this many stored entries or nodes) a call
// drops the GIL, and the parallel paths split.
constexpr std::int64_t kParallelWork = 1 << 15;

// A canonical CSR graph read in place (no ownership): data is null when only the pattern is read.
struct View {
    std::int64_t n = 0;
    const std::int64_t* indptr = nullptr;
    const std::int64_t* indices = nullptr;
    const double* data = nullptr;
    std::int64_t nnz() const { return indptr[n]; }
};

// The arrays the Python layer passes as a View (data may be None). TypeError "func(): ..." when they are
// not contiguous int64 (float64) arrays of matching lengths.
View view_arg(
    nanobind::handle indptr, nanobind::handle indices, nanobind::handle data, std::int64_t n, const char* func
);

// The transpose of a graph (entry (i, j) becomes (j, i)): sparse's canonical transpose (the stable counting
// sort of docs/design/sparse.md §2.6, a unique result whatever its path), with the values `values` (the
// float64 weights, or any array of nnz entries when only the pattern is read; then view.data is null).
// Made with the GIL held; it holds arrays, so it is destroyed with the GIL held too.
struct Transposed {
    sparse::Compressed arrays;
    View view;
};
Transposed transpose(const View& g, const NDArray& indptr, const NDArray& indices, const NDArray& values, bool weights);

// Whether a graph is its own undirected reading: its pattern equals its transpose's and, with `weights`
// (and g.data), every stored weight is the undirected_weight of the two directions (bit for bit). One pass
// over the entries without making the transpose (a cursor per row); no Python object. no_diagonal: also
// false when a diagonal entry is stored.
bool symmetric(const View& g, bool weights, bool no_diagonal = false);

// The undirected weight of the two directions when both are stored (docs/design/graph.md §2.1): the
// smaller; NaN when either is NaN (numpy's minimum), and of 0.0 and -0.0 the -0.0, so that the
// result does not depend on which direction is which (G and its transpose read the same).
inline double undirected_weight(double a, double b) {
    if (a != a || b != b) return a != a ? a : b;
    if (a < b) return a;
    if (b < a) return b;
    return std::signbit(a) ? a : b;
}

}  // namespace sd::graph
