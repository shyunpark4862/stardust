// What kneighbors and radius_neighbors (csrc/spatial/neighbors.cpp) take from the distances
// (csrc/spatial/distance.cpp): a cdist computed block by block (docs/design/spatial.md §3.6, decision SPT38).
#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <string>

namespace sd::spatial {

// The cdist of fixed arguments (the same as _cdist's), one block at a time: the rows [i0, i1) of XA
// against the rows [j0, j1) of XB into out ((i1 - i0) x (j1 - j0), row-major), through _cdist's own
// drivers and kernels, so every value has _cdist's bits. Built with the GIL (it reads the arguments,
// which the caller keeps alive); block() touches no Python object and may run in a pool task (its own
// split then runs serially: the pool does not nest).
class CdistBlocks {
public:
    CdistBlocks(
        nanobind::handle xa,
        nanobind::handle xb,
        const std::string& metric,
        double p,
        nanobind::handle w,
        nanobind::handle aux,
        const char* func
    );
    ~CdistBlocks();
    CdistBlocks(const CdistBlocks&) = delete;
    CdistBlocks& operator=(const CdistBlocks&) = delete;

    std::int64_t rows_a() const;
    std::int64_t rows_b() const;
    std::int64_t cols() const;
    // The work of one pair in pair-terms (cdist's task sizes).
    std::int64_t pair_cost() const;
    // 1 for unweighted euclidean, 2 for unweighted sqeuclidean (the metrics of the GEMM pre-selection,
    // docs/design/spatial.md §3.2), else 0.
    int gemm_kind() const;
    // The float64 rows (gemm_kind() != 0 only): rows_a() x cols() and rows_b() x cols(), row-major.
    const double* a_f64() const;
    const double* b_f64() const;
    void block(std::int64_t i0, std::int64_t i1, std::int64_t j0, std::int64_t j1, double* out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sd::spatial
