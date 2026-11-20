// The argument and result helpers of sd.htest's lane kernels (htest/parametric.cpp, htest/gof.cpp):
// the samples come as 2-D float64 arrays, one lane (one test of a batch) per row, and the
// results go back as 1-D float64 arrays of the lanes.
#pragma once

#include "array/ndarray.hpp"
#include "array/ops/pieces.hpp"
#include "array/parallel_limits.hpp"
#include "bind/pyutil.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace sd::htest {

inline const NDArray& rows_arg(nanobind::handle o) {
    const NDArray& a = require_ndarray(o);
    if (a.dtype != DType::Float64 || a.ndim != 2) throw nanobind::type_error("htest: expects a 2-D float64 array");
    return a;
}
inline const NDArray& vec_arg(nanobind::handle o, std::int64_t n) {
    const NDArray& a = require_ndarray(o);
    if (a.dtype != DType::Float64 || a.ndim != 1 || a.shape[0] != n)
        throw nanobind::type_error("htest: expects a 1-D float64 array of the lanes");
    return a;
}
inline double vec_at(const NDArray& a, std::int64_t i) {
    return *reinterpret_cast<const double*>(a.data + i * a.strides[0]);
}

// k new 1-D float64 arrays of n elements and their data
struct Outs {
    std::vector<NDArray> arrays;
    std::vector<double*> p;
    Outs(int k, std::int64_t n) {
        const std::int64_t shape[] = {n};
        for (int i = 0; i < k; ++i) {
            arrays.push_back(new_array(DType::Float64, shape));
            p.push_back(reinterpret_cast<double*>(arrays.back().data));
        }
    }
    nanobind::tuple tuple() {
        nanobind::list l;
        for (NDArray& a : arrays) l.append(wrap(std::move(a)));
        return nanobind::tuple(l);
    }
};

// fn(r) for every lane on the pool, `cost` the work of one lane in elements (roughly)
template <class F>
void each_lane(std::int64_t lanes, std::int64_t cost, F&& fn) {
    const std::int64_t c = std::max<std::int64_t>(1, cost);
    const std::int64_t per = std::max<std::int64_t>(1, kParallelTask / c);
    for_pieces(lanes, per, lanes * c, [&](std::int64_t, std::int64_t b, std::int64_t e) {
        for (std::int64_t r = b; r < e; ++r) fn(r);
    });
}

}  // namespace sd::htest
