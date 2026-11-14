// bincount and histogram (docs/api/array.md §7.12; T0b), numpy's signatures.
//
// histogram follows numpy's _histograms_impl.histogram step by step, so the
// edges, counts and (uniform bins) weighted sums are numpy's bits: the edges
// come from linspace, each value's bin from numpy's float formula and its
// two corrections, and weights are summed per 65536-element block (a
// bincount from 0.0) and then added block by block.

#include "array/ops/histogram.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/kernels/scalar_ops.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/creation.hpp"
#include "array/ops/reduction.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace sd {

namespace {

constexpr std::int64_t kBlock = 65536;  // numpy's BLOCK

// x in C order as a contiguous array of dtype d.
NDArray c_order(const NDArray& x, DType d) {
    const std::int64_t n = x.size();
    const std::int64_t shape[] = {n};
    NDArray out = new_array(d, shape);
    if (n == 0) return out;
    NDArray dst = out;
    std::int64_t stride = itemsize(d);
    dst.ndim = x.ndim;
    for (int k = x.ndim - 1; k >= 0; --k) {
        dst.shape[k] = x.shape[k];
        dst.strides[k] = stride;
        stride *= x.shape[k];
    }
    dst.update_flags();
    assign(dst, x);
    return out;
}

// ------------------------------------------------------------ bincount

nb::object bincount(nb::handle x_obj, nb::handle weights_obj, nb::handle minlength_obj) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype != DType::Bool && x.dtype != DType::Int64)
        throw nb::type_error((std::string("'x' must be a bool or int64 array, not ") + dtype_name(x.dtype)).c_str());
    if (x.ndim != 1) throw nb::value_error("'x' must be 1-D");
    const std::int64_t minlength = parse_int(minlength_obj, "minlength");
    if (minlength < 0) throw nb::value_error("'minlength' must be non-negative");
    const NDArray xi = c_order(x, DType::Int64);
    const std::int64_t* v = reinterpret_cast<const std::int64_t*>(xi.data);
    const std::int64_t n = xi.size();
    std::int64_t top = -1;
    for (std::int64_t i = 0; i < n; ++i) {
        if (v[i] < 0) throw nb::value_error("'x' must be non-negative");
        top = std::max(top, v[i]);
    }
    const std::int64_t shape[] = {std::max(top + 1, minlength)};
    if (weights_obj.is_none()) {
        NDArray out = new_array(DType::Int64, shape, /*zeroed=*/true);
        std::int64_t* o = reinterpret_cast<std::int64_t*>(out.data);
        for (std::int64_t i = 0; i < n; ++i) ++o[v[i]];
        return wrap(std::move(out));
    }
    const NDArray& w = require_ndarray(weights_obj, "weights");
    if (w.dtype == DType::Complex128) throw nb::type_error("'weights' must be real, not complex128");
    if (w.ndim != 1 || w.shape[0] != n) throw nb::value_error("'weights' must be 1-D and as long as 'x'");
    const NDArray wf = c_order(w, DType::Float64);
    const double* p = reinterpret_cast<const double*>(wf.data);
    NDArray out = new_array(DType::Float64, shape, /*zeroed=*/true);
    double* o = reinterpret_cast<double*>(out.data);
    for (std::int64_t i = 0; i < n; ++i) o[v[i]] += p[i];  // in order, as numpy
    return wrap(std::move(out));
}

// ----------------------------------------------------------- histogram

struct Edges {
    NDArray edges;           // float64 (uniform), or the given array
    bool uniform = false;
    double first = 0, last = 0;
    std::int64_t bins = 0;
};

std::pair<double, double> outer_edges(const std::vector<double>& a, nb::handle range) {
    double first, last;
    if (!range.is_none()) {
        if (!(PyTuple_Check(range.ptr()) || PyList_Check(range.ptr())) || nb::len(range) != 2)
            throw nb::type_error(("'range' must be None or a (min, max) pair, not " + py_type_name(range)).c_str());
        const std::optional<double> lo = real_value(range[0]), hi = real_value(range[1]);
        for (int k = 0; k < 2; ++k)
            if (!(k ? hi : lo))
                throw nb::type_error(("'range' must hold two ints or floats, not " + py_type_name(range[k])).c_str());
        first = *lo;
        last = *hi;
        if (first > last) throw nb::value_error("'range' must have min <= max");
        if (!std::isfinite(first) || !std::isfinite(last)) throw nb::value_error("'range' must be finite");
    } else if (a.empty()) {
        first = 0;
        last = 1;
    } else {
        // numpy's a.min() / a.max(): NaN propagates, and is then not finite.
        first = a[0];
        last = a[0];
        for (double v : a) {
            if (std::isnan(v) || std::isnan(first)) first = last = std::numeric_limits<double>::quiet_NaN();
            else {
                first = std::min(first, v);
                last = std::max(last, v);
            }
        }
        if (!std::isfinite(first) || !std::isfinite(last))
            throw nb::value_error("the data's range is not finite (NaN or inf); pass 'range'");
    }
    if (first == last) {
        first -= 0.5;
        last += 0.5;
    }
    return {first, last};
}

Edges make_edges(const std::vector<double>& a, nb::handle bins, nb::handle range) {
    Edges e;
    std::int64_t nb_ = 0;
    if (index_value(bins, nb_) == IndexResult::Ok) {
        if (nb_ < 1) throw nb::value_error("'bins' must be positive, when an int");
        const auto [first, last] = outer_edges(a, range);
        e.uniform = true;
        e.first = first;
        e.last = last;
        e.bins = nb_;
        const nb::object lin = linspace(nb::float_(first), nb::float_(last), nb::int_(nb_ + 1), nb::none(), nb::bool_(true));
        e.edges = as_ndarray(lin);
        const double* p = reinterpret_cast<const double*>(e.edges.data);
        for (std::int64_t i = 0; i < nb_; ++i)
            if (!(p[i] < p[i + 1]))
                throw nb::value_error(("too many bins for the data range: cannot make " + std::to_string(nb_) +
                                       " finite-sized bins")
                                          .c_str());
        return e;
    }
    if (!is_ndarray(bins)) throw nb::type_error(("'bins' must be an int or a 1-D array, not " + py_type_name(bins)).c_str());
    const NDArray& b = as_ndarray(bins);
    if (b.ndim != 1) throw nb::value_error("'bins' must be 1-D, when an array");
    if (b.dtype != DType::Int64 && b.dtype != DType::Float64)
        throw nb::type_error((std::string("'bins' must be int64 or float64, not ") + dtype_name(b.dtype)).c_str());
    e.edges = c_order(b, b.dtype);  // numpy returns the given edges (and dtype)
    const NDArray ef = c_order(b, DType::Float64);
    const double* p = reinterpret_cast<const double*>(ef.data);
    for (std::int64_t i = 0; i + 1 < b.shape[0]; ++i)
        if (p[i] > p[i + 1]) throw nb::value_error("'bins' must increase monotonically, when an array");
    e.bins = b.shape[0] - 1;
    return e;
}

// The bin of every kept value of a uniform histogram, as numpy computes it;
// -1 for a value outside [first, last] (or NaN).
std::int64_t uniform_bin(double v, const Edges& e, const double* edge) {
    if (!(v >= e.first && v <= e.last)) return -1;
    const double norm_denom = e.last - e.first;
    const double f = ((v - e.first) / norm_denom) * static_cast<double>(e.bins);
    auto i = static_cast<std::int64_t>(f);
    if (i == e.bins) --i;
    if (v < edge[i]) --i;
    if (v >= edge[i + 1] && i != e.bins - 1) ++i;
    return i;
}

// The bin of v among arbitrary sorted edges, the last bin closed: numpy's
// searchsorted 'left' on the inner edges, 'right' on the last.
std::int64_t edges_bin(double v, const double* edge, std::int64_t bins) {
    if (std::isnan(v) || v < edge[0] || v > edge[bins]) return -1;
    if (v == edge[bins]) return bins - 1;  // the last bin is closed on the right
    return (std::upper_bound(edge, edge + bins + 1, v) - edge) - 1;
}

nb::object histogram(nb::handle x_obj, nb::handle bins_obj, nb::handle range_obj, nb::handle weights_obj,
                     nb::handle density_obj) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype == DType::Complex128) throw nb::type_error("not supported for complex128");
    const bool density = parse_bool(density_obj, "density");
    const NDArray xf = c_order(x, DType::Float64);
    const double* xs = reinterpret_cast<const double*>(xf.data);
    const std::vector<double> a(xs, xs + xf.size());
    const Edges e = make_edges(a, bins_obj, range_obj);
    const NDArray ef = c_order(e.edges, DType::Float64);
    const double* edge = reinterpret_cast<const double*>(ef.data);
    const std::int64_t n = static_cast<std::int64_t>(a.size()), bins = e.bins;
    auto bin_of = [&](double v) { return e.uniform ? uniform_bin(v, e, edge) : edges_bin(v, edge, bins); };

    const std::int64_t hshape[] = {bins};
    NDArray hist;
    if (weights_obj.is_none()) {
        hist = new_array(DType::Int64, hshape, /*zeroed=*/true);
        std::int64_t* h = reinterpret_cast<std::int64_t*>(hist.data);
        for (double v : a)
            if (const std::int64_t i = bin_of(v); i >= 0) ++h[i];
    } else {
        const NDArray& w = require_ndarray(weights_obj, "weights");
        if (!std::ranges::equal(w.shape_span(), x.shape_span()))
            throw nb::value_error("'weights' must have the same shape as 'x'");
        const DType wt = w.dtype == DType::Bool ? DType::Int64 : w.dtype;  // bool weights count as int64
        hist = new_array(wt, hshape, /*zeroed=*/true);
        visit_dtype(wt, [&]<typename T>(std::type_identity<T>) {
            if constexpr (!std::is_same_v<T, bool>) {
                const NDArray wc = c_order(w, wt);
                const T* ws = reinterpret_cast<const T*>(wc.data);
                T* h = reinterpret_cast<T*>(hist.data);
                if (e.uniform) {
                    // numpy: per block, np.bincount(indices, weights) -- a sum
                    // from 0.0 in double (complex: real and imaginary apart),
                    // cast back to the weights' dtype -- then n += block.
                    using Acc = std::conditional_t<std::is_same_v<T, complex128>, complex128, double>;
                    std::vector<Acc> blk(bins);
                    for (std::int64_t b0 = 0; b0 < n; b0 += kBlock) {
                        std::fill(blk.begin(), blk.end(), Acc(0));
                        for (std::int64_t k = b0; k < std::min(n, b0 + kBlock); ++k)
                            if (const std::int64_t i = bin_of(a[k]); i >= 0) {
                                if constexpr (std::is_same_v<T, complex128>)
                                    blk[i] = {blk[i].real() + ws[k].real(), blk[i].imag() + ws[k].imag()};
                                else blk[i] += static_cast<double>(ws[k]);
                            }
                        for (std::int64_t i = 0; i < bins; ++i) {
                            if constexpr (std::is_same_v<T, complex128>) h[i] = {h[i].real() + blk[i].real(), h[i].imag() + blk[i].imag()};
                            else h[i] = h[i] + static_cast<T>(blk[i]);
                        }
                    }
                } else {
                    // numpy: per block, sort by value, cumulative sums of the
                    // weights read at the edges, accumulated; n = diff.
                    std::vector<T> cum(bins + 1, T(0));
                    std::vector<std::int64_t> order;
                    std::vector<T> cw;
                    for (std::int64_t b0 = 0; b0 < n; b0 += kBlock) {
                        const std::int64_t len = std::min(n, b0 + kBlock) - b0;
                        order.resize(len);
                        std::iota(order.begin(), order.end(), b0);
                        std::stable_sort(order.begin(), order.end(), [&](std::int64_t p, std::int64_t q) {
                            const double u = a[p], v = a[q];
                            return u < v || (std::isnan(v) && !std::isnan(u));
                        });
                        cw.assign(len + 1, T(0));
                        for (std::int64_t k = 0; k < len; ++k) {
                            if constexpr (std::is_same_v<T, complex128>)
                                cw[k + 1] = {cw[k].real() + ws[order[k]].real(), cw[k].imag() + ws[order[k]].imag()};
                            else cw[k + 1] = cw[k] + ws[order[k]];
                        }
                        // searchsorted of each edge in the sorted block: left for the inner edges, right for the last.
                        auto less = [&](double u, double v) { return u < v || (std::isnan(v) && !std::isnan(u)); };
                        std::vector<double> sorted(len);
                        for (std::int64_t k = 0; k < len; ++k) sorted[k] = a[order[k]];
                        for (std::int64_t j = 0; j <= bins; ++j) {
                            const auto at = j < bins ? std::lower_bound(sorted.begin(), sorted.end(), edge[j], less)
                                                     : std::upper_bound(sorted.begin(), sorted.end(), edge[j], less);
                            const T add = cw[at - sorted.begin()];
                            if constexpr (std::is_same_v<T, complex128>) cum[j] = {cum[j].real() + add.real(), cum[j].imag() + add.imag()};
                            else cum[j] = cum[j] + add;
                        }
                    }
                    for (std::int64_t i = 0; i < bins; ++i) {
                        if constexpr (std::is_same_v<T, complex128>) h[i] = {cum[i + 1].real() - cum[i].real(), cum[i + 1].imag() - cum[i].imag()};
                        else h[i] = cum[i + 1] - cum[i];
                    }
                }
            }
        });
    }

    nb::object edges_out = wrap(NDArray(e.edges));
    if (density) {
        // numpy: n / db / n.sum(), db = diff(edges) as float64.
        const nb::object total = sum(wrap(NDArray(hist)), nb::none(), nb::none(), nb::bool_(false));
        NDArray out = new_array(hist.dtype == DType::Complex128 ? DType::Complex128 : DType::Float64, hshape);
        visit_dtype(hist.dtype, [&]<typename T>(std::type_identity<T>) {
            if constexpr (!std::is_same_v<T, bool>) {
                const T* h = reinterpret_cast<const T*>(hist.data);
                if constexpr (std::is_same_v<T, complex128>) {
                    const complex128 t = nb::cast<complex128>(total);
                    complex128* o = reinterpret_cast<complex128*>(out.data);
                    for (std::int64_t i = 0; i < bins; ++i) {
                        const double db = edge[i + 1] - edge[i];
                        o[i] = ops::cdiv(complex128(h[i].real() / db, h[i].imag() / db), t);
                    }
                } else {
                    const double t = nb::cast<double>(total);
                    double* o = reinterpret_cast<double*>(out.data);
                    for (std::int64_t i = 0; i < bins; ++i) o[i] = static_cast<double>(h[i]) / (edge[i + 1] - edge[i]) / t;
                }
            }
        });
        hist = std::move(out);
    }
    return nb::module_::import_("stardust._results").attr("HistogramResult")(wrap(std::move(hist)), edges_out);
}

}  // namespace

void bind_histogram(nb::module_& m) {
    using nb::arg;
    m.def("bincount", named("bincount", &bincount), arg().none(), nb::kw_only(), arg("weights").none() = nb::none(),
          arg("minlength").none() = 0);
    m.def("histogram", named("histogram", &histogram), arg().none(), nb::kw_only(), arg("bins").none() = 10,
          arg("range").none() = nb::none(), arg("weights").none() = nb::none(), arg("density").none() = false);
}

}  // namespace sd
