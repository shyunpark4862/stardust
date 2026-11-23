// bincount and histogram (docs/api/array.md §7.12; T0b), numpy's signatures.
//
// histogram follows numpy's _histograms_impl.histogram step by step, so the
// edges, counts and the weighted sums of uniform bins are numpy's bits: the
// edges come from linspace, each value's bin from numpy's float formula and its
// two corrections, and weights are summed per 65536-element block (a
// bincount from 0.0) and then added block by block. With given edges the
// weights are summed per bin in the same blocks, not as numpy's differences of
// sorted cumulative sums, so float weights there are not numpy's last bits
// (docs/api/array.md §7.12).

#include "array/ops/histogram.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/kernels/scalar_ops.hpp"
#include "array/ops/copy.hpp"
#include "array/ops/creation.hpp"
#include "array/ops/reduction.hpp"
#include "array/ops/pieces.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace sd {

namespace {

constexpr std::int64_t kBlock = 65536;  // numpy's BLOCK

// x in C order as a contiguous array of dtype d: x itself when it already
// is one (no copy), else a converted copy.
NDArray c_order(const NDArray& x, DType d) {
    if (x.dtype == d && x.c_contiguous) return x;
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

// Counting in parallel: [0, n) in kGroups contiguous ranges, each into its own
// counts, then summed. Integer counts, so the result is the same for any
// split. Used when the per-group counts are small next to the input.
constexpr std::int64_t kGroups = 16;

bool use_groups(std::int64_t n, std::int64_t bins) {
    return n >= kParallelThreshold && bins * kGroups <= n && parallel::available();
}

// counts[bin_of(i)] += 1 for i in [0, n), skipping bins < 0.
template <typename BinOf>
void count_into(std::int64_t* counts, std::int64_t bins, std::int64_t n, BinOf&& bin_of) {
    if (!use_groups(n, bins)) {
        const parallel::Unlocked unlocked(n >= kParallelThreshold);
        for (std::int64_t i = 0; i < n; ++i)
            if (const std::int64_t b = bin_of(i); b >= 0) ++counts[b];
        return;
    }
    std::vector<std::int64_t> local(static_cast<size_t>(kGroups * bins), 0);
    for_pieces(n, (n + kGroups - 1) / kGroups, n, [&](std::int64_t g, std::int64_t b0, std::int64_t e0) {
        std::int64_t* c = local.data() + g * bins;
        for (std::int64_t i = b0; i < e0; ++i)
            if (const std::int64_t b = bin_of(i); b >= 0) ++c[b];
    });
    for (std::int64_t g = 0; g < kGroups; ++g)
        for (std::int64_t b = 0; b < bins; ++b) counts[b] += local[g * bins + b];
}

// ------------------------------------------------------------ bincount

nb::object bincount(nb::handle x_obj, nb::handle weights_obj, nb::handle minlength_obj) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype != DType::Bool && x.dtype != DType::Int64)
        throw nb::type_error((std::string("'x' must be a bool or int64 array, not ") + dtype_name(x.dtype)).c_str());
    if (x.ndim != 1) throw nb::value_error("'x' must be 1-D");
    const std::int64_t minlength = parse_int(minlength_obj, "minlength");
    if (minlength < 0) throw nb::value_error("'minlength' must be >= 0");
    const NDArray xi = c_order(x, DType::Int64);
    const std::int64_t* v = reinterpret_cast<const std::int64_t*>(xi.data);
    const std::int64_t n = xi.size();
    // min and max in one branch-free pass (in parallel pieces when long).
    std::int64_t lo = 0, top = -1;
    {
        std::vector<std::int64_t> mins(kGroups, 0), maxs(kGroups, -1);
        for_pieces(n, (n + kGroups - 1) / kGroups, n, [&](std::int64_t g, std::int64_t b, std::int64_t e) {
            std::int64_t a = 0, c = -1;
            for (std::int64_t i = b; i < e; ++i) {
                a = std::min(a, v[i]);
                c = std::max(c, v[i]);
            }
            mins[g] = a;
            maxs[g] = c;
        });
        lo = *std::min_element(mins.begin(), mins.end());
        top = *std::max_element(maxs.begin(), maxs.end());
    }
    if (lo < 0) throw nb::value_error("'x' must be >= 0");
    if (top == std::numeric_limits<std::int64_t>::max()) throw nb::value_error("array is too big");
    const std::int64_t shape[] = {std::max(top + 1, minlength)};
    if (weights_obj.is_none()) {
        NDArray out = new_array(DType::Int64, shape, /*zeroed=*/true);
        count_into(reinterpret_cast<std::int64_t*>(out.data), shape[0], n, [v](std::int64_t i) { return v[i]; });
        return wrap(std::move(out));
    }
    const NDArray& w = require_ndarray(weights_obj, "weights");
    if (w.dtype == DType::Complex128) throw nb::type_error("'weights' must be real, not complex128");
    if (w.ndim != 1 || w.shape[0] != n) throw nb::value_error("'weights' must be 1-D and as long as 'x'");
    const NDArray wf = c_order(w, DType::Float64);
    const double* p = reinterpret_cast<const double*>(wf.data);
    NDArray out = new_array(DType::Float64, shape, /*zeroed=*/true);
    double* o = reinterpret_cast<double*>(out.data);
    const std::int64_t bins = shape[0];
    if (n >= kParallelThreshold && bins * kGroups <= n) {
        // kGroups fixed ranges of [0, n), chosen by n and the length alone (not
        // by whether the pool is free): each sums its weights per bin from 0.0
        // in element order, and the ranges are added in order. The same bits
        // on any number of threads (docs/design/array.md §6.13).
        std::vector<double> local(static_cast<size_t>(kGroups * bins), 0.0);
        for_pieces(n, (n + kGroups - 1) / kGroups, n, [&](std::int64_t g, std::int64_t b0, std::int64_t e0) {
            double* c = local.data() + g * bins;
            for (std::int64_t i = b0; i < e0; ++i) c[v[i]] += p[i];
        });
        for (std::int64_t g = 0; g < kGroups; ++g)
            for (std::int64_t b = 0; b < bins; ++b) o[b] += local[g * bins + b];
    } else {
        const parallel::Unlocked unlocked(n >= kParallelThreshold);
        for (std::int64_t i = 0; i < n; ++i) o[v[i]] += p[i];  // in element order
    }
    return wrap(std::move(out));
}

// ----------------------------------------------------------- histogram

struct Edges {
    NDArray edges;  // float64 (uniform), or the given array
    bool uniform = false;
    double first = 0, last = 0;
    std::int64_t bins = 0;
};

std::pair<double, double> outer_edges(const double* a, std::int64_t n, nb::handle range) {
    double first, last;
    if (!range.is_none()) {
        if (!(PyTuple_Check(range.ptr()) || PyList_Check(range.ptr())) || nb::len(range) != 2)
            throw nb::type_error(("'range' must be None or a (min, max) pair, not " + py_type_name(range)).c_str());
        const std::optional<double> lo = real_value(range[0], "range"), hi = real_value(range[1], "range");
        for (int k = 0; k < 2; ++k)
            if (!(k ? hi : lo))
                throw nb::type_error(("'range' must hold two ints or floats, not " + py_type_name(range[k])).c_str());
        first = *lo;
        last = *hi;
        if (first > last) throw nb::value_error("'range' must have min <= max");
        if (!std::isfinite(first) || !std::isfinite(last)) throw nb::value_error("'range' must be finite");
    } else if (n == 0) {
        first = 0;
        last = 1;
    } else {
        // numpy's a.min() / a.max(); any NaN or inf makes the range not finite.
        // Pieces that do not run keep (a[0], a[0], no NaN): neutral for the min/max.
        std::vector<double> mins(kGroups, a[0]), maxs(kGroups, a[0]);
        std::vector<char> nan(kGroups, 0);
        for_pieces(n, (n + kGroups - 1) / kGroups, n, [&](std::int64_t g, std::int64_t b, std::int64_t e) {
            double lo_ = a[b], hi_ = a[b];
            bool any_nan = false;
            for (std::int64_t i = b; i < e; ++i) {
                any_nan |= a[i] != a[i];
                lo_ = std::min(lo_, a[i]);
                hi_ = std::max(hi_, a[i]);
            }
            mins[g] = lo_;
            maxs[g] = hi_;
            nan[g] = any_nan;
        });
        first = *std::min_element(mins.begin(), mins.end());
        last = *std::max_element(maxs.begin(), maxs.end());
        if (std::any_of(nan.begin(), nan.end(), [](char c) { return c != 0; }))
            first = std::numeric_limits<double>::quiet_NaN();
        if (!std::isfinite(first) || !std::isfinite(last))
            throw nb::value_error("the data's range is not finite (NaN or inf); pass 'range'");
    }
    if (first == last) {
        first -= 0.5;
        last += 0.5;
    }
    return {first, last};
}

Edges make_edges(const double* a, std::int64_t n, nb::handle bins, nb::handle range) {
    Edges e;
    std::int64_t nb_ = 0;
    if (index_value(bins, nb_) == IndexResult::Ok) {
        if (nb_ < 1) throw nb::value_error("'bins' must be positive, when an int");
        // bins + 1 edges; the largest int64 has no successor.
        if (nb_ == std::numeric_limits<std::int64_t>::max()) throw nb::value_error("array is too big");
        const auto [first, last] = outer_edges(a, n, range);
        e.uniform = true;
        e.first = first;
        e.last = last;
        e.bins = nb_;
        const nb::object lin =
            linspace(nb::float_(first), nb::float_(last), nb::int_(nb_ + 1), nb::none(), nb::bool_(true));
        e.edges = as_ndarray(lin);
        const double* p = reinterpret_cast<const double*>(e.edges.data);
        for (std::int64_t i = 0; i < nb_; ++i)
            if (!(p[i] < p[i + 1]))
                throw nb::value_error(("too many bins for the data range: cannot make " + std::to_string(nb_) +
                                       " finite-sized bins")
                                          .c_str());
        return e;
    }
    if (!is_ndarray(bins))
        throw nb::type_error(("'bins' must be an int or a 1-D array, not " + py_type_name(bins)).c_str());
    const NDArray& b = as_ndarray(bins);
    if (b.ndim != 1) throw nb::value_error("'bins' must be 1-D, when an array");
    if (b.dtype != DType::Int64 && b.dtype != DType::Float64)
        throw nb::type_error((std::string("'bins' must be int64 or float64, not ") + dtype_name(b.dtype)).c_str());
    e.edges = c_order(b, b.dtype);  // numpy returns the given edges (and dtype)
    const NDArray ef = c_order(b, DType::Float64);
    const double* p = reinterpret_cast<const double*>(ef.data);
    for (std::int64_t i = 0; i + 1 < b.shape[0]; ++i)
        if (p[i] > p[i + 1]) throw nb::value_error("'bins' must increase monotonically, when an array");
    e.bins = std::max<std::int64_t>(b.shape[0] - 1, 0);  // no edges or one edge: no bins (numpy's empty result)
    return e;
}

// The bin of every kept value of a uniform histogram, as numpy computes it;
// -1 for a value outside [first, last] (or NaN).
// Held by value where it is used: stores into a double buffer (the weighted
// block sums) would otherwise make the compiler reload these doubles on every
// element (1T weighted histogram, 1e6 values: 3.7 -> 2.2 ms).
struct UniformBins {
    double first, last, bins_d;
    std::int64_t bins;
    const double* edge;
    std::int64_t operator()(double v) const {
        if (!(v >= first && v <= last)) return -1;
        const double f = ((v - first) / (last - first)) * bins_d;
        auto i = static_cast<std::int64_t>(f);
        if (i == bins) --i;
        if (v < edge[i]) --i;
        if (v >= edge[i + 1] && i != bins - 1) ++i;
        return i;
    }
};

// The bin of v among arbitrary sorted edges, the last bin closed: numpy's
// searchsorted 'left' on the inner edges, 'right' on the last.
std::int64_t edges_bin(double v, const double* edge, std::int64_t bins) {
    if (std::isnan(v) || v < edge[0] || v > edge[bins]) return -1;
    if (v == edge[bins]) return bins - 1;  // the last bin is closed on the right
    return (std::upper_bound(edge, edge + bins + 1, v) - edge) - 1;
}

nb::object histogram(
    nb::handle x_obj, nb::handle bins_obj, nb::handle range_obj, nb::handle weights_obj, nb::handle density_obj
) {
    const NDArray& x = require_ndarray(x_obj);
    if (x.dtype == DType::Complex128) throw nb::type_error("not supported for complex128");
    const bool density = parse_bool(density_obj, "density");
    const NDArray xf = c_order(x, DType::Float64);  // x itself when C-ordered float64
    const double* a = reinterpret_cast<const double*>(xf.data);
    const std::int64_t n = xf.size();
    const Edges e = make_edges(a, n, bins_obj, range_obj);
    const NDArray ef = c_order(e.edges, DType::Float64);
    const double* edge = reinterpret_cast<const double*>(ef.data);
    const std::int64_t bins = e.bins;
    const UniformBins ub{e.first, e.last, static_cast<double>(e.bins), e.bins, edge};
    auto bin_of = [ub, edge, bins, uniform = e.uniform](double v) {
        return uniform ? ub(v) : edges_bin(v, edge, bins);
    };

    const std::int64_t hshape[] = {bins};
    NDArray hist;
    if (bins == 0) {  // an edges array of length 0 or 1: nothing to count (and edge[] may be empty)
        hist = new_array(
            weights_obj.is_none() || require_ndarray(weights_obj, "weights").dtype == DType::Bool
                ? DType::Int64
                : require_ndarray(weights_obj, "weights").dtype,
            hshape
        );
    } else if (weights_obj.is_none()) {
        hist = new_array(DType::Int64, hshape, /*zeroed=*/true);
        count_into(reinterpret_cast<std::int64_t*>(hist.data), bins, n, [&](std::int64_t i) { return bin_of(a[i]); });
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
                // Per block of kBlock values, the weights are summed per bin
                // from zero in element order, then the blocks are added to the
                // result in block order. Uniform bins follow numpy (np.bincount
                // per block: a double sum, cast back to the weights' dtype);
                // given edges sum in the weights' own dtype (int64 exactly).
                // The blocks depend on n only and are computed in parallel
                // when memory allows: the same bits on any number of threads.
                auto weighted = [&]<typename Acc>(std::type_identity<Acc>, auto bin) {
                    const std::int64_t nblk = (n + kBlock - 1) / kBlock;
                    auto block_sums = [&](std::int64_t blk, Acc* out_) {
                        std::fill(out_, out_ + bins, Acc(0));
                        const auto u = bin;  // locals: no reloads after each store
                        const double* av = a;
                        const T* wv = ws;
                        for (std::int64_t k = blk * kBlock, end = std::min(n, (blk + 1) * kBlock); k < end; ++k)
                            if (const std::int64_t i = u(av[k]); i >= 0) {
                                if constexpr (std::is_same_v<T, complex128>)
                                    out_[i] = {out_[i].real() + wv[k].real(), out_[i].imag() + wv[k].imag()};
                                else if constexpr (std::is_same_v<Acc, std::int64_t>)  // wraps, as int64 sums do
                                    out_[i] = static_cast<std::int64_t>(
                                        static_cast<std::uint64_t>(out_[i]) + static_cast<std::uint64_t>(wv[k])
                                    );
                                else out_[i] += static_cast<double>(wv[k]);
                            }
                    };
                    auto add_block = [&](const Acc* blk) {
                        for (std::int64_t i = 0; i < bins; ++i) {
                            if constexpr (std::is_same_v<T, complex128>)
                                h[i] = {h[i].real() + blk[i].real(), h[i].imag() + blk[i].imag()};
                            else if constexpr (std::is_same_v<T, std::int64_t>)
                                h[i] = static_cast<std::int64_t>(
                                    static_cast<std::uint64_t>(h[i]) +
                                    static_cast<std::uint64_t>(static_cast<T>(blk[i]))
                                );
                            else h[i] = h[i] + static_cast<T>(blk[i]);
                        }
                    };
                    if (nblk > 1 && nblk * bins <= 2 * n && parallel::available()) {
                        std::vector<Acc> all(static_cast<size_t>(nblk * bins));
                        parallel::parallel_for(nblk, [&](std::int64_t b) { block_sums(b, all.data() + b * bins); });
                        for (std::int64_t b = 0; b < nblk; ++b) add_block(all.data() + b * bins);
                    } else {
                        const parallel::Unlocked unlocked(nblk > 1);  // as the split above: no GIL from two blocks on
                        std::vector<Acc> blk(bins);
                        for (std::int64_t b = 0; b < nblk; ++b) {
                            block_sums(b, blk.data());
                            add_block(blk.data());
                        }
                    }
                };
                using Uniform = std::conditional_t<std::is_same_v<T, complex128>, complex128, double>;
                if (e.uniform) weighted(std::type_identity<Uniform>{}, ub);
                else weighted(std::type_identity<T>{}, [edge, bins](double v) { return edges_bin(v, edge, bins); });
            }
        });
    }

    // The arrays of a result object are read-only (API.md §2). The edges may be
    // the caller's own array (numpy returns them as given): the result holds a
    // read-only view of it, and the caller's array stays as it was.
    NDArray edges_view = e.edges;
    edges_view.writeable = false;
    nb::object edges_out = wrap(std::move(edges_view));
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
                    for (std::int64_t i = 0; i < bins; ++i)
                        o[i] = static_cast<double>(h[i]) / (edge[i + 1] - edge[i]) / t;
                }
            }
        });
        hist = std::move(out);
    }
    hist.writeable = false;
    return nb::module_::import_("stardust._results").attr("HistogramResult")(wrap(std::move(hist)), edges_out);
}

}  // namespace

void bind_histogram(nb::module_& m) {
    using nb::arg;
    m.def(
        "bincount", named("bincount", &bincount), nb::sig("def bincount(x, /, *, weights=None, minlength=0)"),
        arg().none(), nb::kw_only(), arg("weights").none() = nb::none(), arg("minlength").none() = 0
    );
    m.def(
        "histogram", named("histogram", &histogram),
        nb::sig("def histogram(x, /, *, bins=10, range=None, weights=None, density=False)"), arg().none(),
        nb::kw_only(), arg("bins").none() = 10, arg("range").none() = nb::none(), arg("weights").none() = nb::none(),
        arg("density").none() = false
    );
}

}  // namespace sd
