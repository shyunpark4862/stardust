// Indexing and assignment of sd.sparse, and the diagonal (docs/design/sparse.md §3.3, milestone SP3).
//
// The Python layer (src/stardust/sparse/_index.py) parses a key into one selection per axis and
// hands the kernels below normalized selections: None (every index of the axis), a tuple
// (start, step, count) (a slice), or a contiguous 1-D int64 sd.ndarray of indices already in
// [0, size) (repeats and any order allowed).
//
//   _normalize_index(idx, size, axis, /)
//       A new read-only copy of the int64 indices with the negative ones counted from the end;
//       IndexError for the first one out of [-size, size), in the array's words.
//   _select_compressed(indptr, indices, data, n_minor, major, minor, /)
//       The outer selection of canonical compressed arrays: line t of the result is line
//       major[t], holding at position p the value at minor[p]. Canonical again: within a line
//       the positions increase. (indptr, indices, data), new.
//   _select_coo(major, minor, data, row_sel, col_sel, /)
//       The same outer selection of COO entries as stored (repeats and order kept, as scipy's):
//       every entry gives one result entry per pair of positions selecting it, in entry order,
//       then row position, then column position. major None: a 1-D array (row_sel unused).
//       (rows or None, cols, data), new.
//   _pick_compressed(indptr, indices, data, major, minor, /)
//   _pick_coo(major, minor, data, pmajor, pminor, /)
//       Element picking (inner indexing): for every pair k the stored value at
//       (major[k], minor[k]). (positions k, values): the compressed form gives each found pair
//       once in k order; COO gives one result per stored entry at the pair (repeats kept), in
//       entry order, then k.
//   _compressed_value(indptr, indices, data, line, pos, /)
//       The Python scalar stored at (line, pos) of canonical compressed arrays, or a zero of the
//       dtype.
//   _coo_value(major, minor, data, i, j, /)
//       The value at (i, j) of COO entries: the repeats there summed left to right in stored
//       order (the canonical form's value); a 1-element array, or an empty one when nothing is
//       stored there. major None: 1-D (i unused).
//   _assign_compressed(indptr, indices, data, n_minor, major, minor, values, /, *, insert)
//       Sets the value at each (major[k], minor[k]); a position that repeats takes its last
//       value. A stored position is changed in place in `data` (when writeable), also when new
//       ones come with it; a new one is inserted (insert=True), which rebuilds the arrays:
//       (indptr, indices, data), new. None when every value went in place.
//   _coo_remove(major, minor, data, tmajor, tminor, /)
//       The COO entries at none of the target positions (repeats too), order kept:
//       (rows or None, cols, data), or None when nothing goes.
//   _diagonal_compressed(indptr, indices, data, rows, cols, k, /, *, by_rows)
//   _diagonal_coo(major, minor, data, rows, cols, k, /)
//       The k-th diagonal as a dense 1-D array: zeros plus the stored values (the dense array's
//       values, so a stored -0.0 is +0.0 and COO repeats add up in stored order).
//
// Every output element is written by one task and tasks are cut by sizes alone, so the results
// do not depend on the thread count.
#include "array/convert.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using std::int64_t;

nb::object readonly(NDArray&& a) {
    a.writeable = false;
    return wrap(std::move(a));
}

NDArray new_values(DType dtype, int64_t n) {
    const int64_t shape[] = {n};
    return new_array(dtype, shape);
}

inline void copy_value(std::byte* dst, int64_t i, const std::byte* src, int64_t k, int64_t size) {
    switch (size) {  // a constant size: the copy is one load and store
        case 8: std::memcpy(dst + i * 8, src + k * 8, 8); return;
        case 16: std::memcpy(dst + i * 16, src + k * 16, 16); return;
        case 1: dst[i] = src[k]; return;
        default: std::memcpy(dst + i * size, src + k * size, static_cast<std::size_t>(size));
    }
}

// ---- selections ----

// The selection of one axis: which source index each result position takes.
class AxisMap {
public:
    enum Kind { All, Range, Array };

    // lookup=false: only source() is used (no tables)
    AxisMap(nb::handle spec, int64_t n_axis, const char* func, bool lookup = true) : n_(n_axis) {
        if (spec.is_none()) {
            kind_ = All;
            count_ = n_axis;
            return;
        }
        if (PyTuple_Check(spec.ptr())) {
            kind_ = Range;
            start_ = nb::cast<int64_t>(spec[0]);
            step_ = nb::cast<int64_t>(spec[1]);
            count_ = nb::cast<int64_t>(spec[2]);
            if (step_ == 0) throw nb::value_error((std::string(func) + "(): step 0").c_str());
            if (lookup && step_ != 1 && step_ != -1 && n_ <= 16 * count_ + 65536) {
                // the position of each source index (-1: none): a load instead of a division
                rtable_.assign(static_cast<std::size_t>(n_), -1);
                for (int64_t t = 0; t < count_; ++t) rtable_[static_cast<std::size_t>(start_ + t * step_)] = t;
            }
            return;
        }
        kind_ = Array;
        const NDArray& a = vector_arg(spec, func, DType::Int64);
        sel_ = i64(a);
        count_ = a.shape[0];
        if (!lookup) return;
        const parallel::Unlocked unlocked(count_ >= kParallelThreshold);
        increasing_ = true;
        for (int64_t t = 1; t < count_; ++t)
            if (sel_[t] < sel_[t - 1]) {
                increasing_ = false;
                break;
            }
        pos_.resize(static_cast<std::size_t>(count_));
        if (n_ <= 16 * count_ + 65536) {  // a table of where each source index starts in pos_
            table_.assign(static_cast<std::size_t>(n_ + 1), 0);
            for (int64_t t = 0; t < count_; ++t) ++table_[static_cast<std::size_t>(sel_[t] + 1)];
            for (int64_t c = 0; c < n_; ++c)
                table_[static_cast<std::size_t>(c + 1)] += table_[static_cast<std::size_t>(c)];
            std::vector<int64_t> fill(table_.begin(), table_.end() - 1);
            for (int64_t t = 0; t < count_; ++t)
                pos_[static_cast<std::size_t>(fill[static_cast<std::size_t>(sel_[t])]++)] = t;
            // how many positions select each source index, saturated at 255: a byte table that
            // stays in cache answers the common "none" without the wide table (SP8)
            hits_.resize(static_cast<std::size_t>(n_));
            for (int64_t c = 0; c < n_; ++c) {
                const int64_t h = table_[static_cast<std::size_t>(c + 1)] - table_[static_cast<std::size_t>(c)];
                if (h >= 255) saturated_ = true;
                hits_[static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(std::min<int64_t>(255, h));
            }
        } else {  // the positions sorted by source index (stably), searched
            std::iota(pos_.begin(), pos_.end(), int64_t{0});
            if (!increasing_)
                std::stable_sort(pos_.begin(), pos_.end(), [&](int64_t a, int64_t b) { return sel_[a] < sel_[b]; });
            sorted_.resize(static_cast<std::size_t>(count_));
            for (int64_t t = 0; t < count_; ++t)
                sorted_[static_cast<std::size_t>(t)] = sel_[pos_[static_cast<std::size_t>(t)]];
        }
    }

    Kind kind() const { return kind_; }
    // An index array with the byte table and no source index picked 255 times or more: hits()[c]
    // positions pick source index c, they are pos()[table()[c] ...] (else null).
    const std::uint8_t* hits() const { return hits_.empty() || saturated_ ? nullptr : hits_.data(); }
    const int64_t* table() const { return table_.data(); }
    const int64_t* pos() const { return pos_.data(); }
    int64_t count() const { return count_; }
    int64_t start() const { return start_; }
    int64_t step() const { return step_; }
    bool unit() const { return kind_ == All || (kind_ == Range && step_ == 1); }
    // +1: the positions selecting a source index increase with it; -1: they decrease; 0: neither
    int order() const {
        if (kind_ == All) return 1;
        if (kind_ == Range) return step_ > 0 ? 1 : -1;
        return increasing_ ? 1 : 0;
    }

    // the source index of result position t
    int64_t source(int64_t t) const {
        if (kind_ == All) return t;
        if (kind_ == Range) return start_ + t * step_;
        return sel_[t];
    }

    int64_t count_of(int64_t c) const {
        switch (kind_) {
            case All: return 1;
            case Range: {
                int64_t t;
                return range_pos(c, t) ? 1 : 0;
            }
            default: {
                if (!hits_.empty() && c >= 0 && c < n_) {
                    const std::uint8_t h = hits_[static_cast<std::size_t>(c)];
                    if (h < 255) return h;
                }
                int64_t b, e;
                span(c, b, e);
                return e - b;
            }
        }
    }

    // f(p) for every result position p selecting source index c, in increasing p
    template <class F>
    void each(int64_t c, F&& f) const {
        switch (kind_) {
            case All: f(c); return;
            case Range: {
                int64_t t;
                if (range_pos(c, t)) f(t);
                return;
            }
            default: {
                if (!hits_.empty() && c >= 0 && c < n_ && hits_[static_cast<std::size_t>(c)] == 0) return;
                int64_t b, e;
                span(c, b, e);
                for (int64_t s = b; s < e; ++s) f(pos_[static_cast<std::size_t>(s)]);
            }
        }
    }

private:
    Kind kind_ = All;
    int64_t n_, start_ = 0, step_ = 1, count_ = 0;
    const int64_t* sel_ = nullptr;
    bool increasing_ = true;
    std::vector<int64_t> pos_, table_, sorted_, rtable_;
    std::vector<std::uint8_t> hits_;
    bool saturated_ = false;

    bool range_pos(int64_t c, int64_t& t) const {
        if (!rtable_.empty()) {
            t = rtable_[static_cast<std::size_t>(c)];
            return t >= 0;
        }
        const int64_t d = c - start_;
        if (step_ == 1 || step_ == -1) {
            t = step_ == 1 ? d : -d;
            return t >= 0 && t < count_;
        }
        if (step_ > 0) {
            if (d < 0 || d % step_ != 0) return false;
            t = d / step_;
        } else {
            if (d > 0 || (-d) % (-step_) != 0) return false;
            t = d / step_;
        }
        return t < count_;
    }

    void span(int64_t c, int64_t& b, int64_t& e) const {
        if (c < 0 || c >= n_) {
            b = e = 0;
            return;
        }
        if (!table_.empty()) {
            b = table_[static_cast<std::size_t>(c)];
            e = table_[static_cast<std::size_t>(c + 1)];
            return;
        }
        const auto r = std::equal_range(sorted_.begin(), sorted_.end(), c);
        b = r.first - sorted_.begin();
        e = r.second - sorted_.begin();
    }
};

int64_t lines_per_task(int64_t line_len) { return std::max<int64_t>(1, kChunk / std::max<int64_t>(line_len, 1)); }

// The first index of the sorted [b, e) not less than pos: a branch-free count on a short line (a
// search mispredicts more there), a binary search on a long one.
inline const int64_t* lower(const int64_t* b, const int64_t* e, int64_t pos) {
    if (e - b <= 32) {
        int64_t c = 0;
        for (const int64_t* q = b; q < e; ++q) c += *q < pos;
        return b + c;
    }
    return std::lower_bound(b, e, pos);
}

// ---- _normalize_index ----

nb::object normalize_index(nb::handle idx_h, int64_t size, int64_t axis) {
    const NDArray& idx = vector_arg(idx_h, "_normalize_index", DType::Int64);
    const int64_t n = idx.shape[0];
    const int64_t* in = i64(idx);
    NDArray out = new_index(n);
    int64_t* o = i64(out);
    int64_t bad = -1;
    {
        const parallel::Unlocked unlocked(n >= kParallelThreshold);
        for (int64_t k = 0; k < n; ++k) {
            const int64_t v = in[k];
            if (v < -size || v >= size) {
                bad = k;
                break;
            }
            o[k] = v < 0 ? v + size : v;
        }
    }
    if (bad >= 0)
        throw nb::index_error(("index " + std::to_string(in[bad]) + " is out of bounds for axis " +
                               std::to_string(axis) + " with size " + std::to_string(size))
                                  .c_str());
    return readonly(std::move(out));
}

// ---- _select_compressed ----

nb::tuple select_compressed(
    nb::handle indptr_h,
    nb::handle indices_h,
    nb::handle data_h,
    int64_t n_minor,
    nb::handle major_h,
    nb::handle minor_h
) {
    const NDArray& indptr = vector_arg(indptr_h, "_select_compressed", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_select_compressed", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_select_compressed");
    const int64_t n_major = indptr.shape[0] - 1, nnz = indices.shape[0];
    const AxisMap lines(major_h, n_major, "_select_compressed", false);
    const AxisMap minor(minor_h, n_minor, "_select_compressed");
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const int64_t n_out = lines.count(), size = data.itemsize();
    const int64_t avg = n_major > 0 ? nnz / n_major : 0;
    const int64_t per = lines_per_task(avg), work = n_out + n_out * avg;

    // the entries [b, e) of source line `line` that the minor selection can take
    auto bounds = [&](int64_t line, int64_t& b, int64_t& e) {
        b = ip[line];
        e = ip[line + 1];
        if (minor.kind() == AxisMap::Range && minor.step() == 1) {
            const int64_t lo = minor.start(), hi = minor.start() + minor.count();
            if (e - b <= 32) {  // a short line: a scan that stops early mispredicts least here
                while (b < e && ix[b] < lo) ++b;
                int64_t f = b;
                while (f < e && ix[f] < hi) ++f;
                e = f;
            } else {
                b = std::lower_bound(ix + b, ix + e, lo) - ix;
                e = std::lower_bound(ix + b, ix + e, hi) - ix;
            }
        }
    };

    NDArray out_ptr = new_indptr(n_out);
    int64_t* op = i64(out_ptr);
    op[0] = 0;
    const std::uint8_t* hits = minor.hits();
    const int64_t* table = minor.table();
    const int64_t* pos = minor.pos();
    for_chunks(
        n_out,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t t = lo; t < hi; ++t) {
                int64_t b, e;
                bounds(lines.source(t), b, e);
                if (minor.unit()) {
                    op[t + 1] = e - b;
                } else if (hits) {
                    int64_t c = 0;
                    for (int64_t k = b; k < e; ++k) c += hits[ix[k]];
                    op[t + 1] = c;
                } else {
                    int64_t c = 0;
                    for (int64_t k = b; k < e; ++k) c += minor.count_of(ix[k]);
                    op[t + 1] = c;
                }
            }
        },
        per, work
    );
    for (int64_t t = 0; t < n_out; ++t) op[t + 1] += op[t];
    const int64_t total = op[n_out];
    NDArray out_ix = new_index(total);
    NDArray out_v = new_values(data.dtype, total);
    int64_t* ox = i64(out_ix);
    std::byte* ov = out_v.data;
    const std::byte* dv = data.data;
    const int64_t shift = minor.kind() == AxisMap::Range ? minor.start() : 0;
    const int order = minor.order();
    for_chunks(
        n_out,
        [&](int64_t, int64_t lo, int64_t hi) {
            std::vector<std::pair<int64_t, int64_t>> tmp;
            for (int64_t t = lo; t < hi; ++t) {
                int64_t b, e, w = op[t];
                bounds(lines.source(t), b, e);
                if (minor.unit()) {
                    for (int64_t k = b; k < e; ++k) ox[w + k - b] = ix[k] - shift;
                    std::memcpy(ov + w * size, dv + b * size, static_cast<std::size_t>((e - b) * size));
                } else if (order > 0 && hits) {
                    for (int64_t k = b; k < e; ++k) {
                        const int64_t c = ix[k];
                        if (!hits[c]) continue;
                        for (int64_t q = table[c]; q < table[c + 1]; ++q) {
                            ox[w] = pos[q];
                            copy_value(ov, w, dv, k, size);
                            ++w;
                        }
                    }
                } else if (order > 0) {
                    for (int64_t k = b; k < e; ++k)
                        minor.each(ix[k], [&](int64_t p) {
                            ox[w] = p;
                            copy_value(ov, w, dv, k, size);
                            ++w;
                        });
                } else if (order < 0) {
                    for (int64_t k = e - 1; k >= b; --k)
                        minor.each(ix[k], [&](int64_t p) {
                            ox[w] = p;
                            copy_value(ov, w, dv, k, size);
                            ++w;
                        });
                } else {
                    tmp.clear();
                    if (hits) {
                        for (int64_t k = b; k < e; ++k) {
                            const int64_t c = ix[k];
                            if (!hits[c]) continue;
                            for (int64_t q = table[c]; q < table[c + 1]; ++q) tmp.emplace_back(pos[q], k);
                        }
                    } else {
                        for (int64_t k = b; k < e; ++k) minor.each(ix[k], [&](int64_t p) { tmp.emplace_back(p, k); });
                    }
                    if (tmp.size() <= 16) {  // the positions of a line are distinct
                        for (std::size_t i = 1; i < tmp.size(); ++i)
                            for (std::size_t j = i; j > 0 && tmp[j - 1].first > tmp[j].first; --j)
                                std::swap(tmp[j - 1], tmp[j]);
                    } else {
                        std::sort(tmp.begin(), tmp.end());
                    }
                    for (const auto& [p, k] : tmp) {
                        ox[w] = p;
                        copy_value(ov, w, dv, k, size);
                        ++w;
                    }
                }
            }
        },
        per, work
    );
    return nb::make_tuple(readonly(std::move(out_ptr)), readonly(std::move(out_ix)), wrap(std::move(out_v)));
}

const int64_t* opt_index(nb::handle o, const char* func, int64_t n) {
    if (o.is_none()) return nullptr;
    const NDArray& a = vector_arg(o, func, DType::Int64);
    if (a.shape[0] != n) throw nb::value_error((std::string(func) + "(): lengths differ").c_str());
    return i64(a);
}

// ---- _select_coo ----

nb::tuple select_coo(
    nb::handle major_h, nb::handle minor_h, nb::handle data_h, nb::handle shape_h, nb::handle row_h, nb::handle col_h
) {
    const NDArray& minor = vector_arg(minor_h, "_select_coo", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_select_coo");
    const int64_t nnz = minor.shape[0], size = data.itemsize();
    const int64_t* ma = opt_index(major_h, "_select_coo", nnz);
    const int64_t* mi = i64(minor);
    std::vector<int64_t> shape;
    for (nb::handle v : shape_h) shape.push_back(nb::cast<int64_t>(v));
    const bool two = ma != nullptr;
    const AxisMap rows(two ? row_h : nb::none(), two ? shape[0] : 1, "_select_coo");
    const AxisMap cols(col_h, shape.back(), "_select_coo");
    auto count = [&](int64_t k) {
        const int64_t c = cols.count_of(mi[k]);
        return two && c ? c * rows.count_of(ma[k]) : c;
    };
    const int64_t nt = num_chunks(nnz);
    std::vector<int64_t> start(static_cast<std::size_t>(nt + 1), 0);
    for_chunks(nnz, [&](int64_t t, int64_t lo, int64_t hi) {
        int64_t c = 0;
        for (int64_t k = lo; k < hi; ++k) c += count(k);
        start[static_cast<std::size_t>(t + 1)] = c;
    });
    for (int64_t t = 0; t < nt; ++t) start[static_cast<std::size_t>(t + 1)] += start[static_cast<std::size_t>(t)];
    const int64_t total = start[static_cast<std::size_t>(nt)];
    NDArray out_c = new_index(total);
    NDArray out_r = two ? new_index(total) : NDArray();
    NDArray out_v = new_values(data.dtype, total);
    int64_t* oc = i64(out_c);
    int64_t* orow = two ? i64(out_r) : nullptr;
    for_chunks(nnz, [&](int64_t t, int64_t lo, int64_t hi) {
        int64_t w = start[static_cast<std::size_t>(t)];
        for (int64_t k = lo; k < hi; ++k) {
            if (!cols.count_of(mi[k])) continue;
            if (two) {
                rows.each(ma[k], [&](int64_t p) {
                    cols.each(mi[k], [&](int64_t q) {
                        orow[w] = p;
                        oc[w] = q;
                        copy_value(out_v.data, w, data.data, k, size);
                        ++w;
                    });
                });
            } else {
                cols.each(mi[k], [&](int64_t q) {
                    oc[w] = q;
                    copy_value(out_v.data, w, data.data, k, size);
                    ++w;
                });
            }
        }
    });
    nb::object r = two ? readonly(std::move(out_r)) : nb::none();
    return nb::make_tuple(r, readonly(std::move(out_c)), wrap(std::move(out_v)));
}

// ---- picking ----

nb::tuple pick_compressed(
    nb::handle indptr_h, nb::handle indices_h, nb::handle data_h, nb::handle pm_h, nb::handle pn_h
) {
    const NDArray& indptr = vector_arg(indptr_h, "_pick_compressed", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_pick_compressed", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_pick_compressed");
    const NDArray& pmn = vector_arg(pn_h, "_pick_compressed", DType::Int64);
    const int64_t K = pmn.shape[0], size = data.itemsize();
    const int64_t* pm = opt_index(pm_h, "_pick_compressed", K);
    const int64_t* pn = i64(pmn);
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    std::vector<int64_t> off(static_cast<std::size_t>(K));
    const int64_t nt = num_chunks(K);
    std::vector<int64_t> start(static_cast<std::size_t>(nt + 1), 0);
    for_chunks(K, [&](int64_t t, int64_t lo, int64_t hi) {
        int64_t c = 0;
        for (int64_t k = lo; k < hi; ++k) {
            const int64_t line = pm ? pm[k] : 0;
            const int64_t* b = ix + ip[line];
            const int64_t* e = ix + ip[line + 1];
            const int64_t* f = lower(b, e, pn[k]);
            off[static_cast<std::size_t>(k)] = (f != e && *f == pn[k]) ? f - ix : -1;
            c += off[static_cast<std::size_t>(k)] >= 0;
        }
        start[static_cast<std::size_t>(t + 1)] = c;
    });
    for (int64_t t = 0; t < nt; ++t) start[static_cast<std::size_t>(t + 1)] += start[static_cast<std::size_t>(t)];
    const int64_t total = start[static_cast<std::size_t>(nt)];
    NDArray pos = new_index(total);
    NDArray vals = new_values(data.dtype, total);
    int64_t* po = i64(pos);
    for_chunks(K, [&](int64_t t, int64_t lo, int64_t hi) {
        int64_t w = start[static_cast<std::size_t>(t)];
        for (int64_t k = lo; k < hi; ++k)
            if (off[static_cast<std::size_t>(k)] >= 0) {
                po[w] = k;
                copy_value(vals.data, w, data.data, off[static_cast<std::size_t>(k)], size);
                ++w;
            }
    });
    return nb::make_tuple(readonly(std::move(pos)), wrap(std::move(vals)));
}

// Target pairs sorted by (major, minor), stably: order[s] is the input position of the s-th.
struct SortedPairs {
    std::vector<int64_t> order;
    const int64_t *ma = nullptr, *mi = nullptr;

    SortedPairs(const int64_t* major, const int64_t* minor, int64_t n) : ma(major), mi(minor) {
        order.resize(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), int64_t{0});
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return less(a, b); });
    }
    bool less(int64_t a, int64_t b) const {
        const int64_t ra = ma ? ma[a] : 0, rb = ma ? ma[b] : 0;
        return ra < rb || (ra == rb && mi[a] < mi[b]);
    }
    // [b, e) of `order` whose pairs equal (r, c)
    void span(int64_t r, int64_t c, int64_t& b, int64_t& e) const {
        auto key_less = [&](int64_t s, std::pair<int64_t, int64_t> key) {
            const int64_t rs = ma ? ma[s] : 0;
            return rs < key.first || (rs == key.first && mi[s] < key.second);
        };
        auto less_key = [&](std::pair<int64_t, int64_t> key, int64_t s) {
            const int64_t rs = ma ? ma[s] : 0;
            return key.first < rs || (key.first == rs && key.second < mi[s]);
        };
        const auto key = std::make_pair(r, c);
        b = std::lower_bound(order.begin(), order.end(), key, key_less) - order.begin();
        e = std::upper_bound(order.begin() + b, order.end(), key, less_key) - order.begin();
    }
};

nb::tuple pick_coo(nb::handle major_h, nb::handle minor_h, nb::handle data_h, nb::handle pm_h, nb::handle pn_h) {
    const NDArray& minor = vector_arg(minor_h, "_pick_coo", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_pick_coo");
    const NDArray& pmn = vector_arg(pn_h, "_pick_coo", DType::Int64);
    const int64_t nnz = minor.shape[0], K = pmn.shape[0], size = data.itemsize();
    const int64_t* ma = opt_index(major_h, "_pick_coo", nnz);
    const int64_t* mi = i64(minor);
    const int64_t* pm = opt_index(pm_h, "_pick_coo", K);
    const int64_t* pn = i64(pmn);
    std::optional<SortedPairs> sorted;
    {
        const parallel::Unlocked unlocked(K >= kParallelThreshold);
        sorted.emplace(pm, pn, K);
    }
    const SortedPairs& sp = *sorted;
    const int64_t nt = num_chunks(nnz);
    std::vector<int64_t> start(static_cast<std::size_t>(nt + 1), 0);
    for_chunks(
        nnz,
        [&](int64_t t, int64_t lo, int64_t hi) {
            int64_t c = 0;
            for (int64_t k = lo; k < hi; ++k) {
                int64_t b, e;
                sp.span(ma ? ma[k] : 0, mi[k], b, e);
                c += e - b;
            }
            start[static_cast<std::size_t>(t + 1)] = c;
        },
        kChunk, nnz + K
    );
    for (int64_t t = 0; t < nt; ++t) start[static_cast<std::size_t>(t + 1)] += start[static_cast<std::size_t>(t)];
    const int64_t total = start[static_cast<std::size_t>(nt)];
    NDArray pos = new_index(total);
    NDArray vals = new_values(data.dtype, total);
    int64_t* po = i64(pos);
    for_chunks(
        nnz,
        [&](int64_t t, int64_t lo, int64_t hi) {
            int64_t w = start[static_cast<std::size_t>(t)];
            for (int64_t k = lo; k < hi; ++k) {
                int64_t b, e;
                sp.span(ma ? ma[k] : 0, mi[k], b, e);
                for (int64_t s = b; s < e; ++s) {
                    po[w] = sp.order[static_cast<std::size_t>(s)];
                    copy_value(vals.data, w, data.data, k, size);
                    ++w;
                }
            }
        },
        kChunk, nnz + K
    );
    return nb::make_tuple(readonly(std::move(pos)), wrap(std::move(vals)));
}

nb::object compressed_value(nb::handle indptr_h, nb::handle indices_h, nb::handle data_h, int64_t line, int64_t pos) {
    const NDArray& indptr = vector_arg(indptr_h, "_compressed_value", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_compressed_value", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_compressed_value");
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const int64_t* b = ix + ip[line];
    const int64_t* e = ix + ip[line + 1];
    const int64_t* f = lower(b, e, pos);
    if (f != e && *f == pos) return load_python(data.dtype, data.data + (f - ix) * data.itemsize());
    return visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) { return to_python(T(0)); });
}

nb::object coo_value(nb::handle major_h, nb::handle minor_h, nb::handle data_h, int64_t i, int64_t j) {
    const NDArray& minor = vector_arg(minor_h, "_coo_value", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_coo_value");
    const int64_t nnz = minor.shape[0];
    const int64_t* ma = opt_index(major_h, "_coo_value", nnz);
    const int64_t* mi = i64(minor);
    return visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        bool found = false;
        T acc{};
        {
            const parallel::Unlocked unlocked(nnz >= kParallelThreshold);
            for (int64_t k = 0; k < nnz; ++k)
                if (mi[k] == j && (!ma || ma[k] == i)) {
                    acc = found ? add_value(acc, v[k]) : v[k];
                    found = true;
                }
        }
        NDArray out = new_values(data.dtype, found ? 1 : 0);
        if (found) typed<T>(out)[0] = acc;
        return wrap(std::move(out));
    });
}

// ---- assignment ----

nb::object assign_compressed(
    nb::handle indptr_h,
    nb::handle indices_h,
    nb::handle data_h,
    int64_t n_minor,
    nb::handle tm_h,
    nb::handle tn_h,
    nb::handle values_h,
    bool insert
) {
    const NDArray& indptr = vector_arg(indptr_h, "_assign_compressed", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_assign_compressed", DType::Int64);
    NDArray& data = const_cast<NDArray&>(vector_arg(data_h, "_assign_compressed"));
    const NDArray& tn_a = vector_arg(tn_h, "_assign_compressed", DType::Int64);
    const NDArray& values = vector_arg(values_h, "_assign_compressed", data.dtype);
    const int64_t K = tn_a.shape[0], size = data.itemsize(), n_major = indptr.shape[0] - 1;
    if (values.shape[0] != K) throw nb::value_error("_assign_compressed(): lengths differ");
    (void)n_minor;
    const int64_t* tm = opt_index(tm_h, "_assign_compressed", K);
    const int64_t* tn = i64(tn_a);
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);

    // the distinct targets in (line, position) order, each with its last value
    std::vector<int64_t> uniq;  // input positions
    std::vector<int64_t> off;   // offset in the stored entries, or -1
    int64_t n_new = 0;
    {
        const parallel::Unlocked unlocked(K >= kParallelThreshold);
        const SortedPairs sp(tm, tn, K);
        for (int64_t s = 0; s < K; ++s) {
            const int64_t k = sp.order[static_cast<std::size_t>(s)];
            if (s + 1 < K) {
                const int64_t nk = sp.order[static_cast<std::size_t>(s + 1)];
                if ((tm ? tm[nk] : 0) == (tm ? tm[k] : 0) && tn[nk] == tn[k]) continue;  // a later one wins
            }
            uniq.push_back(k);
        }
        off.resize(uniq.size());
        for (std::size_t u = 0; u < uniq.size(); ++u) {
            const int64_t k = uniq[u], line = tm ? tm[k] : 0;
            const int64_t* b = ix + ip[line];
            const int64_t* e = ix + ip[line + 1];
            const int64_t* f = lower(b, e, tn[k]);
            off[u] = (f != e && *f == tn[k]) ? f - ix : -1;
            n_new += off[u] < 0;
        }
    }
    if (n_new == 0 || !insert) {
        NDArray* target = &data;
        NDArray fresh;
        if (!data.writeable) {  // a read-only values array (given with copy=None): a new one
            fresh = new_values(data.dtype, data.shape[0]);
            std::memcpy(fresh.data, data.data, static_cast<std::size_t>(data.shape[0] * size));
            target = &fresh;
        }
        for (std::size_t u = 0; u < uniq.size(); ++u)
            if (off[u] >= 0) copy_value(target->data, off[u], values.data, uniq[u], size);
        if (target == &data) return nb::none();
        nb::object ptr = nb::borrow(indptr_h), idx = nb::borrow(indices_h);
        return nb::make_tuple(ptr, idx, wrap(std::move(fresh)));
    }
    // The stored positions first change in place, as without new ones, so that the arrays sharing
    // the values (a .T, to_scipy() views) see them (scipy's _set_many); then the new arrays.
    if (data.writeable)
        for (std::size_t u = 0; u < uniq.size(); ++u)
            if (off[u] >= 0) copy_value(data.data, off[u], values.data, uniq[u], size);
    // per line, the new positions to merge in
    std::vector<int64_t> tptr(static_cast<std::size_t>(n_major + 1), 0);
    for (std::size_t u = 0; u < uniq.size(); ++u) ++tptr[static_cast<std::size_t>((tm ? tm[uniq[u]] : 0) + 1)];
    for (int64_t r = 0; r < n_major; ++r) tptr[static_cast<std::size_t>(r + 1)] += tptr[static_cast<std::size_t>(r)];
    const int64_t nnz = indices.shape[0], total = nnz + n_new;
    NDArray out_ptr = new_indptr(n_major);
    int64_t* op = i64(out_ptr);
    op[0] = 0;
    for (int64_t r = 0; r < n_major; ++r) {
        int64_t c = 0;
        for (int64_t u = tptr[static_cast<std::size_t>(r)]; u < tptr[static_cast<std::size_t>(r + 1)]; ++u)
            c += off[static_cast<std::size_t>(u)] < 0;
        op[r + 1] = op[r] + (ip[r + 1] - ip[r]) + c;
    }
    NDArray out_ix = new_index(total);
    NDArray out_v = new_values(data.dtype, total);
    int64_t* ox = i64(out_ix);
    const int64_t per = lines_per_task(n_major > 0 ? total / n_major : 0);
    for_chunks(
        n_major,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; ++r) {
                int64_t w = op[r], k = ip[r];
                const int64_t e = ip[r + 1];
                int64_t u = tptr[static_cast<std::size_t>(r)];
                const int64_t ue = tptr[static_cast<std::size_t>(r + 1)];
                while (k < e || u < ue) {
                    const int64_t tk = u < ue ? uniq[static_cast<std::size_t>(u)] : -1;
                    if (u >= ue || (k < e && ix[k] < tn[tk])) {
                        ox[w] = ix[k];
                        copy_value(out_v.data, w, data.data, k, size);
                        ++k;
                    } else if (k < e && ix[k] == tn[tk]) {
                        ox[w] = ix[k];
                        copy_value(out_v.data, w, values.data, tk, size);
                        ++k;
                        ++u;
                    } else {
                        ox[w] = tn[tk];
                        copy_value(out_v.data, w, values.data, tk, size);
                        ++u;
                    }
                    ++w;
                }
            }
        },
        per, total + n_major
    );
    return nb::make_tuple(readonly(std::move(out_ptr)), readonly(std::move(out_ix)), wrap(std::move(out_v)));
}

nb::object coo_remove(nb::handle major_h, nb::handle minor_h, nb::handle data_h, nb::handle tm_h, nb::handle tn_h) {
    const NDArray& minor = vector_arg(minor_h, "_coo_remove", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_coo_remove");
    const NDArray& tn_a = vector_arg(tn_h, "_coo_remove", DType::Int64);
    const int64_t nnz = minor.shape[0], K = tn_a.shape[0], size = data.itemsize();
    const int64_t* ma = opt_index(major_h, "_coo_remove", nnz);
    const int64_t* mi = i64(minor);
    const int64_t* tm = opt_index(tm_h, "_coo_remove", K);
    std::optional<SortedPairs> sorted;
    {
        const parallel::Unlocked unlocked(K >= kParallelThreshold);
        sorted.emplace(tm, i64(tn_a), K);
    }
    const SortedPairs& sp = *sorted;
    std::vector<unsigned char> keep(static_cast<std::size_t>(nnz));
    const int64_t nt = num_chunks(nnz);
    std::vector<int64_t> start(static_cast<std::size_t>(nt + 1), 0);
    for_chunks(
        nnz,
        [&](int64_t t, int64_t lo, int64_t hi) {
            int64_t c = 0;
            for (int64_t k = lo; k < hi; ++k) {
                int64_t b, e;
                sp.span(ma ? ma[k] : 0, mi[k], b, e);
                keep[static_cast<std::size_t>(k)] = b == e;
                c += b == e;
            }
            start[static_cast<std::size_t>(t + 1)] = c;
        },
        kChunk, nnz + K
    );
    for (int64_t t = 0; t < nt; ++t) start[static_cast<std::size_t>(t + 1)] += start[static_cast<std::size_t>(t)];
    const int64_t total = start[static_cast<std::size_t>(nt)];
    if (total == nnz) return nb::none();
    NDArray out_c = new_index(total);
    NDArray out_r = ma ? new_index(total) : NDArray();
    NDArray out_v = new_values(data.dtype, total);
    int64_t* oc = i64(out_c);
    int64_t* orow = ma ? i64(out_r) : nullptr;
    for_chunks(nnz, [&](int64_t t, int64_t lo, int64_t hi) {
        int64_t w = start[static_cast<std::size_t>(t)];
        for (int64_t k = lo; k < hi; ++k)
            if (keep[static_cast<std::size_t>(k)]) {
                if (orow) orow[w] = ma[k];
                oc[w] = mi[k];
                copy_value(out_v.data, w, data.data, k, size);
                ++w;
            }
    });
    nb::object r = ma ? readonly(std::move(out_r)) : nb::none();
    return nb::make_tuple(r, readonly(std::move(out_c)), wrap(std::move(out_v)));
}

// ---- the diagonal ----

int64_t diagonal_length(int64_t rows, int64_t cols, int64_t k) {
    const int64_t n = k >= 0 ? std::min(rows, cols - k) : std::min(rows + k, cols);
    return std::max<int64_t>(n, 0);
}

nb::object diagonal_compressed(
    nb::handle indptr_h, nb::handle indices_h, nb::handle data_h, int64_t rows, int64_t cols, int64_t k, bool by_rows
) {
    const NDArray& indptr = vector_arg(indptr_h, "_diagonal_compressed", DType::Int64);
    const NDArray& indices = vector_arg(indices_h, "_diagonal_compressed", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_diagonal_compressed");
    const int64_t n = diagonal_length(rows, cols, k), r0 = std::max<int64_t>(-k, 0), c0 = std::max<int64_t>(k, 0);
    const int64_t* ip = i64(indptr);
    const int64_t* ix = i64(indices);
    const int64_t shape[] = {n};
    NDArray out = new_array(data.dtype, shape, true);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        T* o = typed<T>(out);
        for_chunks(n, [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t t = lo; t < hi; ++t) {
                const int64_t line = by_rows ? r0 + t : c0 + t, pos = by_rows ? c0 + t : r0 + t;
                const int64_t* b = ix + ip[line];
                const int64_t* e = ix + ip[line + 1];
                const int64_t* f = lower(b, e, pos);
                if (f != e && *f == pos) o[t] = dense_value(v[f - ix]);
            }
        });
    });
    return wrap(std::move(out));
}

nb::object diagonal_coo(
    nb::handle major_h, nb::handle minor_h, nb::handle data_h, int64_t rows, int64_t cols, int64_t k
) {
    const NDArray& minor = vector_arg(minor_h, "_diagonal_coo", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_diagonal_coo");
    const int64_t nnz = minor.shape[0];
    const int64_t* ma = opt_index(major_h, "_diagonal_coo", nnz);
    if (!ma) throw nb::value_error("_diagonal_coo(): 2-D only");
    const int64_t* mi = i64(minor);
    const int64_t n = diagonal_length(rows, cols, k), r0 = std::max<int64_t>(-k, 0);
    const int64_t shape[] = {n};
    NDArray out = new_array(data.dtype, shape, true);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const T* v = typed<T>(data);
        T* o = typed<T>(out);
        const parallel::Unlocked unlocked(nnz >= kParallelThreshold);
        for (int64_t e = 0; e < nnz; ++e)  // in stored order: repeats add up as in toarray
            if (mi[e] - ma[e] == k) o[ma[e] - r0] = add_value(o[ma[e] - r0], v[e]);
    });
    return wrap(std::move(out));
}

}  // namespace

void bind_index(nb::module_& m) {
    m.def("_normalize_index", &normalize_index);
    m.def(
        "_select_compressed", &select_compressed, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"),
        nb::arg("n_minor"), nb::arg("major").none(), nb::arg("minor").none()
    );
    m.def(
        "_select_coo", &select_coo, nb::arg("major").none(), nb::arg("minor"), nb::arg("data"), nb::arg("shape"),
        nb::arg("row_sel").none(), nb::arg("col_sel").none()
    );
    m.def(
        "_pick_compressed", &pick_compressed, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"),
        nb::arg("major").none(), nb::arg("minor")
    );
    m.def(
        "_pick_coo", &pick_coo, nb::arg("major").none(), nb::arg("minor"), nb::arg("data"), nb::arg("pmajor").none(),
        nb::arg("pminor")
    );
    m.def("_compressed_value", &compressed_value);
    m.def(
        "_coo_value", &coo_value, nb::arg("major").none(), nb::arg("minor"), nb::arg("data"), nb::arg("i"), nb::arg("j")
    );
    m.def(
        "_assign_compressed", &assign_compressed, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"),
        nb::arg("n_minor"), nb::arg("major").none(), nb::arg("minor"), nb::arg("values"), nb::kw_only(),
        nb::arg("insert")
    );
    m.def(
        "_coo_remove", &coo_remove, nb::arg("major").none(), nb::arg("minor"), nb::arg("data"),
        nb::arg("tmajor").none(), nb::arg("tminor")
    );
    m.def(
        "_diagonal_compressed", &diagonal_compressed, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"),
        nb::arg("rows"), nb::arg("cols"), nb::arg("k"), nb::kw_only(), nb::arg("by_rows")
    );
    m.def(
        "_diagonal_coo", &diagonal_coo, nb::arg("major").none(), nb::arg("minor"), nb::arg("data"), nb::arg("rows"),
        nb::arg("cols"), nb::arg("k")
    );
}

}  // namespace sd::sparse
