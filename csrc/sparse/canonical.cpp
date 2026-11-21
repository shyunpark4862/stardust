// The canonical compressed form of sd.sparse (docs/design/sparse.md §2.3, §3.1): building CSR
// and CSC from coordinates, the transpose that turns one into the other, and the checks of the
// arrays a user hands in.
//
//   _compress(n_major, major, minor, data, /, *, share)
//       The canonical compressed arrays (indptr, indices, data) of COO entries: major None for
//       a 1-D array. Already canonical entries (major non-decreasing, minor strictly increasing
//       within a major index) give indptr, the very `minor` object (an index array the caller
//       owns, read-only) and `data` itself (share=True) or a copy; otherwise the entries are
//       sorted by (major, minor) and duplicates summed left to right in input order, into new
//       arrays.
//   _transpose(n_major, n_minor, indptr, indices, data, /)
//       The canonical compressed arrays of the transpose (CSR -> CSC and back): new arrays.
//   _compressed_in(n_major, n_minor, indptr, indices, data, /, *, minor_axis, share, func)
//       A user's (data, indices, indptr): indptr and indices are copied and checked (ValueError,
//       messages prefixed "func(): "); a canonical input keeps `data` (share=True) or a copy,
//       any other is sorted and summed like _compress.
//   _index_max(x, /, *, axis, func) / _index_copy(x, bound, /, *, axis, func)
//       The largest value of an index array (-1 when empty; a negative value is a ValueError),
//       and a checked copy of it (every value in [0, bound)), read-only.
//   _is_canonical(major, minor, /)
//       Whether COO entries are canonical (row-major order, no duplicates).
//   _sum_duplicates(n_major, major, minor, data, /)
//       COO entries in canonical order, duplicates summed: (major or None, minor, data), new.
//
// Every result is the same for any thread count: the direct and the bucket path compute the same
// function (the output is ordered by (major, minor), every sort is stable, and a sum of
// duplicates runs in input order on both), and the bucket path cuts its work by sizes alone.
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <arm_neon.h>

#include <nanobind/stl/string.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

const NDArray& vector_arg(nb::handle o, const char* func, DType dtype) {
    if (!is_ndarray(o)) throw nb::type_error((std::string(func) + "(): expected an sd.ndarray").c_str());
    const NDArray& a = as_ndarray(o);
    if (a.ndim != 1 || !a.c_contiguous || (dtype != DType::String && a.dtype != dtype))
        throw nb::type_error((std::string(func) + "(): wrong array layout").c_str());
    return a;
}

NDArray new_index(std::int64_t n, bool zeroed) {
    const std::int64_t shape[] = {n};
    return new_array(DType::Int64, shape, zeroed);
}

NDArray new_indptr(std::int64_t lines, bool zeroed) {
    if (lines == std::numeric_limits<std::int64_t>::max()) throw nb::value_error("array is too big");
    return new_index(lines + 1, zeroed);
}

namespace {

using std::int64_t;

// An entry of one line: its minor index and value. Entries arrive in input order, and every sort
// below is stable, so equal minors stay in input order and their sum runs left to right.
template <class T>
struct Pair {
    int64_t minor;
    T v;
};

template <class T>
struct Entry {
    int64_t major;
    int64_t minor;
    T v;
};

// One line: p[0, n) are its entries in input order. Unless `distinct` (a transpose: the minors
// already increase strictly), orders them by minor (stably) and sums equal minors left to right.
// Writes the distinct entries to out_idx / out_val; returns how many.
template <class T>
int64_t emit_line(Pair<T>* p, int64_t n, bool distinct, int64_t* out_idx, T* out_val) {
    if (distinct) {
        for (int64_t i = 0; i < n; ++i) {
            out_idx[i] = p[i].minor;
            out_val[i] = p[i].v;
        }
        return n;
    }
    if (n <= 16) {
        // Rank sort of keys minor * 16 + position (distinct, so the order is stable): no
        // data-dependent branch, about half the time of the insertion sort on random lines.
        uint64_t any = 0;
        for (int64_t i = 0; i < n; ++i) any |= static_cast<uint64_t>(p[i].minor);
        if ((any >> 59) == 0) {
            alignas(16) uint64_t key[16], sorted[16];
            for (int64_t i = 0; i < 16; ++i)
                key[i] = i < n ? (static_cast<uint64_t>(p[i].minor) << 4) | static_cast<uint64_t>(i) : ~uint64_t(0);
            for (int64_t i = 0; i < n; ++i) {
                const uint64x2_t ki = vdupq_n_u64(key[i]);
                uint64x2_t a0 = vcltq_u64(vld1q_u64(key), ki), a1 = vcltq_u64(vld1q_u64(key + 2), ki);
                uint64x2_t a2 = vcltq_u64(vld1q_u64(key + 4), ki), a3 = vcltq_u64(vld1q_u64(key + 6), ki);
                if (n > 8) {
                    a0 = vaddq_u64(a0, vcltq_u64(vld1q_u64(key + 8), ki));
                    a1 = vaddq_u64(a1, vcltq_u64(vld1q_u64(key + 10), ki));
                    a2 = vaddq_u64(a2, vcltq_u64(vld1q_u64(key + 12), ki));
                    a3 = vaddq_u64(a3, vcltq_u64(vld1q_u64(key + 14), ki));
                }
                const uint64x2_t a = vaddq_u64(vaddq_u64(a0, a1), vaddq_u64(a2, a3));
                // each true lane is all ones (-1): the rank is minus the lane sum
                sorted[-static_cast<int64_t>(vgetq_lane_u64(a, 0) + vgetq_lane_u64(a, 1))] = key[i];
            }
            int64_t w = 0;
            for (int64_t i = 0; i < n;) {
                const int64_t m = static_cast<int64_t>(sorted[i] >> 4);
                T acc = p[sorted[i] & 15].v;
                int64_t j = i + 1;
                for (; j < n && static_cast<int64_t>(sorted[j] >> 4) == m; ++j)
                    acc = add_value(acc, p[sorted[j] & 15].v);
                out_idx[w] = m;
                out_val[w] = acc;
                ++w;
                i = j;
            }
            return w;
        }
    }
    if (n <= 32) {
        for (int64_t i = 1; i < n; ++i) {
            const Pair<T> x = p[i];
            int64_t j = i;
            for (; j > 0 && p[j - 1].minor > x.minor; --j) p[j] = p[j - 1];
            p[j] = x;
        }
    } else {
        std::stable_sort(p, p + n, [](const Pair<T>& a, const Pair<T>& b) { return a.minor < b.minor; });
    }
    int64_t w = 0;
    for (int64_t i = 0; i < n;) {
        const int64_t m = p[i].minor;
        T acc = p[i].v;
        int64_t j = i + 1;
        for (; j < n && p[j].minor == m; ++j) acc = add_value(acc, p[j].v);
        out_idx[w] = m;
        out_val[w] = acc;
        ++w;
        i = j;
    }
    return w;
}

// The direct path: a counting sort by major index (indptr is the counter), then each line
// sorted and summed. The distinct entries end up at [0, return value) of out.
template <class T>
int64_t compress_direct(
    int64_t n_major,
    const int64_t* major,
    const int64_t* minor,
    const T* val,
    int64_t nnz,
    bool presorted,
    int64_t* indptr,
    int64_t* out_idx,
    T* out_val
) {
    std::fill(indptr, indptr + n_major + 1, 0);
    for (int64_t k = 0; k < nnz; ++k) ++indptr[(major ? major[k] : 0) + 1];
    for (int64_t r = 0; r < n_major; ++r) indptr[r + 1] += indptr[r];
    if (presorted) {
        for (int64_t k = 0; k < nnz; ++k) {
            const int64_t p = indptr[major ? major[k] : 0]++;
            out_idx[p] = minor[k];
            out_val[p] = val[k];
        }
        for (int64_t r = n_major; r > 0; --r) indptr[r] = indptr[r - 1];
        indptr[0] = 0;
        return nnz;
    }
    std::vector<Pair<T>> tmp(static_cast<std::size_t>(nnz));
    for (int64_t k = 0; k < nnz; ++k)
        tmp[static_cast<std::size_t>(indptr[major ? major[k] : 0]++)] = {minor[k], val[k]};
    for (int64_t r = n_major; r > 0; --r) indptr[r] = indptr[r - 1];
    indptr[0] = 0;
    int64_t w = 0;
    for (int64_t r = 0; r < n_major; ++r) {
        const int64_t s = indptr[r], e = indptr[r + 1];
        indptr[r] = w;
        w += emit_line(tmp.data() + s, e - s, false, out_idx + w, out_val + w);
    }
    indptr[n_major] = w;
    return w;
}

// The LSD path (one thread): a stable counting sort of the entries by minor, then a stable
// counting sort by major straight into the output lines. Both are stable, so a line ends up in
// minor order with equal minors in input order, and an entry equal to the line's last is added to
// it as it arrives (left to right); the lines are closed up when duplicates left gaps. No
// comparison sort: sorting random lines of ten entries cost about 5 ns an entry, this path about
// 2 in all (M4 Pro, 2026-11-22). Needs n_minor counters: returns -1 (nothing written) when the
// largest minor index is too large for that, and the caller takes another path.
template <class T>
int64_t compress_lsd(
    int64_t n_major,
    const int64_t* major,
    const int64_t* minor,
    const T* val,
    int64_t nnz,
    int64_t* indptr,
    int64_t* out_idx,
    T* out_val
) {
    int64_t n_minor = 0;
    for (int64_t k = 0; k < nnz; ++k) n_minor = std::max(n_minor, minor[k] + 1);
    if (n_minor > 2 * nnz + 65536) return -1;
    struct Line {
        int64_t pos, last;  // next free slot, minor of the last entry written (-1: none yet)
    };
    std::unique_ptr<int64_t[]> cmin(new int64_t[static_cast<std::size_t>(n_minor + 1)]());
    std::unique_ptr<Line[]> line(new Line[static_cast<std::size_t>(n_major + 1)]);
    for (int64_t r = 0; r <= n_major; ++r) line[r] = {0, -1};
    for (int64_t k = 0; k < nnz; ++k) ++cmin[minor[k] + 1];
    if (major)
        for (int64_t k = 0; k < nnz; ++k) ++line[major[k] + 1].pos;
    else line[1].pos = nnz;
    for (int64_t c = 0; c < n_minor; ++c) cmin[c + 1] += cmin[c];
    for (int64_t r = 0; r < n_major; ++r) line[r + 1].pos += line[r].pos;
    for (int64_t r = 0; r <= n_major; ++r) indptr[r] = line[r].pos;
    struct MajorValue {
        int64_t major;
        T v;
    };
    std::unique_ptr<MajorValue[]> tmp(new MajorValue[static_cast<std::size_t>(nnz)]);
    for (int64_t k = 0; k < nnz; ++k) tmp[cmin[minor[k]]++] = {major ? major[k] : 0, val[k]};
    // cmin[m] is now the end of minor m's entries
    bool dups = false;
    for (int64_t m = 0, p = 0; m < n_minor; ++m) {
        for (const int64_t e = cmin[m]; p < e; ++p) {
            Line& d = line[tmp[p].major];
            if (d.last == m) {
                out_val[d.pos - 1] = add_value(out_val[d.pos - 1], tmp[p].v);
                dups = true;
                continue;
            }
            d.last = m;
            out_idx[d.pos] = m;
            out_val[d.pos] = tmp[p].v;
            ++d.pos;
        }
    }
    if (!dups) return nnz;
    int64_t w = 0;
    for (int64_t r = 0; r < n_major; ++r) {
        const int64_t s = indptr[r], e = line[r].pos;
        indptr[r] = w;
        if (w != s)
            for (int64_t k = s; k < e; ++k) {
                out_idx[w + k - s] = out_idx[k];
                out_val[w + k - s] = out_val[k];
            }
        w += e - s;
    }
    indptr[n_major] = w;
    return w;
}

// Where the distinct entries of the bucket path sit before compaction.
struct Segments {
    std::vector<int64_t> src, len, dst;
};

// The bucket path: entries go to buckets of 2^shift consecutive major indices (a stable scatter:
// chunks of entries in order), then each bucket is sorted by (major, minor) and summed on its
// own -- cache-sized pieces, in parallel when the pool is free. Bucket and chunk sizes depend on
// n_major and nnz alone. The distinct entries of bucket b sit at [src[b], src[b] + len[b]) and
// belong at dst[b].
template <class T>
int64_t compress_buckets(
    int64_t n_major,
    const int64_t* major,
    const int64_t* minor,
    const T* val,
    int64_t nnz,
    bool presorted,
    int64_t* indptr,
    int64_t* out_idx,
    T* out_val,
    Segments& seg
) {
    const int64_t target = std::clamp<int64_t>(nnz >> 11, 64, 1024);  // fewer buckets balance badly on the pool
    int shift = 0;
    while (shift < 62 && ((n_major - 1) >> shift) + 1 > target) ++shift;
    const int64_t width = int64_t(1) << shift;
    auto line_end = [&](int64_t r0) { return n_major - r0 <= width ? n_major : r0 + width; };
    const int64_t nb_ = ((n_major - 1) >> shift) + 1;
    constexpr int64_t kScatter = 1 << 14;
    const int64_t nc = num_chunks(nnz, kScatter);
    std::vector<int64_t> hist(static_cast<std::size_t>(nc * nb_), 0);
    parallel::parallel_for(nc, [&](int64_t c) {
        int64_t* h = hist.data() + c * nb_;
        const int64_t hi = std::min(nnz, (c + 1) * kScatter);
        for (int64_t k = c * kScatter; k < hi; ++k) ++h[(major ? major[k] : 0) >> shift];
    });
    std::vector<int64_t> bstart(static_cast<std::size_t>(nb_ + 1));
    int64_t run = 0;
    for (int64_t b = 0; b < nb_; ++b) {
        bstart[b] = run;
        for (int64_t c = 0; c < nc; ++c) {
            const int64_t v = hist[c * nb_ + b];
            hist[c * nb_ + b] = run;
            run += v;
        }
    }
    bstart[nb_] = run;
    std::vector<Entry<T>> tmp(static_cast<std::size_t>(nnz));
    parallel::parallel_for(nc, [&](int64_t c) {
        int64_t* h = hist.data() + c * nb_;
        const int64_t hi = std::min(nnz, (c + 1) * kScatter);
        for (int64_t k = c * kScatter; k < hi; ++k) {
            const int64_t r = major ? major[k] : 0;
            tmp[static_cast<std::size_t>(h[r >> shift]++)] = {r, minor[k], val[k]};
        }
    });
    std::vector<int64_t> uniq(static_cast<std::size_t>(nb_));
    parallel::parallel_for(nb_, [&](int64_t b) {
        const int64_t r0 = b << shift, r1 = line_end(r0);
        Entry<T>* e = tmp.data() + bstart[b];
        const int64_t n = bstart[b + 1] - bstart[b], span = r1 - r0;
        const int64_t base = bstart[b];
        if (span <= 4 * n + 1024) {
            // counting sort by major (stable)
            std::vector<int64_t> start(static_cast<std::size_t>(span + 1), 0);
            for (int64_t i = 0; i < n; ++i) ++start[e[i].major - r0 + 1];
            for (int64_t r = 0; r < span; ++r) start[r + 1] += start[r];
            if (presorted) {
                // the final positions are known: write the output directly
                for (int64_t r = 0; r < span; ++r) indptr[r0 + r] = start[r + 1] - start[r];
                for (int64_t i = 0; i < n; ++i) {
                    const int64_t p = base + start[e[i].major - r0]++;
                    out_idx[p] = e[i].minor;
                    out_val[p] = e[i].v;
                }
                uniq[b] = n;
                return;
            }
            std::vector<Pair<T>> lines(static_cast<std::size_t>(n));
            std::vector<int64_t> next(start.begin(), start.end() - 1);
            for (int64_t i = 0; i < n; ++i) lines[next[e[i].major - r0]++] = {e[i].minor, e[i].v};
            int64_t w = base;
            for (int64_t r = 0; r < span; ++r) {
                const int64_t s = start[r];
                const int64_t c = emit_line(lines.data() + s, start[r + 1] - s, false, out_idx + w, out_val + w);
                indptr[r0 + r] = c;
                w += c;
            }
            uniq[b] = w - base;
        } else {
            // few entries over many lines: a stable sort of the bucket by major
            std::stable_sort(e, e + n, [](const Entry<T>& a, const Entry<T>& x) { return a.major < x.major; });
            std::fill(indptr + r0, indptr + r1, 0);
            std::vector<Pair<T>> lines(static_cast<std::size_t>(n));
            for (int64_t i = 0; i < n; ++i) lines[i] = {e[i].minor, e[i].v};
            int64_t w = base;
            for (int64_t i = 0; i < n;) {
                int64_t j = i + 1;
                while (j < n && e[j].major == e[i].major) ++j;
                const int64_t c = emit_line(lines.data() + i, j - i, presorted, out_idx + w, out_val + w);
                indptr[e[i].major] = c;
                w += c;
                i = j;
            }
            uniq[b] = w - base;
        }
    });
    std::vector<int64_t> ustart(static_cast<std::size_t>(nb_ + 1), 0);
    for (int64_t b = 0; b < nb_; ++b) ustart[b + 1] = ustart[b] + uniq[b];
    parallel::parallel_for(nb_, [&](int64_t b) {
        const int64_t r0 = b << shift, r1 = line_end(r0);
        int64_t acc = ustart[b];
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t c = indptr[r];
            indptr[r] = acc;
            acc += c;
        }
    });
    indptr[n_major] = ustart[nb_];
    seg.src.assign(bstart.begin(), bstart.end() - 1);
    seg.len = std::move(uniq);
    seg.dst.assign(ustart.begin(), ustart.end() - 1);
    return ustart[nb_];
}

// Which path: they all give the same arrays, so the choice is speed alone, by size. The bucket
// path pays for its extra pass with cache-sized pieces and the pool. Measured on random matrices
// with 10 entries a line (M4 Pro, 2026-11-21): with the pool free it wins from 2^16 entries; on one
// thread, for a transpose from 2^20, and when lines are sorted and summed from 2^21 against the LSD
// path (2026-11-22; before that path, from 2^17 against the direct one).
bool use_buckets(int64_t nnz, bool presorted) {
    if (nnz >= (presorted ? int64_t(1) << 20 : int64_t(1) << 21)) return true;
    return nnz >= (int64_t(1) << 16) && parallel::available();
}

// The distinct entries of `out` (segments) moved to arrays of exactly `total` entries.
void shrink(Compressed& out, int64_t total, const Segments& seg) {
    NDArray idx = new_index(total);
    const int64_t shape[] = {total};
    NDArray val = new_array(out.data.dtype, shape);
    const int64_t size = out.data.itemsize();
    const parallel::Unlocked unlocked(total >= kParallelThreshold);
    auto move = [&](int64_t s) {
        const int64_t n = seg.len[s];
        if (n == 0) return;
        std::memcpy(i64(idx) + seg.dst[s], i64(out.indices) + seg.src[s], static_cast<std::size_t>(n) * 8);
        std::memcpy(
            val.data + seg.dst[s] * size, out.data.data + seg.src[s] * size, static_cast<std::size_t>(n * size)
        );
    };
    const int64_t ns = static_cast<int64_t>(seg.len.size());
    if (total >= kParallelThreshold) parallel::parallel_for(ns, move);
    else
        for (int64_t s = 0; s < ns; ++s) move(s);
    out.indices = std::move(idx);
    out.data = std::move(val);
}

}  // namespace

Compressed compress(
    int64_t n_major, const int64_t* major, const int64_t* minor, const NDArray& data, int64_t nnz, bool presorted
) {
    Compressed out;
    out.indptr = new_indptr(n_major);
    out.indices = new_index(nnz);
    const int64_t shape[] = {nnz};
    out.data = new_array(data.dtype, shape);
    int64_t total = 0;
    Segments seg;
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const parallel::Unlocked unlocked(nnz + n_major >= kParallelThreshold);
        const T* val = typed<T>(data);
        if (use_buckets(nnz, presorted) && n_major > 0) {
            total = compress_buckets<T>(
                n_major, major, minor, val, nnz, presorted, i64(out.indptr), i64(out.indices), typed<T>(out.data), seg
            );
        } else {
            total = presorted
                        ? -1
                        : compress_lsd<T>(
                              n_major, major, minor, val, nnz, i64(out.indptr), i64(out.indices), typed<T>(out.data)
                          );
            if (total < 0)
                total = compress_direct<T>(
                    n_major, major, minor, val, nnz, presorted, i64(out.indptr), i64(out.indices), typed<T>(out.data)
                );
            seg.src = {0};
            seg.len = {total};
            seg.dst = {0};
        }
    });
    if (total < nnz) {
        // The distinct entries already sit at [0, total) when there is one segment: keep the
        // arrays and shorten them (a few duplicates), or copy when much of them would be unused.
        if (seg.len.size() == 1 && total * 8 >= nnz * 7) {
            out.indices.shape[0] = total;
            out.data.shape[0] = total;
        } else {
            shrink(out, total, seg);
        }
    }
    return out;
}

bool is_canonical(const int64_t* major, const int64_t* minor, int64_t nnz) {
    std::atomic<bool> ok{true};
    for_chunks(nnz, [&](int64_t, int64_t lo, int64_t hi) {
        if (!ok.load(std::memory_order_relaxed)) return;
        for (int64_t k = std::max<int64_t>(lo, 1); k < hi; ++k) {
            const int64_t a = major ? major[k - 1] : 0, b = major ? major[k] : 0;
            if (b < a || (b == a && minor[k] <= minor[k - 1])) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
        }
    });
    return ok.load();
}

void indptr_of_sorted(const int64_t* major, int64_t nnz, int64_t n_major, int64_t* indptr) {
    if (!major) {
        indptr[0] = 0;
        if (n_major > 0) indptr[1] = nnz;
        return;
    }
    for_chunks(
        nnz,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t k = lo; k < hi; ++k) {
                const int64_t prev = k == 0 ? -1 : major[k - 1];
                for (int64_t r = prev + 1; r <= major[k]; ++r) indptr[r] = k;
            }
        },
        kChunk, nnz + n_major
    );
    const int64_t first = nnz == 0 ? 0 : major[nnz - 1] + 1;
    for_chunks(n_major + 1 - first, [&](int64_t, int64_t lo, int64_t hi) {
        for (int64_t r = first + lo; r < first + hi; ++r) indptr[r] = nnz;
    });
}

namespace {

// ---- transpose ----

template <class T>
void transpose_serial(
    int64_t n_major, int64_t n_minor, const int64_t* ip, const int64_t* ix, const T* v, int64_t* op, int64_t* ox, T* ov
) {
    const int64_t nnz = ip[n_major];
    std::fill(op, op + n_minor + 1, 0);
    for (int64_t k = 0; k < nnz; ++k) ++op[ix[k] + 1];
    for (int64_t c = 0; c < n_minor; ++c) op[c + 1] += op[c];
    for (int64_t r = 0; r < n_major; ++r)
        for (int64_t k = ip[r]; k < ip[r + 1]; ++k) {
            const int64_t p = op[ix[k]]++;
            ox[p] = r;
            ov[p] = v[k];
        }
    for (int64_t c = n_minor; c > 0; --c) op[c] = op[c - 1];
    op[0] = 0;
}

// The transpose on one thread for larger inputs: entries go by blocks of 2^shift columns to a
// buffer (key = row * 2^shift + column within the block, and the value; in row order, so stable),
// then each block -- cache-sized -- counts its columns and places its entries. The plain scatter
// writes to n_minor places at random; this writes in order but twice. Measured (M4 Pro,
// 2026-11-22, random matrices with 10 entries a row): 1.2 times the plain scatter at nnz 10^6 and
// 3.5 times at 10^7. Returns false (nothing written) when the key does not fit 64 bits.
template <class T>
bool transpose_blocks(
    int64_t n_major, int64_t n_minor, const int64_t* ip, const int64_t* ix, const T* v, int64_t* op, int64_t* ox, T* ov
) {
    const int64_t nnz = ip[n_major];
    // about 2048 entries a block
    int shift = 0;
    const double per_column = static_cast<double>(nnz) / static_cast<double>(n_minor);
    while (shift < 16 && static_cast<double>(int64_t(2) << shift) * per_column <= 2048.0) ++shift;
    if (n_major > (int64_t(1) << (63 - shift))) return false;
    const int64_t width = int64_t(1) << shift, mask = width - 1;
    const int64_t nb = ((n_minor - 1) >> shift) + 1;
    std::unique_ptr<int64_t[]> next(new int64_t[static_cast<std::size_t>(nb + 1)]());
    for (int64_t k = 0; k < nnz; ++k) ++next[(ix[k] >> shift) + 1];
    for (int64_t b = 0; b < nb; ++b) next[b + 1] += next[b];
    std::vector<int64_t> start(next.get(), next.get() + nb + 1);
    struct KeyValue {
        uint64_t key;
        T v;
    };
    std::unique_ptr<KeyValue[]> tmp(new KeyValue[static_cast<std::size_t>(nnz)]);
    for (int64_t r = 0; r < n_major; ++r) {
        const uint64_t rk = static_cast<uint64_t>(r) << shift;
        for (int64_t k = ip[r]; k < ip[r + 1]; ++k)
            tmp[next[ix[k] >> shift]++] = {rk | static_cast<uint64_t>(ix[k] & mask), v[k]};
    }
    std::unique_ptr<int64_t[]> cnt(new int64_t[static_cast<std::size_t>(width + 1)]);
    for (int64_t b = 0; b < nb; ++b) {
        const int64_t s = start[b], e = start[b + 1];
        const int64_t c0 = b << shift, w = std::min(width, n_minor - c0);
        std::fill(cnt.get(), cnt.get() + w + 1, 0);
        for (int64_t q = s; q < e; ++q) ++cnt[(tmp[q].key & mask) + 1];
        cnt[0] = s;
        for (int64_t j = 0; j < w; ++j) cnt[j + 1] += cnt[j];
        std::copy(cnt.get(), cnt.get() + w, op + c0);
        for (int64_t q = s; q < e; ++q) {
            const int64_t p = cnt[tmp[q].key & mask]++;
            ox[p] = static_cast<int64_t>(tmp[q].key >> shift);
            ov[p] = tmp[q].v;
        }
    }
    op[n_minor] = nnz;
    return true;
}

NDArray expand_rows(const NDArray& indptr, int64_t n_major, int64_t nnz);

}  // namespace

Compressed transpose(
    int64_t n_major, int64_t n_minor, const NDArray& indptr, const NDArray& indices, const NDArray& data
) {
    const int64_t nnz = indices.shape[0];
    // one thread from 2^15 entries: the blocked transpose (below that the plain scatter fits the cache)
    const bool blocks = nnz >= kParallelThreshold && n_minor > 0 && !parallel::available();
    if (!blocks && use_buckets(nnz, true)) {
        const NDArray rows = expand_rows(indptr, n_major, nnz);
        return compress(n_minor, i64(indices), i64(rows), data, nnz, true);
    }
    Compressed out;
    out.indptr = new_indptr(n_minor);
    out.indices = new_index(nnz);
    const int64_t shape[] = {nnz};
    out.data = new_array(data.dtype, shape);
    visit_dtype(data.dtype, [&]<class T>(std::type_identity<T>) {
        const parallel::Unlocked unlocked(nnz + n_minor >= kParallelThreshold);
        if (blocks && transpose_blocks<T>(
                          n_major, n_minor, i64(indptr), i64(indices), typed<T>(data), i64(out.indptr),
                          i64(out.indices), typed<T>(out.data)
                      ))
            return;
        transpose_serial<T>(
            n_major, n_minor, i64(indptr), i64(indices), typed<T>(data), i64(out.indptr), i64(out.indices),
            typed<T>(out.data)
        );
    });
    return out;
}

namespace {

NDArray expand_rows(const NDArray& indptr, int64_t n_major, int64_t nnz) {
    NDArray rows = new_index(nnz);
    const int64_t* ip = i64(indptr);
    int64_t* out = i64(rows);
    for_chunks(
        n_major,
        [&](int64_t, int64_t lo, int64_t hi) {
            for (int64_t r = lo; r < hi; ++r)
                for (int64_t k = ip[r]; k < ip[r + 1]; ++k) out[k] = r;
        },
        kChunk, nnz + n_major
    );
    return rows;
}

// A new contiguous copy of a contiguous 1-D array.
nb::object copy_of(const NDArray& a) {
    NDArray c = new_array(a.dtype, a.shape_span());
    {
        const parallel::Unlocked unlocked(a.shape[0] >= kParallelThreshold);
        std::memcpy(c.data, a.data, static_cast<std::size_t>(a.nbytes()));
    }
    return wrap(std::move(c));
}

nb::tuple triple(Compressed&& c) {
    return nb::make_tuple(wrap(std::move(c.indptr)), wrap(std::move(c.indices)), wrap(std::move(c.data)));
}

const int64_t* opt_major(nb::handle o, const char* func, int64_t nnz) {
    if (o.is_none()) return nullptr;
    const NDArray& a = vector_arg(o, func, DType::Int64);
    if (a.shape[0] != nnz) throw nb::value_error((std::string(func) + "(): lengths differ").c_str());
    return i64(a);
}

// ---- user input ----

std::string prefix(const std::string& func) { return func + "(): "; }

// Reads element k of a 1-D int64 array of any stride.
inline int64_t at(const NDArray& a, int64_t k) { return *reinterpret_cast<const int64_t*>(a.data + k * a.strides[0]); }

int64_t index_max(nb::handle xh, int axis, const std::string& func) {
    const NDArray& x = require_ndarray(xh, "x");
    const int64_t n = x.shape[0];
    const int64_t nt = num_chunks(n);
    std::vector<int64_t> mx(static_cast<std::size_t>(std::max<int64_t>(nt, 1)), -1);
    parallel::FirstRow neg;
    for_chunks(n, [&](int64_t t, int64_t lo, int64_t hi) {
        int64_t m = -1;
        for (int64_t k = lo; k < hi; ++k) {
            const int64_t v = at(x, k);
            if (v < 0) {
                neg.offer(k);
                return;
            }
            m = std::max(m, v);
        }
        mx[t] = m;
    });
    if (neg.any())
        throw nb::value_error((prefix(func) + "index " + std::to_string(at(x, neg.row())) + " on axis " +
                               std::to_string(axis) + " is negative")
                                  .c_str());
    int64_t m = -1;
    for (int64_t v : mx) m = std::max(m, v);
    return m;
}

nb::object index_copy(nb::handle xh, int64_t bound, int axis, const std::string& func) {
    const NDArray& x = require_ndarray(xh, "x");
    const int64_t n = x.shape[0];
    NDArray out = new_index(n);
    int64_t* o = i64(out);
    parallel::FirstRow bad;
    for_chunks(n, [&](int64_t, int64_t lo, int64_t hi) {
        for (int64_t k = lo; k < hi; ++k) {
            const int64_t v = at(x, k);
            if (v < 0 || v >= bound) {
                bad.offer(k);
                return;
            }
            o[k] = v;
        }
    });
    if (bad.any())
        throw nb::value_error((prefix(func) + "index " + std::to_string(at(x, bad.row())) +
                               " is out of bounds for axis " + std::to_string(axis) + " with size " +
                               std::to_string(bound))
                                  .c_str());
    out.writeable = false;
    return wrap(std::move(out));
}

nb::tuple compressed_in(
    int64_t n_major,
    int64_t n_minor,
    nb::handle indptr_h,
    nb::handle indices_h,
    nb::handle data_h,
    int minor_axis,
    bool share,
    const std::string& func
) {
    const NDArray& ip_in = vector_arg(indptr_h, "_compressed_in", DType::Int64);
    const NDArray& ix_in = vector_arg(indices_h, "_compressed_in", DType::Int64);
    const NDArray& data = vector_arg(data_h, "_compressed_in");
    const int64_t nnz = ix_in.shape[0];
    if (ip_in.shape[0] != n_major + 1 || data.shape[0] != nnz)
        throw nb::value_error("_compressed_in(): lengths differ");
    NDArray indptr = new_indptr(n_major);
    NDArray indices = new_index(nnz);
    const int64_t* ips = i64(ip_in);
    const int64_t* ixs = i64(ix_in);
    int64_t* ip = i64(indptr);
    int64_t* ix = i64(indices);
    // indptr: starts at 0, never decreases, ends at nnz
    parallel::FirstRow down;
    for_chunks(n_major + 1, [&](int64_t, int64_t lo, int64_t hi) {
        for (int64_t r = lo; r < hi; ++r) {
            ip[r] = ips[r];
            if (r > 0 && ips[r] < ips[r - 1]) {
                down.offer(r);
                return;
            }
        }
    });
    if (ips[0] != 0)
        throw nb::value_error((prefix(func) + "'indptr' must start at 0, not " + std::to_string(ips[0])).c_str());
    if (down.any()) throw nb::value_error((prefix(func) + "'indptr' must not decrease").c_str());
    if (ips[n_major] != nnz)
        throw nb::value_error((prefix(func) + "'indptr' must end at the number of stored entries, " +
                               std::to_string(nnz) + ", not " + std::to_string(ips[n_major]))
                                  .c_str());
    // indices: in range; lines sorted without duplicates?
    parallel::FirstRow bad;
    std::atomic<bool> sorted{true};
    for_chunks(
        n_major,
        [&](int64_t, int64_t lo, int64_t hi) {
            bool ok = true;
            for (int64_t r = lo; r < hi; ++r)
                for (int64_t k = ip[r]; k < ip[r + 1]; ++k) {
                    const int64_t v = ixs[k];
                    if (v < 0 || v >= n_minor) {
                        bad.offer(k);
                        return;
                    }
                    if (k > ip[r] && v <= ixs[k - 1]) ok = false;
                    ix[k] = v;
                }
            if (!ok) sorted.store(false, std::memory_order_relaxed);
        },
        kChunk, n_major + nnz
    );
    if (bad.any())
        throw nb::value_error((prefix(func) + "index " + std::to_string(ixs[bad.row()]) +
                               " is out of bounds for axis " + std::to_string(minor_axis) + " with size " +
                               std::to_string(n_minor))
                                  .c_str());
    indptr.writeable = false;
    indices.writeable = false;
    if (sorted.load())
        return nb::make_tuple(
            wrap(std::move(indptr)), wrap(std::move(indices)), share ? nb::borrow(data_h) : copy_of(data)
        );
    const NDArray rows = expand_rows(indptr, n_major, nnz);
    Compressed c = compress(n_major, i64(rows), i64(indices), data, nnz, false);
    c.indptr.writeable = false;
    c.indices.writeable = false;
    return triple(std::move(c));
}

}  // namespace

NDArray expand(const NDArray& indptr, int64_t n_major, int64_t nnz) { return expand_rows(indptr, n_major, nnz); }

void bind_canonical(nb::module_& m) {
    m.def(
        "_compress",
        [](int64_t n_major, nb::handle major_h, nb::handle minor_h, nb::handle data_h, bool share) {
            const NDArray& minor = vector_arg(minor_h, "_compress", DType::Int64);
            const NDArray& data = vector_arg(data_h, "_compress");
            const int64_t nnz = minor.shape[0];
            if (data.shape[0] != nnz) throw nb::value_error("_compress(): lengths differ");
            const int64_t* major = opt_major(major_h, "_compress", nnz);
            if (is_canonical(major, i64(minor), nnz)) {
                NDArray indptr = new_indptr(n_major);
                indptr_of_sorted(major, nnz, n_major, i64(indptr));
                indptr.writeable = false;
                return nb::make_tuple(
                    wrap(std::move(indptr)), nb::borrow(minor_h), share ? nb::borrow(data_h) : copy_of(data)
                );
            }
            Compressed c = compress(n_major, major, i64(minor), data, nnz, false);
            c.indptr.writeable = false;
            c.indices.writeable = false;
            return triple(std::move(c));
        },
        nb::arg("n_major"), nb::arg("major").none(), nb::arg("minor"), nb::arg("data"), nb::kw_only(), nb::arg("share")
    );
    m.def(
        "_transpose",
        [](int64_t n_major, int64_t n_minor, nb::handle indptr_h, nb::handle indices_h, nb::handle data_h) {
            const NDArray& indptr = vector_arg(indptr_h, "_transpose", DType::Int64);
            const NDArray& indices = vector_arg(indices_h, "_transpose", DType::Int64);
            const NDArray& data = vector_arg(data_h, "_transpose");
            if (indptr.shape[0] != n_major + 1 || data.shape[0] != indices.shape[0])
                throw nb::value_error("_transpose(): lengths differ");
            Compressed c = transpose(n_major, n_minor, indptr, indices, data);
            c.indptr.writeable = false;
            c.indices.writeable = false;
            return triple(std::move(c));
        }
    );
    m.def(
        "_compressed_in", &compressed_in, nb::arg("n_major"), nb::arg("n_minor"), nb::arg("indptr"), nb::arg("indices"),
        nb::arg("data"), nb::kw_only(), nb::arg("minor_axis"), nb::arg("share"), nb::arg("func")
    );
    m.def("_index_max", &index_max, nb::arg("x"), nb::kw_only(), nb::arg("axis"), nb::arg("func"));
    m.def("_index_copy", &index_copy, nb::arg("x"), nb::arg("bound"), nb::kw_only(), nb::arg("axis"), nb::arg("func"));
    m.def(
        "_is_canonical",
        [](nb::handle major_h, nb::handle minor_h) {
            const NDArray& minor = vector_arg(minor_h, "_is_canonical", DType::Int64);
            return is_canonical(opt_major(major_h, "_is_canonical", minor.shape[0]), i64(minor), minor.shape[0]);
        },
        nb::arg("major").none(), nb::arg("minor")
    );
    m.def(
        "_sum_duplicates",
        [](int64_t n_major, nb::handle major_h, nb::handle minor_h, nb::handle data_h) {
            const NDArray& minor = vector_arg(minor_h, "_sum_duplicates", DType::Int64);
            const NDArray& data = vector_arg(data_h, "_sum_duplicates");
            const int64_t nnz = minor.shape[0];
            const int64_t* major = opt_major(major_h, "_sum_duplicates", nnz);
            Compressed c = compress(n_major, major, i64(minor), data, nnz, false);
            const int64_t m = c.indices.shape[0];
            nb::object rows = nb::none();
            if (major) {
                NDArray r = expand_rows(c.indptr, n_major, m);
                r.writeable = false;
                rows = wrap(std::move(r));
            }
            c.indices.writeable = false;
            return nb::make_tuple(rows, wrap(std::move(c.indices)), wrap(std::move(c.data)));
        },
        nb::arg("n_major"), nb::arg("major").none(), nb::arg("minor"), nb::arg("data")
    );
}

}  // namespace sd::sparse
