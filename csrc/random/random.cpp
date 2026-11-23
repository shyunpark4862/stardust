// The kernels of sd.random (docs/api/random.md, docs/design/random.md): uniform
// float64, bounded int64, raw words and exponential race keys, element e of a call
// at stream position pos being e = pos + (C-order index). Every element is a pure
// function of (key, e) (philox.hpp), so the pieces on the pool may fall anywhere:
// the bits do not depend on the thread count.
//
// The Generator object, seeding, argument checks and the composite tools
// (choice, permutation) are Python (src/stardust/random/__init__.py); these are
// the submodule stardust._core.random.

#include "random/random.hpp"
#include "random/philox.hpp"

#include "bind/errors.hpp"
#include "bind/pyutil.hpp"
#include "array/convert.hpp"
#include "array/ops/pieces.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace sd {

namespace {

using random::Key;
using random::Stream;

// Work per element, for the pool's threshold: a Philox block is ~10 ns and serves four.
inline constexpr std::int64_t kWordWork = 8;
// Rejected elements of integers() collected before they are finished (four at a time).
inline constexpr int kRejBatch = 64;

NDArray result_array(DType dtype, nb::handle shape_obj) { return new_array(dtype, parse_shape_arg(shape_obj).span()); }

nb::object uniform(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, nb::handle shape_obj) {
    NDArray out = result_array(DType::Float64, shape_obj);
    double* o = reinterpret_cast<double*>(out.data);
    const std::int64_t n = out.size();
    for_pieces(n, kParallelTask, n * kWordWork, [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
        Stream s({k0, k1});
        for (std::int64_t i = i0; i < i1; ++i) o[i] = random::to_unit(s.first(pos + static_cast<std::uint64_t>(i)));
    });
    return wrap(std::move(out));
}

nb::object bits(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, std::int64_t n) {
    const std::int64_t shape[] = {n};
    NDArray out = new_array(DType::Int64, shape);
    auto* o = reinterpret_cast<std::uint64_t*>(out.data);
    for_pieces(n, kParallelTask, n * kWordWork, [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
        Stream s({k0, k1});
        for (std::int64_t i = i0; i < i1; ++i) o[i] = s.first(pos + static_cast<std::uint64_t>(i));
    });
    return wrap(std::move(out));
}

// The offset (bytes) of C-order element i of an array of this shape and these strides.
std::int64_t offset_of(const NDArray& a, std::int64_t i) {
    std::int64_t off = 0;
    for (int d = a.ndim - 1; d >= 0; --d) {
        off += (i % a.shape[d]) * a.strides[d];
        i /= a.shape[d];
    }
    return off;
}

// integers' elements [i0, i1) for one range r (> 0) whose rejections are rare: each handled
// where it comes, as bounded() does (its own function: compiled apart from the batched path,
// the loop keeps the plain one's speed).
[[gnu::noinline]] void ints_few_rejections(
    Key key,
    std::uint64_t pos,
    std::int64_t i0,
    std::int64_t i1,
    std::int64_t l,
    std::uint64_t r,
    std::uint64_t t,
    std::int64_t* o
) {
    Stream s(key);
    for (std::int64_t i = i0; i < i1; ++i) {
        const std::uint64_t e = pos + static_cast<std::uint64_t>(i);
        unsigned __int128 m = static_cast<unsigned __int128>(s.first(e)) * r;
        for (std::uint64_t k = 0; static_cast<std::uint64_t>(m) < t; ++k)
            m = static_cast<unsigned __int128>(s.extra(e, k)) * r;
        o[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(l) + static_cast<std::uint64_t>(m >> 64));
    }
}

// [low, high) (or [low, high] with endpoint) for int64 arrays low and high of the
// result's shape (any strides: broadcast views). The checks (low < high) are the
// caller's; here low + bounded(high - low) in unsigned arithmetic, which is exact
// for any int64 pair with low <= high.
nb::object integers(
    std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, const NDArray& lo, const NDArray& hi, bool endpoint
) {
    NDArray out = new_array(DType::Int64, lo.shape_span());
    auto* o = reinterpret_cast<std::int64_t*>(out.data);
    const std::int64_t n = out.size();
    bool scalar = true;  // low and high the same for every element (the usual call)
    for (int d = 0; d < lo.ndim; ++d) scalar = scalar && lo.strides[d] == 0 && hi.strides[d] == 0;
    const Key key{k0, k1};
    if (scalar && n > 0) {
        // One range for all (the usual call): its threshold 2^64 mod r once, not a
        // division in every element that needs the check (half of them at r = 2^63).
        const std::int64_t l = *reinterpret_cast<const std::int64_t*>(lo.data);
        const std::int64_t h = *reinterpret_cast<const std::int64_t*>(hi.data);
        const std::uint64_t r = static_cast<std::uint64_t>(h) - static_cast<std::uint64_t>(l) + (endpoint ? 1 : 0);
        const std::uint64_t t = r == 0 ? 0 : (0 - r) % r;
        for_pieces(n, kParallelTask, n * kWordWork, [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
            Stream s(key);
            if (r == 0) {
                for (std::int64_t i = i0; i < i1; ++i)
                    o[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(l) + s.first(pos + std::uint64_t(i)));
                return;
            }
            // Lemire's rejection takes element e's extra words in order (bounded()). Rejections
            // are common for a range just above a power of two (half the words at r = 2^63 + 1),
            // so the rejected elements are collected and their first extra block (words 0-3)
            // computed four elements at a time (philox4: four independent chains, not one
            // latency-bound block each); words 4 on come one block at a time as before. The
            // same words, in the same order: the same values.
            if (r <= (std::uint64_t(1) << 60)) {  // rejections below 1 in 16 words
                ints_few_rejections(key, pos, i0, i1, l, r, t, o);
                return;
            }
            std::int64_t rej[kRejBatch + 1];
            int nrej = 0;
            // the rejected elements rej[0 .. cnt), four at a time
            auto finish = [&](int cnt) {
                for (int g = 0; g < cnt; g += 4) {
                    const int lanes = std::min(4, cnt - g);
                    std::uint64_t c0[4];
                    for (int q = 0; q < 4; ++q) c0[q] = pos + static_cast<std::uint64_t>(rej[g + (q < lanes ? q : 0)]);
                    std::array<std::uint64_t, 16> words;
                    random::philox4(c0, key, words, 1);
                    for (int q = 0; q < lanes; ++q) {
                        const std::uint64_t e = c0[q];
                        unsigned __int128 m = 0;
                        std::uint64_t k = 0;
                        do {
                            const std::uint64_t w = k < 4 ? words[4 * q + k] : s.extra(e, k);
                            m = static_cast<unsigned __int128>(w) * r;
                            ++k;
                        } while (static_cast<std::uint64_t>(m) < t);
                        o[rej[g + q]] = static_cast<std::int64_t>(
                            static_cast<std::uint64_t>(l) + static_cast<std::uint64_t>(m >> 64)
                        );
                    }
                }
            };
            // branch-free: every element writes its candidate and its index into rej, which
            // advances only for a rejection (about half the words at r = 2^63 + 1: a branch on
            // it would be mispredicted half the time)
            for (std::int64_t i = i0; i < i1; ++i) {
                const unsigned __int128 m = static_cast<unsigned __int128>(s.first(pos + std::uint64_t(i))) * r;
                o[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(l) + static_cast<std::uint64_t>(m >> 64));
                rej[nrej] = i;
                nrej += static_cast<std::uint64_t>(m) < t ? 1 : 0;
                if (nrej == kRejBatch) {
                    finish(kRejBatch);
                    nrej = 0;
                }
            }
            if (nrej) finish(nrej);
        });
        return wrap(std::move(out));
    }
    for_pieces(n, kParallelTask, n * kWordWork, [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
        Stream s(key);
        for (std::int64_t i = i0; i < i1; ++i) {
            const std::int64_t l = *reinterpret_cast<const std::int64_t*>(lo.data + (scalar ? 0 : offset_of(lo, i)));
            const std::int64_t h = *reinterpret_cast<const std::int64_t*>(hi.data + (scalar ? 0 : offset_of(hi, i)));
            const std::uint64_t r = static_cast<std::uint64_t>(h) - static_cast<std::uint64_t>(l) + (endpoint ? 1 : 0);
            const std::uint64_t e = pos + static_cast<std::uint64_t>(i);
            const std::uint64_t v = random::bounded(s.first(e), r, [&](std::uint64_t k) { return s.extra(e, k); });
            o[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(l) + v);
        }
    });
    return wrap(std::move(out));
}

// E_i / p_i with E_i a standard exponential (-log of a uniform in (0, 1]): the
// arrival times of a race whose k first finishers are a weighted sample of k
// without replacement (Efraimidis and Spirakis, 2006). p_i = 0 never arrives (inf).
// A positive p_i below about 2e-307 would overflow to the same inf and lose to a
// zero by index; its key is the largest double instead. Such elements are then
// ordered by index among themselves: each has p_i below 2.05e-307 (E_i is at most
// 53 log 2 = 36.7, at the smallest uniform), so n of them hold below n * 2.05e-307.
nb::object race_keys(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, const NDArray& p) {
    NDArray out = new_array(DType::Float64, p.shape_span());
    double* o = reinterpret_cast<double*>(out.data);
    const double* pp = reinterpret_cast<const double*>(p.data);
    const std::int64_t n = out.size();
    for_pieces(n, kParallelTask, n * kWordWork, [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
        Stream s({k0, k1});
        for (std::int64_t i = i0; i < i1; ++i) {
            const double u = (static_cast<double>(s.first(pos + static_cast<std::uint64_t>(i)) >> 11) + 1.0) * 0x1p-53;
            o[i] = pp[i] > 0.0 ? std::min(-std::log(u) / pp[i], std::numeric_limits<double>::max()) : HUGE_VAL;
        }
    });
    return wrap(std::move(out));
}

// The k smallest race keys' indices in increasing (key, index) order: argsort(race_keys(...),
// stable)[:k], without sorting all n: one pass keeps the k smallest so far. Once k are held,
// an element's key is computed only when it can enter: -log(u) >= 1 - u (exact for these u,
// multiples of 2^-53 in (0, 1]), so 1 - u > T p (1 + 2^-40), T the largest key held, proves
// the key above T (the margin covers the roundings of log, the quotient and
// the product) and the element is passed over; near the top of the race the bound is tight
// (E is small), so almost every key computed is one that enters. The keys that are computed are
// race_keys' own: the same indices as sorting all of them.
nb::object race_choice(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, const NDArray& p, std::int64_t k) {
    const std::int64_t n = p.size();
    k = std::clamp<std::int64_t>(k, 0, n);
    const double* pp = reinterpret_cast<const double*>(p.data);
    struct Entry {
        double key;
        std::int64_t idx;
    };
    // (key, index) order; keys are never NaN
    auto less = [](const Entry& a, const Entry& b) { return a.key < b.key || (a.key == b.key && a.idx < b.idx); };
    // the candidates: the k smallest so far, then the elements below their largest T (at most
    // k more, after which the k smallest are selected again and T lowered): an amortized
    // selection per candidate in place of a heap's sift with its unpredictable branches
    std::vector<Entry> buf;
    {
        nb::gil_scoped_release release;
        Stream s({k0, k1});
        auto key_of = [&](double u, std::int64_t i) {
            return pp[i] > 0.0 ? std::min(-std::log(u) / pp[i], std::numeric_limits<double>::max()) : HUGE_VAL;
        };
        const auto kk = static_cast<std::size_t>(k);
        buf.reserve(2 * kk);
        auto select = [&]() {  // keep the k smallest; T their largest
            std::nth_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(kk - 1), buf.end(), less);
            buf.resize(kk);
            return buf[kk - 1].key;
        };
        constexpr double kMargin = 1.0 + 0x1p-40;
        double T = HUGE_VAL;
        std::int64_t i = 0;
        for (; i < n && buf.size() < kk; ++i) {
            const double u = (static_cast<double>(s.first(pos + static_cast<std::uint64_t>(i)) >> 11) + 1.0) * 0x1p-53;
            buf.push_back({key_of(u, i), i});
        }
        if (k > 0) T = select();
        for (; i < n && k > 0; ++i) {
            const double u = (static_cast<double>(s.first(pos + static_cast<std::uint64_t>(i)) >> 11) + 1.0) * 0x1p-53;
            if (1.0 - u > T * pp[i] * kMargin) continue;  // its key is above T (see above)
            const double key = key_of(u, i);
            if (!(key < T)) continue;  // a later index never wins a tie
            buf.push_back({key, i});
            if (buf.size() == 2 * kk) T = select();
        }
        if (buf.size() > kk) select();
        std::sort(buf.begin(), buf.end(), less);
    }
    const std::int64_t shape[] = {k};
    NDArray out = new_array(DType::Int64, shape);
    auto* o = reinterpret_cast<std::int64_t*>(out.data);
    for (std::int64_t q = 0; q < k; ++q) o[q] = buf[static_cast<std::size_t>(q)].idx;
    return wrap(std::move(out));
}

// One uniform float64 (element pos of the stream) as a Python float: random()'s scalar call.
double uniform1(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos) {
    return random::to_unit(random::philox({pos >> 2, 0, 0, 0}, {k0, k1})[pos & 3]);
}
// ... with the element reserved from the generator's state here (no Python lock, no tuples)
double uniform1_state(random::State& st) {
    const random::State::Taken t = st.reserve(1);
    return uniform1(t.k0, t.k1, t.pos);
}

// The first k entries of a Fisher-Yates shuffle of 0 .. n-1 (k <= n): step i swaps
// entry i with entry i + bounded(n - i), its draw from element pos + i of the stream
// (extra words for rejections), so the result depends on (key, pos, n, k) only. k = n is
// a permutation, shuffled in the result itself. For a small sample of a large population
// the entries touched live in a hash table (O(k) time and memory; numpy's choice without
// replacement does the same in effect); else in a full array of n beside the result.
// The memory of each path is checked before the stream moves (random/__init__.py
// _shuffle_bytes, docs/design/random.md §4).
struct Slot {
    std::int64_t key, value;
};
// The table's bytes: 16 * bit_ceil(max(2 k, 16)), below 64 k from k = 5 and 256 for k <= 4:
// max(64 k, 256) bounds it (_shuffle_bytes).
inline constexpr std::int64_t kSlotsPerEntry = 2;
inline constexpr std::int64_t kAhead = 16;          // draws made ahead of the table work
inline constexpr std::int64_t kPrefetchMin = 1 << 19;  // shuffled entries from which the array path prefetches

nb::object fisher_yates(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, std::int64_t n, std::int64_t k) {
    const std::int64_t shape[] = {k};
    NDArray out = new_array(DType::Int64, shape);
    auto* o = reinterpret_cast<std::int64_t*>(out.data);
    Stream s({k0, k1});
    auto draw = [&](std::int64_t i) {
        const std::uint64_t e = pos + static_cast<std::uint64_t>(i);
        return i + static_cast<std::int64_t>(random::bounded(
                       s.first(e), static_cast<std::uint64_t>(n - i), [&](std::uint64_t t) { return s.extra(e, t); }
                   ));
    };
    {
        nb::gil_scoped_release release;  // the loop only: wrap() below needs the GIL
        // the swaps of steps 0 .. k-1 in a (filled with 0 .. n-1). The draws do not depend on
        // the array: once it leaves the cache (from kPrefetchMin entries) step i + kAhead's is
        // made kAhead steps early and its entry prefetched (measured: 10^7 100 -> 66 ms, 10^6
        // 3.3 -> 3.1; in cache, 10^5, the plain loop is 15% faster)
        auto shuffle = [&](std::int64_t* a) {
            if (n < kPrefetchMin) {
                for (std::int64_t i = 0; i < k; ++i) std::swap(a[i], a[draw(i)]);
                return;
            }
            std::int64_t ahead[kAhead];
            const std::int64_t first = std::min<std::int64_t>(k, kAhead);
            for (std::int64_t t = 0; t < first; ++t) {
                ahead[t] = draw(t);
                __builtin_prefetch(&a[ahead[t]], 1);
            }
            for (std::int64_t i = 0; i < k; ++i) {
                const std::int64_t j = ahead[i % kAhead];
                if (const std::int64_t t = i + kAhead; t < k) {
                    ahead[i % kAhead] = draw(t);
                    __builtin_prefetch(&a[ahead[i % kAhead]], 1);
                }
                std::swap(a[i], a[j]);
            }
        };
        if (k == n) {
            for (std::int64_t i = 0; i < n; ++i) o[i] = i;
            shuffle(o);
        } else if (k <= n / 16) {
            // The entries that are not their own index, in a flat open-addressing table: a power
            // of two of at least kSlotsPerEntry * k slots (key -1 is empty), Fibonacci hashing,
            // linear probing. At most k keys (one per step), so the load stays at most 1/2. The
            // node-based std::unordered_map took 51 ms for k = 1e6 of n = 1e10 (numpy 10 ms).
            const std::size_t cap =
                std::bit_ceil(static_cast<std::size_t>(std::max<std::int64_t>(kSlotsPerEntry * k, 16)));
            const int shift = 64 - std::countr_zero(cap);
            std::vector<Slot> table(cap, Slot{-1, 0});
            auto home = [&](std::int64_t key) {
                return static_cast<std::size_t>((static_cast<std::uint64_t>(key) * 0x9E3779B97F4A7C15ull) >> shift);
            };
            auto slot = [&](std::int64_t key) -> Slot& {
                std::size_t h = home(key);
                while (table[h].key != -1 && table[h].key != key) h = (h + 1) & (cap - 1);
                return table[h];
            };
            // The draws do not depend on the table: step i + kAhead's is made kAhead steps early
            // and its two home slots are prefetched, so the random reads overlap.
            std::int64_t ahead[kAhead];
            const std::int64_t first = std::min<std::int64_t>(k, kAhead);
            for (std::int64_t t = 0; t < first; ++t) {
                ahead[t] = draw(t);
                __builtin_prefetch(&table[home(ahead[t])], 1);
                __builtin_prefetch(&table[home(t)]);
            }
            for (std::int64_t i = 0; i < k; ++i) {
                const std::int64_t j = ahead[i % kAhead];
                if (const std::int64_t t = i + kAhead; t < k) {
                    ahead[i % kAhead] = draw(t);
                    __builtin_prefetch(&table[home(ahead[i % kAhead])], 1);
                    __builtin_prefetch(&table[home(t)]);
                }
                const Slot& si = slot(i);
                const std::int64_t vi = si.key == -1 ? i : si.value;
                Slot& sj = slot(j);
                const std::int64_t vj = sj.key == -1 ? j : sj.value;
                sj = Slot{j, vi};
                o[i] = vj;
            }
        } else {
            // not zero-filled first (8n bytes): filled with 0 .. n-1 directly
            std::unique_ptr<std::int64_t[]> a(new std::int64_t[static_cast<std::size_t>(n)]);
            for (std::int64_t i = 0; i < n; ++i) a[static_cast<std::size_t>(i)] = i;
            shuffle(a.get());
            std::copy(a.get(), a.get() + k, o);
        }
    }
    return wrap(std::move(out));
}

// One Philox block, for the known-answer tests.
nb::tuple philox_block(
    std::uint64_t c0, std::uint64_t c1, std::uint64_t c2, std::uint64_t c3, std::uint64_t k0, std::uint64_t k1
) {
    const random::Block b = random::philox({c0, c1, c2, c3}, {k0, k1});
    return nb::make_tuple(b[0], b[1], b[2], b[3]);
}

}  // namespace

void bind_random(nb::module_& core) {
    nb::module_ m = core.def_submodule("random", "sd.random kernels (docs/design/random.md)");
    m.def("uniform", &uniform);
    m.def("bits", &bits);
    m.def("integers", &integers);
    m.def("race_keys", &race_keys);
    m.def("race_choice", &race_choice);
    m.def("uniform1", &uniform1);
    m.def("uniform1_state", &uniform1_state);
    nb::class_<random::State>(m, "State")
        .def(nb::init<std::uint64_t, std::uint64_t, std::uint64_t>())
        .def(
            "reserve",
            [](random::State& s, std::uint64_t n) {
                const random::State::Taken t = s.reserve(n);
                return nb::make_tuple(nb::make_tuple(t.k0, t.k1), t.pos);
            }
        )
        .def(
            "get",
            [](random::State& s) {
                const random::State::Taken t = s.get();
                return nb::make_tuple(nb::make_tuple(t.k0, t.k1), t.pos);
            }
        )
        .def("set", &random::State::set);
    m.def("fisher_yates", &fisher_yates);
    m.def("philox_block", &philox_block);
}

}  // namespace sd
