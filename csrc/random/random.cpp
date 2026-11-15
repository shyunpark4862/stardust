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

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sd {

namespace {

using random::Key;
using random::Stream;

// Work per element, for the pool's threshold: a Philox block is ~10 ns and serves four.
inline constexpr std::int64_t kWordWork = 8;

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

// [low, high) (or [low, high] with endpoint) for int64 arrays low and high of the
// result's shape (any strides: broadcast views). The checks (low < high) are the
// caller's; here low + bounded(high - low) in unsigned arithmetic, which is exact
// for any int64 pair with low <= high.
nb::object integers(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, const NDArray& lo, const NDArray& hi,
                    bool endpoint) {
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
            for (std::int64_t i = i0; i < i1; ++i) {
                const std::uint64_t e = pos + static_cast<std::uint64_t>(i);
                const std::uint64_t w = s.first(e);
                std::uint64_t v;
                if (r == 0) {
                    v = w;
                } else {
                    unsigned __int128 m = static_cast<unsigned __int128>(w) * r;
                    for (std::uint64_t k = 0; static_cast<std::uint64_t>(m) < t; ++k)  // rare: Lemire's rejection
                        m = static_cast<unsigned __int128>(s.extra(e, k)) * r;
                    v = static_cast<std::uint64_t>(m >> 64);
                }
                o[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(l) + v);
            }
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
nb::object race_keys(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, const NDArray& p) {
    NDArray out = new_array(DType::Float64, p.shape_span());
    double* o = reinterpret_cast<double*>(out.data);
    const double* pp = reinterpret_cast<const double*>(p.data);
    const std::int64_t n = out.size();
    for_pieces(n, kParallelTask, n * kWordWork, [&](std::int64_t, std::int64_t i0, std::int64_t i1) {
        Stream s({k0, k1});
        for (std::int64_t i = i0; i < i1; ++i) {
            const double u = (static_cast<double>(s.first(pos + static_cast<std::uint64_t>(i)) >> 11) + 1.0) * 0x1p-53;
            o[i] = pp[i] > 0.0 ? -std::log(u) / pp[i] : HUGE_VAL;
        }
    });
    return wrap(std::move(out));
}

// The first k entries of a Fisher-Yates shuffle of 0 .. n-1 (k <= n): step i swaps
// entry i with entry i + bounded(n - i), its draw from element pos + i of the stream
// (extra words for rejections), so the result depends on (key, pos, n, k) only. For a
// small sample of a large population the entries touched live in a hash map (O(k)
// time and memory; numpy's choice without replacement does the same in effect); else
// in the full array. k = n is a permutation.
nb::object fisher_yates(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos, std::int64_t n, std::int64_t k) {
    const std::int64_t shape[] = {k};
    NDArray out = new_array(DType::Int64, shape);
    auto* o = reinterpret_cast<std::int64_t*>(out.data);
    Stream s({k0, k1});
    auto draw = [&](std::int64_t i) {
        const std::uint64_t e = pos + static_cast<std::uint64_t>(i);
        return i + static_cast<std::int64_t>(
                       random::bounded(s.first(e), static_cast<std::uint64_t>(n - i), [&](std::uint64_t t) { return s.extra(e, t); }));
    };
    {
    nb::gil_scoped_release release;  // the loop only: wrap() below needs the GIL
    if (k <= n / 16) {
        std::unordered_map<std::int64_t, std::int64_t> moved;  // entries that are not their own index
        moved.reserve(static_cast<std::size_t>(2 * k));
        auto at = [&](std::int64_t i) {
            const auto it = moved.find(i);
            return it == moved.end() ? i : it->second;
        };
        for (std::int64_t i = 0; i < k; ++i) {
            const std::int64_t j = draw(i);
            const std::int64_t vi = at(i), vj = at(j);
            moved[j] = vi;
            o[i] = vj;
        }
    } else {
        std::vector<std::int64_t> a(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) a[static_cast<std::size_t>(i)] = i;
        for (std::int64_t i = 0; i < k; ++i) std::swap(a[static_cast<std::size_t>(i)], a[static_cast<std::size_t>(draw(i))]);
        std::copy(a.begin(), a.begin() + k, o);
    }
    }
    return wrap(std::move(out));
}

// One Philox block, for the known-answer tests.
nb::tuple philox_block(std::uint64_t c0, std::uint64_t c1, std::uint64_t c2, std::uint64_t c3, std::uint64_t k0,
                       std::uint64_t k1) {
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
    m.def("fisher_yates", &fisher_yates);
    m.def("philox_block", &philox_block);
}

}  // namespace sd
