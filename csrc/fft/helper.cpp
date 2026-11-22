// The helpers of sd.fft that are computed natively (docs/api/fft.md §5; docs/design/fft.md §2.7;
// milestone FT1). The Python layer (src/stardust/fft/_helper.py) checks the arguments.
//
//   _freq(n, val, real)          fftfreq / rfftfreq: k * val for numpy's k (0, 1, ..., -(n // 2), ..., -1;
//                                real: 0 .. n // 2), each k converted exactly and multiplied once:
//                                numpy's own formula, so the same bits (decision FT26)
//   _fast_len(target, real, prev) next_fast_len / prev_fast_len: the smallest 11-smooth (real: 5-smooth)
//                                number >= target, or the largest <= target (prev); -1 when the next one
//                                is beyond int64 (decision FT18)
#include <cstdint>
#include <vector>

#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "fft/fft.hpp"

namespace nb = nanobind;

namespace sd::fft {

namespace {

nb::object freq(int64_t n, double val, bool real) {
    const int64_t count = real ? n / 2 + 1 : n;
    NDArray out = new_array(DType::Float64, std::vector<int64_t>{count});
    double* d = reinterpret_cast<double*>(out.data);
    if (real) {
        for (int64_t k = 0; k < count; ++k) d[k] = static_cast<double>(k) * val;
    } else {
        const int64_t half = (n - 1) / 2 + 1;  // entries 0 .. half - 1 are k, the rest k - n
        for (int64_t k = 0; k < half; ++k) d[k] = static_cast<double>(k) * val;
        for (int64_t k = half; k < n; ++k) d[k] = static_cast<double>(k - n) * val;
    }
    return wrap(std::move(out));
}

using u128 = unsigned __int128;

int64_t fast_len(int64_t target, bool real, bool prev) {
    const u128 t = static_cast<u128>(target);
    const u128 limit = static_cast<u128>(INT64_MAX);
    const int primes_c[] = {3, 5, 7, 11};
    const int nprimes = real ? 2 : 4;
    u128 best = prev ? 0 : ~u128{0};
    // every product of the odd primes up to (prev: target; next: 2 target), then times powers of 2
    std::vector<u128> odd{1};
    const u128 cap = prev ? t : 2 * t;
    for (int i = 0; i < nprimes; ++i) {
        const u128 p = static_cast<u128>(primes_c[i]);
        const size_t have = odd.size();
        for (size_t j = 0; j < have; ++j) {
            for (u128 v = odd[j] * p; v <= cap; v *= p) odd.push_back(v);
        }
    }
    for (u128 q : odd) {
        if (prev) {
            u128 v = q;
            if (v > t) continue;
            while (2 * v <= t) v *= 2;
            if (v > best) best = v;
        } else {
            u128 v = q;
            while (v < t) v *= 2;
            if (v < best) best = v;
        }
    }
    if (best > limit) return -1;
    return static_cast<int64_t>(best);
}

}  // namespace

void bind_helper(nb::module_& m) {
    m.def("_freq", &freq, nb::arg("n"), nb::arg("val"), nb::arg("real"));
    m.def("_fast_len", &fast_len, nb::arg("target"), nb::arg("real"), nb::arg("prev"));
}

}  // namespace sd::fft
