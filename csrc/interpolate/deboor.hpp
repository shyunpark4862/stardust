// The B-spline basis that sd.interpolate's B-splines (bspline.cpp) and the grid's tensor-product splines (grid.cpp)
// share (docs/design/interpolate.md §3.4, §3.7): the knot interval of a point and de Boor's recursion for the k + 1
// basis functions there, one code so that a 1-D grid spline gives a B-spline's bits.
//
//   knot_interval  the l in [k, n - 1] with t[l] <= x < t[l + 1] (x >= t[n]: n - 1; x < t[k]: k) from the count of
//                  the knots at or below x (base/interval.hpp)
//   basis_values   B_{l-k} .. B_l of degree k at x, or their derivatives of order nu: scipy's recursion (_deBoor_D)
//                  with the weights as two quotients (t[l+m] - x) / (t[l+m] - t[l+m-j]) and (x - t[l+m-j]) / (...),
//                  each in [0, 1] inside the base interval, so every step is a convex combination and no step
//                  overflows; the derivative steps are scipy's, w = j h / (t[l+m] - t[l+m-j])
//   basis_fixed    the same for a degree known at compile time and nu = 0, unrolled (the same operations in the same
//                  order, so the same bits)
#pragma once

#include <algorithm>
#include <cstdint>

namespace sd::interpolate {

// The interval of x (not NaN) from its count of knots at or below it.
inline std::int64_t knot_interval(std::int64_t count, std::int64_t k, std::int64_t n) {
    return std::clamp<std::int64_t>(count - 1, k, n - 1);
}


// h[0 .. k] = B_{l-k+r}^{(nu)}(x) for nu <= k; hh: k + 1 doubles of workspace.
inline void basis_values(
    const double* t, std::int64_t k, std::int64_t l, double x, std::int64_t nu, double* h, double* hh
) {
    h[0] = 1.0;
    for (std::int64_t j = 1; j <= k - nu; ++j) {
        std::copy(h, h + j, hh);
        h[0] = 0.0;
        for (std::int64_t m = 1; m <= j; ++m) {
            const double xb = t[l + m], xa = t[l + m - j];
            if (xb == xa) {
                h[m] = 0.0;
                continue;
            }
            const double d = xb - xa;
            const double left = (xb - x) / d, right = (x - xa) / d;
            h[m - 1] = h[m - 1] + hh[m - 1] * left;
            h[m] = hh[m - 1] * right;
        }
    }
    for (std::int64_t j = k - nu + 1; j <= k; ++j) {
        std::copy(h, h + j, hh);
        h[0] = 0.0;
        for (std::int64_t m = 1; m <= j; ++m) {
            const double xb = t[l + m], xa = t[l + m - j];
            if (xb == xa) {
                h[m] = 0.0;
                continue;
            }
            const double w = static_cast<double>(j) * hh[m - 1] / (xb - xa);
            h[m - 1] = h[m - 1] - w;
            h[m] = w;
        }
    }
}

// basis_values for a degree K known at compile time and nu = 0: the same operations in the same order (so the
// same bits), unrolled (evaluation's common case; the loops over a runtime k were 2x its time).
template <int K>
inline void basis_fixed(const double* t, std::int64_t l, double x, double* h) {
    h[0] = 1.0;
    for (int j = 1; j <= K; ++j) {
        double hh[K];
        for (int m = 0; m < j; ++m) hh[m] = h[m];
        h[0] = 0.0;
        for (int m = 1; m <= j; ++m) {
            const double xb = t[l + m], xa = t[l + m - j];
            if (xb == xa) {
                h[m] = 0.0;
                continue;
            }
            const double d = xb - xa;
            const double left = (xb - x) / d, right = (x - xa) / d;
            h[m - 1] = h[m - 1] + hh[m - 1] * left;
            h[m] = hh[m - 1] * right;
        }
    }
}


}  // namespace sd::interpolate
