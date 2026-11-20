// sin x and cos x of the same x in one call (Apple's __sincos_stret, one
// argument reduction for both). The build keeps -fno-builtin-sin/-cos, so the
// compiler never merges the two by itself (where it did depended on inlining,
// and a function could give other bits in another path: f(x)[i] == f(x[i]));
// the merge is written here, the same function on every path.
#pragma once

#include <cmath>
#include <math.h>

namespace sd {

struct SinCos {
    double s, c;
};

inline SinCos sincos(double x) {
#if defined(__APPLE__)
    const struct __double2 r = __sincos_stret(x);
    return {r.__sinval, r.__cosval};
#else
    return {std::sin(x), std::cos(x)};
#endif
}

}  // namespace sd
