// PCHIP's node slope (scipy's PchipInterpolator, Fritsch-Butland) that the 1-D PchipInterpolator (cubic.cpp) and the
// grid's local pchip (grid.cpp) share, so that both compute every slope with the same operations: the 1-D slopes
// are scipy's bits (decision IP42), and a grid's pchip is the 1-D interpolant along each axis (§3.7).
//
//   interior   the weighted harmonic mean 1 / ((w1 / ml + w2 / mr) / (w1 + w2)), w1 = 2 hr + hl, w2 = hr + 2 hl,
//              0 when the two neighbouring slopes ml, mr differ in sign or one is 0
//   edge       the one-sided three-point estimate ((2 h0 + h1) m0 - h0 m1) / (h0 + h1), kept monotone
#pragma once

#include <cmath>

namespace sd::interpolate {

inline int pchip_sgn(double v) { return (v > 0.0) - (v < 0.0); }

// a node between two intervals: their slopes ml (left), mr (right) and the weights w1, w2 of their widths
inline double pchip_interior(double w1, double w2, double ml, double mr) {
    if (pchip_sgn(mr) != pchip_sgn(ml) || mr == 0.0 || ml == 0.0) return 0.0;
    const double whmean = (w1 / ml + w2 / mr) / (w1 + w2);
    return 1.0 / whmean;
}

// scipy's PchipInterpolator._edge_case: an end node, h0 and m0 its interval's width and slope, h1 and m1 the next's
inline double pchip_edge(double h0, double h1, double m0, double m1) {
    const double d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if (pchip_sgn(d) != pchip_sgn(m0)) return 0.0;
    if (pchip_sgn(m0) != pchip_sgn(m1) && std::fabs(d) > 3.0 * std::fabs(m0)) return 3.0 * m0;
    return d;
}

}  // namespace sd::interpolate
