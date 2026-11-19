// The steps of the difference Jacobians in the least squares iterations
// (docs/design/optimize.md §3.1): scipy 1.18.1's approx_derivative ('2-point', '3-point';
// scipy/optimize/_numdiff.py: _compute_absolute_step and _adjust_scheme_to_bounds) and
// MINPACK's fdjac2 step (lmdif, regression.nls's iteration).
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace sd::optimize {

// sqrt(eps) = 2^-26: the relative step of fdjac2 (epsfcn = eps) and of scipy's '2-point'.
inline double rel_step_2point() { return std::sqrt(std::numeric_limits<double>::epsilon()); }
// eps^(1/3): scipy's '3-point'.
inline double rel_step_3point() { return std::pow(std::numeric_limits<double>::epsilon(), 1.0 / 3.0); }

// scipy's _compute_absolute_step for one variable: rstep sign(x) max(1, |x|) by default
// (sign(0) = 1); with a relative step r, r sign(x) |x|, unless that step vanishes in x + h.
inline double absolute_step(double x, double rstep, bool has_rel, double rel) {
    const double sign = x >= 0 ? 1.0 : -1.0;
    const double dflt = rstep * sign * std::max(1.0, std::abs(x));
    if (!has_rel) return dflt;
    const double h = rel * sign * std::abs(x);
    return (x + h) - x == 0 ? dflt : h;
}

// scipy's _adjust_scheme_to_bounds, '1-sided' with one step: the step turned back when
// it leaves [lb, ub] and fits on the other side, else the distance to the farther bound.
inline double adjust_one_sided(double x, double h, double lb, double ub) {
    const double lower = x - lb, upper = ub - x;
    const double xh = x + h;
    const bool violated = xh < lb || xh > ub;
    const bool fitting = std::abs(h) <= std::max(lower, upper);
    if (violated && fitting) return -h;
    if (!fitting) return upper >= lower ? upper : -lower;
    return h;
}

// The same, '2-sided' with one step (scipy's '3-point'): central when both x - h and
// x + h are inside, else one-sided (x + h, x + 2h) towards the farther bound with h at most
// half that distance, back to central with the smaller distance when it is no shorter.
// Returns the step; one_sided tells the stencil.
inline double adjust_two_sided(double x, double h, double lb, double ub, bool& one_sided) {
    h = std::abs(h);
    one_sided = false;
    const double lower = x - lb, upper = ub - x;
    if (lower >= h && upper >= h) return h;
    double ha;
    if (upper >= lower) ha = std::min(h, 0.5 * upper);
    else ha = -std::min(h, 0.5 * lower);
    one_sided = true;
    const double min_dist = std::min(upper, lower);
    if (std::abs(ha) <= min_dist) {
        ha = min_dist;
        one_sided = false;
    }
    return ha;
}

}  // namespace sd::optimize
