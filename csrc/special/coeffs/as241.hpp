// Rational approximations of the normal quantile, PPND16.
//
// Source: M. J. Wichura (1988), "Algorithm AS 241: The Percentage Points of
// the Normal Distribution", Applied Statistics 37(3), 477-484. The decimal
// values are the paper's, as transcribed in R's src/nmath/qnorm.c (read
// 2026-11-14) -- published constants of an algorithm, copied unchanged.
// Highest power first; every denominator ends in the constant term 1.
//
// With q = p - 1/2 and s = sqrt(-log(min(p, 1 - p))):
//   |q| <= 0.425:  q A(r) / B(r),   r = 0.425^2 - q^2
//   s <= 5:        C(s - 1.6) / D(s - 1.6)
//   s > 5:         E(s - 5) / F(s - 5)   (fitted for s <= 27, i.e.
//                                          min(p, 1 - p) >= 2.5e-317)
// negated for q < 0. Wichura: "accurate to about 1 part in 10^16".
#pragma once

#include <array>

namespace sd::special::as241 {

inline constexpr double kSplit1 = 0.425;
inline constexpr double kConst1 = 0.180625;  // 0.425^2
inline constexpr double kSplit2 = 5.0;
inline constexpr double kConst2 = 1.6;

inline constexpr std::array<double, 8> kA = {
    2509.0809287301226727, 33430.575583588128105, 67265.770927008700853, 45921.953931549871457,
    13731.693765509461125, 1971.5909503065514427, 133.14166789178437745, 3.387132872796366608};
inline constexpr std::array<double, 8> kB = {
    5226.495278852854561, 28729.085735721942674, 39307.89580009271061, 21213.794301586595867,
    5394.1960214247511077, 687.1870074920579083, 42.313330701600911252, 1.0};
inline constexpr std::array<double, 8> kC = {
    7.7454501427834140764e-4, .0227238449892691845833, .24178072517745061177, 1.27045825245236838258,
    3.64784832476320460504, 5.7694972214606914055, 4.6303378461565452959, 1.42343711074968357734};
inline constexpr std::array<double, 8> kD = {
    1.05075007164441684324e-9, 5.475938084995344946e-4, .0151986665636164571966, .14810397642748007459,
    .68976733498510000455, 1.6763848301838038494, 2.05319162663775882187, 1.0};
inline constexpr std::array<double, 8> kE = {
    2.01033439929228813265e-7, 2.71155556874348757815e-5, .0012426609473880784386, .026532189526576123093,
    .29656057182850489123, 1.7848265399172913358, 5.4637849111641143699, 6.6579046435011037772};
inline constexpr std::array<double, 8> kF = {
    2.04426310338993978564e-15, 1.4215117583164458887e-7, 1.8463183175100546818e-5, 7.868691311456132591e-4,
    .0148753612908506148525, .13692988092273580531, .59983220655588793769, 1.0};

}  // namespace sd::special::as241
