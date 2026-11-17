// Rational approximations of erf and erfc from Cephes ndtr.c.
//
// Source: Cephes Math Library Release 2.9 (June 2000), ndtr.c, Copyright 1984,
// 1987, 1988, 1992, 2000 by Stephen L. Moshier, as distributed in SciPy
// (BSD 3-clause; scipy/xsf include/xsf/cephes/ndtr.h, read 2026-11-14). The
// decimal values are copied unchanged. Highest power first (polevl order);
// the denominators Q, S, U have an implicit leading 1 (p1evl).
//
//   erf(x)  = x T(x^2) / U(x^2)                   |x| < 1
//   erfc(x) = exp(-x^2) P(x) / Q(x)               1 <= x < 8
//   erfc(x) = exp(-x^2) R(x) / S(x)               x >= 8
//
// Moshier's stated accuracy (IEEE, peak relative error): erf 3.7e-16 on
// [0, 1]; erfc 5.7e-14 on [0, 26.6417], his exp(-x^2) included.
#pragma once

#include <array>

namespace sd::special::cephes {

inline constexpr std::array<double, 9> kNdtrP = {
    2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0,
    4.86371970985681366614E1,   1.96520832956077098242E2,  5.26445194995477358631E2,
    9.34528527171957607540E2,   1.02755188689515710272E3,  5.57535335369399327526E2};
inline constexpr std::array<double, 8> kNdtrQ = {
    1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2, 9.75708501743205489753E2,
    1.82390916687909736289E3, 2.24633760818710981792E3, 1.65666309194161350182E3, 5.57535340817727675546E2};
inline constexpr std::array<double, 6> kNdtrR = {
    5.64189583547755073984E-1, 1.27536670759978104416E0, 5.01905042251180477414E0,
    6.16021097993053585195E0,  7.40974269950448939160E0, 2.97886665372100240670E0};
inline constexpr std::array<double, 6> kNdtrS = {
    2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1,
    1.70814450747565897222E1, 9.60896809063285878198E0, 3.36907645100081516050E0};
inline constexpr std::array<double, 5> kNdtrT = {
    9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3, 7.00332514112805075473E3,
    5.55923013010394962768E4};
inline constexpr std::array<double, 5> kNdtrU = {
    3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3, 2.26290000613890934246E4,
    4.92673942608635921086E4};

}  // namespace sd::special::cephes
