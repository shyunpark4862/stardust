// Gamma candidate B and the Stirling and digamma series (docs/design/special.md §5.2).
//
// Source: Cephes Math Library 2.9, gamma.c and psi.c (Stephen L. Moshier),
// as distributed in SciPy (BSD 3-clause; scipy/xsf include/xsf/cephes/gamma.h,
// psi.h, read 2026-11-14), and the digamma rational on [1, 2] that SciPy took
// from Boost.Math (digamma.hpp, Boost Software License 1.0). Decimal values
// copied unchanged; highest power first (polevl).
#pragma once

#include <array>

namespace sd::special::cephes {

// Gamma(2 + x) = P(x) / Q(x) on [0, 1)
inline constexpr std::array<double, 7> kGammaP = {
    1.60119522476751861407E-4, 1.19135147006586384913E-3, 1.04213797561761569935E-2, 4.76367800457137231464E-2,
    2.07448227648435975150E-1, 4.94214826801497100753E-1, 9.99999999999999996796E-1};
inline constexpr std::array<double, 8> kGammaQ = {
    -2.31581873324120129819E-5, 5.39605580493303397842E-4, -4.45641913851797240494E-3, 1.18139785222060435552E-2,
    3.58236398605498653373E-2,  -2.34591795718243348568E-1, 7.14304917030273074085E-2,  1.00000000000000000320E0};
// Stirling's series for Gamma, x >= 33: 1 + STIR(1/x) / x
inline constexpr std::array<double, 5> kGammaStir = {7.87311395793093628397E-4, -2.29549961613378126380E-4,
                                                      -2.68132617805781232825E-3, 3.47222221605458667310E-3,
                                                      8.33333333333482257126E-2};
inline constexpr double kMaxStir = 143.01608;
// Stirling's series for log Gamma, x >= 13: A(1/x^2) / x
inline constexpr std::array<double, 5> kLgamA = {8.11614167470508450300E-4, -5.95061904284301438324E-4,
                                                  7.93650340457716943945E-4, -2.77777777730099687205E-3,
                                                  8.33333333333331927722E-2};
// digamma's asymptotic series, x >= 10: A(1/x^2) / x^2
inline constexpr std::array<double, 7> kPsiA = {8.33333333333333333333E-2,  -2.10927960927960927961E-2,
                                                 7.57575757575757575758E-3,  -4.16666666666666666667E-3,
                                                 3.96825396825396825397E-3,  -8.33333333333333333333E-3,
                                                 8.33333333333333333333E-2};
// digamma on [1, 2] (Boost): (x - root) (Y + P(x - 1) / Q(x - 1)), root in three parts
inline constexpr double kPsiY = 0.99558162689208984;  // a float in the source: exactly this double
inline constexpr double kPsiRoot1 = 1569415565.0 / 1073741824.0;
inline constexpr double kPsiRoot2 = (381566830.0 / 1073741824.0) / 1073741824.0;
inline constexpr double kPsiRoot3 = 0.9016312093258695918615325266959189453125e-19;
inline constexpr std::array<double, 6> kPsiP = {-0.0020713321167745952, -0.045251321448739056, -0.28919126444774784,
                                                 -0.65031853770896507,   -0.32555031186804491,  0.25479851061131551};
inline constexpr std::array<double, 7> kPsiQ = {-0.55789841321675513e-6, 0.0021284987017821144, 0.054151797245674225,
                                                 0.43593529692665969,    1.4606242909763515,    2.0767117023730469,
                                                 1.0};
// Euler-Maclaurin coefficients (2k)! / B_2k for zeta (zeta.c)
inline constexpr std::array<double, 12> kZetaA = {12.0,
                                                  -720.0,
                                                  30240.0,
                                                  -1209600.0,
                                                  47900160.0,
                                                  -1.8924375803183791606e9,
                                                  7.47242496e10,
                                                  -2.950130727918164224e12,
                                                  1.1646782814350067249e14,
                                                  -4.5979787224074726105e15,
                                                  1.8152105401943546773e17,
                                                  -7.1661652561756670113e18};

}  // namespace sd::special::cephes
