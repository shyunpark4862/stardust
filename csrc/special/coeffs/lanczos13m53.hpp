// The Lanczos approximation lanczos13m53 (gamma candidate A, docs/design/special.md §5.2).
//
// Source: Boost.Math 1.8x, include/boost/math/special_functions/lanczos.hpp,
// struct lanczos13m53 (John Maddock, Paul Bristow; Boost Software License
// 1.0; read 2026-11-14). Decimal values copied unchanged, lowest power first
// (Boost's order). With g = kG,
//     Gamma(z) = L(z) (z + g - 1/2)^(z - 1/2) e^-(z + g - 1/2),  z > 0
// where L(z) = sum num[i] z^i / sum den[i] z^i, den = z(z+1)...(z+11).
#pragma once

#include <array>

namespace sd::special::lanczos {

inline constexpr double kG = 6.024680040776729583740234375;
inline constexpr std::array<double, 13> kNum = {
    23531376880.41075968857200767445163675473, 42919803642.64909876895789904700198885093,
    35711959237.35566804944018545154716670596, 17921034426.03720969991975575445893111267,
    6039542586.35202800506429164430729792107,  1439720407.311721673663223072794912393972,
    248874557.8620541565114603864132294232163, 31426415.58540019438061423162831820536287,
    2876370.628935372441225409051620849613599, 186056.2653952234950402949897160456992822,
    8071.672002365816210638002902272250613822, 210.8242777515793458725097339207133627117,
    2.506628274631000270164908177133837338626};
inline constexpr std::array<double, 13> kDen = {0.0,      39916800.0, 120543840.0, 150917976.0, 105258076.0,
                                                45995730.0, 13339535.0, 2637558.0,   357423.0,    32670.0,
                                                1925.0,     66.0,       1.0};

}  // namespace sd::special::lanczos
