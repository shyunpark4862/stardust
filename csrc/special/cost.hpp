// Per-element cost of each special function, for the parallel split
// (docs/design/special.md §3.3).
//
// The array's iterator goes parallel when the work reaches a threshold and
// cuts it into tasks of a fixed size, both counted in units of one add per
// element (array/parallel_limits.hpp). Special functions cost from a few to
// hundreds of adds per element, so each passes its weight -- its cost in
// adds -- and an expensive function splits at a smaller size, into tasks of
// fewer elements, the same way a lane kernel does (docs/design/array.md §4.5).
//
// The costs are serial times per element on this machine (Apple M4 Pro, one
// thread, inputs spread over the function's guaranteed region), measured by
// bench/special/cost.py. A factor of two either way changes only where the
// split starts, never a result. The values are not re-measured when a function
// gets faster: from 48.6 ns (kMaxWeight adds) up the weight is capped, so the
// exact value of a heavy function's constant has no effect.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sd::special {

// add on contiguous float64, per element (bench/special/cost.py).
inline constexpr double kAddNs = 0.19;  // measured 0.185 (2026-11-13)

// The weight NdIter::run takes: the cost in adds, at least 1 and at most
// kMaxWeight. NdIter cuts tasks of 4096 / weight elements: uncapped, the S4
// functions (weight 800-8500) got one element per task, and 10 threads
// contending for the pool's counter reached only 5.4-7.8x of one thread;
// capped at 256 (16-element tasks) 8.1-8.9x, at 64 about the same
// (bench/special/s4/threads.md). 256 keeps the parallel start at 128 elements.
inline constexpr std::int64_t kMaxWeight = 256;
constexpr std::int64_t weight_of(double ns) {
    const double w = ns / kAddNs;
    return w < 1.0 ? 1 : std::min<std::int64_t>(static_cast<std::int64_t>(w + 0.5), kMaxWeight);
}

namespace cost {
// The engine's test functions (special/ops/testing.hpp).
inline constexpr double kFma3 = 0.24;
inline constexpr double kSum4 = 0.30;
inline constexpr double kPowers4 = 0.45;
inline constexpr double kCfTan = 16;
// S1 (special/ops/logistic.hpp), bench/special/s1/cost.md
inline constexpr double kExpit = 1.3;
inline constexpr double kLogit = 3.5;
inline constexpr double kLogExpit = 14.5;
inline constexpr double kXlogy = 3.5;
inline constexpr double kXlog1py = 4.8;
inline constexpr double kEntr = 2.9;
inline constexpr double kRelEntr = 10.8;
// S2 (special/ops/erf.hpp), bench/special/s2/cost.md
inline constexpr double kErf = 5.4;
inline constexpr double kErfc = 5.6;
inline constexpr double kErfcx = 7.5;
inline constexpr double kErfinv = 7.2;
inline constexpr double kErfcinv = 8.7;
inline constexpr double kNdtr = 6.3;
inline constexpr double kLogNdtr = 14.6;
inline constexpr double kNdtri = 8.1;
// S3 (special/ops/gamma.hpp, zeta.hpp), bench/special/s3/cost.md
inline constexpr double kGamma = 18;
inline constexpr double kRgamma = 20;
inline constexpr double kGammaln = 11;
inline constexpr double kDigamma = 9;
inline constexpr double kPolygamma = 123;
inline constexpr double kBeta = 60;
inline constexpr double kBetaln = 32;  // log of Boost beta for a + b < 40 (S11)
inline constexpr double kBinom = 104;
inline constexpr double kZeta = 18;  // Boost for x > -60 (special/boost.hpp)
inline constexpr double kHurwitz = 87;
// S4 (special/ops/incgamma.hpp, incbeta.hpp), bench/special/s4/cost.md
inline constexpr double kGammainc = 146;
inline constexpr double kGammaincc = 146;
inline constexpr double kGammaincinv = 370;  // with Boost's regions (special/boost.hpp)
inline constexpr double kGammainccinv = 302;
inline constexpr double kBetainc = 401;
inline constexpr double kBetaincc = 399;
inline constexpr double kBetaincinv = 1025;
inline constexpr double kBetainccinv = 1172;
// T2's noncentral distributions: the bench/special/t2/speed.py mix, one thread (all capped at kMaxWeight)
inline constexpr double kChndtr = 940, kChndtrc = 870, kChndtrix = 6500;
inline constexpr double kNctdtr = 3300, kNctdtrc = 3300, kNctdtrit = 21000;
inline constexpr double kNcfdtr = 2600, kNcfdtrc = 1800, kNcfdtri = 16000;
// S5 (special/ops/expint.hpp, ellip.hpp), bench/special/s5/cost.md
inline constexpr double kExpi = 12;  // Boost for x >= -700
inline constexpr double kExpn = 98;
inline constexpr double kEllipk = 8.0;
inline constexpr double kEllipe = 9.7;
inline constexpr double kEllipkAgm = 30;   // the comparison variants
inline constexpr double kEllipeCarlson = 65;
// S6 (special/ops/bessel01.hpp, bessel.hpp, airy.hpp), bench/special/s6/cost.md
inline constexpr double kI0 = 4.5, kI0e = 11, kI1 = 4.7, kI1e = 12;  // Boost (and its value times e^-+x, S11)
inline constexpr double kK0 = 11, kK0e = 13.5, kK1 = 11, kK1e = 13.5;
inline constexpr double kJ0 = 11, kJ1 = 12, kY0 = 14, kY1 = 13;
inline constexpr double kJv = 750, kYv = 750, kIv = 222, kKv = 151;  // the reference mix: |v| up to 1000 (Debye from 30, S11)
inline constexpr double kAiry = 270;
// S7 (special/ops/lambertw.hpp, hyper.hpp), bench/special/s7/cost.md
inline constexpr double kLambertw = 11, kLambertwM1 = 18;  // Boost away from the branch point
inline constexpr double kHyp1f1 = 1440, kHyp2f1 = 2200;  // the reference mix (weight capped at kMaxWeight)
// S8 (special/ops/wofz.hpp, loggamma.hpp, lambertw.hpp), bench/special/s8/cost.md
inline constexpr double kWofz = 57, kErfC = 47, kErfcC = 57, kErfcxC = 47;  // the reference mix
inline constexpr double kLoggammaC = 111, kLambertwC = 130;  // (all capped at kMaxWeight)
}  // namespace cost

}  // namespace sd::special
