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
// thread, 10^6 elements over the function's reference inputs), from the table
// special/cost_table.hpp that `python -m bench.speed cost` writes from the
// special baseline (bench/speed/results/special.json; bench/README.md). A name
// without a row does not compile. A factor of two either way changes only where
// the split starts, never a result; from 48.6 ns (kMaxWeight adds) up the weight
// is capped, so the exact value of a heavy function's constant has no effect.
// The constants below map each operation to the measured function it costs like.
#pragma once

#include "special/cost_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sd::special {

// add on contiguous float64, per element.
inline constexpr double kAddNs = cost_table::ns("_add");

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
inline constexpr double kFma3 = cost_table::ns("_fma3");
inline constexpr double kSum4 = cost_table::ns("_sum4");
inline constexpr double kPowers4 = cost_table::ns("_powers4");
inline constexpr double kCfTan = cost_table::ns("_cf_tan");
// S1 (special/ops/logistic.hpp)
inline constexpr double kExpit = cost_table::ns("expit");
inline constexpr double kLogit = cost_table::ns("logit");
inline constexpr double kLogExpit = cost_table::ns("log_expit");
inline constexpr double kXlogy = cost_table::ns("xlogy");
inline constexpr double kXlog1py = cost_table::ns("xlog1py");
inline constexpr double kEntr = cost_table::ns("entr");
inline constexpr double kRelEntr = cost_table::ns("rel_entr");
// S2 (special/ops/erf.hpp)
inline constexpr double kErf = cost_table::ns("erf");
inline constexpr double kErfc = cost_table::ns("erfc");
inline constexpr double kErfcx = cost_table::ns("erfcx");
inline constexpr double kErfinv = cost_table::ns("erfinv");
inline constexpr double kErfcinv = cost_table::ns("erfcinv");
inline constexpr double kNdtr = cost_table::ns("ndtr");
inline constexpr double kLogNdtr = cost_table::ns("log_ndtr");
inline constexpr double kNdtri = cost_table::ns("ndtri");
// S3 (special/ops/gamma.hpp, zeta.hpp)
inline constexpr double kGamma = cost_table::ns("gamma");
inline constexpr double kRgamma = cost_table::ns("rgamma");
inline constexpr double kGammaln = cost_table::ns("gammaln");
inline constexpr double kDigamma = cost_table::ns("digamma");
inline constexpr double kPolygamma = cost_table::ns("polygamma");
inline constexpr double kBeta = cost_table::ns("beta");
inline constexpr double kBetaln = cost_table::ns("betaln");  // log of Boost beta for a + b < 40 (S11)
inline constexpr double kBinom = cost_table::ns("binom");
inline constexpr double kZeta = cost_table::ns("zeta");  // Boost for x > -60 (special/boost.hpp)
inline constexpr double kHurwitz = cost_table::ns("zeta_hurwitz");
// S4 (special/ops/incgamma.hpp, incbeta.hpp)
inline constexpr double kGammainc = cost_table::ns("gammainc");
inline constexpr double kGammaincc = cost_table::ns("gammaincc");
inline constexpr double kGammaincinv = cost_table::ns("gammaincinv");  // with Boost's regions (special/boost.hpp)
inline constexpr double kGammainccinv = cost_table::ns("gammainccinv");
inline constexpr double kBetainc = cost_table::ns("betainc");
inline constexpr double kBetaincc = cost_table::ns("betaincc");
inline constexpr double kBetaincinv = cost_table::ns("betaincinv");
inline constexpr double kBetainccinv = cost_table::ns("betainccinv");
// T2's noncentral distributions (all capped at kMaxWeight)
inline constexpr double kChndtr = cost_table::ns("chndtr"), kChndtrc = cost_table::ns("chndtrc"), kChndtrix = cost_table::ns("chndtrix");
inline constexpr double kNctdtr = cost_table::ns("nctdtr"), kNctdtrc = cost_table::ns("nctdtrc"), kNctdtrit = cost_table::ns("nctdtrit");
inline constexpr double kNcfdtr = cost_table::ns("ncfdtr"), kNcfdtrc = cost_table::ns("ncfdtrc"), kNcfdtri = cost_table::ns("ncfdtri");
// S5 (special/ops/expint.hpp, ellip.hpp)
inline constexpr double kExpi = cost_table::ns("expi");  // Boost for x >= -700
inline constexpr double kExpn = cost_table::ns("expn");
inline constexpr double kEllipk = cost_table::ns("ellipk");
inline constexpr double kEllipe = cost_table::ns("ellipe");
inline constexpr double kEllipkAgm = 30;   // the comparison variants (not in the speed cases)
inline constexpr double kEllipeCarlson = 65;
// S6 (special/ops/bessel01.hpp, bessel.hpp, airy.hpp)
inline constexpr double kI0 = cost_table::ns("i0"), kI0e = cost_table::ns("i0e"), kI1 = cost_table::ns("i1"), kI1e = cost_table::ns("i1e");  // Boost (and its value times e^-+x, S11)
inline constexpr double kK0 = cost_table::ns("k0"), kK0e = cost_table::ns("k0e"), kK1 = cost_table::ns("k1"), kK1e = cost_table::ns("k1e");
inline constexpr double kJ0 = cost_table::ns("j0"), kJ1 = cost_table::ns("j1"), kY0 = cost_table::ns("y0"), kY1 = cost_table::ns("y1");
inline constexpr double kJv = cost_table::ns("jv"), kYv = cost_table::ns("yv"), kIv = cost_table::ns("iv"), kKv = cost_table::ns("kv");  // the reference mix: |v| up to 1000 (Debye from 30, S11)
inline constexpr double kAiry = cost_table::ns("airy_ai");
// S7 (special/ops/lambertw.hpp, hyper.hpp)
inline constexpr double kLambertw = cost_table::ns("lambertw"), kLambertwM1 = cost_table::ns("lambertw_m1");  // Boost away from the branch point
inline constexpr double kHyp1f1 = cost_table::ns("hyp1f1"), kHyp2f1 = cost_table::ns("hyp2f1");  // the reference mix (weight capped at kMaxWeight)
// S8 (special/ops/wofz.hpp, loggamma.hpp, lambertw.hpp)
inline constexpr double kWofz = cost_table::ns("wofz"), kErfC = cost_table::ns("erf_complex"), kErfcC = cost_table::ns("erfc_complex"), kErfcxC = cost_table::ns("erfcx_complex");  // the reference mix
inline constexpr double kLoggammaC = cost_table::ns("loggamma_complex"), kLambertwC = cost_table::ns("lambertw_c0");  // (all capped at kMaxWeight)
}  // namespace cost

}  // namespace sd::special
