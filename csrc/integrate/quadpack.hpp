// QUADPACK's adaptive routines of quad (docs/design/integrate.md §3.1, decision C5): without a
// weight dqagse (QAGS, a finite range), dqagie (QAGI, an infinite range mapped onto (0, 1]) and
// dqagpe (QAGP, break points), with their parts dqk21, dqk15i, dqpsrt and dqelg; with a weight
// (milestone C5) dqawoe (QAWO, cos or sin of omega x on a finite range), dqawfe (QAWF, on
// [a, inf) by cycles), dqawse (QAWS, the algebraic-logarithmic end point weights) and dqawce
// (QAWC, Cauchy's principal value), with their rules dqk15w, dqc25f, dqc25s, dqc25c, the
// Chebyshev series dqcheb, the moments dqmomo and LINPACK's dgtsl. Each is a statement by
// statement translation of the netlib Fortran original (Piessens, de Doncker-Kapenga, Ueberhuber,
// Kahaner 1983), with its control flow (the labels are the original's), its order of operations
// and its decimal literals, so that the points where the integrand is called and the results are
// scipy 1.18.1's (its C port of the same Fortran) bit for bit. Where scipy's port differs from
// the original and the tests follow it, the routine says so (dqc25s, dgtsl, dqawoe; decisions
// C75, C76, C79); where it is wrong, the original stands (dqc25f's row m, dqawfe at omega = 0;
// C77, C78).
//
// scipy's C port is built with clang's default -ffp-contract=on, which fuses a * b + c within one
// expression into a fused multiply-add; this build has contraction off (API.md §4, D-33), so the
// fused operations are written out with std::fma at those places: the Kronrod rules' running
// sums (resg, resk, resabs, resasc: s = s + w * v), and in the weighted rules every product that
// meets a sum in one expression of the original: dqcheb's combinations, dqc25f's points
// (centr -+ hlgth x(i) are one expression there, unlike dqk21's), moments and sums, dgtsl's
// elimination and back substitution, dqmomo's recurrences, the series sums of dqc25s and dqc25c,
// dqawfe's abserr + 10 correc. The operand clang fuses is the left product when both sides are
// products. Nothing else is fused (no product meets a sum within one expression in the drivers,
// dqelg or dqk21's points). docs/design/integrate.md §7.10 and §7.11 have the comparisons with
// scipy's records and the ablations that settle this.
//
// The arrays are 1-based like the original's (element 0 unused). The integrand is any callable
// double(double); a C++ exception from it (a Python error) leaves the routine by unwinding.
//
// Each rule first lists its points in the order it calls the integrand, computed by the very
// expressions of the calls, and hands them to prefetch(f, xs, n): an integrand with a member
// prefetch (quad's vectorized=True, decision C127) evaluates them in one call and serves the
// rule's calls from that, a plain callable ignores it. The arithmetic after it is unchanged, so
// the points, the values and every result are the same either way.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sd::integrate::quadpack {

constexpr double kEpmach = std::numeric_limits<double>::epsilon();  // d1mach(4)
constexpr double kUflow = std::numeric_limits<double>::min();        // d1mach(1)
constexpr double kOflow = std::numeric_limits<double>::max();        // d1mach(2)

// The work arrays of one call (dqagse's alist, blist, rlist, elist, iord; dqagpe's level, pts,
// ndin), 1-based, and the results. The interval arrays grow with `last` (the routines touch
// elements 1 ... last only), so a large limit costs nothing until it is used.
struct Work {
    std::vector<double> alist, blist, rlist, elist, pts;
    std::vector<int> iord, level, ndin, nnlog;
    double result = 0.0, abserr = 0.0;
    std::int64_t neval = 0;
    int ier = 0, last = 0;
    // dqawoe's Chebyshev moments (chebmo, 25 per row, rows 1 ... momcom + 1 grown as computed) and
    // dqc25f's row m, which a call with ksave = 1 takes from the call before it
    std::vector<double> chebmo;
    int momcom = 0, mom_row = 0;

    void reset(int limit, int npts2 = 0) {
        const auto n = static_cast<std::size_t>(std::min(limit, 64)) + 1;
        alist.assign(n, 0.0);
        blist.assign(n, 0.0);
        rlist.assign(n, 0.0);
        elist.assign(n, 0.0);
        iord.assign(n, 0);
        level.assign(n, 0);
        nnlog.assign(n, 0);
        pts.assign(static_cast<std::size_t>(npts2) + 1, 0.0);
        ndin.assign(static_cast<std::size_t>(npts2) + 1, 0);
        result = abserr = 0.0;
        neval = 0;
        ier = last = 0;
    }

    // room for the intervals 1 ... n
    void grow(int n) {
        const auto need = static_cast<std::size_t>(n) + 1;
        if (need <= alist.size()) return;
        const std::size_t m = std::max(need, 2 * alist.size());
        alist.resize(m, 0.0);
        blist.resize(m, 0.0);
        rlist.resize(m, 0.0);
        elist.resize(m, 0.0);
        iord.resize(m, 0);
        level.resize(m, 0);
        nnlog.resize(m, 0);
    }
};

// ---- the rules

// The points of one rule application, for an integrand that evaluates them together (see the top).
template <class F>
inline void prefetch(F& f, const double* xs, int n) {
    if constexpr (requires { f.prefetch(xs, n); }) f.prefetch(xs, n);
}

// dqk21: the 21-point Gauss-Kronrod rule (10-point Gauss) on [a, b].
inline constexpr double kXgk21[11] = {
    0.995657163025808080735527280689003, 0.973906528517171720077964012084452, 0.930157491355708226001207180059508,
    0.865063366688984510732096688423493, 0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
    0.562757134668604683339000099272694, 0.433395394129247190799265943165784, 0.294392862701460198131126603103866,
    0.148874338981631210884826001129720, 0.000000000000000000000000000000000,
};
inline constexpr double kWgk21[11] = {
    0.011694638867371874278064396062192, 0.032558162307964727478818972459390, 0.054755896574351996031381300244580,
    0.075039674810919952767043140916190, 0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
    0.123491976262065851077958109831074, 0.134709217311473325928054001771707, 0.142775938577060080797094273138717,
    0.147739104901338491374841515972068, 0.149445554002916905664936468389821,
};
inline constexpr double kWg10[5] = {
    0.066671344308688137593568809893332, 0.149451349150580593145776339657697, 0.219086362515982043995534934228163,
    0.269266719309996355091226921569469, 0.295524224714752870173892994651338,
};

// QUADPACK's error estimate from |K - G| (both rules' difference times the half length), resasc
// and resabs (the last two already scaled).
inline double kronrod_error(double abserr, double resabs, double resasc) {
    if (resasc != 0.0 && abserr != 0.0) abserr = resasc * std::min(1.0, std::pow(200.0 * abserr / resasc, 1.5));
    if (resabs > kUflow / (50.0 * kEpmach)) abserr = std::max((kEpmach * 50.0) * resabs, abserr);
    return abserr;
}

template <class F>
void qk21(F& f, double a, double b, double& result, double& abserr, double& resabs, double& resasc) {
    double fv1[10], fv2[10];
    const double centr = 0.5 * (a + b);
    const double hlgth = 0.5 * (b - a);
    const double dhlgth = std::abs(hlgth);
    double resg = 0.0;
    {
        double xs[21];
        int n = 0;
        xs[n++] = centr;
        for (int j = 1; j <= 5; ++j) {
            const double absc = hlgth * kXgk21[2 * j - 1];
            xs[n++] = centr - absc;
            xs[n++] = centr + absc;
        }
        for (int j = 1; j <= 5; ++j) {
            const double absc = hlgth * kXgk21[2 * j - 2];
            xs[n++] = centr - absc;
            xs[n++] = centr + absc;
        }
        prefetch(f, xs, n);
    }
    const double fc = f(centr);
    double resk = kWgk21[10] * fc;
    resabs = std::abs(resk);
    for (int j = 1; j <= 5; ++j) {  // the Gauss points
        const int jtw = 2 * j - 1;  // the original's jtw = 2j, 0-based
        const double absc = hlgth * kXgk21[jtw];
        const double fval1 = f(centr - absc);
        const double fval2 = f(centr + absc);
        fv1[jtw] = fval1;
        fv2[jtw] = fval2;
        const double fsum = fval1 + fval2;
        resg = std::fma(kWg10[j - 1], fsum, resg);
        resk = std::fma(kWgk21[jtw], fsum, resk);
        resabs = std::fma(kWgk21[jtw], std::abs(fval1) + std::abs(fval2), resabs);
    }
    for (int j = 1; j <= 5; ++j) {  // the Kronrod points
        const int jtwm1 = 2 * j - 2;  // the original's 2j - 1, 0-based
        const double absc = hlgth * kXgk21[jtwm1];
        const double fval1 = f(centr - absc);
        const double fval2 = f(centr + absc);
        fv1[jtwm1] = fval1;
        fv2[jtwm1] = fval2;
        const double fsum = fval1 + fval2;
        resk = std::fma(kWgk21[jtwm1], fsum, resk);
        resabs = std::fma(kWgk21[jtwm1], std::abs(fval1) + std::abs(fval2), resabs);
    }
    const double reskh = resk * 0.5;
    resasc = kWgk21[10] * std::abs(fc - reskh);
    for (int j = 0; j < 10; ++j)
        resasc = std::fma(kWgk21[j], std::abs(fv1[j] - reskh) + std::abs(fv2[j] - reskh), resasc);
    result = resk * hlgth;
    resabs = resabs * dhlgth;
    resasc = resasc * dhlgth;
    abserr = kronrod_error(std::abs((resk - resg) * hlgth), resabs, resasc);
}

// dqk15i: the 15-point Gauss-Kronrod rule (7-point Gauss) on [a, b] within (0, 1] of the
// transformed integrand of an infinite range: x = boun + dinf (1 - t)/t, both halves for inf = 2.
inline constexpr double kXgk15[8] = {
    0.991455371120812639206854697526329, 0.949107912342758524526189684047851, 0.864864423359769072789712788640926,
    0.741531185599394439863864773280788, 0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
    0.207784955007898467600689403773245, 0.000000000000000000000000000000000,
};
inline constexpr double kWgk15[8] = {
    0.022935322010529224963732008058970, 0.063092092629978553290700663189204, 0.104790010322250183839876322541518,
    0.140653259715525918745189590510238, 0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
    0.204432940075298892414161999234649, 0.209482141084727828012999174891714,
};
inline constexpr double kWg15i[8] = {
    0.0, 0.129484966168869693270611432679082, 0.0, 0.279705391489276667901467771423780,
    0.0, 0.381830050505118944950369775488975, 0.0, 0.417959183673469387755102040816327,
};

template <class F>
void qk15i(
    F& f, double boun, int inf, double a, double b, double& result, double& abserr, double& resabs, double& resasc
) {
    double fv1[7], fv2[7];
    const double dinf = static_cast<double>(std::min(1, inf));
    const double centr = 0.5 * (a + b);
    const double hlgth = 0.5 * (b - a);
    const double tabsc1 = boun + dinf * (1.0 - centr) / centr;
    {
        double xs[30];
        int n = 0;
        xs[n++] = tabsc1;
        if (inf == 2) xs[n++] = -tabsc1;
        for (int j = 0; j < 7; ++j) {
            const double absc = hlgth * kXgk15[j];
            const double absc1 = centr - absc;
            const double absc2 = centr + absc;
            const double t1 = boun + dinf * (1.0 - absc1) / absc1;
            const double t2 = boun + dinf * (1.0 - absc2) / absc2;
            xs[n++] = t1;
            xs[n++] = t2;
            if (inf == 2) {
                xs[n++] = -t1;
                xs[n++] = -t2;
            }
        }
        prefetch(f, xs, n);
    }
    double fval1 = f(tabsc1);
    if (inf == 2) fval1 = fval1 + f(-tabsc1);
    const double fc = (fval1 / centr) / centr;
    double resg = kWg15i[7] * fc;
    double resk = kWgk15[7] * fc;
    resabs = std::abs(resk);
    for (int j = 0; j < 7; ++j) {
        const double absc = hlgth * kXgk15[j];
        const double absc1 = centr - absc;
        const double absc2 = centr + absc;
        const double t1 = boun + dinf * (1.0 - absc1) / absc1;
        const double t2 = boun + dinf * (1.0 - absc2) / absc2;
        double v1 = f(t1);
        double v2 = f(t2);
        if (inf == 2) v1 = v1 + f(-t1);
        if (inf == 2) v2 = v2 + f(-t2);
        v1 = (v1 / absc1) / absc1;
        v2 = (v2 / absc2) / absc2;
        fv1[j] = v1;
        fv2[j] = v2;
        const double fsum = v1 + v2;
        resg = std::fma(kWg15i[j], fsum, resg);
        resk = std::fma(kWgk15[j], fsum, resk);
        resabs = std::fma(kWgk15[j], std::abs(v1) + std::abs(v2), resabs);
    }
    const double reskh = resk * 0.5;
    resasc = kWgk15[7] * std::abs(fc - reskh);
    for (int j = 0; j < 7; ++j)
        resasc = std::fma(kWgk15[j], std::abs(fv1[j] - reskh) + std::abs(fv2[j] - reskh), resasc);
    result = resk * hlgth;
    resasc = resasc * hlgth;
    resabs = resabs * hlgth;
    abserr = kronrod_error(std::abs((resk - resg) * hlgth), resabs, resasc);
}

// ---- the list of intervals and the epsilon algorithm

// dqpsrt: keeps iord(1..) the intervals in decreasing order of their error estimate (only the
// largest jupbn when last > limit/2 + 2) and gives the next one to bisect (maxerr, ermax).
inline void qpsrt(
    int limit,
    int last,
    int& maxerr,
    double& ermax,
    const std::vector<double>& elist,
    std::vector<int>& iord,
    int& nrmax
) {
    int i = 0, k = 0, jupbn, jbnd, ibeg, isucc;
    double errmax, errmin;
    if (last > 2) goto L10;
    iord[1] = 1;
    iord[2] = 2;
    goto L90;
L10:
    errmax = elist[maxerr];
    if (nrmax != 1) {
        const int ido = nrmax - 1;
        for (i = 1; i <= ido; ++i) {
            isucc = iord[nrmax - 1];
            if (errmax <= elist[isucc]) break;
            iord[nrmax] = isucc;
            nrmax = nrmax - 1;
        }
    }
    jupbn = last;
    if (last > (limit / 2 + 2)) jupbn = (limit - last) + 3;  // limit + 3 - last, without overflow
    errmin = elist[last];
    jbnd = jupbn - 1;
    ibeg = nrmax + 1;
    if (ibeg <= jbnd) {
        for (i = ibeg; i <= jbnd; ++i) {
            isucc = iord[i];
            if (errmax >= elist[isucc]) goto L60;
            iord[i - 1] = isucc;
        }
    }
    iord[jbnd] = maxerr;
    iord[jupbn] = last;
    goto L90;
L60:
    iord[i - 1] = maxerr;
    k = jbnd;
    for (int j = i; j <= jbnd; ++j) {
        isucc = iord[k];
        if (errmin < elist[isucc]) goto L80;
        iord[k + 1] = isucc;
        k = k - 1;
    }
    iord[i] = last;
    goto L90;
L80:
    iord[k + 1] = last;
L90:
    maxerr = iord[nrmax];
    ermax = elist[maxerr];
}

// dqelg: Wynn's epsilon algorithm on the sequence of the integral's approximations epstab(1..n),
// n >= 1; gives the extrapolated result and its error estimate, updates the table, n and the last
// three results.
inline void qelg(int& n, double* epstab, double& result, double& abserr, double* res3la, int& nres) {
    // epstab is 1-based, of 52 + 1 elements; res3la 1-based, of 3 + 1
    int i, ib, ib2, ie, indx, k1, k2, k3, newelm, num;
    const int limexp = 50;
    double delta1, delta2, delta3, e0, e1, e1abs, e2, e3, epsinf, err1, err2, err3, error, res, ss, tol1, tol2, tol3;
    nres = nres + 1;
    abserr = kOflow;
    result = epstab[n];
    if (n < 3) goto L100;
    epstab[n + 2] = epstab[n];
    newelm = (n - 1) / 2;
    epstab[n] = kOflow;
    num = n;
    k1 = n;
    for (i = 1; i <= newelm; ++i) {
        k2 = k1 - 1;
        k3 = k1 - 2;
        res = epstab[k1 + 2];
        e0 = epstab[k3];
        e1 = epstab[k2];
        e2 = res;
        e1abs = std::abs(e1);
        delta2 = e2 - e1;
        err2 = std::abs(delta2);
        tol2 = std::max(std::abs(e2), e1abs) * kEpmach;
        delta3 = e1 - e0;
        err3 = std::abs(delta3);
        tol3 = std::max(e1abs, std::abs(e0)) * kEpmach;
        if (!(err2 > tol2 || err3 > tol3)) {
            // e0, e1 and e2 are equal to within machine accuracy: convergence is assumed
            result = res;
            abserr = err2 + err3;
            goto L100;
        }
        e3 = epstab[k1];
        epstab[k1] = e1;
        delta1 = e1 - e3;
        err1 = std::abs(delta1);
        tol1 = std::max(e1abs, std::abs(e3)) * kEpmach;
        if (err1 <= tol1 || err2 <= tol2 || err3 <= tol3) goto L20;
        ss = 1.0 / delta1 + 1.0 / delta2 - 1.0 / delta3;
        epsinf = std::abs(ss * e1);
        if (epsinf > 1.0e-4) goto L30;
    L20:
        n = i + i - 1;
        goto L50;
    L30:
        res = e1 + 1.0 / ss;
        epstab[k1] = res;
        k1 = k1 - 2;
        error = err2 + std::abs(res - e2) + err3;
        if (error > abserr) continue;
        abserr = error;
        result = res;
    }
L50:
    if (n == limexp) n = 2 * (limexp / 2) - 1;
    ib = 1;
    if ((num / 2) * 2 == num) ib = 2;
    ie = newelm + 1;
    for (i = 1; i <= ie; ++i) {
        ib2 = ib + 2;
        epstab[ib] = epstab[ib2];
        ib = ib2;
    }
    if (num != n) {
        indx = num - n + 1;
        for (i = 1; i <= n; ++i) {
            epstab[i] = epstab[indx];
            indx = indx + 1;
        }
    }
    if (nres >= 4) goto L90;
    res3la[nres] = result;
    abserr = kOflow;
    goto L100;
L90:
    abserr = std::abs(result - res3la[3]) + std::abs(result - res3la[2]) + std::abs(result - res3la[1]);
    res3la[1] = res3la[2];
    res3la[2] = res3la[3];
    res3la[3] = result;
L100:
    abserr = std::max(abserr, 5.0 * kEpmach * std::abs(result));
}

// ---- the drivers

// The adaptive loop shared by dqagse and dqagie (the two are the same routine around their
// rules): rule(a, b, result, abserr, resabs, resasc) is dqk21 on [a, b] or dqk15i on [a, b] of
// (0, 1]; a, b the first interval, small0 dqagse's |b - a| * 0.375 or dqagie's 0.375, nfev the
// evaluations of one rule.
template <class Rule>
void qags_loop(Rule& rule, double a, double b, double small0, double epsabs, double epsrel, int limit, Work& w) {
    double abseps, area, area1, area12, area2, a1, a2, b1, b2, correc = 0.0, defabs, defab1, defab2, dres, erlarg = 0.0,
                                                               erlast, errbnd, errmax, error1, error2, erro12, errsum,
                                                               ertest = 0.0, resabs, reseps, small = 0.0;
    double res3la[4] = {0.0, 0.0, 0.0, 0.0}, rlist2[53];
    int id, ierro, iroff1, iroff2, iroff3, jupbnd, k, ksgn, ktmin, maxerr, nres, nrmax, numrl2;
    bool extrap, noext;
    auto& alist = w.alist;
    auto& blist = w.blist;
    auto& rlist = w.rlist;
    auto& elist = w.elist;
    auto& iord = w.iord;
    int& ier = w.ier;
    int& last = w.last;
    double& result = w.result;
    double& abserr = w.abserr;

    ier = 0;
    last = 0;
    result = 0.0;
    abserr = 0.0;
    alist[1] = a;
    blist[1] = b;
    rlist[1] = 0.0;
    elist[1] = 0.0;
    ierro = 0;
    rule(a, b, result, abserr, defabs, resabs);
    dres = std::abs(result);
    errbnd = std::max(epsabs, epsrel * dres);
    last = 1;
    rlist[1] = result;
    elist[1] = abserr;
    iord[1] = 1;
    if (abserr <= 1.0e2 * kEpmach * defabs && abserr > errbnd) ier = 2;
    if (limit == 1) ier = 1;
    if (ier != 0 || (abserr <= errbnd && abserr != resabs) || abserr == 0.0) goto L140;
    rlist2[1] = result;
    errmax = abserr;
    maxerr = 1;
    area = result;
    errsum = abserr;
    abserr = kOflow;
    nrmax = 1;
    nres = 0;
    numrl2 = 2;
    ktmin = 0;
    extrap = false;
    noext = false;
    iroff1 = 0;
    iroff2 = 0;
    iroff3 = 0;
    ksgn = -1;
    if (dres >= (1.0 - 50.0 * kEpmach) * defabs) ksgn = 1;
    for (last = 2; last <= limit; ++last) {
        w.grow(last);
        a1 = alist[maxerr];
        b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
        a2 = b1;
        b2 = blist[maxerr];
        erlast = errmax;
        rule(a1, b1, area1, error1, resabs, defab1);
        rule(a2, b2, area2, error2, resabs, defab2);
        area12 = area1 + area2;
        erro12 = error1 + error2;
        errsum = errsum + erro12 - errmax;
        area = area + area12 - rlist[maxerr];
        if (!(defab1 == error1 || defab2 == error2)) {
            if (!(std::abs(rlist[maxerr] - area12) > 1.0e-5 * std::abs(area12) || erro12 < 0.99 * errmax)) {
                if (extrap) iroff2 = iroff2 + 1;
                if (!extrap) iroff1 = iroff1 + 1;
            }
            if (last > 10 && erro12 > errmax) iroff3 = iroff3 + 1;
        }
        rlist[maxerr] = area1;
        rlist[last] = area2;
        errbnd = std::max(epsabs, epsrel * std::abs(area));
        if (iroff1 + iroff2 >= 10 || iroff3 >= 20) ier = 2;
        if (iroff2 >= 5) ierro = 3;
        if (last == limit) ier = 1;
        if (std::max(std::abs(a1), std::abs(b2)) <= (1.0 + 100.0 * kEpmach) * (std::abs(a2) + 1000.0 * kUflow)) ier = 4;
        if (error2 > error1) {
            alist[maxerr] = a2;
            alist[last] = a1;
            blist[last] = b1;
            rlist[maxerr] = area2;
            rlist[last] = area1;
            elist[maxerr] = error2;
            elist[last] = error1;
        } else {
            alist[last] = a2;
            blist[maxerr] = b1;
            blist[last] = b2;
            elist[maxerr] = error1;
            elist[last] = error2;
        }
        qpsrt(limit, last, maxerr, errmax, elist, iord, nrmax);
        if (errsum <= errbnd) goto L115;
        if (ier != 0) goto L100;
        if (last == 2) {
            small = small0;
            erlarg = errsum;
            ertest = errbnd;
            rlist2[2] = area;
            continue;
        }
        if (noext) continue;
        erlarg = erlarg - erlast;
        if (std::abs(b1 - a1) > small) erlarg = erlarg + erro12;
        if (!extrap) {
            if (std::abs(blist[maxerr] - alist[maxerr]) > small) continue;
            extrap = true;
            nrmax = 2;
        }
        if (ierro != 3 && erlarg > ertest) {
            id = nrmax;
            jupbnd = last;
            if (last > (2 + limit / 2)) jupbnd = (limit - last) + 3;
            bool more = false;
            for (k = id; k <= jupbnd; ++k) {
                maxerr = iord[nrmax];
                errmax = elist[maxerr];
                if (std::abs(blist[maxerr] - alist[maxerr]) > small) {
                    more = true;
                    break;
                }
                nrmax = nrmax + 1;
            }
            if (more) continue;
        }
        numrl2 = numrl2 + 1;
        rlist2[numrl2] = area;
        qelg(numrl2, rlist2, reseps, abseps, res3la, nres);
        ktmin = ktmin + 1;
        if (ktmin > 5 && abserr < 1.0e-3 * errsum) ier = 5;
        if (abseps < abserr) {
            ktmin = 0;
            abserr = abseps;
            result = reseps;
            correc = erlarg;
            ertest = std::max(epsabs, epsrel * std::abs(reseps));
            if (abserr <= ertest) goto L100;
        }
        if (numrl2 == 1) noext = true;
        if (ier == 5) goto L100;
        maxerr = iord[1];
        errmax = elist[maxerr];
        nrmax = 1;
        extrap = false;
        small = small * 0.5;
        erlarg = errsum;
    }
L100:
    if (abserr == kOflow) goto L115;
    if (ier + ierro == 0) goto L110;
    if (ierro == 3) abserr = abserr + correc;
    if (ier == 0) ier = 3;
    if (result != 0.0 && area != 0.0) {
        if (abserr / std::abs(result) > errsum / std::abs(area)) goto L115;
        goto L110;
    }
    if (abserr > errsum) goto L115;
    if (area == 0.0) goto L130;
L110:
    if (ksgn == -1 && std::max(std::abs(result), std::abs(area)) <= defabs * 1.0e-2) goto L130;
    if (1.0e-2 > (result / area) || (result / area) > 1.0e2 || errsum > std::abs(area)) ier = 6;
    goto L130;
L115:
    result = 0.0;
    for (k = 1; k <= last; ++k) result = result + rlist[k];
    abserr = errsum;
L130:
    if (ier > 2) ier = ier - 1;
L140:
    return;
}

// dqagse: QAGS on the finite [a, b]. Inputs are checked by the caller (ier 6 is not reached).
template <class F>
void qagse(F& f, double a, double b, double epsabs, double epsrel, int limit, Work& w) {
    w.reset(limit);
    auto rule = [&](double x, double y, double& r, double& e, double& ra, double& rs) { qk21(f, x, y, r, e, ra, rs); };
    qags_loop(rule, a, b, std::abs(b - a) * 0.375, epsabs, epsrel, limit, w);
    w.neval = 42 * static_cast<std::int64_t>(w.last) - 21;
}

// dqagie: QAGI on [bound, inf) (inf = 1), (-inf, bound] (inf = -1) or the real line (inf = 2).
template <class F>
void qagie(F& f, double bound, int inf, double epsabs, double epsrel, int limit, Work& w) {
    w.reset(limit);
    const double boun = inf == 2 ? 0.0 : bound;
    auto rule = [&](double x, double y, double& r, double& e, double& ra, double& rs) {
        qk15i(f, boun, inf, x, y, r, e, ra, rs);
    };
    qags_loop(rule, 0.0, 1.0, 0.375, epsabs, epsrel, limit, w);
    w.neval = 30 * static_cast<std::int64_t>(w.last) - 15;
    if (inf == 2) w.neval = 2 * w.neval;
}

// dqagpe: QAGP on [a, b] (a < b) with the npts break points points[0..npts) inside (a, b), sorted
// and distinct (scipy's np.unique and its filter); limit > npts.
template <class F>
void qagpe(F& f, double a, double b, const double* points, int npts, double epsabs, double epsrel, int limit, Work& w) {
    const int npts2 = npts + 2;
    w.reset(limit, npts2);
    w.grow(npts2);
    double abseps, area, area1, area12, area2, a1, a2, b1, b2, correc = 0.0, defabs, defab1, defab2, dres, erlarg,
                                                               erlast, errbnd, errmax, error1, erro12, error2, errsum,
                                                               ertest, resa, resabs, reseps, sign, temp;
    double res3la[4] = {0.0, 0.0, 0.0, 0.0}, rlist2[53];
    int i, id, ierro, ind1, ind2, ip1, iroff1, iroff2, iroff3, j, jlow, jupbnd,
        k = 0, ksgn, ktmin, levcur, levmax, maxerr, nint, nintp1, nres, nrmax, numrl2;
    bool extrap, noext;
    auto& alist = w.alist;
    auto& blist = w.blist;
    auto& rlist = w.rlist;
    auto& elist = w.elist;
    auto& iord = w.iord;
    auto& level = w.level;
    auto& pts = w.pts;
    auto& ndin = w.ndin;
    int& ier = w.ier;
    int& last = w.last;
    std::int64_t& neval = w.neval;
    double& result = w.result;
    double& abserr = w.abserr;

    ier = 0;
    neval = 0;
    last = 0;
    result = 0.0;
    abserr = 0.0;
    alist[1] = a;
    blist[1] = b;
    rlist[1] = 0.0;
    elist[1] = 0.0;
    iord[1] = 0;
    level[1] = 0;
    sign = 1.0;
    if (a > b) sign = -1.0;
    pts[1] = std::min(a, b);
    for (i = 1; i <= npts; ++i) pts[i + 1] = points[i - 1];
    pts[npts + 2] = std::max(a, b);
    nint = npts + 1;
    a1 = pts[1];
    if (npts != 0) {
        nintp1 = nint + 1;
        for (i = 1; i <= nint; ++i) {
            ip1 = i + 1;
            for (j = ip1; j <= nintp1; ++j) {
                if (pts[i] <= pts[j]) continue;
                temp = pts[i];
                pts[i] = pts[j];
                pts[j] = temp;
            }
        }
    }
    resabs = 0.0;
    for (i = 1; i <= nint; ++i) {
        b1 = pts[i + 1];
        qk21(f, a1, b1, area1, error1, defabs, resa);
        abserr = abserr + error1;
        result = result + area1;
        ndin[i] = 0;
        if (error1 == resa && error1 != 0.0) ndin[i] = 1;
        resabs = resabs + defabs;
        level[i] = 0;
        elist[i] = error1;
        alist[i] = a1;
        blist[i] = b1;
        rlist[i] = area1;
        iord[i] = i;
        a1 = b1;
    }
    errsum = 0.0;
    for (i = 1; i <= nint; ++i) {
        if (ndin[i] == 1) elist[i] = abserr;
        errsum = errsum + elist[i];
    }
    last = nint;
    neval = 21 * static_cast<std::int64_t>(nint);
    dres = std::abs(result);
    errbnd = std::max(epsabs, epsrel * dres);
    if (abserr <= 1.0e2 * kEpmach * resabs && abserr > errbnd) ier = 2;
    if (nint != 1) {
        for (i = 1; i <= npts; ++i) {
            jlow = i + 1;
            ind1 = iord[i];
            for (j = jlow; j <= nint; ++j) {
                ind2 = iord[j];
                if (elist[ind1] > elist[ind2]) continue;
                ind1 = ind2;
                k = j;
            }
            if (ind1 == iord[i]) continue;
            iord[k] = iord[i];
            iord[i] = ind1;
        }
        if (limit < npts2) ier = 1;
    }
    if (ier != 0 || abserr <= errbnd) goto L210;
    rlist2[1] = result;
    maxerr = iord[1];
    errmax = elist[maxerr];
    area = result;
    nrmax = 1;
    nres = 0;
    numrl2 = 1;
    ktmin = 0;
    extrap = false;
    noext = false;
    erlarg = errsum;
    ertest = errbnd;
    levmax = 1;
    iroff1 = 0;
    iroff2 = 0;
    iroff3 = 0;
    ierro = 0;
    abserr = kOflow;
    ksgn = -1;
    if (dres >= (1.0 - 50.0 * kEpmach) * resabs) ksgn = 1;
    for (last = npts2; last <= limit; ++last) {
        w.grow(last);
        levcur = level[maxerr] + 1;
        a1 = alist[maxerr];
        b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
        a2 = b1;
        b2 = blist[maxerr];
        erlast = errmax;
        qk21(f, a1, b1, area1, error1, resa, defab1);
        qk21(f, a2, b2, area2, error2, resa, defab2);
        neval = neval + 42;
        area12 = area1 + area2;
        erro12 = error1 + error2;
        errsum = errsum + erro12 - errmax;
        area = area + area12 - rlist[maxerr];
        if (!(defab1 == error1 || defab2 == error2)) {
            if (!(std::abs(rlist[maxerr] - area12) > 1.0e-5 * std::abs(area12) || erro12 < 0.99 * errmax)) {
                if (extrap) iroff2 = iroff2 + 1;
                if (!extrap) iroff1 = iroff1 + 1;
            }
            if (last > 10 && erro12 > errmax) iroff3 = iroff3 + 1;
        }
        level[maxerr] = levcur;
        level[last] = levcur;
        rlist[maxerr] = area1;
        rlist[last] = area2;
        errbnd = std::max(epsabs, epsrel * std::abs(area));
        if (iroff1 + iroff2 >= 10 || iroff3 >= 20) ier = 2;
        if (iroff2 >= 5) ierro = 3;
        if (last == limit) ier = 1;
        if (std::max(std::abs(a1), std::abs(b2)) <= (1.0 + 100.0 * kEpmach) * (std::abs(a2) + 1000.0 * kUflow)) ier = 4;
        if (error2 > error1) {
            alist[maxerr] = a2;
            alist[last] = a1;
            blist[last] = b1;
            rlist[maxerr] = area2;
            rlist[last] = area1;
            elist[maxerr] = error2;
            elist[last] = error1;
        } else {
            alist[last] = a2;
            blist[maxerr] = b1;
            blist[last] = b2;
            elist[maxerr] = error1;
            elist[last] = error2;
        }
        qpsrt(limit, last, maxerr, errmax, elist, iord, nrmax);
        if (errsum <= errbnd) goto L190;
        if (ier != 0) goto L170;
        if (noext) continue;
        erlarg = erlarg - erlast;
        if (levcur + 1 <= levmax) erlarg = erlarg + erro12;
        if (!extrap) {
            if (level[maxerr] + 1 <= levmax) continue;
            extrap = true;
            nrmax = 2;
        }
        if (ierro != 3 && erlarg > ertest) {
            id = nrmax;
            jupbnd = last;
            if (last > (2 + limit / 2)) jupbnd = (limit - last) + 3;
            bool more = false;
            for (k = id; k <= jupbnd; ++k) {
                maxerr = iord[nrmax];
                errmax = elist[maxerr];
                if (level[maxerr] + 1 <= levmax) {
                    more = true;
                    break;
                }
                nrmax = nrmax + 1;
            }
            if (more) continue;
        }
        numrl2 = numrl2 + 1;
        rlist2[numrl2] = area;
        if (numrl2 > 2) {
            qelg(numrl2, rlist2, reseps, abseps, res3la, nres);
            ktmin = ktmin + 1;
            if (ktmin > 5 && abserr < 1.0e-3 * errsum) ier = 5;
            if (abseps < abserr) {
                ktmin = 0;
                abserr = abseps;
                result = reseps;
                correc = erlarg;
                ertest = std::max(epsabs, epsrel * std::abs(reseps));
                if (abserr < ertest) goto L170;
            }
            if (numrl2 == 1) noext = true;
            if (ier >= 5) goto L170;
        }
        maxerr = iord[1];
        errmax = elist[maxerr];
        nrmax = 1;
        extrap = false;
        levmax = levmax + 1;
        erlarg = errsum;
    }
L170:
    if (abserr == kOflow) goto L190;
    if ((ier + ierro) == 0) goto L180;
    if (ierro == 3) abserr = abserr + correc;
    if (ier == 0) ier = 3;
    if (result != 0.0 && area != 0.0) {
        if (abserr / std::abs(result) > errsum / std::abs(area)) goto L190;
        goto L180;
    }
    if (abserr > errsum) goto L190;
    if (area == 0.0) goto L210;
L180:
    if (ksgn == -1 && std::max(std::abs(result), std::abs(area)) <= resabs * 1.0e-2) goto L210;
    if (1.0e-2 > (result / area) || (result / area) > 1.0e2 || errsum > std::abs(area)) ier = 6;
    goto L210;
L190:
    result = 0.0;
    for (k = 1; k <= last; ++k) result = result + rlist[k];
    abserr = errsum;
L210:
    if (ier > 2) ier = ier - 1;
    result = result * sign;
}

// ---- the weighted routines (milestone C5): QAWO, QAWF, QAWS, QAWC

// dqk15w: the 15-point Gauss-Kronrod rule (7-point Gauss) for f(x) w(x) on [a, b], w one of
// QUADPACK's weight functions (dqwgtf, dqwgts, dqwgtc). Its constants are dqk15w's own 16-digit
// literals (not dqk15i's).
inline constexpr double kXgk15w[8] = {
    0.9914553711208126e+00, 0.9491079123427585e+00, 0.8648644233597691e+00, 0.7415311855993944e+00,
    0.5860872354676911e+00, 0.4058451513773972e+00, 0.2077849550078985e+00, 0.0000000000000000e+00,
};
inline constexpr double kWgk15w[8] = {
    0.2293532201052922e-01, 0.6309209262997855e-01, 0.1047900103222502e+00, 0.1406532597155259e+00,
    0.1690047266392679e+00, 0.1903505780647854e+00, 0.2044329400752989e+00, 0.2094821410847278e+00,
};
inline constexpr double kWg7w[4] = {
    0.1294849661688697e+00,
    0.2797053914892767e+00,
    0.3818300505051189e+00,
    0.4179591836734694e+00,
};

template <class F, class W>
void qk15w(F& f, const W& w, double a, double b, double& result, double& abserr, double& resabs, double& resasc) {
    double fv1[8], fv2[8];  // 1-based
    const double centr = 0.5 * (a + b);
    const double hlgth = 0.5 * (b - a);
    const double dhlgth = std::abs(hlgth);
    {
        double xs[15];
        int n = 0;
        xs[n++] = centr;
        for (int j = 1; j <= 3; ++j) {
            const double absc = hlgth * kXgk15w[2 * j - 1];
            xs[n++] = centr - absc;
            xs[n++] = centr + absc;
        }
        for (int j = 1; j <= 4; ++j) {
            const double absc = hlgth * kXgk15w[2 * j - 2];
            xs[n++] = centr - absc;
            xs[n++] = centr + absc;
        }
        prefetch(f, xs, n);
    }
    const double fc = f(centr) * w(centr);
    double resg = kWg7w[3] * fc;
    double resk = kWgk15w[7] * fc;
    resabs = std::abs(resk);
    for (int j = 1; j <= 3; ++j) {
        const int jtw = j * 2;
        const double absc = hlgth * kXgk15w[jtw - 1];
        const double absc1 = centr - absc;
        const double absc2 = centr + absc;
        const double fval1 = f(absc1) * w(absc1);
        const double fval2 = f(absc2) * w(absc2);
        fv1[jtw] = fval1;
        fv2[jtw] = fval2;
        const double fsum = fval1 + fval2;
        resg = std::fma(kWg7w[j - 1], fsum, resg);
        resk = std::fma(kWgk15w[jtw - 1], fsum, resk);
        resabs = std::fma(kWgk15w[jtw - 1], std::abs(fval1) + std::abs(fval2), resabs);
    }
    for (int j = 1; j <= 4; ++j) {
        const int jtwm1 = j * 2 - 1;
        const double absc = hlgth * kXgk15w[jtwm1 - 1];
        const double absc1 = centr - absc;
        const double absc2 = centr + absc;
        const double fval1 = f(absc1) * w(absc1);
        const double fval2 = f(absc2) * w(absc2);
        fv1[jtwm1] = fval1;
        fv2[jtwm1] = fval2;
        const double fsum = fval1 + fval2;
        resk = std::fma(kWgk15w[jtwm1 - 1], fsum, resk);
        resabs = std::fma(kWgk15w[jtwm1 - 1], std::abs(fval1) + std::abs(fval2), resabs);
    }
    const double reskh = resk * 0.5;
    resasc = kWgk15w[7] * std::abs(fc - reskh);
    for (int j = 1; j <= 7; ++j)
        resasc = std::fma(kWgk15w[j - 1], std::abs(fv1[j] - reskh) + std::abs(fv2[j] - reskh), resasc);
    result = resk * hlgth;
    resabs = resabs * dhlgth;
    resasc = resasc * dhlgth;
    abserr = kronrod_error(std::abs((resk - resg) * hlgth), resabs, resasc);
}

// The weight functions dqwgtf (cos or sin of omega x), dqwgts ((x - a)^alfa (b - x)^beta and its
// logarithms) and dqwgtc (1/(x - c)).
struct WeightF {
    double omega;
    int integr;
    double operator()(double x) const {
        const double omx = omega * x;
        return integr == 1 ? std::cos(omx) : std::sin(omx);
    }
};

struct WeightS {
    double a, b, alfa, beta;
    int integr;
    double operator()(double x) const {
        const double xma = x - a;
        const double bmx = b - x;
        double v = std::pow(xma, alfa) * std::pow(bmx, beta);
        if (integr == 2) v = v * std::log(xma);
        if (integr == 3) v = v * std::log(bmx);
        if (integr == 4) v = v * std::log(xma) * std::log(bmx);
        return v;
    }
};

struct WeightC {
    double c;
    double operator()(double x) const { return 1.0 / (x - c); }
};

// The 25 points cos(k pi/24), k = 1 ... 11, of the Clenshaw-Curtis rules (dqc25f, dqc25s, dqc25c).
inline constexpr double kXcc[12] = {
    0.0,
    0.991444861373810411144557526928563,
    0.965925826289068286749743199728897,
    0.923879532511286756128183189396788,
    0.866025403784438646763723170752936,
    0.793353340291235164579776961501299,
    0.707106781186547524400844362104849,
    0.608761429008720639416097542898164,
    0.500000000000000000000000000000000,
    0.382683432365089771728459984030399,
    0.258819045102520762348898837624048,
    0.130526192220051591548406227895489,
};

// dqcheb: the Chebyshev series coefficients of degree 12 and 24 of the 25 values fval(1..25)
// (fval is overwritten). Arrays are 1-based.
inline void qcheb(const double* x, double* fval, double* cheb12, double* cheb24) {
    double v[13], alam, alam1, alam2, part1, part2, part3;
    for (int i = 1; i <= 12; ++i) {
        const int j = 26 - i;
        v[i] = fval[i] - fval[j];
        fval[i] = fval[i] + fval[j];
    }
    alam1 = v[1] - v[9];
    alam2 = x[6] * (v[3] - v[7] - v[11]);
    cheb12[4] = alam1 + alam2;
    cheb12[10] = alam1 - alam2;
    alam1 = v[2] - v[8] - v[10];
    alam2 = v[4] - v[6] - v[12];
    alam = std::fma(x[3], alam1, x[9] * alam2);
    cheb24[4] = cheb12[4] + alam;
    cheb24[22] = cheb12[4] - alam;
    alam = std::fma(x[9], alam1, -(x[3] * alam2));
    cheb24[10] = cheb12[10] + alam;
    cheb24[16] = cheb12[10] - alam;
    part1 = x[4] * v[5];
    part2 = x[8] * v[9];
    part3 = x[6] * v[7];
    alam1 = v[1] + part1 + part2;
    alam2 = std::fma(x[10], v[11], std::fma(x[2], v[3], part3));
    cheb12[2] = alam1 + alam2;
    cheb12[12] = alam1 - alam2;
    alam = std::fma(x[1], v[2], x[3] * v[4]);
    alam = std::fma(x[5], v[6], alam);
    alam = std::fma(x[7], v[8], alam);
    alam = std::fma(x[9], v[10], alam);
    alam = std::fma(x[11], v[12], alam);
    cheb24[2] = cheb12[2] + alam;
    cheb24[24] = cheb12[2] - alam;
    alam = std::fma(x[11], v[2], -(x[9] * v[4]));
    alam = std::fma(x[7], v[6], alam);
    alam = std::fma(-x[5], v[8], alam);
    alam = std::fma(x[3], v[10], alam);
    alam = std::fma(-x[1], v[12], alam);
    cheb24[12] = cheb12[12] + alam;
    cheb24[14] = cheb12[12] - alam;
    alam1 = v[1] - part1 + part2;
    alam2 = std::fma(x[2], v[11], std::fma(x[10], v[3], -part3));
    cheb12[6] = alam1 + alam2;
    cheb12[8] = alam1 - alam2;
    alam = std::fma(x[5], v[2], -(x[9] * v[4]));
    alam = std::fma(-x[1], v[6], alam);
    alam = std::fma(-x[11], v[8], alam);
    alam = std::fma(x[3], v[10], alam);
    alam = std::fma(x[7], v[12], alam);
    cheb24[6] = cheb12[6] + alam;
    cheb24[20] = cheb12[6] - alam;
    alam = std::fma(x[7], v[2], -(x[3] * v[4]));
    alam = std::fma(-x[11], v[6], alam);
    alam = std::fma(x[1], v[8], alam);
    alam = std::fma(-x[9], v[10], alam);
    alam = std::fma(-x[5], v[12], alam);
    cheb24[8] = cheb12[8] + alam;
    cheb24[18] = cheb12[8] - alam;
    for (int i = 1; i <= 6; ++i) {
        const int j = 14 - i;
        v[i] = fval[i] - fval[j];
        fval[i] = fval[i] + fval[j];
    }
    alam1 = std::fma(x[8], v[5], v[1]);
    alam2 = x[4] * v[3];
    cheb12[3] = alam1 + alam2;
    cheb12[11] = alam1 - alam2;
    cheb12[7] = v[1] - v[5];
    alam = std::fma(x[10], v[6], std::fma(x[2], v[2], x[6] * v[4]));
    cheb24[3] = cheb12[3] + alam;
    cheb24[23] = cheb12[3] - alam;
    alam = x[6] * (v[2] - v[4] - v[6]);
    cheb24[7] = cheb12[7] + alam;
    cheb24[19] = cheb12[7] - alam;
    alam = std::fma(x[2], v[6], std::fma(x[10], v[2], -(x[6] * v[4])));
    cheb24[11] = cheb12[11] + alam;
    cheb24[15] = cheb12[11] - alam;
    for (int i = 1; i <= 3; ++i) {
        const int j = 8 - i;
        v[i] = fval[i] - fval[j];
        fval[i] = fval[i] + fval[j];
    }
    cheb12[5] = std::fma(x[8], v[3], v[1]);
    cheb12[9] = std::fma(-x[8], fval[3], fval[1]);
    alam = x[4] * v[2];
    cheb24[5] = cheb12[5] + alam;
    cheb24[21] = cheb12[5] - alam;
    alam = std::fma(x[8], fval[2], -fval[4]);
    cheb24[9] = cheb12[9] + alam;
    cheb24[17] = cheb12[9] - alam;
    cheb12[1] = fval[1] + fval[3];
    alam = fval[2] + fval[4];
    cheb24[1] = cheb12[1] + alam;
    cheb24[25] = cheb12[1] - alam;
    cheb12[13] = v[1] - v[3];
    cheb24[13] = cheb12[13];
    alam = 1.0 / 6.0;
    for (int i = 2; i <= 12; ++i) cheb12[i] = cheb12[i] * alam;
    alam = 0.5 * alam;
    cheb12[1] = cheb12[1] * alam;
    cheb12[13] = cheb12[13] * alam;
    for (int i = 2; i <= 24; ++i) cheb24[i] = cheb24[i] * alam;
    cheb24[1] = 0.5 * alam * cheb24[1];
    cheb24[25] = 0.5 * alam * cheb24[25];
}

// dgtsl: the tridiagonal system of order n with subdiagonal c(2..n), diagonal d, superdiagonal
// e(1..n-1) and right-hand side b (all 1-based, overwritten; b the solution) by Gaussian
// elimination with partial pivoting, in the form of scipy's C port: rows are swapped when
// |d(k)| < |c(k+1)| (LINPACK's dgtsl also on a tie), c(k+1) keeps row k's second superdiagonal,
// and the elimination stops at row n - 1 (LINPACK's goes on to row n): the last row is solved
// as if its subdiagonal were 0. The moments of dqc25f come out within the same error of their
// exact values either way (docs/design/integrate.md §7.11). dqc25f never meets a zero pivot.
inline void dgtsl(int n, double* c, double* d, double* e, double* b) {
    for (int k = 1; k <= n - 2; ++k) {
        if (std::abs(d[k]) < std::abs(c[k + 1])) {
            const double t = -d[k] / c[k + 1];
            const double dk1 = d[k + 1];
            d[k] = c[k + 1];
            d[k + 1] = std::fma(t, dk1, e[k]);
            c[k + 1] = e[k + 1];
            e[k] = dk1;
            e[k + 1] = t * e[k + 1];
            const double bk = b[k];
            b[k] = b[k + 1];
            b[k + 1] = std::fma(t, b[k + 1], bk);
        } else {
            const double t = -c[k + 1] / d[k];
            d[k + 1] = std::fma(t, e[k], d[k + 1]);
            b[k + 1] = std::fma(t, b[k], b[k + 1]);
            c[k + 1] = 0.0;
        }
    }
    b[n] = b[n] / d[n];
    b[n - 1] = std::fma(-e[n - 1], b[n], b[n - 1]) / d[n - 1];
    for (int k = n - 2; k >= 1; --k) b[k] = std::fma(-c[k + 1], b[k + 2], std::fma(-e[k], b[k + 1], b[k])) / d[k];
}

// dqc25f: the integral of f(x) cos(omega x) (integr 1) or sin(omega x) (2) on [a, b], omega >= 0:
// dqk15w when |omega (b - a)/2| <= 2, else the 25-point Clenshaw-Curtis rule with the modified
// Chebyshev moments of level nrmom (computed and stored in w.chebmo's row momcom + 1 unless
// already there; ksave = 1 takes the row of the call before, the other half of the same bisection).
template <class F>
void qc25f(
    F& f,
    double a,
    double b,
    double omega,
    int integr,
    int nrmom,
    int maxp1,
    int ksave,
    double& result,
    double& abserr,
    int& neval,
    double& resabs,
    double& resasc,
    Work& w
) {
    double cheb12[14], cheb24[26], d[26], d1[26], d2[26], fval[26], v[29];
    double ac, an, an2, as, asap, ass, conc, cons, cospar, estc, ests, par2, par22, sinpar, resc12, resc24, ress12,
        ress24;
    int noequ = 25, noeq1 = 24;
    // the row of the moments: kept between calls (the Fortran's m, which a call with ksave = 1 and
    // nrmom >= momcom, once max_p1 rows are taken, reads from the call before); scipy's C port
    // starts it at row 1 on every call, the moments of level 0 (C77)
    int& m = w.mom_row;
    const double centr = 0.5 * (b + a);
    const double hlgth = 0.5 * (b - a);
    const double parint = omega * hlgth;
    if (std::abs(parint) <= 2.0) {
        const WeightF wf{omega, integr};
        qk15w(f, wf, a, b, result, abserr, resabs, resasc);
        neval = 15;
        return;
    }
    conc = hlgth * std::cos(centr * omega);
    cons = hlgth * std::sin(centr * omega);
    resasc = kOflow;
    neval = 25;
    if (!(nrmom < w.momcom || ksave == 1)) {
        m = w.momcom + 1;
        if (w.chebmo.size() < static_cast<std::size_t>(m) * 25 + 1)
            w.chebmo.resize(static_cast<std::size_t>(m) * 25 + 1, 0.0);
        double* mo = w.chebmo.data() + static_cast<std::size_t>(m - 1) * 25;  // mo[k] = chebmo(m, k)
        par2 = parint * parint;
        par22 = par2 + 2.0;
        sinpar = std::sin(parint);
        cospar = std::cos(parint);
        // the moments of the cosine
        v[1] = 2.0 * sinpar / parint;
        v[2] = std::fma(8.0, cospar, (par2 + par2 - 8.0) * sinpar / parint) / par2;
        v[3] = std::fma(32.0 * (par2 - 12.0), cospar, (2.0 * std::fma(par2 - 80.0, par2, 192.0) * sinpar) / parint) /
               (par2 * par2);
        ac = 8.0 * cospar;
        as = 24.0 * parint * sinpar;
        if (std::abs(parint) <= 24.0) {
            an = 6.0;
            for (int k = 1; k <= noeq1; ++k) {
                an2 = an * an;
                d[k] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
                d2[k] = (an - 1.0) * (an - 2.0) * par2;
                d1[k + 1] = (an + 3.0) * (an + 4.0) * par2;
                v[k + 3] = std::fma(-(an2 - 4.0), ac, as);
                an = an + 2.0;
            }
            an2 = an * an;
            d[noequ] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
            v[noequ + 3] = std::fma(-(an2 - 4.0), ac, as);
            v[4] = std::fma(-(56.0 * par2), v[3], v[4]);
            ass = parint * sinpar;
            asap = std::fma(std::fma(210.0, par2, -1.0), cospar, -(std::fma(105.0, par2, -63.0) * ass)) / an2;
            asap = std::fma(15.0, ass, std::fma(-std::fma(-15.0, par2, 1.0), cospar, asap)) / an2;
            asap = std::fma(3.0, ass, asap - cospar) / an2;
            asap = (asap - cospar) / an2;
            v[noequ + 3] = std::fma(-(2.0 * asap * par2 * (an - 1.0)), an - 2.0, v[noequ + 3]);
            dgtsl(noequ, d1, d, d2, v + 3);
        } else {
            an = 4.0;
            for (int i = 4; i <= 13; ++i) {
                an2 = an * an;
                const double t = std::fma(an2 - 4.0, std::fma(2.0 * (par22 - an2 - an2), v[i - 1], -ac), as);
                v[i] = std::fma(-(par2 * (an + 1.0) * (an + 2.0)), v[i - 2], t) / (par2 * (an - 1.0) * (an - 2.0));
                an = an + 2.0;
            }
        }
        for (int j = 1; j <= 13; ++j) mo[2 * j - 1] = v[j];
        // the moments of the sine
        v[1] = 2.0 * std::fma(-parint, cospar, sinpar) / par2;
        v[2] = (18.0 - 48.0 / par2) * sinpar / par2 + (-2.0 + 48.0 / par2) * cospar / parint;
        ac = -24.0 * parint * cospar;
        as = -8.0 * sinpar;
        if (std::abs(parint) <= 24.0) {
            an = 5.0;
            for (int k = 1; k <= noeq1; ++k) {
                an2 = an * an;
                d[k] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
                d2[k] = (an - 1.0) * (an - 2.0) * par2;
                d1[k + 1] = (an + 3.0) * (an + 4.0) * par2;
                v[k + 2] = std::fma(an2 - 4.0, as, ac);
                an = an + 2.0;
            }
            an2 = an * an;
            d[noequ] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
            v[noequ + 2] = std::fma(an2 - 4.0, as, ac);
            v[3] = std::fma(-(42.0 * par2), v[2], v[3]);
            ass = parint * cospar;
            asap = std::fma(std::fma(105.0, par2, -63.0), ass, std::fma(210.0, par2, -1.0) * sinpar) / an2;
            asap = std::fma(-15.0, ass, std::fma(std::fma(15.0, par2, -1.0), sinpar, asap)) / an2;
            asap = (std::fma(-3.0, ass, asap) - sinpar) / an2;
            asap = (asap - sinpar) / an2;
            v[noequ + 2] = std::fma(-(2.0 * asap * par2 * (an - 1.0)), an - 2.0, v[noequ + 2]);
            dgtsl(noequ, d1, d, d2, v + 2);
        } else {
            an = 3.0;
            for (int i = 3; i <= 12; ++i) {
                an2 = an * an;
                const double t = std::fma(an2 - 4.0, std::fma(2.0 * (par22 - an2 - an2), v[i - 1], as), ac);
                v[i] = std::fma(-(par2 * (an + 1.0) * (an + 2.0)), v[i - 2], t) / (par2 * (an - 1.0) * (an - 2.0));
                an = an + 2.0;
            }
        }
        for (int j = 1; j <= 12; ++j) mo[2 * j] = v[j];
    }
    if (nrmom < w.momcom) m = nrmom + 1;
    if (w.momcom < (maxp1 - 1) && nrmom >= w.momcom) w.momcom = w.momcom + 1;
    const double* mo = w.chebmo.data() + static_cast<std::size_t>(m - 1) * 25;
    // the integrand at the 25 points
    {
        double xs[25];
        int n = 0;
        xs[n++] = centr + hlgth;
        xs[n++] = centr;
        xs[n++] = centr - hlgth;
        for (int i = 2; i <= 12; ++i) {
            xs[n++] = std::fma(hlgth, kXcc[i - 1], centr);
            xs[n++] = std::fma(-hlgth, kXcc[i - 1], centr);
        }
        prefetch(f, xs, n);
    }
    fval[1] = 0.5 * f(centr + hlgth);
    fval[13] = f(centr);
    fval[25] = 0.5 * f(centr - hlgth);
    for (int i = 2; i <= 12; ++i) {
        const int isym = 26 - i;
        fval[i] = f(std::fma(hlgth, kXcc[i - 1], centr));
        fval[isym] = f(std::fma(-hlgth, kXcc[i - 1], centr));
    }
    qcheb(kXcc, fval, cheb12, cheb24);
    // the integrals by the Chebyshev series of degree 12 and 24
    resc12 = cheb12[13] * mo[13];
    ress12 = 0.0;
    int k = 11;
    for (int j = 1; j <= 6; ++j) {
        resc12 = std::fma(cheb12[k], mo[k], resc12);
        ress12 = std::fma(cheb12[k + 1], mo[k + 1], ress12);
        k = k - 2;
    }
    resc24 = cheb24[25] * mo[25];
    ress24 = 0.0;
    resabs = std::abs(cheb24[25]);
    k = 23;
    for (int j = 1; j <= 12; ++j) {
        resc24 = std::fma(cheb24[k], mo[k], resc24);
        ress24 = std::fma(cheb24[k + 1], mo[k + 1], ress24);
        resabs = resabs + std::abs(cheb24[k]) + std::abs(cheb24[k + 1]);
        k = k - 2;
    }
    estc = std::abs(resc24 - resc12);
    ests = std::abs(ress24 - ress12);
    resabs = resabs * std::abs(hlgth);
    if (integr == 2) {
        result = std::fma(conc, ress24, cons * resc24);
        abserr = std::abs(conc * ests) + std::abs(cons * estc);
    } else {
        result = std::fma(conc, resc24, -(cons * ress24));
        abserr = std::abs(conc * estc) + std::abs(cons * ests);
    }
    // The rounding floor eps resabs (decision C123; not in QUADPACK or scipy): the difference of
    // the degree 12 and 24 series does not see the rounding of the Chebyshev coefficients and the
    // moments, which leaves about eps times the integral of |f| (resabs, the coefficients' sum)
    // in the result. Where |integral| << integral of |f| (a large omega) the estimate fell 1e8
    // times below the true error; the Kronrod rules' floor is 50 eps resabs (kronrod_error).
    if (resabs > kUflow / kEpmach) abserr = std::max(abserr, kEpmach * resabs);
}

// dqawoe: QAWO, the integral of f(x) cos(omega x) (integr 1) or sin(omega x) (2) on the finite
// [a, b], with dqc25f's rules and the epsilon algorithm. icall > 1 (dqawfe's later cycles) keeps
// the moments of the calls before (the cycles have the same length). Inputs are checked by the
// caller (ier 6 is not reached).
template <class F>
void qawoe(
    F& f,
    double a,
    double b,
    double omega,
    int integr,
    double epsabs,
    double epsrel,
    int limit,
    int icall,
    int maxp1,
    Work& w
) {
    w.reset(limit);
    double abseps, area, area1, area12, area2, a1, a2, b1, b2,
        correc = 0.0, defab1, defab2, defabs, domega, dres, erlarg = 0.0, erlast, errbnd, errmax, error1, erro12,
        error2, errsum, ertest = 0.0, resabs, reseps, small, width;
    double res3la[4] = {0.0, 0.0, 0.0, 0.0}, rlist2[53];
    int id, ierro, iroff1, iroff2, iroff3, jupbnd, k, ksgn, ktmin, maxerr, nev, nres, nrmax, nrmom, numrl2;
    bool extrap, noext, extall;
    int nev0 = 0;
    auto& alist = w.alist;
    auto& blist = w.blist;
    auto& rlist = w.rlist;
    auto& elist = w.elist;
    auto& iord = w.iord;
    auto& nnlog = w.nnlog;
    int& ier = w.ier;
    int& last = w.last;
    double& result = w.result;
    double& abserr = w.abserr;

    ier = 0;
    w.neval = 0;
    last = 0;
    result = 0.0;
    abserr = 0.0;
    alist[1] = a;
    blist[1] = b;
    rlist[1] = 0.0;
    elist[1] = 0.0;
    iord[1] = 0;
    nnlog[1] = 0;
    domega = std::abs(omega);
    nrmom = 0;
    if (icall <= 1) w.momcom = 0;
    qc25f(f, a, b, domega, integr, nrmom, maxp1, 0, result, abserr, nev0, defabs, resabs, w);
    w.neval = nev0;
    dres = std::abs(result);
    errbnd = std::max(epsabs, epsrel * dres);
    rlist[1] = result;
    elist[1] = abserr;
    iord[1] = 1;
    if (abserr <= 100.0 * kEpmach * defabs && abserr > errbnd) ier = 2;
    if (limit == 1) ier = 1;
    if (ier != 0 || abserr <= errbnd) goto L200;
    errmax = abserr;
    maxerr = 1;
    area = result;
    errsum = abserr;
    abserr = kOflow;
    nrmax = 1;
    extrap = false;
    noext = false;
    ierro = 0;
    iroff1 = 0;
    iroff2 = 0;
    iroff3 = 0;
    ktmin = 0;
    small = std::abs(b - a) * 0.75;
    nres = 0;
    numrl2 = 0;
    extall = false;
    if (!(0.5 * std::abs(b - a) * domega > 2.0)) {
        numrl2 = 1;
        extall = true;
        rlist2[1] = result;
    }
    if (0.25 * std::abs(b - a) * domega <= 2.0) extall = true;
    ksgn = -1;
    if (dres >= (1.0 - 50.0 * kEpmach) * defabs) ksgn = 1;
    for (last = 2; last <= limit; ++last) {
        w.grow(last);
        nrmom = nnlog[maxerr] + 1;
        a1 = alist[maxerr];
        b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
        a2 = b1;
        b2 = blist[maxerr];
        erlast = errmax;
        qc25f(f, a1, b1, domega, integr, nrmom, maxp1, 0, area1, error1, nev, resabs, defab1, w);
        w.neval += nev;
        qc25f(f, a2, b2, domega, integr, nrmom, maxp1, 1, area2, error2, nev, resabs, defab2, w);
        w.neval += nev;
        area12 = area1 + area2;
        erro12 = error1 + error2;
        errsum = errsum + erro12 - errmax;
        area = area + area12 - rlist[maxerr];
        if (!(defab1 == error1 || defab2 == error2)) {
            if (!(std::abs(rlist[maxerr] - area12) > 1.0e-5 * std::abs(area12) || erro12 < 0.99 * errmax)) {
                if (extrap) iroff2 = iroff2 + 1;
                if (!extrap) iroff1 = iroff1 + 1;
            }
            if (last > 10 && erro12 > errmax) iroff3 = iroff3 + 1;
        }
        rlist[maxerr] = area1;
        rlist[last] = area2;
        nnlog[maxerr] = nrmom;
        nnlog[last] = nrmom;
        errbnd = std::max(epsabs, epsrel * std::abs(area));
        if (iroff1 + iroff2 >= 10 || iroff3 >= 20) ier = 2;
        if (iroff2 >= 5) ierro = 3;
        if (last == limit) ier = 1;
        // the interval is too small: scipy's C port of dqawoe has epmach where the original has uflow
        // (1000 epmach = 2.2e-13; the other routines have 1000 uflow), and its stopping is kept
        if (std::max(std::abs(a1), std::abs(b2)) <= (1.0 + 100.0 * kEpmach) * (std::abs(a2) + 1000.0 * kEpmach))
            ier = 4;
        if (error2 > error1) {
            alist[maxerr] = a2;
            alist[last] = a1;
            blist[last] = b1;
            rlist[maxerr] = area2;
            rlist[last] = area1;
            elist[maxerr] = error2;
            elist[last] = error1;
        } else {
            alist[last] = a2;
            blist[maxerr] = b1;
            blist[last] = b2;
            elist[maxerr] = error1;
            elist[last] = error2;
        }
        qpsrt(limit, last, maxerr, errmax, elist, iord, nrmax);
        if (errsum <= errbnd) goto L170;
        if (ier != 0) goto L150;
        if (last == 2 && extall) {  // L120
            small = small * 0.5;
            numrl2 = numrl2 + 1;
            rlist2[numrl2] = area;
            ertest = errbnd;  // L130
            erlarg = errsum;
            continue;
        }
        if (noext) continue;
        if (extall) {
            erlarg = erlarg - erlast;
            if (std::abs(b1 - a1) > small) erlarg = erlarg + erro12;
            if (extrap) goto L70;
        }
        // L50
        width = std::abs(blist[maxerr] - alist[maxerr]);
        if (width > small) continue;
        if (!extall) {
            small = small * 0.5;
            if (0.25 * width * domega > 2.0) continue;
            extall = true;
            ertest = errbnd;  // L130
            erlarg = errsum;
            continue;
        }
        extrap = true;  // L60
        nrmax = 2;
    L70:
        if (!(ierro == 3 || erlarg <= ertest)) {
            jupbnd = last;
            if (last > (limit / 2 + 2)) jupbnd = (limit - last) + 3;
            id = nrmax;
            bool more = false;
            for (k = id; k <= jupbnd; ++k) {
                maxerr = iord[nrmax];
                errmax = elist[maxerr];
                if (std::abs(blist[maxerr] - alist[maxerr]) > small) {
                    more = true;
                    break;
                }
                nrmax = nrmax + 1;
            }
            if (more) continue;
        }
        numrl2 = numrl2 + 1;  // L90
        rlist2[numrl2] = area;
        if (numrl2 >= 3) {
            qelg(numrl2, rlist2, reseps, abseps, res3la, nres);
            ktmin = ktmin + 1;
            if (ktmin > 5 && abserr < 1.0e-3 * errsum) ier = 5;
            if (abseps < abserr) {
                ktmin = 0;
                abserr = abseps;
                result = reseps;
                correc = erlarg;
                ertest = std::max(epsabs, epsrel * std::abs(reseps));
                if (abserr <= ertest) goto L150;
            }
            if (numrl2 == 1) noext = true;  // L100
            if (ier == 5) goto L150;
        }
        maxerr = iord[1];  // L110
        errmax = elist[maxerr];
        nrmax = 1;
        extrap = false;
        small = small * 0.5;
        erlarg = errsum;
    }
L150:
    if (abserr == kOflow || nres == 0) goto L170;
    if (ier + ierro == 0) goto L165;
    if (ierro == 3) abserr = abserr + correc;
    if (ier == 0) ier = 3;
    if (result != 0.0 && area != 0.0) {
        if (abserr / std::abs(result) > errsum / std::abs(area)) goto L170;
        goto L165;
    }
    if (abserr > errsum) goto L170;
    if (area == 0.0) goto L190;
L165:
    if (ksgn == -1 && std::max(std::abs(result), std::abs(area)) <= defabs * 1.0e-2) goto L190;
    if (1.0e-2 > (result / area) || (result / area) > 1.0e2 || errsum >= std::abs(area)) ier = 6;
    goto L190;
L170:
    result = 0.0;
    for (k = 1; k <= last; ++k) result = result + rlist[k];
    abserr = errsum;
L190:
    if (ier > 2) ier = ier - 1;
L200:
    if (integr == 2 && omega < 0.0) result = -result;
}

// dqawfe's results beyond Work's: the integral, error and ier of each cycle (1-based, lst of them)
struct Cycles {
    std::vector<double> rslst, erlst;
    std::vector<int> ierlst;
    double result = 0.0, abserr = 0.0;
    std::int64_t neval = 0;
    int ier = 0, lst = 0;
};

// dqawfe: QAWF, the integral of f(x) cos(omega x) (integr 1) or sin(omega x) (2) on [a, inf): QAWO
// on the cycles [a + (k - 1) c, a + k c], c = (2 floor|omega| + 1) pi/|omega|, k = 1 ... limlst,
// and the epsilon algorithm on their partial sums. omega = 0 is QAGI of f on [a, inf) for the
// cosine (the original's dqagie(f, 0, 1, ...) integrates from 0 whatever a is; docs/design/
// integrate.md C78) and 0 for the sine. epsabs > 0, limlst >= 3 (checked by the caller).
template <class F>
void qawfe(
    F& f, double a, double omega, int integr, double epsabs, int limlst, int limit, int maxp1, Work& w, Cycles& r
) {
    constexpr double p = 0.9;
    constexpr double pi = 3.14159265358979323846264338327950;
    double abseps, c1, c2, correc, cycle, dl, drl, ep, eps, epsa, errsum, fact, p1, reseps;
    double psum[53], res3la[4] = {0.0, 0.0, 0.0, 0.0};
    int ktmin, ll = 0, nres, numrl2;
    r.rslst.assign(2, 0.0);
    r.erlst.assign(2, 0.0);
    r.ierlst.assign(2, 0);
    double& result = r.result;
    double& abserr = r.abserr;
    int& ier = r.ier;
    int& lst = r.lst;
    result = 0.0;
    abserr = 0.0;
    r.neval = 0;
    lst = 0;
    ier = 0;
    if (omega == 0.0) {
        if (integr == 1) {
            qagie(f, a, 1, epsabs, 0.0, limit, w);
            result = w.result;
            abserr = w.abserr;
            r.neval = w.neval;
            ier = w.ier;
        }
        r.rslst[1] = result;
        r.erlst[1] = abserr;
        r.ierlst[1] = ier;
        lst = 1;
        return;
    }
    {
        const auto l = static_cast<std::int64_t>(std::abs(omega));  // the original's integer l = dabs(omega)
        dl = static_cast<double>(2 * l + 1);
    }
    cycle = dl * pi / std::abs(omega);
    ier = 0;
    ktmin = 0;
    r.neval = 0;
    numrl2 = 0;
    nres = 0;
    c1 = a;
    c2 = cycle + a;
    p1 = 1.0 - p;
    eps = epsabs;
    if (epsabs > kUflow / p1) eps = epsabs * p1;
    ep = eps;
    fact = 1.0;
    correc = 0.0;
    abserr = 0.0;
    errsum = 0.0;
    for (lst = 1; lst <= limlst; ++lst) {
        epsa = eps * fact;
        qawoe(f, c1, c2, omega, integr, epsa, 0.0, limit, lst, maxp1, w);
        if (r.rslst.size() <= static_cast<std::size_t>(lst)) {
            r.rslst.resize(static_cast<std::size_t>(lst) + 1, 0.0);
            r.erlst.resize(static_cast<std::size_t>(lst) + 1, 0.0);
            r.ierlst.resize(static_cast<std::size_t>(lst) + 1, 0);
        }
        r.rslst[lst] = w.result;
        r.erlst[lst] = w.abserr;
        r.ierlst[lst] = w.ier;
        r.neval += w.neval;
        fact = fact * p;
        errsum = errsum + r.erlst[lst];
        drl = 50.0 * std::abs(r.rslst[lst]);
        if ((errsum + drl) <= epsabs && lst >= 6) goto L80;
        correc = std::max(correc, r.erlst[lst]);
        if (r.ierlst[lst] != 0) eps = std::max(ep, correc * p1);
        if (r.ierlst[lst] != 0) ier = 7;
        if (ier == 7 && (errsum + drl) <= correc * 10.0 && lst > 5) goto L80;
        numrl2 = numrl2 + 1;
        if (lst <= 1) {
            psum[1] = r.rslst[1];
        } else {
            psum[numrl2] = psum[ll] + r.rslst[lst];
            if (lst != 2) {
                if (lst == limlst) ier = 1;
                qelg(numrl2, psum, reseps, abseps, res3la, nres);
                ktmin = ktmin + 1;
                if (ktmin >= 15 && abserr <= 1.0e-3 * (errsum + drl)) ier = 4;
                if (!(abseps > abserr && lst != 3)) {
                    abserr = abseps;
                    result = reseps;
                    ktmin = 0;
                    if (std::fma(10.0, correc, abserr) <= epsabs || (abserr <= epsabs && 10.0 * correc >= epsabs))
                        goto L60;
                }
                if (ier != 0 && ier != 7) goto L60;  // L30
            }
        }
        ll = numrl2;  // L40
        c1 = c2;
        c2 = c2 + cycle;
    }
L60:
    abserr = std::fma(10.0, correc, abserr);
    if (ier == 0) return;
    if (result != 0.0 && psum[numrl2] != 0.0) {
        if (abserr / std::abs(result) > (errsum + drl) / std::abs(psum[numrl2])) goto L80;  // L70
        if (ier >= 1 && ier != 7) abserr = abserr + drl;
        return;
    }
    if (abserr > errsum) goto L80;
    if (psum[numrl2] == 0.0) return;
    if (abserr / std::abs(result) > (errsum + drl) / std::abs(psum[numrl2])) goto L80;  // L70
    if (ier >= 1 && ier != 7) abserr = abserr + drl;
    return;
L80:
    result = psum[numrl2];
    abserr = errsum + drl;
}

// dqmomo: the modified Chebyshev moments of the weights (x - a)^alfa (b - x)^beta on [a, b] and of
// their logarithms (ri, rj; rg, rh), 1-based of 25.
inline void qmomo(double alfa, double beta, double* ri, double* rj, double* rg, double* rh, int integr) {
    const double alfp1 = alfa + 1.0;
    const double betp1 = beta + 1.0;
    const double alfp2 = alfa + 2.0;
    const double betp2 = beta + 2.0;
    const double ralf = std::pow(2.0, alfp1);
    const double rbet = std::pow(2.0, betp1);
    ri[1] = ralf / alfp1;
    rj[1] = rbet / betp1;
    ri[2] = ri[1] * alfa / alfp2;
    rj[2] = rj[1] * beta / betp2;
    double an = 2.0, anm1 = 1.0;
    for (int i = 3; i <= 25; ++i) {
        ri[i] = -std::fma(an * (an - alfp2), ri[i - 1], ralf) / (anm1 * (an + alfp1));
        rj[i] = -std::fma(an * (an - betp2), rj[i - 1], rbet) / (anm1 * (an + betp1));
        anm1 = an;
        an = an + 1.0;
    }
    if (integr != 1) {
        if (integr != 3) {
            rg[1] = -ri[1] / alfp1;
            rg[2] = -(ralf + ralf) / (alfp2 * alfp2) - rg[1];
            an = 2.0;
            anm1 = 1.0;
            int im1 = 2;
            for (int i = 3; i <= 25; ++i) {
                rg[i] = -std::fma(anm1, ri[i], std::fma(an * (an - alfp2), rg[im1], -(an * ri[im1]))) /
                        (anm1 * (an + alfp1));
                anm1 = an;
                an = an + 1.0;
                im1 = i;
            }
        }
        if (integr != 2) {
            rh[1] = -rj[1] / betp1;
            rh[2] = -(rbet + rbet) / (betp2 * betp2) - rh[1];
            an = 2.0;
            anm1 = 1.0;
            int im1 = 2;
            for (int i = 3; i <= 25; ++i) {
                rh[i] = -std::fma(anm1, rj[i], std::fma(an * (an - betp2), rh[im1], -(an * rj[im1]))) /
                        (anm1 * (an + betp1));
                anm1 = an;
                an = an + 1.0;
                im1 = i;
            }
            for (int i = 2; i <= 25; i += 2) rh[i] = -rh[i];
        }
    }
    for (int i = 2; i <= 25; i += 2) rj[i] = -rj[i];
}

// dqc25s: the integral of f(x) w(x) on [bl, br] within [a, b], w dqwgts's weight: the modified
// 25-point Clenshaw-Curtis rule on an interval with the end point a or b (with alfa != 0 or a log
// of that end), else dqk15w. The netlib original's typo res24 = res12 + cheb24(i) rg(i) in the
// loop of the a end's log moments is corrected, as scipy's C port has it (docs/design/integrate.md
// §7.6, §7.11).
template <class F>
void qc25s(
    F& f,
    double a,
    double b,
    double bl,
    double br,
    double alfa,
    double beta,
    const double* ri,
    const double* rj,
    const double* rg,
    const double* rh,
    double& result,
    double& abserr,
    double& resasc,
    int integr,
    int& nev
) {
    double cheb12[14], cheb24[26], fval[26], centr, dc, factor, fix, hlgth, res12, res24, resabs, u;
    nev = 25;
    if (bl == a && (alfa != 0.0 || integr == 2 || integr == 4)) {
        // the end point a: the factor (b - x)^beta (and its log) into the values
        hlgth = 0.5 * (br - bl);
        centr = 0.5 * (br + bl);
        fix = b - centr;
        {
            double xs[25];
            int n = 0;
            xs[n++] = hlgth + centr;
            xs[n++] = centr;
            xs[n++] = centr - hlgth;
            for (int i = 2; i <= 12; ++i) {
                const double uu = hlgth * kXcc[i - 1];
                xs[n++] = uu + centr;
                xs[n++] = centr - uu;
            }
            prefetch(f, xs, n);
        }
        fval[1] = 0.5 * f(hlgth + centr) * std::pow(fix - hlgth, beta);
        fval[13] = f(centr) * std::pow(fix, beta);
        fval[25] = 0.5 * f(centr - hlgth) * std::pow(fix + hlgth, beta);
        for (int i = 2; i <= 12; ++i) {
            u = hlgth * kXcc[i - 1];
            const int isym = 26 - i;
            fval[i] = f(u + centr) * std::pow(fix - u, beta);
            fval[isym] = f(centr - u) * std::pow(fix + u, beta);
        }
        factor = std::pow(hlgth, alfa + 1.0);
        result = 0.0;
        abserr = 0.0;
        res12 = 0.0;
        res24 = 0.0;
        if (integr <= 2) {
            qcheb(kXcc, fval, cheb12, cheb24);
            for (int i = 1; i <= 13; ++i) {
                res12 = std::fma(cheb12[i], ri[i], res12);
                res24 = std::fma(cheb24[i], ri[i], res24);
            }
            for (int i = 14; i <= 25; ++i) res24 = std::fma(cheb24[i], ri[i], res24);
            if (integr != 1) {
                dc = std::log(br - bl);
                result = res24 * dc;
                abserr = std::abs((res24 - res12) * dc);
                res12 = 0.0;
                res24 = 0.0;
                for (int i = 1; i <= 13; ++i) {
                    res12 = std::fma(cheb12[i], rg[i], res12);
                    res24 = std::fma(cheb24[i], rg[i], res24);  // the original: res12 + cheb24(i) rg(i)
                }
                for (int i = 14; i <= 25; ++i) res24 = std::fma(cheb24[i], rg[i], res24);
            }
        } else {
            fval[1] = fval[1] * std::log(fix - hlgth);
            fval[13] = fval[13] * std::log(fix);
            fval[25] = fval[25] * std::log(fix + hlgth);
            for (int i = 2; i <= 12; ++i) {
                u = hlgth * kXcc[i - 1];
                const int isym = 26 - i;
                fval[i] = fval[i] * std::log(fix - u);
                fval[isym] = fval[isym] * std::log(fix + u);
            }
            qcheb(kXcc, fval, cheb12, cheb24);
            for (int i = 1; i <= 13; ++i) {
                res12 = std::fma(cheb12[i], ri[i], res12);
                res24 = std::fma(cheb24[i], ri[i], res24);
            }
            for (int i = 14; i <= 25; ++i) res24 = std::fma(cheb24[i], ri[i], res24);
            if (integr != 3) {
                dc = std::log(br - bl);
                result = res24 * dc;
                abserr = std::abs((res24 - res12) * dc);
                res12 = 0.0;
                res24 = 0.0;
                for (int i = 1; i <= 13; ++i) {
                    res12 = std::fma(cheb12[i], rg[i], res12);
                    res24 = std::fma(cheb24[i], rg[i], res24);
                }
                for (int i = 14; i <= 25; ++i) res24 = std::fma(cheb24[i], rg[i], res24);
            }
        }
        result = (result + res24) * factor;
        abserr = (abserr + std::abs(res24 - res12)) * factor;
        return;
    }
    if (br == b && (beta != 0.0 || integr == 3 || integr == 4)) {
        // the end point b: the factor (x - a)^alfa (and its log) into the values
        hlgth = 0.5 * (br - bl);
        centr = 0.5 * (br + bl);
        fix = centr - a;
        {
            double xs[25];
            int n = 0;
            xs[n++] = hlgth + centr;
            xs[n++] = centr;
            xs[n++] = centr - hlgth;
            for (int i = 2; i <= 12; ++i) {
                const double uu = hlgth * kXcc[i - 1];
                xs[n++] = uu + centr;
                xs[n++] = centr - uu;
            }
            prefetch(f, xs, n);
        }
        fval[1] = 0.5 * f(hlgth + centr) * std::pow(fix + hlgth, alfa);
        fval[13] = f(centr) * std::pow(fix, alfa);
        fval[25] = 0.5 * f(centr - hlgth) * std::pow(fix - hlgth, alfa);
        for (int i = 2; i <= 12; ++i) {
            u = hlgth * kXcc[i - 1];
            const int isym = 26 - i;
            fval[i] = f(u + centr) * std::pow(fix + u, alfa);
            fval[isym] = f(centr - u) * std::pow(fix - u, alfa);
        }
        factor = std::pow(hlgth, beta + 1.0);
        result = 0.0;
        abserr = 0.0;
        res12 = 0.0;
        res24 = 0.0;
        if (!(integr == 2 || integr == 4)) {
            qcheb(kXcc, fval, cheb12, cheb24);
            for (int i = 1; i <= 13; ++i) {
                res12 = std::fma(cheb12[i], rj[i], res12);
                res24 = std::fma(cheb24[i], rj[i], res24);
            }
            for (int i = 14; i <= 25; ++i) res24 = std::fma(cheb24[i], rj[i], res24);
            if (integr != 1) {
                dc = std::log(br - bl);
                result = res24 * dc;
                abserr = std::abs((res24 - res12) * dc);
                res12 = 0.0;
                res24 = 0.0;
                for (int i = 1; i <= 13; ++i) {
                    res12 = std::fma(cheb12[i], rh[i], res12);
                    res24 = std::fma(cheb24[i], rh[i], res24);
                }
                for (int i = 14; i <= 25; ++i) res24 = std::fma(cheb24[i], rh[i], res24);
            }
        } else {
            fval[1] = fval[1] * std::log(hlgth + fix);
            fval[13] = fval[13] * std::log(fix);
            fval[25] = fval[25] * std::log(fix - hlgth);
            for (int i = 2; i <= 12; ++i) {
                u = hlgth * kXcc[i - 1];
                const int isym = 26 - i;
                fval[i] = fval[i] * std::log(u + fix);
                fval[isym] = fval[isym] * std::log(fix - u);
            }
            qcheb(kXcc, fval, cheb12, cheb24);
            for (int i = 1; i <= 13; ++i) {
                res12 = std::fma(cheb12[i], rj[i], res12);
                res24 = std::fma(cheb24[i], rj[i], res24);
            }
            for (int i = 14; i <= 25; ++i) res24 = std::fma(cheb24[i], rj[i], res24);
            if (integr != 2) {
                dc = std::log(br - bl);
                result = res24 * dc;
                abserr = std::abs((res24 - res12) * dc);
                res12 = 0.0;
                res24 = 0.0;
                for (int i = 1; i <= 13; ++i) {
                    res12 = std::fma(cheb12[i], rh[i], res12);
                    res24 = std::fma(cheb24[i], rh[i], res24);
                }
                for (int i = 14; i <= 25; ++i) res24 = std::fma(cheb24[i], rh[i], res24);
            }
        }
        result = (result + res24) * factor;
        abserr = (abserr + std::abs(res24 - res12)) * factor;
        return;
    }
    const WeightS ws{a, b, alfa, beta, integr};
    qk15w(f, ws, bl, br, result, abserr, resabs, resasc);
    nev = 15;
}

// dqawse: QAWS, the integral of f(x) w(x) on [a, b], a < b, w = (x - a)^alfa (b - x)^beta times
// log(x - a) (integr 2), log(b - x) (3) or both (4); alfa, beta > -1, limit >= 2 (checked by the
// caller).
template <class F>
void qawse(
    F& f, double a, double b, double alfa, double beta, int integr, double epsabs, double epsrel, int limit, Work& w
) {
    w.reset(limit);
    w.grow(2);
    double ri[26], rj[26], rg[26], rh[26];
    double area, area1, area12, area2, a1, a2, b1, b2, centre, errbnd, errmax, error1, erro12, error2, errsum, resas1,
        resas2;
    int iroff1, iroff2, k, maxerr, nev, nrmax;
    auto& alist = w.alist;
    auto& blist = w.blist;
    auto& rlist = w.rlist;
    auto& elist = w.elist;
    auto& iord = w.iord;
    int& ier = w.ier;
    int& last = w.last;
    double& result = w.result;
    double& abserr = w.abserr;
    w.neval = 0;
    last = 0;
    rlist[1] = 0.0;
    elist[1] = 0.0;
    iord[1] = 0;
    result = 0.0;
    abserr = 0.0;
    ier = 0;
    qmomo(alfa, beta, ri, rj, rg, rh, integr);
    centre = 0.5 * (b + a);
    qc25s(f, a, b, a, centre, alfa, beta, ri, rj, rg, rh, area1, error1, resas1, integr, nev);
    w.neval = nev;
    qc25s(f, a, b, centre, b, alfa, beta, ri, rj, rg, rh, area2, error2, resas2, integr, nev);
    last = 2;
    w.neval += nev;
    result = area1 + area2;
    abserr = error1 + error2;
    errbnd = std::max(epsabs, epsrel * std::abs(result));
    if (error2 > error1) {
        alist[1] = centre;
        alist[2] = a;
        blist[1] = b;
        blist[2] = centre;
        rlist[1] = area2;
        rlist[2] = area1;
        elist[1] = error2;
        elist[2] = error1;
    } else {
        alist[1] = a;
        alist[2] = centre;
        blist[1] = centre;
        blist[2] = b;
        rlist[1] = area1;
        rlist[2] = area2;
        elist[1] = error1;
        elist[2] = error2;
    }
    iord[1] = 1;
    iord[2] = 2;
    if (limit == 2) ier = 1;
    if (abserr <= errbnd || ier == 1) return;
    errmax = elist[1];
    maxerr = 1;
    nrmax = 1;
    area = result;
    errsum = abserr;
    iroff1 = 0;
    iroff2 = 0;
    for (last = 3; last <= limit; ++last) {
        w.grow(last);
        a1 = alist[maxerr];
        b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
        a2 = b1;
        b2 = blist[maxerr];
        qc25s(f, a, b, a1, b1, alfa, beta, ri, rj, rg, rh, area1, error1, resas1, integr, nev);
        w.neval += nev;
        qc25s(f, a, b, a2, b2, alfa, beta, ri, rj, rg, rh, area2, error2, resas2, integr, nev);
        w.neval += nev;
        area12 = area1 + area2;
        erro12 = error1 + error2;
        errsum = errsum + erro12 - errmax;
        area = area + area12 - rlist[maxerr];
        if (!(a == a1 || b == b2) && !(resas1 == error1 || resas2 == error2)) {
            if (std::abs(rlist[maxerr] - area12) < 1.0e-5 * std::abs(area12) && erro12 >= 0.99 * errmax)
                iroff1 = iroff1 + 1;
            if (last > 10 && erro12 > errmax) iroff2 = iroff2 + 1;
        }
        rlist[maxerr] = area1;
        rlist[last] = area2;
        errbnd = std::max(epsabs, epsrel * std::abs(area));
        if (!(errsum <= errbnd)) {
            if (last == limit) ier = 1;
            if (iroff1 >= 6 || iroff2 >= 20) ier = 2;
            if (std::max(std::abs(a1), std::abs(b2)) <= (1.0 + 100.0 * kEpmach) * (std::abs(a2) + 1000.0 * kUflow))
                ier = 3;
        }
        if (error2 > error1) {
            alist[maxerr] = a2;
            alist[last] = a1;
            blist[last] = b1;
            rlist[maxerr] = area2;
            rlist[last] = area1;
            elist[maxerr] = error2;
            elist[last] = error1;
        } else {
            alist[last] = a2;
            blist[maxerr] = b1;
            blist[last] = b2;
            elist[maxerr] = error1;
            elist[last] = error2;
        }
        qpsrt(limit, last, maxerr, errmax, elist, iord, nrmax);
        if (ier != 0 || errsum <= errbnd) break;
    }
    result = 0.0;
    for (k = 1; k <= last; ++k) result = result + rlist[k];
    abserr = errsum;
}

// dqc25c: the integral of f(x)/(x - c) on [a, b]: the modified 25-point Clenshaw-Curtis rule when
// c is near or inside (|2c - b - a| < 1.1 (b - a)), else dqk15w; krul counts (down) the intervals
// where the rule of the modified moments was not used, as the original.
template <class F>
void qc25c(F& f, double a, double b, double c, double& result, double& abserr, int& krul, int& neval) {
    double cheb12[14], cheb24[26], fval[26], ak22, amom0, amom1, amom2, centr, hlgth, res12, res24, resabs, resasc, u;
    const double cc = (std::fma(2.0, c, -b) - a) / (b - a);
    if (std::abs(cc) >= 1.1) {
        krul = krul - 1;
        const WeightC wc{c};
        qk15w(f, wc, a, b, result, abserr, resabs, resasc);
        neval = 15;
        if (resasc == abserr) krul = krul + 1;
        return;
    }
    hlgth = 0.5 * (b - a);
    centr = 0.5 * (b + a);
    neval = 25;
    {
        double xs[25];
        int n = 0;
        xs[n++] = hlgth + centr;
        xs[n++] = centr;
        xs[n++] = centr - hlgth;
        for (int i = 2; i <= 12; ++i) {
            const double uu = hlgth * kXcc[i - 1];
            xs[n++] = uu + centr;
            xs[n++] = centr - uu;
        }
        prefetch(f, xs, n);
    }
    fval[1] = 0.5 * f(hlgth + centr);
    fval[13] = f(centr);
    fval[25] = 0.5 * f(centr - hlgth);
    for (int i = 2; i <= 12; ++i) {
        u = hlgth * kXcc[i - 1];
        const int isym = 26 - i;
        fval[i] = f(u + centr);
        fval[isym] = f(centr - u);
    }
    qcheb(kXcc, fval, cheb12, cheb24);
    amom0 = std::log(std::abs((1.0 - cc) / (1.0 + cc)));
    amom1 = std::fma(cc, amom0, 2.0);
    res12 = std::fma(cheb12[1], amom0, cheb12[2] * amom1);
    res24 = std::fma(cheb24[1], amom0, cheb24[2] * amom1);
    for (int k = 3; k <= 13; ++k) {
        amom2 = std::fma(2.0 * cc, amom1, -amom0);
        ak22 = static_cast<double>((k - 2) * (k - 2));
        if ((k / 2) * 2 == k) amom2 = amom2 - 4.0 / (ak22 - 1.0);
        res12 = std::fma(cheb12[k], amom2, res12);
        res24 = std::fma(cheb24[k], amom2, res24);
        amom0 = amom1;
        amom1 = amom2;
    }
    for (int k = 14; k <= 25; ++k) {
        amom2 = std::fma(2.0 * cc, amom1, -amom0);
        ak22 = static_cast<double>((k - 2) * (k - 2));
        if ((k / 2) * 2 == k) amom2 = amom2 - 4.0 / (ak22 - 1.0);
        res24 = std::fma(cheb24[k], amom2, res24);
        amom0 = amom1;
        amom1 = amom2;
    }
    result = res24;
    abserr = std::abs(res24 - res12);
}

// dqawce: QAWC, the Cauchy principal value of the integral of f(x)/(x - c) on [a, b], c != a, b
// (checked by the caller).
template <class F>
void qawce(F& f, double a, double b, double c, double epsabs, double epsrel, int limit, Work& w) {
    w.reset(limit);
    double aa, area, area1, area12, area2, a1, a2, b1, b2, bb, errbnd, errmax, error1, erro12, error2, errsum;
    int iroff1, iroff2, k, krule, maxerr, nev, nrmax, neval0 = 0;
    auto& alist = w.alist;
    auto& blist = w.blist;
    auto& rlist = w.rlist;
    auto& elist = w.elist;
    auto& iord = w.iord;
    int& ier = w.ier;
    int& last = w.last;
    double& result = w.result;
    double& abserr = w.abserr;
    w.neval = 0;
    last = 0;
    alist[1] = a;
    blist[1] = b;
    rlist[1] = 0.0;
    elist[1] = 0.0;
    iord[1] = 0;
    result = 0.0;
    abserr = 0.0;
    aa = a;
    bb = b;
    if (!(a <= b)) {
        aa = b;
        bb = a;
    }
    ier = 0;
    krule = 1;
    qc25c(f, aa, bb, c, result, abserr, krule, neval0);
    w.neval = neval0;
    last = 1;
    rlist[1] = result;
    elist[1] = abserr;
    iord[1] = 1;
    alist[1] = a;
    blist[1] = b;
    errbnd = std::max(epsabs, epsrel * std::abs(result));
    if (limit == 1) ier = 1;
    if (abserr < std::min(1.0e-2 * std::abs(result), errbnd) || ier == 1) goto L70;
    alist[1] = aa;
    blist[1] = bb;
    rlist[1] = result;
    errmax = abserr;
    maxerr = 1;
    area = result;
    errsum = abserr;
    nrmax = 1;
    iroff1 = 0;
    iroff2 = 0;
    for (last = 2; last <= limit; ++last) {
        w.grow(last);
        a1 = alist[maxerr];
        b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
        b2 = blist[maxerr];
        if (c <= b1 && c > a1) b1 = 0.5 * (c + b2);
        if (c > b1 && c < b2) b1 = 0.5 * (a1 + c);
        a2 = b1;
        krule = 2;
        qc25c(f, a1, b1, c, area1, error1, krule, nev);
        w.neval += nev;
        qc25c(f, a2, b2, c, area2, error2, krule, nev);
        w.neval += nev;
        area12 = area1 + area2;
        erro12 = error1 + error2;
        errsum = errsum + erro12 - errmax;
        area = area + area12 - rlist[maxerr];
        if (std::abs(rlist[maxerr] - area12) < 1.0e-5 * std::abs(area12) && erro12 >= 0.99 * errmax && krule == 0)
            iroff1 = iroff1 + 1;
        if (last > 10 && erro12 > errmax && krule == 0) iroff2 = iroff2 + 1;
        rlist[maxerr] = area1;
        rlist[last] = area2;
        errbnd = std::max(epsabs, epsrel * std::abs(area));
        if (!(errsum <= errbnd)) {
            if (iroff1 >= 6 && iroff2 > 20) ier = 2;
            if (last == limit) ier = 1;
            if (std::max(std::abs(a1), std::abs(b2)) <= (1.0 + 100.0 * kEpmach) * (std::abs(a2) + 1000.0 * kUflow))
                ier = 3;
        }
        if (error2 > error1) {
            alist[maxerr] = a2;
            alist[last] = a1;
            blist[last] = b1;
            rlist[maxerr] = area2;
            rlist[last] = area1;
            elist[maxerr] = error2;
            elist[last] = error1;
        } else {
            alist[last] = a2;
            blist[maxerr] = b1;
            blist[last] = b2;
            elist[maxerr] = error1;
            elist[last] = error2;
        }
        qpsrt(limit, last, maxerr, errmax, elist, iord, nrmax);
        if (ier != 0 || errsum <= errbnd) break;
    }
    result = 0.0;
    for (k = 1; k <= last; ++k) result = result + rlist[k];
    abserr = errsum;
L70:
    if (aa == b) result = -result;
}

}  // namespace sd::integrate::quadpack
