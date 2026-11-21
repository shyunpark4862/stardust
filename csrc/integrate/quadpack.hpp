// QUADPACK's adaptive routines of quad without a weight (docs/design/integrate.md §3.1, decision
// C5): dqagse (QAGS, a finite range), dqagie (QAGI, an infinite range mapped onto (0, 1]) and dqagpe
// (QAGP, break points), with their parts dqk21, dqk15i, dqpsrt and dqelg. Each is a statement by
// statement translation of the netlib Fortran original (Piessens, de Doncker-Kapenga, Ueberhuber,
// Kahaner 1983), with its control flow (the labels are the original's), its order of operations
// and its decimal literals, so that the points where the integrand is called and the results are
// scipy 1.18.1's (its C port of the same Fortran) bit for bit.
//
// scipy's C port is built with clang's default -ffp-contract=on, which fuses a * b + c within one
// expression into a fused multiply-add; this build has contraction off (API.md §4, D-33), so the
// fused operations are written out with std::fma at those places: the Kronrod rules' four
// running sums (resg, resk, resabs, resasc: s = s + w * v). The points (centr -+ hlgth * xgk) are
// two statements in the original and are not fused, nor is anything in the drivers or dqelg (no
// product meets a sum within one expression there). docs/design/integrate.md §7.10 has the
// comparison with scipy's records that settles this.
//
// The arrays are 1-based like the original's (element 0 unused). The integrand is any callable
// double(double); a C++ exception from it (a Python error) leaves the routine by unwinding.
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
    std::vector<int> iord, level, ndin;
    double result = 0.0, abserr = 0.0;
    std::int64_t neval = 0;
    int ier = 0, last = 0;

    void reset(int limit, int npts2 = 0) {
        const auto n = static_cast<std::size_t>(std::min(limit, 64)) + 1;
        alist.assign(n, 0.0);
        blist.assign(n, 0.0);
        rlist.assign(n, 0.0);
        elist.assign(n, 0.0);
        iord.assign(n, 0);
        level.assign(n, 0);
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
    }
};

// ---- the rules

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

}  // namespace sd::integrate::quadpack
