// minimize(method='l-bfgs-b') (docs/design/optimize.md §3.5): L-BFGS-B 3.0 (Byrd, Lu, Nocedal and
// Zhu 1995; Morales and Nocedal 2011) as scipy 1.18.1 runs it: its C translation
// (scipy/optimize/src/lbfgsb.c, setulb and mainlb with their subroutines) driven by
// _minimize_lbfgsb, the same steps (O9).
//
// Each iteration finds the generalized Cauchy point of the quadratic model on the projected
// steepest-descent path (cauchy: the breakpoints in a heap, hpsolb; the compact form of the
// limited-memory matrix, bmv), minimizes the model over the free variables (formk: the
// factorized middle matrix K; cmprlb: the reduced gradient; subsm: the subspace step, projected,
// and backtracked to the box when it is not a descent direction), and searches the line to it
// with MINPACK-2's dcsrch (ftol 1e-3, gtol 0.9, xtol 0.1, the step at most to the box,
// max_ls trials; too many trials restore the point and restart without memory, or stop with
// ABNORMAL when the memory is already empty). The pair (s, y) updates the memory unless
// s^T y <= eps (-g^T s) (matupd, formt). It stops when the projected gradient's largest entry is
// at most pgtol = gtol, when (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= factr eps = ftol, or at
// scipy's driver caps: max_iter iterations, more than max_nfev evaluations (tested after an
// iteration).
//
// The BLAS calls are scipy's (Accelerate's ddot, dnrm2, daxpy, dscal, dcopy; dpotrf and dtrtrs
// for the Cholesky factors), as are scipy's departures from the Fortran original (the 2-norm by
// dnrm2, the Cauchy point's test for the last breakpoint, which never holds in 0-based indices).
// scipy compiles lbfgsb.c with clang's default -ffp-contract=on, which fuses a * b + c within an
// expression into one rounding; this file asks for the same contraction (the pragma below), so
// the expressions here, written as scipy's, round as scipy's do.
// One result differs from scipy's (O59): after ABNORMAL, x and g go back to the last iterate and
// so does f here; scipy's driver passes f to setulb by value and keeps the last trial's value.
#include "optimize/minimize.hpp"

#include "optimize/objective.hpp"

#include "bind/errors.hpp"
#include "linalg/lapack.hpp"
#include "linalg/types.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#pragma clang fp contract(on)

namespace sd::optimize {

namespace {

using vec = std::vector<double>;
using linalg::lint;
using ivec = std::vector<lint>;
namespace lapack = linalg::lapack;

constexpr double epsmach = 2.220446049250313e-016;

enum Status : int {
    START = 0,
    NEW_X = 1,
    RESTART = 2,
    FG = 3,
    CONVERGENCE = 4,
    STOP = 5,
    WARNING = 6,
    ERROR = 7,
    ABNORMAL = 8
};
enum StatusMsg : int {
    NO_MSG = 0,
    FG_START = 301,
    FG_LNSRCH = 302,
    CONV_GRAD = 401,
    CONV_F = 402,
    STOP_ITER = 502,
    STOP_ITERC = 504,
    WARN_ROUND = 601,
    WARN_STPMAX = 602,
    WARN_STPMIN = 603,
    WARN_XTOL = 604,
    ERROR_STP1 = 706,
    ERROR_STP2 = 707,
    ERROR_INITG = 710,
};

double ddot(lint n, const double* x, const double* y) { return cblas_ddot(n, x, 1, y, 1); }
double dnrm2(lint n, const double* x) { return cblas_dnrm2(n, x, 1); }
void daxpy(lint n, double a, const double* x, double* y) { cblas_daxpy(n, a, x, 1, y, 1); }
void dscal(lint n, double a, double* x) { cblas_dscal(n, a, x, 1); }
void dcopy(lint n, const double* x, double* y) { cblas_dcopy(n, x, 1, y, 1); }

// ---- dcsrch and dcstep (MINPACK-2, scipy's C translation)

struct Dcsrch {
    bool brackt = false;
    int stage = 0;
    double ginit = 0, gtest = 0, gx = 0, gy = 0, finit = 0, fx = 0, fy = 0, stx = 0, sty = 0, stmin = 0, stmax = 0;
    double width = 0, width1 = 0;
};

void dcstep(
    double* stx,
    double* fx,
    double* dx,
    double* sty,
    double* fy,
    double* dy,
    double* stp,
    double fp,
    double dp,
    bool* brackt,
    double stpmin,
    double stpmax
) {
    double gamma, p, q, r, s, sgnd, stpc, stpf, stpq, theta;
    sgnd = dp * (*dx / std::fabs(*dx));
    if (fp > *fx) {
        theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        s = std::fmax(std::fabs(theta), std::fmax(std::fabs(*dx), std::fabs(dp)));
        gamma = s * std::sqrt(std::pow(theta / s, 2.0) - (*dx / s) * (dp / s));
        if (*stp < *stx) gamma = -gamma;
        p = (gamma - *dx) + theta;
        q = ((gamma - *dx) + gamma) + dp;
        r = p / q;
        stpc = *stx + r * (*stp - *stx);
        stpq = *stx + ((*dx / ((*fx - fp) / (*stp - *stx) + *dx)) / 2.0) * (*stp - *stx);
        if (std::fabs(stpc - *stx) < std::fabs(stpq - *stx)) stpf = stpc;
        else stpf = stpc + (stpq - stpc) / 2.0;
        *brackt = true;
    } else if (sgnd < 0.0) {
        theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        s = std::fmax(std::fabs(theta), std::fmax(std::fabs(*dx), std::fabs(dp)));
        gamma = s * std::sqrt(std::pow(theta / s, 2.0) - (*dx / s) * (dp / s));
        if (*stp > *stx) gamma = -gamma;
        p = (gamma - dp) + theta;
        q = ((gamma - dp) + gamma) + *dx;
        r = p / q;
        stpc = *stp + r * (*stx - *stp);
        stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        if (std::fabs(stpc - *stp) > std::fabs(stpq - *stp)) stpf = stpc;
        else stpf = stpq;
        *brackt = true;
    } else if (std::fabs(dp) < std::fabs(*dx)) {
        theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        s = std::fmax(std::fabs(theta), std::fmax(std::fabs(*dx), std::fabs(dp)));
        gamma = s * std::sqrt(std::fmax(0.0, std::pow(theta / s, 2.0) - (*dx / s) * (dp / s)));
        if (*stp > *stx) gamma = -gamma;
        p = (gamma - dp) + theta;
        q = (gamma + (*dx - dp)) + gamma;
        r = p / q;
        if ((r < 0.0) && (gamma != 0.0)) stpc = *stp + r * (*stx - *stp);
        else if (*stp > *stx) stpc = stpmax;
        else stpc = stpmin;
        stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        if (*brackt) {
            if (std::fabs(stpc - *stp) < std::fabs(stpq - *stp)) stpf = stpc;
            else stpf = stpq;
            if (*stp > *stx) stpf = std::fmin(*stp + 0.66 * (*sty - *stp), stpf);
            else stpf = std::fmax(*stp + 0.66 * (*sty - *stp), stpf);
        } else {
            if (std::fabs(stpc - *stp) > std::fabs(stpq - *stp)) stpf = stpc;
            else stpf = stpq;
            stpf = std::fmin(stpmax, stpf);
            stpf = std::fmax(stpmin, stpf);
        }
    } else {
        if (*brackt) {
            theta = 3.0 * (fp - *fy) / (*sty - *stp) + *dy + dp;
            s = std::fmax(std::fabs(theta), std::fmax(std::fabs(*dy), std::fabs(dp)));
            gamma = s * std::sqrt(std::pow(theta / s, 2.0) - (*dy / s) * (dp / s));
            if (*stp > *sty) gamma = -gamma;
            p = (gamma - dp) + theta;
            q = ((gamma - dp) + gamma) + *dy;
            r = p / q;
            stpc = *stp + r * (*sty - *stp);
            stpf = stpc;
        } else if (*stp > *stx) {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
    }
    if (fp > *fx) {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if (sgnd < 0.0) {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }
    *stp = stpf;
}

void dcsrch(
    double f,
    double g,
    double* stp,
    double ftol,
    double gtol,
    double xtol,
    double stpmin,
    double stpmax,
    int* task,
    int* task_msg,
    Dcsrch& S
) {
    const double xtrapl = 1.1, xtrapu = 4.0;
    if (*task == START) {
        if (*stp < stpmin) {
            *task = ERROR;
            *task_msg = ERROR_STP1;
        }
        if (*stp > stpmax) {
            *task = ERROR;
            *task_msg = ERROR_STP2;
        }
        if (g >= 0.0) {
            *task = ERROR;
            *task_msg = ERROR_INITG;
        }
        if (*task == ERROR) return;
        S.brackt = false;
        S.stage = 1;
        S.finit = f;
        S.ginit = g;
        S.gtest = ftol * S.ginit;
        S.width = stpmax - stpmin;
        S.width1 = S.width / 0.5;
        S.stx = 0.0;
        S.fx = S.finit;
        S.gx = S.ginit;
        S.sty = 0.0;
        S.fy = S.finit;
        S.gy = S.ginit;
        S.stmin = 0.0;
        S.stmax = *stp + xtrapu * (*stp);
        *task = FG;
        *task_msg = NO_MSG;
        return;
    }
    const double ftest = S.finit + (*stp) * S.gtest;
    if ((S.stage == 1) && (f <= ftest) && (g >= 0.0)) S.stage = 2;
    if ((S.brackt) && ((*stp <= S.stmin) || (*stp >= S.stmax))) {
        *task = WARNING;
        *task_msg = WARN_ROUND;
    }
    if ((S.brackt) && ((S.stmax - S.stmin) <= xtol * S.stmax)) {
        *task = WARNING;
        *task_msg = WARN_XTOL;
    }
    if ((*stp == stpmax) && (f <= ftest) && (g <= S.gtest)) {
        *task = WARNING;
        *task_msg = WARN_STPMAX;
    }
    if ((*stp == stpmin) && ((f > ftest) || (g >= S.gtest))) {
        *task = WARNING;
        *task_msg = WARN_STPMIN;
    }
    if ((f <= ftest) && (std::fabs(g) <= gtol * (-S.ginit))) *task = CONVERGENCE;
    if ((*task == WARNING) || (*task == CONVERGENCE)) return;
    if ((S.stage == 1) && (f <= S.fx) && (f > ftest)) {
        const double fm = f - (*stp) * S.gtest;
        double fxm = S.fx - S.stx * S.gtest;
        double fym = S.fy - S.sty * S.gtest;
        const double gm = g - S.gtest;
        double gxm = S.gx - S.gtest;
        double gym = S.gy - S.gtest;
        dcstep(&S.stx, &fxm, &gxm, &S.sty, &fym, &gym, stp, fm, gm, &S.brackt, S.stmin, S.stmax);
        S.fx = fxm + S.stx * S.gtest;
        S.fy = fym + S.sty * S.gtest;
        S.gx = gxm + S.gtest;
        S.gy = gym + S.gtest;
    } else {
        dcstep(&S.stx, &S.fx, &S.gx, &S.sty, &S.fy, &S.gy, stp, f, g, &S.brackt, S.stmin, S.stmax);
    }
    if (S.brackt) {
        if (std::fabs(S.sty - S.stx) >= 0.66 * S.width1) *stp = S.stx + 0.5 * (S.sty - S.stx);
        S.width1 = S.width;
        S.width = std::fabs(S.sty - S.stx);
    }
    if (S.brackt) {
        S.stmin = std::fmin(S.stx, S.sty);
        S.stmax = std::fmax(S.stx, S.sty);
    } else {
        S.stmin = *stp + xtrapl * (*stp - S.stx);
        S.stmax = *stp + xtrapu * (*stp - S.stx);
    }
    *stp = std::fmax(*stp, stpmin);
    *stp = std::fmin(*stp, stpmax);
    if (((S.brackt) && (*stp <= S.stmin || *stp >= S.stmax)) || ((S.brackt) && (S.stmax - S.stmin <= xtol * S.stmax)))
        *stp = S.stx;
    *task = FG;
    *task_msg = NO_MSG;
}

// ---- the subroutines of mainlb (0-based; ws, wy n x m and sy, ss, wt m x m, wn, snd 2m x 2m,
// column-major as in scipy)

void active(
    lint n,
    const double* l,
    const double* u,
    const lint* nbd,
    double* x,
    lint* iwhere,
    bool* prjctd,
    bool* cnstnd,
    bool* boxed
) {
    *prjctd = false;
    *cnstnd = false;
    *boxed = true;
    for (lint i = 0; i < n; i++) {
        if (nbd[i] > 0) {
            if ((nbd[i] <= 2) && (x[i] <= l[i])) {
                if (x[i] < l[i]) {
                    *prjctd = true;
                    x[i] = l[i];
                }
            } else if ((nbd[i] >= 2) && (x[i] >= u[i])) {
                if (x[i] > u[i]) {
                    *prjctd = true;
                    x[i] = u[i];
                }
            }
        }
    }
    for (lint i = 0; i < n; i++) {
        if (nbd[i] != 2) *boxed = false;
        if (nbd[i] == 0) {
            iwhere[i] = -1;
        } else {
            *cnstnd = true;
            if ((nbd[i] == 2) && (u[i] - l[i] <= 0.0)) iwhere[i] = 3;
            else iwhere[i] = 0;
        }
    }
}

void bmv(lint m, const double* sy, double* wt, lint col, const double* v, double* p, lint* info) {
    if (col == 0) return;
    p[col] = v[col];
    for (lint i = 1; i < col; i++) {
        const lint i2 = col + i;
        double ssum = 0.0;
        for (lint k = 0; k <= i - 1; k++) ssum = ssum + sy[i + m * k] * v[k] / sy[k + m * k];
        p[i2] = v[i2] + ssum;
    }
    *info = lapack::trtrs('U', 'T', 'N', col, 1, wt, m, &p[col], col);
    if (*info != 0) return;
    for (lint i = 0; i < col; i++) p[i] = v[i] / std::sqrt(sy[i + m * i]);
    *info = lapack::trtrs('U', 'N', 'N', col, 1, wt, m, &p[col], col);
    if (*info != 0) return;
    for (lint i = 0; i < col; i++) p[i] = -p[i] / std::sqrt(sy[i + m * i]);
    for (lint i = 0; i < col; i++) {
        double ssum = 0.0;
        for (lint k = i + 1; k < col; k++) ssum = ssum + sy[k + m * i] * p[col + k] / sy[i + m * i];
        p[i] = p[i] + ssum;
    }
}

void hpsolb(lint n, double* t, lint* iorder, lint iheap) {
    lint i, j, k, indxin, indxout;
    double ddum, out;
    if (iheap == 0) {
        for (k = 2; k <= n + 1; k++) {
            ddum = t[k - 1];
            indxin = iorder[k - 1];
            i = k;
            while (true) {
                if (i > 1) {
                    j = i / 2;
                    if (ddum < t[j - 1]) {
                        t[i - 1] = t[j - 1];
                        iorder[i - 1] = iorder[j - 1];
                        i = j;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            t[i - 1] = ddum;
            iorder[i - 1] = indxin;
        }
    }
    if (n > 0) {
        i = 1;
        out = t[0];
        indxout = iorder[0];
        ddum = t[n];
        indxin = iorder[n];
        while (true) {
            j = i + i;
            if (j <= n) {
                if (t[j] < t[j - 1]) j++;
                if (t[j - 1] < ddum) {
                    t[i - 1] = t[j - 1];
                    iorder[i - 1] = iorder[j - 1];
                    i = j;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        t[i - 1] = ddum;
        iorder[i - 1] = indxin;
        t[n] = out;
        iorder[n] = indxout;
    }
}

void cauchy(
    lint n,
    const double* x,
    const double* l,
    const double* u,
    const lint* nbd,
    const double* g,
    lint* iorder,
    lint* iwhere,
    double* t,
    double* d,
    double* xcp,
    lint m,
    const double* wy,
    const double* ws,
    const double* sy,
    double* wt,
    double theta,
    lint col,
    lint head,
    double* p,
    double* c,
    double* wbp,
    double* v,
    lint* nseg,
    double sbgnrm,
    lint* info
) {
    bool bnded, xlower, xupper;
    lint i, j, col2, nfree, nbreak, pointr, ibp, nleft, ibkmin, iter;
    double f1, f2, dt, dtm, tsum, dibp, zibp, dibp2, bkmin, tu, tl, wmc, wmp, wmw, tj, tj0, neggi, f2_org;
    f1 = 0.0;
    tl = 0.0;
    tu = 0.0;
    if (sbgnrm <= 0.0) {
        dcopy(n, x, xcp);
        return;
    }
    bnded = true;
    nfree = n;
    nbreak = -1;
    ibkmin = 0;
    bkmin = 0.0;
    col2 = 2 * col;
    for (i = 0; i < col2; i++) p[i] = 0.0;
    for (i = 0; i < n; i++) {
        neggi = -g[i];
        if ((iwhere[i] != 3) && (iwhere[i] != -1)) {
            if (nbd[i] <= 2) tl = x[i] - l[i];
            if (nbd[i] >= 2) tu = u[i] - x[i];
            xlower = ((nbd[i] <= 2) && (tl <= 0.0));
            xupper = ((nbd[i] >= 2) && (tu <= 0.0));
            iwhere[i] = 0;
            if (xlower) {
                if (neggi <= 0.0) iwhere[i] = 1;
            } else if (xupper) {
                if (neggi >= 0.0) iwhere[i] = 2;
            } else {
                if (std::fabs(neggi) <= 0.0) iwhere[i] = -3;
            }
        }
        pointr = head;
        if ((iwhere[i] != 0) && (iwhere[i] != -1)) {
            d[i] = 0.0;
        } else {
            d[i] = neggi;
            f1 = f1 - neggi * neggi;
            for (j = 0; j < col; j++) {
                p[j] = p[j] + wy[i + n * pointr] * neggi;
                p[col + j] = p[col + j] + ws[i + n * pointr] * neggi;
                pointr = (pointr + 1) % m;
            }
            if ((nbd[i] <= 2) && (nbd[i] != 0) && (neggi < 0.0)) {
                nbreak += 1;
                iorder[nbreak] = i;
                t[nbreak] = tl / (-neggi);
                if ((nbreak == 0) || (t[nbreak] < bkmin)) {
                    bkmin = t[nbreak];
                    ibkmin = nbreak;
                }
            } else if ((nbd[i] >= 2) && (neggi > 0.0)) {
                nbreak += 1;
                iorder[nbreak] = i;
                t[nbreak] = tu / neggi;
                if ((nbreak == 0) || (t[nbreak] < bkmin)) {
                    bkmin = t[nbreak];
                    ibkmin = nbreak;
                }
            } else {
                nfree -= 1;
                iorder[nfree] = i;
                if (std::fabs(neggi) > 0.0) bnded = false;
            }
        }
    }
    if (theta != 1.0) dscal(col, theta, &p[col]);
    dcopy(n, x, xcp);
    if ((nbreak == -1) && (nfree == n)) return;
    for (j = 0; j < col2; j++) c[j] = 0.0;
    f2 = -(theta)*f1;
    f2_org = f2;
    if (col > 0) {
        bmv(m, sy, wt, col, p, v, info);
        if (*info != 0) return;
        f2 = f2 - ddot(col2, v, p);
    }
    dtm = -f1 / f2;
    tsum = 0.0;
    *nseg = 1;
    nleft = nbreak;
    iter = 0;
    tj = 0.0;
    while (nbreak > -1) {
        tj0 = tj;
        if (iter == 0) {
            tj = bkmin;
            ibp = iorder[ibkmin];
        } else {
            if (iter == 1) {
                if (ibkmin != nbreak) {
                    t[ibkmin] = t[nbreak];
                    iorder[ibkmin] = iorder[nbreak];
                }
            }
            hpsolb(nleft, t, iorder, iter - 1);
            tj = t[nleft];
            ibp = iorder[nleft];
        }
        dt = tj - tj0;
        if (dtm < dt) {
            if (dtm <= 0.0) dtm = 0.0;
            tsum = tsum + dtm;
            daxpy(n, tsum, d, xcp);
            if (col > 0) daxpy(col2, dtm, p, c);
            return;
        }
        tsum = tsum + dt;
        nleft -= 1;
        iter += 1;
        dibp = d[ibp];
        d[ibp] = 0.0;
        if (dibp > 0.0) {
            zibp = u[ibp] - x[ibp];
            xcp[ibp] = u[ibp];
            iwhere[ibp] = 2;
        } else {
            zibp = l[ibp] - x[ibp];
            xcp[ibp] = l[ibp];
            iwhere[ibp] = 1;
        }
        if (nleft == -1 && nbreak == n) {
            dtm = dt;
            if (col > 0) daxpy(col2, dtm, p, c);
            return;
        }
        *nseg += 1;
        dibp2 = std::pow(dibp, 2.0);
        f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp;
        f2 = f2 - theta * dibp2;
        if (col > 0) {
            daxpy(col2, dt, p, c);
            pointr = head;
            for (j = 0; j < col; j++) {
                wbp[j] = wy[ibp + n * pointr];
                wbp[col + j] = theta * ws[ibp + n * pointr];
                pointr = (pointr + 1) % m;
            }
            bmv(m, sy, wt, col, wbp, v, info);
            if (*info != 0) return;
            wmc = ddot(col2, c, v);
            wmp = ddot(col2, p, v);
            wmw = ddot(col2, wbp, v);
            dibp = -dibp;
            daxpy(col2, dibp, wbp, p);
            dibp = -dibp;
            f1 = f1 + dibp * wmc;
            f2 = f2 + dibp * 2.0 * wmp - dibp2 * wmw;
        }
        f2 = std::fmax(epsmach * f2_org, f2);
        if (nleft >= 0) {
            dtm = -f1 / f2;
        } else if (bnded) {
            dtm = 0.0;
            break;
        } else {
            dtm = -f1 / f2;
            break;
        }
    }
    if (dtm <= 0.0) dtm = 0.0;
    tsum = tsum + dtm;
    daxpy(n, tsum, d, xcp);
    if (col > 0) daxpy(col2, dtm, p, c);
}

void cmprlb(
    lint n,
    lint m,
    const double* x,
    const double* g,
    const double* ws,
    const double* wy,
    const double* sy,
    double* wt,
    const double* z,
    double* r,
    double* wa,
    const lint* index,
    double theta,
    lint col,
    lint head,
    lint nfree,
    bool cnstnd,
    lint* info
) {
    if ((!cnstnd) && (col > 0)) {
        for (lint i = 0; i < n; i++) r[i] = -g[i];
    } else {
        for (lint i = 0; i < nfree; i++) {
            const lint k = index[i];
            r[i] = -theta * (z[k] - x[k]) - g[k];
        }
        bmv(m, sy, wt, col, &wa[2 * m], wa, info);
        if (*info != 0) {
            *info = -8;
            return;
        }
        lint pointr = head;
        for (lint j = 0; j < col; j++) {
            const double a1 = wa[j];
            const double a2 = theta * wa[col + j];
            for (lint i = 0; i < nfree; i++) {
                const lint k = index[i];
                r[i] = r[i] + wy[k + n * pointr] * a1 + ws[k + n * pointr] * a2;
            }
            pointr = (pointr + 1) % m;
        }
    }
}

void formk(
    lint n,
    lint nsub,
    const lint* ind,
    lint nenter,
    lint ileave,
    const lint* indx2,
    lint iupdat,
    bool updatd,
    double* wn,
    double* wn1,
    lint m,
    const double* ws,
    const double* wy,
    const double* sy,
    double theta,
    lint col,
    lint head,
    lint* info
) {
    lint m2, ipntr, jpntr, iy, is, jy, js, is1, js1, k1, i, k, col2, pbegin, pend, dbegin, dend, upcl, temp_int;
    double temp1, temp2, temp3, temp4;
    if (updatd) {
        if (iupdat > m) {
            for (jy = 0; jy < m - 1; jy++) {
                js = m + jy;
                temp_int = m - (jy + 1);
                dcopy(temp_int, &wn1[(jy + 1) + 2 * m * (jy + 1)], &wn1[jy + 2 * m * jy]);
                dcopy(temp_int, &wn1[(js + 1) + 2 * m * (js + 1)], &wn1[js + 2 * m * js]);
                temp_int = m - 1;
                dcopy(temp_int, &wn1[(m + 1) + 2 * m * (jy + 1)], &wn1[m + 2 * m * jy]);
            }
        }
        pbegin = 0;
        pend = nsub;
        dbegin = nsub;
        dend = n;
        iy = col - 1;
        is = m + col - 1;
        ipntr = head + col - 1;
        if (ipntr >= m) ipntr -= m;
        jpntr = head;
        for (jy = 0; jy < col; jy++) {
            js = m + jy;
            temp1 = 0.0;
            temp2 = 0.0;
            temp3 = 0.0;
            for (k = pbegin; k < pend; k++) {
                k1 = ind[k];
                temp1 = temp1 + wy[k1 + n * ipntr] * wy[k1 + n * jpntr];
            }
            for (k = dbegin; k < dend; k++) {
                k1 = ind[k];
                temp2 = temp2 + ws[k1 + n * ipntr] * ws[k1 + n * jpntr];
                temp3 = temp3 + ws[k1 + n * ipntr] * wy[k1 + n * jpntr];
            }
            wn1[iy + 2 * m * jy] = temp1;
            wn1[is + 2 * m * js] = temp2;
            wn1[is + 2 * m * jy] = temp3;
            jpntr = (jpntr + 1) % m;
        }
        jy = col - 1;
        jpntr = head + col - 1;
        if (jpntr >= m) jpntr -= m;
        ipntr = head;
        for (i = 0; i < col; i++) {
            is = m + i;
            temp3 = 0.0;
            for (k = pbegin; k < pend; k++) {
                k1 = ind[k];
                temp3 = temp3 + ws[k1 + n * ipntr] * wy[k1 + n * jpntr];
            }
            ipntr = (ipntr + 1) % m;
            wn1[is + 2 * m * jy] = temp3;
        }
        upcl = col - 1;
    } else {
        upcl = col;
    }
    ipntr = head;
    for (iy = 0; iy < upcl; iy++) {
        is = m + iy;
        jpntr = head;
        for (jy = 0; jy <= iy; jy++) {
            js = m + jy;
            temp1 = 0.0;
            temp2 = 0.0;
            temp3 = 0.0;
            temp4 = 0.0;
            for (k = 0; k < nenter; k++) {
                k1 = indx2[k];
                temp1 = temp1 + wy[k1 + n * ipntr] * wy[k1 + n * jpntr];
                temp2 = temp2 + ws[k1 + n * ipntr] * ws[k1 + n * jpntr];
            }
            for (k = ileave; k < n; k++) {
                k1 = indx2[k];
                temp3 = temp3 + wy[k1 + n * ipntr] * wy[k1 + n * jpntr];
                temp4 = temp4 + ws[k1 + n * ipntr] * ws[k1 + n * jpntr];
            }
            wn1[iy + 2 * m * jy] = wn1[iy + 2 * m * jy] + temp1 - temp3;
            wn1[is + 2 * m * js] = wn1[is + 2 * m * js] - temp2 + temp4;
            jpntr = (jpntr + 1) % m;
        }
        ipntr = (ipntr + 1) % m;
    }
    ipntr = head;
    for (is = m; is < m + upcl; is++) {
        jpntr = head;
        for (jy = 0; jy < upcl; jy++) {
            temp1 = 0.0;
            temp3 = 0.0;
            for (k = 0; k < nenter; k++) {
                k1 = indx2[k];
                temp1 = temp1 + ws[k1 + n * ipntr] * wy[k1 + n * jpntr];
            }
            for (k = ileave; k < n; k++) {
                k1 = indx2[k];
                temp3 = temp3 + ws[k1 + n * ipntr] * wy[k1 + n * jpntr];
            }
            if (is <= jy + m) wn1[is + 2 * m * jy] = wn1[is + 2 * m * jy] + temp1 - temp3;
            else wn1[is + 2 * m * jy] = wn1[is + 2 * m * jy] - temp1 + temp3;
            jpntr = (jpntr + 1) % m;
        }
        ipntr = (ipntr + 1) % m;
    }
    m2 = 2 * m;
    for (iy = 0; iy < col; iy++) {
        is = col + iy;
        is1 = m + iy;
        for (jy = 0; jy <= iy; jy++) {
            js = col + jy;
            js1 = m + jy;
            wn[jy + m2 * iy] = wn1[iy + m2 * jy] / theta;
            wn[js + m2 * is] = wn1[is1 + m2 * js1] * theta;
        }
        for (jy = 0; jy < iy; jy++) wn[jy + m2 * is] = -wn1[is1 + m2 * jy];
        for (jy = iy; jy < col; jy++) wn[jy + m2 * is] = wn1[is1 + m2 * jy];
        wn[iy + m2 * iy] = wn[iy + m2 * iy] + sy[iy + m * iy];
    }
    *info = lapack::potrf('U', col, wn, m2);
    if (*info != 0) {
        *info = -1;
        return;
    }
    col2 = 2 * col;
    *info = lapack::trtrs('U', 'T', 'N', col, col, wn, m2, &wn[m2 * col], m2);
    for (is = col; is < col2; is++)
        for (js = is; js < col2; js++) wn[is + m2 * js] = wn[is + m2 * js] + ddot(col, &wn[m2 * is], &wn[m2 * js]);
    *info = lapack::potrf('U', col, &wn[col + m2 * col], m2);
    if (*info != 0) *info = -2;
}

void formt(lint m, double* wt, const double* sy, const double* ss, lint col, double theta, lint* info) {
    for (lint j = 0; j < col; j++) wt[m * j] = theta * ss[m * j];
    for (lint i = 1; i < col; i++) {
        for (lint j = i; j < col; j++) {
            const lint k1 = (i > j ? j : i);
            double ddum = 0.0;
            for (lint k = 0; k < k1; k++) ddum = ddum + ((sy[i + m * k] * sy[j + m * k]) / sy[k + m * k]);
            wt[i + m * j] = ddum + theta * ss[i + m * j];
        }
    }
    *info = lapack::potrf('U', col, wt, m);
    if (*info != 0) *info = -3;
}

void freev(
    lint n,
    lint* nfree,
    lint* idx,
    lint* nenter,
    lint* ileave,
    lint* idx2,
    const lint* iwhere,
    bool* wrk,
    bool updatd,
    bool cnstnd,
    lint iter
) {
    *nenter = 0;
    *ileave = n;
    if ((iter > 0) && cnstnd) {
        for (lint i = 0; i < *nfree; i++) {
            const lint k = idx[i];
            if (iwhere[k] > 0) {
                *ileave -= 1;
                idx2[*ileave] = k;
            }
        }
        for (lint i = *nfree; i < n; i++) {
            const lint k = idx[i];
            if (iwhere[k] <= 0) {
                idx2[*nenter] = k;
                *nenter += 1;
            }
        }
    }
    *wrk = ((*ileave < n) || (*nenter > 0) || updatd);
    *nfree = 0;
    lint iact = n;
    for (lint i = 0; i < n; i++) {
        if (iwhere[i] <= 0) {
            idx[*nfree] = i;
            *nfree += 1;
        } else {
            iact -= 1;
            idx[iact] = i;
        }
    }
}

void matupd(
    lint n,
    lint m,
    double* ws,
    double* wy,
    double* sy,
    double* ss,
    const double* d,
    const double* r,
    lint* itail,
    lint iupdat,
    lint* col,
    lint* head,
    double* theta,
    double rr,
    double dr,
    double stp,
    double dtd
) {
    if (iupdat <= m) {
        *col = iupdat;
        *itail = (*head + iupdat - 1) % m;
    } else {
        *itail = (*itail + 1) % m;
        *head = (*head + 1) % m;
    }
    dcopy(n, d, &ws[(*itail) * n]);
    dcopy(n, r, &wy[(*itail) * n]);
    *theta = rr / dr;
    if (iupdat > m) {
        for (lint j = 1; j < *col; j++) {
            dcopy(j, &ss[1 + m * j], &ss[m * (j - 1)]);
            dcopy(*col - j, &sy[j + m * j], &sy[(j - 1) + m * (j - 1)]);
        }
    }
    lint pointr = *head;
    for (lint j = 0; j < *col - 1; j++) {
        sy[*col - 1 + m * j] = ddot(n, d, &wy[pointr * n]);
        ss[j + m * (*col - 1)] = ddot(n, &ws[pointr * n], &d[0]);
        pointr = (pointr + 1) % m;
    }
    ss[(*col - 1) + m * (*col - 1)] = (stp == 1.0 ? dtd : stp * stp * dtd);
    sy[(*col - 1) + m * (*col - 1)] = dr;
}

void projgr(
    lint n, const double* l, const double* u, const lint* nbd, const double* x, const double* g, double* sbgnrm
) {
    *sbgnrm = 0.0;
    for (lint i = 0; i < n; i++) {
        double gi = g[i];
        if (gi != gi) {
            *sbgnrm = gi;
            return;
        }
        if (nbd[i] != 0) {
            if (gi < 0.0) {
                if (nbd[i] >= 2) gi = std::fmax(x[i] - u[i], gi);
            } else {
                if (nbd[i] <= 2) gi = std::fmin(x[i] - l[i], gi);
            }
        }
        *sbgnrm = std::fmax(*sbgnrm, std::fabs(gi));
    }
}

void subsm(
    lint n,
    lint m,
    lint nsub,
    const lint* ind,
    const double* l,
    const double* u,
    const lint* nbd,
    double* x,
    double* d,
    double* xp,
    const double* ws,
    const double* wy,
    double theta,
    const double* xx,
    const double* gg,
    lint col,
    lint head,
    lint* iword,
    double* wv,
    double* wn,
    lint* info
) {
    lint pointr, m2, col2, ibd, jy, js, i, j, k;
    double alpha, xk, dk, temp, temp1, temp2, dd_p;
    if (nsub <= 0) return;
    pointr = head;
    for (i = 0; i < col; i++) {
        temp1 = 0.0;
        temp2 = 0.0;
        for (j = 0; j < nsub; j++) {
            k = ind[j];
            temp1 += wy[k + n * pointr] * d[j];
            temp2 += ws[k + n * pointr] * d[j];
        }
        wv[i] = temp1;
        wv[col + i] = theta * temp2;
        pointr = (pointr + 1) % m;
    }
    m2 = 2 * m;
    col2 = 2 * col;
    *info = lapack::trtrs('U', 'T', 'N', col2, 1, wn, m2, wv, m2);
    if (*info != 0) return;
    for (i = 0; i < col; i++) wv[i] = -wv[i];
    *info = lapack::trtrs('U', 'N', 'N', col2, 1, wn, m2, wv, m2);
    if (*info != 0) return;
    pointr = head;
    for (jy = 0; jy < col; jy++) {
        js = col + jy;
        for (i = 0; i < nsub; i++) {
            k = ind[i];
            d[i] = d[i] + (wy[k + n * pointr] * wv[jy] / theta) + (ws[k + n * pointr] * wv[js]);
        }
        pointr = (pointr + 1) % m;
    }
    temp = 1.0 / theta;
    dscal(nsub, temp, d);
    *iword = 0;
    dcopy(n, x, xp);
    for (i = 0; i < nsub; i++) {
        k = ind[i];
        dk = d[i];
        xk = x[k];
        if (nbd[k] != 0) {
            if (nbd[k] == 1) {
                x[k] = std::fmax(l[k], xk + dk);
                if (x[k] == l[k]) *iword = 1;
            } else {
                if (nbd[k] == 2) {
                    xk = std::fmax(l[k], xk + dk);
                    x[k] = std::fmin(u[k], xk);
                    if ((x[k] == l[k]) || (x[k] == u[k])) *iword = 1;
                } else {
                    if (nbd[k] == 3) {
                        x[k] = std::fmin(u[k], xk + dk);
                        if (x[k] == u[k]) *iword = 1;
                    }
                }
            }
        } else {
            x[k] = xk + dk;
        }
    }
    if (*iword == 0) return;
    dd_p = 0.0;
    for (i = 0; i < n; i++) dd_p = dd_p + (x[i] - xx[i]) * gg[i];
    if (dd_p > 0.0) {
        dcopy(n, xp, x);
    } else {
        return;
    }
    alpha = 1.0;
    temp1 = 1.0;
    ibd = 0;
    for (i = 0; i < nsub; i++) {
        k = ind[i];
        dk = d[i];
        if (nbd[k] != 0) {
            if ((dk < 0.0) && (nbd[k] <= 2)) {
                temp2 = l[k] - x[k];
                if (temp2 >= 0.0) temp1 = 0.0;
                else if (dk * alpha < temp2) temp1 = temp2 / dk;
            } else if ((dk > 0.0) && (nbd[k] >= 2)) {
                temp2 = u[k] - x[k];
                if (temp2 <= 0.0) temp1 = 0.0;
                else if (dk * alpha > temp2) temp1 = temp2 / dk;
            }
            if (temp1 < alpha) {
                alpha = temp1;
                ibd = i;
            }
        }
    }
    if (alpha < 1.0) {
        dk = d[ibd];
        k = ind[ibd];
        if (dk > 0.0) {
            x[k] = u[k];
            d[ibd] = 0.0;
        } else if (dk < 0.0) {
            x[k] = l[k];
            d[ibd] = 0.0;
        }
    }
    for (i = 0; i < nsub; i++) {
        k = ind[i];
        x[k] = x[k] + alpha * d[i];
    }
}

// ---- mainlb with scipy's driver loop (_minimize_lbfgsb) around it

// A part of the workspace (Lbfgsb::buf).
struct Part {
    double* p = nullptr;
    double* data() const { return p; }
    double& operator[](std::size_t i) const { return p[i]; }
};

struct Lbfgsb {
    Objective& obj;
    lint n, m;
    vec x;
    vec l, u;
    ivec nbd;
    double factr, pgtol;
    lint maxls;
    std::int64_t maxiter, maxfun;

    // scipy's workspace wa, partitioned as setulb does: the parts start where scipy's do, so they
    // have scipy's 16-byte alignment (Accelerate's ddot sums in an order that depends on it).
    vec buf;
    Part ws, wy, sy, ss, wt, wn, snd, z, r, d, t, xp, wa;
    vec g;
    ivec index, iwhere, indx2;
    double f = 0;

    bool prjctd = false, cnstnd = false, boxed = false, updatd = false, wrk = false;
    lint nintol = 0, iback = 0, nskip = 0, head = 0, col = 0, iter = 0, itail = 0, iupdat = 0;
    lint nseg = 0, nfgv = 0, info = 0, ifun = 0, iword = 0, nfree = 0, nact = 0, ileave = 0, nenter = 0;
    double theta = 1.0, fold = 0, dr = 0, rr = 0, tol = 0, xstep = 0, sbgnrm = 0, stpmx = 0, dnorm = 0, dtd = 0;
    double gd = 0, gdold = 0, stp = 0;
    int task = START, task_msg = NO_MSG;
    int ls_task = START, ls_msg = NO_MSG;
    Dcsrch ls;
    std::int64_t n_iterations = 0;
    bool big = false;

    Lbfgsb(Objective& o, lint m_) : obj(o), n(static_cast<lint>(o.n)), m(m_) {
        const auto un = static_cast<std::size_t>(n), um = static_cast<std::size_t>(m);
        buf.assign(2 * um * un + 5 * un + 11 * um * um + 8 * um, 0.0);
        std::size_t at = 0;
        for (auto [part, size] :
             {std::pair<Part*, std::size_t>{&ws, um * un},
              {&wy, um * un},
              {&sy, um * um},
              {&ss, um * um},
              {&wt, um * um},
              {&wn, 4 * um * um},
              {&snd, 4 * um * um},
              {&z, un},
              {&r, un},
              {&d, un},
              {&t, un},
              {&xp, un},
              {&wa, 8 * um}}) {
            part->p = buf.data() + at;
            at += size;
        }
        g.assign(un, 0.0);
        index.assign(un, 0);
        iwhere.assign(un, 0);
        indx2.assign(un, 0);
        big = 10.0 * double(n) * double(m + 1) > 2e5;  // the 10 us rule
    }

    void evaluate() {
        obj.value_and_gradient(x.data());
        f = obj.f;
        g = obj.g;
    }

    void reset() {
        info = 0;
        col = 0;
        head = 0;
        theta = 1.0;
        iupdat = 0;
        updatd = false;
    }

    // The search direction: the Cauchy point and the subspace minimization (no Python calls).
    // false when a factorization failed and the memory was reset (scipy's `continue`).
    bool direction() {
        iword = -1;
        if (cnstnd || col == 0) {
            cauchy(
                n, x.data(), l.data(), u.data(), nbd.data(), g.data(), indx2.data(), iwhere.data(), t.data(), d.data(),
                z.data(), m, wy.data(), ws.data(), sy.data(), wt.data(), theta, col, head, wa.data(),
                &wa[static_cast<std::size_t>(2 * m)], &wa[static_cast<std::size_t>(4 * m)],
                &wa[static_cast<std::size_t>(6 * m)], &nseg, sbgnrm, &info
            );
            if (info != 0) {
                reset();
                return false;
            }
            nintol += nseg;
            freev(n, &nfree, index.data(), &nenter, &ileave, indx2.data(), iwhere.data(), &wrk, updatd, cnstnd, iter);
            nact = n - nfree;
        } else {
            dcopy(n, x.data(), z.data());
            wrk = updatd;
            nseg = 0;
        }
        if ((nfree != 0) && (col != 0)) {
            if (wrk)
                formk(
                    n, nfree, index.data(), nenter, ileave, indx2.data(), iupdat, updatd, wn.data(), snd.data(), m,
                    ws.data(), wy.data(), sy.data(), theta, col, head, &info
                );
            if (info != 0) {
                reset();
                return false;
            }
            cmprlb(
                n, m, x.data(), g.data(), ws.data(), wy.data(), sy.data(), wt.data(), z.data(), r.data(), wa.data(),
                index.data(), theta, col, head, nfree, cnstnd, &info
            );
            if (info == 0)
                subsm(
                    n, m, nfree, index.data(), l.data(), u.data(), nbd.data(), z.data(), r.data(), xp.data(), ws.data(),
                    wy.data(), theta, x.data(), g.data(), col, head, &iword, wa.data(), wn.data(), &info
                );
            if (info != 0) {
                reset();
                return false;
            }
        }
        for (lint i = 0; i < n; i++)
            d[static_cast<std::size_t>(i)] = z[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)];
        return true;
    }

    // lnsrlb; first: the line search starts (scipy: task_msg is not FG_LNSRCH).
    void lnsrlb(bool first) {
        const double ftol = 1e-3, gtol = 0.9, xtol = 0.1;
        if (first) {
            dnorm = dnrm2(n, d.data());
            dtd = std::pow(dnorm, 2.0);
            stpmx = 1.0e+10;
            if (cnstnd) {
                if (iter == 0) {
                    stpmx = 1.0;
                } else {
                    for (lint i = 0; i < n; i++) {
                        const auto ui = static_cast<std::size_t>(i);
                        const double a1 = d[ui];
                        if (nbd[ui] != 0) {
                            if ((a1 < 0.0) && (nbd[ui] <= 2)) {
                                const double a2 = l[ui] - x[ui];
                                if (a2 >= 0.0) stpmx = 0.0;
                                else if (a1 * stpmx < a2) stpmx = a2 / a1;
                            } else if ((a1 > 0.0) && (nbd[ui] >= 2)) {
                                const double a2 = u[ui] - x[ui];
                                if (a2 <= 0.0) stpmx = 0.0;
                                else if (a1 * stpmx > a2) stpmx = a2 / a1;
                            }
                        }
                    }
                }
            }
            if ((iter == 0) && (!boxed)) stp = std::fmin(1.0 / dnorm, stpmx);
            else stp = 1.0;
            std::copy(x.begin(), x.end(), t.data());
            std::copy(g.begin(), g.end(), r.data());
            fold = f;
            ifun = 0;
            iback = 0;
            ls_task = START;
            ls_msg = NO_MSG;
        }
        gd = ddot(n, g.data(), d.data());
        if (ifun == 0) {
            gdold = gd;
            if (gd >= 0.0) {
                info = -4;
                return;
            }
        }
        dcsrch(f, gd, &stp, ftol, gtol, xtol, 0.0, stpmx, &ls_task, &ls_msg, ls);
        xstep = stp * dnorm;
        if ((ls_task != CONVERGENCE) && (ls_task != WARNING)) {
            task = FG;
            task_msg = FG_LNSRCH;
            ifun += 1;
            nfgv += 1;
            iback = ifun - 1;
            if (stp == 1.0) {
                dcopy(n, z.data(), x.data());
            } else {
                for (lint i = 0; i < n; i++) {
                    const auto ui = static_cast<std::size_t>(i);
                    x[ui] = stp * d[ui] + t[ui];
                    if ((nbd[ui] == 1) || (nbd[ui] == 2)) x[ui] = std::fmax(x[ui], l[ui]);
                    if ((nbd[ui] == 2) || (nbd[ui] == 3)) x[ui] = std::fmin(x[ui], u[ui]);
                }
            }
        } else {
            task = NEW_X;
            task_msg = NO_MSG;
        }
    }

    // The memory update after an iteration (no Python calls); false: skipped or reset.
    bool update() {
        for (lint i = 0; i < n; i++) {
            const auto ui = static_cast<std::size_t>(i);
            r[ui] = g[ui] - r[ui];
        }
        rr = std::pow(dnrm2(n, r.data()), 2.0);
        double ddum;
        if (stp == 1.0) {
            dr = gd - gdold;
            ddum = -gdold;
        } else {
            dr = (gd - gdold) * stp;
            dscal(n, stp, d.data());
            ddum = -gdold * stp;
        }
        if (dr <= epsmach * ddum) {
            nskip += 1;
            updatd = false;
            return false;
        }
        updatd = true;
        iupdat += 1;
        matupd(
            n, m, ws.data(), wy.data(), sy.data(), ss.data(), d.data(), r.data(), &itail, iupdat, &col, &head, &theta,
            rr, dr, stp, dtd
        );
        formt(m, wt.data(), sy.data(), ss.data(), col, theta, &info);
        if (info != 0) {
            reset();
            return false;
        }
        return true;
    }

    void run() {
        tol = factr * epsmach;
        nfree = n;
        active(n, l.data(), u.data(), nbd.data(), x.data(), iwhere.data(), &prjctd, &cnstnd, &boxed);
        task = FG;
        task_msg = FG_START;
        evaluate();
        obj.require_finite_start();
        nfgv = 1;
        projgr(n, l.data(), u.data(), nbd.data(), x.data(), g.data(), &sbgnrm);
        if (sbgnrm <= pgtol) {
            task = CONVERGENCE;
            task_msg = CONV_GRAD;
            return;
        }
        for (;;) {
            bool ok;
            {
                std::optional<nb::gil_scoped_release> rel;
                if (big) rel.emplace();
                ok = direction();
            }
            if (!ok) continue;
            bool restart = false;
            for (bool first = true;; first = false) {
                lnsrlb(first);
                if ((info != 0) || (iback >= maxls)) {
                    std::copy(t.data(), t.data() + n, x.begin());
                    std::copy(r.data(), r.data() + n, g.begin());
                    f = fold;
                    if (col == 0) {
                        if (info == 0) {
                            info = -9;
                            nfgv--;
                            ifun--;
                            iback--;
                        }
                        task = ABNORMAL;
                        task_msg = NO_MSG;
                        iter++;
                        return;
                    }
                    if (info == 0) nfgv--;
                    reset();
                    task = RESTART;
                    task_msg = NO_MSG;
                    restart = true;
                    break;
                }
                if ((task == FG) && (task_msg == FG_LNSRCH)) {
                    evaluate();
                    continue;
                }
                iter++;
                projgr(n, l.data(), u.data(), nbd.data(), x.data(), g.data(), &sbgnrm);
                // the driver at NEW_X (_minimize_lbfgsb)
                n_iterations += 1;
                if (n_iterations >= maxiter) {
                    task = STOP;
                    task_msg = STOP_ITERC;
                    return;
                }
                if (obj.nfev > maxfun) {
                    task = STOP;
                    task_msg = STOP_ITER;
                    return;
                }
                break;
            }
            if (restart) continue;
            if (sbgnrm <= pgtol) {
                task = CONVERGENCE;
                task_msg = CONV_GRAD;
                return;
            }
            const double ddum = std::fmax(std::fmax(std::fabs(fold), std::fabs(f)), 1.0);
            if ((fold - f) <= tol * ddum) {
                task = CONVERGENCE;
                task_msg = CONV_F;
                if (iback >= 10) info = -5;
                return;
            }
            {
                std::optional<nb::gil_scoped_release> rel;
                if (big) rel.emplace();
                update();
            }
        }
    }
};

// lbfgsb_solve(func, fun, jac, check, mode, x0, lb, ub, m, factr, pgtol, steps, max_nfev, max_iter, max_ls)
//   -> (x, fun, jac, task, task_msg, n_iter, nfev, njev, S (m, n), Y (m, n), n_corrs)
// lb, ub: the bounds (+-inf for none); mode, steps: as bfgs_solve. task, task_msg: scipy's
// codes (task CONVERGENCE, STOP, ABNORMAL; its message number). S, Y: the stored corrections in
// the order scipy keeps them (its hess_inv uses the first n_corrs rows).
nb::object lbfgsb_solve(
    const std::string& func,
    nb::callable fun,
    nb::object jac,
    nb::callable check,
    int mode,
    const vec& x0,
    const vec& lb,
    const vec& ub,
    std::int64_t m,
    double factr,
    double pgtol,
    const vec& steps,
    std::int64_t max_nfev,
    std::int64_t max_iter,
    std::int64_t max_ls
) {
    const char* fn = func.c_str();
    return call_named(fn, [&]() -> nb::object {
        const std::size_t n = x0.size();
        Objective obj;
        obj.fun = fun;
        obj.jac = jac;
        obj.check = check;
        obj.fn = fn;
        obj.n = n;
        obj.mode = mode;
        obj.steps = steps;
        obj.lb = lb;
        obj.ub = ub;
        Lbfgsb L(obj, static_cast<lint>(m));
        L.x = x0;
        L.l.assign(n, 0.0);
        L.u.assign(n, 0.0);
        L.nbd.assign(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const bool lo = std::isfinite(lb[i]), hi = std::isfinite(ub[i]);
            if (lo) L.l[i] = lb[i];
            if (hi) L.u[i] = ub[i];
            L.nbd[i] = lo ? (hi ? 2 : 1) : (hi ? 3 : 0);
        }
        L.factr = factr;
        L.pgtol = pgtol;
        L.maxls = static_cast<lint>(max_ls);
        L.maxiter = max_iter;
        L.maxfun = max_nfev;
        L.run();
        const lint ncorr = std::min(L.iupdat, L.m);
        return nb::make_tuple(
            out_vector(L.x.data(), n), L.f, out_vector(L.g.data(), n), L.task, L.task_msg, L.n_iterations, obj.nfev,
            obj.ngev, out_matrix(L.ws.data(), static_cast<std::size_t>(L.m), n),
            out_matrix(L.wy.data(), static_cast<std::size_t>(L.m), n), static_cast<std::int64_t>(ncorr)
        );
    });
}

}  // namespace

void bind_lbfgsb(nb::module_& m) {
    m.def(
        "lbfgsb_solve", &lbfgsb_solve, nb::arg("func"), nb::arg("fun"), nb::arg("jac").none(), nb::arg("check"),
        nb::arg("mode"), nb::arg("x0"), nb::arg("lb"), nb::arg("ub"), nb::arg("m"), nb::arg("factr"), nb::arg("pgtol"),
        nb::arg("steps"), nb::arg("max_nfev"), nb::arg("max_iter"), nb::arg("max_ls")
    );
}

}  // namespace sd::optimize
