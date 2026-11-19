"""Build and run the Fortran originals of MINPACK and L-BFGS-B 3.0 and compare them with scipy
1.18.1's ports on the same problems (docs/design/optimize.md §7, decision O13). Writes the
fixed data of tests/optimize/data/:

  minpack_drivers.txt     the summary tables of MINPACK's four test drivers, run unchanged
  lbfgsb_drivers.txt      the output of L-BFGS-B 3.0's driver1-3 (f77), run unchanged
  minpack_functions.json  SSQFCN/SSQJAC and VECFCN/VECJAC at each driver run's starting point
                          and at a second point, to check tests/optimize/mgh.py and systems.py
  fortran_minpack.json    LMDER1, LMDIF1, HYBRJ1, HYBRD1 on the drivers' runs, full precision
  fortran_lbfgsb.json     setulb with scipy's driver loop on MINPACK's least squares problems
                          (f = sum F^2) with and without bounds, and on driver1's problem
  fortran_vs_scipy.json   for each run: the same run in scipy (leastsq, fsolve, minimize
                          L-BFGS-B with the drivers' options) and where the two first differ

The sources are not part of this repository. They are, from netlib and from Nocedal's page:

  https://www.netlib.org/minpack/ex/file05   MINPACK, double precision (CHKDER ... RWUPDT)
  https://www.netlib.org/minpack/ex/file15   test driver for HYBRD1 (with VECFCN, INITPT)
  https://www.netlib.org/minpack/ex/file16   test driver for HYBRJ1 (with VECFCN, VECJAC, INITPT)
  https://www.netlib.org/minpack/ex/file17   test driver for LMDER1 (with SSQFCN, SSQJAC, INITPT)
  https://www.netlib.org/minpack/ex/file19   test driver for LMDIF1 (with SSQFCN, INITPT)
  https://www.netlib.org/minpack/ex/file21   data for the HYBRD/HYBRJ drivers
  https://www.netlib.org/minpack/ex/file22   data for the LMDER/LMDIF drivers
  https://users.iems.northwestern.edu/~nocedal/Software/Lbfgsb.3.0.tar.gz

Put the seven netlib files in one directory and unpack the tarball there or in a
subdirectory (it makes Lbfgsb.3.0/). file05 has no DPMPAR (MINPACK's machine constants); this script supplies one
from Fortran's EPSILON, TINY and HUGE. Everything is compiled with

    gfortran -O2 -ffp-contract=off -std=legacy

(no fused multiply-add, so that each operation rounds once, as in Python). The drivers' own
programs are run as they are; for full precision this script writes its own drivers (below),
which call the same MINPACK routines and the drivers' test-function subroutines, extracted
from file16 and file17 (INITPT renamed per driver so that both can be linked).

Run:  uv run --no-sync python tools/gen_optimize_fortran.py SRC_DIR BUILD_DIR
      (BUILD_DIR is created; SRC_DIR is only read.)
"""

import json
import math
import pathlib
import re
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "tests" / "optimize" / "data"
FC = "/opt/homebrew/bin/gfortran"
FFLAGS = ["-O2", "-ffp-contract=off", "-std=legacy", "-w"]
EPS = float(np.finfo(float).eps)
TOL = math.sqrt(EPS)  # the drivers' TOL = DSQRT(DPMPAR(1))

DPMPAR = """\
      DOUBLE PRECISION FUNCTION DPMPAR(I)
      INTEGER I
      IF (I .EQ. 1) DPMPAR = EPSILON(1.0D0)
      IF (I .EQ. 2) DPMPAR = TINY(1.0D0)
      IF (I .EQ. 3) DPMPAR = HUGE(1.0D0)
      RETURN
      END
"""

# ---- this script's drivers: the original drivers' logic, printed at full precision, with a
# trace of every point at which the function (f), a difference (d) or the Jacobian (j) is
# evaluated. Output lines: "RUN ..." then "X ...", "F ..."; traces go to unit 20.

LSQ_DRIVER = """\
module counts
  implicit none
  integer :: nprob = 0, nf = 0, nd = 0, nj = 0
end module counts

subroutine fcnder(m, n, x, fvec, fjac, ldfjac, iflag)
  use counts
  implicit none
  integer m, n, ldfjac, iflag
  double precision x(n), fvec(m), fjac(ldfjac, n)
  if (iflag == 1) then
    call ssqfcn(m, n, x, fvec, nprob)
    nf = nf + 1
    write (20, '(a,*(1x,es25.17e3))') 'f', x
  else if (iflag == 2) then
    call ssqjac(m, n, x, fjac, ldfjac, nprob)
    nj = nj + 1
    write (20, '(a,*(1x,es25.17e3))') 'j', x
  end if
end subroutine fcnder

subroutine fcndif(m, n, x, fvec, iflag)
  use counts
  implicit none
  integer m, n, iflag
  double precision x(n), fvec(m)
  call ssqfcn(m, n, x, fvec, nprob)
  if (iflag == 1) then
    nf = nf + 1
    write (20, '(a,*(1x,es25.17e3))') 'f', x
  else
    nd = nd + 1
    write (20, '(a,*(1x,es25.17e3))') 'd', x
  end if
end subroutine fcndif

program lsqdrv
  use counts
  implicit none
  integer :: n, m, ntries, k, info, ic, which
  integer :: iwa(40)
  double precision :: factor, tol, x(40), fvec(65), fjac(65, 40), wa(2865), fnorm
  double precision, external :: dpmpar, enorm
  external fcnder, fcndif
  character(len=8) :: arg
  call get_command_argument(1, arg)
  which = 1
  if (arg == 'lmdif') which = 2
  tol = sqrt(dpmpar(1))
  open (20, file='trace_' // trim(arg) // '.txt', status='replace')
  ic = 0
  do
    read (5, '(4i5)') nprob, n, m, ntries
    if (nprob <= 0) exit
    factor = 1.0d0
    do k = 1, ntries
      ic = ic + 1
      call initls(n, x, nprob, factor)
      nf = 0; nd = 0; nj = 0
      write (20, '(a,4i6,es25.17e3)') 'RUN', ic, nprob, n, m, factor
      if (which == 1) then
        call lmder1(fcnder, m, n, x, fvec, fjac, 65, tol, info, iwa, wa, 265)
      else
        call lmdif1(fcndif, m, n, x, fvec, tol, info, iwa, wa, 2865)
      end if
      call ssqfcn(m, n, x, fvec, nprob)
      fnorm = enorm(m, fvec)
      write (*, '(a,5i6,es25.17e3,4i8,es25.17e3)') 'RUN', ic, nprob, n, m, info, factor, nf, nd, nj, 0, fnorm
      write (*, '(a,*(1x,es25.17e3))') 'X', x(1:n)
      factor = 10.0d0 * factor
    end do
  end do
  close (20)
end program lsqdrv
"""

SYS_DRIVER = """\
module counts
  implicit none
  integer :: nprob = 0, nf = 0, nd = 0, nj = 0
end module counts

subroutine fcnjac(n, x, fvec, fjac, ldfjac, iflag)
  use counts
  implicit none
  integer n, ldfjac, iflag
  double precision x(n), fvec(n), fjac(ldfjac, n)
  if (iflag == 1) then
    call vecfcn(n, x, fvec, nprob)
    nf = nf + 1
    write (20, '(a,*(1x,es25.17e3))') 'f', x
  else if (iflag == 2) then
    call vecjac(n, x, fjac, ldfjac, nprob)
    nj = nj + 1
    write (20, '(a,*(1x,es25.17e3))') 'j', x
  end if
end subroutine fcnjac

subroutine fcnhyb(n, x, fvec, iflag)
  use counts
  implicit none
  integer n, iflag
  double precision x(n), fvec(n)
  call vecfcn(n, x, fvec, nprob)
  if (iflag == 1) then
    nf = nf + 1
    write (20, '(a,*(1x,es25.17e3))') 'f', x
  else
    nd = nd + 1
    write (20, '(a,*(1x,es25.17e3))') 'd', x
  end if
end subroutine fcnhyb

program sysdrv
  use counts
  implicit none
  integer :: n, ntries, k, info, ic, which
  double precision :: factor, tol, x(40), fvec(40), fjac(40, 40), wa(2660), fnorm
  double precision, external :: dpmpar, enorm
  external fcnjac, fcnhyb
  character(len=8) :: arg
  call get_command_argument(1, arg)
  which = 1
  if (arg == 'hybrd') which = 2
  tol = sqrt(dpmpar(1))
  open (20, file='trace_' // trim(arg) // '.txt', status='replace')
  ic = 0
  do
    read (5, '(3i5)') nprob, n, ntries
    if (nprob <= 0) exit
    factor = 1.0d0
    do k = 1, ntries
      ic = ic + 1
      call initsy(n, x, nprob, factor)
      nf = 0; nd = 0; nj = 0
      write (20, '(a,3i6,es25.17e3)') 'RUN', ic, nprob, n, factor
      if (which == 1) then
        call hybrj1(fcnjac, n, x, fvec, fjac, 40, tol, info, wa, 1060)
      else
        call hybrd1(fcnhyb, n, x, fvec, tol, info, wa, 2660)
      end if
      call vecfcn(n, x, fvec, nprob)
      fnorm = enorm(n, fvec)
      write (*, '(a,5i6,es25.17e3,4i8,es25.17e3)') 'RUN', ic, nprob, n, n, info, factor, nf, nd, nj, 0, fnorm
      write (*, '(a,*(1x,es25.17e3))') 'X', x(1:n)
      factor = 10.0d0 * factor
    end do
  end do
  close (20)
end program sysdrv
"""

# The functions at each run's starting point and at x0 + 0.1 (j/n): "LSQ ic nprob n m factor",
# then "P" (the point), "F" (residuals), "J" (column-major Jacobian), twice; the same for SYS.
EVAL_DRIVER = """\
program evaldrv
  implicit none
  integer :: nprob, n, m, ntries, k, ic, j, p
  double precision :: factor, x(40), fvec(65), fjac(65, 40), sfjac(40, 40)
  open (21, file='lsq.dat', status='old')
  ic = 0
  do
    read (21, '(4i5)') nprob, n, m, ntries
    if (nprob <= 0) exit
    factor = 1.0d0
    do k = 1, ntries
      ic = ic + 1
      call initls(n, x, nprob, factor)
      do p = 1, 2
        if (p == 2) then
          do j = 1, n
            x(j) = x(j) + 0.1d0 * dble(j) / dble(n)
          end do
        end if
        call ssqfcn(m, n, x, fvec, nprob)
        call ssqjac(m, n, x, fjac, 65, nprob)
        write (*, '(a,5i6,es25.17e3)') 'LSQ', ic, nprob, n, m, p, factor
        write (*, '(a,*(1x,es25.17e3))') 'P', x(1:n)
        write (*, '(a,*(1x,es25.17e3))') 'F', fvec(1:m)
        write (*, '(a,*(1x,es25.17e3))') 'J', fjac(1:m, 1:n)
      end do
      factor = 10.0d0 * factor
    end do
  end do
  close (21)
  open (22, file='sys.dat', status='old')
  ic = 0
  do
    read (22, '(3i5)') nprob, n, ntries
    if (nprob <= 0) exit
    factor = 1.0d0
    do k = 1, ntries
      ic = ic + 1
      call initsy(n, x, nprob, factor)
      do p = 1, 2
        if (p == 2) then
          do j = 1, n
            x(j) = x(j) + 0.1d0 * dble(j) / dble(n)
          end do
        end if
        call vecfcn(n, x, fvec, nprob)
        call vecjac(n, x, sfjac, 40, nprob)
        write (*, '(a,5i6,es25.17e3)') 'SYS', ic, nprob, n, n, p, factor
        write (*, '(a,*(1x,es25.17e3))') 'P', x(1:n)
        write (*, '(a,*(1x,es25.17e3))') 'F', fvec(1:n)
        write (*, '(a,*(1x,es25.17e3))') 'J', sfjac(1:n, 1:n)
      end do
      factor = 10.0d0 * factor
    end do
  end do
  close (22)
end program evaldrv
"""

# L-BFGS-B with scipy 1.18.1's driver loop (_lbfgsb_py._minimize_lbfgsb): count a NEW_X as an
# iteration, stop after it when the iterations reach maxiter or the evaluations exceed maxfun.
# Input (unit 5), one run per record:
#   id kind nprob n m factor maxcor factr pgtol maxiter maxfun
#   x0(1..n) / nbd(1..n) / l(1..n) / u(1..n)
# kind 1: f = sum F^2 of MINPACK's problem nprob (SSQFCN, SSQJAC); kind 2: driver1's function.
LBFGSB_DRIVER = """\
subroutine fg(kind, nprob, n, m, x, f, g)
  implicit none
  integer kind, nprob, n, m, i, j
  double precision x(n), f, g(n), fvec(65), fjac(65, 40), t1, t2
  if (kind == 1) then
    call ssqfcn(m, n, x, fvec, nprob)
    call ssqjac(m, n, x, fjac, 65, nprob)
    f = 0.0d0
    do i = 1, m
      f = f + fvec(i) * fvec(i)
    end do
    do j = 1, n
      g(j) = 0.0d0
      do i = 1, m
        g(j) = g(j) + fjac(i, j) * fvec(i)
      end do
      g(j) = 2.0d0 * g(j)
    end do
  else
    f = 0.25d0 * (x(1) - 1.0d0) * (x(1) - 1.0d0)
    do i = 2, n
      f = f + (x(i) - x(i - 1) * x(i - 1)) * (x(i) - x(i - 1) * x(i - 1))
    end do
    f = 4.0d0 * f
    t1 = x(2) - x(1) * x(1)
    g(1) = 2.0d0 * (x(1) - 1.0d0) - 1.6d1 * x(1) * t1
    do i = 2, n - 1
      t2 = t1
      t1 = x(i + 1) - x(i) * x(i)
      g(i) = 8.0d0 * t2 - 1.6d1 * x(i) * t1
    end do
    g(n) = 8.0d0 * t1
  end if
end subroutine fg

program lbdrv
  implicit none
  integer, parameter :: nmax = 40, mmax = 20
  character(len=60) :: task, csave
  character(len=40) :: id
  logical :: lsave(4)
  integer :: n, m, kind, nprob, mres, maxiter, maxfun, niter, nfev, isave(44), nbd(nmax), iwa(3 * nmax), ios
  double precision :: f, factr, pgtol, factor, x(nmax), l(nmax), u(nmax), g(nmax), dsave(29)
  double precision :: wa(2 * mmax * nmax + 5 * nmax + 11 * mmax * mmax + 8 * mmax)
  open (20, file='trace_lbfgsb.txt', status='replace')
  do
    read (5, *, iostat=ios) id, kind, nprob, n, mres, factor, m, factr, pgtol, maxiter, maxfun
    if (ios /= 0) exit
    read (5, *) x(1:n)
    read (5, *) nbd(1:n)
    read (5, *) l(1:n)
    read (5, *) u(1:n)
    write (20, '(a,1x,a)') 'RUN', trim(id)
    task = 'START'
    niter = 0
    nfev = 0
    f = 0.0d0
    g(1:n) = 0.0d0
    do
      call setulb(n, m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, task, -1, csave, lsave, isave, dsave)
      if (task(1:2) == 'FG') then
        call fg(kind, nprob, n, mres, x, f, g)
        nfev = nfev + 1
        write (20, '(a,*(1x,es25.17e3))') 'f', x(1:n)
      else if (task(1:5) == 'NEW_X') then
        niter = niter + 1
        if (niter >= maxiter) then
          task = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
        else if (nfev > maxfun) then
          task = 'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
        end if
      else
        exit
      end if
    end do
    write (*, '(a,1x,a,3i8,es25.17e3)') 'RUN', trim(id), niter, nfev, isave(30), f
    write (*, '(a,1x,a)') 'TASK', trim(task)
    write (*, '(a,*(1x,es25.17e3))') 'X', x(1:n)
    write (*, '(a,*(1x,es25.17e3))') 'G', g(1:n)
  end do
  close (20)
end program lbdrv
"""


def extract(text, name, rename=None):
    """The subroutine `name` of a fixed-form source (from its statement to its END)."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if re.match(rf"^      SUBROUTINE {name}\(", ln))
    end = next(i for i in range(start + 1, len(lines)) if re.match(r"^      END\s*$", lines[i]))
    body = "\n".join(lines[start : end + 1]) + "\n"
    return body.replace(f"SUBROUTINE {name}(", f"SUBROUTINE {rename}(") if rename else body


def sh(cmd, cwd, stdin=None):
    with open(stdin) if stdin else open("/dev/null") as fin:
        r = subprocess.run(cmd, cwd=cwd, stdin=fin, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}: {r.stderr}")
    return r.stdout


def build(src, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "dpmpar.f").write_text(DPMPAR)
    f16, f17 = (src / "file16").read_text(), (src / "file17").read_text()
    # test functions of the drivers; INITPT is renamed per driver
    (out / "lsqfun.f").write_text(extract(f17, "SSQFCN") + extract(f17, "SSQJAC") + extract(f17, "INITPT", "INITLS"))
    (out / "sysfun.f").write_text(extract(f16, "VECFCN") + extract(f16, "VECJAC") + extract(f16, "INITPT", "INITSY"))
    for name, text in (
        ("lsqdrv.f90", LSQ_DRIVER),
        ("sysdrv.f90", SYS_DRIVER),
        ("evaldrv.f90", EVAL_DRIVER),
        ("lbdrv.f90", LBFGSB_DRIVER),
    ):
        (out / name).write_text(text)
    sh([FC, *FFLAGS, "-x", "f77", "-c", str(src / "file05"), "-o", "minpack.o"], out)
    sh([FC, *FFLAGS, "-c", "dpmpar.f", "lsqfun.f", "sysfun.f"], out)
    # the original drivers, unchanged
    for drv in ("file15", "file16", "file17", "file19"):
        sh([FC, *FFLAGS, "-x", "f77", str(src / drv), "-x", "none", "minpack.o", "dpmpar.o", "-o", f"orig_{drv}"], out)
    sh([FC, *FFLAGS, "lsqdrv.f90", "lsqfun.o", "minpack.o", "dpmpar.o", "-o", "lsqdrv"], out)
    sh([FC, *FFLAGS, "sysdrv.f90", "sysfun.o", "minpack.o", "dpmpar.o", "-o", "sysdrv"], out)
    sh([FC, *FFLAGS, "evaldrv.f90", "lsqfun.o", "sysfun.o", "-o", "evaldrv"], out)
    lb = next(p for p in sorted(src.rglob("Lbfgsb.3.0")) if p.is_dir())  # where the tarball was unpacked
    libs = [str(lb / f) for f in ("lbfgsb.f", "linpack.f", "blas.f", "timer.f")]
    for k in (1, 2, 3):
        sh([FC, *FFLAGS, str(lb / f"driver{k}.f"), *libs, "-o", f"lbfgsb_driver{k}"], out)
    sh([FC, *FFLAGS, "lbdrv.f90", *libs, "lsqfun.o", "dpmpar.o", "-o", "lbdrv"], out)


def floats(line):
    return [float(v) for v in line.split()[1:]]


def parse_runs(text):
    """RUN lines of this script's MINPACK drivers -> list of dicts."""
    runs, cur = [], None
    for ln in text.splitlines():
        if ln.startswith("RUN"):
            v = ln.split()
            cur = {
                "ic": int(v[1]),
                "nprob": int(v[2]),
                "n": int(v[3]),
                "m": int(v[4]),
                "info": int(v[5]),
                "factor": float(v[6]),
                "calls_f": int(v[7]),
                "calls_d": int(v[8]),
                "calls_j": int(v[9]),
                "fnorm": float(v[11]),
            }
            runs.append(cur)
        elif ln.startswith("X"):
            cur["x"] = floats(ln)
    return runs


def parse_trace(path):
    """{run index: [(kind, x tuple), ...]} from a trace file (RUN headers)."""
    out, cur = {}, None
    for ln in pathlib.Path(path).read_text().splitlines():
        if ln.startswith("RUN"):
            key = ln.split()[1]
            cur = out.setdefault(int(key) if key.isdigit() else key, [])
        else:
            v = ln.split()
            cur.append((v[0], tuple(float(t) for t in v[1:])))
    return out


def parse_eval(text):
    out, cur = [], None
    for ln in text.splitlines():
        tag = ln.split()[0]
        if tag in ("LSQ", "SYS"):
            v = ln.split()
            cur = {
                "set": tag.lower(),
                "ic": int(v[1]),
                "nprob": int(v[2]),
                "n": int(v[3]),
                "m": int(v[4]),
                "point": int(v[5]),
                "factor": float(v[6]),
            }
            out.append(cur)
        elif tag == "P":
            cur["x"] = floats(ln)
        elif tag == "F":
            cur["f"] = floats(ln)
        elif tag == "J":  # column-major (m x n) -> rows
            vals = floats(ln)
            m, n = cur["m"], cur["n"]
            cur["jac"] = [[vals[j * m + i] for j in range(n)] for i in range(m)]
    return out


# ---- the L-BFGS-B runs: (id, kind, nprob, n, m residuals, factor, maxcor, x0, nbd, l, u)


def lbfgsb_runs():
    from tests.optimize import functions, mgh

    runs = []
    for c in mgh.MINPACK_LSQ_CASES:
        p = c.problem
        x0 = c.x0()
        runs.append((f"{c.id}", 1, p.minpack, c.n, c.m, c.factor, 10, x0, [0] * c.n, [0.0] * c.n, [0.0] * c.n))
        if c.factor == 1.0:
            lo = [v - 0.5 * (abs(v) + 1.0) for v in x0]
            hi = [v + 0.5 * (abs(v) + 1.0) for v in x0]
            runs.append((f"{c.id}-box", 1, p.minpack, c.n, c.m, c.factor, 10, x0, [2] * c.n, lo, hi))
    lo = [b[0] for b in functions.LBFGSB_DRIVER1_BOUNDS]
    hi = [b[1] for b in functions.LBFGSB_DRIVER1_BOUNDS]
    for m in (5, 10):
        runs.append((f"driver1-m{m}", 2, 0, 25, 0, 1.0, m, functions.LBFGSB_DRIVER1_X0, [2] * 25, lo, hi))
    return runs


FTOL_SCIPY = 2.220446049250313e-09  # minimize(method='L-BFGS-B')'s default ftol
GTOL_SCIPY = 1e-05


def lbfgsb_input(runs):
    lines = []
    for rid, kind, nprob, n, mres, factor, maxcor, x0, nbd, lo, hi in runs:
        factr = 1e7 if rid.startswith("driver1") else FTOL_SCIPY / EPS
        lines.append(f"'{rid}' {kind} {nprob} {n} {mres} {factor!r} {maxcor} {factr!r} {GTOL_SCIPY!r} 15000 15000")
        lines.append(" ".join(repr(float(v)) for v in x0))
        lines.append(" ".join(str(v) for v in nbd))
        lines.append(" ".join(repr(float(v)) for v in lo))
        lines.append(" ".join(repr(float(v)) for v in hi))
    return "\n".join(lines) + "\n"


def parse_lbfgsb(text):
    out, cur = {}, None
    for ln in text.splitlines():
        if ln.startswith("RUN"):
            v = ln.split()
            cur = {"n_iter": int(v[2]), "nfev": int(v[3]), "isave30": int(v[4]), "fun": float(v[5])}
            out[v[1]] = cur
        elif ln.startswith("TASK"):
            cur["task"] = ln[5:].strip()
        elif ln.startswith("X"):
            cur["x"] = floats(ln)
        elif ln.startswith("G"):
            cur["grad"] = floats(ln)
    return out


# ---- scipy on the same runs


class Recorder:
    """Wraps numpy functions to record each evaluation point (kind, x)."""

    def __init__(self):
        self.trace = []

    def wrap(self, kind, fn):
        def g(x, *a):
            self.trace.append((kind, tuple(float(v) for v in x)))
            return fn(x, *a)

        return g


def first_difference(a, b, kinds=True):
    """(index of the first evaluation that differs, the largest relative difference of the
    points over the common prefix of equal kinds)."""
    first = None
    worst = 0.0
    for i, ((ka, xa), (kb, xb)) in enumerate(zip(a, b)):
        same_kind = (ka == kb) if kinds else True
        if not same_kind or len(xa) != len(xb):
            if first is None:
                first = i
            break
        d = max((abs(u - v) / max(abs(u), abs(v), 1e-300) if u != v else 0.0) for u, v in zip(xa, xb))
        if d > 0 and first is None:
            first = i
        worst = max(worst, d)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    return first, worst


def scipy_minpack(kind, case, warn_ok=True):
    """scipy's run matching LMDER1 / LMDIF1 / HYBRJ1 / HYBRD1 (the drivers' options)."""
    import warnings

    from scipy.optimize import fsolve, leastsq

    from tests.optimize import mgh, systems

    rec = Recorder()
    if kind in ("lmder", "lmdif"):
        F, J, _, _ = mgh.as_numpy(case)
        x0 = np.array(case.x0())
        n = case.n
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "lmder":
                x, cov, info, msg, ier = leastsq(
                    rec.wrap("f", F),
                    x0,
                    Dfun=rec.wrap("j", J),
                    full_output=True,
                    ftol=TOL,
                    xtol=TOL,
                    gtol=0.0,
                    maxfev=100 * (n + 1),
                )
            else:
                x, cov, info, msg, ier = leastsq(
                    rec.wrap("f", F),
                    x0,
                    full_output=True,
                    ftol=TOL,
                    xtol=TOL,
                    gtol=0.0,
                    maxfev=200 * (n + 1),
                    epsfcn=None,
                )
        fnorm = float(np.sqrt(np.sum(F(x) ** 2)))
    else:
        F, J = systems.as_numpy(case)
        x0 = np.array(case.x0())
        n = case.n
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "hybrj":
                x, info, ier, msg = fsolve(
                    rec.wrap("f", F),
                    x0,
                    fprime=rec.wrap("j", J),
                    full_output=True,
                    xtol=TOL,
                    maxfev=100 * (n + 1),
                    diag=np.ones(n),
                    factor=100,
                )
            else:
                x, info, ier, msg = fsolve(
                    rec.wrap("f", F), x0, full_output=True, xtol=TOL, maxfev=200 * (n + 1), diag=np.ones(n), factor=100
                )
        fnorm = float(np.sqrt(np.sum(F(x) ** 2)))
    # scipy's wrappers evaluate fun (and the Jacobian, when given) once at x0 to check their
    # shapes before MINPACK starts: drop those calls
    nchk = 2 if kind in ("lmder", "hybrj") else 1
    x0t = tuple(float(v) for v in x0)
    assert [t[1] for t in rec.trace[:nchk]] == [x0t] * nchk
    trace = rec.trace[nchk:]
    # scipy's C port of MINPACK then evaluates fun at x0 once more before the routine's own
    # first call (one call more than the Fortran; MINPACK's nfev does not count it)
    extra = 0
    while len(trace) > 1 and trace[0] == trace[1] == ("f", x0t):
        trace = trace[1:]
        extra += 1
    return {
        "x": [float(v) for v in x],
        "info": int(ier),
        "nfev": int(info["nfev"]),
        "njev": int(info.get("njev", 0)),
        "fnorm": fnorm,
        "calls": len(trace),
        "extra_calls_at_x0": extra,
    }, trace


def scipy_lbfgsb(run):
    import warnings

    from scipy.optimize import minimize

    from tests.optimize import functions, mgh

    rid, kind, nprob, n, mres, factor, maxcor, x0, nbd, lo, hi = run
    if kind == 1:
        case = mgh.Case(mgh.MINPACK_LSQ[nprob], n, mres, factor)

        def fg(x):
            xl = [np.float64(v) for v in x]
            r = case.fun(xl)
            jac = case.jac(xl)
            f = 0.0
            for v in r:
                f = f + v * v
            g = []
            for j in range(n):
                s = 0.0
                for i in range(mres):
                    s = s + jac[i][j] * r[i]
                g.append(2.0 * s)
            return f, np.array(g)
    else:

        def fg(x):
            xl = [np.float64(v) for v in x]
            return functions.lbfgsb_driver1(xl), np.array(functions.lbfgsb_driver1_der(xl))

    rec = Recorder()
    bounds = None if nbd[0] == 0 else list(zip(lo, hi))
    ftol = 1e7 * EPS if rid.startswith("driver1") else FTOL_SCIPY
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = minimize(
            rec.wrap("f", fg),
            np.array(x0, dtype=float),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxcor=maxcor, ftol=ftol, gtol=GTOL_SCIPY, maxiter=15000, maxfun=15000, maxls=20),
        )
    return {
        "x": [float(v) for v in r.x],
        "fun": float(r.fun),
        "n_iter": int(r.nit),
        "nfev": int(r.nfev),
        "status": int(r.status),
        "message": str(r.message),
    }, rec.trace


def main(src, out):
    np.seterr(all="ignore")  # the problems overflow at some trial points, as in Fortran
    src, out = pathlib.Path(src).resolve(), pathlib.Path(out).resolve()
    build(src, out)
    (out / "lsq.dat").write_text((src / "file22").read_text())
    (out / "sys.dat").write_text((src / "file21").read_text())

    # the original drivers, unchanged: keep their summary tables
    texts = []
    for drv, data in (("file15", "file21"), ("file16", "file21"), ("file17", "file22"), ("file19", "file22")):
        t = sh([f"./orig_{drv}"], out, stdin=src / data)
        k = t.index("SUMMARY OF")
        texts.append(f"== {drv} < {data}\n" + t[k - 1 :].rstrip() + "\n")
    (DATA / "minpack_drivers.txt").write_text("".join(texts))
    lbt = []
    for k in (1, 2, 3):
        t = sh([f"./lbfgsb_driver{k}"], out)
        t = "\n".join(ln for ln in t.splitlines() if "time" not in ln.lower()).strip()
        lbt.append(f"== driver{k}.f\n{t}\n")
    (DATA / "lbfgsb_drivers.txt").write_text("".join(lbt))

    # the functions at the drivers' starting points
    ev = parse_eval(sh(["./evaldrv"], out))
    (DATA / "minpack_functions.json").write_text(json.dumps(ev, separators=(",", ":")) + "\n")

    # this script's drivers at full precision, and scipy on the same runs
    from tests.optimize import mgh, systems

    results, compare = {}, {}
    for kind, prog, data, cases in (
        ("lmder", "lsqdrv", "lsq.dat", mgh.MINPACK_LSQ_CASES),
        ("lmdif", "lsqdrv", "lsq.dat", mgh.MINPACK_LSQ_CASES),
        ("hybrj", "sysdrv", "sys.dat", systems.CASES),
        ("hybrd", "sysdrv", "sys.dat", systems.CASES),
    ):
        runs = parse_runs(sh([f"./{prog}", kind], out, stdin=out / data))
        traces = parse_trace(out / f"trace_{kind}.txt")
        assert len(runs) == len(cases)
        results[kind] = []
        compare[kind] = []
        for run, case in zip(runs, cases):
            assert run["n"] == case.n and run["factor"] == case.factor
            results[kind].append({"id": case.id, **run})
            sres, strace = scipy_minpack(kind, case)
            ftrace = traces[run["ic"]]
            first, worst = first_difference(ftrace, strace, kinds=kind in ("lmder", "hybrj"))
            fx, sx = np.array(run["x"]), np.array(sres["x"])
            compare[kind].append(
                {
                    "id": case.id,
                    "fortran": {"info": run["info"], "calls": len(ftrace), "fnorm": run["fnorm"]},
                    "scipy": {
                        "info": sres["info"],
                        "calls": sres["calls"],
                        "nfev": sres["nfev"],
                        "njev": sres["njev"],
                        "extra_calls_at_x0": sres["extra_calls_at_x0"],
                        "fnorm": sres["fnorm"],
                        "x": sres["x"],
                    },
                    "first_difference": first,
                    "max_rel_point_difference": worst,
                    "x_rel_difference": float(np.max(np.abs(fx - sx)) / max(np.max(np.abs(fx)), 1e-300)),
                    "same_trace": first is None,
                }
            )
    (DATA / "fortran_minpack.json").write_text(json.dumps(results, indent=1) + "\n")

    runs = lbfgsb_runs()
    (out / "lbfgsb.dat").write_text(lbfgsb_input(runs))
    lres = parse_lbfgsb(sh(["./lbdrv"], out, stdin=out / "lbfgsb.dat"))
    ltr = parse_trace(out / "trace_lbfgsb.txt")
    (DATA / "fortran_lbfgsb.json").write_text(json.dumps(lres, indent=1) + "\n")
    compare["lbfgsb"] = []
    for run in runs:
        rid = run[0]
        sres, strace = scipy_lbfgsb(run)
        first, worst = first_difference(ltr[rid], strace)
        fr = lres[rid]
        compare["lbfgsb"].append(
            {
                "id": rid,
                "fortran": {"n_iter": fr["n_iter"], "nfev": fr["nfev"], "fun": fr["fun"], "task": fr["task"]},
                "scipy": sres,
                "first_difference": first,
                "max_rel_point_difference": worst,
                "x_rel_difference": float(
                    np.max(np.abs(np.array(fr["x"]) - np.array(sres["x"]))) / max(np.max(np.abs(fr["x"])), 1e-300)
                ),
                "same_trace": first is None,
            }
        )
    (DATA / "fortran_vs_scipy.json").write_text(json.dumps(compare, indent=1) + "\n")
    for kind, rows in compare.items():
        same = sum(r["same_trace"] for r in rows)
        print(f"{kind}: {same}/{len(rows)} runs evaluate at bit-identical points in the same order")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
