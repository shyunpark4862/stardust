"""Build the Fortran originals of QUADPACK (netlib) and compare them with scipy 1.18.1's quad on the
integrate test collection (docs/design/integrate.md §7.6, decision C17). Writes the fixed data of
tests/integrate/data/:

  fortran_quadpack.json   per one-dimensional integral of tests/integrate/problems.py (INTEGRALS),
                          the QUADPACK routine scipy's quad would call (dqagse, dqagie, dqagpe,
                          dqawoe, dqawfe, dqawse, dqawce) with scipy's arguments (tolerances
                          1.49e-8, limit 50, maxp1 50, limlst 50; reversed limits swapped and the
                          sign changed, (-inf, b] with a Fourier weight mapped by x -> -x as scipy
                          does): integral, error, nfev, status (ier), last, the number of calls
                          and of dgtsl calls (below); and dqng (which scipy never calls) on the
                          finite integrals with an analytic integrand and no weight; and the check
                          of the earlier dgtsl stand-in against LINPACK's original (below)
  fortran_vs_scipy.json   per integral: whether the Fortran run and scipy's (scipy_quad.json,
                          scipy_calls.npz) give the same integral, error, nfev and status to the bit,
                          and the same points in the same order; where they differ, the first point
                          that differs and the integral's distance in ULP

The sources are not part of this repository: the 21 files of https://www.netlib.org/quadpack/
(dqagse dqk21 dqelg dqpsrt dqagie dqk15i dqagpe dqawoe dqc25f dqk15w dqwgtf dqcheb dqawfe dqawse
dqc25s dqmomo dqwgts dqawce dqc25c dqwgtc dqng, .f) and LINPACK's tridiagonal solver
https://www.netlib.org/linpack/dgtsl.f (dqc25f, for QAWO and QAWF, solves the Chebyshev moments'
boundary value problem with it when |omega (b - a)/2| <= 24), fetched into build/comparators/quadpack/
with their SHA256SUMS (this tool checks them and records the digests). The two routines they call
outside QUADPACK and LINPACK are written here:

  d1mach   the machine constants, from Fortran's TINY, HUGE and EPSILON (as tools/gen_optimize_fortran.py's
           DPMPAR): d1mach(4) = 2^-52, the one QUADPACK's tests use with d1mach(1) and (2)
  xerror   a no-op (dqng's error report)

Before LINPACK's file was fetched, dgtsl was a stand-in written here from the routine's documented
algorithm (DGTSL_STANDIN). It is no longer in the build. Each run also builds a second library with
the stand-in in place of the original (BUILD_DIR/standin; the stand-in counts its calls), runs the
collection again and records whether every result and every point is the same to the bit
("dgtsl_standin" in fortran_quadpack.json), and each record's dgtsl_calls comes from that count.

Everything is compiled with

    gfortran -O2 -ffp-contract=off -std=legacy -shared -fPIC

(no fused multiply-add, so that each operation rounds once) into one library, which this script
loads with ctypes: the integrands are the same Python functions (problems.MATH) that scipy's
records call, so the two runs see the same values, and each call's point is recorded.

Run:  .venv/bin/python tools/gen_integrate_fortran.py SRC_DIR BUILD_DIR
      (SRC_DIR is only read; BUILD_DIR is created and holds the replacements and the library.)

With --probe-contract the library is built a second time with -ffp-contract=fast (FMA where the
compiler likes, as clang's default contraction does in scipy's C port) into BUILD_DIR/contract,
and only the comparison with scipy is printed; nothing is written.

With --probe-qc25s the library is built a second time with dqc25s.f's typo corrected (QC25S_FIX:
the loop of the a end's log moments adds cheb24(i) rg(i) to res12, not to res24) into
BUILD_DIR/qc25s, and the weighted runs' comparison with scipy is printed; nothing is written
(docs/design/integrate.md §7.11: the QAWS log example's 4e-9 from scipy is this line).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "tests" / "integrate" / "data"
FC = "/opt/homebrew/bin/gfortran"
FFLAGS = ["-O2", "-ffp-contract=off", "-std=legacy", "-w", "-shared", "-fPIC"]
SOURCES = (
    "dqagse dqk21 dqelg dqpsrt dqagie dqk15i dqagpe dqawoe dqc25f dqk15w dqwgtf dqcheb dqawfe dqawse "
    "dqc25s dqmomo dqwgts dqawce dqc25c dqwgtc dqng dgtsl"
).split()  # QUADPACK's 21 and LINPACK's dgtsl
EPS = 1.49e-8  # scipy quad's epsabs and epsrel
LIMIT, MAXP1, LIMLST = 50, 50, 50

D1MACH = """\
c     d1mach: machine constants for IEEE double precision, written for
c     tools/gen_integrate_fortran.py (not fetched).
      double precision function d1mach(i)
      integer i
      d1mach = 0.0d0
      if (i .eq. 1) d1mach = tiny(1.0d0)
      if (i .eq. 2) d1mach = huge(1.0d0)
      if (i .eq. 3) d1mach = epsilon(1.0d0)/2.0d0
      if (i .eq. 4) d1mach = epsilon(1.0d0)
      if (i .eq. 5) d1mach = log10(2.0d0)
      return
      end
"""

XERROR = """\
c     xerror: a no-op error report, written for tools/gen_integrate_fortran.py.
      subroutine xerror(messg,nmessg,nerr,level)
      integer messg,nmessg,nerr,level
      return
      end
"""

# dgtsl: the stand-in used before LINPACK's file was fetched (see the module's docstring), kept only
# for the comparison with the original. c(2..n) the subdiagonal,
# d the diagonal, e(1..n-1) the superdiagonal, b the right-hand side; on return b holds the
# solution and info is 0 (or k when the pivot of step k is zero). Rows are kept shifted so that
# c(k), d(k), e(k) are row k's entries in columns k, k+1, k+2. The common block counts the calls.
DGTSL_STANDIN = """\
c     dgtsl: a stand-in for LINPACK's routine, written for
c     tools/gen_integrate_fortran.py from its documented algorithm
c     (the comparison build only; the real build uses LINPACK's file).
      subroutine dgtsl(n,c,d,e,b,info)
      integer n,info,k,kb,kp1,nm1,nm2,ncalls
      double precision c(*),d(*),e(*),b(*),t
      common /sdgtsl/ ncalls
      ncalls = ncalls + 1
      info = 0
      c(1) = d(1)
      nm1 = n - 1
      if (nm1 .lt. 1) go to 40
      d(1) = e(1)
      e(1) = 0.0d0
      e(n) = 0.0d0
      do 30 k = 1, nm1
        kp1 = k + 1
        if (dabs(c(kp1)) .lt. dabs(c(k))) go to 10
        t = c(kp1)
        c(kp1) = c(k)
        c(k) = t
        t = d(kp1)
        d(kp1) = d(k)
        d(k) = t
        t = e(kp1)
        e(kp1) = e(k)
        e(k) = t
        t = b(kp1)
        b(kp1) = b(k)
        b(k) = t
   10   continue
        if (c(k) .ne. 0.0d0) go to 20
        info = k
        return
   20   continue
        t = -c(kp1)/c(k)
        c(kp1) = d(kp1) + t*d(k)
        d(kp1) = e(kp1) + t*e(k)
        e(kp1) = 0.0d0
        b(kp1) = b(kp1) + t*b(k)
   30 continue
   40 continue
      if (c(n) .ne. 0.0d0) go to 50
      info = n
      return
   50 continue
      nm2 = n - 2
      b(n) = b(n)/c(n)
      if (n .eq. 1) return
      b(nm1) = (b(nm1) - d(nm1)*b(n))/c(nm1)
      if (nm2 .lt. 1) return
      do 60 kb = 1, nm2
        k = nm2 - kb + 1
        b(k) = (b(k) - d(k)*b(k+1) - e(k)*b(k+2))/c(k)
   60 continue
      return
      end
      block data sdgtbd
      integer ncalls
      common /sdgtsl/ ncalls
      data ncalls /0/
      end
"""

REPLACEMENTS = {"d1mach.f": D1MACH, "xerror.f": XERROR}


def check_sources(src):
    """The 22 files are there and match SHA256SUMS: {name: sha256}."""
    sums = {}
    for line in (src / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, name = line.split()
            sums[name.lstrip("*")] = h
    out = {}
    for s in SOURCES:
        name = f"{s}.f"
        h = hashlib.sha256((src / name).read_bytes()).hexdigest()
        if sums.get(name) != h:
            raise SystemExit(f"{name}: sha256 {h} is not SHA256SUMS's {sums.get(name)}")
        out[name] = h
    return out


# dqc25s.f's typo (netlib's original, loop 50): the log moments' sum of degree 24 restarts from
# the degree-12 sum; scipy's C port has res24 = res24 + ... (QC25S_FIX, --probe-qc25s only)
QC25S_FIX = ("        res24 = res12+cheb24(i)*rg(i)", "        res24 = res24+cheb24(i)*rg(i)")


def build(src, out, flags=None, standin=False, fix_qc25s=False):
    """The library of the sources and REPLACEMENTS; standin: DGTSL_STANDIN in place of dgtsl.f;
    fix_qc25s: dqc25s.f with QC25S_FIX (a copy in out)."""
    out.mkdir(parents=True, exist_ok=True)
    files = []
    texts = {**REPLACEMENTS, **({"dgtsl_standin.f": DGTSL_STANDIN} if standin else {})}
    if fix_qc25s:
        text = (src / "dqc25s.f").read_text()
        if text.count(QC25S_FIX[0]) != 1:
            raise SystemExit("dqc25s.f: the line of QC25S_FIX is not there once")
        texts["dqc25s_fixed.f"] = text.replace(*QC25S_FIX)
    for name, text in texts.items():
        (out / name).write_text(text)
        files.append(str(out / name))
    skip = {"dgtsl"} if standin else set()
    if fix_qc25s:
        skip.add("dqc25s")
    files += [str(src / f"{s}.f") for s in SOURCES if s not in skip]
    lib = out / "libquadpack.dylib"
    subprocess.run([FC, *(flags or FFLAGS), "-o", str(lib), *files], check=True)
    version = subprocess.run([FC, "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return lib, version


# ---- the library through ctypes

D, I = ctypes.c_double, ctypes.c_int
FUNC = ctypes.CFUNCTYPE(D, ctypes.POINTER(D))


class Quadpack:
    def __init__(self, lib):
        self.lib = ctypes.CDLL(str(lib))
        try:  # the stand-in's call counter (the original has none)
            self.count = ctypes.c_int.in_dll(self.lib, "sdgtsl_")
        except ValueError:
            self.count = None

    @staticmethod
    def _d(v):
        return ctypes.byref(D(v))

    @staticmethod
    def _i(v):
        return ctypes.byref(I(v))

    @staticmethod
    def _arr(n, dtype=np.float64):
        a = np.zeros(n, dtype=dtype)
        return a, a.ctypes.data_as(ctypes.c_void_p)


def run_routine(qp, p):
    """The routine scipy's quad calls for p, with scipy's arguments: (record, points in order)."""
    from tests.integrate.problems import MATH

    pts = []
    flip = p.b < p.a
    a, b = min(p.a, p.b), max(p.a, p.b)
    expr = p.expr
    sign_map = 1.0
    if p.weight in ("cos", "sin") and a == -math.inf:
        # scipy: thefunc(x) = func(-x) (cos) or -func(-x) (sin) on [-b, inf)
        sign_map = -1.0 if p.weight == "sin" else 1.0

        def g(x):
            pts.append(-x)
            return sign_map * expr(-x, MATH)

    else:

        def g(x):
            pts.append(x)
            return expr(x, MATH)

    cb = FUNC(lambda xp: float(g(xp[0])))
    result, abserr, neval, ier, last = D(0), D(0), I(0), I(0), I(0)
    R = ctypes.byref
    alist, al = qp._arr(LIMIT)
    blist, bl = qp._arr(LIMIT)
    rlist, rl = qp._arr(LIMIT)
    elist, el = qp._arr(LIMIT)
    iord, io = qp._arr(LIMIT, np.int32)
    if qp.count is not None:
        qp.count.value = 0
    extra = {}
    if p.weight is None and not p.points:
        if math.isinf(a) or math.isinf(b):
            if a != -math.inf:
                bound, inf = a, 1
            elif b == math.inf:
                bound, inf = 0.0, 2
            else:
                bound, inf = b, -1
            routine = "dqagie"
            qp.lib.dqagie_(cb, qp._d(bound), qp._i(inf), qp._d(EPS), qp._d(EPS), qp._i(LIMIT), R(result), R(abserr),
                           R(neval), R(ier), al, bl, rl, el, io, R(last))  # fmt: skip
        else:
            routine = "dqagse"
            qp.lib.dqagse_(cb, qp._d(a), qp._d(b), qp._d(EPS), qp._d(EPS), qp._i(LIMIT), R(result), R(abserr), R(neval),
                           R(ier), al, bl, rl, el, io, R(last))  # fmt: skip
    elif p.weight is None:
        routine = "dqagpe"
        inner = sorted({v for v in p.points if a < v < b})
        npts2 = len(inner) + 2
        points, pp = qp._arr(npts2)
        points[: len(inner)] = inner
        ptsv, ptp = qp._arr(npts2)
        level, lv = qp._arr(LIMIT, np.int32)
        ndin, nd = qp._arr(npts2, np.int32)
        qp.lib.dqagpe_(cb, qp._d(a), qp._d(b), qp._i(npts2), pp, qp._d(EPS), qp._d(EPS), qp._i(LIMIT), R(result),
                       R(abserr), R(neval), R(ier), al, bl, rl, el, ptp, io, lv, nd, R(last))  # fmt: skip
    elif p.weight in ("cos", "sin"):
        integr = 1 if p.weight == "cos" else 2
        nnlog, nn = qp._arr(LIMIT, np.int32)
        chebmo, cm = qp._arr(MAXP1 * 25)
        if math.isinf(a) or math.isinf(b):
            routine = "dqawfe"
            lo = a if a != -math.inf else -b
            rslst, rs = qp._arr(LIMLST)
            erlst, er = qp._arr(LIMLST)
            ierlst, ie = qp._arr(LIMLST, np.int32)
            lst = I(0)
            qp.lib.dqawfe_(cb, qp._d(lo), qp._d(float(p.wvar)), qp._i(integr), qp._d(EPS), qp._i(LIMLST), qp._i(LIMIT),
                           qp._i(MAXP1), R(result), R(abserr), R(neval), R(ier), rs, er, ie, R(lst), al, bl, rl, el, io,
                           nn, cm)  # fmt: skip
            last = lst
            extra = {"rslst": rslst[: lst.value].tolist(), "ierlst": ierlst[: lst.value].tolist()}
        else:
            routine = "dqawoe"
            momcom = I(0)
            qp.lib.dqawoe_(cb, qp._d(a), qp._d(b), qp._d(float(p.wvar)), qp._i(integr), qp._d(EPS), qp._d(EPS),
                           qp._i(LIMIT), qp._i(1), qp._i(MAXP1), R(result), R(abserr), R(neval), R(ier), R(last),
                           al, bl, rl, el, io, nn, R(momcom), cm)  # fmt: skip
    elif p.weight.startswith("alg"):
        routine = "dqawse"
        integr = {"alg": 1, "alg-loga": 2, "alg-logb": 3, "alg-log": 4}[p.weight]
        al_, be_ = p.wvar
        qp.lib.dqawse_(cb, qp._d(a), qp._d(b), qp._d(float(al_)), qp._d(float(be_)), qp._i(integr), qp._d(EPS),
                       qp._d(EPS), qp._i(LIMIT), R(result), R(abserr), R(neval), R(ier), al, bl, rl, el, io,
                       R(last))  # fmt: skip
    else:
        routine = "dqawce"
        qp.lib.dqawce_(cb, qp._d(a), qp._d(b), qp._d(float(p.wvar)), qp._d(EPS), qp._d(EPS), qp._i(LIMIT), R(result),
                       R(abserr), R(neval), R(ier), al, bl, rl, el, io, R(last))  # fmt: skip
    value = -result.value if flip else result.value
    rec = {
        "routine": routine,
        "integral": value,
        "error": abserr.value,
        "nfev": neval.value,
        "status": ier.value,
        "last": last.value,
        "calls": len(pts),
        "dgtsl_calls": qp.count.value if qp.count is not None else None,
        **extra,
    }
    return rec, np.array(pts, dtype=np.float64)


def run_qng(qp, p):
    from tests.integrate.problems import MATH

    pts = []

    def g(x):
        pts.append(x)
        return p.expr(x, MATH)

    cb = FUNC(lambda xp: float(g(xp[0])))
    result, abserr, neval, ier = D(0), D(0), I(0), I(0)
    R = ctypes.byref
    qp.lib.dqng_(cb, qp._d(p.a), qp._d(p.b), qp._d(EPS), qp._d(EPS), R(result), R(abserr), R(neval), R(ier))
    return {"routine": "dqng", "integral": result.value, "error": abserr.value, "nfev": neval.value,
            "status": ier.value, "calls": len(pts)}  # fmt: skip


def ulps(a, b):
    import struct

    def key(x):
        (i,) = struct.unpack("<q", struct.pack("<d", x))
        return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)

    if math.isnan(a) or math.isnan(b):
        return None
    return abs(key(a) - key(b))


def compare(rec, pts, sp, spts):
    same = {k: _bits(rec[k]) == _bits(sp[k]) for k in ("integral", "error", "nfev", "status")}
    out = {"same_result": all(same.values()), "same": same}
    n = min(len(pts), len(spts))
    diff = np.nonzero(pts[:n].view(np.int64) != spts[:n].view(np.int64))[0]
    if diff.size == 0 and len(pts) == len(spts):
        out["same_calls"] = True
    else:
        out["same_calls"] = False
        i = int(diff[0]) if diff.size else n
        out["first_difference"] = i
        out["fortran_point"] = float(pts[i]) if i < len(pts) else None
        out["scipy_point"] = float(spts[i]) if i < len(spts) else None
        out["fortran_calls"], out["scipy_calls"] = len(pts), len(spts)
    if not same["integral"]:
        out["integral_ulps"] = ulps(rec["integral"], sp["integral"])
    return out


def _bits(v):
    if isinstance(v, float):
        return np.float64(v).view(np.int64).item()
    return v


def main(argv=None):
    import argparse

    from tests.integrate import problems as P

    ap = argparse.ArgumentParser(prog="tools/gen_integrate_fortran.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=pathlib.Path, help="the folder of the 22 Fortran files and SHA256SUMS")
    ap.add_argument("build", type=pathlib.Path, help="the folder for the replacements and the library")
    ap.add_argument("--probe-contract", action="store_true", help="also build with -ffp-contract=fast and compare")
    ap.add_argument("--probe-qc25s", action="store_true", help="also build with dqc25s.f's typo corrected and compare")
    a = ap.parse_args(argv)
    digests = check_sources(a.src.resolve())
    if a.probe_qc25s:
        lib, _ = build(a.src.resolve(), a.build.resolve() / "qc25s", fix_qc25s=True)
        return _probe(Quadpack(lib), weighted=True)
    if a.probe_contract:
        flags = [f if f != "-ffp-contract=off" else "-ffp-contract=fast" for f in FFLAGS]
        lib, _ = build(a.src.resolve(), a.build.resolve() / "contract", flags)
        return _probe(Quadpack(lib))
    lib, version = build(a.src.resolve(), a.build.resolve())
    qp = Quadpack(lib)
    standin = Quadpack(build(a.src.resolve(), a.build.resolve() / "standin", standin=True)[0])
    scipy_quad = json.loads((DATA / "scipy_quad.json").read_text())
    calls = np.load(DATA / "scipy_calls.npz")
    head = {
        "compiler": version,
        "flags": " ".join(FFLAGS),
        "sources": "https://www.netlib.org/quadpack/ (21 files), https://www.netlib.org/linpack/dgtsl.f",
        "sha256": digests,
        "replacements": {"d1mach": "Fortran TINY, HUGE, EPSILON", "xerror": "no-op"},
        "replacement_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in REPLACEMENTS.items()},
    }  # fmt: skip
    rows, cmp_rows, standin_differs, standin_calls = [], [], [], 0
    for p in P.INTEGRALS:
        if p.a == p.b:
            continue
        rec, pts = run_routine(qp, p)
        srec, spts = run_routine(standin, p)  # the stand-in, for the comparison and the dgtsl count
        rec["dgtsl_calls"] = srec.pop("dgtsl_calls")
        standin_calls += rec["dgtsl_calls"]
        orig = {k: v for k, v in rec.items() if k != "dgtsl_calls"}
        if json.dumps(orig, sort_keys=True) != json.dumps(srec, sort_keys=True) or not np.array_equal(
            pts.view(np.int64), spts.view(np.int64)
        ):
            standin_differs.append(p.id)
        rows.append((p.id, rec))
        cmp_rows.append((p.id, {"routine": rec["routine"], "dgtsl_calls": rec["dgtsl_calls"],
                                **compare(rec, pts, scipy_quad[p.id], calls[f"quad/{p.id}"])}))  # fmt: skip
    for p in P.INTEGRALS:
        if p.numeric and p.weight is None and not p.points and math.isfinite(p.a) and math.isfinite(p.b):
            rows.append((f"dqng/{p.id}", run_qng(qp, p)))
    head["dgtsl_standin"] = {
        "sha256": hashlib.sha256(DGTSL_STANDIN.encode()).hexdigest(),
        "runs": len(cmp_rows),
        "runs_calling_dgtsl": sum(1 for _, r in rows if r.get("dgtsl_calls")),
        "dgtsl_calls": standin_calls,
        "differ": standin_differs,
    }
    _write("fortran_quadpack.json", head, rows)
    same_standin = len(cmp_rows) - len(standin_differs)
    print(f"the dgtsl stand-in against LINPACK's: {same_standin} of {len(cmp_rows)} runs "
          f"the same to the bit ({head['dgtsl_standin']['runs_calling_dgtsl']} call dgtsl, "
          f"{standin_calls} calls)")  # fmt: skip
    same = sum(1 for _, c in cmp_rows if c["same_result"] and c["same_calls"])
    _write("fortran_vs_scipy.json", {"scipy": "1.18.1", "same": same, "of": len(cmp_rows)}, cmp_rows)
    print(f"{same} of {len(cmp_rows)} runs agree with scipy to the bit (results and call order)")
    for pid, c in cmp_rows:
        if not (c["same_result"] and c["same_calls"]):
            print(f"  {pid} ({c['routine']}, dgtsl {c['dgtsl_calls']}): {json.dumps(c)}")
    return 0


def _probe(qp, weighted=False):
    from tests.integrate import problems as P

    scipy_quad = json.loads((DATA / "scipy_quad.json").read_text())
    calls = np.load(DATA / "scipy_calls.npz")
    same = {"all": 0, "calls": 0, "integral": 0, "error": 0}
    n = 0
    for p in P.INTEGRALS:
        if weighted and p.weight is None:
            continue
        rec, pts = run_routine(qp, p)
        c = compare(rec, pts, scipy_quad[p.id], calls[f"quad/{p.id}"])
        if weighted:
            print(f"  {p.id} ({rec['routine']}): calls {c['same_calls']}, {json.dumps(c['same'])}")
        n += 1
        same["all"] += c["same_result"] and c["same_calls"]
        same["calls"] += c["same_calls"]
        same["integral"] += c["same"]["integral"]
        same["error"] += c["same"]["error"]
    print(f"{'dqc25s corrected' if weighted else '-ffp-contract=fast'}: of {n} runs, {same['all']} agree with scipy "
          f"to the bit; the same calls "
          f"{same['calls']}, integral {same['integral']}, error {same['error']}")  # fmt: skip
    return 0


def _write(name, head, rows):
    lines = [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in head.items()]
    lines += [f"{json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}" for k, v in rows]
    (DATA / name).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(f"{len(rows)} records -> {(DATA / name).relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
