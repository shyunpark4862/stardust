"""One-off probes of sd.sparse's foundations (docs/design/sparse.md §1.4, §2.2, §7): Accelerate's
Sparse Solvers and the cost of int64 indices in SpMV. The results are recorded in
docs/design/sparse.md §7; this tool reruns them.

    .venv/bin/python tools/sparse_probe.py                     # every section
    .venv/bin/python tools/sparse_probe.py sing spmv           # only these sections
    .venv/bin/python tools/sparse_probe.py det --jobs 2        # det in both thread modes at once

Sections (positional; default all):
  det    every factorization x ordering of Accelerate on ten matrix kinds and three sizes: 20 factor +
         solve runs in one process, the distinct result byte patterns (tools/sparse_probe_accelerate.cpp)
  conc   the same factor + solve on 4 threads at once against the serial bytes
  sing   structurally and numerically singular matrices: status, the reportError messages, the
         solution's inf / NaN, the residual of a finite wrong solution
  speed  factor and solve times and Accelerate's peak allocation of LU/AMD, LU/Metis, LU/MTMetis
         and Cholesky/AMD on 2-D and 3-D Laplacians
  iter   Accelerate's iterative solvers (CG, restarted GMRES, DQGMRES, LSMR) on the grids and at the
         tolerances of the iterative speed cases (SP5), against which sd.sparse's own are compared
  spmv   one CSR SpMV / SpMM kernel (row-sequential std::fma sums) with int32 and with int64
         indices on 1 and 10 threads (tools/sparse_probe_spmv.cpp), and its bits against scipy's
         csr_array @ x, csc_array @ x and csr_array @ X

The Accelerate sections run twice, in Accelerate's default thread mode and with
VECLIB_MAXIMUM_THREADS=1, as separate processes (--jobs of them at once, default 1: the timings
want a quiet machine; det and conc are about bytes, not time, and may share it). The sources are
compiled into build/sparse_probe/ with sd's flags (-O3 -mcpu=native -ffp-contract=off). The
programs' lines go to the terminal and to build/logs/sparse_probe/<stamp>.log. The det section
needs up to about 10 GB of memory and an hour; MTMetis on the block-diagonal kind is slow in the
default mode (about 100 s a factorization at n = 9e4).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import cli  # noqa: E402

BUILD = ROOT / "build" / "sparse_probe"
FLAGS = ["-std=c++20", "-O3", "-mcpu=native", "-ffp-contract=off"]
PROGRAMS = {
    "accelerate": (ROOT / "tools" / "sparse_probe_accelerate.cpp", ["-framework", "Accelerate"]),
    "spmv": (ROOT / "tools" / "sparse_probe_spmv.cpp", []),
}
ACCELERATE_SECTIONS = ("det", "conc", "sing", "speed", "iter")
SECTIONS = ACCELERATE_SECTIONS + ("spmv",)


def build(name: str) -> pathlib.Path:
    src, extra = PROGRAMS[name]
    BUILD.mkdir(parents=True, exist_ok=True)
    exe = BUILD / name
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["clang++", *FLAGS, "-o", str(exe), str(src), *extra], check=True)
    return exe


def run(cmd, env_extra, log) -> str:
    env = dict(os.environ, **env_extra)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    tag = " ".join(f"{k}={v}" for k, v in env_extra.items()) or "default thread mode"
    out = f"### {' '.join(map(str, cmd[1:]))} ({tag})\n{p.stdout}"
    if p.returncode:
        out += f"(exit status {p.returncode})\n{p.stderr[-2000:]}"
    elif p.stderr.strip():
        out += "stderr:\n" + p.stderr
    with log.open("a") as f:
        f.write(out + "\n")
    return out


def spmv_bits_against_scipy(d: pathlib.Path) -> str:
    """The kernel's y = A @ x and Y = A @ X (dumped for nnz <= 1e6) against scipy bit for bit."""
    import numpy as np
    import scipy
    import scipy.sparse as sp

    lines = [f"scipy {scipy.__version__}"]
    for meta in sorted(d.glob("*.meta")):
        name = meta.stem
        n, nnz, k = (int(v) for v in meta.read_text().split())

        def raw(ext, dtype, name=name):
            return np.fromfile(d / f"{name}.{ext}", dtype=dtype)

        indptr, indices, data = raw("indptr", np.int64), raw("indices", np.int64), raw("data", np.float64)
        x, y = raw("x", np.float64), raw("y", np.float64)
        xk, zk = raw("xk", np.float64).reshape(n, k), raw("zk", np.float64).reshape(n, k)
        a64 = sp.csr_array((data, indices, indptr), shape=(n, n))
        a32 = sp.csr_array((data, indices.astype(np.int32), indptr.astype(np.int32)), shape=(n, n))
        same = {
            "csr64 @ x": (a64 @ x).tobytes() == y.tobytes(),
            "csr32 @ x": (a32 @ x).tobytes() == y.tobytes(),
            "csc @ x": (a32.tocsc() @ x).tobytes() == y.tobytes(),
            "csr32 @ X16": np.ascontiguousarray(a32 @ xk).tobytes() == zk.tobytes(),
        }
        lines.append(
            f"{name:26s} nnz={nnz:8d} " + "  ".join(f"{t}: {'same' if v else 'DIFFER'}" for t, v in same.items())
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tools/sparse_probe.py", description=__doc__.split("\n\n")[0])
    p.add_argument(
        "sections", nargs="*", metavar="SECTION", help=f"sections to run (default all of {', '.join(SECTIONS)})"
    )
    cli.add_jobs(p, default=1)
    args = p.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        p.error(f"unknown section {unknown[0]!r} (choose from {', '.join(SECTIONS)})")
    chosen = [s for s in SECTIONS if not args.sections or s in args.sections]
    log = cli.log_file("sparse_probe")
    print(f"log: {log.relative_to(ROOT)}", flush=True)

    jobs = []
    if any(s in ACCELERATE_SECTIONS for s in chosen):
        exe = build("accelerate")
        for s in chosen:
            if s in ACCELERATE_SECTIONS:
                jobs += [([exe, s], {}), ([exe, s], {"VECLIB_MAXIMUM_THREADS": "1"})]
    with ThreadPoolExecutor(args.jobs) as pool:
        for out in pool.map(lambda j: run(j[0], j[1], log), jobs):
            print(out, flush=True)
    if "spmv" in chosen:
        exe = build("spmv")
        with tempfile.TemporaryDirectory() as d:
            print(run([exe, d], {}, log), flush=True)
            check = spmv_bits_against_scipy(pathlib.Path(d))
        with log.open("a") as f:
            f.write(check)
        print(check, flush=True)
    return cli.OK


if __name__ == "__main__":
    sys.exit(main())
