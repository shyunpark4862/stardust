"""The real matrices of sd.sparse's tests from the SuiteSparse Matrix Collection (decision SP20,
docs/design/sparse.md §7.2): eight Matrix Market archives the user approved, fetched into
build/sparse/suitesparse/ (outside version control, like optimize's Fortran originals, O13).
Nothing else is downloaded.

    .venv/bin/python tools/sparse_suitesparse.py                  # fetch the missing ones (8 jobs)
    .venv/bin/python tools/sparse_suitesparse.py HB/sherman3      # only these
    .venv/bin/python tools/sparse_suitesparse.py --jobs 2
    .venv/bin/python tools/sparse_suitesparse.py --list           # the matrices and their state

Each archive is checked before it is unpacked: gzip data (not the server's error page) and,
where SHA256 holds a hash, that hash. An archive without a recorded hash is unpacked and its
hash printed, to be recorded in SHA256. Only the .mtx members are extracted (tarfile's "data"
filter: no links, no paths outside the folder), to build/sparse/suitesparse/<group>/<name>.mtx.
The tests that read them (tests/sparse/test_suitesparse.py) are in the external tier and skip
when a file is missing. Exit status: 0 every chosen matrix is in place, 1 a download or a check
failed, 2 a usage error. The log: build/logs/sparse_suitesparse/<stamp>.log.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import cli  # noqa: E402

DEST = ROOT / "build" / "sparse" / "suitesparse"
URL = "https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"

# (group/name, kind, rows, cols, nonzeros as the collection lists them); decision SP20
MATRICES = (
    ("HB/bcsstk14", "structural, symmetric positive definite", 1806, 1806, 63454),
    ("Boeing/msc10848", "structural, symmetric positive definite", 10848, 10848, 1229776),
    ("HB/sherman3", "fluid dynamics, unsymmetric, condition 5.0e17", 5005, 5005, 20033),
    ("Goodwin/goodwin", "fluid dynamics, unsymmetric, condition 2.0e10", 7320, 7320, 324772),
    ("Hamm/memplus", "circuit, unsymmetric", 17758, 17758, 99147),
    ("GHS_indef/cvxqp3", "optimization KKT, symmetric indefinite", 17500, 17500, 114962),
    ("HB/well1850", "least squares, rectangular", 1850, 712, 8755),
    ("HB/young1c", "acoustics, complex unsymmetric", 841, 841, 4089),
)
NAMES = tuple(m[0] for m in MATRICES)

# sha256 of each archive as downloaded; None until the first successful fetch records it
SHA256 = dict.fromkeys(NAMES)


def mtx_path(name: str) -> pathlib.Path:
    return DEST / f"{name}.mtx"


def available(name: str) -> bool:
    return mtx_path(name).exists()


def load(name: str):
    """(shape, row, col, data) of a fetched matrix, symmetric ones expanded to both triangles;
    FileNotFoundError when it is not fetched."""
    import numpy as np
    import scipy.io

    p = mtx_path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} (fetch it with .venv/bin/python tools/sparse_suitesparse.py {name})")
    a = scipy.io.mmread(str(p), spmatrix=False).tocoo()
    return a.shape, a.coords[0].astype(np.int64), a.coords[1].astype(np.int64), np.asarray(a.data)


def fetch(name: str, log: pathlib.Path) -> tuple[str, bool, str]:
    group, base = name.split("/")
    url = URL.format(group=group, name=base)
    out = mtx_path(name)
    if out.exists():
        return name, True, "already present"
    with tempfile.TemporaryDirectory() as tmp:
        arc = pathlib.Path(tmp) / f"{base}.tar.gz"
        try:
            with urllib.request.urlopen(url, timeout=300) as r, arc.open("wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as e:  # HTTP errors, refused connections
            return name, False, f"download failed: {e}"
        data = arc.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if data[:2] != b"\x1f\x8b":
            return name, False, f"not a gzip archive ({len(data)} bytes; the server's error page?)"
        want = SHA256[name]
        if want is not None and digest != want:
            return name, False, f"sha256 {digest} does not match the recorded {want}"
        with tarfile.open(arc) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".mtx")]
            main = [m for m in members if pathlib.Path(m.name).name == f"{base}.mtx"]
            if not main:
                return name, False, "no <name>.mtx in the archive"
            tar.extractall(tmp, members=main, filter="data")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(pathlib.Path(tmp) / main[0].name, out)
        note = f"{len(data)} bytes, sha256 {digest}" + ("" if want else " (record it in SHA256)")
    with log.open("a") as f:
        f.write(f"{name}: {note}\n")
    return name, True, note


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tools/sparse_suitesparse.py", description=__doc__.split("\n\n")[0])
    p.add_argument("names", nargs="*", metavar="GROUP/NAME", help="matrices to fetch (default all eight)")
    p.add_argument("--list", action="store_true", help="list the matrices and whether they are fetched")
    cli.add_jobs(p)
    args = p.parse_args(argv)
    unknown = [n for n in args.names if n not in NAMES]
    if unknown:
        p.error(f"unknown matrix {unknown[0]!r} (choose from {', '.join(NAMES)})")
    chosen = [n for n in NAMES if not args.names or n in args.names]
    if args.list:
        for name, kind, m, n, nnz in MATRICES:
            if name in chosen:
                state = "fetched" if available(name) else "missing"
                print(f"{name:20s} {m:>6} x {n:<6} nnz {nnz:>8}  {state:8s} {kind}")
        return cli.OK
    log = cli.log_file("sparse_suitesparse")
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda n: fetch(n, log), chosen))
    for name, ok, note in results:
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {note}")
    return cli.OK if all(ok for _, ok, _ in results) else cli.FAIL


if __name__ == "__main__":
    sys.exit(main())
