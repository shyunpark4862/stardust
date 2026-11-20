"""One R process per worker for the comparator columns (python -m bench.accuracy run --compare).

R is optional: when Rscript is not on PATH, available() is False and the R columns read "not
run". A request writes each input as raw little-endian float64 (exactly the doubles the other
libraries get), sends R one expression over the names a, b, c, ..., and reads the result back
the same way. The process stays up between requests (packages load once).
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

_DRIVER = r"""
options(warn = -1)
.sd_read <- function(path, n) readBin(path, what = "double", n = n, size = 8, endian = "little")
.sd_write <- function(x, path) writeBin(as.double(x), path, size = 8, endian = "little")
.sd_in <- file("stdin")
open(.sd_in)
repeat {
  line <- readLines(con = .sd_in, n = 1)
  if (length(line) == 0) break
  res <- tryCatch({ source(line, local = new.env()); "ok" }, error = function(e) paste("error:", conditionMessage(e)))
  cat("@@sd@@", gsub("\n", " ", res), "\n", sep = "")
  flush(stdout())
}
"""

_PROC = [None]
_DIR = [None]


def available() -> bool:
    return shutil.which("R") is not None


def version() -> str | None:
    if not available():
        return None
    out = subprocess.run(["R", "--version"], capture_output=True, text=True).stdout
    return out.split()[2] if out.startswith("R version") else None


def _start():
    import atexit

    atexit.register(close)
    _DIR[0] = pathlib.Path(tempfile.mkdtemp(prefix="sd-accuracy-r-"))
    drv = _DIR[0] / "driver.R"
    drv.write_text(_DRIVER)
    _PROC[0] = subprocess.Popen(
        ["R", "--vanilla", "--slave", "-f", str(drv)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )


def evaluate(expr: str, *args, packages=(), n_out=None) -> np.ndarray:
    """expr evaluated in R over a, b, c, ... (the args as double vectors, broadcast to one
    length first); the result as a float64 array of that length (or n_out)."""
    if _PROC[0] is None or _PROC[0].poll() is not None:
        _start()
    arrs = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in args))
    n = arrs[0].size if arrs else (n_out or 1)
    d = _DIR[0]
    names = "abcdefgh"
    lines = [f"suppressPackageStartupMessages(library({p}))" for p in packages]
    for nm, a in zip(names, arrs):
        f = d / f"{nm}.bin"
        np.ascontiguousarray(a.ravel(), dtype="<f8").tofile(f)
        lines.append(f'{nm} <- .sd_read("{f}", {n})')
    out = d / "out.bin"
    out.unlink(missing_ok=True)
    lines.append(f'.sd_write(({expr}), "{out}")')
    script = d / "req.R"
    script.write_text("\n".join(lines) + "\n")
    try:
        _PROC[0].stdin.write(f"{script}\n")
        _PROC[0].stdin.flush()
        status = _status()
    except BaseException:  # interrupted or broken: R may still be computing; start over next time
        kill()
        raise
    if status != "ok":
        raise RuntimeError(f"R: {status} in {expr}")
    res = np.fromfile(out, dtype="<f8")
    return res.reshape(arrs[0].shape) if arrs and res.size == n else res


def evaluate_named(expr: str, data: dict) -> np.ndarray:
    """expr over named double vectors of any lengths (data: name -> array, read in row-major
    order); the result as a float64 vector."""
    if _PROC[0] is None or _PROC[0].poll() is not None:
        _start()
    d = _DIR[0]
    lines = []
    for nm, a in data.items():
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel(), dtype="<f8")
        f = d / f"{nm}.bin"
        a.tofile(f)
        lines.append(f'{nm} <- .sd_read("{f}", {a.size})')
    out = d / "out.bin"
    out.unlink(missing_ok=True)
    lines.append(f'.sd_write(({{ {expr} }}), "{out}")')
    script = d / "req.R"
    script.write_text("\n".join(lines) + "\n")
    try:
        _PROC[0].stdin.write(f"{script}\n")
        _PROC[0].stdin.flush()
        status = _status()
    except BaseException:
        kill()
        raise
    if status != "ok":
        raise RuntimeError(f"R: {status}")
    return np.fromfile(out, dtype="<f8")


def _status() -> str:
    """The driver's answer line; anything else R printed (a package's start-up message) is skipped."""
    while True:
        line = _PROC[0].stdout.readline()
        if not line:
            raise RuntimeError("R exited")
        if line.startswith("@@sd@@"):
            return line[6:].strip()


def forget():
    """In a forked child: drop the parent's session without touching it (the child starts its own)."""
    _PROC[0] = None
    _DIR[0] = None


def kill():
    if _PROC[0] is not None:
        _PROC[0].kill()
        _PROC[0].wait()
        _PROC[0] = None
    if _DIR[0] is not None:
        shutil.rmtree(_DIR[0], ignore_errors=True)
        _DIR[0] = None


def close():
    if _PROC[0] is not None:
        try:
            _PROC[0].stdin.close()
            _PROC[0].wait(timeout=10)
        except Exception:
            _PROC[0].kill()
        _PROC[0] = None
    if _DIR[0] is not None:
        shutil.rmtree(_DIR[0], ignore_errors=True)
        _DIR[0] = None
