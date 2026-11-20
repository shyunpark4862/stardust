"""One R process for a whole run.

One `R --vanilla` process reads requests on stdin: each request names an R file that
loads its data, runs its setup and times one expression with the same batch rule as
harness.measure (warm-up, doubling batch size, min / median of the batches), then prints
one result line. Packages stay loaded between requests, so no case times R's start-up.

R runs on one thread: R_ENV limits its BLAS (this machine's R links OpenBLAS built with
OpenMP, which reads OMP_NUM_THREADS; OPENBLAS_NUM_THREADS covers a pthreads build).

The clock is Sys.time() (microseconds; proc.time()'s elapsed counts whole milliseconds
here, too coarse for 20 ms batches). R's garbage collector cannot be switched off inside
a batch, as Python's is: it runs once (gc()) before the batches instead.

Data travels as raw float64 files (row-major, as numpy writes them); a matrix is
rebuilt with byrow = TRUE. A rotated case (RRef.sets) loads every input set into its own
environment, and the timed function calls the expression on each set in turn.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

from . import harness

# R's threads: one, for its BLAS and anything else that uses OpenMP
R_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

_DRIVER = r"""
options(warn = -1)
.sd_now <- function() as.numeric(Sys.time())
.sd_nan <- function(v) if (is.numeric(v) || is.complex(v)) mean(is.na(v)) else -1
.sd_time <- function(f, target, batches, min_batches, budget, slow) {
  t0 <- .sd_now(); v <- f(); first <- .sd_now() - t0
  nan <- .sd_nan(v); rm(v)
  invisible(gc(FALSE))
  if (first >= slow) {
    t0 <- .sd_now(); f(); return(list(times = .sd_now() - t0, calls = 1L, nan = nan))
  }
  n <- 1L; per <- first
  if (first < target) repeat {
    t0 <- .sd_now(); for (i in seq_len(n)) f(); dt <- .sd_now() - t0
    per <- dt / n
    if (dt >= target || n >= 2^24) break
    n <- n * 2L
  }
  k <- batches
  if (per * n * k > budget) k <- max(min_batches, as.integer(budget / (per * n)))
  times <- numeric(k)
  for (j in seq_len(k)) { t0 <- .sd_now(); for (i in seq_len(n)) f(); times[j] <- (.sd_now() - t0) / n }
  list(times = times, calls = n, nan = nan)
}
.sd_emit <- function(name, u, dir) {
  if (methods::is(u, "Matrix")) u <- as.matrix(u)
  if (is.logical(u)) u <- as.numeric(u)
  if (!(is.numeric(u) || is.complex(u))) { cat("SDVALUE", name, "-", class(u)[1], "\n"); return(invisible()) }
  d <- dim(u); if (is.null(d)) d <- length(u)
  if (length(d) == 2) u <- t(u)
  u <- as.vector(u)
  f <- file.path(dir, paste0("v_", name, ".bin"))
  if (is.complex(u)) writeBin(as.vector(rbind(Re(u), Im(u))), f) else writeBin(as.double(u), f)
  cat("SDVALUE", name, if (is.complex(u)) "c" else "f", d, "\n")
}
.sd_value <- function(v, dir) {
  if (is.list(v) && !is.data.frame(v)) {
    nms <- names(v); if (is.null(nms)) nms <- rep("", length(v))
    for (i in seq_along(v)) .sd_emit(if (nzchar(nms[i])) nms[i] else as.character(i), v[[i]], dir)
  } else .sd_emit("value", v, dir)
}
.sd_info <- function() {
  s <- sessionInfo()
  cat("SDRESULT", R.version$major, R.version$minor, "\n")
  cat("SDBLAS", s$BLAS, "\n")
  cat("SDLAPACK", s$LAPACK, "\n")
  cat("SDOMP", Sys.getenv("OMP_NUM_THREADS"), Sys.getenv("OPENBLAS_NUM_THREADS"), "\n")
}
.sd_read <- function(path, dims) {
  v <- readBin(path, "double", prod(dims))
  if (length(dims) == 1) v else matrix(v, dims[1], dims[2], byrow = TRUE)
}
con <- file("stdin", open = "r")
cat("SDREADY\n"); flush(stdout())
repeat {
  line <- readLines(con, n = 1)
  if (length(line) == 0 || line == "quit") break
  res <- tryCatch({ source(line, local = new.env()); NULL },
                  error = function(e) paste("ERROR", gsub("\n", " ", conditionMessage(e))))
  if (!is.null(res)) cat("SDRESULT", res, "\n")
  cat("SDDONE\n"); flush(stdout())
}
"""


class RSession:
    """A persistent R process. Use as a context manager, or call close()."""

    def __init__(self, rscript: str | None = None):
        r = rscript or shutil.which("R")
        if r is None:
            raise RuntimeError("R is not installed")
        self._dir = pathlib.Path(tempfile.mkdtemp(prefix="sdbench-r-"))
        driver = self._dir / "driver.R"
        driver.write_text(_DRIVER)
        self._stderr = open(self._dir / "stderr.txt", "w")  # a file: a full pipe would block R
        self._proc = subprocess.Popen(
            [r, "--vanilla", "--no-echo", "-f", str(driver)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            env={**os.environ, **R_ENV},
            text=True,
            bufsize=1,
        )
        self._loaded: set[str] = set()
        self._count = 0
        self._read_until("SDREADY")

    def _read_until(self, marker: str) -> list[str]:
        assert self._proc.stdout is not None
        lines = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                self._stderr.flush()
                err = (self._dir / "stderr.txt").read_text()
                raise RuntimeError(f"R exited: {err.strip()[-2000:]}")
            line = line.rstrip("\n")
            if line.strip() == marker:
                return lines
            lines.append(line)

    def _source(self, code: str) -> list[str]:
        self._count += 1
        path = self._dir / f"req{self._count}.R"
        path.write_text(code)
        assert self._proc.stdin is not None
        self._proc.stdin.write(f"{path}\n")
        self._proc.stdin.flush()
        out = self._read_until("SDDONE")
        path.unlink(missing_ok=True)
        for ln in out:
            if ln.startswith("SDRESULT ERROR"):
                raise RuntimeError(ln[len("SDRESULT ERROR ") :].strip())
        return out

    def version(self) -> dict:
        return {"R": self.info().get("version")} if self.info().get("version") else {}

    def info(self) -> dict:
        """R's version, the BLAS and LAPACK it loaded, and its thread variables."""
        if getattr(self, "_info", None) is None:
            out: dict = {}
            for ln in self._source(".sd_info()\n"):
                parts = ln.split()
                if not parts:
                    continue
                if parts[0] == "SDRESULT" and len(parts) >= 3:
                    out["version"] = f"{parts[1]}.{parts[2]}"
                elif parts[0] == "SDBLAS":
                    out["blas"] = " ".join(parts[1:])
                elif parts[0] == "SDLAPACK":
                    out["lapack"] = " ".join(parts[1:])
                elif parts[0] == "SDOMP":
                    out["threads"] = {"OMP_NUM_THREADS": R_ENV["OMP_NUM_THREADS"], "seen": parts[1:]}
            self._info = out
        return self._info

    def package_versions(self, packages) -> dict:
        out = {}
        for p in sorted(set(packages)):
            lines = self._source(f'cat("SDRESULT", as.character(packageVersion("{p}")), "\\n")\n')
            for ln in lines:
                if ln.startswith("SDRESULT"):
                    out[p] = ln.split()[1]
        return out

    def time(
        self,
        expr: str,
        data: dict[str, np.ndarray],
        *,
        setup: str = "",
        packages=(),
        sets: list[dict[str, np.ndarray]] | None = None,
        target_s: float = harness.TARGET_S,
        batches: int = harness.BATCHES,
        min_batches: int = harness.MIN_BATCHES,
        budget_s: float = harness.BUDGET_S,
    ) -> harness.Sample:
        """Time the R expression `expr` (a call, evaluated each time) after loading the
        arrays in `data` under their names and running `setup`. With `sets` (a list of
        such dicts), each set is loaded into its own environment (setup runs in each) and
        the calls go through the sets in turn."""
        lines, files = self._prepare(expr, data, setup, packages, sets)
        lines.append(
            f".sd_r <- .sd_time(.sd_f, {target_s}, {batches}, {min_batches}, {budget_s}, {harness.SLOW_CALL_S})"
        )
        lines.append('cat("SDRESULT", .sd_r$calls, .sd_r$nan, .sd_r$times, "\\n")')
        out = self._run(lines, files, packages)
        for ln in out:
            if ln.startswith("SDRESULT"):
                parts = ln.split()[1:]
                nan = float(parts[1])
                return harness.Sample([float(v) for v in parts[2:]], int(float(parts[0])), nan if nan >= 0 else None)
        raise RuntimeError(f"R gave no result: {json.dumps(out)[-500:]}")

    def value(
        self, expr: str, data: dict[str, np.ndarray], *, setup: str = "", packages=(), sets=None, value: str = ""
    ):
        """The value of one call (on the first set of a rotated case), for the results
        check: a float64 or complex128 array in row-major order, or for an R list a dict
        of its numeric components by name (by position, "1", "2", ..., when unnamed). A
        component that is not numeric is the string of its R class. `value`: an R expression
        of the result `.v` to return instead (e.g. "coef(.v)")."""
        lines, files = self._prepare(expr, data, setup, packages, sets)
        lines.append(".v <- .sd_f()")
        lines.append(f'.sd_value({value or ".v"}, "{self._dir}")')
        out = self._run(lines, files, packages)
        vals: dict = {}
        for ln in out:
            parts = ln.split()
            if not parts or parts[0] != "SDVALUE":
                continue
            name, kind = parts[1], parts[2]
            if kind == "-":
                vals[name] = " ".join(parts[3:])
                continue
            path = self._dir / f"v_{name}.bin"
            a = np.fromfile(path, dtype=np.float64)
            path.unlink(missing_ok=True)
            if kind == "c":
                a = a[0::2] + 1j * a[1::2]
            dims = [int(d) for d in parts[3:]]
            vals[name] = a.reshape(dims) if dims and int(np.prod(dims)) == a.size else a
        if set(vals) == {"value"}:
            return vals["value"]
        return vals

    def _prepare(self, expr, data, setup, packages, sets):
        lines = []
        for p in packages:
            if p not in self._loaded:
                lines.append(f"suppressMessages(library({p}))")
        # library() attaches to the search path, which outlives the request's environment.
        files: list[pathlib.Path] = []
        if not sets:
            lines += self._load(data, files, "")
            lines.append(setup)
            lines.append(f".sd_f <- function() {{ {expr} }}")
            return lines, files
        lines.append(".sd_envs <- list()")
        for i, one in enumerate(sets):
            lines.append(".sd_e <- new.env(parent = environment())")
            lines += [f"assign('{ln.split(' <- ')[0]}', {ln.split(' <- ', 1)[1]}, envir = .sd_e)"
                      for ln in self._load(one, files, f"s{i}_")]  # fmt: skip
            if setup:
                lines.append(f"local({{ {setup} }}, envir = .sd_e)")
            lines.append(f".sd_envs[[{i + 1}]] <- .sd_e")
        lines.append(
            f".sd_fs <- lapply(.sd_envs, function(e) {{ g <- function() {{ {expr} }}; environment(g) <- e; g }})"
        )
        lines.append(".sd_k <- 0L; .sd_m <- length(.sd_fs)")
        lines.append(".sd_f <- function() { .sd_k <<- if (.sd_k < .sd_m) .sd_k + 1L else 1L; .sd_fs[[.sd_k]]() }")
        return lines, files

    def _load(self, data, files, prefix) -> list[str]:
        lines = []
        for name, arr in data.items():
            a = np.ascontiguousarray(arr, dtype=np.float64)
            if a.ndim not in (1, 2):
                raise ValueError(f"R data {name!r}: 1 or 2 dimensions, got {a.ndim}")
            path = self._dir / f"d{self._count + 1}_{prefix}{name}.bin"
            a.tofile(path)
            files.append(path)
            dims = ", ".join(str(d) for d in a.shape)
            lines.append(f'{name} <- .sd_read("{path}", c({dims}))')
        return lines

    def _run(self, lines, files, packages) -> list[str]:
        try:
            out = self._source("\n".join(lines) + "\n")
        finally:
            for f in files:
                f.unlink(missing_ok=True)
        self._loaded.update(packages)
        return out

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except OSError, subprocess.SubprocessError:
                self._proc.kill()
        self._stderr.close()
        shutil.rmtree(self._dir, ignore_errors=True)

    def __enter__(self) -> RSession:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
