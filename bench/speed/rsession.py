"""One R process for a whole run.

The area benches started Rscript once per case, which also timed R's start-up when a
script was careless and loaded the packages again each time. Here one `R --vanilla`
process reads requests on stdin: each request names an R file that loads its data,
runs its setup and times one expression with the same batch rule as harness.measure
(warm-up, doubling batch size, min / median of the batches), then prints one result
line. Packages stay loaded between requests.

Data travels as raw float64 files (row-major, as numpy writes them); a matrix is
rebuilt with byrow = TRUE.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

from . import harness

_DRIVER = r"""
options(warn = -1)
.sd_now <- function() as.numeric(Sys.time())
.sd_time <- function(f, target, batches, min_batches, budget, slow) {
  t0 <- .sd_now(); f(); first <- .sd_now() - t0
  if (first >= slow) { t0 <- .sd_now(); f(); return(list(times = .sd_now() - t0, calls = 1L)) }
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
  list(times = times, calls = n)
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
                raise RuntimeError(ln[len("SDRESULT ERROR "):].strip())
        return out

    def version(self) -> dict:
        out = self._source('cat("SDRESULT", R.version$major, R.version$minor, "\\n")\n')
        for ln in out:
            if ln.startswith("SDRESULT"):
                _, major, minor = ln.split()[:3]
                return {"R": f"{major}.{minor}"}
        return {}

    def package_versions(self, packages) -> dict:
        out = {}
        for p in sorted(set(packages)):
            lines = self._source(f'cat("SDRESULT", as.character(packageVersion("{p}")), "\\n")\n')
            for ln in lines:
                if ln.startswith("SDRESULT"):
                    out[p] = ln.split()[1]
        return out

    def time(self, expr: str, data: dict[str, np.ndarray], *, setup: str = "", packages=(),
             target_s: float = harness.TARGET_S, batches: int = harness.BATCHES,
             min_batches: int = harness.MIN_BATCHES, budget_s: float = harness.BUDGET_S) -> harness.Sample:
        """Time the R expression `expr` (a call, evaluated each time) after loading the
        arrays in `data` under their names and running `setup`."""
        lines = []
        for p in packages:
            if p not in self._loaded:
                lines.append(f"suppressMessages(library({p}))")
        # library() attaches to the search path, which outlives the request's environment.
        files = []
        for name, arr in data.items():
            a = np.ascontiguousarray(arr, dtype=np.float64)
            if a.ndim not in (1, 2):
                raise ValueError(f"R data {name!r}: 1 or 2 dimensions, got {a.ndim}")
            path = self._dir / f"d{self._count + 1}_{name}.bin"
            a.tofile(path)
            files.append(path)
            dims = ", ".join(str(d) for d in a.shape)
            lines.append(f'{name} <- .sd_read("{path}", c({dims}))')
        lines.append(setup)
        lines.append(f".sd_f <- function() {{ {expr} }}")
        lines.append(f".sd_r <- .sd_time(.sd_f, {target_s}, {batches}, {min_batches}, {budget_s}, {harness.SLOW_CALL_S})")
        lines.append('cat("SDRESULT", .sd_r$calls, .sd_r$times, "\\n")')
        try:
            out = self._source("\n".join(lines) + "\n")
        finally:
            for f in files:
                f.unlink(missing_ok=True)
        self._loaded.update(packages)
        for ln in out:
            if ln.startswith("SDRESULT"):
                parts = ln.split()[1:]
                return harness.Sample([float(v) for v in parts[1:]], int(float(parts[0])))
        raise RuntimeError(f"R gave no result: {json.dumps(out)[-500:]}")

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except (OSError, subprocess.SubprocessError):
                self._proc.kill()
        self._stderr.close()
        shutil.rmtree(self._dir, ignore_errors=True)

    def __enter__(self) -> RSession:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
