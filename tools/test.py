"""The test suite in parallel: the tests without the serial mark on pytest-xdist workers, then
the serial ones alone (tests/README.md §1; tests/conftest.py has the tiers and the marks).

    .venv/bin/python tools/test.py [-n N] [PYTEST ARGS ...]     # e.g. tests/frame, -k merge, --tier deep

-n is the worker count (default 6, at most 8: timed 2026-11-20 on this Mac when the suite had
7,404 tests, serial 145 s / 3.7 GB, 4 workers 59 s / 7.8 GB, 6 workers 46 s / 9.4 GB, 8 workers
48 s / 11 GB with a GIL test failing under the load; tests/README.md §1 has the current count
and how to take it). Every worker keeps sd's default thread count (2 threads per worker
measured slower). Plain `pytest` stays serial. The exit status is the worse of the two
runs; "no tests collected" in either counts as success.
"""

import argparse
import subprocess
import sys
import time

MAX_WORKERS = 8
NO_TESTS = 5  # pytest's exit status when nothing was selected


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-n", type=int, default=6, help=f"workers (default 6, at most {MAX_WORKERS})")
    args, rest = ap.parse_known_args()
    n = max(1, min(args.n, MAX_WORKERS))
    base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *rest]
    status = 0
    for label, extra in (("parallel", ["-n", str(n), "-m", "not serial"]), ("serial", ["-m", "serial"])):
        t0 = time.perf_counter()
        print(f"== {label}: {' '.join(extra)}", flush=True)
        code = subprocess.run(base + extra).returncode
        print(f"== {label}: exit {code} in {time.perf_counter() - t0:.0f} s", flush=True)
        if code not in (0, NO_TESTS):
            status = status or code
    return status


if __name__ == "__main__":
    sys.exit(main())
