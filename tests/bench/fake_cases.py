"""Fake cases for the supervisor tests (SDBENCH_CASES_MODULE=tests.bench.fake_cases).

SDBENCH_PID_LOG=<file>: the "pid" case's setup appends its worker's pid to that file.
"""

import os
import time

from bench.speed.registry import Case, Check, Ref


def _hog(d):
    import numpy as np

    blocks = []

    def f():
        # Keep allocating and touching memory until the supervisor kills the worker.
        while True:
            blocks.append(np.ones(16 * 2**20))  # 128 MB per block
            time.sleep(0.05)

    return f


def _boom(d):
    def f():
        raise ValueError("boom")

    return f


def _pid_setup():
    path = os.environ.get("SDBENCH_PID_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(f"{os.getpid()}\n")
    return {}


def cases():
    fast = {"timing": {"target_s": 0.001, "batches": 3, "budget_s": 0.05}}
    return [
        Case(
            area="array",
            group="fake",
            name="sum",
            n=1000,
            setup=lambda: {"x": list(range(1000))},
            sd=lambda d: lambda: sum(d["x"]),
            refs=(Ref("numpy", lambda d: lambda: len(d["x"])),),
            quick=True,
            cost=("special", "_fake"),
            **fast,
        ),
        Case(area="array", group="fake", name="sleep", sd=lambda d: lambda: time.sleep(60), threads=(1,), **fast),
        Case(area="array", group="fake", name="pid", setup=_pid_setup, sd=lambda d: lambda: None, threads=(1,), **fast),
        Case(
            area="array",
            group="fake",
            name="check",
            n=3,
            sd=lambda d: lambda: [1.0, 2.0, float("nan")],
            refs=(
                Ref("numpy", lambda d: lambda: [1.0, 2.0 + 1e-9, 5.0]),  # agrees; a NaN only sd gives
                Ref("other", lambda d: lambda: [9.0, 9.0, 9.0]),  # not the first comparator: not checked
            ),
            threads=(1,),
            **fast,
        ),
        Case(
            area="array",
            group="fake",
            name="wrong",
            n=3,
            sd=lambda d: lambda: [1.0, 2.0, 3.0],
            refs=(
                Ref("numpy", lambda d: lambda: [3.0, 2.5, 1.0]),
                Ref("numpy swapped", lambda d: lambda: [3.0, 2.5, 1.0], differs="takes the arguments reversed"),
            ),
            threads=(1,),
            **fast,
        ),
        Case(
            area="array",
            group="fake",
            name="skipped",
            sd=lambda d: lambda: object(),
            refs=(Ref("numpy", lambda d: lambda: object()),),
            threads=(1,),
            check=Check(skip="no values to compare"),
            **fast,
        ),
        Case(area="array", group="fake", name="hog", sd=_hog, threads=(1,), **fast),
        Case(
            area="array",
            group="fake",
            name="boom",
            sd=_boom,
            threads=(1,),
            refs=(Ref("numpy", lambda d: lambda: None),),
            **fast,
        ),
        Case(
            area="array",
            group="fake",
            name="sum2",
            n=10,
            setup=lambda: {"x": list(range(10))},
            sd=lambda d: lambda: sum(d["x"]),
            quick=True,
            **fast,
        ),
    ]
