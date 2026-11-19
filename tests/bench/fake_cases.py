"""Fake cases for the supervisor tests (SDBENCH_CASES_MODULE=tests.bench.fake_cases)."""

import time

from bench.speed.registry import Case, Ref


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
