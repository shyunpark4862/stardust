"""M0: the extension builds, imports, and was compiled for the intended target."""

import pytest

import stardust_pandas as sdpd


def test_import():
    assert sdpd._core.__name__ == "stardust_pandas._core"


def test_build_targets_this_mac():
    info = sdpd._core._build_info()
    assert info["cplusplus"] >= 202002  # C++20
    assert info["neon"]


@pytest.fixture
def restore_threads():
    n = sdpd.get_num_threads()
    yield
    sdpd.set_num_threads(n)


def test_pool_defaults_to_performance_cores():
    assert sdpd.get_num_threads() == 10  # M4 Pro: 10 P-cores (DESIGN.md §1.3)


@pytest.mark.parametrize("threads", [1, 2, 10])
def test_pool_runs_every_task_once(restore_threads, threads):
    sdpd.set_num_threads(threads)
    assert sdpd.get_num_threads() == threads
    assert sdpd._core._pool_smoke_test(10_000) == [t * t for t in range(10_000)]


@pytest.mark.parametrize("n", [0, 1025])
def test_set_num_threads_rejects_out_of_range(restore_threads, n):
    with pytest.raises(ValueError):
        sdpd.set_num_threads(n)
