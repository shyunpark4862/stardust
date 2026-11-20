"""The extension builds, imports, and was compiled for the intended target."""

import pytest

import stardust as sd


def test_import():
    assert sd._core.__name__ == "stardust._core"


def test_build_targets_this_mac():
    info = sd._core._build_info()
    assert info["cplusplus"] >= 202002  # C++20
    assert info["neon"]
    assert info["fcma"]  # complex multiply-accumulate, used by the complex multiply kernel


def test_accelerate_is_linked_with_ilp64():
    info = sd._core._build_info()
    assert info["blas_ilp64"]
    assert info["accelerate_ddot"] == 32.0


def test_pool_defaults_to_performance_cores():
    assert sd.get_num_threads() == 10  # M4 Pro: 10 P-cores


@pytest.mark.parametrize("threads", [1, 2, 10])
def test_pool_runs_every_task_once(restore_threads, threads):
    sd.set_num_threads(threads)
    assert sd.get_num_threads() == threads
    assert sd._core._pool_smoke_test(10_000) == [t * t for t in range(10_000)]


@pytest.mark.parametrize("n", [0, 1025])
def test_set_num_threads_rejects_out_of_range(restore_threads, n):
    with pytest.raises(ValueError):
        sd.set_num_threads(n)
