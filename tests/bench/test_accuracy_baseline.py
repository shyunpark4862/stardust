"""The accuracy against the baselines, whole library (deep, serial: bench.accuracy runs its own
worker processes): a run of every sweep, then its comparison with bench/accuracy/results, which
fails on a value beyond its contract or a measure worse than the baseline."""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.deep, pytest.mark.serial]


def run(args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_accuracy_is_no_worse_than_the_baselines(tmp_path):
    run_file = tmp_path / "accuracy.json"
    code, out = run(["-m", "bench.accuracy", "run", "--out", str(run_file)])
    assert code == 0, out[-20000:]
    code, out = run(["-m", "bench.accuracy", "compare", str(run_file)])
    assert code == 0, out[-20000:]
