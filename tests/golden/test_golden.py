"""The deep tier's whole-library checks: the golden outputs (tools/golden.py check, bit for bit at
1, 3 and 10 threads) and the accuracy against the baselines (bench.accuracy run and compare:
beyond a contract, or worse than the baseline in any measure). Both run their own processes,
so they are serial too."""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.deep, pytest.mark.serial]


def run(args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_golden_outputs_are_unchanged():
    code, out = run(["tools/golden.py", "check"])
    assert code == 0, out[-20000:]


def test_accuracy_is_no_worse_than_the_baselines(tmp_path):
    run_file = tmp_path / "accuracy.json"
    code, out = run(["-m", "bench.accuracy", "run", "--out", str(run_file)])
    assert code == 0, out[-20000:]
    code, out = run(["-m", "bench.accuracy", "compare", str(run_file)])
    assert code == 0, out[-20000:]
