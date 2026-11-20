"""The test tools' own rules: tools/test.py's options and exit status (procrun), tools/coverage.py's
options, tools/check_refs.py's references and paths, tools/check.py's reference sizes."""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load(name):
    """tools/<name>.py as a module (tools/test.py would clash with the standard library's test)."""
    spec = importlib.util.spec_from_file_location(f"sd_tools_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


procrun = load("procrun")
testrun = load("test")
coverage = load("coverage")
check_refs = load("check_refs")
check = load("check")


@pytest.mark.parametrize(
    "codes, status",
    [
        ([0, 0], 0),
        ([0, 5], 0),
        ([5, 0], 0),
        ([5, 5], 1),  # nothing selected anywhere
        ([1, 0], 1),
        ([0, 1], 1),
        ([1, 2], 2),  # interrupted
        ([4, 0], 2),  # pytest's usage error
        ([3, 5], 2),  # internal error
    ],
)
def test_pytest_status(codes, status):
    assert procrun.pytest_status(codes) == status


@pytest.mark.parametrize("arg", ["-k", "-kfoo", "-m", "-mserial", "-n", "-n4", "--numprocesses=4", "--tier"])
def test_refused_pytest_args(arg):
    assert procrun.refused_pytest_args(["-x", arg]).startswith(repr(arg))
    assert procrun.refused_pytest_args(["-x", "-q", "tests/frame", "--durations=5"]) is None


def test_test_options():
    args, rest = testrun.parse(["special", "array", "--match", "a", "--match", "b*", "--jobs", "3", "--", "-x"])
    assert args.areas == ["array", "special"] and args.match == ["a", "b*"] and args.jobs == 3 and rest == ["-x"]
    args, rest = testrun.parse([])
    assert args.areas == [] and args.jobs == 8 and rest == []


@pytest.mark.parametrize(
    "argv",
    [["nowhere"], ["-n", "4"], ["--jobs", "11"], ["--", "-k", "x"], ["--", "-m", "serial"], ["--case", "x"]],
    ids=repr,
)
def test_test_usage_errors(argv, capsys):
    with pytest.raises(SystemExit) as e:
        testrun.parse(argv)
    assert e.value.code == 2


@pytest.mark.parametrize("argv", [["run", "nowhere"], ["run", "--", "-k", "x"], ["build", "--jobs", "0"]], ids=repr)
def test_coverage_usage_errors(argv, capsys):
    with pytest.raises(SystemExit) as e:
        coverage.main(argv)
    assert e.value.code == 2


def test_coverage_crashes_list_is_empty():
    assert coverage.CRASHES == set()


def test_no_test_selected_is_a_failure():
    r = subprocess.run(
        [
            sys.executable,
            "tools/test.py",
            "--jobs",
            "1",
            "--match",
            "no_test_has_this_name",
            "--",
            "tests/test_tools.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, r.stdout[-3000:]
    assert "no tests selected" in r.stdout


def test_check_refs_finds_broken_sections_and_paths():
    # The lines are written with "@" for "/" and "#" for "§", so tools/check_refs.py, which reads
    # this file too, does not see them as references.
    lines = {
        "tools/golden.py": [
            "see DESIGN.md #1.6 and docs@api@array.md #99",  # one good, one broken section
            "tools@golden.py, tools@nothere.py, tests@golden@cases_<영역>.py, bench@<area>@x.md",
            "tests@stats@refdata.CLASSES, bench@speed@results@x.json, bench@array@speed.md",
            "bench@merge@golden (at commit 4151d2a)",
            "커밋 4151d2a의 bench@merge@golden",  # the same, as the Korean documents write it
        ],
        "ROADMAP.md": ["bench@nothere@at@all.md"],  # the record of what was done: paths not checked
    }
    lines = {k: [ln.replace("@", "/").replace("#", "§") for ln in v] for k, v in lines.items()}
    refs, bad_refs, paths, bad_paths = check_refs.check(lines)
    assert (refs, bad_refs) == (2, 1)
    assert bad_paths == 1  # the one that is not there


def test_check_refs_knows_every_area_document():
    for area in ("array", "linalg", "frame", "special", "stats", "regression", "random", "optimize"):
        assert f"docs/api/{area}.md" in check_refs.DOCS and f"docs/design/{area}.md" in check_refs.DOCS


def test_reference_sizes_in_mib(tmp_path, monkeypatch, capsys):
    (tmp_path / "tests" / "stats" / "ref").mkdir(parents=True)
    (tmp_path / "tests" / "stats" / "ref_mv").mkdir()
    (tmp_path / "tests" / "stats" / "data").mkdir()
    (tmp_path / "tests" / "stats" / "ref" / "a.npz").write_bytes(b"\0" * 2**20)
    (tmp_path / "tests" / "stats" / "ref_mv" / "b.npz").write_bytes(b"\0" * 2**19)
    (tmp_path / "tests" / "stats" / "data" / "c.json").write_bytes(b"\0" * 2**21)  # not reference data
    (tmp_path / "tests" / "frame").mkdir()
    monkeypatch.setattr(check, "ROOT", tmp_path)
    assert check.ref_sizes() == {"stats": 1.5}
    monkeypatch.setattr(check, "REF_LIMIT_MIB", 1)
    assert check.check_ref_sizes() == 1
    assert "tests/stats: reference folders hold 1.50 MiB" in capsys.readouterr().out
