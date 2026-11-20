"""The golden outputs (tools/golden.py): the whole check bit for bit at 1, 3 and 10 threads (deep,
serial: it runs its own worker processes), and in quick the case files' coverage of the public
API and the tool's encoding, comparison and record rules."""

import ast
import importlib
import io
import json
import pathlib
import re
import subprocess
import sys
import tokenize

import numpy as np
import pytest

import stardust as sd
from bench import areas

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden"
for p in (str(ROOT / "tools"), str(GOLDEN)):
    if p not in sys.path:
        sys.path.insert(0, p)

import golden  # noqa: E402  (tools/golden.py)


def run(args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.deep
@pytest.mark.serial
def test_golden_outputs_are_unchanged():
    code, out = run(["tools/golden.py", "check"])
    assert code == 0, out[-20000:]


# ---- the case files cover the public API


def namespace(area):
    return sd if area == "array" else getattr(sd, area)


def used_words(path):
    """The identifiers of a case file's code and the words of its strings, without comments,
    docstrings and the EXCLUDED and THREAD_DEPENDENT lists: a public name counts as covered when
    the code calls it or a case is named after it."""
    src = path.read_text()
    docs = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("EXCLUDED", "THREAD_DEPENDENT") for t in node.targets
        ):
            docs.update(range(node.lineno, node.end_lineno + 1))
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            v = body[0].value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                docs.update(range(body[0].lineno, body[0].end_lineno + 1))
    words = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME:
            words.add(tok.string)
        elif tok.type in (tokenize.STRING, tokenize.FSTRING_MIDDLE) and tok.start[0] not in docs:
            words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", tok.string))
    return words


@pytest.fixture(scope="module")
def case_modules():
    lib = importlib.import_module("_lib").Lib()
    out = {}
    for area in areas.AREAS:
        mod = importlib.import_module(f"cases_{area}")
        names = [name for name, _ in mod.cases(lib)]
        out[area] = (mod, names)
    return out


@pytest.mark.parametrize("area", areas.AREAS)
def test_every_public_name_has_a_case_or_a_reason(area, case_modules):
    mod, names = case_modules[area]
    words = used_words(GOLDEN / f"cases_{area}.py")
    for name in names:
        words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", name))
    excluded = getattr(mod, "EXCLUDED", {})
    public = list(namespace(area).__all__)
    missing = [n for n in public if n not in words and n not in excluded]
    assert not missing, f"{area}: public names without a golden case or an EXCLUDED reason: {missing}"
    for n, why in excluded.items():
        assert n in public, f"{area}: EXCLUDED names {n!r}, not a public name"
        assert n not in words, f"{area}: EXCLUDED names {n!r}, which a case calls"
        assert isinstance(why, str) and why.strip(), f"{area}: EXCLUDED[{n!r}] needs a reason"


@pytest.mark.parametrize("area", areas.AREAS)
def test_thread_dependent_cases_exist_and_say_why(area, case_modules):
    mod, names = case_modules[area]
    assert len(set(names)) == len(names), f"{area}: duplicate case names"
    for n, why in getattr(mod, "THREAD_DEPENDENT", {}).items():
        assert n in names, f"{area}: THREAD_DEPENDENT names {n!r}, not a case"
        assert isinstance(why, str) and why.strip(), f"{area}: THREAD_DEPENDENT[{n!r}] needs a reason"


# ---- encoding and comparison


@pytest.fixture(scope="module")
def enc():
    return golden.Encoder(importlib.import_module("_lib").Lib())


def stored(enc, x):
    """What check compares: the encoding through JSON and back."""
    return json.loads(json.dumps(enc(x)))


def differences(enc, got, want):
    out = []
    golden.diff(stored(enc, got), stored(enc, want), "", out)
    return out


class OtherValueError(Exception):
    pass


OtherValueError.__name__ = OtherValueError.__qualname__ = "ValueError"


@pytest.mark.parametrize(
    "got, want",
    [
        (True, 1),
        (0, False),
        ([True, False], [1, 0]),
        (1.5, np.float64(1.5)),
        (1 + 2j, np.complex128(1 + 2j)),
        (np.int64(3), 3),
        (ValueError("x"), OtherValueError("x")),
        (1.0, 1),
        (-0.0, 0.0),
        ((1, 2), [1, 2]),
        ("a", np.str_("a")),
    ],
    ids=repr,
)
def test_a_change_of_type_or_value_is_a_difference(enc, got, want):
    a, b = stored(enc, got), stored(enc, want)
    assert not golden.same(a, b)
    assert differences(enc, got, want)


def test_an_equal_result_is_equal(enc):
    x = sd.asarray(np.arange(5000.0))
    assert golden.same(stored(enc, x), stored(enc, sd.asarray(np.arange(5000.0))))
    assert golden.same(stored(enc, [ValueError("x"), 1.5, True]), stored(enc, [ValueError("x"), 1.5, True]))


def test_errors_and_warnings_name_their_module():
    e = golden.Encoder(importlib.import_module("_lib").Lib())(sd.linalg.LinAlgError("singular"))
    assert e["error"].endswith(".LinAlgError") and e["error"].startswith("stardust")
    assert golden.qualname(ValueError) == "builtins.ValueError"


def test_a_large_array_difference_is_placed_and_counted(enc):
    base = np.arange(5000.0)
    changed = base.copy()
    changed[[10, 4000, 4001]] += 1.0
    (path, text), *rest = differences(enc, sd.asarray(changed.reshape(50, 100)), sd.asarray(base.reshape(50, 100)))
    assert text.startswith("2 of 20 chunks of 256 elements differ (5000 elements); the first is elements 0-255")
    one_ulp = base.copy()
    one_ulp[1234] = np.nextafter(one_ulp[1234], 1e9)
    (path, text), *rest = differences(enc, sd.asarray(one_ulp), sd.asarray(base))
    assert "1 of 20 chunks" in text and "elements 1024-1279" in text


def test_chunks_stay_few_on_long_arrays():
    assert golden.chunk_len(5000) == golden.CHUNK
    n = 10**6
    assert -(-n // golden.chunk_len(n)) <= golden.MAX_CHUNKS


def test_nan_bits_are_named_and_any_nan_forgets_them(enc):
    nan = np.full(5000, np.nan)
    other = nan.copy()
    other[7] = -np.nan
    a, b = stored(enc, sd.asarray(other)), stored(enc, sd.asarray(nan))
    out = []
    golden.diff(a, b, "", out)
    assert out == [("", "only NaN bits differ (sign or payload; 5000 elements, sha)")]
    assert golden.same(golden.any_nan(a), golden.any_nan(b))


def test_float_difference_shows_the_ulp_distance(enc):
    (path, text), *_ = differences(enc, 1.0, float(np.nextafter(1.0, 2.0)))
    assert text.endswith(", 1 ulp")
    (path, text), *_ = differences(enc, -0.0, 0.0)
    assert "ulp" not in text


# ---- threads and record


def test_merge_threads_keeps_out_thread_dependent_results():
    per = {1: {"a": {"r": 1}, "b": {"r": True}}, 3: {"a": {"r": 1}, "b": {"r": 1}}}
    merged, unstable = golden.merge_threads(per)
    assert merged == {"a": {"r": 1}} and unstable == ["b"]  # True and 1 differ
    merged, unstable = golden.merge_threads(per, allowed={"b": "why"})
    assert unstable == [] and merged["b"] == {"per_threads": {"1": {"r": True}, "3": {"r": 1}}}


class FakeCases:
    THREAD_DEPENDENT = {"allowed": "the reason"}

    @staticmethod
    def cases(L):
        for name in ("same", "allowed", "differs", "new"):
            yield name, None


def fake_record(monkeypatch, tmp_path, per, argv):
    """golden.main(argv) with run_all's results given: the record logic without workers."""
    out = tmp_path / "random.json"
    monkeypatch.setattr(golden, "load_area", lambda area: (None, FakeCases))
    monkeypatch.setattr(golden, "count_cases", lambda areas, patterns: {a: 4 for a in areas})
    threads = sorted(per)
    names = [n for n in ("same", "allowed", "differs", "new") if any(n in per[t] for t in threads)]
    res = {t: {n: per[t][n] for n in names} for t in threads}
    monkeypatch.setattr(
        golden, "run_all", lambda areas, ths, jobs, patterns, con, counts: ({"random": res}, {}, 0.0, [])
    )
    monkeypatch.setattr(golden.cli, "log_file", lambda tool: tmp_path / "log.txt")
    code = golden.main(["record", "random", "--out", str(out), "--threads", ",".join(map(str, threads)), *argv])
    return code, json.loads(out.read_text())


def test_record_refuses_a_thread_dependent_result(monkeypatch, tmp_path):
    per = {
        1: {"same": {"r": 1}, "allowed": {"r": 1}, "differs": {"r": 1}},
        3: {"same": {"r": 1}, "allowed": {"r": 2}, "differs": {"r": 2}},
    }
    code, doc = fake_record(monkeypatch, tmp_path, per, [])
    assert code == 1
    assert doc["format"] == golden.FORMAT and doc["threads"] == [1, 3]
    assert set(doc["cases"]) == {"same", "allowed"}  # "differs" is not stored
    assert doc["cases"]["allowed"] == {"per_threads": {"1": {"r": 1}, "3": {"r": 2}}}


def test_partial_record_replaces_matching_cases_and_drops_gone_ones(monkeypatch, tmp_path):
    out = tmp_path / "random.json"
    old = {"same": {"r": 0}, "allowed": {"r": 0}, "gone one": {"r": 0}, "kept": {"r": 0}}
    out.write_text(json.dumps({"format": golden.FORMAT, "area": "random", "threads": [1, 10], "cases": old}))
    per = {3: {"same": {"r": 5}}}
    code, doc = fake_record(monkeypatch, tmp_path, per, ["--match", "same", "--match", "gone*"])
    assert code == 0
    assert doc["cases"] == {"same": {"r": 5}, "allowed": {"r": 0}, "kept": {"r": 0}}
    assert doc["threads"] == [1, 3, 10]


def test_partial_record_will_not_rewrite_an_old_format_file(monkeypatch, tmp_path):
    out = tmp_path / "random.json"
    out.write_text(json.dumps({"format": 1, "area": "random", "threads": [1], "cases": {"same": {"r": 0}}}))
    monkeypatch.setattr(golden.cli, "log_file", lambda tool: tmp_path / "log.txt")
    code = golden.main(["record", "random", "--out", str(out), "--match", "same"])
    assert code == 2
    assert json.loads(out.read_text())["format"] == 1


def test_options_follow_the_shared_conventions(tmp_path):
    code, out = run(["tools/golden.py", "list", "nowhere"])
    assert code == 2 and "unknown area 'nowhere'" in out
    code, out = run(["tools/golden.py", "check", "--case", "x"])
    assert code == 2
    code, out = run(["tools/golden.py", "check", "random", "special", "--out", str(tmp_path / "x.json")])
    assert code == 2 and "exactly one AREA" in out
    code, out = run(["tools/golden.py", "check", "random", "--jobs", "11"])
    assert code == 2


def test_check_reports_a_thread_count_not_recorded(tmp_path):
    out = tmp_path / "random.json"
    code, text = run(["tools/golden.py", "record", "random", "--match", "default_rng 12345", "--threads", "1",
                      "--jobs", "1", "--out", str(out)])  # fmt: skip
    assert code == 0, text
    doc = json.loads(out.read_text())
    name = next(iter(doc["cases"]))
    doc["cases"][name] = {"per_threads": {"1": doc["cases"][name]}}
    out.write_text(json.dumps(doc))
    code, text = run(["tools/golden.py", "check", "random", "--match", name, "--threads", "1,2", "--jobs", "1",
                      "--out", str(out)])  # fmt: skip
    assert code == 1 and "not recorded at 2 threads" in text, text


def test_frame_csv_inputs_go_to_the_run_folder(monkeypatch, tmp_path):
    frame_cases = importlib.import_module("cases_frame")
    monkeypatch.setattr(frame_cases, "_CSV_DIR", [])
    monkeypatch.setenv("GOLDEN_TMP", str(tmp_path))
    p = frame_cases._csv_path("a,b\n1,2\n")
    assert p.parent == tmp_path / "frame-csv" and p.read_bytes() == b"a,b\n1,2\n"
    assert frame_cases._csv_path(b"a,b\n1,2\n") == p
    assert [q.name for q in p.parent.iterdir()] == [p.name]  # no partly written file left
