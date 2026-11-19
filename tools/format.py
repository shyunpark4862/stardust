"""Format the Python and C++ sources: ruff's safe lint fixes, ruff format, clang-format.

    .venv/bin/python tools/format.py              # the whole repository
    .venv/bin/python tools/format.py csrc/linalg  # only these files or directories

The rules are in pyproject.toml ([tool.ruff]), .clang-format and .clang-format-ignore (vendored
and generated files, which are never touched). Only fixes that ruff marks safe are applied; the
remaining lint findings are printed. tools/check.py checks the same without changing anything.
"""

import concurrent.futures
import difflib
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CPP_SUFFIXES = (".cpp", ".hpp", ".h", ".cc")
CLANG_FORMAT_MAJOR = 23  # formatting changes between major versions
JOBS = 8  # at most 8 processes at a time
CHUNK = 24


def tool(name):
    path = shutil.which(name)
    if path is None:
        sys.exit(f"{name} not found on PATH (brew install {'llvm' if name == 'clang-format' else name})")
    return path


def clang_format():
    exe = tool("clang-format")
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"version (\d+)\.", out)
    if m is None or int(m.group(1)) != CLANG_FORMAT_MAJOR:
        sys.exit(f"clang-format {CLANG_FORMAT_MAJOR}.x is required, found: {out.strip()}")
    return exe


def cpp_files(paths):
    """Tracked and untracked (not ignored) C++ files under `paths`; .clang-format-ignore applies later."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted({f for f in out.splitlines() if f.endswith(CPP_SUFFIXES) and (ROOT / f).is_file()})


def chunks(items, n):
    return [items[i : i + n] for i in range(0, len(items), n)]


def run_clang_format(exe, files, check, diff):
    """Formats `files` in place, or (check) reports the files that would change. Returns 0 or 1."""
    if not files:
        return 0
    if check and diff:
        return clang_format_diff(exe, files)
    args = ["--dry-run", "--Werror"] if check else ["-i"]
    with concurrent.futures.ThreadPoolExecutor(JOBS) as ex:
        runs = list(
            ex.map(
                lambda part: subprocess.run([exe, *args, *part], cwd=ROOT, capture_output=True, text=True),
                chunks(files, CHUNK),
            )
        )
    status = 0
    for r in runs:
        if r.returncode != 0:
            status = 1
            if check:  # one line per file instead of clang-format's per-line warnings
                bad = sorted(set(re.findall(r"^(\S+?):\d+:\d+: error:", r.stderr, re.M)))
                for f in bad:
                    print(f"Would reformat: {f}")
                if not bad:
                    sys.stderr.write(r.stderr)
            else:
                sys.stderr.write(r.stderr)
    return status


def clang_format_diff(exe, files):
    """Prints a unified diff of what clang-format would change; 1 if anything would."""

    def one(f):
        src = (ROOT / f).read_text()
        r = subprocess.run(
            [exe, f"--assume-filename={f}"], cwd=ROOT, input=src, capture_output=True, text=True, check=True
        )
        if r.stdout == src or (r.stdout == "" and src != ""):  # unchanged, or skipped by .clang-format-ignore
            return ""
        lines = difflib.unified_diff(
            src.splitlines(keepends=True), r.stdout.splitlines(keepends=True), f"a/{f}", f"b/{f}"
        )
        return "".join(lines)

    with concurrent.futures.ThreadPoolExecutor(JOBS) as ex:
        diffs = [d for d in ex.map(one, files) if d]
    sys.stdout.write("".join(diffs))
    return 1 if diffs else 0


def main(argv, check=False):
    diff = "--diff" in argv
    paths = [a for a in argv if a != "--diff"] or ["."]
    # ruff reads every file it is given as Python: give it directories and Python files only.
    py_paths = [p for p in paths if (ROOT / p).is_dir() or p.endswith((".py", ".pyi"))]
    ruff = tool("ruff")
    cf = clang_format()
    status = 0
    if not py_paths:
        steps = []
    elif check:
        steps = [
            [ruff, "format", "--check", *(["--diff"] if diff else []), *py_paths],
            [ruff, "check", "--no-fix", *py_paths],
        ]
    else:
        # Format, fix, format: the formatter splits `import a; import b` into lines the import
        # sorter then sees, and the sorter can leave lines for the formatter.
        steps = [[ruff, "format", *py_paths], [ruff, "check", "--fix", *py_paths], [ruff, "format", *py_paths]]
    for cmd in steps:
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode >= 2 or (check and r.returncode != 0):
            status = 1
    if run_clang_format(cf, cpp_files(paths), check, diff):
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
