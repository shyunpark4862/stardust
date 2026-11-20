"""Check formatting and lint without changing any file, and (on the whole repository) the size of
each area's reference data; exits 1 if anything would change or a size is over its limit.

    .venv/bin/python tools/check.py               # the whole repository
    .venv/bin/python tools/check.py src tests     # only these files or directories
    .venv/bin/python tools/check.py --diff        # also print what the formatters would change

Runs `ruff format --check`, `ruff check` and `clang-format --dry-run` with the same settings
as tools/format.py, which applies them. The reference data of an area is the sum of its
reference folders, tests/<area>/ref and tests/<area>/ref_*, in MiB (2^20 bytes); each area
may hold REF_LIMIT_MIB.
"""

import pathlib
import sys

from format import main as format_main

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF_LIMIT_MIB = 60


def ref_sizes():
    """{area: MiB of its reference folders} for the areas under tests/ that have one."""
    out = {}
    for area in sorted(p for p in (ROOT / "tests").iterdir() if p.is_dir()):
        folders = [d for d in area.iterdir() if d.is_dir() and (d.name == "ref" or d.name.startswith("ref_"))]
        if folders:
            size = sum(f.stat().st_size for d in folders for f in d.rglob("*") if f.is_file())
            out[area.name] = size / 2**20
    return out


def check_ref_sizes():
    """Prints the areas over REF_LIMIT_MIB; 1 if there is one."""
    over = {a: s for a, s in ref_sizes().items() if s > REF_LIMIT_MIB}
    for a, s in over.items():
        print(f"tests/{a}: reference folders hold {s:.2f} MiB, over the limit of {REF_LIMIT_MIB} MiB")
    return 1 if over else 0


def main(argv):
    status = format_main(argv, check=True)
    if not [a for a in argv if a != "--diff"]:
        status = max(status, check_ref_sizes())
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
