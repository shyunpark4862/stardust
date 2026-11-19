"""Check formatting and lint without changing any file; exits 1 if anything would change.

    .venv/bin/python tools/check.py               # the whole repository
    .venv/bin/python tools/check.py src tests     # only these files or directories
    .venv/bin/python tools/check.py --diff        # also print what the formatters would change

Runs `ruff format --check`, `ruff check` and `clang-format --dry-run` with the same settings
as tools/format.py, which applies them.
"""

import sys

from format import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], check=True))
