"""Superseded: csrc/stats/cost_table.hpp (and csrc/special/cost_table.hpp) are now written
from the speed baselines by `python -m bench.speed cost` (bench/README.md). This script
used to parse bench/stats/t4, t5, t6 speed.md; it now runs the new generator.

    .venv/bin/python tools/gen_stats_cost.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bench.speed.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["cost", *sys.argv[1:]]))
