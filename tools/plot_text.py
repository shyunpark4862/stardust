"""Record the text the plot tests and golden cases use (tests/plot/data/text): the layout of
every string of tests/plot/scenes.py and of the figures of tests/plot/figures.py, and the outlines of
their glyphs, typeset with this machine's TeX Live. The tests read the folder frozen (docs/design/plot.md §8).

    .venv/bin/python tools/plot_text.py            # add what is missing
    .venv/bin/python tools/plot_text.py --fresh    # typeset everything again (after a TeX update)
"""

import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stardust.plot import _text  # noqa: E402
from tests.plot import figures, scenes  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fresh", action="store_true", help="remove the recorded folder first")
    args = ap.parse_args()
    if args.fresh and scenes.TEXT.exists():
        shutil.rmtree(scenes.TEXT)
    with _text.cache(scenes.TEXT):
        for family in _text.FAMILIES:
            e = _text.engine(family)
            before = len(e.strings)
            strings = list(scenes.STRINGS) + list(scenes.NUMBERS)
            for make in scenes.SCENES.values():
                strings += make(family).strings()
            e.layout(list(dict.fromkeys(strings)))
            for make in figures.FIGURES.values():  # a figure typesets what its layout needs
                make(family).to_svg()
            print(f"{family}: {len(e.strings)} strings ({len(e.strings) - before} new), {e.tex_runs} TeX runs")
    n = sum(1 for _ in scenes.TEXT.rglob("*.json*"))
    size = sum(f.stat().st_size for f in scenes.TEXT.rglob("*") if f.is_file())
    print(f"{scenes.TEXT.relative_to(ROOT)}: {n} files, {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
