"""Draw the figures of the pages from the pages' own examples.

    ../.venv/bin/python tools/plot_figures.py [page ...]
    ../.venv/bin/python tools/plot_figures.py --check

A page shows a figure with `<PlotImage name="..." alt="..." />` placed after
the ```python block that builds it. The example blocks of a page run in
order in one namespace, as in check_examples.py. When a tag is reached, the
`Figure` in the variable its `of` attribute names (default `fig`) is drawn
into public/plot/<name>.svg, or public/plot/<name>.png when the tag says
`format="png"` (3D figures, whose surfaces are pictures anyway).

src/data/plots.json records, for every figure, the page it belongs to, its
format and its outer size in mm; the PlotImage component reads it.

A page is named by its path under src/content without the extension
(`api/plot/function`). With no page every page that has a PlotImage tag is
drawn, and files no page uses any more are removed. `--check` draws in
memory and lists missing, stale and unused files without writing anything.
"""

from __future__ import annotations

import doctest
import fcntl
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import stardust as sd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_examples import FENCE, FLAGS, _dedent  # noqa: E402

SITE = Path(__file__).resolve().parent.parent
CONTENT = SITE / "src" / "content"
OUT = SITE / "public" / "plot"
MANIFEST = SITE / "src" / "data" / "plots.json"
TAG = re.compile(r"<PlotImage\b(.*?)/>", re.S)
ATTR = re.compile(r'(\w+)="([^"]*)"')
DPI = 300


def figures_of(path: Path) -> tuple[list[tuple[str, str, bytes, float, float]], int]:
    """Draws the figures of one page: [(name, format, data, width, height)], failures."""
    text = path.read_text(encoding="utf-8")
    events = [(m.start(), "block", m.group(2)) for m in FENCE.finditer(text) if ">>> " in m.group(2)]
    events += [(m.start(), "tag", m.group(1)) for m in TAG.finditer(text)]
    events.sort()
    globs = {"sd": sd}
    runner = doctest.DocTestRunner(optionflags=FLAGS, verbose=False)
    parser = doctest.DocTestParser()
    figures = []
    failures = 0
    for _, kind, body in events:
        if kind == "block":
            test = parser.get_doctest(_dedent(body), globs, path.stem, str(path), 0)
            log = io.StringIO()
            with redirect_stdout(log):
                result = runner.run(test, clear_globs=False, out=log.write)
            # DocTest runs in a copy of the namespace; carry it to the next block.
            globs = test.globs
            if result.failed:
                failures += result.failed
                print(log.getvalue(), end="")
            continue
        attrs = dict(ATTR.findall(body))
        name, var, fmt = attrs.get("name"), attrs.get("of", "fig"), attrs.get("format", "svg")
        where = f"{path.relative_to(CONTENT)}: PlotImage {name!r}"
        if not name or not re.fullmatch(r"[a-z0-9-]+", name):
            raise SystemExit(f"{where}: name must be lower-case letters, digits and '-'")
        fig = globs.get(var)
        if not isinstance(fig, sd.plot.Figure):
            raise SystemExit(f"{where}: no Figure in the variable {var!r} at this point")
        if fmt == "svg":
            data = fig.to_svg().encode("utf-8")
        elif fmt == "png":
            data = fig.to_png(dpi=DPI)
        else:
            raise SystemExit(f"{where}: format must be 'svg' or 'png', not {fmt!r}")
        figures.append((name, fmt, data, float(fig.width), float(fig.height)))
    return figures, failures


def page_id(path: Path) -> str:
    return str(path.relative_to(CONTENT).with_suffix(""))


def main(argv: list[str]) -> int:
    check = "--check" in argv
    targets = [a for a in argv if a != "--check"]
    pages = sorted(p for p in CONTENT.glob("*/*/*.mdx") if "<PlotImage" in p.read_text(encoding="utf-8"))
    if targets:
        chosen = []
        for t in targets:
            path = CONTENT / f"{t}.mdx"
            if not path.exists():
                raise SystemExit(f"no page: {t}")
            chosen.append(path)
        pages = chosen
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    drawn = {page_id(p) for p in pages}
    if not targets:
        # A full run redraws everything: entries of pages without figures go too.
        drawn |= {e["page"] for e in manifest.values()}
    manifest = {n: e for n, e in manifest.items() if e["page"] not in drawn}
    problems = failures = 0
    files: dict[str, bytes] = {}
    for path in pages:
        figs, failed = figures_of(path)
        failures += failed
        for name, fmt, data, width, height in figs:
            if name in manifest or f"{name}.svg" in files or f"{name}.png" in files:
                raise SystemExit(f"{page_id(path)}: figure name {name!r} is used twice")
            manifest[name] = {
                "page": page_id(path),
                "format": fmt,
                "width": round(width, 4),
                "height": round(height, 4),
            }
            files[f"{name}.{fmt}"] = data
    used = {f"{n}.{e['format']}" for n, e in manifest.items()}
    for fname, data in sorted(files.items()):
        target = OUT / fname
        if target.exists() and target.read_bytes() == data:
            continue
        problems += 1
        print(f"{'stale' if target.exists() else 'missing'}: public/plot/{fname}")
        if not check:
            OUT.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if not targets:
        for target in sorted(OUT.glob("*")) if OUT.exists() else []:
            if target.name not in used:
                problems += 1
                print(f"unused: public/plot/{target.name}")
                if not check:
                    target.unlink()
    text = json.dumps(dict(sorted(manifest.items())), ensure_ascii=False, indent=1) + "\n"
    if not MANIFEST.exists() or MANIFEST.read_text(encoding="utf-8") != text:
        problems += 1
        print("stale: src/data/plots.json")
        if not check:
            MANIFEST.write_text(text, encoding="utf-8")
    print(f"{len(files)} figures from {len(pages)} pages, {failures} failing examples")
    if check:
        return 1 if (problems or failures) else 0
    return 1 if failures else 0


if __name__ == "__main__":
    # Several runs at once would lose each other's entries in the manifest.
    with open(SITE / ".plot-figures.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sys.exit(main(sys.argv[1:]))
