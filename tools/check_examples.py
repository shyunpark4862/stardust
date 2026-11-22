"""Run every interactive example in the API pages and compare the output.

    ../stardust/.venv/bin/python tools/check_examples.py [area[/slug] ...]
    ../stardust/.venv/bin/python tools/check_examples.py --coverage array

Examples are the ```python blocks that contain `>>> ` prompts. The blocks of
one page run in order in one namespace, so a later block can use names an
earlier one defined. `sd` (stardust) is always available.

`--coverage <area>` lists the public names of that area that no page documents
(the `names:` lists in the frontmatter) and names documented twice. For the
classes in MEMBERS, every public member counts as a name too (`Series.head`).

`--links` checks that every `/api/<area>/<slug>/` and `/internals/...` link and
every `seeAlso:` slug points at an existing page. If the site has been built
(`pnpm build`), it also checks every in-site link of the built pages, `#anchor`
included, against the ids those pages really have.

With no arguments every API area and every internals area is checked. A
target that names no page is an error.
"""

from __future__ import annotations

import doctest
import html
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import stardust as sd

SITE = Path(__file__).resolve().parent.parent
CONTENT = SITE / "src" / "content"
DIST = SITE / "dist"
ROOT = CONTENT / "api"
# Blocks may be indented (inside list items or components); the closing
# fence has the same indent as the opening one.
FENCE = re.compile(r"^([ \t]*)```python\n(.*?)^\1```", re.M | re.S)
FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

NAMESPACES = {
    "array": sd,
    "linalg": sd.linalg,
    "frame": sd.frame,
    "special": sd.special,
    "stats": sd.stats,
    "random": sd.random,
    "regression": sd.regression,
    "optimize": sd.optimize,
    "sparse": sd.sparse,
    "htest": sd.htest,
    "integrate": sd.integrate,
    "differentiate": sd.differentiate,
    "plot": sd.plot,
}


def check_page(path: Path) -> tuple[int, int]:
    """Returns (failures, examples) for one page."""
    text = path.read_text(encoding="utf-8")
    blocks = [b for _, b in FENCE.findall(text) if ">>> " in b]
    if not blocks:
        return 0, 0
    # MDX indents blocks inside components; doctest needs the prompts flush left.
    source = "\n".join(_dedent(b) for b in blocks)
    globs = {"sd": sd}
    test = doctest.DocTestParser().get_doctest(source, globs, path.stem, str(path), 0)
    runner = doctest.DocTestRunner(optionflags=FLAGS, verbose=False)
    runner.run(test, clear_globs=False)
    return runner.failures, runner.tries


def _dedent(block: str) -> str:
    lines = block.split("\n")
    indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    lines = [l[indent:] for l in lines]
    # A blank line inside printed output (3-D arrays print one between their
    # 2-D blocks) would end the expected output for doctest. Mark it instead,
    # so the page can show the output exactly as Python prints it.
    for i in range(1, len(lines) - 1):
        nxt = lines[i + 1]
        if not lines[i].strip() and lines[i - 1].strip() and nxt.strip() and not nxt.startswith(">>>"):
            lines[i] = "<BLANKLINE>"
    return "\n".join(lines)


def names_of(path: Path) -> list[str]:
    head = path.read_text(encoding="utf-8").split("---")[1]
    m = re.search(r"^names:\s*\[(.*?)\]", head, re.M | re.S)
    return [n.strip() for n in m.group(1).split(",") if n.strip()] if m else []


# Classes whose public members are documented by name too, as `Class.member`.
MEMBERS = {
    "frame": ("Series", "DataFrame", "Index", "RangeIndex"),
    "plot": ("Figure",),
}

# Areas whose class methods repeat module functions of the same name
# (`sd.plot.line` and `Figure.line`): the page of the function documents both.
SHARED_METHODS = {"plot"}


def coverage(area: str) -> int:
    ns = NAMESPACES[area]
    public = set(ns.__all__)
    for cls in MEMBERS.get(area, ()):
        public |= {f"{cls}.{m}" for m in dir(getattr(ns, cls)) if not m.startswith("_")}
    seen: dict[str, str] = {}
    twice = []
    for path in sorted((ROOT / area).glob("*.mdx")):
        for name in names_of(path):
            if name in seen:
                twice.append(f"{name}: {seen[name]} and {path.stem}")
            seen[name] = path.stem
    covered = set(seen)
    if area in SHARED_METHODS:
        covered |= {n for n in public if "." in n and n.split(".", 1)[1] in seen}
    missing = sorted(public - covered)
    unknown = sorted(set(seen) - public)
    print(f"{area}: {len(public) - len(missing)}/{len(public)} public names documented")
    if missing:
        print("  missing:", " ".join(missing))
    if unknown:
        print("  not public:", " ".join(unknown))
    for line in twice:
        print("  documented twice:", line)
    return 1 if (unknown or twice) else 0


LINK = re.compile(r"/(api|internals)/([a-z]+)/([a-z0-9_-]+)/")


def links() -> int:
    pages = {(sec, p.parent.name, p.stem) for sec in ("api", "internals") for p in (CONTENT / sec).glob("*/*.mdx")}
    bad = 0
    for sec in ("api", "internals"):
        for path in sorted((CONTENT / sec).glob("*/*.mdx")):
            text = path.read_text(encoding="utf-8")
            targets = set(LINK.findall(text))
            m = re.search(r"^seeAlso:\s*\[(.*?)\]", text.split("---")[1], re.M)
            if m:
                targets |= {(sec, path.parent.name, s.strip()) for s in m.group(1).split(",") if s.strip()}
            for t in sorted(targets - pages):
                print(f"  {sec}/{path.parent.name}/{path.stem}: no page /{t[0]}/{t[1]}/{t[2]}/")
                bad += 1
    bad += anchors()
    print(f"{bad} broken links")
    return 1 if bad else 0


HREF = re.compile(r'href="(/[^"]*)"')
ID = re.compile(r'\bid="([^"]+)"')


def anchors() -> int:
    """Checks the built pages: every in-site link must reach a page, and its
    #anchor an id on that page. Headings get their ids at build time, so this
    is the only place a renamed heading shows up as a broken link."""
    index = DIST / "index.html"
    if not index.exists():
        print("  (anchors not checked: run `pnpm build` first)")
        return 0
    built = index.stat().st_mtime
    sources = [p for d in ("src", "public") for p in (SITE / d).rglob("*") if p.is_file()]
    if any(p.stat().st_mtime > built for p in sources):
        print("  (warning: dist/ is older than the sources; run `pnpm build` for current anchors)")
    pages: dict[str, tuple[set[str], str]] = {}
    for f in DIST.rglob("index.html"):
        rel = f.parent.relative_to(DIST).as_posix()
        text = f.read_text(encoding="utf-8")
        pages["/" if rel == "." else f"/{rel}/"] = ({html.unescape(i) for i in ID.findall(text)}, text)
    bad = 0
    for page, (_, text) in sorted(pages.items()):
        for href in sorted(set(HREF.findall(text))):
            path, _, frag = html.unescape(href).partition("#")
            if path.startswith("//") or "." in path.rsplit("/", 1)[-1]:
                continue  # another host, or a file such as /favicon.svg
            target = unquote(path or page)
            if target not in pages:
                print(f"  {page}: no page {target}")
                bad += 1
            elif frag and unquote(frag) not in pages[target][0]:
                print(f"  {page}: no #{unquote(frag)} on {target}")
                bad += 1
    return bad


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--links":
        return links()
    if argv and argv[0] == "--coverage":
        unknown = [a for a in argv[1:] if a not in NAMESPACES]
        if unknown:
            print(f"unknown area: {' '.join(unknown)} (one of {', '.join(NAMESPACES)})", file=sys.stderr)
            return 2
        return max((coverage(a) for a in argv[1:] or ["array"]), default=0)
    targets = argv or [
        f"{sec}/{p.name}" if sec == "internals" else p.name
        for sec in ("api", "internals")
        for p in sorted((CONTENT / sec).iterdir())
        if p.is_dir()
    ]
    paths: list[Path] = []
    for t in targets:
        # "internals[/<area>[/<slug>]]" checks the internals articles instead.
        t = t.strip("/")
        p = CONTENT / t if t == "internals" or t.startswith("internals/") else ROOT / t
        if p.is_dir():
            paths += sorted(p.rglob("*.mdx"))
        elif p.with_suffix(".mdx").is_file():
            paths.append(p.with_suffix(".mdx"))
        else:
            print(f"no page or area: {t}", file=sys.stderr)
            return 2
    failed = tried = 0
    for path in paths:
        f, n = check_page(path)
        failed += f
        tried += n
        if n:
            print(f"{'FAIL' if f else 'ok  '} {path.parent.name}/{path.stem}  ({n} examples)")
    print(f"{tried - failed}/{tried} examples pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
