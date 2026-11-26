"""Run the interactive examples of the site's pages and compare their output.

    .sdbuild/stardust/.venv/bin/python tools/check_examples.py [target ...]
    .sdbuild/stardust/.venv/bin/python tools/check_examples.py --coverage AREA [AREA ...]
    .sdbuild/stardust/.venv/bin/python tools/check_examples.py --links

Run from site/ with the pinned build's interpreter; the script stops if
`stardust` is imported from anywhere else.

Targets: `linalg` (every page of an area), `linalg/solve` (one page),
`areas/linalg` (an overview), `start` or `start/conventions`. With no target
every page is checked: src/content/api, src/content/areas and
src/content/start. Examples are the ```python blocks that contain `>>> `
prompts. The blocks of one page run in order in one namespace, so a later
block can use names an earlier one defined. `sd` (stardust) is always
available. The output must match exactly, spaces and line breaks included;
only `...` in the expected output matches any text (doctest's ELLIPSIS), as
pages use it to cut a long error message short. A blank line inside printed
output is written as it prints.

`--coverage AREA` compares the `names:` of the area's pages with the public
names of the pinned build in notes/api-inventory.json: the module's names and,
for every public class, its public members as `Class.member` (a member
inherited from another public class of the area counts on that class only).
It lists names no page documents, names documented twice and names that are
not public. AREA is the site's area key (`array` for the top-level `sd`).

`--links` checks that every in-site link written in the pages
(`/api/<area>/<slug>/`, `/api/<area>/`, `/start/<slug>/`, in the body or in
frontmatter text) and every `seeAlso` and `of` reaches an
existing page. `glance`, `returnedBy` and method `slug` entries without a page
are listed as pending (the templates show them without a link). If the site has been built
(`pnpm build`), it also checks every in-site link of the built pages, `#anchor`
included, every in-site `src` and file link against dist/, and the search
index. A site built under a sub-path is checked with the same setting it was
built with (`SITE_BASE=/stardust`).
"""

from __future__ import annotations

import doctest
import html
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import stardust as sd

SITE = Path(__file__).resolve().parent.parent
if not Path(sd.__file__).resolve().is_relative_to(SITE / ".sdbuild"):
    sys.exit(f"stardust comes from {sd.__file__}, not the pinned build: run .sdbuild/stardust/.venv/bin/python")

CONTENT = SITE / "src" / "content"
API = CONTENT / "api"
AREAS = CONTENT / "areas"
START = CONTENT / "start"
DIST = SITE / "dist"
INVENTORY = SITE / "notes" / "api-inventory.json"
# Blocks may be indented (inside list items or components); the closing
# fence has the same indent as the opening one.
FENCE = re.compile(r"^([ \t]*)```python\n(.*?)^\1```", re.M | re.S)
# Exact comparison: no NORMALIZE_WHITESPACE, so array reprs are checked
# column for column, and `1` does not pass for `True`. ELLIPSIS stays for
# shortened error messages.
FLAGS = doctest.ELLIPSIS | doctest.DONT_ACCEPT_TRUE_FOR_1

# Site area key -> key in api-inventory.json (the array area is the top-level `sd`).
INVENTORY_KEY = {"array": "sd"}


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


# ---- frontmatter (a small reader for the fields these checks need)


def frontmatter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.split("---", 2)[1] if text.startswith("---") else ""


def flow_list(head: str, key: str) -> list[str]:
    """`key: [a, b]` at the top level of the frontmatter."""
    m = re.search(rf"^{key}:\s*\[(.*?)\]", head, re.M | re.S)
    return [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()] if m else []


def scalar(head: str, key: str) -> str | None:
    m = re.search(rf"^{key}:\s*(.+?)\s*$", head, re.M)
    return m.group(1).strip("'\"") if m else None


def method_slugs(head: str) -> list[str]:
    return re.findall(r"\bslug:\s*['\"]?([\w./-]+)", head)


# ---- coverage


def public_names(area: str) -> set[str]:
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))["areas"][INVENTORY_KEY.get(area, area)]["names"]
    public = set(inv)
    classes = {n: v for n, v in inv.items() if v["kind"] == "class"}
    for name, info in classes.items():
        bases = [b for b in info.get("bases", []) if b in classes and b != name]
        for member in info.get("members", {}):
            if member.startswith("_"):
                continue
            if any(member in classes[b].get("members", {}) for b in bases):
                continue
            public.add(f"{name}.{member}")
    return public


# Areas whose class methods repeat module functions of the same name
# (`sd.plot.line` and `Figure.line`): the page of the function documents both.
SHARED_METHODS = {"plot"}


def coverage(area: str) -> int:
    public = public_names(area)
    seen: dict[str, str] = {}
    twice = []
    for path in sorted((API / area).glob("*.mdx")):
        for name in flow_list(frontmatter(path), "names"):
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


# ---- links

LINK = re.compile(r"\]\((/(?:api|start)/[^)\s#]*)(?:#[^)\s]*)?\)|href=\"(/(?:api|start)/[^\"#]*)")


def site_pages() -> set[str]:
    pages = {"/", "/api/", "/api/all/", "/start/"}
    pages |= {f"/api/{p.parent.name}/{p.stem}/" for p in API.glob("*/*.mdx")}
    pages |= {f"/api/{p.stem}/" for p in AREAS.glob("*.mdx")}
    pages |= {f"/start/{p.stem}/" for p in START.glob("*.mdx")}
    return pages


def links() -> int:
    pages = site_pages()
    bad = 0
    pending: list[str] = []

    def ref(area: str, r: str) -> str:
        return f"/api/{r}/" if "/" in r else f"/api/{area}/{r}/"

    sources = [(p, p.parent.name) for p in API.glob("*/*.mdx")] + [(p, p.stem) for p in AREAS.glob("*.mdx")]
    sources += [(p, None) for p in START.glob("*.mdx")]
    for path, area in sorted(sources):
        name = path.relative_to(CONTENT).with_suffix("").as_posix()
        text = path.read_text(encoding="utf-8")
        targets = {a or b for a, b in LINK.findall(text)}
        head = frontmatter(path)
        if area is not None:
            targets |= {ref(area, s) for s in flow_list(head, "seeAlso")}
            if (of := scalar(head, "of")) is not None:
                targets.add(ref(area, of))
            for s in flow_list(head, "glance") + flow_list(head, "returnedBy") + method_slugs(head):
                if ref(area, s) not in pages:
                    pending.append(f"{name}: {ref(area, s)}")
        for t in sorted(targets - pages):
            print(f"  {name}: no page {t}")
            bad += 1
    for line in pending:
        print(f"  pending (shown without a link): {line}")
    bad += anchors()
    print(f"{bad} broken links")
    return 1 if bad else 0


URL = re.compile(r'\b(?:href|src)="(/[^"]*)"')
ID = re.compile(r'\bid="([^"]+)"')
# The sub-path the site is built under (astro.config.mjs reads the same
# variable): "" at the root, "/stardust" under GitHub Pages.
BASE = os.environ.get("SITE_BASE", "/").rstrip("/")


def anchors() -> int:
    """Checks the built pages: every in-site link must reach a page, and its
    #anchor an id on that page. Links to files must reach a file."""
    index = DIST / "index.html"
    if not index.exists():
        print("  (built pages not checked: run `pnpm build` first)")
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
    bad = checked = 0
    for page, (_, text) in sorted(pages.items()):
        for href in sorted(set(URL.findall(text))):
            if href.startswith("//"):
                continue  # another host
            checked += 1
            href = html.unescape(href)
            if BASE and not (href == BASE or href.startswith(f"{BASE}/")):
                print(f"  {page}: {href} is outside the base {BASE}/")
                bad += 1
                continue
            path, _, frag = href[len(BASE) :].partition("#")
            path = unquote(path.partition("?")[0]) or "/"
            if "." in path.rsplit("/", 1)[-1]:
                if not (DIST / path.lstrip("/")).is_file():
                    print(f"  {page}: no file {path}")
                    bad += 1
            elif path not in pages:
                print(f"  {page}: no page {path}")
                bad += 1
            elif frag and unquote(frag) not in pages[path][0]:
                print(f"  {page}: no #{unquote(frag)} on {path}")
                bad += 1
    # The search index holds paths from the site root; the browser adds the base.
    search = DIST / "search.json"
    for item in json.loads(search.read_text(encoding="utf-8")) if search.exists() else []:
        checked += 1
        if item["h"] not in pages:
            print(f"  search.json: no page {item['h']}")
            bad += 1
    print(f"  {len(pages)} built pages, {checked} in-site URLs checked (base {BASE or '/'})")
    return bad


# ---- main


def resolve(target: str) -> list[Path] | None:
    t = target.strip("/")
    for root, prefix in ((START, "start"), (AREAS, "areas")):
        if t == prefix:
            return sorted(root.glob("*.mdx"))
        if t.startswith(prefix + "/"):
            p = (root / t[len(prefix) + 1 :]).with_suffix(".mdx")
            return [p] if p.is_file() else None
    p = API / t
    if p.is_dir():
        return sorted(p.glob("*.mdx"))
    if p.with_suffix(".mdx").is_file():
        return [p.with_suffix(".mdx")]
    return None


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--links":
        return links()
    if argv and argv[0] == "--coverage":
        areas = json.loads(INVENTORY.read_text(encoding="utf-8"))["areas"]
        known = {k for k in areas if k != "sd"} | {"array"}
        unknown = [a for a in argv[1:] if a not in known]
        if unknown or len(argv) < 2:
            print(f"name one or more areas: {', '.join(sorted(known))}", file=sys.stderr)
            return 2
        return max(coverage(a) for a in argv[1:])
    paths: list[Path] = []
    if not argv:
        paths = sorted(API.glob("*/*.mdx")) + sorted(AREAS.glob("*.mdx")) + sorted(START.glob("*.mdx"))
    for t in argv:
        found = resolve(t)
        if found is None:
            print(f"no page or area: {t}", file=sys.stderr)
            return 2
        paths += found
    failed = tried = 0
    for path in paths:
        f, n = check_page(path)
        failed += f
        tried += n
        if n:
            print(f"{'FAIL' if f else 'ok  '} {path.relative_to(CONTENT).with_suffix('').as_posix()}  ({n} examples)")
    print(f"{tried - failed}/{tried} examples pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
