"""Every document reference in the code, tests and documents points at a section that exists,
and every repository path they name exists.

    .venv/bin/python tools/check_refs.py

A reference is `<doc> §N[.M]` (several sections may follow one name: `§1.2, §3`), where <doc>
is a root document (`API.md`, `DESIGN.md`, `ARCHITECTURE.md`, `ROADMAP.md`, also `DESIGN §N`
and `ARCHITECTURE §N`) or an area's document, written with its path (`docs/api/frame.md`) or
short (`frame.md`). The areas are the documents in docs/api/ and docs/design/; a short name
means the design document when written in docs/design/, the api document elsewhere.

A path is a word that starts with one of the repository's top folders (bench/, csrc/, docs/,
src/, tests/, tools/, third_party/). It must exist in the working tree, except:
  - a pattern (`<area>`, `cases_<영역>.py`, `*`, `{a,b}`) is not checked;
  - a Python name after a module's path (`tests/stats/refdata.CLASSES`) needs the module;
  - a path written with `(at commit SHA)` (in Korean, `커밋 SHA의`) on the same line must exist
    at that commit;
  - what the measurements write (GENERATED: the speed baselines and their tables) may be
    missing, and so may another project's path (EXTERNAL: R's sources);
  - ROADMAP.md, the record of what was done when, may name paths that are gone.
Prints each broken reference and path; exits 1 if there is any.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOT_DOCS = ("API.md", "DESIGN.md", "ARCHITECTURE.md", "ROADMAP.md")
AREAS = tuple(sorted({p.stem for kind in ("api", "design") for p in (ROOT / "docs" / kind).glob("*.md")}))
DOCS = {
    **{d: d for d in ROOT_DOCS},
    **{f"docs/{k}/{a}.md": f"docs/{k}/{a}.md" for k in ("api", "design") for a in AREAS},
}
SCAN = [
    "bench",
    "csrc",
    "src",
    "tests",
    "tools",
    "docs/api",
    "docs/design",
    *ROOT_DOCS,
    "README.md",
    "CMakeLists.txt",
]
SUFFIXES = {".cpp", ".hpp", ".py", ".md", ".txt", ".sh"}
NAME = (
    r"(?<![\w/.-])((?:docs/(?:api|design)/)?(?:" + "|".join(map(re.escape, AREAS)) + r")\.md"
    r"|API\.md|DESIGN(?:\.md)?|ARCHITECTURE(?:\.md)?|ROADMAP\.md)"
)
REF = re.compile(NAME + r"((?:(?:[ ,]|and|및|와|과)+§\d+(?:\.\d+)*)+)")
TOPS = ("bench", "csrc", "docs", "src", "tests", "tools", "third_party")
PATH = re.compile(r"(?<![\w./-])((?:" + "|".join(TOPS) + r")/[A-Za-z0-9_./*<>{},-]*[A-Za-z0-9_*>}/])")
AT_COMMIT = re.compile(r"at commit ([0-9a-f]{7,40})\b|커밋 ([0-9a-f]{7,40})의")
GENERATED = (
    re.compile(r"bench/speed/results(/.*)?"),  # python -m bench.speed promote: the baselines
    re.compile(r"bench/[a-z]+/speed\.md"),  # python -m bench.speed table, from the baselines
)
EXTERNAL = (re.compile(r"src/nmath/.*"),)  # R's sources (csrc/special, csrc/stats cite them)
NO_PATH_CHECK = {"ROADMAP.md"}


def headings(path):
    text = (ROOT / path).read_text()
    return set(re.findall(r"^#+\s+§?(\d+(?:\.\d+)*)[.\s]", text, re.M))


def resolve(name, where):
    if name in ("DESIGN", "ARCHITECTURE"):  # the root documents, written short
        return name + ".md"
    if name.startswith("docs/") or name in DOCS:
        return name
    area = name[: -len(".md")]
    kind = "design" if "docs/design/" in str(where) else "api"
    return f"docs/{kind}/{area}.md"


def files():
    for top in SCAN:
        base = ROOT / top
        found = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.suffix in SUFFIXES)
        for f in found:
            if "refmap" in f.name or "/golden/" in str(f) and f.suffix == ".json":
                continue
            yield f


def path_problem(p, line, cache={}):  # noqa: B006 -- a cache of `git cat-file` answers
    """Why the path p written on `line` is broken, or None."""
    if any(c in p for c in "<>*{}"):
        return None
    if any(g.fullmatch(p) for g in GENERATED + EXTERNAL):
        return None
    module = re.fullmatch(r"(.+)\.([A-Za-z_]\w*)", p)
    if module and not (ROOT / p).exists() and (ROOT / (module.group(1) + ".py")).is_file():
        return None
    m = AT_COMMIT.search(line)
    if m:
        sha = m.group(1) or m.group(2)
        key = (sha, p)
        if key not in cache:
            r = subprocess.run(["git", "cat-file", "-e", f"{sha}:{p}"], cwd=ROOT, capture_output=True)
            cache[key] = r.returncode == 0
        return None if cache[key] else f"not at commit {sha}"
    if (ROOT / p).exists():
        return None
    return "no such file or folder"


def check(lines_of):
    """(section references checked, broken ones, paths checked, broken ones) over the files of
    lines_of: {relative path: lines}; prints each broken one."""
    known = {doc: headings(doc) for doc in DOCS.values() if (ROOT / doc).exists()}
    refs = bad_refs = paths = bad_paths = 0
    for rel, lines in lines_of.items():
        for lineno, line in enumerate(lines, 1):
            for m in REF.finditer(line):
                doc = resolve(m.group(1), rel)
                for sec in re.findall(r"§(\d+(?:\.\d+)*)", m.group(2)):
                    refs += 1
                    if doc not in known or sec not in known[doc]:
                        bad_refs += 1
                        print(f"{rel}:{lineno}: {m.group(1)} §{sec} -> no such section in {doc}")
            if rel in NO_PATH_CHECK:
                continue
            for m in PATH.finditer(line):
                after = line[m.end() : m.end() + 1]
                if after and after in "<{*":  # cases_<영역>.py: a pattern
                    continue
                p = m.group(1).rstrip("/.,")
                paths += 1
                why = path_problem(p, line)
                if why:
                    bad_paths += 1
                    print(f"{rel}:{lineno}: {p} -> {why}")
    return refs, bad_refs, paths, bad_paths


def main():
    lines_of = {str(f.relative_to(ROOT)): f.read_text(errors="replace").splitlines() for f in files()}
    refs, bad_refs, paths, bad_paths = check(lines_of)
    print(f"{refs} references checked, {bad_refs} broken; {paths} paths checked, {bad_paths} broken")
    sys.exit(1 if bad_refs or bad_paths else 0)


if __name__ == "__main__":
    main()
