"""Every document reference in the code, tests and documents points at a section
that exists.

    .venv/bin/python tools/check_refs.py

A reference is `<doc> §N[.M]` (several sections may follow one name: `§1.2, §3`),
where <doc> is one of the documents below, written with its path
(`docs/api/frame.md`), with a short name inside docs/ (`frame.md`,
`linalg.md`), or bare for the root documents (`API.md`, `DESIGN.md`,
`ARCHITECTURE.md`, also `DESIGN §N` and `ARCHITECTURE §N`). Short names `array.md`, `linalg.md`, `frame.md` mean the
api document when written in docs/api/ or code, the design document when
written in docs/design/.
Prints each broken reference; exits 1 if there is any.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = {
    "API.md": "API.md",
    "DESIGN.md": "DESIGN.md",
    "ARCHITECTURE.md": "ARCHITECTURE.md",
    "ROADMAP.md": "ROADMAP.md",
    **{f"docs/{k}/{a}.md": f"docs/{k}/{a}.md" for k in ("api", "design") for a in ("array", "linalg", "frame")},
}
SCAN = [
    "csrc",
    "src",
    "tests",
    "tools",
    "docs/api",
    "docs/design",
    "bench/merge",
    "API.md",
    "DESIGN.md",
    "ARCHITECTURE.md",
    "ROADMAP.md",
    "README.md",
    "CMakeLists.txt",
]
SUFFIXES = {".cpp", ".hpp", ".py", ".md", ".txt", ".sh"}
NAME = r"(?<![\w/.-])((?:docs/(?:api|design)/)?(?:array|linalg|frame)\.md|API\.md|DESIGN(?:\.md)?|ARCHITECTURE(?:\.md)?|ROADMAP\.md)"
REF = re.compile(NAME + r"((?:(?:[ ,]|and|및|와|과)+§\d+(?:\.\d+)*)+)")


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


def main():
    known = {doc: headings(doc) for doc in DOCS.values() if (ROOT / doc).exists()}
    bad = checked = 0
    for top in SCAN:
        base = ROOT / top
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.suffix in SUFFIXES]
        for f in files:
            if "refmap" in f.name or "/golden/" in str(f) and f.suffix == ".json":
                continue
            for lineno, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                for m in REF.finditer(line):
                    doc = resolve(m.group(1), f.relative_to(ROOT))
                    for sec in re.findall(r"§(\d+(?:\.\d+)*)", m.group(2)):
                        checked += 1
                        if doc not in known or sec not in known[doc]:
                            bad += 1
                            print(f"{f.relative_to(ROOT)}:{lineno}: {m.group(1)} §{sec} -> no such section in {doc}")
    print(f"{checked} references checked, {bad} broken")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
