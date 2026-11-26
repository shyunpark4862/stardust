# Writer brief (common part)

You write reference pages for the English documentation site of the Python library stardust, in
/Users/sanghyunpark/Desktop/stardust/site (Astro, MDX content).

Read first, in full, before writing anything:
1. site/WRITING.md — binding rules. §2 (voice: professional library reference, imperative summaries, impersonal,
   no "you") and §4 (page anatomy) matter most. Re-read §2 before your final pass.
2. site/plans/<area>.md — the outline and the "공통 결정" (shared decisions) section. The shared decisions are binding:
   other agents write sibling pages in parallel and all pages must agree in terms, type notation, group names and slugs.
3. The model pages named in the shared decisions. Match their frontmatter shape, tone, and example style.

Facts: only from the pinned build (site/.sdbuild/stardust: source, tests, docs) and from running
site/.sdbuild/stardust/.venv/bin/python. Never read or run the repository root's src/, csrc/, docs/, .venv.
The old Korean pages in /Users/sanghyunpark/Desktop/stardust/site-legacy/src/content/api/<area>/ may be read for
ideas about what matters to explain; do not translate them and re-verify every fact.

Rules: create or edit only the files assigned to you. Do not edit the area overview, templates, tools, or other
agents' pages (if you think one needs a change, say so in your report). No git commands that change anything, no
`pnpm build`, no installs or downloads, no subagents, no long-running servers. At most 4 processes at a time.
Record library anomalies in site/notes/anomalies/<area>.md (append; `## title`, reproduction, expectation) and
document what the library actually does.

Before reporting, for every page: `cd site && .sdbuild/stardust/.venv/bin/python tools/check_examples.py <area>/<slug>`,
`node tools/fmcheck.mjs <file>`, `node tools/mdxcheck.mjs <file>`; all must pass (warnings about pages that do not
exist yet but are in the outline's slug list are fine).

Report in Korean, ≤ 20 lines: pages written, check results, judgment calls, any slug/group you changed or needs
changing, facts that sibling pages should know (e.g. a rule you documented that they should link to), anomalies.
