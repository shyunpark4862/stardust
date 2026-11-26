# Writing the stardust site

This file is the brief every writing and verification agent reads first. The site is English only.
Paths are relative to the repository root unless they start with `site/`.

## 1. Ground rules

- **Pinned build only.** Every fact comes from the pinned build in `site/.sdbuild/stardust` (commit in `site/README.md`).
  Run code with `site/.sdbuild/stardust/.venv/bin/python` (check `sd.__file__` points into `.sdbuild`).
  Never read or run the live working tree (`src/`, `csrc/`, `docs/` at the repository root); those change under you.
- **Sources, in order of authority:** the code in the clone (`.sdbuild/stardust/src/stardust/<area>/`, `csrc/<area>/`),
  then actually running it, then `.sdbuild/stardust/tests/<area>/`, `.sdbuild/stardust/API.md`,
  `.sdbuild/stardust/docs/api/<area>.md`, `.sdbuild/stardust/docs/design/<area>.md`. When a document and the code
  disagree, the code is the fact. Docstrings in the build are mostly one-line signatures.
- `site/API-INVENTORY.md` (and `site/notes/api-inventory.json`) lists every public name with its signature. Page
  coverage and signatures are checked against it.
- The old Korean site in `site-legacy/src/content/api/<area>/` may be read for ideas (what to explain, good examples),
  but it can be out of date and must not be translated wholesale. Re-check every fact against the pinned build.
- Touch only the files your brief names, inside `site/`. Never edit anything outside `site/` (not the library, not
  `site-legacy/`). No git commands that change anything, no `pnpm install`, no downloads or installs of any kind.
- Library problems you find (wrong results, crashes, misleading errors, signature/doc mismatches) are **not** fixed and
  not worked around silently. Record each in `site/notes/anomalies/<area>.md` as `## <short title>` with a minimal
  reproduction and what you expected; then write the page about what the library actually does today.

## 2. Audience and voice

- Readers know basic Python. They do **not** know NumPy, SciPy, pandas or R. Never explain something by pointing to
  those libraries ("works like numpy.x"). Explain it directly; define a term the first time a page uses it, briefly, or
  link to the area overview's Concepts section or to Conventions (`/start/conventions/`).
- **Register: the voice of a professional library reference** (NumPy, SciPy, the Python standard library, R help
  pages). Precise, neutral, impersonal, concise. Not a tutorial, blog or chat voice. Concretely:
  - Summaries are imperative and start with a verb: "Solve a linear system.", "Return the indices that sort an array.",
    "Compute the matrix exponential." Classes: a noun phrase: "Result of a two-sample t-test."
  - Parameter and return descriptions are noun phrases or terse clauses: "Coefficient matrix.", "If True, ...",
    "Number of samples. Must be positive." Not "This is the matrix you want to ...".
  - Description and Notes are third person and declarative ("The function returns ...", "`solve` raises
    `LinAlgError` when ..."). Avoid "you", "we", "let's", questions, exclamations, rhetorical setups, analogies and
    reassurance ("don't worry", "easy", "just", "simply", "of course").
  - Explanations a beginner needs are given as definitions and facts, compactly, not as conversation. Being
    self-contained for readers who do not know NumPy/SciPy/R does not mean being chatty.
  - Example lead-ins are short statements: "Solve a 2-by-2 system:", "Several right-hand sides at once:".
- Present tense, short sentences, concrete words. No marketing tone, no filler ("powerful", "seamlessly",
  "it's worth noting", "Note that ..." openers, "In summary").
- Describe what the library does now. No history: no dates, commits, milestones, "now", "no longer", "new", "was
  changed", "fixed". No meta talk: no "the design document says", "according to the API doc", decision numbers
  (`SP3`, `R12`, `D-33` ...), section signs.
- Hide the implementation. Do not describe internal data structures, C++ classes, threading internals, or which
  external library does the work. Name an algorithm only when the user needs it to choose or to understand results
  (e.g. "Nelder–Mead uses no derivatives", "the LU factorization uses partial pivoting"), and only in Notes.
- No speed figures. Accuracy statements only when the code or the area's accuracy record supports them; prefer
  qualitative wording.
- Determinism is stated as: same machine, same number of threads, same input → the same bits every run. Say more only
  where an area guarantees more.
- American spelling. Function and parameter names always in backticks.
- Avoid em dashes as sentence glue; use a comma, a colon, parentheses or two sentences.

## 3. Site structure

- `/` home. `/start/` Install, `/start/tour/` a short tour, `/start/conventions/` shared rules (array shapes, `axis`,
  missing values, result objects and `summary()`, random seeds, determinism, errors).
- `/api/` all areas. `/api/<area>/` area overview. `/api/<area>/<slug>/` one page per function, per closely related
  family that shares one explanation (`sin`/`cos`/`tan`; `zeros`/`ones`/`full`/`empty`), per class, or per method of a
  large class. `/api/all/` alphabetical index of every public name.
- Pages live in `site/src/content/api/<area>/<slug>.mdx`; overviews in `site/src/content/areas/<area>.mdx`; Getting
  started pages in `site/src/content/start/<slug>.mdx`. The schema is `site/src/content.config.ts`.

## 4. Page anatomy

### Function page (`kind: function`)

Frontmatter (rendered by the template, not written in the body):

```yaml
kind: function
title: solve                      # the name, or a short English title for a family page
names: [solve]                    # every public name this page documents (members as Class.method)
group: Linear systems             # must match a group title in the area overview
order: 1                          # position inside the group
signatures:                       # exactly the pinned signature(s), prefixed with the full path
  - "sd.linalg.solve(a, b, /)"
summary: Solve a linear system a @ x = b for x.   # one imperative sentence, ends with a period
params:
  - name: a
    type: array of shape (..., n, n)
    kind: positional-only          # optional: positional-only | keyword-only
    default: None                  # optional; omit when required
    desc: The coefficient matrix. ...
    choices:                       # optional: value | meaning rows, shown as a table
      - [ '"two-sided"', means differ ]
returns:                           # one entry, or several for a tuple (give each a name)
  - type: ndarray of shape (..., n) or (..., n, k)
    desc: ...
raises:
  - type: LinAlgError
    desc: when a is singular to working precision.
seeAlso: [lstsq, linalg/inv, frame/DataFrame]   # same-area slug, or area/slug
```

Body (MDX), in this order; leave out a section that would be empty:

1. **Description**: 1–4 short paragraphs right under the frontmatter, no heading. What it does and when to use it.
2. `## Examples`: 2–4 groups, each introduced by one sentence. `>>>` examples with real output. The last group may show
   a common mistake or edge case (including the error it raises).
3. `<Reference />`: the template's Parameters / Returns / Raises block.
4. `## Notes`: definitions, formulas, numerical caveats, the method when it matters. Math only here.
5. `## References`: numbered, only for real published sources the page relies on.
6. `## For NumPy, SciPy and R users`: only when there is something useful to say. Name the nearest equivalents
   (`numpy.linalg.solve`, `scipy.stats.ttest_ind`, R `t.test`) and list real differences (defaults, argument names,
   result shape or names, NaN handling) as 2–6 bullets. Verify every claim about NumPy/SciPy/pandas by running them in
   the pinned interpreter (they are installed there). For R, state only what you are sure of from R's documentation
   or by running `Rscript` if available; otherwise leave R out.

The template adds the See also list at the end from `seeAlso`.

### Class and result-object page (`kind: class`)

Frontmatter as above, plus:

```yaml
kind: class
returnedBy: [ttest_ind, ttest_rel]        # functions that return it (slugs)
attributes:
  - { name: statistic, type: float, desc: ... }
methods:
  - { name: summary, signature: "summary()", summary: Return a printable report. }
  - { name: confidence_interval, signature: "confidence_interval(level=0.95)", summary: ..., slug: TTestResult-confidence_interval }  # slug only when the method has its own page
```

Body: description; `## Examples` (create it, read two or three attributes, print `summary()`); `<Reference />` (renders
Attributes and Methods tables); `## The summary() report` when the class has one, showing the output and saying which
attribute each part comes from; `## Methods` with an `### name` subsection per method that has no page of its own;
then Notes / References / For NumPy, SciPy and R users. Small result objects keep all methods on one page. Large
classes (`ndarray`, `DataFrame`, `Series`, sparse arrays, `Generator`, plot figures) get one page per method
(`kind: method`, `of: <Class slug>`), shaped like a function page.

### Area overview (`site/src/content/areas/<area>.mdx`)

```yaml
title: Linear algebra
ns: sd.linalg
lead: One to three sentences: what this area covers and what it does not (and where to go instead).
glance: [solve, lstsq, eig, svd]      # 4–8 most-used slugs, shown as cards
groups:
  - { title: Linear systems, desc: One sentence about the group. }
```

Body: `## Concepts` (short: shared definitions, shapes, the result-object pattern; math allowed); `<FunctionTables />`
(renders every group as a table of name + summary, in `groups` order); `<ResultTable />` when the area returns result
objects; `## For NumPy, SciPy and R users` (a 3-column table stardust | NumPy/SciPy | R for the main functions, then
3–8 bullets of the important differences).

## 5. Examples

- Python fenced blocks with `>>> ` / `... ` prompts and the exact output the pinned build prints. All blocks on a page
  run in order in one namespace; `import stardust as sd` is implied (write it once in the first block anyway).
- Deterministic: seed random generators, keep outputs short (slice or round when an array is long).
- Each example finishes in well under a second. Show printed `summary()` reports in full when the page is about them.
- Check with `cd site && .sdbuild/stardust/.venv/bin/python tools/check_examples.py <area>/<slug>`.

## 6. Math and markup

- Inline `$...$`, display `$$...$$` (rendered at build time). A single variable in running text that is not code is
  math (`$n$`); code identifiers stay in backticks. Never fake math with Unicode (no `x̂`, `‖x‖`, `2⁻⁵³`).
  Frontmatter, captions and headings are plain text.
- `<` and `>` inside math: `\lt`, `\gt`. Inside a table cell use `\lvert ... \rvert` instead of `|`.
- Links: `/api/<area>/<slug>/` to existing pages only (or slugs your brief says will exist). `/start/conventions/#...`
  anchors only as listed in your brief.
- Components available in MDX: `Callout` (`type="note" | "tip" | "warn"`), and the figure components in
  `site/src/components/fig/` (`Plot`, `PlaneFig`, `ArrayFig`, `Steps`). Figures only where they really help (0–2 per
  page); figure data comes from actually running the code, exported as `export const` at the top of the page.
  `sd.plot` images appear only on plot pages.

## 7. Checks before you report

- `node tools/fmcheck.mjs <files>` (frontmatter), `node tools/mdxcheck.mjs <files>` (MDX syntax) when present.
- Example checker on every page you touched; coverage check for the area
  (`tools/check_examples.py --coverage <area>`); link check (`--links`).
- Do not run `pnpm build` unless your brief says so; never commit.

## 8. Report

End with a short report in Korean: files created or changed, example-check results, judgment calls you made,
anomalies recorded (count and file), anything you could not verify.
