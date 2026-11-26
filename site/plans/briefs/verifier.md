# Verifier brief (common part)

You verify and correct reference pages of the stardust English documentation site in
/Users/sanghyunpark/Desktop/stardust/site. Another agent wrote them; assume nothing is right until checked.

Read first, in full: site/WRITING.md, site/plans/<area>.md (outline + "공통 결정"), site/plans/briefs/writer.md
(the rules the writers had; the same restrictions on sources, files, git, installs apply to you).

For every assigned page check, and fix directly in the page:
1. Facts: every statement in description, params, returns, raises, notes and the comparison section is true for the
   pinned build. Do not trust the page: read the source in site/.sdbuild/stardust and run it. Try edge cases the page
   mentions (and obvious ones it does not: empty input, 0-D, wrong dtype, negative axis, NaN). Each `raises` entry must
   be reproducible; each claim about NumPy/SciPy/pandas must be reproduced in the pinned interpreter; R claims with
   Rscript if available, otherwise remove claims you cannot confirm.
2. Completeness: every parameter documented with correct type/default/kind; return shape and dtype stated; important
   behaviors a user would hit are not missing.
3. Voice (WRITING.md §2): professional library reference. Imperative verb-first summaries, noun-phrase parameter
   descriptions, impersonal declarative prose; no "you/we/let's", questions, chatty or tutorial tone, filler,
   history, meta talk, implementation internals, speed claims, em-dash glue, Unicode fake math.
4. Structure (WRITING.md §4): section order, exact heading names, `<Reference />` placement, empty sections omitted.
5. Consistency with the shared decisions and with sibling pages: terms, type notation, group names, how shared rules
   are linked rather than re-explained, same wording for the same fact across pages.
6. Examples: correct, minimal, deterministic, output exact (the checker compares exactly, spaces included).

Then run check_examples, fmcheck, mdxcheck on each page and `tools/check_examples.py --coverage <area>` and `--links`.
Issues in pages that are not assigned to you: do not edit; list them in site/notes/consistency/<milestone>.md
(file, line or quote, what is inconsistent, suggested fix).

Report in Korean, ≤ 25 lines: pages checked, number and kinds of corrections (with the 5 most significant),
remaining doubts, cross-page issues recorded, anomalies recorded.
