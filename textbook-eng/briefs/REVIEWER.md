# Brief for the reviewer

You review one finished draft chapter of the English textbook series "Foundations of Scientific Computing" independently of its author and FIX what you find, directly in the files. The manager's message names the chapter.

Work only inside /Users/sanghyunpark/Desktop/stardust/textbook-eng/. Edit the chapter file, references.bib, and the Notation part of the volume's main file. Never modify, build or run the rest of the repository. No downloads or installs; you may read web pages for bibliography checks.

Read first, in full: GUIDE.md (binding rules, Korean), the chapter plan in plans/, the comments in sdbook-eng.cls, and the author's notes passed on by the manager.

Notation chapter: it must stay short (two or three pages per volume). Remove entries the chapter added that are local to one section, universally known, or variants of existing entries; add nothing unless later chapters need it.

Four passes.
1. Bibliography. For every \cite key of the chapter verify the entry against Crossref (https://api.crossref.org/works/<DOI> or ?query.bibliographic=...), the publisher or a library catalogue: authors, title, venue, volume, issue, pages, year, publisher, edition; add DOIs. Correct wrong data, never invent it. Read the sentence around each citation and check that the source supports the claim. Statements with a known origin, historical claims and standard facts that lack a source get one (verified); what cannot be verified is softened or removed from the text.
2. Mathematics, algorithms and logic. Check every definition, statement, proof step, worked example, pseudocode line, table entry and number (recompute with plain python3). Look for circular arguments, hidden assumptions, quantifier mistakes, false statements, proofs that do not prove what is stated, and clashes with earlier chapters and with the conventions in GUIDE.md. Fix errors; make the conservative fix where it is a judgment call.
3. Compliance with GUIDE.md, rule by rule (reader level and patient explanation, review-mode or full-proof mode as the volume requires, abstract one paragraph of at most about 120 words, no exercises, footnotes avoided, advanced material only in note boxes, no language syntax, no stardust API explanations in Volumes I–IV, every figure and table referred to, thorough citations, Notes section, design rules of §7). Fix violations.
   Also enforce the inline-fraction rule of GUIDE.md §3: no \frac (or \tfrac, \dfrac) inside inline mathematics; rewrite with a solidus and the parentheses needed to keep the meaning, or move a crowded formula to a display. Rewrite $(1/2)x$-style forms as $x/2$, $(a+b)/2$.
   Likewise no written-out matrix or column vector inline: move it to a display.
   Remove run-in pseudo-environment labels (\emph{Example.}, \emph{Examples.}, \emph{Remark.}, \paragraph{...}) and make the sentence lead in by itself.
4. Prose. Formal academic English, clear and unhurried, British spelling. Remove filler, repeated phrasings, inflated wording, mechanical parallelism, overused words ("crucial", "key", "fundamental", "it is worth noting") and excess dashes; vary sentence openings; keep terminology consistent. Keep all content; never shorten explanations for brevity.

Then `tools/build.sh volN` until warnings 0, render every page with a figure, table, algorithm or box (pdftoppm -r 80 into a temp directory outside the repository, deleted afterwards) and confirm margins, placement, overlaps, fills and axes on top.

Report (Korean, concise): errors found and fixed with locations; judgment calls; bibliography changes and anything unverifiable; guideline violations fixed; scale of prose edits; build result; what remains unresolved. Do not commit.
