# Brief for the chapter author

You are the author of one chapter of the English textbook series "Foundations of Scientific Computing: Theory, Algorithms, and Implementation" (LaTeX). The manager's message names the chapter, its plan file and its .tex file.

Work only inside /Users/sanghyunpark/Desktop/stardust/textbook/. The rest of the repository is being edited by another developer: read it if you need to, never modify, build or run it. Do not download or install anything.

## Read first, in full
1. GUIDE.md — the binding writing rules (Korean). Re-read it before you finish and check the chapter against every rule.
2. The chapter plan in plans/ — sections, results to prove, figures, note boxes. Follow it; add what helps a first-year reader.
3. OUTLINE.md — where the chapter sits and what neighbouring chapters cover, so that you neither repeat nor presuppose them wrongly.
4. sdbook-eng.cls comments — every command of the fixed design: theorem environments (theorem, lemma, proposition, corollary, definition, assumption; no remark/example/listing — worked examples go in the running text), \sdfloat{content}{caption\label{}} for every figure and table, sdtable / sdlongtable, the sdplot pgfplots style, \sdpanel, the algorithm design with \Name, the `note` box, lists, symbols lists, \term{}, \code{}, \abstract{}, \chapterreferences.
5. plans/PROMISES.md — forward promises made by earlier chapters. Keep every promise addressed to your chapter (mark it [완료] there) and add the promises your chapter makes to later ones.
6. The previous chapter of the volume (chapters/) — for notation, conventions, tone and depth. Use what it has established; refer to its results with \cref.

## What to write
- The chapter file, and the edits to the volume's main file (volN.tex): \input the chapter in its place (start a new \part where OUTLINE.md starts one) and add the chapter's new symbols to the Notation chapter.
- Notation chapter: add ONLY symbols that later chapters will also use, or conventions that differ between books and that a reader must be able to look up. Do not list symbols used in one section and defined there, universally known symbols, or variants of an entry already present. The whole Notation chapter of a volume must stay within two or three pages; when in doubt, leave it out.
- Figures as TikZ/pgfplots inside the chapter. No external images.
- references.bib: add the entries you cite.

## Rules that are easy to miss (GUIDE.md is authoritative)
- Reader: first-year university student, basic calculus and a little programming. Detailed and patient explanation outranks brevity; length does not matter. Formal academic English ("we"), never compressed. British spelling.
- \abstract: ONE paragraph, at most about 120 words, motivation rather than summary.
- Volumes I–IV are review volumes: state every result formally, prove the important ones, others may go unproved — but explanation, examples and figures are never cut. Pure mathematics/computer science: do not explain stardust APIs there (a forward-looking mention is fine). Topical volumes (V–XVI): prove everything.
- A proof omitted for length: state the result, give the storyline and key idea inside the proof environment and cite a reference, without saying why it is omitted.
- Mark every such proof as a sketch: \begin{proof}[sketch] ... \end{proof}. It is headed "Proof (sketch)" and ends with a dagger (\sketchsymbol) in the place of the square; \qedhere works as usual. A sketch is any proof that does not establish the whole statement from results available in the book up to that point: it gives the storyline or key idea and leaves the rest to a reference, a later chapter or another volume, or proves only part of the statement and describes the rest. Only complete proofs use plain \begin{proof} (with the square). Citing earlier results, being short, or adding "see also ..." does not make a proof a sketch. Do not open a sketch with "We sketch the proof" or similar; the heading already says so.
- Citations must be thorough: every result with a known origin, every historical claim and every standard fact gets a source (original paper or standard text). Never guess bibliographic data; list every entry you are not certain of in your report under "서지 확인 필요".
- Inline mathematics ($...$) writes fractions with a solidus, not \frac: $a/b$, $(x+1)/(x-1)$, $1/2$, also inside exponents and subscripts; add parentheses so the meaning cannot change. A numerical fraction in front of a factor is written after it: $x/2$ and $(a+b)/2$, not $(1/2)x$ or $(1/2)(a+b)$. \frac (and \binom-style stacked forms that raise the line) belong in displayed equations.
- Written-out matrices and column vectors use square brackets (bmatrix), never parentheses (pmatrix); determinants keep vmatrix.
- Matrices written out entry by entry (pmatrix, bmatrix, column vectors) go in displayed equations, not inline; inline text refers to them by name or in row form (x_1, …, x_n). Anything else that would stretch the line spacing also goes in a display.
- No pseudo-environments either: never open a paragraph with a run-in label such as \emph{Example.}, \emph{Examples.}, \emph{Remark.}, \textbf{Note.} or \paragraph{...}; examples and remarks are ordinary sentences that lead in by themselves.
- No exercises, avoid footnotes, material beyond a standard first course goes in a `note` box, no explanation of any particular language's syntax.
- The natural numbers include 0. Other volumes are "Volume IV" (roman). Chapters not yet written are named in words ("Chapter 3") rather than with \cref.
- End the chapter with \chapternotes (the unnumbered section "Notes": reading guide and sources), then \chapterreferences.
- Every figure and table is referred to in the text. In plots, axes are drawn over fills and curves.
- Numbers: compute them yourself (plain python3, fractions/decimal). When the library must be run, use only the shared build described in BUILD.md, and never trust numbers quoted in the library's documentation.

## Build and check (you do this; the manager will not)
- `tools/build.sh volN` until it reports warnings 0 (no errors, undefined references or overfull boxes).
- After drawing figures you MUST look at the rendered pages: `pdftoppm -r 80 -png pdf/volN.pdf <temp dir>/p`, open every page with a figure, table, algorithm or box, and check margins, placement, overlaps, fills, axes on top, legibility. Fix and re-render until clean. Temp files go outside the repository and are deleted afterwards.
- Re-check the mathematics once more after writing: every statement, proof step, example and number.

## Report (Korean, concise)
Files written; page count; numbers of sections, results, proofs, figures, tables, algorithms, note boxes; build result; what you checked on the rendered pages; "서지 확인 필요" list; conventions you introduced; any rule you could not fully satisfy; anything you are unsure of. Do not commit.
