%% sdbook-eng.cls -- Foundations of Scientific Computing (English edition)
%%
%% A thin layer over Springer Nature's SNmono (v6.00), which is loaded unchanged:
%% trim size, text block, chapter/section heads, running heads, captions, the
%% theorem look and the 10/12pt Times text all come from SNmono. This file only
%% adds what the book needs on top: fonts for pdfLaTeX, cross-references,
%% pseudocode, plots, tables and per-chapter bibliographies.
%% Engine: pdfLaTeX (latexmk, see latexmkrc).
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sdbook-eng}[2026/11/25 v0.1 English edition on SNmono]

% ---------------------------------------------------------------------------
% 0. Base class
% ---------------------------------------------------------------------------
% sectrefs: a reference list at the end of each chapter
% nospthms: the theorem environments are rebuilt below with amsthm in exactly
%           SNmono's style, so that cleveref can tell a lemma from a theorem
% noalg: leave out the pseudocode setup (for documents that load another package)
\newif\ifsd@alg \sd@algtrue
\DeclareOption{noalg}{\sd@algfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{SNmono}}
\ProcessOptions\relax
\LoadClass[sectrefs,nospthms]{SNmono}

% ---------------------------------------------------------------------------
% 1. Fonts: Times text and math (newtx), Helvetica, Inconsolata
% ---------------------------------------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage{amsmath,amssymb,mathtools}
\let\Bbbk\relax
\RequirePackage{newtxtext}
\RequirePackage[varvw]{newtxmath}
\RequirePackage[scaled=0.92]{helvet}
\RequirePackage[varqu,scaled=0.95]{zi4}
\RequirePackage{microtype}

% ---------------------------------------------------------------------------
% 2. Cross-references (cleveref must come before the theorem environments)
% ---------------------------------------------------------------------------
\RequirePackage[hidelinks,bookmarksnumbered]{hyperref}
\RequirePackage[capitalise,noabbrev]{cleveref}
\crefname{algorithm}{Algorithm}{Algorithms}
% Captions and references say "Figure" and "Table" in full (user's choice 2026-11-25;
% SNmono's own word is "Fig.", set again at \begin{document}, hence AtBeginDocument)
\crefname{figure}{Figure}{Figures}
\Crefname{figure}{Figure}{Figures}
\crefname{table}{Table}{Tables}
\AtBeginDocument{\def\figurename{Figure}\def\tablename{Table}}
\crefname{assumption}{Assumption}{Assumptions}

% ---------------------------------------------------------------------------
% 3. Theorem environments -- SNmono's own look, rebuilt with amsthm
% ---------------------------------------------------------------------------
% theorem, lemma, proposition, corollary: bold head, italic body
% definition: bold head, upright body
% All share one counter numbered within the chapter (SNmono's envcountsame + envcountchap).
\RequirePackage{amsthm}
\RequirePackage{aliascnt}
\newtheoremstyle{spthm}{\topsep}{\topsep}{\itshape}{0pt}{\bfseries}{}{0.5em}%
  {\thmname{#1}\thmnumber{ #2}\thmnote{ (#3)}}
\newtheoremstyle{spdef}{\topsep}{\topsep}{\upshape}{0pt}{\bfseries}{}{0.5em}%
  {\thmname{#1}\thmnumber{ #2}\thmnote{ (#3)}}
\newcommand\sd@newthm[2]{%
  \newaliascnt{#1}{theorem}\newtheorem{#1}[#1]{#2}\aliascntresetthe{#1}}
\theoremstyle{spthm}
\newtheorem{theorem}{Theorem}[chapter]
\sd@newthm{lemma}{Lemma}
\sd@newthm{proposition}{Proposition}
\sd@newthm{corollary}{Corollary}
\theoremstyle{spdef}
\sd@newthm{definition}{Definition}
\sd@newthm{assumption}{Assumption}
% Proof: italic head as in SNmono, the box set flush right
\renewcommand\qedsymbol{\hbox{\rlap{$\sqcap$}$\sqcup$}}
\renewenvironment{proof}[1][\proofname]{\par
  \pushQED{\qed}\normalfont\topsep6\p@\@plus6\p@\relax
  \trivlist\item[\hskip\labelsep\itshape #1\@addpunct{.}]\ignorespaces
}{\popQED\endtrivlist\@endpefalse}

% ---------------------------------------------------------------------------
% 4. Pseudocode: a ruled float, caption below like figures and tables
% ---------------------------------------------------------------------------
\RequirePackage{float}
% Design chosen 2026-11-25: rule above; Input and Output; the procedure name in small
% capitals (\Name{...}); numbered lines (small numbers) with indentation guides and no
% "end" lines; comments in typewriter type after "//"; rule below; "Algorithm 2.1" and an
% optional one-line description underneath as a caption, like figures and tables.
% (wrapped in a macro: skipping it with \iffalse would trip over the \iffalse inside)
\newcommand\sd@setupalg{%
\newcommand\sd@algcapt[2]{{\small\textbf{##1}\enskip ##2\par}}
\newcommand\fs@sdruled{\def\@fs@cfont{\bfseries}\let\@fs@capt\sd@algcapt
  \def\@fs@pre{\hrule height.6pt depth0pt \kern4pt}%
  \def\@fs@mid{\kern3pt\hrule height.6pt\kern6pt}%
  \def\@fs@post{}\let\@fs@iftopcapt\iffalse}
\floatstyle{sdruled}
\newfloat{algorithm}{htbp}{loa}[chapter]
\floatname{algorithm}{Algorithm}
\floatstyle{plain}
\RequirePackage[noEnd=true,indLines=true,italicComments=false,commentColor=black,
  beginComment={\ttfamily//~},beginLComment={\ttfamily//~},endLComment={}]{algpseudocodex}
\algrenewcommand\algorithmicrequire{\textbf{Input:}}
\algrenewcommand\algorithmicensure{\textbf{Output:}}
\algrenewcommand\alglinenumber[1]{{\scriptsize ##1}}
\tikzset{algpxIndentLine/.style={draw=black, line width=0.3pt}}
\AtBeginEnvironment{algorithmic}{\small}
% the procedure name with its parameters, placed after Input and Output: \Name{NextAfter}{a,b};
% a thin full-width rule separates Input and Output from the name
% \Name closes the list that holds Input and Output, draws the rule across the full width
% outside any list, and opens the numbered body (Input and Output carry no numbers, so the
% body still starts at line 1)
\newcommand\Name[2]{\end{algorithmic}\par\vskip-\lastskip\vskip3\p@
  \hrule height0.4\p@\vskip3\p@
  \begin{algorithmic}[1]%
  \item[\proc{##1}$(##2)$]}
}
\newcommand\proc[1]{\textsc{#1}}
\ifsd@alg\sd@setupalg\fi

% ---------------------------------------------------------------------------
% 5. Figures: black-and-white pgfplots
% ---------------------------------------------------------------------------
\RequirePackage{graphicx}
\RequirePackage{tikz}
\RequirePackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{arrows.meta,calc,positioning,decorations.pathreplacing,patterns}
% series are told apart by line style: solid, dashed, dotted, dash-dotted, grey
\pgfplotscreateplotcyclelist{sdbw}{%
  {black, solid, line width=0.7pt},
  {black, dashed, line width=0.7pt},
  {black, densely dotted, line width=0.8pt},
  {black, dashdotted, line width=0.7pt},
  {black!45, solid, line width=0.7pt},
  {black!45, dashed, line width=0.7pt}}
\pgfplotsset{
  % function plots, after the Nature figure style but in the serif text face
  % (user's choice 2026-11-25): left and bottom axis lines only, short outward ticks,
  % no grid, no frame around the legend. The legend sits inside the plot; a figure
  % leaves room for it at the top (one row: legend columns=-1) or moves it with
  % legend pos / legend style={at=..., anchor=...}.
  sdplot/.style={
    axis x line*=bottom, axis y line*=left,
    axis on top,   % axis lines and ticks are drawn over fills and curves (2026-11-25)
    tick align=outside, tick style={black, line width=0.4pt},
    major tick length=2.5pt, minor tick length=0pt,
    axis line style={black, line width=0.5pt},
    every tick label/.append style={font=\scriptsize},
    label style={font=\scriptsize},
    legend style={font=\scriptsize, draw=none, fill=none, inner sep=1pt, row sep=-1pt,
      /tikz/every even column/.append style={column sep=5pt}},
    legend cell align=left,
    legend pos=north east,
    cycle list name=sdbw,
    mark size=1.4pt,
    width=0.62\textwidth, height=0.36\textwidth},
  % a one-row legend centred above the data, inside the plot
  sdlegendtop/.style={legend style={at={(0.5,0.97)}, anchor=north}, legend columns=-1},
  % diagrams: axes with arrow heads, no grid
  sddiagram/.style={
    axis lines=middle, axis line style={-{Latex[length=1.6mm]}, line width=0.4pt},
    axis on top,
    xlabel style={anchor=west, font=\scriptsize},
    ylabel style={anchor=south, font=\scriptsize},
    every tick label/.append style={font=\scriptsize},
    tick style={black}, cycle list name=sdbw}}
\colorlet{sdfill}{black!12}

% Captions: SNmono's (bold label, small type, flush left), below the float
\RequirePackage{caption}
\captionsetup{font=small, labelfont=bf, labelsep=quad, justification=justified,
  singlelinecheck=false, skip=6pt}
\captionsetup[figure]{position=bottom, skip=8pt}
\captionsetup[table]{position=bottom, skip=8pt}

% \sdfloat{<picture or table>}{<caption, \label inside>}, in a figure or a table: the
% caption goes to the right of the content when at least \sd@sidemin is left beside it,
% and below it otherwise. A side caption is set flush left in the remaining width, its
% last line level with the bottom of the content. \sdfigure is the same command.
% Panels: \sdpanel{A}{<picture>} sets a bold panel letter above the top left corner of
% the picture; panels sit side by side with \hfill and in rows separated by \sdpanelrow.
% A figure made of panels takes one caption below it that names the panels: (A) ..., (B) ...
% In the text: Figure~\ref{fig:x}B.
% Panels keep the type size of every other figure (\scriptsize); only the plot gets smaller.
\newcommand\sdpanel[2]{\begin{tabular}[t]{@{}l@{}}{\small\bfseries #1}\\[-1pt]#2\end{tabular}}
\newcommand\sdpanelrow{\par\vskip8\p@}
% A panel figure too tall for one page is written as two figures on consecutive pages
% (the same convention as sdlongtable): the first carries no caption and ends with
% \sdcontinuedfoot ("Continued on the next page"); the second starts with
% \sdcontinuedhead ("Figure 2.1 (continued)") and ends with the full caption. Split only
% a figure that cannot fit on one page, and give both parts the placement [t].
\newcommand\sdcontinuedfoot{\par\vskip4\p@{\small\itshape\hfill Continued on the next page\par}}
\newcommand\sdcontinuedhead{{\small\textbf{\figurename\hspace{0.25em}%
  \ifnum\c@chapter>\z@\thechapter.\fi\the\numexpr\c@figure+1\relax}\enskip(continued)\par}\vskip6\p@}
\newdimen\sd@sidemin \sd@sidemin=4cm
\newdimen\sd@capwd
\newsavebox\sd@figbox
\newcommand\sdfloat[2]{%
  \sbox\sd@figbox{#1}%
  \sd@capwd=\dimexpr\linewidth-\wd\sd@figbox-\figgap\relax
  \ifdim\sd@capwd<\sd@sidemin
    \centering\usebox\sd@figbox\par\caption{#2}%
  \else
    \noindent\usebox\sd@figbox\hfill
    \begin{minipage}[b]{\sd@capwd}%
      \captionsetup{skip=0pt, justification=raggedright}\caption{#2}%
    \end{minipage}\par
  \fi}
\let\sdfigure\sdfloat

% ---------------------------------------------------------------------------
% 6. Tables and code
% ---------------------------------------------------------------------------
\RequirePackage{booktabs}
% sdtable (style B, chosen 2026-11-25): booktabs rules (\toprule, \midrule, \cmidrule,
% \bottomrule), no vertical rules, the table at its natural width; placed with \sdfloat,
% which centres it with the caption below, or puts the caption beside a narrow table.
% The baseline is the bottom row, so that a side caption ends level with the table.
\RequirePackage{tabularray}
\UseTblrLibrary{booktabs}
\NewTblrEnviron{sdtable}
\SetTblrOuter[sdtable]{baseline=b}
\SetTblrInner[sdtable]{cells={font=\small}, rowsep=1.5pt, colsep=6pt}
% sdlongtable: a table that may run over a page (not a float; it stays where it is
% written). Its caption is set after the table, across the full text width like any
% other caption; a page it leaves says "Continued on the next page", and a page it
% continues on repeats the header rows (rowhead=<n>) under "Table 1.4 (continued)".
% The booktabs rules go in the specification so that they are redrawn on every page:
%   \begin{sdlongtable}{<caption, \label inside>}{colspec={...}, rowhead=1,
%     hline{1,Z}={0.8pt}, hline{2}={0.4pt}} ... \end{sdlongtable}
% (tabularray's own caption would be confined to the width of the table, so the table
% itself carries none and the number it will get is computed in advance.)
\NewTblrTheme{sd}{%
  \DefTblrTemplate{conthead-text}{default}{(continued)}%
  \DefTblrTemplate{contfoot-text}{default}{\itshape Continued on the next page}%
  \SetTblrStyle{conthead}{font=\small}%
  \SetTblrStyle{contfoot}{font=\small}%
  \DefTblrTemplate{conthead}{default}{\textbf{Table\hspace{0.25em}\sd@longtag}\enskip\UseTblrTemplate{conthead-text}{default}}%
  \DefTblrTemplate{firsthead}{default}{}%
  \DefTblrTemplate{middlehead,lasthead}{default}{\UseTblrTemplate{conthead}{default}}%
  \DefTblrTemplate{firstfoot,middlefoot}{default}{\UseTblrTemplate{contfoot}{default}}%
  \DefTblrTemplate{lastfoot}{default}{}}
\NewTblrEnviron{sd@longtblr}
\SetTblrOuter[sd@longtblr]{long, theme=sd, entry=none, label=none, presep=\sd@blockskip, postsep=0pt}
\SetTblrInner[sd@longtblr]{cells={font=\small}, rowsep=1.5pt, colsep=6pt}
\def\sd@longtag{}
\NewDocumentEnvironment{sdlongtable}{m m +b}{%
  \edef\sd@longtag{\ifnum\c@chapter>\z@\thechapter.\fi\the\numexpr\c@table+1\relax}%
  \begin{sd@longtblr}{#2}#3\end{sd@longtblr}%
  \par\nobreak\vskip8\p@
  \captionof{table}{#1}%
  \par\addvspace{\sd@blockskip}}{}

% Side notes (history, implementation remarks): a light grey background without a frame,
% small type, bold title on its own line (style C, chosen 2026-11-25).
%   \begin{note}{Kahan and IEEE 754} ... \end{note}
\RequirePackage[most]{tcolorbox}
% It may break across pages: the grey band simply continues on the next page.
\newtcolorbox{note}[1]{enhanced, breakable, frame hidden, colback=black!7, arc=0pt, outer arc=0pt,
  fontupper=\small, fonttitle=\small\bfseries, coltitle=black,
  left=8pt, right=8pt, top=6pt, bottom=6pt, before skip=\sd@blockskip, after skip=\sd@blockskip,
  title={#1}, attach title to upper={\par\smallskip}}

% ---------------------------------------------------------------------------
% 7. Bibliography: IEEE numeric style, one list at the end of each chapter
% ---------------------------------------------------------------------------
\RequirePackage[backend=biber, style=ieee, refsection=chapter,
  doi=true, url=false, isbn=false, eprint=false]{biblatex}
\renewcommand*{\bibfont}{\small}
\newcommand\chapterreferences{\printbibliography[heading=subbibliography, title=References]}

% ---------------------------------------------------------------------------
% 7a. Running heads: SNmono's, but the chapter title without its number (2026-11-25);
% the section head on odd pages keeps its number.
% ---------------------------------------------------------------------------
\def\ps@headings{\let\@mkboth\markboth
   \let\@oddfoot\@empty\let\@evenfoot\@empty
   \def\@evenhead{\runheadsize\runheadstyle\rlap{\thepage}\hfil\leftmark}%
   \def\@oddhead{\runheadsize\runheadstyle\rightmark\hfil\llap{\thepage}}%
   \def\chaptermark##1{\markboth{##1}{##1}}%
   \def\sectionmark##1{\markright{{\ifnum\c@secnumdepth>\z@
      \thesection\seccounterend\hskip\betweenumberspace\fi ##1}}}}
\pagestyle{headings}

% ---------------------------------------------------------------------------
% 7b. Placement and spacing of blocks (2026-11-25). Figures, tables, algorithms and notes
% are set where they are written whenever they fit ("h" first). One vertical gap,
% \sd@blockskip, separates every such block from the text around it and from each other,
% whether it stays in place or floats to the top or bottom of a page.
% ---------------------------------------------------------------------------
\newskip\sd@blockskip \sd@blockskip=14\p@ \@plus 3\p@ \@minus 2\p@
\setlength\intextsep{\sd@blockskip}
\setlength\textfloatsep{\sd@blockskip}
\setlength\floatsep{\sd@blockskip}
\def\fps@figure{htbp}
\def\fps@table{htbp}
\renewcommand\topfraction{.85}
\renewcommand\bottomfraction{.6}
\renewcommand\textfraction{.1}

% ---------------------------------------------------------------------------
% 8a. Chapter heads: SNmono's, set 48pt lower on the page (2026-11-25). The text below a
% head still starts 167pt from the top, as in SNmono, so only the head moves.
% ---------------------------------------------------------------------------
\RequirePackage{etoolbox}
\pretocmd{\@makechapterhead}{\vspace*{48\p@}}{}{\ClassError{sdbook-eng}{patch failed}{}}
\pretocmd{\@makeschapterhead}{\vspace*{48\p@}}{}{\ClassError{sdbook-eng}{patch failed}{}}

% The chapter abstract is set off from the text: a blank line's worth of space follows it
% and the first paragraph of the text is not indented (2026-11-25).
\def\@abstract#1{\noindent\textbf{\abstractname}\enskip #1\par
  \addvspace{\baselineskip}\@afterindentfalse\@afterheading}

% ---------------------------------------------------------------------------
% 8b. Lists (2026-11-25): itemize keeps SNmono's labels (bullet, then en dash); enumerate
% numbers (i), (ii), ... and then (a), (b), ... Indentation is SNmono's own (labels set
% right-aligned in its \leftmargini, \leftmarginii); only the labels and spacing change.
% ---------------------------------------------------------------------------
\RequirePackage{enumitem}
\setlist{itemsep=1pt, parsep=0pt, topsep=3pt, partopsep=0pt}
\setlist[itemize,1]{label=\textbullet}
\setlist[itemize,2]{label=\textendash}
\setlist[enumerate,1]{label=(\roman*), ref=(\roman*)}
\setlist[enumerate,2]{label=(\alph*), ref=(\alph*)}

% ---------------------------------------------------------------------------
% 8c. Notation (2026-11-25): an unnumbered chapter right after the contents that lists
% the symbols of the volume once, briefly. \notationchapter starts it (SNmono's
% \Extrachap: unnumbered, with a contents entry and running heads); the symbols go in
% "symbols" lists, one per group, each under \symbolgroup{<name>}.
%   \notationchapter
%   \symbolgroup{Sets and numbers}
%   \begin{symbols} \item[$\mathbb R$] the real numbers ... \end{symbols}
% ---------------------------------------------------------------------------
\newcommand\notationchapter{\Extrachap{Notation}}
\newcommand\symbolgroup[1]{\par\addvspace{12\p@}\noindent{\bfseries #1}\par\nobreak\vskip2\p@}
\newlist{symbols}{description}{1}
\setlist[symbols]{font=\normalfont, labelwidth=7.5em, labelsep=1em, leftmargin=8.5em,
  align=left, itemsep=0pt, topsep=2pt}

% ---------------------------------------------------------------------------
% 8. Title page: SNmono's, without the publisher line
% ---------------------------------------------------------------------------
% \volume{1}{Calculus, Linear Algebra, and Differential Equations} sets the two lines
% under the title: "Volume 1", then the volume's own title in smaller type. The series
% title is set in italics (2026-11-25).
\newcommand\volume[2]{\gdef\sd@volno{#1}\gdef\sd@voltitle{#2}}
\def\sd@volno{}\def\sd@voltitle{}
\def\@maketitle{\newpage\null\vskip 2em
\begingroup
  \def\and{\unskip, }\parindent=\z@\pretolerance=10000\rightskip=\z@ \@plus 3cm
  {\LARGE\lineskip .5em \@author\par}%
  \vskip 2cm
  {\Huge\itshape \@title \par}%
  \vskip 1cm
  \ifx\sd@volno\@empty
    \if!\@subtitle!\else{\LARGE\ignorespaces\@subtitle \par}\vskip 1cm\fi
  \else
    {\LARGE Volume \sd@volno\par}\vskip 6\p@
    {\Large \sd@voltitle\par}\vskip 1cm
  \fi
  \ifx\@date\@empty\else{\large \@date}\par\vskip 1.5em\fi
  \vfill
\endgroup}

% ---------------------------------------------------------------------------
% 9. Small things
% ---------------------------------------------------------------------------
\setcounter{tocdepth}{1}            % chapters and sections in the contents
\newcommand\term[1]{\emph{#1}}      % a defined term at its first use
\newcommand\code[1]{\texttt{#1}}
\newcommand\fl{\mathop{\mathrm{fl}}\nolimits}   % the rounding map
\endinput
