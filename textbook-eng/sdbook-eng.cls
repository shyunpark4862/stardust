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
% \chapternotes opens the unnumbered Notes section that ends a chapter; it and the
% reference list set the odd-page running head, which would otherwise keep the last
% numbered section.
\newcommand\chapternotes{\section*{Notes}\markright{Notes}}
\newcommand\chapterreferences{\printbibliography[heading=subbibliography, title=References]\markright{References}}

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

% The chapter abstract is set in \small and set off from the text: a blank line's worth of space follows it
% and the first paragraph of the text is not indented (2026-11-25).
\def\@abstract#1{{\small\noindent\textbf{\abstractname}\enskip #1\par}%
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
% under the title: "Volume I" (roman numeral), then the volume's own title in smaller type. The series
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
    {\LARGE Volume \uppercase\expandafter{\romannumeral\sd@volno}\par}\vskip 6\p@
    {\Large \sd@voltitle\par}\vskip 1cm
  \fi
  \ifx\@date\@empty\else{\large \@date}\par\vskip 1.5em\fi
  \vfill
\endgroup}
% Background (2026-11-26, chosen by the user from a set of alternates; the lettering above is
% unchanged): in the lower outer corner, a lattice of dots 14pt apart fading in three flat
% steps and running off the edges; over it 150 points drawn (seed 20261126) from the normal
% distribution with unit variances and correlation 0.7, 34pt to the unit, as small open
% circles, with the concentration ellipses at Mahalanobis distance 1 and 2 (semi-axes
% sqrt(1.7) and sqrt(0.3)) and the two principal axes. Greyscale only, thin lines.
% It is drawn in page coordinates (origin at the lower left corner of the page, unit 1pt,
% anchored to the page, not to the text block) on the title page only, and only when that
% page is a right-hand page. A volume can put its own motif in its preamble with
%   \renewcommand\sdcoverbackground{<TikZ code in page coordinates>}
% (\sd@coverW and \sd@coverH are the page width and height), or none with
% \renewcommand\sdcoverbackground{}.
\newcommand\sd@coverlattice[2]{% dots through (#1,#2), inside an ellipse about the corner
  \foreach \sd@i in {-24,...,23}{\foreach \sd@j in {-7,...,30}{%
    \pgfmathsetmacro\sd@x{#1+14*\sd@i}\pgfmathsetmacro\sd@y{#2+14*\sd@j}%
    \pgfmathsetmacro\sd@r{sqrt(((\sd@coverW-\sd@x)/410)^2+(\sd@y/400)^2)}%
    \ifdim\sd@r\p@<\p@
      \pgfmathsetmacro\sd@t{\sd@r<0.62 ? 30 : (\sd@r<0.82 ? 21 : 13)}%
      \fill[black!\sd@t] (\sd@x,\sd@y) circle (0.8);
    \fi}}}
\newcommand\sd@coverpoints{%
\foreach \sd@p in {{(425.5,149.0)},{(427.4,140.6)},{(421.5,178.8)},{(414.9,125.6)},{(412.6,149.0)},{(444.0,131.8)},{(498.1,185.7)},{(408.4,150.5)},{(406.6,155.3)},{(482.8,205.9)},{(419.8,102.3)},{(406.5,161.2)},{(400.7,148.4)},{(431.7,177.6)},{(471.1,184.7)},{(441.7,171.3)},{(462.7,165.6)},{(479.1,158.3)},{(411.6,123.1)},{(350.4,49.0)},{(441.2,133.8)},{(460.7,139.6)},{(452.4,166.9)},{(428.1,144.3)},{(455.0,216.0)},{(476.1,201.0)},{(425.8,180.2)},{(441.4,148.3)},{(410.5,110.9)},{(398.9,148.1)},{(432.3,141.5)},{(428.3,164.6)},{(439.9,140.1)},{(387.3,102.7)},{(487.0,256.6)},{(445.2,194.6)},{(462.6,161.5)},{(453.3,169.7)},{(395.4,160.7)},{(399.9,121.8)},{(431.5,163.4)},{(375.4,144.4)},{(389.1,96.7)},{(371.1,70.6)},{(415.6,139.5)},{(419.0,151.7)},{(419.8,147.3)},{(490.1,223.6)},{(454.6,125.1)},{(480.5,194.5)},{(450.8,141.6)},{(439.5,157.1)},{(491.3,210.5)},{(453.8,199.3)},{(392.3,122.1)},{(506.9,228.1)},{(431.9,171.7)},{(465.2,180.0)},{(392.1,119.5)},{(522.4,223.0)},{(491.6,185.7)},{(426.3,140.8)},{(435.7,136.7)},{(413.4,159.0)},{(450.4,183.5)},{(450.2,140.0)},{(412.0,138.1)},{(433.9,153.9)},{(426.5,147.5)},{(397.9,101.9)},{(390.9,83.6)},{(400.3,122.7)},{(432.1,134.4)},{(445.5,180.8)},{(476.8,178.8)},{(460.6,164.9)},{(405.5,152.0)},{(448.8,192.7)},{(470.9,159.7)},{(379.1,84.4)},{(447.9,153.3)},{(481.8,142.0)},{(372.0,128.3)},{(444.3,141.4)},{(481.5,174.2)},{(448.6,150.6)},{(414.2,163.3)},{(443.9,173.5)},{(465.8,189.0)},{(535.8,244.7)},{(472.9,181.5)},{(463.4,145.1)},{(449.9,161.2)},{(484.9,205.0)},{(412.4,118.7)},{(461.2,203.0)},{(389.5,87.8)},{(449.9,157.0)},{(385.5,107.9)},{(433.4,150.3)},{(506.7,199.3)},{(386.0,102.3)},{(461.9,184.2)},{(471.7,157.1)},{(378.4,117.7)},{(384.5,92.3)},{(490.8,188.9)},{(394.7,113.4)},{(382.2,132.8)},{(419.1,123.6)},{(484.4,198.8)},{(444.9,180.8)},{(448.6,139.8)},{(418.0,136.8)},{(447.9,108.4)},{(405.0,138.8)},{(446.0,200.9)},{(431.4,172.8)},{(456.3,167.7)},{(431.2,146.1)},{(379.0,142.7)},{(393.6,165.9)},{(424.4,176.6)},{(450.1,139.6)},{(461.4,145.9)},{(440.3,154.6)},{(433.2,124.1)},{(390.6,88.1)},{(446.3,136.2)},{(467.6,215.6)},{(379.8,78.0)},{(469.7,193.6)},{(449.0,137.4)},{(473.3,192.3)},{(424.6,128.9)},{(346.7,85.2)},{(479.8,133.4)},{(414.3,141.5)},{(474.2,165.3)},{(384.7,134.2)},{(417.6,133.4)},{(442.7,165.5)},{(420.0,103.6)},{(397.0,109.2)},{(415.4,141.3)},{(413.2,155.5)},{(328.3,114.6)},{(391.5,90.2)},{(410.3,157.8)},{(476.4,205.6)}}{\draw \sd@p circle (1.5);}}
\newcommand\sd@coverbivariate{%
  \def\sd@cx{434}\def\sd@cy{154}\def\sd@sc{34}%
  \sd@coverlattice{\sd@cx}{\sd@cy}%
  \begin{scope}[line width=0.3pt]
  \begin{scope}[rotate around={45:(\sd@cx,\sd@cy)}]
    \draw[black!22] (\sd@cx-2.6*\sd@sc*1.3038,\sd@cy) -- (\sd@cx+2.6*\sd@sc*1.3038,\sd@cy);
    \draw[black!22] (\sd@cx,\sd@cy-2.6*\sd@sc*0.5477) -- (\sd@cx,\sd@cy+2.6*\sd@sc*0.5477);
    \draw[black!38, line width=0.4pt] (\sd@cx,\sd@cy) ellipse ({\sd@sc*1.3038} and {\sd@sc*0.5477});
    \draw[black!38, line width=0.4pt] (\sd@cx,\sd@cy) ellipse ({2*\sd@sc*1.3038} and {2*\sd@sc*0.5477});
  \end{scope}
  \begin{scope}[black!45]\sd@coverpoints\end{scope}
  \end{scope}}
\newcommand\sdcoverbackground{\sd@coverbivariate}
\newcommand\sd@drawcover{%
  \if@twoside\ifodd\c@page\else\let\sdcoverbackground\relax\fi\fi
  \ifx\sdcoverbackground\relax\else
    \pgfmathsetmacro\sd@coverW{\paperwidth/1pt}\pgfmathsetmacro\sd@coverH{\paperheight/1pt}%
    \begin{tikzpicture}[remember picture, overlay, x=1pt, y=1pt]
    \begin{scope}[shift={(current page.south west)}]
    \clip (0,0) rectangle (\sd@coverW,\sd@coverH);
    \sdcoverbackground
    \end{scope}\end{tikzpicture}%
  \fi}
\let\sd@maketitle@lettering\@maketitle
\def\@maketitle{\AddToHookNext{shipout/background}{\sd@drawcover}\sd@maketitle@lettering}

% ---------------------------------------------------------------------------
% 9. Small things
% ---------------------------------------------------------------------------
\setcounter{tocdepth}{1}            % chapters and sections in the contents
\newcommand\term[1]{\emph{#1}}      % a defined term at its first use
\newcommand\code[1]{\texttt{#1}}
\newcommand\fl{\mathop{\mathrm{fl}}\nolimits}   % the rounding map
% Contents: room for section numbers such as 13.11 (SNmono allows only 88.8).
\tocsecnum=30\p@
\tocsectotal=\tocchpnum \advance\tocsectotal by\tocsecnum
\tocsubsectotal=\tocsectotal \advance\tocsubsectotal by\tocsubsecnum
\endinput
