%% sdbook.cls — 《계산하는 수학》 교과서 클래스
%%
%% Springer Nature의 SNmono(v6.00)를 그대로 불러 그 디자인(장·절 머리, 쪽 머리,
%% 차례, 캡션)을 살리고, 그 위에 한글 조판만 얹는다.
%% 원칙: 흑백(그레이스케일), 강조는 볼드, 이탤릭은 쓰지 않는다. SNmono.cls는 고치지 않는다.
%% 지침: GUIDE.md
%% 엔진: XeLaTeX (xetexko)
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sdbook}[2026/11/25 v0.3 Korean edition of SNmono]

% ---------------------------------------------------------------------------
% 0. 바탕 클래스
% ---------------------------------------------------------------------------
% graybox: Springer의 회색 상자(svgraybox)
% sectrefs: 참고 문헌을 장 끝에 둘 수 있게
% nospthms: 정리 환경은 아래에서 amsthm으로 Springer 꼴을 그대로 다시 만든다
%           (cleveref가 정의·보조정리 같은 종류를 가리게 하려는 것)
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{SNmono}}
\ProcessOptions\relax
\LoadClass[graybox,sectrefs,nospthms]{SNmono}

% ---------------------------------------------------------------------------
% 1. 글꼴
% ---------------------------------------------------------------------------
% 수식: newtxmath (mathptmx의 현대판). fontspec보다 먼저 부르고 no-math로 지킨다.
\RequirePackage{amsmath,amssymb}
\let\Bbbk\relax
\RequirePackage[varvw]{newtxmath}
% 수식 안의 숫자·연산자 이름(sin, min, fl …)은 newtx의 Times 계열 글꼴로.
% (newtxtext 없이 부르면 Latin Modern으로 떨어진다)
\DeclareSymbolFont{operators}{OT1}{ntxtlf}{m}{n}
\SetSymbolFont{operators}{bold}{OT1}{ntxtlf}{b}{n}
\DeclareMathAlphabet{\mathrm}{OT1}{ntxtlf}{m}{n}
\SetMathAlphabet{\mathrm}{bold}{OT1}{ntxtlf}{b}{n}
\DeclareMathAlphabet{\mathbf}{OT1}{ntxtlf}{b}{n}
\DeclareMathAlphabet{\mathit}{OT1}{ntxtlf}{m}{it}
\RequirePackage[no-math]{fontspec}
\defaultfontfeatures{Ligatures=TeX}
% 영문: Times New Roman, 산세리프는 Helvetica, 고정폭은 Cascadia Code
\setmainfont{Times New Roman}
\setsansfont{Helvetica}[Scale=MatchLowercase]
\setmonofont{Cascadia Code}[Scale=MatchLowercase]
% 한글: 본문 KoPub World 바탕체(명조), 산세리프 Apple SD 산돌고딕 Neo.
% 한글에는 이탤릭이 없고 어울리지도 않으므로, 이탤릭을 청해도 곧은 글자로 둔다.
\RequirePackage{kotex}
\setmainhangulfont{KoPubWorldBatang_Pro}[
  UprightFont=* Light, BoldFont=* Bold,
  ItalicFont=* Light, BoldItalicFont=* Bold,
  Scale=0.93]
\setsanshangulfont{AppleSDGothicNeo-Regular}[
  BoldFont=AppleSDGothicNeo-Bold,
  ItalicFont=AppleSDGothicNeo-Regular,
  BoldItalicFont=AppleSDGothicNeo-Bold,
  Scale=0.93]
\setmonohangulfont{AppleSDGothicNeo-Regular}[Scale=0.88]
% 강조는 볼드로 통일한다
\DeclareRobustCommand\emph[1]{\textbf{#1}}

% ---------------------------------------------------------------------------
% 2. 판형과 글자 크기
% ---------------------------------------------------------------------------
% Springer 단행본 판형 155 x 235 mm, 본문 틀 117 x 191 mm는 그대로 둔다.
% 한글 본문은 줄 간격이 더 필요하므로 10/12pt를 10/15pt로 넓힌다(35줄).
\renewcommand\normalsize{%
   \@setfontsize\normalsize\@xpt{15}%
   \abovedisplayskip 9\p@ \@plus2\p@ \@minus3\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 6\p@ \@plus2\p@ \@minus2\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}
\renewcommand\small{%
   \@setfontsize\small{8.5}{12.5}%
   \abovedisplayskip 8\p@ \@plus2\p@ \@minus3\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \belowdisplayskip \abovedisplayskip
   \def\@listi{\leftmargin\leftmargini
               \parsep \z@ \@plus\p@ \@minus\p@
               \topsep 6\p@ \@plus2\p@ \@minus4\p@
               \itemsep\z@}}
\let\footnotesize\small
\normalsize
\setlength\footnotesep{9\p@}
\RequirePackage{geometry}
\geometry{paperwidth=155mm, paperheight=235mm,
  textwidth=117mm, textheight=520\p@,   % topskip 10pt + 34 x 15pt
  hmarginratio=1:1, top=22mm, headheight=12\p@, headsep=12\p@,
  footskip=30\p@}
\emergencystretch=1.5em
\tolerance=1000

% ---------------------------------------------------------------------------
% 3. 이름 (한글)
% ---------------------------------------------------------------------------
\def\abstractname{개요}
\def\ackname{감사의 글}
\def\andname{·}
\def\appendixname{부록}
\def\bibname{참고 문헌}
\def\refname{참고 문헌}
\def\chaptername{장}
\def\contentsname{차례}
\def\figurename{Figure}
\def\tablename{Table}
\def\indexname{찾아보기}
\def\listfigurename{그림 차례}
\def\listtablename{표 차례}
\def\partname{부}
\def\prefacename{머리말}
\def\forewordname{추천의 글}
\def\glossaryname{용어}
\def\acronymname{약어}
\def\proofname{Proof}
\def\seename{→}
\def\algorithmname{Algorithm}
% "I부", "1장" 꼴. 장 아래는 1. → 가. → (1) → (가) 순서.
\newcommand\sd@chaplabel[1]{#1장}
\newcommand\sd@partlabel[1]{#1부}
\def\thechapterend{장}
\newcommand\sd@gana[1]{\ifcase#1\or 가\or 나\or 다\or 라\or 마\or 바\or 사\or 아\or
  자\or 차\or 카\or 타\or 파\or 하\else\@ctrerr\fi}
\def\gana#1{\expandafter\sd@gana\csname c@#1\endcsname}
\setcounter{secnumdepth}{4}
\AtEndOfClass{%
  \renewcommand\thesection{\arabic{section}}%
  \renewcommand\thesubsection{\gana{subsection}}%
  \renewcommand\thesubsubsection{\arabic{subsubsection}}%
  \renewcommand\theparagraph{\gana{paragraph}}%
  % 참조할 때는 장과 절을 앞에 붙인다: "1장 2절", "1장 2절 나"
  \renewcommand\p@section{\thechapter장\ }%
  \renewcommand\p@subsection{\thechapter장\ \thesection절\ }%
  \renewcommand\p@subsubsection{\thechapter장\ \thesection절\ \thesubsection\ }}
% 제사(motto)는 이탤릭 없이
\renewcommand\mottostyle{\normalfont\raggedright}
% 쪽 머리의 절 번호도 "2." 꼴로 (\seccounterend는 제목 끝에도 붙으므로 비워 두고 직접 단다)
\RequirePackage{etoolbox}
\patchcmd{\ps@headings}{\thesection\seccounterend}{\thesection.}{}{\ClassWarning{sdbook}{running head patch failed}}
\def\sectionmark#1{\markright{{\ifnum\c@secnumdepth>\z@\thesection.\hskip\betweenumberspace\fi #1}}}
% 번호의 꼴: 1. / 가. / (1) / (가)
\def\@seccntformat#1{\csname sd@fmt@#1\endcsname{\csname the#1\endcsname}\hskip 0.5em\relax}
\def\sd@fmt@section#1{#1.}
\def\sd@fmt@subsection#1{#1.}
\def\sd@fmt@subsubsection#1{(#1)}
\def\sd@fmt@paragraph#1{(#1)}
% 목록: 1) → 가)
\renewcommand\theenumi{\arabic{enumi}}
\renewcommand\labelenumi{\theenumi)}
\renewcommand\theenumii{\gana{enumii}}
\renewcommand\labelenumii{\theenumii)}
\renewcommand\p@enumii{\theenumi)\,}

% ---------------------------------------------------------------------------
% 4. 부와 장의 머리 — SNmono의 꼴 그대로, 번호 표기만 한국어 어순으로
% ---------------------------------------------------------------------------
\def\@part[#1]#2{\ifnum \c@secnumdepth >-2\relax
        \refstepcounter{part}%
        \addcontentsline{toc}{part}{\sd@partlabel{\thepart}\hspace{\betweenumberspace}#1}%
      \else
        \addcontentsline{toc}{part}{#1}\fi
   \markboth{}{}%
   {\raggedleft
    \hyphenpenalty \@M
    \interlinepenalty\@M
    \ifnum \c@secnumdepth >-2\relax
      \normalfont\partnumsize\partnumstyle
      \sd@partlabel{\thepart}\par
      \vskip 2\p@ \fi
    \partsize\partstyle #2\par}\@endpart}

\def\@makechapterhead#1{{\parindent\z@\raggedright\normalfont
  \hyphenpenalty \@M
  \interlinepenalty\@M
  \if@chapnum
     \chapnumsize\chapnumstyle
     \sd@chaplabel{\thechapter}\par
     \vskip 2\p@
  \fi
  \chapsize\chapstyle
  \ignorespaces#1\par\nobreak
  \processchapsubtit
  \processchapauthor
  \processmotto
  \ifdim\pagetotal>167\p@
     \vskip 11\p@
  \else
     \@tempdima=167\p@\advance\@tempdima by-\pagetotal
     \vskip\@tempdima
  \fi}}

% 속표지: SNmono 꼴에서 날짜와 출판사 줄만 바꾼다
\date{}
\def\publisher#1{\gdef\@publisher{#1}}
\def\@publisher{}
\def\@maketitle{\newpage
 \null
 \vskip 2em
\begingroup
  \def\and{\unskip, }%
  \parindent=\z@
  \pretolerance=10000
  \rightskip=\z@ \@plus 3cm
  {\LARGE\lineskip .5em \@author\par}%
  \vskip 2cm
  {\Huge \@title \par}%
  \vskip 1cm
  \if!\@subtitle!\else
   {\LARGE\ignorespaces\@subtitle \par}%
   \vskip 1cm
  \fi
  \ifx\@date\@empty\else {\large \@date}\par\vskip 1.5em\fi
 \vfill
 {\Large \@publisher\par}%
\endgroup}

% 장 머리글: Springer의 \abstract — 굵은 "개요" 뒤에 바로 본문
\def\@abstract#1{\noindent\textbf{\abstractname}\hskip 0.6em\relax #1\par}

% ---------------------------------------------------------------------------
% 5. 참조 (cleveref는 정리 환경을 만들기 전에 불러야 한다)
% ---------------------------------------------------------------------------
\RequirePackage[hidelinks,bookmarksnumbered,pdfencoding=auto]{hyperref}
\RequirePackage[noabbrev]{cleveref}
\crefformat{chapter}{#2#1장#3}
\Crefformat{chapter}{#2#1장#3}
\crefformat{part}{#2#1부#3}
\Crefformat{part}{#2#1부#3}
\crefformat{section}{#2#1절#3}
\Crefformat{section}{#2#1절#3}
\crefformat{subsection}{#2#1#3}
\Crefformat{subsection}{#2#1#3}
\crefformat{subsubsection}{#2#1#3}
\Crefformat{subsubsection}{#2#1#3}
\crefname{theorem}{Theorem}{Theorem}
\crefname{lemma}{Lemma}{Lemma}
\crefname{proposition}{Proposition}{Proposition}
\crefname{corollary}{Corollary}{Corollary}
\crefname{definition}{Definition}{Definition}
\crefname{assumption}{Assumption}{Assumption}
\crefname{example}{Example}{Example}
\crefname{remark}{Remark}{Remark}
\crefformat{equation}{#2(#1)#3}
\Crefformat{equation}{#2(#1)#3}
\crefname{figure}{Figure}{Figure}
\crefname{table}{Table}{Table}
\crefname{algorithm}{Algorithm}{Algorithm}
\newcommand\crefpairconjunction{, }
\newcommand\crefmiddleconjunction{, }
\newcommand\creflastconjunction{, }

% ---------------------------------------------------------------------------
% 6. 정리 환경 — Springer 꼴(굵은 이름과 번호, 곧은 본문)을 amsthm으로
% ---------------------------------------------------------------------------
\newtheoremstyle{sp}{7\p@ \@plus2\p@ \@minus4\p@}{7\p@ \@plus2\p@ \@minus4\p@}%
  {\upshape}{0pt}{\bfseries}{}{0.5em}%
  {\thmname{#1}\thmnumber{ #2}\thmnote{ (#3)}}
\RequirePackage{aliascnt}
% 정리 계열은 장 단위 번호 하나를 함께 쓴다(Springer의 envcountsame + envcountchap).
\newcommand\sd@newthm[2]{%
  \newaliascnt{#1}{theorem}\newtheorem{#1}[#1]{#2}\aliascntresetthe{#1}}
\theoremstyle{sp}
\newtheorem{theorem}{Theorem}[chapter]
\sd@newthm{lemma}{Lemma}
\sd@newthm{proposition}{Proposition}
\sd@newthm{corollary}{Corollary}
\sd@newthm{definition}{Definition}
\sd@newthm{assumption}{Assumption}
\sd@newthm{example}{Example}
\sd@newthm{remark}{Remark}
\newtheorem*{remark*}{Remark}

% 증명: 굵은 "Proof." 뒤에 본문, 끝에 □ (이탤릭은 쓰지 않는다)
\def\qedsymbol{\hbox{\rlap{$\sqcap$}$\sqcup$}}
\renewenvironment{proof}[1][\proofname]{\par
  \pushQED{\qed}%
  \normalfont \topsep6\p@\@plus6\p@\relax
  \trivlist
  \item[\hskip\labelsep\bfseries #1\@addpunct{.}]\ignorespaces
}{\popQED\endtrivlist\@endpefalse}

% ---------------------------------------------------------------------------
% 7. 상자 — 위아래 가는 선과 굵은 머리말만 둔 깔끔한 상자 (음영 없음)
% ---------------------------------------------------------------------------
\RequirePackage{needspace}
\newcommand\sd@boxhead[2]{\textbf{#1}\ifx\relax#2\relax\else\enspace{\mdseries#2}\fi}
% 선 바로 뒤에는 행간이 붙지 않도록 \hrule과 \kern으로 간격을 직접 정한다
\newenvironment{sdbox}[2]{\par\needspace{5\baselineskip}\addvspace{12\p@}%
  \hrule height 0.6\p@\kern 5\p@
  \noindent{\small\sd@boxhead{#1}{#2}}\par\nobreak\vskip 3\p@
  \small\@afterindentfalse\@afterheading\ignorespaces}%
 {\par\kern 4\p@\hrule height 0.4\p@\addvspace{12\p@}}
% Aside: 역사, 인물, 사고, 여담
\newenvironment{aside}[1][]{\begin{sdbox}{Aside}{#1}}{\end{sdbox}}
% In stardust: 이론이 라이브러리의 어디에 어떻게 들어가는가
\newenvironment{instardust}[1][]{\begin{sdbox}{In stardust}{#1}}{\end{sdbox}}

% ---------------------------------------------------------------------------
% 8. 유사코드
% ---------------------------------------------------------------------------
\RequirePackage{float}
% 위 선 · 본문 · 아래 선, 캡션은 그림·표처럼 아래에.
\newcommand\fs@sdruled{\def\@fs@cfont{\bfseries}\let\@fs@capt\sd@algcapt
  \def\@fs@pre{\hrule height.6pt depth0pt \kern4pt}%
  \def\@fs@mid{\kern3pt\hrule height.6pt\kern6pt}%
  \def\@fs@post{}%
  \let\@fs@iftopcapt\iffalse}
\newcommand\sd@algcapt[2]{{\small\textbf{#1}\quad #2\par}}
\floatstyle{sdruled}
\newfloat{algorithm}{tbp}{loa}[chapter]
\floatname{algorithm}{\algorithmname}
\renewcommand\thealgorithm{\thechapter.\arabic{algorithm}}
\newcommand\listofalgorithms{\listof{algorithm}{알고리즘 차례}}
\floatstyle{plain}
% 주석은 "//"로 시작하고, 회색·이탤릭 없이 검은 곧은 글자로
\RequirePackage[noEnd=false,indLines=true,italicComments=false,commentColor=black,
  beginComment={//~},beginLComment={//~},endLComment={}]{algpseudocodex}
\algrenewcommand\algorithmicrequire{\textbf{Input}}
\algrenewcommand\algorithmicensure{\textbf{Output}}
\algrenewcommand\alglinenumber[1]{{\scriptsize #1}}
\tikzset{algpxIndentLine/.style={draw=black, line width=0.3pt}}
\AtBeginEnvironment{algorithmic}{\small}

% ---------------------------------------------------------------------------
% 9. 그림 — 샘플(수리통계 강의 노트) 풍의 흑백 그래프
% ---------------------------------------------------------------------------
\RequirePackage{graphicx}
\RequirePackage{tikz}
\RequirePackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,calc,positioning,decorations.pathreplacing,patterns}
% 선 모양으로 계열을 가른다: 실선, 파선, 점선, 일점쇄선, 옅은 실선
\pgfplotscreateplotcyclelist{sdbw}{%
  {black, solid, line width=0.7pt},
  {black, dashed, line width=0.7pt},
  {black, densely dotted, line width=0.8pt},
  {black, dashdotted, line width=0.7pt},
  {black!45, solid, line width=0.7pt},
  {black!45, dashed, line width=0.7pt}}
\pgfplotsset{
  % 함수 그래프: 틀 두른 축, 옅은 격자, 안쪽 눈금
  sdplot/.style={
    axis lines=box, grid=major,
    grid style={black!15, line width=0.3pt},
    tick align=inside, tick style={black, line width=0.4pt},
    axis line style={black, line width=0.5pt},
    every tick label/.append style={font=\scriptsize},
    label style={font=\scriptsize},
    legend style={font=\scriptsize, draw=black, line width=0.4pt,
      fill=white, inner sep=2pt, row sep=-1pt},
    legend cell align=left,
    cycle list name=sdbw,
    mark size=1.6pt,
    width=0.85\textwidth, height=0.55\textwidth},
  % 도해: 축만, 화살표, 격자 없음
  sddiagram/.style={
    axis lines=middle, axis line style={-{Latex[length=1.6mm]}, line width=0.4pt},
    xlabel style={anchor=west, font=\scriptsize},
    ylabel style={anchor=south, font=\scriptsize},
    every tick label/.append style={font=\scriptsize},
    tick style={black}, cycle list name=sdbw}}
% 회색 칠: 넓이·영역
\colorlet{sdfill}{black!12}

% 캡션: Springer 꼴(굵은 "Figure 1.1", 작은 글자, 왼쪽 맞춤). 그림·표·알고리즘 모두 아래.
\RequirePackage{caption}
\captionsetup{font=small, labelfont=bf, labelsep=quad, justification=justified,
  singlelinecheck=false, skip=6pt}
\captionsetup[figure]{position=bottom, skip=8pt}
\captionsetup[table]{position=bottom, skip=8pt}

% ---------------------------------------------------------------------------
% 10. 목록, 표, 코드
% ---------------------------------------------------------------------------
\RequirePackage{booktabs}
% 표: 본문 폭을 꽉 채우고, 각 칸의 폭은 내용(자연 폭)에 비례한다.
% (tabularray에서 모든 칸의 계수를 음수로 두면 남는 폭을 자연 폭에 비례해 나눈다)
\RequirePackage{tabularray}
\UseTblrLibrary{booktabs}
\NewTblrEnviron{sdtable}
\SetTblrInner[sdtable]{width=\linewidth, columns={co=-1}, rowsep=1.5pt, colsep=4pt,
  cells={font=\small}}
\RequirePackage{listings}
\lstset{basicstyle=\ttfamily\small, columns=fullflexible,
  keepspaces=true, keywordstyle=\bfseries, commentstyle=\itshape,
  aboveskip=6pt, belowskip=6pt, showstringspaces=false,
  extendedchars=true, xleftmargin=\parindent}

% 차례: 얕은 들여쓰기. 장은 "1장", 절은 "1.", 그 아래는 "가."
\setcounter{tocdepth}{2}
\tocchpnum=2.3em
\tocsecnum=1.5em
\tocsubsecnum=1.5em
\calctocindent
\newdimen\sd@tocsecindent \sd@tocsecindent=1em
\newdimen\sd@tocsubindent \sd@tocsubindent=2.5em
\let\sd@tocnum\@firstofone
\def\numberline#1{\hb@xt@\@tempdima{\sd@tocnum{#1}\hfil}}
\RequirePackage{etoolbox}
\pretocmd{\l@chapter}{\let\sd@tocnum\@firstofone}{}{}
\pretocmd{\l@part}{\let\sd@tocnum\@firstofone}{}{}
\def\l@section{\def\sd@tocnum##1{##1.}\@dottedtocline{1}{\sd@tocsecindent}{\tocsecnum}}
\def\l@subsection{\def\sd@tocnum##1{##1.}\@dottedtocline{2}{\sd@tocsubindent}{\tocsubsecnum}}

% 참고문헌: IEEE 방식(본문 [1], 장 끝에 인용 순서대로). 장마다 끝에 (biblatex + biber).
% 원고에서는 \cite{…}, 장 끝에 \chapterreferences.
\RequirePackage[backend=biber, style=ieee, refsection=chapter,
  doi=true, url=false, isbn=false, eprint=false]{biblatex}
\renewcommand*{\bibfont}{\small}
\DefineBibliographyStrings{english}{andothers={et al\adddot}}
\newcommand\chapterreferences{\printbibliography[heading=subbibliography, title=References]}

% ---------------------------------------------------------------------------
% 11. 기타
% ---------------------------------------------------------------------------
% 용어 첫 등장: 굵은 한글(원어)
\newcommand\term[2]{\textbf{#1}\,(#2)}
\newcommand\code[1]{\texttt{#1}}
% 반올림 사상 fl (합자 없이)
\newcommand\fl{\mathop{\mathrm{f\kern0.03em l}}\nolimits}
\endinput
