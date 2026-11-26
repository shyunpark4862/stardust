#!/bin/sh
# Build the given documents (default: every .tex in this folder) with latexmk and pdfLaTeX.
# Intermediate files go to build/; the finished PDFs are collected in pdf/, apart from everything else.
# Usage: tools/build.sh [name ...]
cd "$(dirname "$0")/.." || exit 1
docs=${*:-$(ls *.tex 2>/dev/null | sed 's/\.tex$//')}
mkdir -p build pdf
for d in $docs; do
  latexmk -pdf "$d.tex" > "build/$d.build.log" 2>&1 || { echo "$d: failed (build/$d.log)"; exit 1; }
  cp "build/$d.pdf" "pdf/$d.pdf"
  pages=$(grep -o 'Output written on [^(]*([0-9]* pages' "build/$d.log" | grep -o '[0-9]* pages')
  warn=$(grep -cE 'undefined|multiply defined|Overfull' "build/$d.log")
  # biber: every WARN and ERROR line, but not the summary lines ("INFO - WARNINGS: n") and not
  # the warning of a volume that cites nothing yet (a new volume's front matter alone)
  bibwarn=0; [ -f "build/$d.blg" ] && bibwarn=$(grep -E 'WARN|ERROR' "build/$d.blg" |
    grep -vE '> INFO - |does not contain any citations' | wc -l | tr -d ' ')
  warn=$((warn + bibwarn))
  echo "$d: $pages, warnings $warn"
done
