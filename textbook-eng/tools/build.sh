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
  echo "$d: $pages, warnings $warn"
done
