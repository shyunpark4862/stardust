#!/bin/sh
# Build the given documents (default: demo) with latexmk and pdfLaTeX; PDFs are copied to pdf/.
# Usage: tools/build.sh [name ...]
cd "$(dirname "$0")/.." || exit 1
docs=${*:-demo}
mkdir -p build pdf
for d in $docs; do
  latexmk -pdf "$d.tex" > "build/$d.build.log" 2>&1 || { echo "$d: failed (build/$d.log)"; exit 1; }
  cp "build/$d.pdf" "pdf/$d.pdf"
  pages=$(grep -o 'Output written on [^(]*([0-9]* pages' "build/$d.log" | grep -o '[0-9]* pages')
  warn=$(grep -cE 'undefined|multiply defined|Overfull' "build/$d.log")
  echo "$d: $pages, warnings $warn"
done
