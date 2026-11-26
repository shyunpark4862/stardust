#!/bin/sh
# Build the given documents with latexmk and pdfLaTeX. Intermediate files go to build/; the
# finished PDFs are collected in pdf/, apart from everything else.
# Usage: tools/build.sh [all | name ...]
#
# Volumes I and II refer to each other (\externalvolume in their preambles): each reads the
# labels of the other from build/vol1.aux or build/vol2.aux. "all", and no argument at all,
# therefore builds vol1 and vol2 in turn, again and again, until neither aux file changes any
# more (at most four rounds; from an empty build/ three or four are needed), then every other
# document in this folder (vol3, ...), which reads both aux files. A single name builds only
# that document, against whatever aux files build/ already holds: after a change that moves
# numbers in Volume I or II, run "all" so that both volumes print the new numbers.
cd "$(dirname "$0")/.." || exit 1
mkdir -p build pdf

# build_doc <name>: build one document and set $line to "<name>: N pages, warnings W".
# latexmk notices a change of the other volume's aux file only once that file has been read;
# when the previous run still had undefined references (for instance because the other aux
# file did not exist yet), the build is therefore forced (-g).
build_doc() {
  d=$1; force=
  [ -f "build/$d.log" ] && grep -q 'There were undefined references' "build/$d.log" && force=-g
  latexmk -pdf $force "$d.tex" > "build/$d.build.log" 2>&1 || { echo "$d: failed (build/$d.log)"; exit 1; }
  cp "build/$d.pdf" "pdf/$d.pdf"
  pages=$(grep -o 'Output written on [^(]*([0-9]* pages' "build/$d.log" | grep -o '[0-9]* pages')
  warn=$(grep -cE 'undefined|multiply defined|Overfull' "build/$d.log")
  # biber: every WARN and ERROR line, but not the summary lines ("INFO - WARNINGS: n") and not
  # the warning of a volume that cites nothing yet (a new volume's front matter alone)
  bibwarn=0; [ -f "build/$d.blg" ] && bibwarn=$(grep -E 'WARN|ERROR' "build/$d.blg" |
    grep -vE '> INFO - |does not contain any citations' | wc -l | tr -d ' ')
  warn=$((warn + bibwarn))
  line="$d: $pages, warnings $warn"
}

# the checksum of the aux files through which Volumes I and II see each other
auxsum() { cat build/vol1.aux build/vol2.aux 2>/dev/null | cksum; }

if [ $# -eq 0 ] || [ "$*" = all ]; then
  round=0
  while :; do
    before=$(auxsum)
    build_doc vol1; line1=$line
    build_doc vol2; line2=$line
    round=$((round + 1))
    [ "$(auxsum)" = "$before" ] && break
    if [ $round -ge 4 ]; then echo "vol1, vol2: aux files still change after $round rounds"; break; fi
  done
  echo "$line1"; echo "$line2"
  for d in $(ls *.tex | sed 's/\.tex$//'); do
    case $d in vol1|vol2) continue ;; esac
    build_doc "$d"; echo "$line"
  done
else
  for d in "$@"; do build_doc "$d"; echo "$line"; done
fi
