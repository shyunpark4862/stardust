#!/bin/sh
# 모든 권을 빌드하고 PDF를 pdf/에 모은다. 권 사이의 참조(xr-hyper)가 서로의 aux를 읽으므로 두 번 돈다.
# 쓰기: tools/build.sh [권 번호 ...]   (번호를 주면 그 권만 두 번)
cd "$(dirname "$0")/.." || exit 1
vols=${*:-"1 2 3 4 5 6"}
for pass in 1 2; do
  for v in $vols; do
    latexmk -xelatex -interaction=nonstopmode "vol$v.tex" > "build/vol$v.build.log" 2>&1 \
      || { echo "vol$v: 실패 (build/vol$v.log)"; exit 1; }
    # 다른 권이 읽을 참조 사본: refsection 라벨을 뺀 \newlabel 줄
    mkdir -p build/xref
    for a in build/chapters/*.aux; do
      grep '^\\newlabel' "$a" | grep -v 'newlabel{refsection:' > "build/xref/$(basename "$a")"
    done
  done
done
# 읽을 PDF는 중간 파일과 떨어진 pdf/에 모은다(권 사이 링크가 같은 폴더의 volN.pdf를 가리킨다)
mkdir -p pdf
for v in $vols; do cp "build/vol$v.pdf" "pdf/vol$v.pdf"; done
for v in $vols; do
  pages=$(grep -o 'Output written on [^(]*([0-9]* pages' "build/vol$v.log" | grep -o '[0-9]* pages')
  warn=$(grep -cE 'undefined|multiply defined|Overfull' "build/vol$v.log")
  echo "vol$v: $pages, 경고 $warn"
done
