#!/bin/zsh
# M11: rebuild the end commit of each optimizing milestone and time them all in
# one sitting with the same script (bench/stage_compare.py), alternating builds
# over ROUNDS rounds. Results: bench/results/final/<tag>.json
#   M7  c74f390  all basic functions, straightforward wrappers (the baseline)
#   M8  de9ebc3  per-call cost
#   M9  d3d84e4  batch parallelism
#   M10 ba7930b  kron, cross, expm (final)
# Usage: zsh bench/stage_compare.sh [ROUNDS]   (from the repository root)
set -e
ROUNDS=${1:-2}
ROOT=$PWD
PY=$ROOT/.venv/bin/python
SITE=$($PY -c "import site; print(site.getsitepackages()[0])")
OUT=$ROOT/bench/results/final
mkdir -p $OUT build/stage
typeset -A COMMITS
COMMITS=(M7 c74f390 M8 de9ebc3 M9 d3d84e4 M10 ba7930b)
TAGS=(M7 M8 M9 M10)

for tag in $TAGS; do
  dir=build/stage/$tag
  if [[ ! -d $dir/pkg/stardust_linalg ]]; then
    rm -rf $dir && mkdir -p $dir
    git worktree add -f --detach $dir/src ${COMMITS[$tag]} > /dev/null
    $PY -m pip install --no-build-isolation --no-deps --target $dir/pkg $dir/src > $dir/build.log 2>&1
    git worktree remove --force $dir/src
  fi
done

rm -f $OUT/*.json(N)
# each build runs as its own process, its package first on the path (-S: no editable install)
run() { PYTHONPATH=$ROOT/build/stage/$1/pkg:$SITE $PY -S bench/stage_compare.py $1 $OUT/$2.json ${@:3}; }
for r in $(seq $ROUNDS); do
  for tag in $TAGS; do
    echo "round $r: $tag"
    run $tag $tag
  done
  echo "round $r: M10 one thread"
  run M10 M10-1thread --threads 1
done
echo "reference (numpy / scipy)"
run M10 ref --ref
