#!/bin/zsh
# M12: rebuild the end commit of M9, M10 and M11 and time them all in one
# sitting with the same script (bench/final_compare.py), alternating with
# pandas and Polars over ROUNDS rounds. Results: bench/results/<dir>/<col>.json
#   M9   64ca0c1  every feature, nothing optimized (the baseline)
#   M10  d6aadd5  single-core optimization
#   M11  e2195f7  multithreading
#   now           the working tree (editable install): after M12's fixes
# Usage: zsh bench/final_compare.sh [ROUNDS] [same|fresh] [COLUMNS...]
#   same : one input per case, called over and over (M12's first run: final/)
#   fresh: a different input each call (final-fresh/); see _harness.per_call_ns_cycle
#   COLUMNS: any of M9 M10 M11-1 M11 now-1 now pandas polars-1 polars (default: all)
set -e
ROUNDS=${1:-2}
MODE=${2:-same}
shift $(( $# < 2 ? $# : 2 ))
if (( $# )); then COLS=($@); else COLS=(M9 M10 M11-1 M11 now-1 now pandas polars-1 polars); fi
ROOT=$PWD
PY=$ROOT/.venv/bin/python
SITE=$($PY -c "import site; print(site.getsitepackages()[0])")
if [[ $MODE == fresh ]]; then OUT=$ROOT/bench/results/final-fresh; FLAG=--fresh; else OUT=$ROOT/bench/results/final; FLAG=; fi
DATA=$ROOT/build/stage/data
mkdir -p $OUT build/stage $DATA
typeset -A COMMITS
COMMITS=(M9 64ca0c1 M10 d6aadd5 M11 e2195f7)

for tag in M9 M10 M11; do
  dir=build/stage/$tag
  if [[ ! -d $dir/pkg/stardust_pandas ]]; then
    rm -rf $dir && mkdir -p $dir
    git worktree add -f --detach $dir/src ${COMMITS[$tag]} > /dev/null
    $PY -m pip install --no-build-isolation --no-deps --target $dir/pkg $dir/src > $dir/build.log 2>&1
    git worktree remove --force $dir/src
  fi
done

for c in $COLS; do rm -f $OUT/$c.json; done
# each build runs as its own process, its package first on the path (-S: no editable install)
sd() { PYTHONPATH=$ROOT/build/stage/$1/pkg:$SITE $PY -S bench/final_compare.py sdpd $OUT/$2.json $DATA ${@:3} ${=FLAG}; }
run() {
  case $1 in
    M9|M10|M11) sd $1 $1 ;;
    M11-1)      sd M11 M11-1 --threads 1 ;;
    now)        $PY bench/final_compare.py sdpd $OUT/now.json $DATA ${=FLAG} ;;
    now-1)      $PY bench/final_compare.py sdpd $OUT/now-1.json $DATA --threads 1 ${=FLAG} ;;
    pandas)     $PY bench/final_compare.py pandas $OUT/pandas.json $DATA ${=FLAG} ;;
    polars-1)   POLARS_MAX_THREADS=1 $PY bench/final_compare.py polars $OUT/polars-1.json $DATA ${=FLAG} ;;
    polars)     $PY bench/final_compare.py polars $OUT/polars.json $DATA ${=FLAG} ;;
  esac
}
for r in $(seq $ROUNDS); do
  for c in $COLS; do echo "round $r ($MODE): $c"; run $c; done
done
