#!/bin/zsh
# The baseline's rows (bench/merge/baseline.sh) measured again on stardust: the
# same old scripts, which import stardust_numpy / stardust_linalg and get
# stardust through bench/merge/shim. Compare with
#   .venv/bin/python bench/merge/baseline_table.py --compare OUTDIR OUTDIR/compare.md
# Usage: zsh bench/merge/rerun.sh OUTDIR [ROUNDS] [AREAS...]   (areas: array linalg; keep the machine quiet)
set -e
OUT=${1:A}  # absolute: final_report.py resolves --out against its own directory
ROUNDS=${2:-2}
shift $(( $# < 2 ? $# : 2 ))
if (( $# )); then AREAS=($@); else AREAS=(array linalg); fi
ROOT=$PWD
PY=$ROOT/.venv/bin/python
export PYTHONPATH=$ROOT/bench/merge/shim
mkdir -p $OUT
rm -f $OUT/*.json(N)
A=bench/array/final_report.py
L=bench/linalg/stage_compare.py
for r in $(seq $ROUNDS); do
  if (( ${AREAS[(Ie)array]} )); then
    echo "round $r: array"
    $PY $A --label legacy --order 1 --out $OUT/array-legacy-r$r.json
    $PY $A --label legacy-1 --order 2 --threads 1 --out $OUT/array-legacy-1-r$r.json
  fi
  if (( ${AREAS[(Ie)linalg]} )); then
    echo "round $r: linalg"
    $PY $L legacy $OUT/linalg-legacy.json
    $PY $L legacy-1 $OUT/linalg-legacy-1.json --threads 1
  fi
done
echo done
