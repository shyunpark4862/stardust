#!/bin/zsh
# Old packages and stardust measured side by side in one sitting: the old
# benchmark scripts run on .venv-legacy (the old packages) and on .venv
# (stardust, through bench/merge/shim), alternating, so the machine's drift
# hits both. Then
#   .venv/bin/python bench/merge/sameseat_table.py OUTDIR > OUTDIR/table.md
# Usage: zsh bench/merge/sameseat.sh OUTDIR [ROUNDS] [AREAS...]
#   areas: array linalg frame (default: all). Keep the machine quiet.
# The numpy/scipy/pandas reference rows run once, in the first round.
set -e
OUT=${1:A}  # absolute: final_report.py resolves --out against its own directory
ROUNDS=${2:-2}
shift $(( $# < 2 ? $# : 2 ))
if (( $# )); then AREAS=($@); else AREAS=(array linalg frame); fi
ROOT=$PWD
OLD=$ROOT/.venv-legacy/bin/python
NEW=$ROOT/.venv/bin/python
SHIM=$ROOT/bench/merge/shim
DATA=$ROOT/build/baseline-data
A=legacy/numpy/bench/final_report.py
L=legacy/linalg/bench/stage_compare.py
F=legacy/pandas/bench/final_compare.py
mkdir -p $OUT
rm -f $OUT/*.json(N)
for r in $(seq $ROUNDS); do
  if (( ${AREAS[(Ie)array]} )); then
    echo "round $r: array"
    $OLD $A --label legacy --order 1 --out $OUT/array-legacy-r$r.json
    PYTHONPATH=$SHIM $NEW $A --label new --order 2 --out $OUT/array-new-r$r.json
    $OLD $A --label legacy1 --order 3 --threads 1 --out $OUT/array-legacy1-r$r.json
    PYTHONPATH=$SHIM $NEW $A --label new1 --order 4 --threads 1 --out $OUT/array-new1-r$r.json
  fi
  if (( ${AREAS[(Ie)linalg]} )); then
    echo "round $r: linalg"
    $OLD $L legacy $OUT/linalg-legacy.json
    PYTHONPATH=$SHIM $NEW $L new $OUT/linalg-new.json
    $OLD $L legacy1 $OUT/linalg-legacy1.json --threads 1
    PYTHONPATH=$SHIM $NEW $L new1 $OUT/linalg-new1.json --threads 1
    if (( r == 1 )); then $OLD $L ref $OUT/linalg-ref.json --ref; fi
  fi
  if (( ${AREAS[(Ie)frame]} )); then
    echo "round $r: frame"  # final_compare.py keeps each row's best over the rounds in one file
    $OLD $F sdpd $OUT/frame-legacy.json $DATA
    PYTHONPATH=$SHIM $NEW $F sdpd $OUT/frame-new.json $DATA
    $OLD $F sdpd $OUT/frame-legacy1.json $DATA --threads 1
    PYTHONPATH=$SHIM $NEW $F sdpd $OUT/frame-new1.json $DATA --threads 1
    if (( r == 1 )); then $OLD $F pandas $OUT/frame-pandas.json $DATA; fi
  fi
done
echo done
