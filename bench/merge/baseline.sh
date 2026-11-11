#!/bin/zsh
# M0: the benchmark baseline (ROADMAP.md, safety net 3). The three old
# packages, built from legacy/ at their final commits (.venv-legacy), timed
# with each project's own final-report script, in one sitting, next to the
# reference libraries. M7 reruns the same rows on stardust.
#   array : legacy/numpy/bench/final_report.py     (default threads, 1 thread, numpy)
#   linalg: legacy/linalg/bench/stage_compare.py   (default threads, 1 thread, numpy/scipy)
#   frame : legacy/pandas/bench/final_compare.py   (default threads, 1 thread, pandas, Polars)
# Results: bench/merge/baseline/<area>-<column>.json
# Usage: zsh bench/merge/baseline.sh [ROUNDS]   (from the repository root; keep the machine quiet)
set -e
ROUNDS=${1:-2}
ROOT=$PWD
PY=$ROOT/.venv-legacy/bin/python
OUT=$ROOT/bench/merge/baseline
DATA=$ROOT/build/baseline-data
mkdir -p $OUT $DATA
rm -f $OUT/*.json(N)

$PY -c "import platform, numpy, scipy, pandas, polars; print('python', platform.python_version(), platform.machine(), 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__, 'polars', polars.__version__)" > $OUT/environment.txt
for n in numpy linalg pandas; do echo "legacy/$n $(git log -1 --format=%h -- legacy/$n)" >> $OUT/environment.txt; done

A=legacy/numpy/bench/final_report.py
L=legacy/linalg/bench/stage_compare.py
F=legacy/pandas/bench/final_compare.py
for r in $(seq $ROUNDS); do
  echo "round $r: array"
  # final_report.py keeps the best of 7 repeats per row and overwrites its output,
  # so each round writes its own file; the table takes the best per row.
  $PY $A --label legacy --order 1 --out $OUT/array-legacy-r$r.json
  $PY $A --label legacy-1 --order 2 --threads 1 --out $OUT/array-legacy-1-r$r.json
  $PY $A --numpy --label numpy --order 3 --out $OUT/array-numpy-r$r.json
  echo "round $r: linalg"   # stage_compare.py keeps each row's best over the rounds itself
  $PY $L legacy $OUT/linalg-legacy.json
  $PY $L legacy-1 $OUT/linalg-legacy-1.json --threads 1
  echo "round $r: frame"    # so does final_compare.py
  $PY $F sdpd $OUT/frame-legacy.json $DATA
  $PY $F sdpd $OUT/frame-legacy-1.json $DATA --threads 1
  $PY $F pandas $OUT/frame-pandas.json $DATA
  POLARS_MAX_THREADS=1 $PY $F polars $OUT/frame-polars-1.json $DATA
  $PY $F polars $OUT/frame-polars.json $DATA
done
echo "reference: linalg (numpy / scipy)"
$PY $L legacy $OUT/linalg-ref.json --ref
echo done
