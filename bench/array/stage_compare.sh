#!/bin/sh
# M10: build stardust as it was at the end of each milestone and measure all
# builds (and numpy) with the same script, bench/final_report.py.
#
# Each commit is checked out into a git worktree under $WORK, configured the
# way scikit-build-core does it here (Release, this venv's Python and
# nanobind), built, and imported from its own src/ directory.
#
#   sh bench/stage_compare.sh [workdir]
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY="$ROOT/.venv/bin/python"
WORK=${1:-${TMPDIR:-/tmp}/stardust-stages}
NB=$("$PY" -m nanobind --cmake_dir)
mkdir -p "$WORK"

stage() {  # order label commit
    order=$1; label=$2; commit=$3; wt="$WORK/$label"
    [ -d "$wt" ] || git -C "$ROOT" worktree add --detach "$wt" "$commit" >/dev/null
    cmake -S "$wt" -B "$wt/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DPython_EXECUTABLE="$PY" -Dnanobind_ROOT="$NB" >/dev/null
    cmake --build "$wt/build" >/dev/null
    cp "$wt"/build/_core*.so "$wt/src/stardust_numpy/"
    "$PY" "$ROOT/bench/final_report.py" --label "$label" --order "$order" --src "$wt/src" \
          --out "results/final/$label.json"
}

stage 1 M6-baseline 014a32b   # all features, plain scalar loops
stage 2 M7-1core    34b5214   # + single-core optimization
stage 3 M8-threads  e93042c   # + thread pool (10 threads)
stage 4 M9-vforce   8e85abc   # + vForce
stage 5 M9plus      5c78115   # + fused gather/scatter
stage 6 M10         caf4482   # + int64 matmul unrolled
# The final build on one thread, for a like-for-like comparison with numpy.
"$PY" "$ROOT/bench/final_report.py" --label "M10-1thread" --order 7 --threads 1 --src "$WORK/M10/src" \
      --out "results/final/M10-1thread.json"
"$PY" "$ROOT/bench/final_report.py" --numpy --out results/final/numpy.json
"$PY" "$ROOT/bench/final_report.py" --table "$ROOT"/bench/results/final/*.json
