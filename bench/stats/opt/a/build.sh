#!/bin/sh
# Builds one of the standalone harnesses of this directory against the tree's headers:
#     bench/stats/opt/a/build.sh sweep        ->  ./sweep
# (sweep, log_dd, vforce_error, ndtri_error). Run from the repository root; the include
# paths of nanobind and Python come from .venv (the headers pull them in, nothing is linked).
set -e
name=$1
py=.venv/bin/python
nb=$($py -c "import nanobind; print(nanobind.include_dir())")
pyinc=$($py -c "import sysconfig; print(sysconfig.get_paths()['include'])")
exec clang++ -O3 -std=c++20 -mcpu=native -ffp-contract=off -Icsrc \
    -isystem third_party/boost_math/include -DBOOST_MATH_STANDALONE \
    -isystem "$nb" -isystem "$pyinc" -undefined dynamic_lookup \
    bench/stats/opt/a/$name.cpp -o $name -framework Accelerate
