cmake_minimum_required(VERSION 3.27)
project(stardust_pandas LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.15 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# NOMINSIZE: nanobind compiles extensions with -Os by default (small
# binaries), which comes after -O3 on the command line and wins. -Os keeps
# loops small: little unrolling, little vectorization (stardust-numpy M7.1).
nanobind_add_module(_core NB_STATIC NOMINSIZE
  csrc/core/buffer.cpp
  csrc/core/column.cpp
  csrc/core/convert.cpp
  csrc/ops/cast.cpp
  csrc/ops/elementwise.cpp
  csrc/ops/hash.cpp
  csrc/ops/reduce.cpp
  csrc/ops/selection.cpp
  csrc/parallel/pool.cpp
  csrc/bind/module.cpp)

target_include_directories(_core PRIVATE csrc)

# The only target is this Mac (Apple M4 Pro): -mcpu=native resolves to apple-m4.
# -ffast-math is deliberately absent: it assumes no NaN/inf, and NaN is an
# ordinary value here (DESIGN.md §2.1).
# -ffp-contract=off: clang would otherwise fuse `a + b*c` into one FMA (one
# rounding instead of two), silently changing results versus the source text
# and versus pandas. Kernels that want FMA ask for it explicitly.
target_compile_options(_core PRIVATE -mcpu=native -ffp-contract=off -Wall -Wextra -Wpedantic)

install(TARGETS _core LIBRARY DESTINATION stardust_pandas)
