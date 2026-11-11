cmake_minimum_required(VERSION 3.27)
project(stardust LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.15 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# One extension module for every area (DESIGN.md §3.2); the areas are its
# submodules. NOMINSIZE: nanobind compiles extensions with -Os by default,
# which comes after -O3 on the command line and wins; -Os keeps loops small
# (little unrolling, little vectorization).
nanobind_add_module(_core NB_STATIC NOMINSIZE
  csrc/bind/module.cpp)

target_include_directories(_core PRIVATE csrc)

# The only target is this Mac (Apple M4 Pro): -mcpu=native resolves to apple-m4.
# -ffast-math is deliberately absent: NaN and inf must keep their meaning.
# -ffp-contract=off: clang would otherwise fuse `a + b*c` into one FMA (one
# rounding instead of two), silently changing results versus the source text
# and versus numpy. Kernels that want FMA ask for it explicitly.
# -fno-builtin-sin/-cos: on Apple targets clang merges sin(x) and cos(x) of the
# same x into one __sincos_stret call, whose results can differ by 1 ULP from
# sin()/cos() called separately (which is what numpy does).
target_compile_options(_core PRIVATE -mcpu=native -ffp-contract=off -fno-builtin-sin -fno-builtin-cos
                       -Wall -Wextra -Wpedantic)

# Accelerate: the new LAPACK interface with 64-bit integers.
target_compile_definitions(_core PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
target_link_libraries(_core PRIVATE "-framework Accelerate")

install(TARGETS _core LIBRARY DESTINATION stardust)
