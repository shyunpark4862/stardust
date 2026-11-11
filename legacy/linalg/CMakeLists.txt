cmake_minimum_required(VERSION 3.27)
project(stardust_linalg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.15 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# NOMINSIZE: nanobind compiles extensions with -Os by default, which comes
# after -O3 on the command line and wins (stardust-numpy measured this in M7.1).
nanobind_add_module(_core NB_STATIC NOMINSIZE
  csrc/core/args.cpp
  csrc/core/batch.cpp
  csrc/core/errors.cpp
  csrc/core/input.cpp
  csrc/core/outblock.cpp
  csrc/core/pyrefs.cpp
  csrc/ops/basic.cpp
  csrc/ops/factor.cpp
  csrc/ops/lu.cpp
  csrc/ops/nonsym.cpp
  csrc/ops/norms.cpp
  csrc/ops/qr.cpp
  csrc/ops/spectral.cpp
  csrc/ops/special.cpp
  csrc/parallel/pool.cpp
  csrc/bind/module.cpp)

target_include_directories(_core PRIVATE csrc)

# Same flags as stardust-numpy (its CMakeLists.txt explains each one).
# The only target is this Mac (Apple M4 Pro). No -ffast-math: NaN/inf must keep
# their meaning (the finiteness check of DESIGN.md §2.4 relies on it).
target_compile_options(_core PRIVATE -mcpu=native -ffp-contract=off
                       -Wall -Wextra -Wpedantic)

# Accelerate: new LAPACK interface with 64-bit integers (DESIGN.md §1.5).
target_compile_definitions(_core PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
target_link_libraries(_core PRIVATE "-framework Accelerate")

install(TARGETS _core LIBRARY DESTINATION stardust_linalg)
