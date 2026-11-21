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
  csrc/base/buffer.cpp
  csrc/random/random.cpp
  csrc/array/convert.cpp
  csrc/array/iter.cpp
  csrc/array/ndarray.cpp
  csrc/array/ops/asarray.cpp
  csrc/array/ops/copy.cpp
  csrc/array/ops/covariance.cpp
  csrc/array/ops/creation.cpp
  csrc/array/ops/dlpack.cpp
  csrc/array/ops/dtype_info.cpp
  csrc/array/ops/histogram.cpp
  csrc/array/ops/indexing.cpp
  csrc/array/ops/manipulation.cpp
  csrc/array/ops/matmul.cpp
  csrc/array/ops/quantile.cpp
  csrc/array/ops/reduction.cpp
  csrc/array/ops/searching.cpp
  csrc/array/ops/shape.cpp
  csrc/array/ops/sort.cpp
  csrc/array/ops/ufunc.cpp
  csrc/linalg/args.cpp
  csrc/linalg/batch.cpp
  csrc/linalg/errors.cpp
  csrc/linalg/input.cpp
  csrc/linalg/outblock.cpp
  csrc/linalg/pyrefs.cpp
  csrc/linalg/ops/basic.cpp
  csrc/linalg/ops/expm.cpp
  csrc/linalg/ops/factor.cpp
  csrc/linalg/ops/lu.cpp
  csrc/linalg/ops/nonsym.cpp
  csrc/linalg/ops/norms.cpp
  csrc/linalg/ops/qr.cpp
  csrc/linalg/ops/spectral.cpp
  csrc/linalg/ops/special.cpp
  csrc/optimize/batch.cpp
  csrc/optimize/bfgs.cpp
  csrc/optimize/hybrd.cpp
  csrc/optimize/lbfgsb.cpp
  csrc/optimize/linear.cpp
  csrc/optimize/lsq.cpp
  csrc/optimize/nelder_mead.cpp
  csrc/optimize/newton_cg.cpp
  csrc/optimize/scalar.cpp
  csrc/optimize/trust.cpp
  csrc/parallel/pool.cpp
  csrc/regression/glm.cpp
  csrc/regression/linear.cpp
  csrc/regression/penalized.cpp
  csrc/regression/quantile.cpp
  csrc/regression/robust.cpp
  csrc/regression/select.cpp
  csrc/regression/slopes.cpp
  csrc/htest/parametric.cpp
  csrc/htest/gof.cpp
  csrc/htest/rank.cpp
  csrc/sparse/direct.cpp
  csrc/sparse/canonical.cpp
  csrc/sparse/convert.cpp
  csrc/sparse/dok.cpp
  csrc/special/reduce.cpp
  csrc/frame/column.cpp
  csrc/frame/convert.cpp
  csrc/frame/interop.cpp
  csrc/frame/io/csv.cpp
  csrc/frame/ops/cast.cpp
  csrc/frame/ops/elementwise.cpp
  csrc/frame/ops/groupby.cpp
  csrc/frame/ops/hash.cpp
  csrc/frame/ops/join.cpp
  csrc/frame/ops/reduce.cpp
  csrc/frame/ops/selection.cpp
  csrc/frame/ops/sort.cpp
  csrc/bind/errors.cpp
  csrc/bind/pyutil.cpp
  csrc/bind/array_bind.cpp
  csrc/bind/linalg_bind.cpp
  csrc/bind/frame_bind.cpp
  csrc/bind/special_bind.cpp
  csrc/bind/stats_bind.cpp
  csrc/bind/stats_sample.cpp
  csrc/bind/stats_desc.cpp
  csrc/bind/stats_mvcdf.cpp
  csrc/bind/stats_matrix.cpp
  csrc/bind/stats_mv.cpp
  csrc/bind/regression_bind.cpp
  csrc/bind/htest_bind.cpp
  csrc/bind/optimize_bind.cpp
  csrc/bind/sparse_bind.cpp
  csrc/bind/module.cpp)

target_include_directories(_core PRIVATE csrc)
# Boost.Math (headers only, the subset sd.special calls; third_party/boost_math/README.md).
# SYSTEM: its headers are not held to our warning flags.
target_include_directories(_core SYSTEM PRIVATE third_party/boost_math/include)
target_compile_definitions(_core PRIVATE BOOST_MATH_STANDALONE)

# The only target is this Mac (Apple M4 Pro): -mcpu=native resolves to apple-m4.
# -ffast-math is deliberately absent: NaN and inf must keep their meaning.
# -ffp-contract=off: the code, not the compiler, decides where an FMA happens.
# Otherwise clang fuses `a + b*c` (one rounding instead of two) wherever its
# inlining allows, so the same formula could give other bits on another path
# (f(x)[i] == f(x[i]), a NEON kernel against its scalar definition). Kernels
# that want FMA write std::fma / vfmaq.
# -fno-builtin-sin/-cos: for the same reason clang does not merge sin(x) and
# cos(x) of one x into a __sincos_stret call by itself; code that wants the
# merged call writes it (sd::sincos, special/sincos.hpp).
target_compile_options(_core PRIVATE -mcpu=native -ffp-contract=off -fno-builtin-sin -fno-builtin-cos
                       -Wall -Wextra -Wpedantic)

# Accelerate: the new LAPACK interface with 64-bit integers.
target_compile_definitions(_core PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
target_link_libraries(_core PRIVATE "-framework Accelerate")

install(TARGETS _core LIBRARY DESTINATION stardust)
