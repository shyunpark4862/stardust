// Basic operations: vecdot, outer, trace, diagonal, matrix_power, kron, cross, block_diag,
// kronsum, bandwidth
// (docs/api/linalg.md §2; docs/design/linalg.md §10.8).
//
// None of these calls LAPACK, so NaN and inf pass through unchecked
// (docs/design/linalg.md §2.3) -- except matrix_power with n < 0, which inverts
// through getrf and so checks like inv (the same section).
#include "linalg/ops/basic.hpp"
#include "bind/errors.hpp"
#include "bind/pyutil.hpp"

#include "linalg/args.hpp"
#include "linalg/cost.hpp"
#include "linalg/fastsum.hpp"
#include "linalg/outblock.hpp"
#include "linalg/pairwise.hpp"
#include "linalg/pyrefs.hpp"
#include "linalg/ops/common.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <deque>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sd::linalg {

namespace {

// Shape and strides of an operand padded on the left to `nd` axes (size 1, stride 0).
struct Padded {
    std::vector<std::int64_t> shape, strides;
    Padded(const Input& x, int nd) : shape(nd, 1), strides(nd, 0) {
        const int lead = nd - x.ndim();
        for (int d = 0; d < x.ndim(); ++d) {
            shape[lead + d] = x.shape(d);
            strides[lead + d] = x.stride(d);
        }
    }
    void erase(int axis) {
        shape.erase(shape.begin() + axis);
        strides.erase(strides.begin() + axis);
    }
};


// --------------------------------------------------------------- vecdot --

nb::object vecdot(nb::handle x1o, nb::handle x2o, nb::handle axis_o) {
    Input x(x1o, "vecdot", "x1");
    Input y(x2o, "vecdot", "x2");
    const int nd = std::max(x.ndim(), y.ndim());
    // The axis counts in the broadcast shape (Array API); it is not broadcast itself.
    const int ax = normalize_axis(parse_int(axis_o, "vecdot", "axis"), nd, "vecdot");
    Padded px(x, nd), py(y, nd);
    if (px.shape[ax] != py.shape[ax])
        throw nb::value_error(("vecdot(): 'x1' and 'x2' must have the same size along the axis, got " +
                               std::to_string(px.shape[ax]) + " and " + std::to_string(py.shape[ax]))
                                  .c_str());
    const std::int64_t len = px.shape[ax], sx = px.strides[ax], sy = py.strides[ax];
    px.erase(ax);
    py.erase(ax);
    Batch b({{px.shape, px.strides}, {py.shape, py.strides}}, "vecdot");
    const bool complex = computes_complex(x.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        return visit_storage(x.dtype(), [&]<class SX>(std::type_identity<SX>) {
            return visit_storage(y.dtype(), [&]<class SY>(std::type_identity<SY>) -> nb::object {
                if constexpr ((std::is_same_v<SX, cplx> || std::is_same_v<SY, cplx>) && !std::is_same_v<T, cplx>) {
                    return nb::none();  // unreachable: a complex input makes T complex
                } else {
                    // sum_i conj(x1[i]) * x2[i]  (x1 is conjugated: Array API). Inputs of the
                    // computing type with positive strides are read in place by the fast sums
                    // (fastsum.hpp: Accelerate's ddot / zdotc on long runs, docs/design/linalg.md
                    // §8.2); any other layout or dtype is converted into two buffers per leaf
                    // and summed by the fused leaves, the chunks the same either way.
                    constexpr bool direct_type = std::is_same_v<SX, T> && std::is_same_v<SY, T>;
                    constexpr std::int64_t isx = sizeof(SX), isy = sizeof(SY);
                    [[maybe_unused]] const bool direct =
                        direct_type && sx > 0 && sy > 0 && sx % isx == 0 && sy % isy == 0;
                    // the result of dot(offsets) for every batch element (a scalar for none)
                    auto all = [&](const auto& dot) -> nb::object {
                        if (b.ndim() == 0) {
                            const std::int64_t off[2] = {0, 0};
                            return py_scalar(dot(off));
                        }
                        Out out = alloc_c(b.shape(), dtype_of<T>());
                        T* res = out.as<T>();
                        run_batch(
                            b, cost(kStream, double(len)), [&](std::int64_t i, const std::int64_t* off, Scratch&) {
                                res[i] = dot(off);
                                return 0;
                            }
                        );
                        return to_array(out);
                    };
                    if constexpr (direct_type) {
                        if (direct)  // a body of its own: small, so it inlines into the batch loop
                            return all([&](const std::int64_t* off) -> T {
                                const T* a = reinterpret_cast<const T*>(x.data() + off[0]);
                                const T* c = reinterpret_cast<const T*>(y.data() + off[1]);
                                if constexpr (std::is_same_v<T, cplx>) return fast::cdot(len, a, sx / isx, c, sy / isy);
                                else return fast::dot(len, a, sx / isx, c, sy / isy);
                            });
                    }
                    return all([&](const std::int64_t* off) -> T {
                        const std::byte* p = x.data() + off[0];
                        const std::byte* q = y.data() + off[1];
                        auto leaf = [&](std::int64_t b0, std::int64_t k) -> T {
                            alignas(16) double ra[kPairwiseBlock], rc[kPairwiseBlock];  // no constructor runs
                            T* a = reinterpret_cast<T*>(ra);
                            T* c = reinterpret_cast<T*>(rc);
                            const std::byte* pp = p + b0 * sx;
                            const std::byte* qq = q + b0 * sy;
                            for (std::int64_t i = 0; i < k; ++i) {
                                a[i] = convert<T>(load<SX>(pp + i * sx));
                                c[i] = convert<T>(load<SY>(qq + i * sy));
                            }
                            if constexpr (std::is_same_v<T, cplx>) return detail::fma_cdot_leaf(a, c, k);
                            else return detail::fma_dot_leaf(a, c, k);
                        };
                        const T s = fast::chunked_sum<T>(len, [&](std::int64_t b0, std::int64_t k) -> T {
                            if constexpr (std::is_same_v<T, cplx>) return pairwise::tree_complex(b0, k, leaf);
                            else return pairwise::tree(b0, k, leaf);
                        });
                        if constexpr (std::is_same_v<T, cplx>) return {0. + s.real(), 0. + s.imag()};
                        else return 0. + s;
                    });
                }
            });
        });
    });
}

// ---------------------------------------------------------------- outer --

nb::object outer(nb::handle x1o, nb::handle x2o) {
    Input x(x1o, "outer", "x1");
    Input y(x2o, "outer", "x2");
    // (..., M) and (..., N) -> (..., M, N); batch axes broadcast (docs/api/linalg.md §2).
    const std::int64_t m = x.shape(-1), n = y.shape(-1), sx = x.stride(-1), sy = y.stride(-1);
    Batch b({{x.batch_shape(1), x.batch_strides(1)}, {y.batch_shape(1), y.batch_strides(1)}}, "outer");
    // The result has one axis more than the inputs: a 32-dimensional input has no result.
    if (b.ndim() + 2 > static_cast<std::size_t>(kMaxNdim))
        throw nb::value_error(("arrays may have at most " + std::to_string(kMaxNdim) + " dimensions").c_str());
    const bool complex = computes_complex(x.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_matrices(b.shape(), m, n, dtype_of<T>());
        T* res = out.as<T>();
        visit_storage(x.dtype(), [&]<class SX>(std::type_identity<SX>) {
            visit_storage(y.dtype(), [&]<class SY>(std::type_identity<SY>) {
                if constexpr (!((std::is_same_v<SX, cplx> || std::is_same_v<SY, cplx>) && !std::is_same_v<T, cplx>)) {
                    run_batch(b, cost(kCopy, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
                        // x1 converted once into scratch; the result column by column (column-major)
                        T* xv = s.get<T>(static_cast<std::size_t>(m));
                        for (std::int64_t r = 0; r < m; ++r) xv[r] = convert<T>(load<SX>(x.data() + off[0] + r * sx));
                        T* o = res + i * m * n;
                        for (std::int64_t c = 0; c < n; ++c) {
                            const T yc = convert<T>(load<SY>(y.data() + off[1] + c * sy));
                            for (std::int64_t r = 0; r < m; ++r) o[c * m + r] = mul(xv[r], yc);  // as sd's x * y
                        }
                        return 0;
                    });
                }
            });
        });
        return to_array(out);
    });
}

// -------------------------------------------------------- trace, diagonal --

// The k-th diagonal of an m x n matrix: its length and the (row, col) where it starts.
struct Diag {
    std::int64_t len, r0, c0;
    Diag(std::int64_t m, std::int64_t n, std::int64_t k) {
        // An offset past the matrix gives an empty diagonal starting at
        // (0, 0): -k would overflow at k = INT64_MIN, and r0 * stride for a
        // huge |k| would too (both undefined behaviour).
        if (k >= n || k <= -m) {
            len = r0 = c0 = 0;
            return;
        }
        r0 = k >= 0 ? 0 : -k;
        c0 = k >= 0 ? k : 0;
        len = std::max<std::int64_t>(0, std::min(m - r0, n - c0));
    }
};

[[noreturn]] void complex_to_real(const char* func, DType to) {
    throw nb::type_error((std::string(func) + "(): cannot convert complex128 to " +
                          (to == DType::Int64 ? "int64" : "float64") +
                          ": the imaginary part would be lost (take the real part explicitly)")
                             .c_str());
}

nb::object trace(nb::handle xo, nb::handle k_o, nb::handle dtype_o) {
    Input x(xo, "trace", "x");
    require_matrix(x);
    const std::int64_t k = parse_int(k_o, "trace", "k");
    const std::optional<DType> want = parse_dtype(dtype_o, "trace");
    // dtype= is an explicit conversion, as in sd.sum (docs/design/linalg.md §2.1; docs/api/array.md §7.5).
    DType acc_dt = computes_complex(x.dtype()) ? DType::Complex128 : DType::Float64;
    if (want) {
        if (*want == DType::Bool) throw nb::type_error("trace(): cannot accumulate in bool; use any() or all()");
        if (x.dtype() == DType::Complex128 && *want != DType::Complex128) complex_to_real("trace", *want);
        acc_dt = *want;
    }
    const Diag dg(x.shape(-2), x.shape(-1), k);
    const std::int64_t step = x.stride(-2) + x.stride(-1);
    const std::int64_t start = dg.r0 * x.stride(-2) + dg.c0 * x.stride(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "trace");
    auto run = [&]<class T>(std::type_identity<T>) -> nb::object {
        return visit_storage(x.dtype(), [&]<class S>(std::type_identity<S>) -> nb::object {
            if constexpr (std::is_same_v<S, cplx> && !std::is_same_v<T, cplx>) {
                return nb::none();  // unreachable: rejected above
            } else {
                auto sum = [&](std::int64_t off) {
                    const std::byte* p = x.data() + off + start;
                    if constexpr (std::is_same_v<T, std::int64_t>) {
                        std::uint64_t acc = 0;  // wraps on overflow, as sd.sum of int64
                        for (std::int64_t i = 0; i < dg.len; ++i)
                            acc += static_cast<std::uint64_t>(convert<T>(load<S>(p + i * step)));
                        return static_cast<std::int64_t>(acc);
                    } else {  // pairwise (docs/design/linalg.md §8.2): the bits of sd.sum(diagonal(x))
                        return pairwise_sum_mt<T>(dg.len, [&](std::int64_t b0, std::int64_t k, T* out) {
                            for (std::int64_t i = 0; i < k; ++i) out[i] = convert<T>(load<S>(p + (b0 + i) * step));
                        });
                    }
                };
                if (b.ndim() == 0) return py_scalar(sum(0));
                Out out = alloc_c(b.shape(), dtype_of<T>());
                T* res = out.as<T>();
                run_batch(b, cost(kStream, double(dg.len)), [&](std::int64_t i, const std::int64_t* off, Scratch&) {
                    res[i] = sum(off[0]);
                    return 0;
                });
                return to_array(out);
            }
        });
    };
    if (acc_dt == DType::Int64) return run(std::type_identity<std::int64_t>{});
    if (acc_dt == DType::Float64) return run(std::type_identity<double>{});
    return run(std::type_identity<cplx>{});
}

nb::object diagonal(nb::handle xo, nb::handle k_o) {
    Input x(xo, "diagonal", "x");
    require_matrix(x);
    const std::int64_t k = parse_int(k_o, "diagonal", "k");
    const Diag dg(x.shape(-2), x.shape(-1), k);
    const std::int64_t step = x.stride(-2) + x.stride(-1);
    const std::int64_t start = dg.r0 * x.stride(-2) + dg.c0 * x.stride(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "diagonal");
    // Extraction, not computation: the input dtype is kept, and the result is a copy
    // (docs/design/linalg.md §10.8).
    Out out = alloc_vectors(b.shape(), dg.len, x.dtype());
    // a typed strided copy (a memcpy call per element cost 2-3x numpy, docs/design/linalg.md §9)
    visit_storage(x.dtype(), [&]<class S>(std::type_identity<S>) {
        S* o = reinterpret_cast<S*>(out.data);
        run_batch(b, cost(kCopy, double(dg.len)), [&](std::int64_t i, const std::int64_t* off, Scratch&) {
            const std::byte* p = x.data() + off[0] + start;
            S* oi = o + i * dg.len;
            for (std::int64_t j = 0; j < dg.len; ++j) oi[j] = load<S>(p + j * step);
            return 0;
        });
        return 0;
    });
    return to_array(out);
}

// --------------------------------------------------------- matrix_power --

nb::object matrix_power(nb::handle xo, nb::handle n_o) {
    Input x(xo, "matrix_power", "x");
    require_square(x);
    const std::int64_t n = parse_int(n_o, "matrix_power", "n");
    const lint m = x.shape(-1);
    // |n| without overflow at INT64_MIN
    const std::uint64_t e = n < 0 ? std::uint64_t(0) - static_cast<std::uint64_t>(n) : static_cast<std::uint64_t>(n);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "matrix_power");
    return with_type(computes_complex(x.dtype()), [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_matrices(b.shape(), m, m, dtype_of<T>());
        T* res = out.as<T>();
        const std::size_t mm = static_cast<std::size_t>(m * m);
        const Failure f = run_batch(
            b, cost(kLU, double(m) * m),
            [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
                T* o = res + i * m * m;
                if (m == 0) return 0;
                if (e == 0) {  // x^0 = I, whatever x holds
                    std::fill(o, o + mm, T(0));
                    for (lint d = 0; d < m; ++d) o[d * m + d] = T(1);
                    return 0;
                }
                // The base: x^-1 (column-major in scratch), or x itself. A float64 or complex128
                // x with unit-stride columns or rows is read in place: a row-major x is the
                // column-major x^T, and every product with the base takes it transposed (no
                // packing; the last bits can differ from a packed copy's). Tiny float64
                // matrices are packed for gemm_square's own kernel (tiny_gemm).
                const T* base;
                lint ldb = m;
                char opb = 'N';
                constexpr std::int64_t sz = sizeof(T);
                const std::byte* xp = x.data() + off[0];
                const bool tiny = std::is_same_v<T, double> && m <= lapack::kTinyGemm;
                if (n < 0) {
                    // x^-e = (x^-1)^e. The inverse goes through LAPACK, so it is checked like inv.
                    T* lu;
                    lint* ipiv;
                    if (const int code = lu_factor<T>(x, off[0], m, s, lu, ipiv)) return code;
                    T* inv = s.get<T>(mm, 1);
                    lu_inverse(lu, ipiv, m, inv);
                    base = inv;
                } else if (!tiny && x.dtype() == dtype_of<T>() && aligned16(xp) && x.stride(-2) == sz &&
                           x.stride(-1) % 16 == 0 && x.stride(-1) / sz >= m) {
                    base = reinterpret_cast<const T*>(xp);
                    ldb = x.stride(-1) / sz;
                } else if (!tiny && x.dtype() == dtype_of<T>() && aligned16(xp) && x.stride(-1) == sz &&
                           x.stride(-2) % 16 == 0 && x.stride(-2) / sz >= m) {
                    base = reinterpret_cast<const T*>(xp);
                    ldb = x.stride(-2) / sz;
                    opb = 'T';
                } else {
                    T* packed = s.get<T>(mm, 1);
                    pack(x.dtype(), xp, m, m, x.stride(-2), x.stride(-1), packed, m, false);
                    base = packed;
                }
                // Binary powering, most significant bit first: r = base; for each further bit,
                // r = r*r, then r = r*base if the bit is set. log2(e) squarings + popcount(e)-1 products.
                // The products ping-pong between two scratch buffers and the last one goes
                // straight to the output slot (no copies of r, docs/design/linalg.md §9).
                int top = 63;
                while (!((e >> top) & 1)) --top;
                LapackSlot<T> os(o, static_cast<std::int64_t>(mm), s, Scratch::kAlign0);
                if (top == 0) {  // x^1 and x^-1: base itself
                    T* d = os.get();
                    for (lint j = 0; j < m; ++j)
                        for (lint r = 0; r < m; ++r) d[j * m + r] = opb == 'N' ? base[j * ldb + r] : base[r * ldb + j];
                    return 0;
                }
                const int total = top + std::popcount(e) - 1;  // products
                T* buf[2] = {s.get<T>(mm, 2), s.get<T>(mm, 3)};
                const T* r = base;
                lint ldr = ldb;
                char opr = opb;
                int done = 0, next = 0;
                auto product = [&](const T* lhs, lint ldl, char opl, const T* rhs, lint ldrh, char oprh) {
                    T* dst = ++done == total ? os.get() : buf[next];
                    if (tiny) lapack::gemm_square(m, lhs, rhs, dst);  // both packed column-major here
                    else lapack::gemm(opl, oprh, m, m, m, lhs, ldl, rhs, ldrh, dst, m);
                    next ^= 1;
                    r = dst;
                    ldr = m;
                    opr = 'N';
                };
                for (int bit = top - 1; bit >= 0; --bit) {
                    product(r, ldr, opr, r, ldr, opr);
                    if ((e >> bit) & 1) product(r, ldr, opr, base, ldb, opb);
                }
                return 0;
            },
            n < 0 ? lu_task_cap(m, m) : 0
        );  // x^-1 is getrf + getrs(I)
        if (f) raise(b, f, "matrix_power", "x");
        return to_array(out);
    });
}

// ----------------------------------------------------------------- kron --

// A kron operand as a matrix: a 1-D (N,) is the one row (1, N) with no batch axes (as
// np.kron and sd.sparse.kron read it; docs/api/linalg.md §2).
struct KronOperand {
    std::int64_t rows, cols, rs, cs;
    std::span<const std::int64_t> bshape, bstrides;
    explicit KronOperand(const Input& x)
        : rows(x.ndim() == 1 ? 1 : x.shape(-2)),
          cols(x.shape(-1)),
          rs(x.ndim() == 1 ? 0 : x.stride(-2)),
          cs(x.stride(-1)),
          bshape(x.ndim() == 1 ? std::span<const std::int64_t>{} : x.batch_shape(2)),
          bstrides(x.ndim() == 1 ? std::span<const std::int64_t>{} : x.batch_strides(2)) {}
};

nb::object kron(nb::handle x1o, nb::handle x2o) {
    Input x(x1o, "kron", "x1");
    Input y(x2o, "kron", "x2");
    // (..., M, N) (x) (..., P, Q) -> (..., M*P, N*Q); the batch axes broadcast like
    // every other matrix function's (np.kron differs for ndim > 2, docs/design/linalg.md §2.7).
    // A 1-D operand is one row; two 1-D operands give the 1-D (N*Q,).
    const KronOperand ox(x), oy(y);
    const std::int64_t m = ox.rows, n = ox.cols, p = oy.rows, q = oy.cols;
    const bool vector = x.ndim() == 1 && y.ndim() == 1;
    Batch b({{ox.bshape, ox.bstrides}, {oy.bshape, oy.bstrides}}, "kron");
    // M*P and N*Q of broadcast views can pass int64; the rest is checked by alloc_matrices.
    std::int64_t rows, cols;
    if (__builtin_mul_overflow(m, p, &rows) || __builtin_mul_overflow(n, q, &cols))
        throw nb::value_error("array is too big");
    const bool complex = computes_complex(x.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        // (1, N*Q) column-major is the contiguous vector (N*Q,)
        Out out = vector ? alloc_vectors(b.shape(), cols, dtype_of<T>())
                         : alloc_matrices(b.shape(), rows, cols, dtype_of<T>());
        T* res = out.as<T>();
        const std::int64_t block = rows * cols;
        run_batch(b, cost(kCopy, double(block)), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            // both operands converted into scratch once, column-major; NaN passes through
            T* a = s.get<T>(static_cast<std::size_t>(m * n), 0);
            T* c = s.get<T>(static_cast<std::size_t>(p * q), 1);
            pack(x.dtype(), x.data() + off[0], m, n, ox.rs, ox.cs, a, m, false);
            pack(y.dtype(), y.data() + off[1], p, q, oy.rs, oy.cs, c, p, false);
            T* o = res + i * block;
            // result column j*q + l, row block i: a[i, j] * c[:, l], with a on the left as
            // np.kron's multiply (the complex product's bits depend on the order)
            for (std::int64_t j = 0; j < n; ++j)
                for (std::int64_t l = 0; l < q; ++l) {
                    T* col = o + (j * q + l) * rows;
                    const T* cl = c + l * p;
                    for (std::int64_t r = 0; r < m; ++r) {
                        const T arj = a[j * m + r];
                        T* dst = col + r * p;
                        for (std::int64_t k = 0; k < p; ++k) dst[k] = mul(arj, cl[k]);
                    }
                }
            return 0;
        });
        return to_array(out);
    });
}

// ---------------------------------------------------------------- cross --

nb::object cross(nb::handle x1o, nb::handle x2o, nb::handle axis_o) {
    Input x(x1o, "cross", "x1");
    Input y(x2o, "cross", "x2");
    const int nd = std::max(x.ndim(), y.ndim());
    // As vecdot: the axis counts in the broadcast shape and is not broadcast itself (Array API).
    const int ax = normalize_axis(parse_int(axis_o, "cross", "axis"), nd, "cross");
    Padded px(x, nd), py(y, nd);
    if (px.shape[ax] != 3 || py.shape[ax] != 3)
        throw nb::value_error(("cross(): 'x1' and 'x2' must have size 3 along the axis, got " +
                               std::to_string(px.shape[ax]) + " and " + std::to_string(py.shape[ax]))
                                  .c_str());
    const std::int64_t sx = px.strides[ax], sy = py.strides[ax];
    px.erase(ax);
    py.erase(ax);
    Batch b({{px.shape, px.strides}, {py.shape, py.strides}}, "cross");
    // The result: the broadcast shape with the vector axis back in its place, C order.
    std::vector<std::int64_t> shape = b.shape();
    shape.insert(shape.begin() + ax, 3);
    std::int64_t inner = 1;  // elements between two components of one result vector
    for (std::size_t d = static_cast<std::size_t>(ax); d < b.ndim(); ++d) inner *= b.shape()[d];
    const bool complex = computes_complex(x.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_c(shape, dtype_of<T>());
        T* res = out.as<T>();
        visit_storage(x.dtype(), [&]<class SX>(std::type_identity<SX>) {
            visit_storage(y.dtype(), [&]<class SY>(std::type_identity<SY>) {
                if constexpr (!((std::is_same_v<SX, cplx> || std::is_same_v<SY, cplx>) && !std::is_same_v<T, cplx>)) {
                    run_batch(b, cost(kCopy, 6), [&](std::int64_t i, const std::int64_t* off, Scratch&) {
                        T u[3], v[3];
                        for (int k = 0; k < 3; ++k) {
                            u[k] = convert<T>(load<SX>(x.data() + off[0] + k * sx));
                            v[k] = convert<T>(load<SY>(y.data() + off[1] + k * sy));
                        }
                        T* o = res + (i / inner) * 3 * inner + i % inner;
                        // numpy's cross: each product rounded, then the difference
                        o[0] = mul(u[1], v[2]) - mul(u[2], v[1]);
                        o[inner] = mul(u[2], v[0]) - mul(u[0], v[2]);
                        o[2 * inner] = mul(u[0], v[1]) - mul(u[1], v[0]);
                        return 0;
                    });
                }
            });
        });
        return to_array(out);
    });
}

// ----------------------------------------------------------- block_diag --

// The arrays on the diagonal of one matrix (docs/api/linalg.md §2): a 1-D (N,) is the row
// (1, N), as kron's operands; the batch axes broadcast like every other matrix function's
// (scipy.linalg.block_diag's batch rule). Extraction, not computation: the result dtype is
// the promotion of the inputs and every value is copied as it is (diagonal keeps the dtype
// too); the rest is 0 (False).
nb::object block_diag(nb::handle arrays_o) {
    if (!PyList_Check(arrays_o.ptr()) && !PyTuple_Check(arrays_o.ptr()))
        throw nb::type_error(
            ("block_diag(): 'arrays' must be a list or tuple of sd.ndarray, not " + py_type_name(arrays_o)).c_str()
        );
    std::deque<std::string> names;
    std::deque<Input> in;
    for (nb::handle h : arrays_o) {
        names.push_back("arrays[" + std::to_string(in.size()) + "]");
        in.emplace_back(h, "block_diag", names.back().c_str());
    }
    if (in.empty()) throw nb::value_error("block_diag(): 'arrays' must hold at least one array");
    std::vector<KronOperand> ops;
    ops.reserve(in.size());
    DType dt = DType::Bool;
    std::size_t nd = 0;
    std::int64_t rows = 0, cols = 0;
    for (const Input& x : in) {
        ops.emplace_back(x);
        dt = promote(dt, x.dtype());
        nd = std::max(nd, ops.back().bshape.size());
        if (__builtin_add_overflow(rows, ops.back().rows, &rows) ||
            __builtin_add_overflow(cols, ops.back().cols, &cols))
            throw nb::value_error("array is too big");
    }
    // the broadcast batch shape (right-aligned, size-1 axes stretch), as Batch's
    std::vector<std::int64_t> bshape(nd, 1);
    for (const KronOperand& o : ops) {
        const std::size_t lead = nd - o.bshape.size();
        for (std::size_t d = 0; d < o.bshape.size(); ++d) {
            std::int64_t& r = bshape[lead + d];
            if (o.bshape[d] == r || o.bshape[d] == 1) continue;
            if (r == 1) {
                r = o.bshape[d];
                continue;
            }
            std::string msg = "block_diag(): batch shapes do not broadcast:";
            for (const KronOperand& q : ops) msg += " " + shape_str(q.bshape);
            throw nb::value_error(msg.c_str());
        }
    }
    // each operand's byte stride per broadcast batch axis (0 where it broadcasts)
    std::vector<std::int64_t> bstride(ops.size() * nd, 0);
    for (std::size_t j = 0; j < ops.size(); ++j) {
        const std::size_t lead = nd - ops[j].bshape.size();
        for (std::size_t d = 0; d < ops[j].bshape.size(); ++d)
            if (ops[j].bshape[d] != 1) bstride[j * nd + lead + d] = ops[j].bstrides[d];
    }
    const std::vector<std::int64_t> zero(nd, 0);
    Batch b({{bshape, zero}}, "block_diag");  // walks the batch (and checks its size)
    Out out = alloc_matrices(b.shape(), rows, cols, dt, true);  // zeros: only the blocks are written
    const std::int64_t block = rows * cols;
    // each operand's first row and column in the result
    std::vector<std::int64_t> r0s(ops.size()), c0s(ops.size());
    for (std::size_t j = 0, r0 = 0, c0 = 0; j < ops.size(); ++j) {
        r0s[j] = static_cast<std::int64_t>(r0);
        c0s[j] = static_cast<std::int64_t>(c0);
        r0 += static_cast<std::size_t>(ops[j].rows);
        c0 += static_cast<std::size_t>(ops[j].cols);
    }
    visit_storage(dt, [&]<class R>(std::type_identity<R>) {
        R* res = out.as<R>();
        // Columns [cb, ce) of operand j's block of result element o; its zeros are already
        // there (a large result is calloc's fresh zero pages, a smaller one is filled per
        // element). Large float64 and complex128 blocks are copied by pack (C-order sources
        // in tiles, docs/design/linalg.md §5.3); small ones by the element loop (pack's call
        // and dispatch cost more there).
        auto columns = [&](R* o, std::size_t j, const std::byte* p, std::int64_t cb, std::int64_t ce) {
            const KronOperand& op = ops[j];
            const std::int64_t r0 = r0s[j], c0 = c0s[j];
            visit_storage(in[j].dtype(), [&]<class S>(std::type_identity<S>) {
                if constexpr (!(std::is_same_v<S, cplx> && !std::is_same_v<R, cplx>)) {
                    if constexpr (std::is_same_v<R, double> || std::is_same_v<R, cplx>) {
                        if (op.rows * (ce - cb) >= 4096) {
                            pack(
                                in[j].dtype(), p + cb * op.cs, op.rows, ce - cb, op.rs, op.cs,
                                o + (c0 + cb) * rows + r0, rows, false
                            );
                            return 0;
                        }
                    }
                    for (std::int64_t c = cb; c < ce; ++c) {
                        R* dst = o + (c0 + c) * rows + r0;
                        for (std::int64_t r = 0; r < op.rows; ++r)
                            dst[r] = convert<R>(load<S>(p + r * op.rs + c * op.cs));
                    }
                }
                return 0;
            });
        };
        auto source = [&](std::int64_t i, std::size_t j) {
            std::int64_t coord[kMaxNdim];
            for (std::int64_t d = static_cast<std::int64_t>(nd) - 1, t = i; d >= 0; --d) {
                coord[d] = t % bshape[d];
                t /= bshape[d];
            }
            std::int64_t off = 0;
            for (std::size_t d = 0; d < nd; ++d) off += coord[d] * bstride[j * nd + d];
            return in[j].data() + off;
        };
        // One large result: its columns on the pool (copies only, so the same result).
        constexpr std::int64_t kStripCols = 64;
        if (b.size() == 1 && block >= (1 << 18) && cols > kStripCols && detail::begin_parallel()) {
            const std::int64_t ntasks = (cols + kStripCols - 1) / kStripCols;
            detail::run_tasks(ntasks, [&](std::int64_t t) {
                const std::int64_t C0 = t * kStripCols, C1 = std::min(cols, C0 + kStripCols);
                if (!out.zeroed) std::fill(res + C0 * rows, res + C1 * rows, R(0));
                for (std::size_t j = 0; j < ops.size(); ++j) {
                    const std::int64_t cb = std::max<std::int64_t>(0, C0 - c0s[j]);
                    const std::int64_t ce = std::min<std::int64_t>(ops[j].cols, C1 - c0s[j]);
                    if (cb < ce) columns(res, j, source(0, j), cb, ce);
                }
            });
            return 0;
        }
        run_batch(b, cost(kCopy, double(block)), [&](std::int64_t i, const std::int64_t*, Scratch&) {
            R* o = res + i * block;
            if (!out.zeroed) std::fill(o, o + block, R(0));
            for (std::size_t j = 0; j < ops.size(); ++j) columns(o, j, source(i, j), 0, ops[j].cols);
            return 0;
        });
        return 0;
    });
    return to_array(out);
}

// -------------------------------------------------------------- kronsum --

// kron(I_n, A) + kron(B, I_m) for A (..., m, m) and B (..., n, n): the (n, n) grid of m x m
// blocks with A on the diagonal blocks and b_ij on the diagonal of block (i, j)
// (sd.sparse.kronsum's sum, docs/api/linalg.md §2). Each entry is a, b or the one sum a + b
// (a_pp + b_ii on the main diagonal; no products with the identity, so NaN and inf stay
// where they are); the rest is +0. Computed as kron: float64 for bool and int64.
nb::object kronsum(nb::handle x1o, nb::handle x2o) {
    Input x(x1o, "kronsum", "x1");
    Input y(x2o, "kronsum", "x2");
    require_square(x);
    require_square(y);
    const std::int64_t m = x.shape(-1), n = y.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}, {y.batch_shape(2), y.batch_strides(2)}}, "kronsum");
    std::int64_t size;
    if (__builtin_mul_overflow(m, n, &size)) throw nb::value_error("array is too big");
    const bool complex = computes_complex(x.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        // zeros to start with when large (calloc), else written around the nonzero entries
        Out out = alloc_matrices(b.shape(), size, size, dtype_of<T>(), true);
        T* res = out.as<T>();
        const std::int64_t block = size * size;
        // column J = j*m + q (column-major): A's column q in block row j, then b_ij at row
        // i*m + q of every block row i (added to a_qq on block row j)
        auto columns = [&](T* o, const T* a, const T* c, std::int64_t J0, std::int64_t J1) {
            for (std::int64_t J = J0; J < J1; ++J) {
                const std::int64_t j = J / m, q = J % m;
                T* col = o + J * size;
                if (!out.zeroed) std::fill(col, col + size, T(0));  // one fill: cheaper than two around A's column
                std::copy(a + q * m, a + (q + 1) * m, col + j * m);
                const T* cj = c + j * n;
                for (std::int64_t r = 0; r < n; ++r) {
                    if (r == j) col[r * m + q] = col[r * m + q] + cj[r];  // kron(I, A) + kron(B, I)
                    else col[r * m + q] = cj[r];
                }
            }
        };
        run_batch(b, cost(kCopy, double(block)), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* a = s.get<T>(static_cast<std::size_t>(m * m), 0);
            T* c = s.get<T>(static_cast<std::size_t>(n * n), 1);
            pack(x.dtype(), x.data() + off[0], m, m, x.stride(-2), x.stride(-1), a, m, false);
            pack(y.dtype(), y.data() + off[1], n, n, y.stride(-2), y.stride(-1), c, n, false);
            T* o = res + i * block;
            // one large result: its columns on the pool (copies only, so the same result)
            constexpr std::int64_t kCols = 64;
            if (b.size() == 1 && block >= (1 << 18) && size > kCols && detail::begin_parallel()) {
                detail::run_tasks((size + kCols - 1) / kCols, [&](std::int64_t t) {
                    columns(o, a, c, t * kCols, std::min(size, (t + 1) * kCols));
                });
            } else {
                columns(o, a, c, 0, size);
            }
            return 0;
        });
        return to_array(out);
    });
}

// ------------------------------------------------------------ bandwidth --

// The lower and upper bandwidth of each matrix: the farthest nonzero diagonal below and
// above the main one, 0 when none (scipy.linalg.bandwidth; docs/api/linalg.md §2). Nonzero
// is != 0: NaN counts, -0.0 does not; a complex entry counts when either part does. Only
// the entries that could widen the band found so far are read, along the memory's
// faster axis.
template <class S>
inline bool nonzero_at(const std::byte* p) {
    const S v = load<S>(p);
    if constexpr (std::is_same_v<S, cplx>) return v.real() != 0 || v.imag() != 0;
    else return v != S(0);
}

template <class S>
std::pair<std::int64_t, std::int64_t> band_of(
    const std::byte* p, std::int64_t m, std::int64_t n, std::int64_t rs, std::int64_t cs
) {
    std::int64_t lower = 0, upper = 0;
    auto at = [&](std::int64_t r, std::int64_t c) { return nonzero_at<S>(p + r * rs + c * cs); };
    if ((rs < 0 ? -rs : rs) <= (cs < 0 ? -cs : cs)) {  // columns are contiguous-ish: scan down columns
        for (std::int64_t c = 0; c < n; ++c) {
            for (std::int64_t r = m - 1; r > c + lower; --r)  // below: from the bottom up
                if (at(r, c)) {
                    lower = r - c;
                    break;
                }
            for (std::int64_t r = 0; r < std::min(m, c - upper); ++r)  // above: from the top down
                if (at(r, c)) {
                    upper = c - r;
                    break;
                }
        }
    } else {  // rows are: scan along rows
        for (std::int64_t r = 0; r < m; ++r) {
            for (std::int64_t c = 0; c < std::min(n, r - lower); ++c)  // below: from the left
                if (at(r, c)) {
                    lower = r - c;
                    break;
                }
            for (std::int64_t c = n - 1; c > r + upper; --c)  // above: from the right
                if (at(r, c)) {
                    upper = c - r;
                    break;
                }
        }
    }
    return {lower, upper};
}

nb::object bandwidth(nb::handle xo) {
    Input x(xo, "bandwidth", "x");
    require_matrix(x);
    const std::int64_t m = x.shape(-2), n = x.shape(-1), rs = x.stride(-2), cs = x.stride(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}}, "bandwidth");
    return visit_storage(x.dtype(), [&]<class S>(std::type_identity<S>) -> nb::object {
        if (b.ndim() == 0) {
            const auto [lo, up] = band_of<S>(x.data(), m, n, rs, cs);
            const nb::object l = nb::int_(lo), u = nb::int_(up);
            return make_result(Result::Bandwidth, {l, u});
        }
        Out lo = alloc_c(b.shape(), DType::Int64), up = alloc_c(b.shape(), DType::Int64);
        std::int64_t *pl = lo.as<std::int64_t>(), *pu = up.as<std::int64_t>();
        run_batch(b, cost(kStream, double(m) * n), [&](std::int64_t i, const std::int64_t* off, Scratch&) {
            const auto [l, u] = band_of<S>(x.data() + off[0], m, n, rs, cs);
            pl[i] = l;
            pu[i] = u;
            return 0;
        });
        const nb::object l = to_array(lo), u = to_array(up);
        return make_result(Result::Bandwidth, {l, u});
    });
}

}  // namespace

void bind_basic(nb::module_& m) {
    m.def("kron", named("kron", &kron), nb::sig("def kron(x1, x2, /)"), nb::arg().none(), nb::arg().none());
    m.def(
        "cross", named("cross", &cross), nb::sig("def cross(x1, x2, /, *, axis=-1)"), nb::arg().none(),
        nb::arg().none(), nb::kw_only(), nb::arg("axis").none() = -1
    );
    m.def(
        "vecdot", named("vecdot", &vecdot), nb::sig("def vecdot(x1, x2, /, *, axis=-1)"), nb::arg().none(),
        nb::arg().none(), nb::kw_only(), nb::arg("axis").none() = -1
    );
    m.def("outer", named("outer", &outer), nb::sig("def outer(x1, x2, /)"), nb::arg().none(), nb::arg().none());
    m.def(
        "trace", named("trace", &trace), nb::sig("def trace(x, /, *, k=0, dtype=None)"), nb::arg().none(),
        nb::kw_only(), nb::arg("k").none() = 0, nb::arg("dtype").none() = nb::none()
    );
    m.def(
        "diagonal", named("diagonal", &diagonal), nb::sig("def diagonal(x, /, *, k=0)"), nb::arg().none(),
        nb::kw_only(), nb::arg("k").none() = 0
    );
    m.def(
        "matrix_power", named("matrix_power", &matrix_power), nb::sig("def matrix_power(x, n, /)"), nb::arg().none(),
        nb::arg().none()
    );
    m.def("block_diag", named("block_diag", &block_diag), nb::sig("def block_diag(arrays, /)"), nb::arg().none());
    m.def("kronsum", named("kronsum", &kronsum), nb::sig("def kronsum(x1, x2, /)"), nb::arg().none(), nb::arg().none());
    m.def("bandwidth", named("bandwidth", &bandwidth), nb::sig("def bandwidth(x, /)"), nb::arg().none());
}

}  // namespace sd::linalg
