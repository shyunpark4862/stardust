// Basic operations: vecdot, outer, trace, diagonal, matrix_power, kron, cross (API.md §1; ROADMAP.md M2, M10).
//
// None of these calls LAPACK, so NaN and inf pass through unchecked
// (DESIGN.md §2.4) -- except matrix_power with n < 0, which inverts through
// getrf and so checks like inv (DESIGN.md §2.11).
#include "linalg/ops/basic.hpp"

#include "linalg/args.hpp"
#include "linalg/cost.hpp"
#include "linalg/outblock.hpp"
#include "linalg/pairwise.hpp"
#include "linalg/ops/common.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace sd::linalg {

namespace {

// Shape and strides of an operand padded on the left to `nd` axes (size 1, stride 0).
struct Padded {
    std::vector<Py_ssize_t> shape, strides;
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

template <class T>
T conj_if_complex(T v) {
    if constexpr (std::is_same_v<T, cplx>) return std::conj(v);
    else return v;
}

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
                    // sum_i conj(x1[i]) * x2[i]  (x1 is conjugated: Array API), the
                    // products summed pairwise (M8): the bits of sdnp.sum(conj(x1) * x2).
                    auto dot = [&](const std::int64_t* off) {
                        const std::byte* p = x.data() + off[0];
                        const std::byte* q = y.data() + off[1];
                        auto fill = [&](std::int64_t b0, std::int64_t k, T* out) {
                            const std::byte* pp = p + b0 * sx;
                            const std::byte* qq = q + b0 * sy;
                            if (sx == static_cast<std::int64_t>(sizeof(SX)) && sy == static_cast<std::int64_t>(sizeof(SY))) {
                                for (std::int64_t i = 0; i < k; ++i)
                                    out[i] = mul(conj_if_complex(convert<T>(load<SX>(pp + i * static_cast<std::int64_t>(sizeof(SX))))),
                                                 convert<T>(load<SY>(qq + i * static_cast<std::int64_t>(sizeof(SY)))));
                            } else {
                                for (std::int64_t i = 0; i < k; ++i)
                                    out[i] = mul(conj_if_complex(convert<T>(load<SX>(pp + i * sx))), convert<T>(load<SY>(qq + i * sy)));
                            }
                        };
                        return pairwise_sum_mt<T>(len, fill);  // on the pool when long (M9)
                    };
                    if (b.ndim() == 0) {
                        std::int64_t off[2] = {0, 0};
                        return py_scalar(dot(off));
                    }
                    Out out = alloc_c(b.shape(), dtype_of<T>());
                    T* res = out.as<T>();
                    run_batch(b, cost(kStream, double(len)), [&](std::int64_t i, const std::int64_t* off, Scratch&) {
                        res[i] = dot(off);
                        return 0;
                    });
                    return to_sdnp(out);
                }
            });
        });
    });
}

// ---------------------------------------------------------------- outer --

nb::object outer(nb::handle x1o, nb::handle x2o) {
    Input x(x1o, "outer", "x1");
    Input y(x2o, "outer", "x2");
    // (..., M) and (..., N) -> (..., M, N); batch axes broadcast (API.md §1, S10).
    const std::int64_t m = x.shape(-1), n = y.shape(-1), sx = x.stride(-1), sy = y.stride(-1);
    Batch b({{x.batch_shape(1), x.batch_strides(1)}, {y.batch_shape(1), y.batch_strides(1)}}, "outer");
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
                            for (std::int64_t r = 0; r < m; ++r) o[c * m + r] = mul(xv[r], yc);  // as sdnp's x * y
                        }
                        return 0;
                    });
                }
            });
        });
        return to_sdnp(out);
    });
}

// -------------------------------------------------------- trace, diagonal --

// The k-th diagonal of an m x n matrix: its length and the (row, col) where it starts.
struct Diag {
    std::int64_t len, r0, c0;
    Diag(std::int64_t m, std::int64_t n, std::int64_t k) {
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
    // dtype= is an explicit conversion, as in sdnp.sum (DESIGN.md §2.11; stardust-numpy API §6).
    DType acc_dt = computes_complex(x.dtype()) ? DType::Complex128 : DType::Float64;
    if (want) {
        if (*want == DType::Bool) throw nb::type_error("trace() cannot accumulate in bool; use any() or all()");
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
                        std::uint64_t acc = 0;  // wraps on overflow, as stardust-numpy's int64 sum
                        for (std::int64_t i = 0; i < dg.len; ++i)
                            acc += static_cast<std::uint64_t>(convert<T>(load<S>(p + i * step)));
                        return static_cast<std::int64_t>(acc);
                    } else {  // pairwise (M8): the bits of sdnp.sum(diagonal(x))
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
                return to_sdnp(out);
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
    // Extraction, not computation: the input dtype is kept (DESIGN.md §2.11). A copy (S10).
    Out out = alloc_vectors(b.shape(), dg.len, x.dtype());
    // a typed strided copy (M8: a memcpy call per element cost 2-3x numpy)
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
    return to_sdnp(out);
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
        const Failure f = run_batch(b, cost(kLU, double(m) * m), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            T* o = res + i * m * m;
            if (m == 0) return 0;
            if (e == 0) {  // x^0 = I, whatever x holds
                std::fill(o, o + mm, T(0));
                for (lint d = 0; d < m; ++d) o[d * m + d] = T(1);
                return 0;
            }
            T* base = s.get<T>(mm, 1);
            if (n < 0) {
                // x^-e = (x^-1)^e. The inverse goes through LAPACK, so it is checked like inv.
                T* lu;
                lint* ipiv;
                if (const int code = lu_factor<T>(x, off[0], m, s, lu, ipiv)) return code;
                lu_inverse(lu, ipiv, m, base);
            } else {
                pack(x.dtype(), x.data() + off[0], m, m, x.stride(-2), x.stride(-1), base, m, false);
            }
            // Binary powering, most significant bit first: r = base; for each further bit,
            // r = r*r, then r = r*base if the bit is set. log2(e) squarings + popcount(e)-1 products.
            // The products ping-pong between two scratch buffers and the last one goes
            // straight to the output slot (M8: no copies of r).
            int top = 63;
            while (!((e >> top) & 1)) --top;
            if (top == 0) {  // x^1 and x^-1: base itself
                std::copy(base, base + mm, o);
                return 0;
            }
            const int total = top + std::popcount(e) - 1;  // gemm calls
            LapackSlot<T> os(o, static_cast<std::int64_t>(mm), s, Scratch::kAlign0);
            T* buf[2] = {s.get<T>(mm, 2), s.get<T>(mm, 3)};
            const T* r = base;
            int done = 0, next = 0;
            auto product = [&](const T* lhs, const T* rhs) {
                T* dst = ++done == total ? os.get() : buf[next];
                lapack::gemm_square(m, lhs, rhs, dst);
                next ^= 1;
                r = dst;
            };
            for (int bit = top - 1; bit >= 0; --bit) {
                product(r, r);
                if ((e >> bit) & 1) product(r, base);
            }
            return 0;
        }, n < 0 ? lu_task_cap(m, m, true) : 0);  // x^-1 is getrf + getrs(I)
        if (f) raise(b, f, "matrix_power", "x");
        return to_sdnp(out);
    });
}

// ----------------------------------------------------------------- kron --

nb::object kron(nb::handle x1o, nb::handle x2o) {
    Input x(x1o, "kron", "x1");
    Input y(x2o, "kron", "x2");
    require_matrix(x);
    require_matrix(y);
    // (..., M, N) (x) (..., P, Q) -> (..., M*P, N*Q); the batch axes broadcast like
    // every other matrix function's (M10, user decision; np.kron differs for ndim > 2).
    const std::int64_t m = x.shape(-2), n = x.shape(-1), p = y.shape(-2), q = y.shape(-1);
    Batch b({{x.batch_shape(2), x.batch_strides(2)}, {y.batch_shape(2), y.batch_strides(2)}}, "kron");
    const bool complex = computes_complex(x.dtype()) || computes_complex(y.dtype());
    return with_type(complex, [&]<class T>(std::type_identity<T>) -> nb::object {
        Out out = alloc_matrices(b.shape(), m * p, n * q, dtype_of<T>());
        T* res = out.as<T>();
        const std::int64_t rows = m * p, block = rows * n * q;
        run_batch(b, cost(kCopy, double(block)), [&](std::int64_t i, const std::int64_t* off, Scratch& s) {
            // both operands converted into scratch once, column-major; NaN passes through
            T* a = s.get<T>(static_cast<std::size_t>(m * n), 0);
            T* c = s.get<T>(static_cast<std::size_t>(p * q), 1);
            pack(x.dtype(), x.data() + off[0], m, n, x.stride(-2), x.stride(-1), a, m, false);
            pack(y.dtype(), y.data() + off[1], p, q, y.stride(-2), y.stride(-1), c, p, false);
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
        return to_sdnp(out);
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
        return to_sdnp(out);
    });
}

}  // namespace

void bind_basic(nb::module_& m) {
    m.def("kron", &kron, nb::arg().none(), nb::arg().none());
    m.def("cross", &cross, nb::arg().none(), nb::arg().none(), nb::kw_only(), nb::arg("axis") = -1);
    m.def("vecdot", &vecdot, nb::arg().none(), nb::arg().none(), nb::kw_only(), nb::arg("axis") = -1);
    m.def("outer", &outer, nb::arg().none(), nb::arg().none());
    m.def("trace", &trace, nb::arg().none(), nb::kw_only(), nb::arg("k") = 0, nb::arg("dtype").none() = nb::none());
    m.def("diagonal", &diagonal, nb::arg().none(), nb::kw_only(), nb::arg("k") = 0);
    m.def("matrix_power", &matrix_power, nb::arg().none(), nb::arg());
}

}  // namespace sd::linalg
