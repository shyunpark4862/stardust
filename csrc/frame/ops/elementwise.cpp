#include "frame/ops/elementwise.hpp"

#include "frame/ops/cast.hpp"

#include "frame/parallel.hpp"

#include "array/kernels/elementwise.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>
#include <cstring>
#include <string>

namespace nb = nanobind;

namespace sd::frame {

namespace {

enum class Op { Add, Sub, Mul, TrueDiv, FloorDiv, Mod, Pow, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, LShift, RShift };

struct OpInfo {
    std::string_view name;
    Op op;
    const char* symbol;
};

constexpr OpInfo kOps[] = {
    {"add", Op::Add, "+"},
    {"sub", Op::Sub, "-"},
    {"mul", Op::Mul, "*"},
    {"truediv", Op::TrueDiv, "/"},
    {"floordiv", Op::FloorDiv, "//"},
    {"mod", Op::Mod, "%"},
    {"pow", Op::Pow, "**"},
    {"eq", Op::Eq, "=="},
    {"ne", Op::Ne, "!="},
    {"lt", Op::Lt, "<"},
    {"le", Op::Le, "<="},
    {"gt", Op::Gt, ">"},
    {"ge", Op::Ge, ">="},
    {"and", Op::And, "&"},
    {"or", Op::Or, "|"},
    {"xor", Op::Xor, "^"},
    {"lshift", Op::LShift, "<<"},
    {"rshift", Op::RShift, ">>"},
};

const OpInfo& parse_op(std::string_view name) {
    for (const OpInfo& info : kOps)
        if (info.name == name) return info;
    throw nb::value_error(("unknown operator '" + std::string(name) + "'").c_str());
}

bool is_arith(Op op) { return op <= Op::Pow; }
bool is_compare(Op op) { return op >= Op::Eq && op <= Op::Ge; }
bool is_numeric(DType dt) { return dt == DType::Int64 || dt == DType::Float64; }

// One input of a kernel: a column, or a one-row scalar read at every row.
// The scalar case is a template parameter so the loop body has no branch.
template <class T, bool Scalar>
struct In {
    const T* p;
    T operator[](std::int64_t i) const {
        if constexpr (Scalar) return p[0];
        else return p[i];
    }
};

template <class A, class B, class F>
void visit_inputs(const Column& a, const Column& b, int scalar_side, F&& f) {
    const A* pa = a.data<A>();
    const B* pb = b.data<B>();
    if (scalar_side == 1) f(In<A, true>{pa}, In<B, false>{pb});
    else if (scalar_side == 2) f(In<A, false>{pa}, In<B, true>{pb});
    else f(In<A, false>{pa}, In<B, false>{pb});
}

// A fixed-width result R = f(x, y). With CanNA the kernel may also turn a
// row into NA: f(x, y, na) sets na (int // 0, int % 0, int ** negative).
template <class R, class A, class B, bool CanNA = false, class F>
Column map2(const Column& a, const Column& b, int scalar_side, DType result, F f) {
    const std::int64_t n = scalar_side == 1 ? b.length : a.length;
    Column out = make_fixed(result, n);
    out.validity = and_validity(a, b, scalar_side, n);
    R* o = reinterpret_cast<R*>(out.values->data());
    // Chunks are multiples of 64 rows, so tasks never share a bitmap byte.
    // Each task copies its inputs into locals: captures live in the closure
    // object, and a store through a uint8_t pointer may alias them, so they
    // would be reloaded every row and the loop would not vectorize (docs/design/frame.md §7.6).
    if constexpr (CanNA) {
        if (!out.validity) add_validity(out);
        auto* bits = reinterpret_cast<std::uint8_t*>(out.validity->data());
        visit_inputs<A, B>(a, b, scalar_side, [&](auto x, auto y) {
            for_chunks(n, [&, x, y](std::int64_t lo, std::int64_t hi) {
                const auto xx = x;  // locals, not captures: see the comment above
                const auto yy = y;
                R* oo = o;
                std::uint8_t* bb = bits;
                for (std::int64_t i = lo; i < hi; ++i) {
                    bool na = false;
                    oo[i] = f(xx[i], yy[i], na);
                    if (na) bitmap::clear(bb, i);
                }
            });
        });
    } else {
        visit_inputs<A, B>(a, b, scalar_side, [&](auto x, auto y) {
            for_chunks(n, [&, x, y](std::int64_t lo, std::int64_t hi) {
                const auto xx = x;  // locals, not captures: see the comment above
                const auto yy = y;
                R* oo = o;
                for (std::int64_t i = lo; i < hi; ++i) oo[i] = f(xx[i], yy[i]);
            });
        });
    }
    finish_fixed(out);
    return out;
}

// --- int64 arithmetic: overflow wraps (docs/api/frame.md §1.5), done in uint64 where
// signed overflow would be undefined behavior in C++. ---

std::int64_t wrap_add(std::int64_t x, std::int64_t y) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(x) + static_cast<std::uint64_t>(y));
}
std::int64_t wrap_sub(std::int64_t x, std::int64_t y) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(x) - static_cast<std::uint64_t>(y));
}
std::int64_t wrap_mul(std::int64_t x, std::int64_t y) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(x) * static_cast<std::uint64_t>(y));
}
std::int64_t wrap_neg(std::int64_t x) { return static_cast<std::int64_t>(0 - static_cast<std::uint64_t>(x)); }

// Floor division and its remainder (the result takes the divisor's sign),
// as in Python. x / -1 is handled apart: INT64_MIN / -1 overflows, which C++
// leaves undefined; wrapping gives INT64_MIN.
std::int64_t int_floordiv(std::int64_t x, std::int64_t y, bool& na) {
    if (y == 0) {
        na = true;
        return 0;
    }
    if (y == -1) return wrap_neg(x);
    std::int64_t q = x / y;
    if (x % y != 0 && ((x < 0) != (y < 0))) --q;
    return q;
}

std::int64_t int_mod(std::int64_t x, std::int64_t y, bool& na) {
    if (y == 0) {
        na = true;
        return 0;
    }
    if (y == -1) return 0;
    std::int64_t r = x % y;
    if (r != 0 && ((r < 0) != (y < 0))) r += y;
    return r;
}

// Exponentiation by squaring, wrapping. A negative exponent has no int
// result: NA (docs/api/frame.md §1.5).
std::int64_t int_pow(std::int64_t x, std::int64_t y, bool& na) {
    if (y < 0) {
        na = true;
        return 0;
    }
    std::uint64_t result = 1, base = static_cast<std::uint64_t>(x);
    for (std::uint64_t e = static_cast<std::uint64_t>(y); e != 0; e >>= 1) {
        if (e & 1) result *= base;
        base *= base;
    }
    return static_cast<std::int64_t>(result);
}

// --- float64 floor division and remainder: numpy's npy_divmod, so results
// match pandas bit for bit (including the signs of zeros). ---

double float_divmod(double a, double b, double* modulus) {
    double mod = std::fmod(a, b);
    if (b == 0.0) {  // b == 0: the quotient is a / b (inf or nan), fmod gives nan
        *modulus = mod;
        return a / b;
    }
    double div = (a - mod) / b;  // very nearly an integer multiple of b
    if (mod != 0.0) {
        if ((b < 0) != (mod < 0)) {  // Python's remainder takes the divisor's sign
            mod += b;
            div -= 1.0;
        }
    } else {
        mod = std::copysign(0.0, b);
    }
    double floordiv;
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;  // snap to the nearest integral value
    } else {
        floordiv = std::copysign(0.0, a / b);
    }
    *modulus = mod;
    return floordiv;
}

double float_floordiv(double a, double b) {
    double mod;
    return float_divmod(a, b, &mod);
}

double float_mod(double a, double b) {
    if (b == 0.0) return std::fmod(a, b);
    double mod;
    float_divmod(a, b, &mod);
    return mod;
}

// float64 ** (int64 is converted first): the array area's kernel (sd::pow_float_run: libm's
// pow, with the exact forms of the exponents 2, 0, 1, -1 and 0.5), so a Series and an array
// give the same bits (docs/api/frame.md §3.2). A scalar side is a stride-0 operand, as in the
// array.
template <class A, class B>
Column float_pow(const Column& a, const Column& b, int scalar_side) {
    const std::int64_t n = scalar_side == 1 ? b.length : a.length;
    Column out = make_fixed(DType::Float64, n);
    out.validity = and_validity(a, b, scalar_side, n);
    double* o = reinterpret_cast<double*>(out.values->data());
    const A* pa = a.data<A>();
    const B* pb = b.data<B>();
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        const std::int64_t len = hi - lo;
        std::vector<double> ta, tb;
        auto operand = [&](const auto* p, bool scalar, std::vector<double>& t, std::int64_t& stride) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
            stride = scalar ? 0 : static_cast<std::int64_t>(sizeof(double));
            if constexpr (std::is_same_v<T, double>) {
                if (scalar) return reinterpret_cast<const std::byte*>(p);
                return reinterpret_cast<const std::byte*>(p + lo);
            } else {
                t.resize(scalar ? 1 : static_cast<std::size_t>(len));
                for (std::size_t i = 0; i < t.size(); ++i) t[i] = static_cast<double>(p[scalar ? 0 : lo + i]);
                return reinterpret_cast<const std::byte*>(t.data());
            }
        };
        std::int64_t sa, sb;
        const std::byte* xa = operand(pa, scalar_side == 1, ta, sa);
        const std::byte* xb = operand(pb, scalar_side == 2, tb, sb);
        sd::pow_float_run(xa, sa, xb, sb, reinterpret_cast<std::byte*>(o + lo), sizeof(double), len);
    });
    finish_fixed(out);
    return out;
}

template <class A, class B>
Column float_arith(Op op, const Column& a, const Column& b, int side) {
    auto run = [&](auto f) { return map2<double, A, B>(a, b, side, DType::Float64, f); };
    auto d = [](auto v) { return static_cast<double>(v); };
    switch (op) {
        case Op::Add: return run([&](auto x, auto y) { return d(x) + d(y); });
        case Op::Sub: return run([&](auto x, auto y) { return d(x) - d(y); });
        case Op::Mul: return run([&](auto x, auto y) { return d(x) * d(y); });
        case Op::TrueDiv: return run([&](auto x, auto y) { return d(x) / d(y); });
        case Op::FloorDiv: return run([&](auto x, auto y) { return float_floordiv(d(x), d(y)); });
        case Op::Mod: return run([&](auto x, auto y) { return float_mod(d(x), d(y)); });
        case Op::Pow: return float_pow<A, B>(a, b, side);
        default: break;
    }
    throw nb::value_error("not an arithmetic operator");
}

Column int_arith(Op op, const Column& a, const Column& b, int side) {
    using I = std::int64_t;
    auto run = [&](auto f) { return map2<I, I, I>(a, b, side, DType::Int64, f); };
    auto run_na = [&](auto f) { return map2<I, I, I, true>(a, b, side, DType::Int64, f); };
    switch (op) {
        // Lambdas, not function pointers: the loop body is compiled once inside
        // a task, where a pointer would be an indirect call per row (docs/design/frame.md §7.6).
        case Op::Add: return run([](I x, I y) { return wrap_add(x, y); });
        case Op::Sub: return run([](I x, I y) { return wrap_sub(x, y); });
        case Op::Mul: return run([](I x, I y) { return wrap_mul(x, y); });
        case Op::TrueDiv: return float_arith<I, I>(op, a, b, side);  // int / int is float64 (docs/api/frame.md §1.5)
        case Op::FloorDiv: return run_na([](I x, I y, bool& na) { return int_floordiv(x, y, na); });
        case Op::Mod: return run_na([](I x, I y, bool& na) { return int_mod(x, y, na); });
        case Op::Pow: return run_na([](I x, I y, bool& na) { return int_pow(x, y, na); });
        default: break;
    }
    throw nb::value_error("not an arithmetic operator");
}

// --- comparison ---

constexpr bool from_three_way(Op op, int c) {
    switch (op) {
        case Op::Eq: return c == 0;
        case Op::Ne: return c != 0;  // NaN != anything is True
        case Op::Lt: return c == -1;
        case Op::Le: return c == -1 || c == 0;
        case Op::Gt: return c == 1;
        case Op::Ge: return c == 1 || c == 0;
        default: return false;
    }
}

// Calls f(std::integral_constant<Op, op>) so the comparison is fixed at
// compile time inside the row loop.
template <class F>
Column with_compare_op(Op op, F f) {
    switch (op) {
        case Op::Eq: return f(std::integral_constant<Op, Op::Eq>{});
        case Op::Ne: return f(std::integral_constant<Op, Op::Ne>{});
        case Op::Lt: return f(std::integral_constant<Op, Op::Lt>{});
        case Op::Le: return f(std::integral_constant<Op, Op::Le>{});
        case Op::Gt: return f(std::integral_constant<Op, Op::Gt>{});
        case Op::Ge: return f(std::integral_constant<Op, Op::Ge>{});
        default: break;
    }
    throw nb::value_error("not a comparison");
}

template <class A, class B>
Column compare_plain(Op op, const Column& a, const Column& b, int side) {
    auto run = [&](auto f) { return map2<std::uint8_t, A, B>(a, b, side, DType::Bool, f); };
    switch (op) {  // the C++ operators on doubles are IEEE 754 comparisons
        case Op::Eq: return run([](A x, B y) -> std::uint8_t { return x == y; });
        case Op::Ne: return run([](A x, B y) -> std::uint8_t { return x != y; });
        case Op::Lt: return run([](A x, B y) -> std::uint8_t { return x < y; });
        case Op::Le: return run([](A x, B y) -> std::uint8_t { return x <= y; });
        case Op::Gt: return run([](A x, B y) -> std::uint8_t { return x > y; });
        case Op::Ge: return run([](A x, B y) -> std::uint8_t { return x >= y; });
        default: break;
    }
    throw nb::value_error("not a comparison");
}

// int64 against float64: the int is converted to float64 first, as in the
// array area, numpy and pandas (docs/api/frame.md §3.3): 2**53 + 1 == 2.0**53.
Column compare_mixed(Op op, const Column& a, const Column& b, int side) {
    using I = std::int64_t;
    if (a.dtype == DType::Int64) return compare_plain<I, double>(op, a, b, side);
    return compare_plain<double, I>(op, a, b, side);
}

Column compare_strings(Op op, const Column& a, const Column& b, int side) {
    const std::int64_t n = side == 1 ? b.length : a.length;
    Column out = make_fixed(DType::Bool, n);
    out.validity = and_validity(a, b, side, n);
    auto* o = reinterpret_cast<std::uint8_t*>(out.values->data());
    for_chunks(
        n,
        [&](std::int64_t lo, std::int64_t hi) {
            for (std::int64_t i = lo; i < hi; ++i) {
                // char_traits<char>::compare compares as unsigned char, so this is
                // UTF-8 byte order, which is code point order (docs/api/frame.md §3.3).
                const int c = a.string_at(side == 1 ? 0 : i).compare(b.string_at(side == 2 ? 0 : i));
                o[i] = from_three_way(op, c < 0 ? -1 : c > 0 ? 1 : 0);
            }
        },
        4
    );
    finish_fixed(out);
    return out;
}

Column concat_strings(const Column& a, const Column& b, int side) {
    const std::int64_t n = side == 1 ? b.length : a.length;
    Column out;
    out.dtype = DType::String;
    out.length = n;
    out.validity = and_validity(a, b, side, n);
    const auto* bits = out.validity ? reinterpret_cast<const std::uint8_t*>(out.validity->data()) : nullptr;
    auto valid = [&](std::int64_t i) { return !bits || bitmap::get(bits, i); };
    auto size = [&](std::int64_t i) -> std::int64_t {
        if (!valid(i)) return 0;  // NA rows stay empty (docs/design/frame.md §4.5)
        return static_cast<std::int64_t>(a.string_at(side == 1 ? 0 : i).size() + b.string_at(side == 2 ? 0 : i).size());
    };
    // Two passes over the same chunks: bytes per chunk, then each chunk
    // writes from its start (the sum of the chunks before it, in order).
    std::vector<std::int64_t> start(static_cast<std::size_t>(num_chunks(n)) + 1, 0);
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t t = 0;
        for (std::int64_t i = lo; i < hi; ++i) t += size(i);
        start[static_cast<std::size_t>(lo / kChunkRows) + 1] = t;
    });
    for (std::size_t c = 1; c < start.size(); ++c) start[c] += start[c - 1];
    out.values = Buffer::allocate(start.back(), /*zeroed=*/false);
    out.offsets = Buffer::allocate((n + 1) * 8, /*zeroed=*/false);
    auto* offs = reinterpret_cast<std::int64_t*>(out.offsets->data());
    auto* bytes = reinterpret_cast<char*>(out.values->data());
    offs[0] = 0;
    for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
        std::int64_t pos = start[static_cast<std::size_t>(lo / kChunkRows)];
        for (std::int64_t i = lo; i < hi; ++i) {
            if (valid(i)) {
                for (std::string_view part : {a.string_at(side == 1 ? 0 : i), b.string_at(side == 2 ? 0 : i)}) {
                    std::memcpy(bytes + pos, part.data(), part.size());
                    pos += static_cast<std::int64_t>(part.size());
                }
            }
            offs[i + 1] = pos;
        }
    });
    out.null_count = bits ? n - bitmap::count_set(bits, 0, n) : 0;
    if (out.null_count == 0) out.validity.reset();
    return out;
}

// --- logic: Kleene three-valued & and | (docs/api/frame.md §1.4) ---

// Rows [lo, n) with lo a multiple of 8: bitmap bytes lo / 8 onward.
template <Op O, int Side>
void logic_rows(
    const std::uint8_t* x,
    const std::uint8_t* vx,
    const std::uint8_t* y,
    const std::uint8_t* vy,
    std::int64_t lo,
    std::int64_t n,
    std::uint8_t* o,
    std::uint8_t* bits
) {
    for (std::int64_t j = lo / 8; j < bitmap::num_bytes(n); ++j) {
        unsigned byte = 0;
        for (std::int64_t i = j * 8; i < std::min(n, j * 8 + 8); ++i) {
            const std::int64_t ia = Side == 1 ? 0 : i, ib = Side == 2 ? 0 : i;
            const unsigned a = x[ia], va = vx[ia], b = y[ib], vb = vy[ib];
            unsigned value, valid;
            if constexpr (O == Op::And) {  // False & anything is False
                value = a & b;
                valid = (va & vb) | (va & (a ^ 1)) | (vb & (b ^ 1));
            } else if constexpr (O == Op::Or) {  // True | anything is True
                value = a | b;
                valid = (va & vb) | (va & a) | (vb & b);
            } else {
                value = a ^ b;
                valid = va & vb;
            }
            o[i] = static_cast<std::uint8_t>(value & valid);  // NA rows hold 0 (docs/design/frame.md §4.5)
            byte |= valid << (i & 7);
        }
        bits[j] = static_cast<std::uint8_t>(byte);
    }
}

Column logic(Op op, const Column& a, const Column& b, int side) {
    const std::int64_t n = side == 1 ? b.length : a.length;
    Column out = make_fixed(DType::Bool, n);
    std::uint8_t* bits = add_validity(out);
    auto* o = reinterpret_cast<std::uint8_t*>(out.values->data());
    const std::vector<std::uint8_t> va = valid_bytes(a), vb = valid_bytes(b);
    const auto* x = a.data<std::uint8_t>();
    const auto* y = b.data<std::uint8_t>();
    auto run = [&](auto op_tag) {
        constexpr Op O = decltype(op_tag)::value;
        for_chunks(n, [&](std::int64_t lo, std::int64_t hi) {
            if (side == 1) logic_rows<O, 1>(x, va.data(), y, vb.data(), lo, hi, o, bits);
            else if (side == 2) logic_rows<O, 2>(x, va.data(), y, vb.data(), lo, hi, o, bits);
            else logic_rows<O, 0>(x, va.data(), y, vb.data(), lo, hi, o, bits);
        });
    };
    if (op == Op::And) run(std::integral_constant<Op, Op::And>{});
    else if (op == Op::Or) run(std::integral_constant<Op, Op::Or>{});
    else run(std::integral_constant<Op, Op::Xor>{});
    finish_fixed(out);
    return out;
}

// --- int64 bitwise operators: the array area's rules (docs/api/frame.md §3.4). The
// shift count is read as unsigned, so 64 or more (negative counts included)
// shifts every bit out: << gives 0, >> the sign (0 or -1). C++ leaves a
// shift by >= 64 undefined, hence the test. ---

Column int_bitwise(Op op, const Column& a, const Column& b, int side) {
    using I = std::int64_t;
    using U = std::uint64_t;
    auto run = [&](auto f) { return map2<I, I, I>(a, b, side, DType::Int64, f); };
    switch (op) {
        case Op::And: return run([](I x, I y) { return x & y; });
        case Op::Or: return run([](I x, I y) { return x | y; });
        case Op::Xor: return run([](I x, I y) { return x ^ y; });
        case Op::LShift:
            return run([](I x, I y) {
                return static_cast<U>(y) >= 64 ? I{0} : static_cast<I>(static_cast<U>(x) << y);
            });
        case Op::RShift:
            return run([](I x, I y) {
                return static_cast<U>(y) >= 64 ? (x < 0 ? I{-1} : I{0}) : x >> y;
            });  // arithmetic
        default: break;
    }
    throw nb::value_error("not a bitwise operator");
}

}  // namespace

Column binary_op(std::string_view name, const Column& a, const Column& b, int scalar_side) {
    const OpInfo& info = parse_op(name);
    const Op op = info.op;
    if (scalar_side == 0 && a.length != b.length) throw nb::value_error("operands must have the same length");
    const DType da = a.dtype, db = b.dtype;
    auto unsupported = [&](const char* why) -> Column {
        throw nb::type_error((std::string("unsupported operand dtypes for ") + info.symbol + ": " + dtype_name(da) +
                              " and " + dtype_name(db) + why)
                                 .c_str());
    };
    using I = std::int64_t;

    if (is_arith(op)) {
        if (da == DType::String || db == DType::String) {
            if (op == Op::Add && da == db) return concat_strings(a, b, scalar_side);
            return unsupported(" (strings support + and comparisons)");
        }
        // bool takes part as int64, as in the array area (docs/api/frame.md §3.2):
        // True + True == 2, and the int rules (NA for // 0) apply from there.
        if (da == DType::Bool || db == DType::Bool)
            return binary_op(
                name, da == DType::Bool ? astype(a, DType::Int64) : a, db == DType::Bool ? astype(b, DType::Int64) : b,
                scalar_side
            );
        if (da == DType::Int64 && db == DType::Int64) return int_arith(op, a, b, scalar_side);
        if (da == DType::Int64) return float_arith<I, double>(op, a, b, scalar_side);
        if (db == DType::Int64) return float_arith<double, I>(op, a, b, scalar_side);
        return float_arith<double, double>(op, a, b, scalar_side);
    }
    if (is_compare(op)) {
        // A bool meeting a number takes part as int64, as in the array area
        // (promote_columns, docs/api/frame.md §3.3): True == 1, False < 0.5.
        if ((da == DType::Bool && is_numeric(db)) || (db == DType::Bool && is_numeric(da)))
            return binary_op(
                name, da == DType::Bool ? astype(a, DType::Int64) : a, db == DType::Bool ? astype(b, DType::Int64) : b,
                scalar_side
            );
        if (is_numeric(da) && is_numeric(db)) {
            if (da == db)
                return da == DType::Int64 ? compare_plain<I, I>(op, a, b, scalar_side)
                                          : compare_plain<double, double>(op, a, b, scalar_side);
            return compare_mixed(op, a, b, scalar_side);
        }
        if (da == DType::Bool && db == DType::Bool)
            return compare_plain<std::uint8_t, std::uint8_t>(op, a, b, scalar_side);
        if (da == DType::String && db == DType::String) return compare_strings(op, a, b, scalar_side);
        return unsupported(" (these dtypes cannot be compared)");
    }
    // & | ^: Kleene logic on two bools (docs/api/frame.md §1.4); otherwise
    // bitwise on int64, a bool taking part as int64 as in the array area.
    // << >> take int64 only, as in the array area.
    const bool shift = op == Op::LShift || op == Op::RShift;
    if (shift) {
        if (da == DType::Int64 && db == DType::Int64) return int_bitwise(op, a, b, scalar_side);
        return unsupported(" (shifts take int64)");
    }
    if (da == DType::Bool && db == DType::Bool) return logic(op, a, b, scalar_side);
    auto int_like = [](DType dt) { return dt == DType::Int64 || dt == DType::Bool; };
    if (int_like(da) && int_like(db))
        return int_bitwise(
            op, da == DType::Bool ? astype(a, DType::Int64) : a, db == DType::Bool ? astype(b, DType::Int64) : b,
            scalar_side
        );
    return unsupported(" (bitwise operators take bool and int64)");
}

Column unary_op(std::string_view name, const Column& a) {
    auto map1 = [&](DType result, auto f) {
        Column out = make_fixed(result, a.length);
        out.validity = rebased_validity(a);
        using T = std::remove_cvref_t<decltype(f(0))>;  // input and result share a type here
        const T* x = a.data<T>();
        T* o = reinterpret_cast<T*>(out.values->data());
        for_chunks(a.length, [&, x, o](std::int64_t lo, std::int64_t hi) {
            const T* xx = x;  // locals, not captures: see map2
            T* oo = o;
            for (std::int64_t i = lo; i < hi; ++i) oo[i] = f(xx[i]);
        });
        finish_fixed(out);
        return out;
    };
    auto unsupported = [&]() -> Column {  // Python's form: -s, abs(s), ~s
        const char* symbol = name == "neg" ? "unary -" : name == "abs" ? "abs()" : "unary ~";
        throw nb::type_error(
            (std::string("unsupported operand dtype for ") + symbol + ": " + dtype_name(a.dtype)).c_str()
        );
    };
    if (name == "neg" || name == "abs") {
        if (a.dtype == DType::Bool) return unary_op(name, astype(a, DType::Int64));  // as in the array area
        const bool neg = name == "neg";
        if (a.dtype == DType::Int64)
            return map1(DType::Int64, [neg](std::int64_t x) { return neg || x < 0 ? wrap_neg(x) : x; });
        if (a.dtype == DType::Float64) return map1(DType::Float64, [neg](double x) { return neg ? -x : std::fabs(x); });
        return unsupported();
    }
    if (name == "invert") {
        if (a.dtype == DType::Bool) return map1(DType::Bool, [](std::uint8_t x) -> std::uint8_t { return !x; });
        if (a.dtype == DType::Int64)
            return map1(DType::Int64, [](std::int64_t x) { return ~x; });  // bitwise, as in the array area
        return unsupported();
    }
    throw nb::value_error(("unknown operator '" + std::string(name) + "'").c_str());
}

}  // namespace sd::frame
