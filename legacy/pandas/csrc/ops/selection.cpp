#include "ops/selection.hpp"

#include <nanobind/nanobind.h>

#include <cstring>
#include <string>

namespace nb = nanobind;

namespace sd {

namespace {

const std::int64_t* positions_of(const Column& positions) {
    if (positions.dtype != DType::Int64 || positions.null_count != 0)
        throw nb::type_error("positions must be int64 without NA");
    return positions.data<std::int64_t>();
}

template <class T>
void gather(const Column& col, const std::int64_t* pos, std::int64_t n, Column& out, std::uint8_t* bits) {
    const T* in = col.data<T>();
    T* o = reinterpret_cast<T*>(out.values->data());
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t p = pos[i];
        if (p < 0) {
            o[i] = T{};
            bitmap::clear(bits, i);
        } else {
            o[i] = in[p];
            if (bits && !col.is_valid(p)) bitmap::clear(bits, i);
        }
    }
}

}  // namespace

Column take(const Column& col, const Column& positions) {
    const std::int64_t* pos = positions_of(positions);
    const std::int64_t n = positions.length;
    bool any_missing = false;
    for (std::int64_t i = 0; i < n; ++i) {
        if (pos[i] < -1 || pos[i] >= col.length) throw nb::index_error("take: position out of range");
        any_missing |= pos[i] == -1;
    }
    Column out;
    out.dtype = col.dtype;
    out.length = n;
    std::uint8_t* bits = nullptr;
    auto make_bits = [&] {
        if (col.validity || any_missing) bits = add_validity(out);
    };
    if (col.dtype == DType::String) {
        out.offsets = Buffer::allocate((n + 1) * 8, /*zeroed=*/false);
        make_bits();
        auto* offs = reinterpret_cast<std::int64_t*>(out.offsets->data());
        offs[0] = 0;
        for (std::int64_t i = 0; i < n; ++i)
            offs[i + 1] = offs[i] + (pos[i] < 0 ? 0 : static_cast<std::int64_t>(col.string_at(pos[i]).size()));
        out.values = Buffer::allocate(offs[n], /*zeroed=*/false);
        auto* bytes = reinterpret_cast<char*>(out.values->data());
        for (std::int64_t i = 0; i < n; ++i) {
            if (pos[i] < 0) {
                bitmap::clear(bits, i);
                continue;
            }
            std::string_view sv = col.string_at(pos[i]);
            std::memcpy(bytes + offs[i], sv.data(), sv.size());
            if (bits && !col.is_valid(pos[i])) bitmap::clear(bits, i);
        }
        out.null_count = bits ? n - bitmap::count_set(bits, 0, n) : 0;
        if (out.null_count == 0) out.validity.reset();
        return out;
    }
    out.values = Buffer::allocate(n * item_size(col.dtype), /*zeroed=*/false);
    make_bits();
    switch (col.dtype) {
    case DType::Bool: gather<std::uint8_t>(col, pos, n, out, bits); break;
    case DType::Int64: gather<std::int64_t>(col, pos, n, out, bits); break;
    case DType::Float64: gather<double>(col, pos, n, out, bits); break;
    case DType::String: break;
    }
    finish_fixed(out);
    return out;
}

Column nonzero(const Column& mask) {
    if (mask.dtype != DType::Bool) throw nb::type_error("a mask must be bool");
    const auto* m = mask.data<std::uint8_t>();
    std::int64_t count = 0;
    for (std::int64_t i = 0; i < mask.length; ++i) count += m[i] && mask.is_valid(i);
    Column out = make_fixed(DType::Int64, count);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    std::int64_t k = 0;
    for (std::int64_t i = 0; i < mask.length; ++i)
        if (m[i] && mask.is_valid(i)) o[k++] = i;  // NA rows hold 0 anyway (ARCHITECTURE.md §2.4)
    return out;
}

Column isna(const Column& col, bool want_na) {
    Column out = make_fixed(DType::Bool, col.length);
    auto* o = reinterpret_cast<std::uint8_t*>(out.values->data());
    const std::uint8_t flip = want_na ? 1 : 0;  // isna is 1 - valid, notna is valid
    auto bits = rebased_validity(col);
    if (!bits) {
        std::memset(o, flip ^ 1, static_cast<std::size_t>(col.length));
        return out;
    }
    const auto* b = reinterpret_cast<const std::uint8_t*>(bits->data());
    for (std::int64_t i = 0; i < col.length; ++i) o[i] = static_cast<std::uint8_t>(bitmap::get(b, i) ^ flip);
    return out;
}

Column fill_na(const Column& col, const Column& value) {
    if (value.dtype != col.dtype || value.length != 1 || value.null_count != 0)
        throw nb::type_error("fill value must be one value of the column's dtype");
    if (col.null_count == 0) return col;
    if (col.dtype == DType::String) {
        const std::string_view fill = value.string_at(0);
        Column out;
        out.dtype = DType::String;
        out.length = col.length;
        out.offsets = Buffer::allocate((col.length + 1) * 8, /*zeroed=*/false);
        auto* offs = reinterpret_cast<std::int64_t*>(out.offsets->data());
        offs[0] = 0;
        for (std::int64_t i = 0; i < col.length; ++i)
            offs[i + 1] = offs[i] + static_cast<std::int64_t>(col.is_valid(i) ? col.string_at(i).size() : fill.size());
        out.values = Buffer::allocate(offs[col.length], /*zeroed=*/false);
        auto* bytes = reinterpret_cast<char*>(out.values->data());
        for (std::int64_t i = 0; i < col.length; ++i) {
            std::string_view sv = col.is_valid(i) ? col.string_at(i) : fill;
            std::memcpy(bytes + offs[i], sv.data(), sv.size());
        }
        return out;
    }
    const std::int64_t size = item_size(col.dtype);
    Column out = make_fixed(col.dtype, col.length);
    std::byte* o = out.values->data();
    std::memcpy(o, col.values->data() + col.offset * size, static_cast<std::size_t>(col.length * size));
    for (std::int64_t i = 0; i < col.length; ++i)
        if (!col.is_valid(i)) std::memcpy(o + i * size, value.values->data() + value.offset * size, static_cast<std::size_t>(size));
    return out;
}

bool equals(const Column& a, const Column& b) {
    if (a.dtype != b.dtype || a.length != b.length || a.null_count != b.null_count) return false;
    for (std::int64_t i = 0; i < a.length; ++i) {
        if (a.is_valid(i) != b.is_valid(i)) return false;
        if (!a.is_valid(i)) continue;
        if (a.dtype == DType::String) {
            if (a.string_at(i) != b.string_at(i)) return false;
        } else {
            const std::int64_t size = item_size(a.dtype);
            if (std::memcmp(a.values->data() + (a.offset + i) * size, b.values->data() + (b.offset + i) * size,
                            static_cast<std::size_t>(size)) != 0)
                return false;
        }
    }
    return true;
}

Column arange(std::int64_t start, std::int64_t step, std::int64_t n) {
    Column out = make_fixed(DType::Int64, n);
    auto* o = reinterpret_cast<std::int64_t*>(out.values->data());
    for (std::int64_t i = 0; i < n; ++i) o[i] = start + i * step;
    return out;
}

}  // namespace sd
