// The frame area's bindings: the submodule stardust._core.frame.

#include "bind/frame_bind.hpp"

#include "bind/errors.hpp"
#include "frame/column.hpp"
#include "frame/convert.hpp"
#include "frame/interop.hpp"
#include "frame/io/csv.hpp"
#include "frame/ops/cast.hpp"
#include "frame/ops/elementwise.hpp"
#include "frame/ops/groupby.hpp"
#include "frame/ops/hash.hpp"
#include "frame/ops/join.hpp"
#include "frame/ops/reduce.hpp"
#include "frame/ops/selection.hpp"
#include "frame/ops/sort.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nb = nanobind;

namespace {

// The optional `func` of a kernel the Python layer calls for a public function:
// that function's name, put in front of the kernel's error messages
// (bind/errors.hpp). None for operators and indexing, which keep Python's form.
using Func = std::optional<std::string>;

const char* name_of(const Func& func) { return func ? func->c_str() : nullptr; }

template <class F>
decltype(auto) named_call(const Func& func, F&& f) {
    return sd::call_named(name_of(func), std::forward<F>(f));
}

const auto kFunc = nb::arg("func").none() = nb::none();

nb::bytes bytes_of(const std::byte* p, std::int64_t n) { return nb::bytes(reinterpret_cast<const char*>(p), static_cast<std::size_t>(n)); }

std::shared_ptr<sd::Buffer> buffer_of(const nb::bytes& b) {
    auto buf = sd::Buffer::allocate(static_cast<std::int64_t>(b.size()), /*zeroed=*/false);
    std::memcpy(buf->data(), b.c_str(), b.size());
    return buf;
}

// (dtype, length, null_count, values, validity or None, offsets or None):
// the rows [offset, offset + length) rebased to 0.
nb::tuple column_state(const sd::frame::Column& c) {
    nb::object validity = nb::none(), offsets = nb::none();
    if (auto v = sd::frame::rebased_validity(c)) validity = bytes_of(v->data(), v->size());
    nb::bytes values;
    if (c.dtype == sd::DType::String) {
        const std::int64_t* o = c.string_offsets();
        std::vector<std::int64_t> rebased(static_cast<std::size_t>(c.length) + 1);
        for (std::int64_t i = 0; i <= c.length; ++i) rebased[static_cast<std::size_t>(i)] = o[i] - o[0];
        offsets = bytes_of(reinterpret_cast<const std::byte*>(rebased.data()), (c.length + 1) * 8);
        values = bytes_of(c.values->data() + o[0], o[c.length] - o[0]);
    } else {
        const std::int64_t size = sd::itemsize(c.dtype);
        values = bytes_of(c.values->data() + c.offset * size, c.length * size);
    }
    return nb::make_tuple(sd::dtype_name(c.dtype), c.length, c.null_count, values, validity, offsets);
}

sd::frame::Column column_from_state(const nb::tuple& state) {
    if (state.size() != 6) throw nb::value_error("not a pickled Column");
    sd::frame::Column c;
    c.dtype = sd::frame::parse_dtype(nb::cast<std::string_view>(state[0]));
    c.length = nb::cast<std::int64_t>(state[1]);
    c.null_count = nb::cast<std::int64_t>(state[2]);
    c.values = buffer_of(nb::cast<nb::bytes>(state[3]));
    if (!state[4].is_none()) c.validity = buffer_of(nb::cast<nb::bytes>(state[4]));
    if (!state[5].is_none()) c.offsets = buffer_of(nb::cast<nb::bytes>(state[5]));
    return c;
}

void bind_column(nb::module_& m) {
    using sd::frame::Column;
    // A value type: returning one moves a few shared_ptrs into the new Python
    // object, which is cheaper than a shared_ptr holder (docs/design/frame.md §3.2).
    nb::class_<Column>(m, "Column")
        .def_prop_ro("dtype", [](const Column& c) { return sd::dtype_name(c.dtype); })
        .def_prop_ro("null_count", [](const Column& c) { return c.null_count; })
        .def("__len__", [](const Column& c) { return c.length; })
        .def("slice", &Column::slice, nb::arg("start"), nb::arg("length"))
        .def("convert", [](const Column& c, std::string_view dtype) { return c.convert(sd::frame::parse_dtype(dtype)); },
             nb::arg("dtype"))
        .def("to_list", &sd::frame::to_list, nb::arg("na").none())
        .def("to_numpy_view", &sd::frame::to_numpy_view)
        .def("to_array_view", &sd::frame::to_array_view, nb::arg("writeable"))
        .def_static("from_array", [](nb::handle arr, const Func& func) {
            return named_call(func, [&] { return sd::frame::from_array(arr); });
        }, nb::arg("arr"), kFunc)
        .def("validity_mask", &sd::frame::validity_mask)
        // For tests: whether two columns share their values Buffer (zero-copy).
        .def("_shares_values_with", [](const Column& a, const Column& b) { return a.values == b.values; })
        .def_static(
            "from_list",
            [](nb::handle seq, nb::handle na, std::optional<std::string_view> dtype, const Func& func) {
                return named_call(func, [&] {
                    std::optional<sd::DType> dt;
                    if (dtype) dt = sd::frame::parse_dtype(*dtype);
                    return sd::frame::from_list(seq, na, dt);
                });
            },
            nb::arg("seq"), nb::arg("na").none(), nb::arg("dtype").none(), kFunc)
        .def_static("from_numpy", [](const nb::ndarray<nb::ro, nb::ndim<1>, nb::device::cpu>& arr, const Func& func) {
            return named_call(func, [&] { return sd::frame::from_numpy(arr); });
        }, nb::arg("arr"), kFunc)
        .def("astype", [](const Column& c, std::string_view dtype, const Func& func) {
            return named_call(func, [&] { return sd::frame::astype(c, sd::frame::parse_dtype(dtype)); });
        }, nb::arg("dtype"), kFunc)
        .def("isna", [](const Column& c) { return sd::frame::isna(c, true); })
        .def("notna", [](const Column& c) { return sd::frame::isna(c, false); })
        .def("fill_na", &sd::frame::fill_na, nb::arg("value"))
        .def("take", &sd::frame::take, nb::arg("positions"))
        .def("nonzero", &sd::frame::nonzero)
        .def("filter", &sd::frame::filter, nb::arg("mask"))
        .def("equals", &sd::frame::equals, nb::arg("other"))
        // A new Column object sharing this one's Buffers. Writers take their
        // own object first, so a write never shows through another object
        // that held the same Column (the Buffers themselves are Copy-on-Write).
        .def("share", [](const Column& c) { return c; })
        .def("set_rows", &sd::frame::set_rows, nb::arg("positions"), nb::arg("values"))
        .def("contains_key", &sd::frame::contains_key, nb::arg("value"))
        // Pickling (DESIGN.md §10.1 #12): the column's own rows as bytes, so a
        // slice pickles only its rows and the bits round-trip exactly.
        .def("__getstate__", &column_state)
        .def("__setstate__", [](Column& c, const nb::tuple& state) { new (&c) Column(column_from_state(state)); });
    nb::class_<sd::frame::LabelTable>(m, "LabelTable")
        .def(nb::init<const Column&>(), nb::arg("labels"))
        .def_prop_ro("is_unique", &sd::frame::LabelTable::is_unique)
        .def("lookup", &sd::frame::LabelTable::lookup, nb::arg("targets"))
        .def("first_positions", &sd::frame::LabelTable::first_positions, nb::arg("targets"))
        .def("all_positions", &sd::frame::LabelTable::all_positions, nb::arg("targets"));
    m.def("concat", &sd::frame::concat, nb::arg("cols"));
    // The common dtype of column dtypes by the frame's promotion
    // (promote_columns): a name, or None when a string meets another dtype.
    m.def("promote_types", [](const std::vector<std::string>& names) -> std::optional<std::string> {
        if (names.empty()) throw nb::value_error("promote_types needs at least one dtype");
        std::optional<sd::DType> d = sd::frame::parse_dtype(names[0]);
        for (std::size_t i = 1; i < names.size() && d; ++i) d = sd::frame::promote_columns(*d, sd::frame::parse_dtype(names[i]));
        if (!d) return std::nullopt;
        return std::string(sd::dtype_name(*d));
    }, nb::arg("dtypes"));
    m.def("columns_to_array", &sd::frame::columns_to_array, nb::arg("cols"), nb::arg("n"));
    m.def("reduce", [](std::string_view op, const Column& col, bool skipna, double correction, const Func& func) {
        return named_call(func, [&] { return sd::frame::reduce(op, col, skipna, correction); });
    }, nb::arg("op"), nb::arg("col"), nb::arg("skipna"), nb::arg("correction"), kFunc);
    nb::class_<sd::frame::Groups>(m, "Groups")
        .def(nb::init<const Column&, std::int64_t>(), nb::arg("codes"), nb::arg("ngroups"))
        .def("sizes", &sd::frame::Groups::sizes)
        .def("reduce", [](const sd::frame::Groups& g, std::string_view op, const Column& col, bool skipna, double correction,
                          const Func& func) {
            return named_call(func, [&] { return g.reduce(op, col, skipna, correction); });
        }, nb::arg("op"), nb::arg("col"), nb::arg("skipna"), nb::arg("correction"), kFunc);
    m.def("sort_groups", &sd::frame::sort_groups, nb::arg("keys"));
    m.def("argsort", &sd::frame::argsort, nb::arg("keys"), nb::arg("ascending"), nb::arg("na_last"));
    m.def("join_indices", [](const std::vector<Column>& left, const std::vector<Column>& right, std::string_view how,
                             const Func& func) {
        return named_call(func, [&] { return sd::frame::join_indices(left, right, how); });
    }, nb::arg("left"), nb::arg("right"), nb::arg("how"), kFunc);
    m.def(
        "read_csv",
        [](nb::bytes data, std::string_view sep, bool header, std::vector<std::string> names,
           std::vector<std::string> usecols, std::unordered_map<std::string, std::string> dtypes,
           std::optional<std::string_view> all_dtype, std::vector<std::string> na_values, std::int64_t nrows) {
            return sd::call_named("read_csv", [&] {
                sd::frame::CsvOptions opt;
                opt.sep = sep[0];
                opt.header = header;
                opt.names = std::move(names);
                opt.usecols = std::move(usecols);
                for (const auto& [k, v] : dtypes) opt.dtypes.emplace(k, sd::frame::parse_dtype(v));
                if (all_dtype) opt.all_dtype = sd::frame::parse_dtype(*all_dtype);
                opt.na_values = std::move(na_values);
                opt.nrows = nrows;
                return sd::frame::read_csv(data.c_str(), data.size(), opt);
            });
        },
        nb::arg("data"), nb::arg("sep"), nb::arg("header"), nb::arg("names"), nb::arg("usecols"), nb::arg("dtypes"),
        nb::arg("all_dtype").none(), nb::arg("na_values"), nb::arg("nrows"));
    m.def("binary_op", &sd::frame::binary_op, nb::arg("op"), nb::arg("a"), nb::arg("b"), nb::arg("scalar_side"));
    m.def("unary_op", &sd::frame::unary_op, nb::arg("op"), nb::arg("a"));
    m.def("isnan", &sd::frame::isnan, nb::arg("col"));
    m.def("arange", &sd::frame::arange, nb::arg("start"), nb::arg("step"), nb::arg("n"));
    // (codes, first, ngroups) of the rows of `cols` taken together.
    m.def("factorize_rows", [](const std::vector<sd::frame::Column>& cols) {
        sd::frame::Factorized f = sd::frame::factorize_rows(cols);
        return std::make_tuple(f.codes, f.first, f.ngroups);
    }, nb::arg("cols"));
    m.def("group_counts", &sd::frame::group_counts, nb::arg("codes"), nb::arg("ngroups"));
    m.def("stable_order", &sd::frame::stable_order, nb::arg("counts"), nb::arg("ascending"));
}

}  // namespace

namespace sd {

void bind_frame(nb::module_& core) {
    nb::module_ m = core.def_submodule("frame", "The frame area (sd.frame)");
    bind_column(m);
}

}  // namespace sd
