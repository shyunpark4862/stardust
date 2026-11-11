// Python entry point of the native core (`stardust_pandas._core`).

#include "core/column.hpp"
#include "core/convert.hpp"
#include "io/csv.hpp"
#include "ops/cast.hpp"
#include "ops/elementwise.hpp"
#include "ops/groupby.hpp"
#include "ops/hash.hpp"
#include "ops/join.hpp"
#include "ops/reduce.hpp"
#include "ops/selection.hpp"
#include "ops/sort.hpp"
#include "parallel/pool.hpp"

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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nb = nanobind;

namespace {

nb::dict build_info() {
    nb::dict info;
    info["compiler"] = std::string(__VERSION__);
    info["cplusplus"] = static_cast<long>(__cplusplus);
#if defined(__ARM_NEON)
    info["neon"] = true;
#else
    info["neon"] = false;
#endif
    return info;
}

// M0 check of the copied pool: task t writes t * t into its own slot, so a
// task that runs twice or never shows up in the result. The result must not
// depend on the thread count.
std::vector<std::int64_t> pool_smoke_test(std::int64_t ntasks) {
    std::vector<std::int64_t> out(static_cast<std::size_t>(ntasks));
    sd::parallel::parallel_for(ntasks, [&](std::int64_t t) { out[static_cast<std::size_t>(t)] = t * t; });
    return out;
}

void bind_column(nb::module_& m) {
    using sd::Column;
    // A value type: returning one moves a few shared_ptrs into the new Python
    // object, which is cheaper than a shared_ptr holder (ARCHITECTURE.md §1.1).
    nb::class_<Column>(m, "Column")
        .def_prop_ro("dtype", [](const Column& c) { return sd::dtype_name(c.dtype); })
        .def_prop_ro("null_count", [](const Column& c) { return c.null_count; })
        .def("__len__", [](const Column& c) { return c.length; })
        .def("slice", &Column::slice, nb::arg("start"), nb::arg("length"))
        .def("convert", [](const Column& c, std::string_view dtype) { return c.convert(sd::parse_dtype(dtype)); },
             nb::arg("dtype"))
        .def("to_list", &sd::to_list, nb::arg("na").none())
        .def("to_numpy_view", &sd::to_numpy_view)
        .def("validity_mask", &sd::validity_mask)
        // For tests: whether two columns share their values Buffer (zero-copy).
        .def("_shares_values_with", [](const Column& a, const Column& b) { return a.values == b.values; })
        .def_static(
            "from_list",
            [](nb::handle seq, nb::handle na, std::optional<std::string_view> dtype) {
                std::optional<sd::DType> dt;
                if (dtype) dt = sd::parse_dtype(*dtype);
                return sd::from_list(seq, na, dt);
            },
            nb::arg("seq"), nb::arg("na").none(), nb::arg("dtype").none())
        .def_static("from_numpy", &sd::from_numpy, nb::arg("arr"))
        .def("astype", [](const Column& c, std::string_view dtype) { return sd::astype(c, sd::parse_dtype(dtype)); },
             nb::arg("dtype"))
        .def("isna", [](const Column& c) { return sd::isna(c, true); })
        .def("notna", [](const Column& c) { return sd::isna(c, false); })
        .def("fill_na", &sd::fill_na, nb::arg("value"))
        .def("take", &sd::take, nb::arg("positions"))
        .def("nonzero", &sd::nonzero)
        .def("filter", &sd::filter, nb::arg("mask"))
        .def("equals", &sd::equals, nb::arg("other"))
        // A new Column object sharing this one's Buffers. Writers take their
        // own object first, so a write never shows through another object
        // that held the same Column (the Buffers themselves are Copy-on-Write).
        .def("share", [](const Column& c) { return c; })
        .def("set_rows", &sd::set_rows, nb::arg("positions"), nb::arg("values"))
        .def("contains_key", &sd::contains_key, nb::arg("value"));
    nb::class_<sd::LabelTable>(m, "LabelTable")
        .def(nb::init<const Column&>(), nb::arg("labels"))
        .def_prop_ro("is_unique", &sd::LabelTable::is_unique)
        .def("lookup", &sd::LabelTable::lookup, nb::arg("targets"))
        .def("first_positions", &sd::LabelTable::first_positions, nb::arg("targets"))
        .def("all_positions", &sd::LabelTable::all_positions, nb::arg("targets"));
    m.def("concat", &sd::concat, nb::arg("cols"));
    m.def("reduce", &sd::reduce, nb::arg("op"), nb::arg("col"), nb::arg("skipna"), nb::arg("ddof"));
    nb::class_<sd::Groups>(m, "Groups")
        .def(nb::init<const Column&, std::int64_t>(), nb::arg("codes"), nb::arg("ngroups"))
        .def("sizes", &sd::Groups::sizes)
        .def("reduce", &sd::Groups::reduce, nb::arg("op"), nb::arg("col"), nb::arg("skipna"), nb::arg("ddof"));
    m.def("sort_groups", &sd::sort_groups, nb::arg("keys"));
    m.def("argsort", &sd::argsort, nb::arg("keys"), nb::arg("ascending"), nb::arg("na_last"));
    m.def("join_indices", &sd::join_indices, nb::arg("left"), nb::arg("right"), nb::arg("how"));
    m.def(
        "read_csv",
        [](nb::bytes data, std::string_view sep, bool header, std::vector<std::string> names,
           std::vector<std::string> usecols, std::unordered_map<std::string, std::string> dtypes,
           std::optional<std::string_view> all_dtype, std::vector<std::string> na_values, std::int64_t nrows) {
            sd::CsvOptions opt;
            opt.sep = sep[0];
            opt.header = header;
            opt.names = std::move(names);
            opt.usecols = std::move(usecols);
            for (const auto& [k, v] : dtypes) opt.dtypes.emplace(k, sd::parse_dtype(v));
            if (all_dtype) opt.all_dtype = sd::parse_dtype(*all_dtype);
            opt.na_values = std::move(na_values);
            opt.nrows = nrows;
            return sd::read_csv(data.c_str(), data.size(), opt);
        },
        nb::arg("data"), nb::arg("sep"), nb::arg("header"), nb::arg("names"), nb::arg("usecols"), nb::arg("dtypes"),
        nb::arg("all_dtype").none(), nb::arg("na_values"), nb::arg("nrows"));
    m.def("binary_op", &sd::binary_op, nb::arg("op"), nb::arg("a"), nb::arg("b"), nb::arg("scalar_side"));
    m.def("unary_op", &sd::unary_op, nb::arg("op"), nb::arg("a"));
    m.def("arange", &sd::arange, nb::arg("start"), nb::arg("step"), nb::arg("n"));
    // (codes, first, ngroups) of the rows of `cols` taken together.
    m.def("factorize_rows", [](const std::vector<sd::Column>& cols) {
        sd::Factorized f = sd::factorize_rows(cols);
        return std::make_tuple(f.codes, f.first, f.ngroups);
    }, nb::arg("cols"));
    m.def("group_counts", &sd::group_counts, nb::arg("codes"), nb::arg("ngroups"));
    m.def("stable_order", &sd::stable_order, nb::arg("counts"), nb::arg("ascending"));
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "stardust-pandas native core";
    m.def("_build_info", &build_info, "Toolchain facts captured at compile time.");
    m.def("_pool_smoke_test", &pool_smoke_test, nb::arg("ntasks"));
    // Results never depend on the thread count (DESIGN.md §2.6).
    m.def("set_num_threads", [](std::int64_t n) {
        if (n < 1 || n > 1024) throw nb::value_error("the number of threads must be between 1 and 1024");
        sd::parallel::set_num_threads(static_cast<int>(n));
    }, nb::arg("n"));
    m.def("get_num_threads", &sd::parallel::num_threads);
    bind_column(m);
}
