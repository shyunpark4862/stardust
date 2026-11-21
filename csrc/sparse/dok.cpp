// The dictionary of keys of sd.sparse's dok_array (docs/design/sparse.md §2.1, §3.1): a hash map
// from (i, j) to a value, open addressing with linear probing, keys and values in flat arrays.
// `A[i, j] = v` is O(1); converting to the other formats sorts the keys, so the result does not
// depend on the order the entries went in. A 1-D array uses the keys (i, 0).
//
//   _DokMap(dtype, shape)       an empty map (dtype an sd dtype object, shape a tuple of 1 or 2)
//   m.get(key) / m.set(key, v)  one element: key an int per axis (a tuple for 2-D, negative
//                               counts from the end); get gives the stored value or a zero of
//                               the dtype, set stores v by the array's assignment rules (an
//                               implicit downcast is a TypeError) and erases the key when v == 0
//                               (scipy's dok_array)
//   m.nnz, m.shape, m.dtype     stored entries, the shape, the dtype object
//   m.copy(), m.transposed()    new maps
//   m.astype(dtype)             a new map, every value converted explicitly (the caller checks
//                               that the conversion is allowed)
//   m.resize(shape)             drops the keys outside the new shape, in place
//   m.to_coo()                  (major or None, minor, data) in row-major order (new arrays;
//                               indices read-only)
//   m.set_many(major, minor, values)
//                               sets each (major[k], minor[k]) in turn (a repeat takes its last
//                               value): values of the map's dtype, a zero erases the key; major
//                               None for 1-D
//   _DokMap.from_coo(dtype, shape, major, minor, data)
//                               a map holding the given entries (distinct positions), stored
//                               zeros included
//
// Errors in get and set are Python's protocol form (IndexError, TypeError), without a name.
#include "array/convert.hpp"
#include "array/ops/copy.hpp"
#include "bind/pyutil.hpp"
#include "parallel/pool.hpp"
#include "sparse/format.hpp"
#include "sparse/parallel.hpp"

#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace sd::sparse {

namespace {

using std::int64_t;

constexpr int64_t kEmpty = std::numeric_limits<int64_t>::min();
constexpr int64_t kGone = kEmpty + 1;

inline std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

std::vector<int64_t> shape_arg(nb::handle shape) {
    std::vector<int64_t> s;
    for (nb::handle v : shape) s.push_back(nb::cast<int64_t>(v));
    if (s.empty() || s.size() > 2) throw nb::value_error("_DokMap: 1-D or 2-D only");
    return s;
}

DType dtype_arg(nb::handle o) {
    const std::optional<DType> d = parse_dtype(o);
    if (!d) throw nb::type_error("_DokMap: dtype required");
    return *d;
}

class DokMap {
public:
    DokMap(nb::handle dtype, nb::handle shape) : DokMap(dtype_arg(dtype), shape_arg(shape)) {}
    DokMap(DType dtype, std::vector<int64_t> shape) : dtype_(dtype), shape_(std::move(shape)) {
        size_ = itemsize(dtype_);
    }

    DType dtype() const { return dtype_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t nnz() const { return count_; }

    nb::object get(nb::handle key) const {
        int64_t i, j;
        parse_key(key, i, j);
        const int64_t s = find(i, j);
        if (s < 0) return visit_dtype(dtype_, [&]<class T>(std::type_identity<T>) { return to_python(T(0)); });
        return load_python(dtype_, val_.data() + s * size_);
    }

    void set(nb::handle key, nb::handle value) {
        int64_t i, j;
        parse_key(key, i, j);
        const std::optional<DType> kind = scalar_dtype(value);
        if (!kind)
            throw nb::type_error(("only a bool, int, float or complex can be assigned to a single element, not " +
                                  py_type_name(value))
                                     .c_str());
        check_assignable(*kind, dtype_);
        visit_dtype(dtype_, [&]<class T>(std::type_identity<T>) {
            const T v = scalar_from_python<T>(value, *kind);
            if (!is_nonzero(v)) {  // 0 and -0.0 erase the key (scipy: `if x`); NaN is stored
                erase(i, j);
                return;
            }
            std::memcpy(slot_for(i, j), &v, sizeof(T));
        });
    }

    void set_many(const int64_t* major, const int64_t* minor, const NDArray& values) {
        const int64_t n = values.shape[0];
        visit_dtype(dtype_, [&]<class T>(std::type_identity<T>) {
            const T* v = typed<T>(values);
            for (int64_t k = 0; k < n; ++k) {
                const int64_t i = major ? major[k] : minor[k], j = major ? minor[k] : 0;
                if (!is_nonzero(v[k])) erase(i, j);
                else std::memcpy(slot_for(i, j), &v[k], sizeof(T));
            }
        });
    }

    DokMap copy() const { return *this; }

    DokMap transposed() const {
        DokMap out(dtype_, shape_.size() == 2 ? std::vector<int64_t>{shape_[1], shape_[0]} : shape_);
        out.reserve(count_);
        for (int64_t s = 0; s < cap_; ++s)
            if (ki_[s] >= 0) std::memcpy(out.slot_for(kj_[s], ki_[s]), val_.data() + s * size_, size_);
        return out;
    }

    DokMap astype(DType to) const {
        DokMap out(to, shape_);
        out.reserve(count_);
        visit_dtype(dtype_, [&]<class F>(std::type_identity<F>) {
            visit_dtype(to, [&]<class T>(std::type_identity<T>) {
                for (int64_t s = 0; s < cap_; ++s)
                    if (ki_[s] >= 0) {
                        F f;
                        std::memcpy(&f, val_.data() + s * size_, sizeof(F));
                        const T t = cast_value<T>(f);
                        std::memcpy(out.slot_for(ki_[s], kj_[s]), &t, sizeof(T));
                    }
            });
        });
        return out;
    }

    void resize(std::vector<int64_t> shape) {
        const int64_t rows = shape.size() == 2 ? shape[0] : std::numeric_limits<int64_t>::max();
        const int64_t cols = shape.back();
        const bool two = shape.size() == 2;
        for (int64_t s = 0; s < cap_; ++s)
            if (ki_[s] >= 0 && (two ? (ki_[s] >= rows || kj_[s] >= cols) : ki_[s] >= cols)) {
                ki_[s] = kGone;
                --count_;
            }
        shape_ = std::move(shape);
    }

    nb::tuple to_coo() const {
        const int64_t n = count_;
        NDArray minor = new_index(n);
        const bool two = shape_.size() == 2;
        NDArray major = two ? new_index(n) : NDArray();
        const int64_t shp[] = {n};
        NDArray data = new_array(dtype_, shp);
        {
            const parallel::Unlocked unlocked(n >= kParallelThreshold);
            std::vector<int64_t> order;
            order.reserve(static_cast<std::size_t>(n));
            for (int64_t s = 0; s < cap_; ++s)
                if (ki_[s] >= 0) order.push_back(s);
            std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                return ki_[a] < ki_[b] || (ki_[a] == ki_[b] && kj_[a] < kj_[b]);
            });
            int64_t* mi = i64(minor);
            int64_t* ma = two ? i64(major) : nullptr;
            for (int64_t k = 0; k < n; ++k) {
                const int64_t s = order[static_cast<std::size_t>(k)];
                if (two) {
                    ma[k] = ki_[s];
                    mi[k] = kj_[s];
                } else {
                    mi[k] = ki_[s];
                }
                std::memcpy(data.data + k * size_, val_.data() + s * size_, size_);
            }
        }
        minor.writeable = false;
        nb::object rows = nb::none();
        if (two) {
            major.writeable = false;
            rows = wrap(std::move(major));
        }
        return nb::make_tuple(rows, wrap(std::move(minor)), wrap(std::move(data)));
    }

    static DokMap from_coo(
        DType dtype, std::vector<int64_t> shape, const int64_t* major, const int64_t* minor, const NDArray& data
    ) {
        DokMap out(dtype, std::move(shape));
        const int64_t n = data.shape[0];
        out.reserve(n);
        for (int64_t k = 0; k < n; ++k)
            std::memcpy(
                out.slot_for(major ? major[k] : minor[k], major ? minor[k] : 0), data.data + k * out.size_, out.size_
            );
        return out;
    }

private:
    DType dtype_;
    std::vector<int64_t> shape_;
    int64_t size_ = 8;      // bytes per value
    int64_t count_ = 0;     // live keys
    int64_t used_ = 0;      // live and erased slots
    int64_t cap_ = 0;       // slots, a power of two (or 0)
    std::vector<int64_t> ki_, kj_;
    std::vector<std::byte> val_;

    int64_t home(int64_t i, int64_t j) const {
        return static_cast<int64_t>(
            mix(static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL + static_cast<std::uint64_t>(j)) &
            static_cast<std::uint64_t>(cap_ - 1)
        );
    }

    int64_t find(int64_t i, int64_t j) const {
        if (cap_ == 0) return -1;
        for (int64_t s = home(i, j);; s = (s + 1) & (cap_ - 1)) {
            if (ki_[s] == kEmpty) return -1;
            if (ki_[s] == i && kj_[s] == j) return s;
        }
    }

    void erase(int64_t i, int64_t j) {
        const int64_t s = find(i, j);
        if (s < 0) return;
        ki_[s] = kGone;
        --count_;
    }

    void reserve(int64_t n) {
        if ((used_ + n) * 2 >= cap_) rehash(n + count_);
    }

    // Room for `n` live keys at a load of at most 1/2; erased slots are dropped.
    void rehash(int64_t n) {
        int64_t cap = 16;
        while (cap < 2 * n + 2) cap *= 2;
        std::vector<int64_t> ki(static_cast<std::size_t>(cap), kEmpty), kj(static_cast<std::size_t>(cap));
        std::vector<std::byte> val(static_cast<std::size_t>(cap * size_));
        std::swap(ki, ki_);
        std::swap(kj, kj_);
        std::swap(val, val_);
        const int64_t old = cap_;
        cap_ = cap;
        used_ = count_;
        for (int64_t s = 0; s < old; ++s)
            if (ki[s] >= 0) {
                int64_t t = home(ki[s], kj[s]);
                while (ki_[t] != kEmpty) t = (t + 1) & (cap_ - 1);
                ki_[t] = ki[s];
                kj_[t] = kj[s];
                std::memcpy(val_.data() + t * size_, val.data() + s * size_, size_);
            }
    }

    // The value slot of (i, j), inserted if absent.
    std::byte* slot_for(int64_t i, int64_t j) {
        if ((used_ + 1) * 2 >= cap_) rehash(count_ + 1);
        int64_t gone = -1;
        for (int64_t s = home(i, j);; s = (s + 1) & (cap_ - 1)) {
            if (ki_[s] == kEmpty) {
                if (gone < 0) {
                    gone = s;
                    ++used_;
                }
                ki_[gone] = i;
                kj_[gone] = j;
                ++count_;
                return val_.data() + gone * size_;
            }
            if (ki_[s] == kGone) {
                if (gone < 0) gone = s;
            } else if (ki_[s] == i && kj_[s] == j) {
                return val_.data() + s * size_;
            }
        }
    }

    int64_t index(nb::handle o, int axis) const {
        const int64_t n = shape_[static_cast<std::size_t>(axis)];
        int64_t v = 0;
        const IndexResult r = index_value(o, v);
        if (r == IndexResult::NotInt)
            throw nb::index_error(
                ("a dok_array element is indexed by one int per axis, not " + py_type_name(o)).c_str()
            );
        if (r == IndexResult::Overflow || v < -n || v >= n)
            throw nb::index_error(("index " + nb::cast<std::string>(nb::str(o)) + " is out of bounds for axis " +
                                   std::to_string(axis) + " with size " + std::to_string(n))
                                      .c_str());
        return v < 0 ? v + n : v;
    }

    void parse_key(nb::handle key, int64_t& i, int64_t& j) const {
        const std::size_t nd = shape_.size();
        if (PyTuple_Check(key.ptr())) {
            const Py_ssize_t n = PyTuple_GET_SIZE(key.ptr());
            if (static_cast<std::size_t>(n) != nd)
                throw nb::index_error(("a " + std::to_string(nd) + "-D dok_array element is indexed by " +
                                       std::to_string(nd) + " int" + (nd == 1 ? "" : "s") + ", got " +
                                       std::to_string(n))
                                          .c_str());
            i = index(nb::handle(PyTuple_GET_ITEM(key.ptr(), 0)), 0);
            j = nd == 2 ? index(nb::handle(PyTuple_GET_ITEM(key.ptr(), 1)), 1) : 0;
            return;
        }
        if (nd != 1)
            throw nb::index_error(("a 2-D dok_array element is indexed by 2 ints, not " + py_type_name(key)).c_str());
        i = index(key, 0);
        j = 0;
    }
};

}  // namespace

void bind_dok(nb::module_& m) {
    nb::class_<DokMap>(m, "_DokMap")
        .def(nb::init<nb::handle, nb::handle>(), nb::arg("dtype"), nb::arg("shape"))
        .def("get", &DokMap::get)
        .def("set", &DokMap::set)
        .def_prop_ro("nnz", &DokMap::nnz)
        .def_prop_ro("dtype", [](const DokMap& d) { return nb::borrow(dtype_object(d.dtype())); })
        .def_prop_ro(
            "shape",
            [](const DokMap& d) {
                nb::list l;
                for (int64_t v : d.shape()) l.append(v);
                return nb::tuple(l);
            }
        )
        .def(
            "set_many",
            [](DokMap& d, nb::handle major_h, nb::handle minor_h, nb::handle values_h) {
                const NDArray& minor = vector_arg(minor_h, "_DokMap.set_many", DType::Int64);
                const NDArray& values = vector_arg(values_h, "_DokMap.set_many", d.dtype());
                if (values.shape[0] != minor.shape[0]) throw nb::value_error("_DokMap.set_many(): lengths differ");
                const int64_t* major = nullptr;
                if (!major_h.is_none()) {
                    const NDArray& ma = vector_arg(major_h, "_DokMap.set_many", DType::Int64);
                    if (ma.shape[0] != minor.shape[0]) throw nb::value_error("_DokMap.set_many(): lengths differ");
                    major = i64(ma);
                }
                d.set_many(major, i64(minor), values);
            },
            nb::arg("major").none(), nb::arg("minor"), nb::arg("values")
        )
        .def("copy", &DokMap::copy)
        .def("transposed", &DokMap::transposed)
        .def("astype", [](const DokMap& d, nb::handle dtype) { return d.astype(dtype_arg(dtype)); })
        .def("resize", [](DokMap& d, nb::handle shape) { d.resize(shape_arg(shape)); })
        .def("to_coo", &DokMap::to_coo)
        .def_static(
            "from_coo",
            [](nb::handle dtype, nb::handle shape, nb::handle major_h, nb::handle minor_h, nb::handle data_h) {
                const NDArray& minor = vector_arg(minor_h, "_DokMap.from_coo", DType::Int64);
                const NDArray& data = vector_arg(data_h, "_DokMap.from_coo");
                const DType d = dtype_arg(dtype);
                if (data.dtype != d || data.shape[0] != minor.shape[0])
                    throw nb::value_error("_DokMap.from_coo(): data does not match");
                const int64_t* major = nullptr;
                if (!major_h.is_none()) {
                    const NDArray& ma = vector_arg(major_h, "_DokMap.from_coo", DType::Int64);
                    if (ma.shape[0] != minor.shape[0]) throw nb::value_error("_DokMap.from_coo(): lengths differ");
                    major = i64(ma);
                }
                return DokMap::from_coo(d, shape_arg(shape), major, i64(minor), data);
            },
            nb::arg("dtype"), nb::arg("shape"), nb::arg("major").none(), nb::arg("minor"), nb::arg("data")
        );
}

}  // namespace sd::sparse
