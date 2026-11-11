#include "core/batch.hpp"

#include "core/pyrefs.hpp"
#include "parallel/pool.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>

namespace sdla {

namespace nb = nanobind;

namespace {

std::string shape_str(std::span<const Py_ssize_t> s) {
    std::string r = "(";
    for (std::size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + (s.size() == 1 ? ",)" : ")");
}

}  // namespace

Batch::Batch(std::initializer_list<BatchOperand> ops, const char* func) : nops_(ops.size()) {
    if (nops_ > kMaxOperands) throw std::logic_error("Batch: too many operands");
    std::size_t nd = 0;
    for (const auto& op : ops) nd = std::max(nd, op.shape.size());
    shape_.assign(nd, 1);
    for (const auto& op : ops) {
        const std::size_t lead = nd - op.shape.size();
        for (std::size_t d = 0; d < op.shape.size(); ++d) {
            const std::int64_t s = op.shape[d];
            std::int64_t& r = shape_[lead + d];
            if (s == r || s == 1) continue;
            if (r == 1) {
                r = s;
                continue;
            }
            std::string msg = std::string(func) + "(): batch shapes do not broadcast:";
            for (const auto& o : ops) msg += " " + shape_str(o.shape);
            throw nb::value_error(msg.c_str());
        }
    }
    strides_.assign(nops_ * nd, 0);
    std::size_t k = 0;
    for (const auto& op : ops) {
        const std::size_t lead = nd - op.shape.size();
        for (std::size_t d = 0; d < op.shape.size(); ++d)
            if (op.shape[d] != 1) strides_[k * nd + lead + d] = op.strides[d];
        ++k;
    }
    for (std::int64_t s : shape_) size_ *= s;
}

void Batch::offsets(std::int64_t idx, std::int64_t* out) const {
    const std::size_t nd = shape_.size();
    for (std::size_t k = 0; k < nops_; ++k) out[k] = 0;
    for (std::size_t d = nd; d-- > 0;) {
        const std::int64_t i = idx % shape_[d];
        idx /= shape_[d];
        for (std::size_t k = 0; k < nops_; ++k) out[k] += i * strides_[k * nd + d];
    }
}

std::string Batch::index_str(std::int64_t idx) const {
    std::vector<std::int64_t> ix(shape_.size());
    for (std::size_t d = shape_.size(); d-- > 0;) {
        ix[d] = idx % shape_[d];
        idx /= shape_[d];
    }
    std::string r = "(";
    for (std::size_t i = 0; i < ix.size(); ++i) r += (i ? ", " : "") + std::to_string(ix[i]);
    return r + (ix.size() == 1 ? ",)" : ")");
}

namespace detail {

bool begin_parallel() {
    if (!parallel::available_here()) return false;  // inside a task: no nesting
    // The thread count follows sdnp.get_num_threads() (DESIGN.md §2.5), read on
    // every call so sdnp.set_num_threads() applies at once (~7 ns, §2.9).
    PyObject* r = PyObject_CallNoArgs(refs().get_num_threads);
    if (!r) throw nb::python_error();
    const long want = PyLong_AsLong(r);
    Py_DECREF(r);
    if (want < 1) return false;
    if (want != parallel::num_threads()) parallel::set_num_threads(static_cast<int>(want));
    return parallel::available();
}

void run_tasks(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn) { parallel::parallel_for(ntasks, fn); }

}  // namespace detail

std::string failure_message(const Batch& b, std::int64_t idx, const char* what) {
    if (b.ndim() == 0) return what;
    return std::string(what) + " (at batch index " + b.index_str(idx) + ")";
}

}  // namespace sdla
