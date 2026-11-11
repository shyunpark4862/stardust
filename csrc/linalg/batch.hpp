// Batch axes: broadcasting them, walking them, and the batch driver every
// LAPACK function goes through (ARCHITECTURE.md §3.4).
#pragma once

#include "linalg/types.hpp"

#include <Python.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sd::linalg {

// One operand's batch part: its batch shape and byte strides.
struct BatchOperand {
    std::span<const Py_ssize_t> shape;
    std::span<const Py_ssize_t> strides;
};

class Batch {
public:
    // Broadcasts the operands' batch shapes (right-aligned, size-1 axes
    // stretch). ValueError if they do not broadcast.
    Batch(std::initializer_list<BatchOperand> ops, const char* func);

    const std::vector<std::int64_t>& shape() const { return shape_; }
    std::int64_t size() const { return size_; }
    std::size_t ndim() const { return shape_.size(); }

    // Byte offset of every operand for batch element `idx` (C order over shape()).
    void offsets(std::int64_t idx, std::int64_t* out) const;

    // "(1, 2)": `idx` as a batch index, for error messages.
    std::string index_str(std::int64_t idx) const;

private:
    std::vector<std::int64_t> shape_;
    std::int64_t size_ = 1;
    std::size_t nops_ = 0;
    std::vector<std::int64_t> strides_;  // nops_ x ndim, 0 on broadcast axes
};

// Workspace sizes from one LAPACK query (lwork = -1): the main workspace and,
// for the routines that have them, the real and integer workspaces.
struct WorkSizes {
    lint work = 1, rwork = 1, iwork = 1;
};

// Scratch memory for a run of batch elements. The driver makes one for a
// serial batch and one per task when it splits (run_batch), so a body may
// reuse it across the matrices it processes.
// Slots 0-3 (and int slot 0) belong to the function bodies; slots 4-5 (and
// int slot 1) to the LAPACK wrappers' workspaces (lapack.hpp); slots 6-8 to
// LapackSlot's stand-ins for misaligned output slots (common.hpp).
struct Scratch {
    static constexpr int kSlots = 9;
    static constexpr int kWork = 4, kRWork = 5, kIWork = 1;
    static constexpr int kAlign0 = 6, kAlign1 = 7, kAlign2 = 8;
    template <class T>
    T* get(std::size_t n, int slot = 0) {
        return reinterpret_cast<T*>(bytes_[slot].get(n * sizeof(T)));
    }
    lint* get_ints(std::size_t n, int slot = 0) { return reinterpret_cast<lint*>(ints_[slot].get(n * sizeof(lint))); }

    // LAPACK's workspace sizes, asked once per routine and argument set (M8,
    // ARCHITECTURE.md §3.4). Within a batch every matrix has the same shape, so
    // only the first one per Scratch queries; the sizes, and so the results' bits, are the
    // ones a per-call query gives (DESIGN.md §2.14: gesdd's path depends on lwork).
    using WorkKey = std::array<lint, 6>;  // routine id, its size arguments, the element size
    template <class Query>
    WorkSizes sizes(const WorkKey& key, Query&& query) {
        for (const auto& [k, v] : work_sizes_)
            if (k == key) return v;
        work_sizes_.emplace_back(key, query());
        return work_sizes_.back().second;
    }

private:
    // A growing buffer that is not zero-filled (M8): every user writes before
    // it reads, and std::vector's zero-fill of a large fresh block cost as much
    // as the packing itself.
    struct Buf {
        std::unique_ptr<std::byte[]> p;
        std::size_t n = 0;
        std::byte* get(std::size_t want) {
            if (want > n) {
                p.reset(new std::byte[want]);
                n = want;
            }
            return p.get();
        }
    };
    Buf bytes_[kSlots];  // independent areas; slot 0 is the default
    Buf ints_[2];
    std::vector<std::pair<WorkKey, WorkSizes>> work_sizes_;
};

// What went wrong with one batch element; code 0 means nothing did.
struct Failure {
    std::int64_t index = -1;
    int code = 0;
    explicit operator bool() const { return code != 0; }
};

// The batch driver. Calls body(idx, offsets, scratch) for each batch element;
// body returns 0 on success or a positive failure code of its own choosing.
// Returns the failure with the smallest index (ARCHITECTURE.md §3.4, A2).
//
// `element_ns` is the estimated serial cost of one element (core/cost.hpp).
// When the whole batch is worth it (kParallelMinNs), the elements are cut into
// tasks of about kTaskNs and run on the thread pool (M9, DESIGN.md §2.18).
// Every element is computed the same way wherever it runs (its own scratch,
// 16-byte aligned arrays, M8), so results are bit-for-bit those of the serial
// loop, for any thread count. Failures: each task stops at its first one, and
// every task skips elements past the smallest failure seen so far; the
// smallest index overall is reported, as in the serial loop.
inline constexpr std::size_t kMaxOperands = 4;
// Measured in M9 (experiments/m9_threshold). Variables only so that the
// experiments can move them (_core._set_tuning); nothing else changes them.
inline double kParallelMinNs = 10000;  // a batch whose estimated serial time is below this runs serially
inline double kTaskNs = 5000;          // estimated serial time of one task

namespace detail {
bool begin_parallel();  // syncs the pool with sdnp.get_num_threads() (needs the GIL); true if it may split
void run_tasks(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn);
}  // namespace detail

// `max_tasks` > 0 caps the number of tasks, and so of threads working at once:
// for the LAPACK paths that run serialized on the AMX unit the cores share
// (core/cost.hpp: lu_task_cap; after M11). Which element runs where does not
// change any result.
template <class F>
Failure run_batch(const Batch& b, double element_ns, F&& body, int max_tasks = 0) {
    const std::int64_t n = b.size();
    // element_ns == 0 means: never split (e.g. small gbsv bands, core/cost.hpp)
    if (n >= 2 && element_ns > 0 && element_ns * static_cast<double>(n) >= kParallelMinNs && detail::begin_parallel()) {
        std::int64_t per = std::max<std::int64_t>(1, static_cast<std::int64_t>(kTaskNs / element_ns));
        if (max_tasks > 0 && (n + per - 1) / per > max_tasks) per = (n + max_tasks - 1) / max_tasks;
        const std::int64_t ntasks = (n + per - 1) / per;
        std::vector<Failure> fails(static_cast<std::size_t>(ntasks));
        std::atomic<std::int64_t> first{std::numeric_limits<std::int64_t>::max()};
        detail::run_tasks(ntasks, [&](std::int64_t t) {
            Scratch scratch;
            std::int64_t offs[kMaxOperands];
            const std::int64_t end = std::min(n, (t + 1) * per);
            for (std::int64_t i = t * per; i < end; ++i) {
                if (i > first.load(std::memory_order_relaxed)) return;  // a smaller failure is already known
                b.offsets(i, offs);
                if (const int code = body(i, static_cast<const std::int64_t*>(offs), scratch)) {
                    fails[static_cast<std::size_t>(t)] = {i, code};
                    std::int64_t cur = first.load(std::memory_order_relaxed);
                    while (i < cur && !first.compare_exchange_weak(cur, i, std::memory_order_relaxed)) {
                    }
                    return;
                }
            }
        });
        Failure best;
        for (const Failure& f : fails)
            if (f && (!best || f.index < best.index)) best = f;
        return best;
    }
    Scratch scratch;
    std::int64_t offs[kMaxOperands];
    for (std::int64_t i = 0; i < n; ++i) {
        b.offsets(i, offs);
        if (const int code = body(i, static_cast<const std::int64_t*>(offs), scratch)) return {i, code};
    }
    return {};
}

// Message for a failed batch element: "<what>" or "<what> (at batch index (1, 2))".
std::string failure_message(const Batch& b, std::int64_t idx, const char* what);

}  // namespace sd::linalg
