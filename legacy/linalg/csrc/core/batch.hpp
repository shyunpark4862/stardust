// Batch axes: broadcasting them, walking them, and the batch driver every
// LAPACK function goes through (ARCHITECTURE.md §3.4).
#pragma once

#include "core/types.hpp"

#include <Python.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sdla {

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

// Scratch memory for one worker. The driver hands each worker its own, so a
// body may reuse it across the matrices it processes.
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
    // only the first one queries; the sizes, and so the results' bits, are the
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
// M1 runs serially, so it can stop at the first failure: in index order that
// is already the smallest one, and nothing of a failed call is returned.
// M9 splits the batch over the thread pool behind this same interface.
inline constexpr std::size_t kMaxOperands = 4;

template <class F>
Failure run_batch(const Batch& b, F&& body) {
    Scratch scratch;
    std::int64_t offs[kMaxOperands];
    for (std::int64_t i = 0; i < b.size(); ++i) {
        b.offsets(i, offs);
        if (const int code = body(i, static_cast<const std::int64_t*>(offs), scratch)) return {i, code};
    }
    return {};
}

// Message for a failed batch element: "<what>" or "<what> (at batch index (1, 2))".
std::string failure_message(const Batch& b, std::int64_t idx, const char* what);

}  // namespace sdla
