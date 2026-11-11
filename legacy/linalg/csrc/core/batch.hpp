// Batch axes: broadcasting them, walking them, and the batch driver every
// LAPACK function goes through (ARCHITECTURE.md §3.4).
#pragma once

#include "core/types.hpp"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
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

// Scratch memory for one worker. The driver hands each worker its own, so a
// body may reuse it across the matrices it processes.
// Slots 0-3 (and int slot 0) belong to the function bodies; slots 4-5 (and
// int slot 1) to the LAPACK wrappers' workspaces (lapack.hpp).
struct Scratch {
    static constexpr int kSlots = 6;
    static constexpr int kWork = 4, kRWork = 5, kIWork = 1;
    std::vector<std::byte> bytes[kSlots];  // independent areas; slot 0 is the default
    std::vector<lint> ints[2];

    template <class T>
    T* get(std::size_t n, int slot = 0) {
        std::vector<std::byte>& b = bytes[slot];
        if (b.size() < n * sizeof(T)) b.resize(n * sizeof(T));
        return reinterpret_cast<T*>(b.data());
    }
    lint* get_ints(std::size_t n, int slot = 0) {
        if (ints[slot].size() < n) ints[slot].resize(n);
        return ints[slot].data();
    }
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
