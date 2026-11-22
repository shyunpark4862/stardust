// The n-dimensional iterator: a scaled-down numpy `nditer`
// (docs/design/array.md §4.2).
//
// Every elementwise operation (copy, cast, add, ...) has the same shape:
// walk N arrays of one common shape in lockstep and apply a 1-D "inner loop"
// kernel. The iterator turns the N-d problem into as few, as long, as
// memory-friendly 1-D runs as possible, in four steps:
//
//   1. broadcast   every operand gets the common shape; broadcast axes get stride 0
//   2. drop        axes of extent 1 (never stepped over)
//   3. reorder     axes sorted by |stride| so the innermost loop walks memory
//                  sequentially (a transposed array is read in memory order)
//   4. coalesce    neighbouring axes that are contiguous for *every* operand are
//                  merged; a contiguous (3, 4, 5) array becomes one run of 60
//
// What remains is driven by an odometer over the outer axes, calling the
// kernel once per innermost run.
#pragma once

#include "array/ndarray.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace sd {

// The most operands one iterator walks: inputs and outputs together.
// hyp2f1(a, b, c, z) -> out and airy(x) -> (ai, aip, bi, bip) need 5
// (docs/design/special.md §3.1).
inline constexpr int kMaxOperands = 5;

// The shortest piece of the innermost axis a parallel run hands out
// (docs/design/array.md §10.4).
inline constexpr std::int64_t kMinInnerPiece = 1024;
// At most this many pieces per run (NdIter::cap_pieces): a long run of a weighted
// kernel was cut into ~0.5 us pieces, and their pickup cost showed (exp of 10^7
// on 10 threads: 19,500 pieces, 2.0 ms; at most 256: 1.15 ms; 1 thread 9.9 ms).
// A constant, so the cut still depends on the shape alone.
inline constexpr std::int64_t kMaxPieces = 256;

// The signature every kernel ("inner loop") has: process n elements; operand
// k starts at args[k] and advances strides[k] bytes per element. As in numpy,
// inputs come first and outputs last.
using Loop = void (*)(std::byte* const* args, std::int64_t n, const std::int64_t* strides, void* ctx);

struct IterOperand {
    std::byte* data;
    std::span<const std::int64_t> shape;
    std::span<const std::int64_t> strides;
};

inline IterOperand operand(const NDArray& a) { return {a.data, a.shape_span(), a.strides_span()}; }

// How far NdIter simplifies the traversal. Full drops extent-1 axes, reorders
// the axes by stride and coalesces them; every operation uses it, except where
// the index order itself matters: the one-index scatter runs Naive (broadcast
// only, C order, one thread) so that with repeated indices the last write wins.
enum class IterMode { Naive, Full };

// How run() may split the work across threads (docs/design/array.md §10.5).
// Results never depend on it -- only which splits are safe does:
//   Elementwise  every element is computed on its own: any axis may be cut
//   Reduction    the last operand is the output, broadcast (stride 0) along
//                the reduced axes; only axes where it has a real stride may
//                be cut, so each output element still folds its inputs in
//                exactly the serial order, on one thread
//   None         never split
enum class Split { None, Elementwise, Reduction };

// The broadcast shape of several shapes (numpy rules: align right; extents
// must match or be 1). Raises ValueError when they are incompatible.
std::vector<std::int64_t> broadcast_shapes(std::span<const std::span<const std::int64_t>> shapes);
// The same into a fixed array, without allocating; returns the ndim.
int broadcast_into(std::span<const std::span<const std::int64_t>> shapes, Dims& out);

class NdIter {
public:
    // Plans the traversal of `ops`, each broadcast to `shape`.
    NdIter(std::span<const std::int64_t> shape, std::span<const IterOperand> ops, IterMode mode = IterMode::Full);

    int nop() const { return nop_; }
    int ndim() const { return ndim_; }
    bool empty() const { return empty_; }
    std::int64_t extent(int d) const { return shape_[d]; }
    std::int64_t stride(int op, int d) const { return strides_[op][d]; }

    // Calls kernel(args, n, strides, ctx) once per innermost run. With enough
    // work, the runs are spread over the thread pool (see Split); the kernel
    // and ctx must then be safe to call from several threads at once.
    // `weight` is the work per element, for kernels that do more than one
    // element's worth per call position (a lane kernel processes a whole lane).
    void run(Loop kernel, void* ctx = nullptr, Split split = Split::Elementwise, std::int64_t weight = 1) const;

    // Allows run() to walk a 2-D iteration in tiles when an operand runs across
    // the inner axis (x.T + y): only for kernels that treat every element on its
    // own (ufuncs), since the inner runs get shorter and come in tile order.
    void allow_tiles() { tiles_ok_ = true; }

    // Caps run()'s pieces at kMaxPieces (longer pieces for long runs). The
    // array's ufuncs set it; plan() of an iterator without it is unchanged.
    void cap_pieces() { capped_ = true; }

    // How run() cuts the work: `tasks` pieces of `chunk` indices along
    // `axis` (tasks == 1: not split), and the work in bytes it counted.
    // Depends on the shape, layout and dtypes only, never on the thread
    // count, so callers may base decisions that change results on it.
    struct Plan {
        int axis = -1;
        std::int64_t chunk = 0, tasks = 1, work = 0;
    };
    Plan plan(Split split, std::int64_t weight = 1) const;

private:
    void run_serial(Loop kernel, void* ctx) const;
    int split_axis(Split split) const;  // -1: no axis may be split

    int nop_ = 0;
    int ndim_ = 0;
    bool empty_ = false;
    // Left uninitialized: only the first ndim_ entries (and nop_ operands)
    // are ever written or read. Zero-filling 1.3 KB per iterator was a
    // visible cost on tiny arrays (docs/design/array.md §11).
    Dims shape_;
    std::array<Dims, kMaxOperands> strides_;
    std::array<std::byte*, kMaxOperands> data_{};
    bool tiles_ok_ = false;
    bool capped_ = false;
};

}  // namespace sd
