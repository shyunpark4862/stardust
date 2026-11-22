// The axis driver (docs/design/fft.md §3.7): one transform along one axis of an array, line by line.
//
// A line (the entries along the transform axis at one position of the other axes) is gathered into a
// task's buffer -- converted to complex128, cut or zero-padded to the transform length -- transformed by
// the plan of (kind, length), scaled by the normalisation factor and scattered into the result. When the
// input line already is what the plan reads (complex128, contiguous, the transform axis last) the plan
// reads it in place and writes the result's line directly: the same operations on the same values. A
// line's bits are those of the line transformed alone: they depend on its values, the kind, the length
// and the factor only, never on the batch, the layout or the thread count.
//
// Splitting: tasks of consecutive lines, as many as make kTaskWork units of work (the plan's cost per
// line, about n log2 n), decided by the sizes alone; a call of kParallelWork units or more splits and runs
// without the GIL. Lines along an axis other than the last are gathered and scattered a block of up to
// kBlockLines at a time (neighbouring lines, so the reads of one position are close together); the
// block changes the memory traffic, not the values.
#pragma once

#include <cstdint>

#include "array/ndarray.hpp"
#include "fft/plan.hpp"

namespace sd::fft {

inline constexpr int64_t kParallelWork = int64_t{1} << 15;  // API.md §3, array.md §10.4
inline constexpr int64_t kTaskWork = int64_t{1} << 12;      // array.md §10.4's kParallelTask
inline constexpr int64_t kBlockLines = 32;
inline constexpr int64_t kBlockDoubles = int64_t{1} << 19;  // a task's line buffers at most (docs/design/fft.md §3.8)

// The result (complex128, C order, the input's shape with `n` along `axis`) of the complex transform
// of every line of x (bool, int64, float64 or complex128) along `axis`, times `scale`. The caller holds
// the GIL and has checked the arguments; `out` is allocated by the caller.
void c2c_axis(const NDArray& x, int axis, int64_t n, bool inverse, double scale, NDArray& out);

}  // namespace sd::fft
