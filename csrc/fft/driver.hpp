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
// line, about n log2 n) and at least lines / kMaxTasks of them (milestone FT3: a task per few lines made
// the shared task counter the bottleneck of large calls), decided by the sizes alone; a call of
// kParallelWork units or more splits and runs without the GIL. Lines along an axis other than the last
// are gathered and scattered a block of up to kBlockLines at a time (neighbouring lines, so the reads of
// one position are close together; the block's buffers are the thread's scratch, their lines padded off
// powers of two); the block changes the memory traffic, not the values.
#pragma once

#include <cstdint>

#include "array/ndarray.hpp"
#include "fft/plan.hpp"

namespace sd::fft {

inline constexpr int64_t kParallelWork = int64_t{1} << 15;  // API.md §3, array.md §10.4
inline constexpr int64_t kTaskWork = int64_t{1} << 12;      // array.md §10.4's kParallelTask
inline constexpr int64_t kMaxTasks = 1024;
inline constexpr int64_t kBlockLines = 128;  // and a block's buffers at most 2^16 doubles each (§3.8)
inline constexpr int64_t kBlockShare = 64;   // a block has at most lines / kBlockShare lines

// The result (complex128, the input's shape with `n` along `axis`) of the complex transform of every
// line of x (bool, int64, float64 or complex128) along `axis`, times `scale`. The caller holds the GIL
// and has checked the arguments; `out` is allocated by the caller (any strides). With `conj_in` a line
// is conjugated as it is gathered (before the zero padding: the padded entries stay +0), with
// `conj_out` the result (both exact). `out` may be x itself (the passes of an n-dimensional transform,
// milestone FT3, when the axis keeps its length): every line is then gathered whole before its result
// is scattered back, which gives the bits of the direct path.
void c2c_axis(
    const NDArray& x,
    int axis,
    int64_t n,
    bool inverse,
    double scale,
    NDArray& out,
    bool conj_in = false,
    bool conj_out = false
);

// The real transforms (milestone FT2, docs/design/fft.md §3.4). An even n runs the real plan of n
// (fft/rfft.hpp); an odd n runs the complex plan of n on the line with zero imaginary parts and keeps the
// Hermitian part of the result, (y_k + conj y_(n-k)) / 2 (decision FT43): the exact projection, so the
// result is exactly Hermitian and its DC imaginary part +0, at one rounding per entry.
//
// r2c_axis: the transform of every line of x (bool, int64 or float64) along `axis`, cut or zero-padded
// to n, times `scale`: the half spectrum (n / 2 + 1 entries: rfft, ihfft) or, with `full`, the whole
// spectrum with y_(n-k) = conj y_k written from the half (fft and ifft of real input, decision FT16), and
// with `conj` the conjugate of either (ihfft, ifft). out: complex128.
void r2c_axis(const NDArray& x, int axis, int64_t n, bool full, bool conj, double scale, NDArray& out);

// c2r_axis: the unnormalised inverse transform of length n of the Hermitian extension of every line of
// x (any numeric dtype; a real dtype is read as complex with zero imaginary parts) along `axis`, of which
// n / 2 + 1 entries are used (cut or zero-padded) and the imaginary parts of the DC and (even n) Nyquist
// entries are dropped, times `scale` (irfft; with `conj`, of conj(x): hfft). out: float64.
void c2r_axis(const NDArray& x, int axis, int64_t n, bool conj, double scale, NDArray& out);

}  // namespace sd::fft
