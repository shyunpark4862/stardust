// Level curves of a function sampled on a grid (docs/design/plot.md §5): marching squares.
//
// The grid F has ny + 1 rows of nx + 1 values. For a level, a corner is inside where its value is
// above the level; a cell whose four corners are finite and not all on one side is crossed.
// Every crossed edge is found once, in the order the cells are walked (row by row) and within a
// cell bottom, right, top, left: `edges` gives that list, `curves` and `inside` take the
// crossing points back (after the caller has polished them with the function) in that order.
// The crossed cells of the last level asked are kept, so the three calls for one level walk
// the grid once.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd::plot {

class Levels {
public:
    Levels(std::vector<double> F, std::vector<double> xs, std::vector<double> ys);

    // Eight numbers per crossed edge: the crossing estimated by linear interpolation (x, y), and
    // the edge's two corners with their values minus the level (ax, ay, a, bx, by, b).
    std::vector<double> edges(double level);

    // The level set as polylines: x, y pairs, a NaN pair between curves; a closed curve repeats
    // its first point. `points` holds x, y of every crossed edge in the order of `edges`. Open
    // curves come first. In a saddle cell the corners' mean decides which edges are joined.
    std::vector<double> curves(double level, const std::vector<double>& points);

    // The region above the level as polygons (ops over "MLZ" and coordinates): a rectangle for
    // every run of cells of a row wholly inside, and the inside polygon of every crossed cell.
    void inside(double level, const std::vector<double>& points, std::string& ops, std::vector<double>& xy);

private:
    void prepare(double level);
    double at(std::size_t j, std::size_t i) const { return F_[j * (nx_ + 1) + i] - level_; }

    std::vector<double> F_, xs_, ys_;
    std::size_t ny_, nx_;
    double level_ = 0.0;
    bool ready_ = false;
    std::vector<std::uint32_t> cells_;  // j * nx + i of the crossed cells, in order
    std::unordered_map<std::int64_t, std::int32_t> index_;  // of every crossed edge
    std::vector<double> edges_;
};

}  // namespace sd::plot
