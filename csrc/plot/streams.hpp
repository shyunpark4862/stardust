// Streamlines of a vector field sampled on a grid (docs/design/plot.md PL49).
//
// The field (u, v) is given at nx x ny points spread evenly over the box [x0, x1] x [y0, y1]
// (row j at y0 + j (y1 - y0) / (ny - 1)) and read between them by bilinear interpolation. The
// lines are spread evenly over the box: a coarse grid of mx x my square cells keeps which cells a
// line has passed through, seeds are tried cell by cell from the edge of the box inwards, and a
// line grows from its seed forward and backward (second-order Runge-Kutta on the field's
// direction, a fifth of a cell a step) until it leaves the box, reaches a cell another line holds,
// comes to a point where the field vanishes or has no value, or turns back on itself. A line
// shorter than `min_cells` cells is dropped and its cells freed.
//
// The result: three numbers a point (x, y, the field's magnitude there) with a NaN triple between
// lines; each line runs along the field.
#pragma once

#include <cstddef>
#include <vector>

namespace sd::plot {

std::vector<double> streamlines(
    const double* u,
    const double* v,
    int nx,
    int ny,
    double x0,
    double x1,
    double y0,
    double y1,
    int mx,
    int my,
    double min_cells
);

}  // namespace sd::plot
