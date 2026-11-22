// Streamlines of a vector field sampled on a grid (streams.hpp has the method).
#include "plot/streams.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sd::plot {

namespace {

struct Field {
    const double *u, *v;
    int nx, ny;
    int mx, my;
    double sx, sy;  // mask units per data unit along x and y

    // the field at a point of mask coordinates (gx in [0, mx], gy in [0, my]), bilinear; false
    // where a corner has no value
    bool at(double gx, double gy, double& fu, double& fv) const {
        const double fx = gx / mx * (nx - 1), fy = gy / my * (ny - 1);
        int i = static_cast<int>(std::floor(fx)), j = static_cast<int>(std::floor(fy));
        i = std::clamp(i, 0, nx - 2);
        j = std::clamp(j, 0, ny - 2);
        const double a = fx - i, b = fy - j;
        const std::size_t k = static_cast<std::size_t>(j) * nx + i;
        const double w00 = (1 - a) * (1 - b), w10 = a * (1 - b), w01 = (1 - a) * b, w11 = a * b;
        fu = w00 * u[k] + w10 * u[k + 1] + w01 * u[k + nx] + w11 * u[k + nx + 1];
        fv = w00 * v[k] + w10 * v[k + 1] + w01 * v[k + nx] + w11 * v[k + nx + 1];
        return std::isfinite(fu) && std::isfinite(fv);
    }

    // the unit direction of the field in mask coordinates (sign: +1 along, -1 against)
    bool dir(double gx, double gy, double sign, double& dx, double& dy) const {
        double fu, fv;
        if (!at(gx, gy, fu, fv)) return false;
        const double du = fu * sx, dv = fv * sy, n = std::hypot(du, dv);
        if (!(n > 1e-300)) return false;
        dx = sign * du / n;
        dy = sign * dv / n;
        return true;
    }
};

}  // namespace

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
) {
    const Field F{u, v, nx, ny, mx, my, mx / (x1 - x0), my / (y1 - y0)};
    std::vector<char> mask(static_cast<std::size_t>(mx) * my, 0);
    const double h = 0.2;  // the step, in cells
    const int most = static_cast<int>(4.0 * (mx + my) / h);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> out;

    // the seeds: the cells from the edge of the box inwards, ring by ring
    std::vector<int> seeds;
    seeds.reserve(static_cast<std::size_t>(mx) * my);
    for (int r = 0; 2 * r < std::min(mx, my) + 1; ++r) {
        const int a = r, b = mx - 1 - r, c = r, d = my - 1 - r;
        if (a > b || c > d) break;
        for (int i = a; i <= b; ++i) seeds.push_back(c * mx + i);
        for (int j = c + 1; j <= d; ++j) seeds.push_back(j * mx + b);
        if (d > c)
            for (int i = b - 1; i >= a; --i) seeds.push_back(d * mx + i);
        if (b > a)
            for (int j = d - 1; j > c; --j) seeds.push_back(j * mx + a);
    }

    std::vector<int> own;
    std::vector<double> fwd, bwd;  // x, y pairs in mask coordinates
    for (const int seed : seeds) {
        if (mask[seed]) continue;
        const double sxp = seed % mx + 0.5, syp = seed / mx + 0.5;
        double t0, t1;
        if (!F.dir(sxp, syp, 1.0, t0, t1)) continue;
        own.assign(1, seed);
        mask[seed] = 1;
        double length = 0.0;
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<double>& path = pass == 0 ? fwd : bwd;
            path.assign({sxp, syp});
            const double sign = pass == 0 ? 1.0 : -1.0;
            double gx = sxp, gy = syp;
            int cell = seed;
            for (int s = 0; s < most; ++s) {
                double ax, ay, bx, by;
                if (!F.dir(gx, gy, sign, ax, ay)) break;
                const double mxp = gx + 0.5 * h * ax, myp = gy + 0.5 * h * ay;
                if (mxp < 0 || mxp > mx || myp < 0 || myp > my || !F.dir(mxp, myp, sign, bx, by)) break;
                const double nxp = gx + h * bx, nyp = gy + h * by;
                if (nxp < 0 || nxp > mx || nyp < 0 || nyp > my) break;
                const int ci = std::min(static_cast<int>(nxp), mx - 1), cj = std::min(static_cast<int>(nyp), my - 1);
                const int c = cj * mx + ci;
                if (c != cell) {
                    if (mask[c]) break;  // another line's cell, or this line's own: stop here
                    mask[c] = 1;
                    own.push_back(c);
                    cell = c;
                }
                gx = nxp;
                gy = nyp;
                path.insert(path.end(), {gx, gy});
                length += h;
            }
        }
        if (length < min_cells) {  // too short to show a flow: its cells are free again
            for (const int c : own) mask[c] = 0;
            mask[seed] = 1;  // but the seed is not tried again
            continue;
        }
        if (!out.empty()) out.insert(out.end(), {nan, nan, nan});
        const auto emit = [&](double gx, double gy) {
            double fu, fv;
            const double speed = F.at(gx, gy, fu, fv) ? std::hypot(fu, fv) : nan;
            out.insert(out.end(), {x0 + gx / F.sx, y0 + gy / F.sy, speed});
        };
        for (std::size_t k = bwd.size(); k >= 4; k -= 2) emit(bwd[k - 2], bwd[k - 1]);  // upstream, reversed
        for (std::size_t k = 0; k + 1 < fwd.size(); k += 2) emit(fwd[k], fwd[k + 1]);
    }
    return out;
}

}  // namespace sd::plot
