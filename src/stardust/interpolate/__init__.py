"""stardust.interpolate: functions through given points, built once and evaluated many times
(docs/api/interpolate.md, docs/design/interpolate.md).

Usage: ``import stardust as sd`` then ``sd.interpolate.CubicSpline(x, y)(x_new)``,
``sd.interpolate.make_interp_spline(x, y, k=3)`` and ``sd.interpolate.RegularGridInterpolator(points, values)(xi)``.
The names are scipy.interpolate's, in one flat namespace (no submodules); inputs and options follow this
library's rules (decisions IP1-IP61 in docs/design/interpolate.md §1.1). numpy's interp is the top-level
sd.interp of the array area.

The public names come milestone by milestone (docs/design/interpolate.md §8): IP2 brought the piecewise
polynomials and the cubic family, IP3 the B-splines, IP4 the polynomial and rational interpolants,
IP5 the regular grids, IP6 the scattered-data interpolants.
"""

from ._aaa import AAA, FloaterHormannInterpolator
from ._bspline import BSpline, make_interp_spline
from ._grid import RegularGridInterpolator, interpn
from ._polyint import BarycentricInterpolator, KroghInterpolator, barycentric_interpolate, krogh_interpolate
from ._ppoly import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
    PPoly,
    pchip_interpolate,
)
from ._rbf import NearestNDInterpolator, RBFInterpolator

__all__ = [
    "PPoly",
    "CubicHermiteSpline",
    "CubicSpline",
    "PchipInterpolator",
    "Akima1DInterpolator",
    "pchip_interpolate",
    "BSpline",
    "make_interp_spline",
    "BarycentricInterpolator",
    "barycentric_interpolate",
    "KroghInterpolator",
    "krogh_interpolate",
    "FloaterHormannInterpolator",
    "AAA",
    "RegularGridInterpolator",
    "interpn",
    "RBFInterpolator",
    "NearestNDInterpolator",
]
