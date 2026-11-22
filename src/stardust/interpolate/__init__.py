"""stardust.interpolate: functions through given points, built once and evaluated many times
(docs/api/interpolate.md, docs/design/interpolate.md).

Usage: ``import stardust as sd`` then ``sd.interpolate.CubicSpline(x, y)(x_new)``,
``sd.interpolate.make_interp_spline(x, y, k=3)`` and ``sd.interpolate.RegularGridInterpolator(points, values)(xi)``.
The names are scipy.interpolate's, in one flat namespace (no submodules); inputs and options follow this
library's rules (decisions IP1-IP35 in docs/design/interpolate.md §1.1). numpy's interp is the top-level
sd.interp of the array area.

The public names come milestone by milestone (docs/design/interpolate.md §8): IP2 brought the piecewise
polynomials and the cubic family, IP3 the B-splines.
"""

from ._bspline import BSpline, make_interp_spline
from ._ppoly import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
    PPoly,
    pchip_interpolate,
)

__all__ = [
    "PPoly",
    "CubicHermiteSpline",
    "CubicSpline",
    "PchipInterpolator",
    "Akima1DInterpolator",
    "pchip_interpolate",
    "BSpline",
    "make_interp_spline",
]
