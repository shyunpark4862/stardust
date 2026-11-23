"""stardust.smooth: smooth curves and densities from noisy data (docs/api/smooth.md, docs/design/smooth.md).

Usage: ``import stardust as sd`` then ``sd.smooth.lowess(x, y).fittedvalues``,
``sd.smooth.smoothing_spline(x, y).spline(x_new)``, ``sd.smooth.kde(data).pdf(points)`` and
``sd.smooth.savgol_filter(x, 11, 3)``. Scatterplot smoothing (lowess, loess), smoothing and penalized splines,
kernel regression, kernel density estimation and Savitzky-Golay filters in one flat namespace (no submodules);
the fitting functions return <Name>Result objects, the scipy-named spline makers return sd.interpolate.BSpline
(decisions SM1-SM142 in docs/design/smooth.md §1.1).
"""

from ._bandwidth import bw_cv_ls, bw_cv_ml, bw_isj, bw_nrd, bw_nrd0, bw_scott, bw_silverman, bw_sj
from ._fitpack import generate_knots, make_splprep, make_splrep
from ._kde import DensityResult, KDEResult, kde
from ._kreg import KernelRegressionResult, bw_dpill, kernel_regression
from ._local import LowessResult, lowess
from ._loess import LoessResult, loess
from ._savgol import savgol_coeffs, savgol_filter
from ._spline import (
    PSplineResult,
    SmoothingSplineResult,
    make_lsq_spline,
    make_smoothing_spline,
    pspline,
    smoothing_spline,
)

__all__ = [
    "lowess",
    "LowessResult",
    "loess",
    "LoessResult",
    "savgol_coeffs",
    "savgol_filter",
    "make_lsq_spline",
    "make_smoothing_spline",
    "make_splrep",
    "make_splprep",
    "generate_knots",
    "smoothing_spline",
    "SmoothingSplineResult",
    "pspline",
    "PSplineResult",
    "kernel_regression",
    "KernelRegressionResult",
    "bw_dpill",
    "kde",
    "KDEResult",
    "DensityResult",
    "bw_scott",
    "bw_silverman",
    "bw_nrd0",
    "bw_nrd",
    "bw_sj",
    "bw_isj",
    "bw_cv_ls",
    "bw_cv_ml",
]
