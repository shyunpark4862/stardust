"""sd.smooth against statsmodels 0.15.0, scipy 1.18.1, scikit-learn 1.9.1 and R 4.6.2's stats and KernSmooth
(docs/design/smooth.md §5, §7.5).

The plan of the area's cases, milestone by milestone: every spec below names its milestone and the sd.smooth
names it needs, and cases() keeps a spec once its names are public (SM1 brings lowess, SM2 loess, SM3 the
splines, SM4 kernel regression, SM5 the densities and bandwidths, SM6 Savitzky-Golay). The sizes are those of
docs/design/smooth.md §5, where milestone SM0 timed the comparators; R's loess at n = 10^5 (70 s) is left out.
Every comparator runs on one thread. A comparator that computes something else by design says so (`differs`):
R's default loess surface, KernSmooth's and statsmodels' binned paths, R's density.

Data: x uniform on [0, 10] (sorted), y = sin x + N(0, 0.3^2); densities standard normal; signals a sine with
noise; numpy arrays made in the setup from fixed seeds, sd gets them as sd.ndarray outside the timing.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ..registry import Case, Check, Ref, RRef, count


@dataclasses.dataclass(frozen=True)
class Spec:
    milestone: str
    group: str
    name: str
    needs: tuple[str, ...]
    size: str
    setup: Callable[[], dict]
    sd: Callable
    refs: tuple = ()
    quick: bool = False
    note: str = ""
    check: Check = Check()


def _scatter(n, p=1, seed=20261123):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, n, p]))
        if p == 1:
            x = np.sort(10.0 * g.random(n))
            y = np.sin(x) + 0.3 * g.standard_normal(n)
        else:
            x = 10.0 * g.random((n, p))
            y = np.sin(x[:, 0]) + np.cos(x[:, 1]) + 0.3 * g.standard_normal(n)
        out = {"x": x, "y": y, "sx": sd.asarray(x), "sy": sd.asarray(y), "r_x": np.ascontiguousarray(x.T.ravel())}
        if p == 1:  # make_lsq_spline's knots: 101 quantiles of x (SM3 milestone)
            t = np.r_[[x[0]] * 3, np.quantile(x, np.linspace(0.0, 1.0, 102)), [x[-1]] * 3]
            out["t"], out["st"] = t, sd.asarray(t)
        return out

    return setup


def _sample(n, p=1, m=512, seed=20261124):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, n, p, m]))
        x = g.standard_normal(n if p == 1 else (n, p))
        q = np.linspace(-4.0, 4.0, m) if p == 1 else g.standard_normal((m, p))
        return {"x": x, "q": q, "sx": sd.asarray(x), "sq": sd.asarray(q)}

    return setup


def _signal(shape, seed=20261125):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, *shape]))
        k = np.arange(shape[-1] if len(shape) == 1 else shape[0])
        x = np.sin(k / 50.0)
        x = (x if len(shape) == 1 else x[:, None]) + 0.2 * g.standard_normal(shape)
        return {"x": x, "sx": sd.asarray(x)}

    return setup


def _sm():
    import stardust as sd

    return sd.smooth


def _lowess():
    from statsmodels.nonparametric.smoothers_lowess import lowess

    return lowess


def _si():
    import scipy.interpolate as si

    return si


def _np_sm():
    import statsmodels.api as sm

    return sm.nonparametric


def _kde_scipy():
    from scipy.stats import gaussian_kde

    return gaussian_kde


def _kd_sklearn():
    from sklearn.neighbors import KernelDensity

    return KernelDensity


def _savgol_scipy():
    from scipy.signal import savgol_filter

    return savgol_filter


R_XY = {"x": "x", "y": "y"}


def _local_specs():
    out = []
    for n, frac, delta, quick in (
        (10**4, 2 / 3, 0.0, True),
        (10**4, 0.1, 0.0, False),
        (10**4, 2 / 3, 0.1, True),
        (10**5, 2 / 3, 0.1, False),
    ):
        tag = f"lowess frac={frac:.3g}" + (" delta=1%" if delta else "")
        refs = [
            Ref(
                "statsmodels",
                lambda d, f=frac, de=delta: (
                    lambda: _lowess()(d["y"], d["x"], frac=f, it=3, delta=de, return_sorted=False)
                ),
            )
        ]
        if frac == 2 / 3:
            # x is sorted in the setup, so R's sorted $y is in the input order
            refs.append(RRef("R", f"lowess(x, y, f = 2/3, iter = 3, delta = {delta!r})$y", data=R_XY))
        out.append(
            Spec(
                "SM1",
                "local",
                tag,
                ("lowess",),
                count(n),
                _scatter(n),
                lambda d, f=frac, de=delta: lambda: _sm().lowess(d["sx"], d["sy"], frac=f, delta=de).fittedvalues,
                tuple(refs),
                quick=quick,
            )
        )
    for n, p in ((10**4, 1), (10**4, 2)):
        rx = "x" if p == 1 else "X1 + X2"
        setup_r = "" if p == 1 else "X1 <- x[seq_len(length(y))]; X2 <- x[length(y) + seq_len(length(y))]"
        # R's direct fit with its default statistics: 'exact' forms the n x n smoother (800 MB, O(n^3) for delta2),
        # where sd computes trace L and delta1 from the rows and delta2 only on request (n <= 4096)
        direct = "loess.control(surface = 'direct')"
        data = R_XY if p == 1 else {"r_x": "x", "y": "y"}  # two columns: x holds the first, then the second
        refs = (
            RRef("R direct", f"fitted(loess(y ~ {rx}, degree = 2, control = {direct}))", data=data, setup=setup_r),
            RRef(
                "R interpolate",
                f"fitted(loess(y ~ {rx}, degree = 2))",
                data=data,
                setup=setup_r,
                differs="R's default surface interpolates between KD-tree vertices: an approximation",
            ),
        )
        out.append(
            Spec(
                "SM2",
                "local",
                f"loess degree=2 p={p}",
                ("loess",),
                count(n),
                _scatter(n, p),
                lambda d: lambda: _sm().loess(d["sx"], d["sy"]).fittedvalues,
                refs,
                quick=p == 1,
            )
        )
    return out


_GRID = np.linspace(0.5, 9.5, 201)  # where --smoke compares two splines of the data on [0, 10]


def _spline_values(r):
    """A spline result as its values on _GRID (a curve: on [0, 1] of its parameter)."""
    if isinstance(r, tuple) and len(r) == 2:  # make_splprep: (spline, u)
        return np.asarray(r[0](np.linspace(0.0, 1.0, 201)))
    if isinstance(r, tuple):  # splrep's (t, c, k)
        import scipy.interpolate as si

        return si.splev(_GRID, r)
    return np.asarray(r(_GRID))


def _spline_check(**kw):
    return Check(sd=_spline_values, ref=lambda label, r: _spline_values(r), **kw)


_SCIPY_LOST = "scipy's fit loses digits at close x (4.2e-3 off a 1024-bit solve at 10^4, 7.6 at 10^5; sd 7.5e-12)"
_SEARCH = "each library's search stops at its own lam (tests/smooth hold sd's values)"


def _spline_specs():
    out = []
    for n in (10**3, 10**4, 10**5):
        refs = (Ref("scipy", lambda d: lambda: _si().make_smoothing_spline(d["x"], d["y"], lam=1.0)),)
        out.append(
            Spec(
                "SM3",
                "spline",
                "make_smoothing_spline lam=1",
                ("make_smoothing_spline",),
                count(n),
                _scatter(n),
                lambda d: lambda: _sm().make_smoothing_spline(d["sx"], d["sy"], lam=1.0),
                refs,
                quick=n == 10**4,
                check=_spline_check(skip=_SCIPY_LOST if n >= 10**4 else ""),
            )
        )
    for n in (10**3, 10**4, 10**5):
        refs = []
        if n <= 10**3:  # scipy's GCV fails on larger random x (docs/design/smooth.md §2.3)
            refs.append(Ref("scipy", lambda d: lambda: _si().make_smoothing_spline(d["x"], d["y"])))
        if n >= 10**4:
            refs.append(RRef("R all knots", "smooth.spline(x, y, all.knots = TRUE)", data=R_XY))
            refs.append(
                RRef(
                    "R",
                    "smooth.spline(x, y)",
                    data=R_XY,
                    differs="R's default keeps about 200 knots (a different estimator)",
                )
            )
        out.append(
            Spec(
                "SM3",
                "spline",
                "smoothing_spline gcv",
                ("smoothing_spline",),
                count(n),
                _scatter(n),
                lambda d: lambda: _sm().smoothing_spline(d["sx"], d["sy"]),
                tuple(refs),
                quick=n == 10**4,
                check=Check(skip=_SEARCH),
            )
        )
    for n in (10**4, 10**5):
        out.append(
            Spec(
                "SM3",
                "spline",
                "pspline gcv",
                ("pspline",),
                count(n),
                _scatter(n),
                lambda d: lambda: _sm().pspline(d["sx"], d["sy"]),
                (
                    RRef(
                        "R mgcv",
                        'gam(y ~ s(x, bs = "ps", k = 40), method = "GCV.Cp")',
                        data=R_XY,
                        packages=("mgcv",),
                        differs="mgcv's P-spline basis and penalty scaling",
                    ),
                ),
                check=Check(skip=_SEARCH),
            )
        )
    for n in (10**3, 10**4, 10**5):
        refs = (
            Ref("scipy", lambda d, n=n: lambda: _si().make_splrep(d["x"], d["y"], s=0.09 * n)),
            Ref(
                "scipy splrep",
                lambda d, n=n: lambda: _si().splrep(d["x"], d["y"], s=0.09 * n),
                lib="scipy",
                differs="the legacy FITPACK knot rule: other knots for the same s",
            ),
        )
        out.append(
            Spec(
                "SM3",
                "spline",
                "make_splrep s=0.09n",
                ("make_splrep",),
                count(n),
                _scatter(n),
                lambda d, n=n: lambda: _sm().make_splrep(d["sx"], d["sy"], s=0.09 * n),
                refs,
                check=_spline_check(),
            )
        )
    # SM3 milestone: the least-squares spline on 100 interior knots, the curve maker, a million points at a given lam
    for n in (10**4, 10**5):
        out.append(
            Spec(
                "SM3",
                "spline",
                "make_lsq_spline 100 knots",
                ("make_lsq_spline",),
                count(n),
                _scatter(n),
                lambda d: lambda: _sm().make_lsq_spline(d["sx"], d["sy"], d["st"]),
                (Ref("scipy", lambda d: lambda: _si().make_lsq_spline(d["x"], d["y"], d["t"])),),
                note="knots: x's 101 quantiles, the ends four times",
                check=_spline_check(),
            )
        )
    out.append(
        Spec(
            "SM3",
            "spline",
            "make_splprep 3-D s=0.05n",
            ("make_splprep",),
            count(10**4),
            _curve(10**4),
            lambda d: lambda: _sm().make_splprep(d["sX"], s=0.05 * 10**4),
            (Ref("scipy", lambda d: lambda: _si().make_splprep(d["X"], s=0.05 * 10**4)),),
            check=_spline_check(),
        )
    )
    out.append(
        Spec(
            "SM3",
            "spline",
            "make_smoothing_spline lam=1",
            ("make_smoothing_spline",),
            count(10**6),
            _scatter(10**6),
            lambda d: lambda: _sm().make_smoothing_spline(d["sx"], d["sy"], lam=1.0),
            (Ref("scipy", lambda d: lambda: _si().make_smoothing_spline(d["x"], d["y"], lam=1.0)),),
            check=_spline_check(skip=_SCIPY_LOST),
        )
    )
    return out


def _curve(n, seed=20261126):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, n]))
        th = np.sort(g.random(n)) * 2 * np.pi
        X = np.stack([np.cos(th), np.sin(2 * th), th / 7]) + 0.1 * g.standard_normal((3, n))
        return {"X": X, "sX": sd.asarray(X)}

    return setup


def _kreg_specs():
    out = []
    for n in (200, 500):
        refs = (
            Ref(
                "statsmodels",
                lambda d: lambda: _np_sm().KernelReg(d["y"], d["x"], "c", reg_type="ll", bw="cv_ls").bw,
                differs="the searches differ (fmin on h against a grid and Brent on log h, decision SM87): the "
                "bandwidths agree to about 1e-5, the criteria at one h to 1e-15",
            ),
        )
        out.append(
            Spec(
                "SM4",
                "kreg",
                "kernel_regression cv_ls",
                ("kernel_regression",),
                str(n),
                _scatter(n),
                lambda d: lambda: _sm().kernel_regression(d["sx"], d["sy"]).bw,
                refs,
                quick=n == 500,
            )
        )
    out.append(
        Spec(
            "SM4",
            "kreg",
            "kernel_regression aicc",
            ("kernel_regression",),
            "500",
            _scatter(500),
            lambda d: lambda: _sm().kernel_regression(d["sx"], d["sy"], degree=0, bw="aicc").bw,
            (
                Ref(
                    "statsmodels",
                    lambda d: lambda: _np_sm().KernelReg(d["y"], d["x"], "c", reg_type="lc", bw="aic").bw,
                    differs="the searches differ (decision SM87): the bandwidths agree to about 1e-5",
                ),
            ),
            note="the local constant: statsmodels' AIC of the local line is another function (decision SM89)",
        )
    )
    out.append(
        Spec(
            "SM4",
            "kreg",
            "bw_dpill",
            ("bw_dpill",),
            count(10**4),
            _scatter(10**4),
            lambda d: lambda: _sm().bw_dpill(d["sx"], d["sy"]),
            (
                RRef(
                    "R KernSmooth",
                    "KernSmooth::dpill(x, y)",
                    data=R_XY,
                    packages=("KernSmooth",),
                    differs="binned on 401 grid points",
                ),
            ),
        )
    )
    refs = (
        Ref("statsmodels", lambda d: lambda: _np_sm().KernelReg(d["y"], d["x"], "c", reg_type="ll", bw=[0.5]).fit()[0]),
    )
    out.append(
        Spec(
            "SM4",
            "kreg",
            "kernel_regression bw=0.5",
            ("kernel_regression",),
            "2000",
            _scatter(2000),
            lambda d: lambda: _sm().kernel_regression(d["sx"], d["sy"], bw=0.5).fittedvalues,
            refs,
        )
    )
    for n in (10**4,):
        out.append(
            Spec(
                "SM4",
                "kreg",
                "kernel_regression degree=0 bw=0.5",
                ("kernel_regression",),
                count(n),
                _scatter(n),
                lambda d: lambda: _sm().kernel_regression(d["sx"], d["sy"], degree=0, bw=0.5).fittedvalues,
                (
                    RRef(
                        "R ksmooth",
                        "ksmooth(x, y, 'normal', bandwidth = 0.5 / 0.3706506, x.points = x)$y",
                        data=R_XY,
                        differs="ksmooth cuts the kernel at 4 standard deviations",
                    ),
                    RRef(
                        "R locpoly",
                        "KernSmooth::locpoly(x, y, degree = 0, bandwidth = 0.5)",
                        data=R_XY,
                        packages=("KernSmooth",),
                        differs="binned on 401 grid points",
                    ),
                ),
            )
        )
    return out


def _kde_specs():
    out = []
    for n in (10**3, 10**4, 10**5):
        refs = (
            Ref("scipy", lambda d: lambda: _kde_scipy()(d["x"])(d["q"])),
            Ref(
                "sklearn",
                lambda d: (
                    lambda: (
                        _kd_sklearn()(bandwidth=0.3, rtol=0, atol=0).fit(d["x"][:, None]).score_samples(d["q"][:, None])
                    )
                ),
                differs="a fixed bandwidth of 0.3",
            ),
        )
        out.append(
            Spec(
                "SM5",
                "kde",
                "kde pdf 512 points",
                ("kde",),
                count(n),
                _sample(n),
                lambda d: lambda: _sm().kde(d["sx"]).pdf(d["sq"]),
                refs,
                quick=n == 10**4,
            )
        )
    out.append(
        Spec(
            "SM5",
            "kde",
            "kde pdf m=n",
            ("kde",),
            count(10**4),
            _sample(10**4, m=10**4),
            lambda d: lambda: _sm().kde(d["sx"]).pdf(d["sq"]),
            (Ref("scipy", lambda d: lambda: _kde_scipy()(d["x"])(d["q"])),),
        )
    )
    out.append(
        Spec(
            "SM5",
            "kde",
            "kde pdf 2-d m=n",
            ("kde",),
            count(10**4),
            _sample(10**4, p=2, m=10**4),
            lambda d: lambda: _sm().kde(d["sx"]).pdf(d["sq"]),
            (Ref("scipy", lambda d: lambda: _kde_scipy()(d["x"].T)(d["q"].T)),),
        )
    )
    for n in (10**4, 10**5, 10**6):

        def fft(d):
            import statsmodels.api as sm

            return lambda: sm.nonparametric.KDEUnivariate(d["x"]).fit(fft=True, gridsize=512, cut=3)

        refs = [Ref("statsmodels", fft, differs="statsmodels' binned FFT density")]
        if n <= 10**5:
            refs.append(RRef("R", "density(x, n = 512)$y", data={"x": "x"}, differs="R's binned FFT density"))
        out.append(
            Spec(
                "SM5",
                "kde",
                "density binned",
                ("kde",),
                count(n),
                _sample(n),
                lambda d: lambda: _sm().kde(d["sx"]).density(n=512),
                tuple(refs),
                quick=n == 10**5,
            )
        )
    for n in (10**4, 10**5):
        out.append(
            Spec(
                "SM5",
                "bandwidth",
                "bw_sj",
                ("bw_sj",),
                count(n),
                _sample(n),
                lambda d: lambda: _sm().bw_sj(d["sx"]),
                (RRef("R", "bw.SJ(x)", data={"x": "x"}),),
                check=Check(rtol=2e-2),  # R's uniroot stops at 0.1 lower (about 1% of hmax); sd solves to 1e-13
            )
        )
    for bw in ("cv_ml", "cv_ls"):
        out.append(
            Spec(
                "SM5",
                "bandwidth",
                f"bw_{bw}",
                (f"bw_{bw}",),
                "200",
                _sample(200),
                lambda d, bw=bw: lambda: getattr(_sm(), f"bw_{bw}")(d["sx"]),
                (Ref("statsmodels", lambda d, bw=bw: lambda: _np_sm().KDEMultivariate(d["x"], "c", bw=bw).bw),),
                check=Check(rtol=1e-3)
                if bw == "cv_ml"
                else Check(
                    skip="the same criterion; on this sample sd's grid finds its smallest value at h = 0.096, "
                    "statsmodels' fmin stops at the local minimum near 0.42 (a larger criterion)"
                ),
            )
        )
    return out


def _savgol_specs():
    out = []
    for shape, window, order, axis, mode, quick in (
        ((10**4,), 11, 3, -1, "interp", False),
        ((10**5,), 11, 3, -1, "interp", True),
        ((10**6,), 11, 3, -1, "interp", False),
        ((10**5,), 101, 3, -1, "interp", False),
        ((10**6,), 101, 3, -1, "interp", True),
        ((1000, 1000), 11, 3, 0, "interp", False),
        # SM6 milestone: the other modes, the last and a middle axis, a long window's end fits
        ((10**6,), 11, 3, -1, "mirror", False),
        ((1000, 1000), 11, 3, -1, "interp", False),
        ((100, 100, 100), 11, 3, 1, "nearest", False),
        ((10**5,), 1001, 5, -1, "interp", False),
    ):
        size = count(shape[0]) if len(shape) == 1 else "(" + ", ".join(map(str, shape)) + ")"
        tag = f"savgol_filter ({window}, {order})" + (f" axis={axis}" if axis != -1 else "")
        tag += f" mode={mode}" if mode != "interp" else ""
        # scipy's lstsq coefficients are wrong from about polyorder 5 at window 1001 (their sum 1 - 1.2e-6)
        differs = "scipy's coefficients lose digits at this window (Vandermonde lstsq)" if window > 101 else ""
        refs = (
            Ref(
                "scipy",
                lambda d, w=window, o=order, a=axis, m=mode: lambda: _savgol_scipy()(d["x"], w, o, axis=a, mode=m),
                differs=differs,
            ),
        )
        out.append(
            Spec(
                "SM6",
                "savgol",
                tag,
                ("savgol_filter",),
                size,
                _signal(shape),
                lambda d, w=window, o=order, a=axis, m=mode: lambda: _sm().savgol_filter(d["sx"], w, o, axis=a, mode=m),
                refs,
                quick=quick,
            )
        )
    return out


SPECS = _local_specs() + _spline_specs() + _kreg_specs() + _kde_specs() + _savgol_specs()


def _public(names) -> bool:
    import stardust as sd

    return all(n in sd.smooth.__all__ for n in names)


def cases() -> list[Case]:
    return [
        Case(
            area="smooth",
            group=s.group,
            name=s.name,
            size=s.size,
            setup=s.setup,
            sd=s.sd,
            refs=s.refs,
            quick=s.quick,
            note=s.note,
            check=s.check,
        )
        for s in SPECS
        if _public(s.needs)
    ]


TITLES = {
    "local": "산점도 평활 (`lowess`, `loess`)",
    "spline": "평활 스플라인",
    "kreg": "핵 회귀",
    "kde": "핵밀도추정",
    "bandwidth": "대역폭 선택",
    "savgol": "Savitzky–Golay",
}
