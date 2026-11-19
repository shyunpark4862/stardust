"""Accuracy of the multivariate normal and t cdfs (T9, csrc/stats/mvcdf.hpp) against the
references of tools/mvcdf_ref.py (tests/stats/ref_mvcdf/).

    .venv/bin/python bench/stats/t9/cdf_accuracy.py      # writes cdf_accuracy.md beside it

  * d = 2, 3: the relative error |v - ref| / ref against rigorous Arb values (hi + lo), by
    region and by the size of the probability, and its ratio to eps max(1, -ln P): the
    rounding of the exponent q ~ -ln P inside exp is what is left in the tails.
    Also against the problem's condition number cond = sum |d log P / d log x_i| over the
    inputs (central differences of our own values): err / (u cond), u = 2^-53, is how many
    input roundings the error is worth.
  * the t at d = 2, 3 (exact scale mixture) against the bvt / tvt references.
  * QMC (normal d >= 4, t d >= 4): the actual error against the one-factor references and
    the reported error estimate (3 standard errors over 12 shifts), at atol 1e-6 and 1e-4.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

import stardust as sd
from stardust._core import stats as C

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "tests" / "stats" / "ref_mvcdf"
EPS = 2.0**-52
INF = math.inf


def mv(cov, x, df, atol=1e-6):
    cov = np.asarray(cov, float)
    x = np.atleast_2d(x)
    if cov.ndim == 2:
        cov = cov[None]
    bt = np.zeros(len(x), np.int64) if len(cov) == 1 else np.arange(len(x), dtype=np.int64)
    df = np.broadcast_to(np.asarray(df, float), (len(cov),)).copy()
    v, e = C.mv_cdf(sd.asarray(cov), sd.asarray(x), sd.asarray(bt), sd.asarray(df), atol, 0)
    return np.asarray(v), np.asarray(e)


def cond(ev, cols):
    """sum_i |d log P / d log x_i| for the input columns cols (a list of arrays)."""
    c = np.zeros(len(cols[0]))
    st = 1e-6
    for i in range(len(cols)):
        up = [x.copy() for x in cols]
        dn = [x.copy() for x in cols]
        up[i] *= 1 + st
        dn[i] *= 1 - st
        with np.errstate(divide="ignore", invalid="ignore"):
            c += np.abs((np.log(ev(*up)) - np.log(ev(*dn))) / (2 * st))
    return c


def cond_line(out, rel, c, mask):
    m = mask & np.isfinite(c) & np.isfinite(rel)
    q = rel[m] / (2.0**-53 * np.maximum(c[m], 1.0))
    out.append(
        f"\nerr / (u cond): max {q.max():.1f}, 99th percentile {np.percentile(q, 99):.1f} "
        f"({m.sum()} points with a finite cond); cond up to {c[m].max():.1e}."
    )


BINS = [(0, 1), (1, 5), (5, 10), (10, 18), (18, 30), (30, 50), (50, 100), (100, 200), (200, 308)]


def table_by_size(out, v, hi, lo, masks):
    """Rows: the -log10 P bins; columns: the named masks (max relative error, ratio)."""
    norm = hi >= 1e-300
    rel = np.full(len(hi), np.nan)
    rel[norm] = np.abs((v[norm] - hi[norm]) - lo[norm]) / hi[norm]
    lp = np.full(len(hi), np.inf)
    lp[norm] = -np.log10(hi[norm])
    out.append(
        "| -log10 P | " + " | ".join(f"{name}: n, max rel, max rel / (eps max(1, -ln P))" for name in masks) + " |"
    )
    out.append("|---|" + "---|" * len(masks))
    for a, b in BINS:
        cells = []
        for m in masks.values():
            s = m & norm & (lp >= a) & (lp < b)
            if s.sum() == 0:
                cells.append("-")
                continue
            ratio = rel[s] / (EPS * np.maximum(1.0, lp[s] * math.log(10)))
            cells.append(f"{s.sum()}, {rel[s].max():.1e}, {ratio.max():.1f}")
        out.append(f"| [{a}, {b}) | " + " | ".join(cells) + " |")
    under = ~norm
    if under.any():
        out.append(f"\nBelow 1e-300 ({under.sum()} points): max |v - ref| = {np.abs(v[under] - hi[under]).max():.1e}.")
    return rel


def bvn_section(out):
    z = np.load(REF / "bvn.npz")
    h, k, r, hi, lo, reg = z["h"], z["k"], z["r"], z["hi"], z["lo"], z["region"]
    M = len(h)
    S = np.tile(np.eye(2), (M, 1, 1))
    S[:, 0, 1] = S[:, 1, 0] = r
    sd.set_num_threads(1)
    t = time.perf_counter()
    v, _ = mv(S, np.c_[h, k], INF)
    t = time.perf_counter() - t
    out.append(f"## d = 2 ({M} points, {t / M * 1e6:.2f} us per point on one thread)\n")
    out.append(
        "Regions: the grid h, k in {-37, ..., 10} x 20 values of r in (-1, 1); random h, k in "
        "[-10, 6], r in (-1, 1); h + k within 1e-12..1 of 0 with r < 0; |r| = 1 - 1e-12..1e-2.\n"
    )
    masks = {"r >= 0": r >= 0, "r < 0": r < 0, "h + k ~ 0, r < 0": reg == 2, "|r| ~ 1": reg == 3}
    rel = table_by_size(out, v, hi, lo, masks)
    for t in (1e-10, 1e-15, 1e-18):
        out.append(f"\nMax relative error where P >= {t:.0e}: {np.nanmax(rel[hi >= t]):.2e} (every r).")

    def ev(H, K, RR):
        S2 = np.tile(np.eye(2), (len(H), 1, 1))
        S2[:, 0, 1] = S2[:, 1, 0] = RR
        return mv(S2, np.c_[H, K], INF)[0]

    cond_line(out, rel, cond(ev, [h, k, r]), hi >= 1e-300)
    return rel


def tvn_section(out):
    z = np.load(REF / "tvn.npz")
    h, R, hi, lo, reg = z["h"], z["r"], z["hi"], z["lo"], z["region"]
    M = len(h)
    S = np.tile(np.eye(3), (M, 1, 1))
    for q, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        S[:, a, b] = S[:, b, a] = R[:, q]
    sd.set_num_threads(1)
    t = time.perf_counter()
    v, _ = mv(S, h, INF)
    t = time.perf_counter() - t
    negs = (R < 0).sum(axis=1)
    out.append(f"\n## d = 3 ({M} points, {t / M * 1e6:.2f} us per point on one thread)\n")
    out.append(
        "Regions: random correlation matrices with h in [-7, 5]; the same with h in [-20, -5]; "
        "three negative correlations; two negative; smallest eigenvalue 1e-9..1e-3.\n"
    )
    masks = {"<= 1 negative r": negs <= 1, "2 negative": negs == 2, "3 negative": negs == 3, "near-singular": reg == 4}
    rel = table_by_size(out, v, hi, lo, masks)
    norm = hi >= 1e-300
    out.append(
        f"\nMax relative error, at most one negative r, P >= 1e-15: "
        f"{np.nanmax(rel[norm & (negs <= 1) & (hi >= 1e-15)]):.2e}"
    )

    def ev(H1, H2, H3, R1, R2, R3):
        S3 = np.tile(np.eye(3), (len(H1), 1, 1))
        for q, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            S3[:, a, b] = S3[:, b, a] = (R1, R2, R3)[q]
        return mv(S3, np.c_[H1, H2, H3], INF)[0]

    c = cond(ev, [h[:, 0], h[:, 1], h[:, 2], R[:, 0], R[:, 1], R[:, 2]])
    out.append(
        f"\nMax relative error, any r, P >= 1e-10 and cond <= 100: "
        f"{np.nanmax(rel[norm & (hi >= 1e-10) & (c <= 100)]):.2e}"
    )
    cond_line(out, rel, c, norm)
    for name, m in masks.items():
        s = m & norm
        if s.any():
            ab = np.abs(v[s] - hi[s])
            out.append(
                f"\n{name}: max relative {np.nanmax(rel[s]):.1e}, where P >= 1e-18 "
                f"{np.nanmax(rel[s & (hi >= 1e-18)]) if (s & (hi >= 1e-18)).any() else math.nan:.1e}; "
                f"max absolute {ab.max():.1e}"
            )
    return rel


def t_section(out):
    out.append("\n## The t at d = 2, 3: the exact scale mixture\n")
    out.append("| set | points | max rel | max rel, P >= 1e-15 | us per point (1 thread, mean) | max us |")
    out.append("|---|---|---|---|---|---|")
    rows = {}
    for name, d in (("bvt", 2), ("tvt", 3)):
        path = REF / f"{name}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        h, R, nu, hi, lo = z["h"], z["r"], z["nu"], z["hi"], z["lo"]
        sd.set_num_threads(1)
        rel, ts = [], []
        for i in range(len(h)):
            S = np.eye(d)
            if d == 2:
                S[0, 1] = S[1, 0] = R[i, 0]
            else:
                for q, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                    S[a, b] = S[b, a] = R[i, q]
            t = time.perf_counter()
            v, e = mv(S, h[i], nu[i])
            ts.append(time.perf_counter() - t)
            assert e[0] == 0
            rel.append(abs((v[0] - hi[i]) - lo[i]) / hi[i])
        rel, ts = np.array(rel), np.array(ts)
        out.append(
            f"| {name} | {len(h)} | {rel.max():.1e} | {rel[hi >= 1e-15].max():.1e} | "
            f"{ts.mean() * 1e6:.0f} | {ts.max() * 1e6:.0f} |"
        )
        rows[name] = (nu, rel)
    for name, (nu, rel) in rows.items():
        out.append(f"\n{name} by df: " + ", ".join(f"{v:g}: {rel[nu == v].max():.1e}" for v in np.unique(nu)))


def factor_rows(path, t):
    z = np.load(path)
    h, lam, d, nu, ref = z["h"], z["lam"], z["d"], z["nu"], z["hi"]
    rows = []
    for atol in (1e-6, 1e-4):
        for i in range(len(d)):
            n = d[i]
            S = np.outer(lam[i, :n], lam[i, :n])
            np.fill_diagonal(S, 1.0)
            v, e = mv(S, h[i, :n], nu[i] if t else INF, atol)
            rows.append((atol, n, abs(v[0] - ref[i]), e[0]))
    return np.array(rows)


def qmc_section(out):
    sd.set_num_threads(10)
    for name, t in (("normal", False), ("t", True)):
        r = factor_rows(REF / f"factor_{'t' if t else 'n'}.npz", t)
        r = r[r[:, 1] >= 4]  # (the t's d = 2, 3 are exact)
        out.append(f"\n## QMC, {name}: actual error vs the estimate (one-factor references)\n")
        out.append("| atol | d | points | max actual | max estimate | actual > estimate | max actual / atol |")
        out.append("|---|---|---|---|---|---|---|")
        for atol in (1e-6, 1e-4):
            for n in np.unique(r[:, 1]):
                s = (r[:, 0] == atol) & (r[:, 1] == n)
                a, e = r[s, 2], r[s, 3]
                out.append(
                    f"| {atol:.0e} | {int(n)} | {s.sum()} | {a.max():.1e} | {e.max():.1e} | "
                    f"{int((a > e).sum())} | {a.max() / atol:.2f} |"
                )


def main():
    out = [
        "# Multivariate normal and t cdf: accuracy (bench/stats/t9/cdf_accuracy.py)\n",
        "Generated by bench/stats/t9/cdf_accuracy.py against tests/stats/ref_mvcdf/ "
        "(tools/mvcdf_ref.py). eps = 2^-52.\n",
    ]
    bvn_section(out)
    tvn_section(out)
    t_section(out)
    qmc_section(out)
    out.append(
        "\nThe t's chi tables (csrc/stats/mvcdf.hpp ChiTable) against the exact inverse "
        "(gammaincinv) on the same lattice, 2e5 evaluations, d = 4..100, df = 1..30: "
        "max |difference| 8.3e-17 (a C++ harness linking the header; r itself is within "
        "1.4e-14 relative over 512k dense samples)."
    )
    text = "\n".join(out) + "\n"
    (Path(__file__).with_suffix(".md")).write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
