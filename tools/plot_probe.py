"""Probe the text pipeline of sd.plot (docs/design/plot.md §3.2). Standard library only.

Measures, on this machine's TeX Live:
  1. the two typesetting paths (pdfLaTeX with a dumped format, XeLaTeX for Hangul) and their time,
  2. whether the same Latin string has the same box in both paths,
  3. whether tick numbers laid out by hand from single-glyph boxes land where LaTeX puts them,
  4. the outline conversion (dvisvgm) time per call and per page.

    python tools/plot_probe.py [--work DIR]

Prints a report; writes nothing outside DIR (default: a fresh temporary directory).
"""

import argparse
import re
import subprocess
import tempfile
import time
from pathlib import Path

SD_GOTHIC = "/System/Library/Fonts/"
KOPUB = str(Path.home() / "Library/Fonts") + "/"

MEASURE = (
    r"\pagestyle{empty}\newbox\b"
    r"\def\one#1{\setbox\b\hbox{#1}\typeout{BOX \the\wd\b,\the\ht\b,\the\dp\b}\shipout\box\b}"
)

PDF_PREAMBLE = {
    "serif": r"\usepackage[T1]{fontenc}\usepackage{newtxtext}\usepackage{newtxmath}",
    "sans": r"\usepackage[T1]{fontenc}\usepackage{tgheros}\renewcommand\familydefault{\sfdefault}\usepackage{newtxsf}",
}
XE_PREAMBLE = {
    "serif": r"\usepackage{newtxtext}\usepackage{newtxmath}\usepackage[no-math]{fontspec}"
    r"\usepackage{xeCJK}\xeCJKsetup{CJKspace=true}"
    r"\setCJKmainfont{KoPubWorld Batang_Pro Light.otf}[Path=" + KOPUB + r",BoldFont={KoPubWorld Batang_Pro Bold.otf}]",
    "sans": r"\usepackage{newtxsf}\usepackage[no-math]{fontspec}\usepackage{xeCJK}\xeCJKsetup{CJKspace=true}"
    r"\setmainfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,"
    r"BoldItalicFont=*-bolditalic,Ligatures=TeX]"
    r"\setCJKsansfont{AppleSDGothicNeo.ttc}[Path=" + SD_GOTHIC + r",FontIndex=0,BoldFont=*,BoldFeatures={FontIndex=6}]"
    r"\setCJKmainfont{AppleSDGothicNeo.ttc}[Path=" + SD_GOTHIC + r",FontIndex=0,BoldFont=*,BoldFeatures={FontIndex=6}]",
}

LATIN = [
    r"Residuals",
    r"Fitted values",
    r"$x$",
    r"$\sin x$",
    r"$-1.5$",
    r"$10^{-3}$",
    r"$0.25$",
    r"$p < 0.001$",
    r"$\hat\beta_1 = 0.4217$",
    r"$\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt\pi}{2}$",
    r"Group A",
    r"$R^2 = 0.804$",
    r"$\Gamma(z)$, $J_\nu(x)$",
    r"\textbf{A}",
]
HANGUL = [r"잔차", r"표준화 잔차", r"적합값 $\hat y$", r"그룹 A"]
NUMBERS = ["0", "1", "-1.5", "0.25", "12.75", "-0.001", "1000", "3.14159", "-273.15", "8", "47", "71", "11", "0.4217"]
EXPONENTS = [("10", "-3"), ("10", "2"), ("10", "-12"), ("2", "8")]


def run(cmd, cwd):
    t = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")
    return time.perf_counter() - t, p


def document(preamble, strings, dump=False, body_only=False):
    head = r"\documentclass{article}\usepackage{amsmath}" + preamble + MEASURE
    body = "\n".join(rf"\one{{{s}}}" for s in strings)
    if body_only:  # the dumped format already holds the preamble
        return "\\begin{document}\n" + body + "\n\\end{document}\n"
    return head + (r"\endofdump" if dump else "") + "\n\\begin{document}\n" + body + "\n\\end{document}\n"


def boxes(log):
    text = Path(log).read_text(errors="replace")
    errors = re.findall(r"^! .*", text, flags=re.M)
    out = [tuple(float(v[:-2]) for v in m.split(",")) for m in re.findall(r"^BOX (\S+)", text, flags=re.M)]
    return out, errors


class Path2:
    def __init__(self, work, name, engine, preamble):
        self.work, self.name, self.engine, self.preamble = work, name, engine, preamble
        self.fmt = None

    def dump(self):
        (self.work / f"{self.name}_fmt.tex").write_text(document(self.preamble, ["x"], dump=True))
        t, p = run(
            [
                self.engine,
                "-ini",
                "-interaction=batchmode",
                f"-jobname={self.name}_fmt",
                f"&{self.engine}",
                "mylatexformat.ltx",
                f"{self.name}_fmt.tex",
            ],
            self.work,
        )
        if (self.work / f"{self.name}_fmt.fmt").exists():
            self.fmt = f"{self.name}_fmt"
        return t

    def typeset(self, strings, job):
        job = f"{self.name}_{job}"
        (self.work / f"{job}.tex").write_text(document(self.preamble, strings, body_only=self.fmt is not None))
        cmd = [self.engine, "-interaction=batchmode"]
        if self.engine == "xelatex":
            cmd.append("-no-pdf")
        if self.fmt:
            cmd.append(f"-fmt={self.fmt}")
        t, _ = run(cmd + [f"{job}.tex"], self.work)
        got, errors = boxes(self.work / f"{job}.log")
        return t, got, errors, job + (".xdv" if self.engine == "xelatex" else ".dvi")


def outlines(work, dvi, pages, extra=()):
    t, p = run(
        ["dvisvgm", "--no-fonts", "--exact-bbox", *extra, f"-p{pages}", "-o", f"{Path(dvi).stem}_%p.svg", dvi], work
    )
    return t, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work")
    args = ap.parse_args()
    work = Path(args.work or tempfile.mkdtemp(prefix="plot_probe_"))
    work.mkdir(parents=True, exist_ok=True)
    print("work:", work)

    for family in ("serif", "sans"):
        print(f"\n== {family} ==")
        pdf = Path2(work, f"{family}_pdf", "latex", PDF_PREAMBLE[family])
        xe = Path2(work, f"{family}_xe", "xelatex", XE_PREAMBLE[family])

        # 1. time: plain, dumped format
        t_plain = min(pdf.typeset(LATIN, "plain")[0] for _ in range(3))
        t_dump = pdf.dump()
        runs = [pdf.typeset(LATIN, "fmt") for _ in range(3)]
        t_fmt, pdf_boxes, errors, dvi = min(runs, key=lambda r: r[0])
        print(
            f"pdfLaTeX {len(LATIN)} strings: plain {t_plain * 1e3:.0f} ms, dumped format {t_fmt * 1e3:.0f} ms "
            f"(dump once {t_dump * 1e3:.0f} ms, format {'ok' if pdf.fmt else 'FAILED'}), errors {errors[:2]}"
        )
        t_one = min(pdf.typeset(LATIN[:1], "one")[0] for _ in range(3))
        print(f"pdfLaTeX 1 string with format: {t_one * 1e3:.0f} ms")

        # 2. the XeLaTeX path: same Latin boxes? Hangul?
        runs = [xe.typeset(LATIN + HANGUL, "all") for _ in range(2)]
        t_xe, xe_boxes, errors, xdv = min(runs, key=lambda r: r[0])
        print(f"XeLaTeX {len(LATIN) + len(HANGUL)} strings: {t_xe * 1e3:.0f} ms, errors {errors[:3]}")
        worst = 0.0
        for s, a, b in zip(LATIN, pdf_boxes, xe_boxes):
            d = max(abs(x - y) for x, y in zip(a, b))
            worst = max(worst, d)
            if d > 1e-4:
                print(f"   box differs {d:.5f} pt: {s}  pdf {a}  xe {b}")
        print(f"Latin boxes pdfLaTeX vs XeLaTeX: worst difference {worst:.5f} pt over {len(LATIN)} strings")
        for s, b in zip(HANGUL, xe_boxes[len(LATIN) :]):
            print(f"   Hangul box {b}: {s}")

        # 3. numbers laid out by hand from single-glyph boxes
        glyphs = list("0123456789") + ["-", "."]
        single = [f"${g}$" for g in glyphs] + [rf"$\scriptstyle {g}$" for g in glyphs]
        whole = [f"${n}$" for n in NUMBERS] + [f"${b}^{{{e}}}$" for b, e in EXPONENTS]
        _, got, errors, _ = pdf.typeset(single + whole, "num")
        w = {g: got[i][0] for i, g in enumerate(glyphs)}
        ws = {g: got[len(glyphs) + i][0] for i, g in enumerate(glyphs)}
        worst = 0.0
        for n, box in zip(NUMBERS, got[len(single) :]):
            mine = sum(w[c] for c in n)
            worst = max(worst, abs(mine - box[0]))
            if abs(mine - box[0]) > 1e-4:
                print(f"   number {n}: by hand {mine:.5f}, LaTeX {box[0]:.5f}")
        print(f"plain numbers by hand vs LaTeX: worst width difference {worst:.5f} pt over {len(NUMBERS)}")
        for (b, e), box in zip(EXPONENTS, got[len(single) + len(NUMBERS) :]):
            mine = sum(w[c] for c in b) + sum(ws[c] for c in e)
            print(
                f"   {b}^{{{e}}}: base + script glyphs {mine:.5f}, LaTeX {box[0]:.5f}, rest {box[0] - mine:.5f} pt "
                f"(height {box[1]:.5f})"
            )

        # 4. outlines
        t_all, rc = outlines(work, dvi, "1-")
        t_1, _ = outlines(work, dvi, "1")
        print(f"dvisvgm (DVI): {len(LATIN)} pages {t_all * 1e3:.0f} ms, 1 page {t_1 * 1e3:.0f} ms, rc {rc}")
        for extra in (
            ["--no-specials"],
            ["--no-mktexmf"],
            ["--no-specials", "--no-mktexmf", "--no-styles"],
            ["--bbox=min"],
        ):
            print(f"   dvisvgm 1 page with {' '.join(extra)}: {outlines(work, dvi, '1', extra)[0] * 1e3:.0f} ms")
        t_all, rc = outlines(work, xdv, "1-")
        t_1, _ = outlines(work, xdv, f"{len(LATIN) + 1}")
        print(
            f"dvisvgm (XDV): {len(LATIN) + len(HANGUL)} pages {t_all * 1e3:.0f} ms, "
            f"1 Hangul page {t_1 * 1e3:.0f} ms, rc {rc}"
        )


if __name__ == "__main__":
    main()
