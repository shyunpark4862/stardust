"""Text for sd.plot: LaTeX typesets every string, the glyph outlines are drawn as paths
(docs/design/plot.md §3.2, decisions PL7, PL8, PL14, PL17, PL22, PL23).

A string is typeset once, at TeX's 10pt, and kept as a layout: its box and the glyphs (font,
code, position) and rules TeX placed, in scaled points, read from the DVI file. The outline of
a glyph is fetched once per font and code (dvisvgm) and kept. Both are stored on disk, so a
string seen before costs no TeX run and a new string made of known glyphs costs one TeX run and
no outline conversion. Other sizes are the 10pt layout scaled.

Two paths: a string without Hangul or other CJK goes through pdfLaTeX with a dumped format,
one with it through XeLaTeX (system fonts). Numbers ("$-1.5$", "$10^{-3}$") are laid out here
from the layouts of their single characters, which is what TeX does with them (no kerning
between math digits; tests/plot/test_text_tex.py compares with TeX glyph by glyph).

The families (PL8): 'serif' is TeX Gyre Termes (newtxtext), KoPubWorld Batang, newtxmath;
'sans' is TeX Gyre Heros, Apple SD Gothic Neo, newtxsf.
"""

import contextlib
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
import xml.etree.ElementTree as ET
from array import array
from concurrent.futures import ThreadPoolExecutor

from . import _dvi, _svgpath

FORMAT = 1  # of the cache files; part of the cache folder's stamp
FAMILIES = ("serif", "sans")
SP_TO_BP = 72.0 / 72.27 / 65536.0
UNIT_BP = 10.0 * 72.0 / 72.27  # TeX's 10pt, the size every string is typeset at
TEN_PT = 10 * 65536
SCRIPTSPACE = 32768  # \scriptspace, 0.5pt, which TeX adds after a superscript

_TEX_DIRS = ("/Library/TeX/texbin",)
_TOOLS = ("latex", "xelatex", "tex", "xetex", "dvisvgm", "kpsewhich")
_KOPUB = str(pathlib.Path.home() / "Library" / "Fonts") + "/"
_SYSTEM = "/System/Library/Fonts/"

_MEASURE = (
    r"\pagestyle{empty}\newbox\sdb"
    r"\def\one#1{\setbox\sdb\hbox{#1}\typeout{SDBOX \number\wd\sdb,\number\ht\sdb,\number\dp\sdb}\shipout\box\sdb}"
)
_HEAD = r"\documentclass{article}\usepackage{amsmath}"
_PDF = {
    "serif": r"\usepackage[T1]{fontenc}\usepackage{newtxtext}\usepackage{newtxmath}",
    "sans": r"\usepackage[T1]{fontenc}\usepackage{tgheros}\renewcommand\familydefault{\sfdefault}\usepackage{newtxsf}",
}
_XE = {
    "serif": r"\usepackage{newtxtext}\usepackage{newtxmath}\usepackage[no-math]{fontspec}"
    r"\usepackage{xeCJK}\xeCJKsetup{CJKspace=true,CJKecglue={}}"
    r"\setCJKmainfont{KoPubWorld Batang_Pro Light.otf}[Path=" + _KOPUB + r",BoldFont={KoPubWorld Batang_Pro Bold.otf}]",
    "sans": r"\usepackage[T1]{fontenc}\usepackage{tgheros}\usepackage{newtxsf}\usepackage[no-math]{fontspec}"
    r"\usepackage{xeCJK}\xeCJKsetup{CJKspace=true,CJKecglue={}}"
    r"\setmainfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,"
    r"BoldItalicFont=*-bolditalic,Ligatures=TeX]"
    r"\setCJKmainfont{AppleSDGothicNeo.ttc}[Path=" + _SYSTEM + r",FontIndex=0,BoldFont=*,BoldFeatures={FontIndex=6}]",
}
_PLAIN = (
    r"\newbox\sdb\def\one#1{\setbox\sdb\hbox{#1}"
    r"\immediate\write16{SDBOX \number\wd\sdb,\number\ht\sdb,\number\dp\sdb}\shipout\box\sdb}"
)

# Typeset with the first batch of a new user cache, so that the outlines of the common glyphs are
# fetched in that one conversion and a later new label costs a TeX run only (PL23).
_LOWER = "abcdefghijklmnopqrstuvwxyz"
_WARM = (
    _LOWER.upper() + " " + _LOWER + r" 0123456789 .,:;!?()[]/-+=*'`\%\&\#",
    r"\textbf{" + _LOWER.upper() + _LOWER + "0123456789}",
    r"\textit{" + _LOWER.upper() + _LOWER + "}",
    "$"
    + _LOWER.upper()
    + _LOWER
    + r"0123456789\alpha\beta\gamma\delta\epsilon\varepsilon\zeta\eta\theta\vartheta\iota\kappa"
    r"\lambda\mu\nu\xi\pi\rho\sigma\tau\upsilon\phi\varphi\chi\psi\omega\Gamma\Delta\Theta\Lambda\Xi\Pi\Sigma\Phi\Psi\Omega$",
    r"$+-\times\cdot\div=<>\leq\geq\approx\neq\equiv\sim\propto\pm\mp\infty\partial\nabla\in\to\mapsto\ldots\cdots"
    r"\int\sum\prod\sqrt{x}\frac{1}{2}()[]\{\}|,.;:!/\prime\ell\hbar\circ\bullet\star\dagger\angle\perp\parallel$",
    "$x^{"
    + _LOWER.upper()
    + _LOWER
    + r"0123456789+-=()\alpha\beta\gamma\theta\lambda\mu\pi\sigma\omega\infty\prime,./*}$",
    r"$x^{y^{"
    + _LOWER
    + r"0123456789+-}}\hat{x}\bar{x}\tilde{x}\vec{x}\dot{x}\ddot{x}\mathrm{"
    + _LOWER.upper()
    + _LOWER
    + r"}"
    r"\mathbf{" + _LOWER.upper() + _LOWER + r"0123456789}\sin\cos\tan\log\ln\exp\lim\max\min$",
    r"$\displaystyle\int_0^1\sum_{i=1}^n\prod\frac{a}{b}\sqrt{\frac{a}{b}}\left(\frac{a}{b}\right)\left[\frac{a}{b}\right]$",
)

_CJK = re.compile("[\u1100-\u11ff\u2e80-\u9fff\ua960-\ua97f\uac00-\ud7ff\uf900-\ufaff\uff00-\uffef]")
_NUMBER = re.compile(r"\$(-?)(\d+)(?:\.(\d+))?\$")
_POWER = re.compile(r"\$(-?)(\d+)(?:\.(\d+))?\^\{(-?)(\d+)\}\$")
_BOX = re.compile(r"^SDBOX (-?\d+),(-?\d+),(-?\d+)", re.M)


class MissingTool(RuntimeError):
    """A TeX Live program is not installed."""


class TypesetError(ValueError):
    """TeX could not typeset a string: .string, .reason (TeX's first error line)."""

    def __init__(self, string, reason):
        super().__init__(f"LaTeX could not typeset {string!r}: {reason}")
        self.string, self.reason = string, reason


class FrozenMiss(LookupError):
    """A frozen cache (the tests' recorded one) does not hold a string."""


class Layout:
    """A typeset string at 10pt: the box (sp), the fonts [(key, size in sp)], the glyphs
    [(font index, code, h, v)] and the rules [(h, v of the bottom, width, height)], origin at the
    box's upper left corner, y down."""

    __slots__ = ("width", "height", "depth", "fonts", "glyphs", "rules")

    def __init__(self, width, height, depth, fonts, glyphs, rules):
        self.width, self.height, self.depth = width, height, depth
        self.fonts, self.glyphs, self.rules = fonts, glyphs, rules

    def box(self, size):
        """(width, height, depth) in bp at font size `size` (bp)."""
        k = size / UNIT_BP * SP_TO_BP
        return self.width * k, self.height * k, self.depth * k

    def to_json(self):
        return {"w": self.width, "h": self.height, "d": self.depth, "f": self.fonts, "g": self.glyphs, "r": self.rules}

    @classmethod
    def from_json(cls, o):
        return cls(
            o["w"], o["h"], o["d"], [tuple(f) for f in o["f"]], [tuple(g) for g in o["g"]], [tuple(r) for r in o["r"]]
        )


_EMPTY = Layout(0, 0, 0, [], [], [])

# ---- where the cache is

_lock = threading.RLock()
_root = None  # (path, frozen) set by cache(); None is the user's cache
_engines = {}
_tools = {}


def tool(name):
    """The path of a TeX Live program, or MissingTool."""
    path = _tools.get(name)
    if path is None:
        path = shutil.which(name)
        if path is None:
            for d in _TEX_DIRS:
                if os.access(os.path.join(d, name), os.X_OK):
                    path = os.path.join(d, name)
                    break
        if path is None:
            raise MissingTool(
                "sd.plot needs 'latex', 'xelatex', 'tex', 'xetex', 'dvisvgm' and 'kpsewhich' (TeX Live) on PATH; "
                f"{name!r} was not found"
            )
        _tools[name] = path
    return path


def _stamp():
    """What the cached layouts depend on: the TeX programs, the preambles, the file format."""
    h = hashlib.sha256()
    h.update(repr((FORMAT, _HEAD, _MEASURE, _PDF, _XE, _PLAIN)).encode())
    for name in ("latex", "xelatex", "dvisvgm"):
        real = os.path.realpath(tool(name))
        st = os.stat(real)
        h.update(f"{real}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _default_root():
    return pathlib.Path.home() / "Library" / "Caches" / "stardust" / "plot" / _stamp()


@contextlib.contextmanager
def cache(path, *, frozen=False):
    """Within the block the text cache is the folder `path`. frozen: a string the folder does not
    hold raises FrozenMiss instead of running TeX (the tests' recorded cache)."""
    global _root
    with _lock:
        old = _root
        _root = (pathlib.Path(path), frozen)
    try:
        yield
    finally:
        with _lock:
            _root = old


def engine(family):
    """The Engine of a font family for the cache in force."""
    if family not in FAMILIES:
        raise ValueError(f"'font' must be 'serif' or 'sans', not {family!r}")
    with _lock:
        root, frozen = _root if _root is not None else (None, False)
        if root is None:
            root = _default_root()
        key = (family, str(root), frozen)
        e = _engines.get(key)
        if e is None:
            e = _engines[key] = Engine(family, root, frozen)
            e.warm = _root is None
        return e


# ---- running TeX


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL)


def _log_error(log, strings):
    """The TypesetError a failed run's log tells: TeX's first '! ' line, and the string TeX was
    in (one box line is written per string done, so their count is the failing one's index)."""
    text = log.read_text(errors="replace") if log.exists() else ""
    m = re.search(r"^! (.*)$", text, re.M)
    reason = "! " + m.group(1).strip() if m else "TeX stopped without an error message"
    k = len(_BOX.findall(text))
    return TypesetError(strings[k] if k < len(strings) else "?", reason)


def _check(s):
    """A string that would swallow its neighbours in the batch is refused before TeX sees it."""
    depth = 0
    for m in re.finditer(r"\\.|[{}%]", s, re.S):
        t = m.group()
        if t == "%":
            raise TypesetError(s, "an unescaped % (write \\%)")
        if t == "{":
            depth += 1
        elif t == "}":
            depth -= 1
            if depth < 0:
                break
    if depth != 0:
        raise TypesetError(s, "unbalanced braces")


def _boxes(log):
    return [tuple(int(v) for v in m) for m in _BOX.findall(log.read_text(errors="replace"))]


class _Tfm:
    """TFM widths by font name, found with kpsewhich (one call for all the names asked)."""

    def __init__(self):
        self.widths = {}

    def load(self, names):
        names = sorted(n for n in names if n not in self.widths)
        if not names:
            return
        out = _run([tool("kpsewhich"), *[n + ".tfm" for n in names]], None).stdout.split("\n")
        paths = {pathlib.Path(p).stem: p for p in out if p.strip()}
        for n in names:
            if n not in paths:
                raise TypesetError("?", f"font metric {n}.tfm was not found")
            self.widths[n] = _dvi.tfm_widths(pathlib.Path(paths[n]).read_bytes())

    def __call__(self, name):
        if name not in self.widths:
            self.load([name])
        return self.widths[name]


_tfm = _Tfm()


def _read_dvi(path, boxes, strings):
    """[(Layout, [Font])] of a DVI or XDV file with one string per page."""
    data = path.read_bytes()
    _tfm.load(_dvi.font_names(data))
    pages = _dvi.pages(data, _tfm)
    if len(pages) != len(strings) or len(boxes) != len(strings):
        raise TypesetError("?", f"TeX made {len(pages)} pages for {len(strings)} strings")
    out = []
    for page, (w, h, d) in zip(pages, boxes):
        fonts = [(f.key, f.size) for f in page.fonts]
        out.append((Layout(w, h, d, fonts, page.glyphs, page.rules), page.fonts))
    return out


# ---- the engine


class Engine:
    def __init__(self, family, root, frozen):
        self.family, self.root, self.frozen = family, pathlib.Path(root), frozen
        self.dir = self.root / family
        self.glyph_dir = self.root / "glyphs"
        self.strings = {}  # string -> Layout
        self.glyphs = {}  # font key -> {code: (ops, xy)} at 10pt, bp, y down, origin the glyph's
        self.numbers = {}  # number string -> Layout laid out here
        self._packed = {}
        self.lock = threading.RLock()
        self.tex_runs = 0  # typesetting runs so far (the tests count them)
        self.warm = False  # the user's cache also fetches the common glyphs with its first batch
        f = self.dir / "strings.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                if line:
                    o = json.loads(line)
                    self.strings[o["s"]] = Layout.from_json(o)

    # -- public

    def layout(self, strings):
        """The layouts of `strings` (a list), typesetting in one batch what is not known."""
        with self.lock:
            need = []
            for s in strings:
                if not s.strip():
                    continue  # nothing to typeset: an empty box
                for a in _atoms(s) or (s,):
                    if a not in self.strings and a not in need:
                        need.append(a)
            if need:
                for s in need:
                    _check(s)
                if self.frozen:
                    raise FrozenMiss(f"the frozen text cache does not hold {need[0]!r} ({self.family})")
                self._typeset(need)
            return [self._get(s) for s in strings]

    def outline(self, font, code):
        """(ops, xy) of a glyph at 10pt; the font's file is read on first use."""
        with self.lock:
            store = self._store(font)
            return store[str(code)]

    def packed(self, font, code):
        """A glyph's outline as the rasterizer takes it: (ops bytes, packed doubles)."""
        got = self._packed.get((font, code))
        if got is None:
            ops, xy = self.outline(font, code)
            got = self._packed[font, code] = (ops.encode(), array("d", xy).tobytes())
        return got

    def path(self, lay, size, x=0.0, y=0.0):
        """The filled outline of a layout at font size `size` (bp) with its baseline's left end at
        (x, y): (ops, xy), y down."""
        k0 = size / UNIT_BP
        ops, xy = [], []
        top = lay.height
        for fi, code, h, v in lay.glyphs:
            key, fsize = lay.fonts[fi]
            gops, gxy = self.outline(key, code)
            kf = fsize / TEN_PT * k0
            ox, oy = x + h * SP_TO_BP * k0, y + (v - top) * SP_TO_BP * k0
            ops.append(gops)
            for i in range(0, len(gxy), 2):
                xy.append(ox + gxy[i] * kf)
                xy.append(oy + gxy[i + 1] * kf)
        k = SP_TO_BP * k0
        for h, v, w, ht in lay.rules:
            x0, y1 = x + h * k, y + (v - top) * k
            x1, y0 = x0 + w * k, y1 - ht * k
            ops.append("MLLLZ")
            xy += (x0, y0, x1, y0, x1, y1, x0, y1)
        return "".join(ops), xy

    # -- numbers laid out from their characters

    def _get(self, s):
        if not s.strip():
            return _EMPTY
        lay = self.strings.get(s)
        if lay is None:
            lay = self.numbers.get(s)
            if lay is None:
                lay = self.numbers[s] = self._number(s)
        return lay

    def _number(self, s):
        m = _POWER.fullmatch(s)
        if m:
            base = self._concat([f"${c}$" for c in _digits(m.group(1), m.group(2), m.group(3))])
            sup = self._concat([rf"$\scriptstyle {c}$" for c in _digits(m.group(4), m.group(5), None)])
            ref = self.strings["$1^{1}$"]
            one, small = self.strings["$1$"], self.strings[r"$\scriptstyle 1$"]
            raised = ref.height - small.height  # how far TeX lifts a superscript's baseline
            height = max(base.height, sup.height + raised)
            fonts = list(base.fonts)
            glyphs = [(fi, c, h, v - base.height + height) for fi, c, h, v in base.glyphs]
            for fi, c, h, v in sup.glyphs:
                f = sup.fonts[fi]
                if f not in fonts:
                    fonts.append(f)
                glyphs.append((fonts.index(f), c, base.width + h, v - sup.height + height - raised))
            assert one.depth == 0
            return Layout(base.width + sup.width + SCRIPTSPACE, height, base.depth, fonts, glyphs, [])
        m = _NUMBER.fullmatch(s)
        return self._concat([f"${c}$" for c in _digits(*m.groups())])

    def _concat(self, atoms):
        """The layouts of single characters side by side on one baseline."""
        lays = [self.strings[a] for a in atoms]
        height = max(lay.height for lay in lays)
        depth = max(lay.depth for lay in lays)
        fonts, glyphs, rules, x = [], [], [], 0
        for lay in lays:
            for fi, c, h, v in lay.glyphs:
                f = lay.fonts[fi]
                if f not in fonts:
                    fonts.append(f)
                glyphs.append((fonts.index(f), c, x + h, v - lay.height + height))
            for h, v, w, ht in lay.rules:
                rules.append((x + h, v - lay.height + height, w, ht))
            x += lay.width
        return Layout(x, height, depth, fonts, glyphs, rules)

    # -- typesetting what is missing

    def _typeset(self, need):
        self.dir.mkdir(parents=True, exist_ok=True)
        warm = self.dir / "warm"
        warming = self.warm and not warm.exists()
        if warming:
            need = list(need) + [s for s in _WARM if s not in need and s not in self.strings]
        latin = [s for s in need if not _CJK.search(s)]
        hangul = [s for s in need if _CJK.search(s)]
        with tempfile.TemporaryDirectory(prefix="sdplot_") as tmp:
            tmp = pathlib.Path(tmp)
            jobs = []
            with ThreadPoolExecutor(2) as pool:
                if latin:
                    jobs.append((latin, pool.submit(self._pdflatex, latin, tmp / "pdf")))
                if hangul:
                    jobs.append((hangul, pool.submit(self._xelatex, hangul, tmp / "xe")))
            done = [(strings, job.result()) for strings, job in jobs]
            fonts = {}
            for _, results in done:
                for lay, page_fonts in results:
                    for fi, code, _, _ in lay.glyphs:
                        fonts.setdefault(page_fonts[fi].key, (page_fonts[fi], set()))[1].add(code)
            self._calibrate(fonts, tmp)
        with open(self.dir / "strings.jsonl", "a") as f:
            for strings, results in done:
                for s, (lay, _) in zip(strings, results):
                    self.strings[s] = lay
                    f.write(json.dumps({"s": s, **lay.to_json()}, ensure_ascii=False, separators=(",", ":")) + "\n")
        if warming:
            warm.touch()  # only once the batch with the common glyphs went through

    def _format(self):
        """The dumped pdfLaTeX format of this family. It lives in the user's cache whatever cache
        is in force (a recorded cache holds layouts and outlines only), made once per stamp."""
        fmt = _default_root() / f"{self.family}.fmt"
        if not fmt.exists():
            fmt.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="sdplot_fmt_") as tmp:
                (pathlib.Path(tmp) / "f.tex").write_text(
                    _HEAD + _PDF[self.family] + _MEASURE + "\\endofdump\n\\begin{document}\\end{document}\n"
                )
                _run(
                    [
                        tool("latex"),
                        "-ini",
                        "-interaction=batchmode",
                        "-jobname=pdf",
                        "&latex",
                        "mylatexformat.ltx",
                        "f.tex",
                    ],
                    tmp,
                )
                made = pathlib.Path(tmp) / "pdf.fmt"
                if not made.exists():
                    raise TypesetError(
                        "?", "the LaTeX format could not be made (newtx, tgheros and mylatexformat are needed)"
                    )
                shutil.copy(made, fmt.with_suffix(".tmp"))
                os.replace(fmt.with_suffix(".tmp"), fmt)
        return fmt

    def _pdflatex(self, strings, tmp):
        tmp.mkdir()
        os.symlink(self._format(), tmp / "pdf.fmt")
        (tmp / "j.tex").write_text(
            "\\begin{document}\n" + "".join(f"\\one{{{s}}}\n" for s in strings) + "\\end{document}\n"
        )
        self.tex_runs += 1
        p = _run(
            [tool("latex"), "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "-fmt=pdf", "j.tex"], tmp
        )
        return self._collect(p, tmp, "j.dvi", strings)

    def _xelatex(self, strings, tmp):
        tmp.mkdir()
        (tmp / "j.tex").write_text(
            _HEAD
            + _XE[self.family]
            + _MEASURE
            + "\n\\begin{document}\n"
            + "".join(f"\\one{{{s}}}\n" for s in strings)
            + "\\end{document}\n"
        )
        self.tex_runs += 1
        p = _run(
            [tool("xelatex"), "-no-pdf", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "j.tex"], tmp
        )
        return self._collect(p, tmp, "j.xdv", strings)

    def _collect(self, p, tmp, out, strings):
        log = tmp / "j.log"
        if p.returncode != 0 or not (tmp / out).exists():
            raise _log_error(log, strings)
        text = log.read_text(errors="replace")
        m = re.search(r"^Missing character: There is no (\S+)", text, re.M)
        if m:
            ch = m.group(1)
            bad = next((s for s in strings if ch in s), "?")
            raise TypesetError(bad, f"the font has no character {ch}")
        return _read_dvi(tmp / out, _boxes(log), strings)

    # -- glyph outlines

    def _store(self, key):
        store = self.glyphs.get(key)
        if store is None:
            f = self.glyph_dir / _file_name(key)
            store = self.glyphs[key] = json.loads(f.read_text()) if f.exists() else {}
        return store

    def _calibrate(self, fonts, tmp):
        """Fetch the outlines of the glyphs not stored yet: fonts is {key: (Font, codes)}. A page
        per glyph, plain TeX for TFM fonts and plain XeTeX for native ones, then dvisvgm."""
        todo = {"tex": [], "xetex": []}
        for key, (font, codes) in fonts.items():
            store = self._store(key)
            miss = sorted(c for c in codes if str(c) not in store)
            if miss:
                todo["xetex" if font.native else "tex"].append((font, miss))
        if not (todo["tex"] or todo["xetex"]):
            return
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(self._glyph_pages, eng, todo[eng], tmp / f"cal_{eng}") for eng in todo if todo[eng]]
        self.glyph_dir.mkdir(parents=True, exist_ok=True)
        touched = set()
        for job in jobs:
            for key, code, outline in job.result():
                self._store(key)[str(code)] = outline
                touched.add(key)
        for key in touched:
            f = self.glyph_dir / _file_name(key)
            f.with_suffix(".tmp").write_text(json.dumps(self.glyphs[key], separators=(",", ":"), sort_keys=True))
            os.replace(f.with_suffix(".tmp"), f)

    def _glyph_pages(self, eng, todo, tmp):
        tmp.mkdir()
        lines, items = [_PLAIN], []
        for n, (font, codes) in enumerate(todo):
            if font.native:
                where = f"[{font.path}:{font.index}]" if font.index else f"[{font.path}]"
                lines.append(f'\\font\\sdf{_alpha(n)}="{where}" at 10pt')
            else:
                lines.append(f"\\font\\sdf{_alpha(n)}={font.key} at 10pt")
            for c in codes:
                lines.append(
                    f"\\one{{\\sdf{_alpha(n)}" + (f"\\XeTeXglyph{c} }}" if font.native else f'\\char"{c:X} }}')
                )
                items.append((font.key, c))
        (tmp / "g.tex").write_text("\n".join(lines) + "\n\\bye\n")
        cmd = (
            [tool(eng), "-interaction=batchmode", "-no-shell-escape"]
            + (["-no-pdf"] if eng == "xetex" else [])
            + ["g.tex"]
        )
        _run(cmd, tmp)
        out = tmp / ("g.xdv" if eng == "xetex" else "g.dvi")
        boxes = _boxes(tmp / "g.log")
        if not out.exists() or len(boxes) != len(items):
            raise TypesetError("?", f"the glyph outlines of {todo[0][0].key} could not be typeset")
        _run([tool("dvisvgm"), "--no-fonts", "--no-specials", "-p1-", "-o", "p%p.svg", out.name], tmp)
        pages = {int(re.sub(r"\D", "", f.stem)): f for f in tmp.glob("p*.svg")}
        res = []
        for i, ((key, code), (_, h, _)) in enumerate(zip(items, boxes), 1):
            ops, xy = _page_outline(pages[i]) if i in pages else ("", [])
            dy = h * SP_TO_BP
            res.append((key, code, [ops, [round(v - dy, 6) if j % 2 else round(v, 6) for j, v in enumerate(xy)]]))
        return res


_ATOMS = tuple(f"${c}$" for c in "0123456789-.") + tuple(rf"$\scriptstyle {c}$" for c in "0123456789-.") + ("$1^{1}$",)


def _atoms(s):
    """The strings a number is laid out from (every digit, sign and point in both sizes, so one
    TeX run serves all numbers), or None for any other string."""
    return _ATOMS if _NUMBER.fullmatch(s) or _POWER.fullmatch(s) else None


def _digits(sign, whole, frac):
    return list(sign + whole + ("." + frac if frac else ""))


def _alpha(n):
    out = ""
    while True:
        out = chr(97 + n % 26) + out
        n = n // 26 - 1
        if n < 0:
            return out


def _file_name(key):
    return re.sub(r"[^A-Za-z0-9._-]", "_", key) + "-" + hashlib.sha256(key.encode()).hexdigest()[:8] + ".json"


_SVG = "{http://www.w3.org/2000/svg}"
_HREF = "{http://www.w3.org/1999/xlink}href"


def _page_outline(path):
    """Every filled shape of a dvisvgm page as one outline (ops, xy) in the page's bp. A <use>
    names a path of <defs> or another <use> there (dvisvgm writes a glyph shared by two fonts
    once and aliases it)."""
    root = ET.parse(path).getroot()
    defs = {e.get("id"): e for d in root.iter(_SVG + "defs") for e in d if e.get("id")}
    ops, xy = [], []

    def draw(e, m):
        tag = e.tag.removeprefix(_SVG)
        t = _svgpath.compose(m, _svgpath.matrix(e.get("transform")))
        if tag == "g":
            for child in e:
                draw(child, t)
        elif tag == "use":
            t = _svgpath.compose(t, (1.0, 0.0, 0.0, 1.0, float(e.get("x", 0)), float(e.get("y", 0))))
            target = defs.get(e.get(_HREF).lstrip("#"))  # a glyph without ink (a space) has none
            if target is not None:
                draw(target, t)
        elif tag == "path":
            o, c = _svgpath.parse(e.get("d"))
            ops.append(o)
            xy.extend(_svgpath.apply(t, c))
        elif tag == "rect":
            x, y, w, h = (float(e.get(k, 0)) for k in ("x", "y", "width", "height"))
            ops.append("MLLLZ")
            xy.extend(_svgpath.apply(t, [x, y, x + w, y, x + w, y + h, x, y + h]))

    for e in root:
        if e.tag != _SVG + "defs":
            draw(e, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    return "".join(ops), xy
