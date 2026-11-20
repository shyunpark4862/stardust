"""Console output of the speed runs, in English: summary blocks laid out like a regression
summary (78 columns, '=' rules, "Key:" on the left and the value on the right of each half),
small tables, a progress bar (tqdm) and event lines printed above it.

The log file next to a run file gets every line with a time stamp, and one plain line per
case instead of the bar. When stderr is not a terminal the plain lines go there too.
"""

from __future__ import annotations

import datetime
import re
import sys
import threading
import unicodedata

WIDTH = 78
HALF = 39
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def width(s: str) -> int:
    """Display width: East Asian wide and full-width characters take two columns."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def ljust(s: str, n: int) -> str:
    return s + " " * max(0, n - width(s))


def rjust(s: str, n: int) -> str:
    return " " * max(0, n - width(s)) + s


def fmt_duration(s: float) -> str:
    s = max(0, int(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def rule(ch: str = "=") -> str:
    return ch * WIDTH


def _cell(key: str, value: str, n: int) -> str | None:
    """'Key:' then the value right-aligned in n columns; None when it does not fit."""
    head = f"{key}:"
    if width(head) + 1 + width(value) > n:
        return None
    return head + rjust(value, n - width(head))


def block(title: str, pairs: list[tuple[str, str]], wide: list[tuple[str, str]] = ()) -> list[str]:
    """A titled block: `pairs` two to a row, then `wide` one to a row ('Key:' padded, then the
    value). A pair too long for half a row gets a row of its own."""
    lines = [title.center(WIDTH).rstrip(), rule()]
    halves: list[str] = []
    rows: list[str] = []
    for key, value in pairs:
        cell = _cell(key, value, HALF - 1)
        if cell is None:
            rows.append(_cell(key, value, WIDTH) or f"{key}: {value}")
        else:
            halves.append(cell)
    for i in range(0, len(halves), 2):
        left = halves[i]
        right = halves[i + 1] if i + 1 < len(halves) else ""
        lines.append((ljust(left, HALF) + " " + right).rstrip())
    lines += rows
    key_w = max((width(k) for k, _ in wide), default=0) + 2
    lines += [ljust(f"{k}:", key_w) + v for k, v in wide]
    lines.append(rule())
    return lines


def table(headers: list[str], rows: list[list[str]], align: str) -> list[str]:
    """A table with a '-' rule under the header; align: one 'l' or 'r' per column. The first
    column takes the width the others leave of WIDTH."""
    ws = [max([width(h)] + [width(r[i]) for r in rows]) for i, h in enumerate(headers)]
    rest = sum(ws[1:]) + 3 * (len(ws) - 1)
    ws[0] = max(ws[0], WIDTH - rest)

    def fmt(cells):
        out = []
        for i, c in enumerate(cells):
            out.append(ljust(c, ws[i]) if align[i] == "l" else rjust(c, ws[i]))
        return "   ".join(out).rstrip()

    return [fmt(headers), rule("-")] + [fmt(r) for r in rows]


def wrap(items: list[tuple[str, str]], indent: int, width_: int, sep: str = " · ") -> list[str]:
    """Join (plain, shown) items with `sep` into lines of at most width_ columns (measured on
    the plain text: the shown text may carry colors), each starting with `indent` spaces."""
    lines, cur, cur_w = [], "", 0
    for plain, shown in items:
        w = width(plain)
        if cur and cur_w + width(sep) + w > width_:
            lines.append(cur)
            cur, cur_w = "", 0
        if not cur:
            cur, cur_w = " " * indent + shown, indent + w
        else:
            cur, cur_w = cur + sep + shown, cur_w + width(sep) + w
    if cur:
        lines.append(cur)
    return lines


def split_label(text: str, indent: int, width_: int) -> list[str]:
    """A long case label broken before ', ' or ' [' so no line is wider than width_."""
    lines, first = [], True
    while width(text) + (0 if first else indent) > width_:
        room = width_ - (0 if first else indent)
        cut = max(text.rfind(", ", 0, room), text.rfind(" [", 0, room))
        if cut <= 0:
            break
        cut += 1 if text[cut] == "," else 0
        lines.append(("" if first else " " * indent) + text[:cut].rstrip())
        text, first = text[cut:].lstrip(), False
    lines.append(("" if first else " " * indent) + text)
    return lines


def case_label(case) -> str:
    """'area · name [size]': a case without its group (the table section)."""
    label = f"{case.area} · {case.name}"
    return f"{label} [{case.size}]" if case.size else label


class Console:
    """Where a run's lines go: the terminal (with a progress bar when it is one) and the log."""

    def __init__(self, log=None, show: bool = True, stream=None):
        self.log = log
        self.show = show
        self.stream = stream or sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.bar = None
        self.lock = threading.RLock()

    def _log(self, text: str) -> None:
        if self.log is not None and text:
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            for line in _ANSI.sub("", text).splitlines() or [""]:
                self.log.write(f"{stamp} {line}\n")

    def print(self, text: str = "") -> None:
        with self.lock:
            self._log(text)
            if not self.show:
                return
            if self.bar is not None:
                self.bar.write(text, file=self.stream)
            else:
                print(text, file=self.stream, flush=True)

    def width(self) -> int:
        """The width result lines are laid out in: the terminal's, kept within 60-100 columns
        (100 when the stream is not a terminal)."""
        if not self.tty:
            return 100
        import shutil

        return max(60, min(100, shutil.get_terminal_size((100, 24)).columns))

    def color(self, text: str, code: str) -> str:
        """text in an ANSI color on a terminal ('31' red, '32' green, '2' dim); plain otherwise."""
        return f"\x1b[{code}m{text}\x1b[0m" if self.tty and self.show else text

    def lines(self, lines: list[str]) -> None:
        self.print("\n".join(lines))

    def event(self, tag: str, what: str, detail: str = "") -> None:
        """An event line above the bar: '  busy     stats · Normal.cdf [10^5]  1.6 cores ...'."""
        self.print(f"  {tag:<8} {what}" + (f"  {detail}" if detail else ""))

    # ------------------------------------------------------------ progress

    def start(self, total: int, desc: str) -> None:
        """A bar over `total` steps (only on a terminal)."""
        with self.lock:
            self.total = total
            if not (self.show and self.tty):
                return
            from tqdm import tqdm

            self.bar = tqdm(
                total=total,
                file=self.stream,
                dynamic_ncols=True,
                leave=True,
                ascii=False,
                bar_format=self._format(desc, 0.0, None, ""),
                mininterval=0.2,
            )

    @staticmethod
    def _format(desc: str, elapsed_s: float, left: float | None, what: str) -> str:
        eta = fmt_duration(left) if left is not None else "?"
        what = what.replace("{", "{{").replace("}", "}}")
        return f"{desc} {{percentage:3.0f}}%|{{bar:30}}| {{n_fmt}}/{{total_fmt}} [{fmt_duration(elapsed_s)} < {eta}] {what}"

    def step(self, done: int, desc: str, what: str, elapsed_s: float, left: float | None, plain: str) -> None:
        """Progress: `done` steps finished, `what` running now. The log (and a non-terminal
        stream) gets `plain` instead of the bar."""
        with self.lock:
            self._log(plain)
            if not self.show:
                return
            if self.bar is None:
                if plain:
                    print(plain, file=self.stream, flush=True)
                return
            self.bar.bar_format = self._format(desc, elapsed_s, left, what)
            self.bar.n = done
            self.bar.refresh()

    def close(self) -> None:
        with self.lock:
            if self.bar is not None:
                self.bar.close()
                self.bar = None
