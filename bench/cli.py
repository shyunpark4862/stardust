"""The command-line conventions shared by every measurement and test tool.

tools/golden.py, tools/test.py, tools/coverage.py, tools/<area>_ref.py, python -m bench.speed
and python -m bench.accuracy all build their options from these helpers, so the same option
has the same name, meaning and default everywhere:

- areas: positional AREA ... (default: all the tool covers), in bench.areas.AREAS order;
- --match PATTERN: keep the cases, sweeps or tests whose id matches (glob with * and ?; repeat
  the option for several patterns; a pattern without a wildcard matches as a substring);
- --jobs N: worker processes (default 8, at most 10);
- --out FILE: the result file to write (always a file, never a folder);
- exit codes: 0 all good; 1 a failure, a difference, a value beyond its contract, a WORSE
  measure, or nothing to compare against; 2 a usage or tool error;
- logs: build/logs/<tool>/<stamp>.log, the newest LOG_KEEP kept.
"""

import argparse
import re
import time
from pathlib import Path

from bench import areas as _areas

OK, FAIL, ERROR = 0, 1, 2
JOBS_DEFAULT, JOBS_MAX = 8, 10
LOG_KEEP = 20
ROOT = Path(__file__).resolve().parent.parent


def add_areas(parser: argparse.ArgumentParser, available=_areas.AREAS) -> None:
    parser.add_argument(
        "areas",
        nargs="*",
        metavar="AREA",
        help=f"areas to cover (default: all of {', '.join(available)})",
    )


def areas(parser: argparse.ArgumentParser, args, available=_areas.AREAS):
    """The chosen areas in AREAS order; a usage error (exit 2, through `parser.error`) for an
    unknown name."""
    try:
        return _areas.check(args.areas, available)
    except ValueError as e:
        parser.error(str(e))


def add_match(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        metavar="PATTERN",
        help="keep ids matching PATTERN (* and ? are wildcards; without them, a substring); repeatable",
    )


def matches(patterns, ident: str) -> bool:
    """True when no pattern is given or one of them matches `ident`. Only `*` and `?` are
    wildcards; `[` and `]` are plain characters, so an id copied from a table matches as is."""
    if not patterns:
        return True
    for p in patterns:
        if "*" in p or "?" in p:
            if _glob(p).fullmatch(ident):
                return True
        elif p in ident:
            return True
    return False


def _glob(pattern: str) -> "re.Pattern[str]":
    parts = []
    for ch in pattern:
        parts.append(".*" if ch == "*" else "." if ch == "?" else re.escape(ch))
    return re.compile("".join(parts), re.DOTALL)


def add_jobs(parser: argparse.ArgumentParser, default: int = JOBS_DEFAULT) -> None:
    parser.add_argument(
        "--jobs",
        type=_jobs,
        default=default,
        metavar="N",
        help=f"worker processes (default {default}, at most {JOBS_MAX})",
    )


def _jobs(text: str) -> int:
    n = int(text)
    if not 1 <= n <= JOBS_MAX:
        raise argparse.ArgumentTypeError(f"--jobs must be from 1 to {JOBS_MAX}, not {n}")
    return n


def add_out(parser: argparse.ArgumentParser, what: str) -> None:
    parser.add_argument("--out", type=Path, metavar="FILE", help=f"write {what} to FILE")


def log_file(tool: str) -> Path:
    """A new log file build/logs/<tool>/<stamp>.log; older logs beyond LOG_KEEP are removed."""
    folder = ROOT / "build" / "logs" / tool
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = folder / f"{stamp}.log"
    n = 1
    while path.exists():
        n += 1
        path = folder / f"{stamp}-{n}.log"
    logs = sorted(folder.glob("*.log"))
    for old in logs[: max(0, len(logs) - (LOG_KEEP - 1))]:
        old.unlink(missing_ok=True)
    return path
