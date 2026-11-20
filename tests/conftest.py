"""Test tiers, marks and hypothesis profiles for the whole suite (tests/README.md §1).

The tier comes from --tier or else SD_TIER (quick by default; tests/_tier.py):
  quick  deselects the tests marked deep or external,
  deep   deselects the external ones,
  all    runs everything.
It also loads the hypothesis profile of the same name: quick is derandomized (the same
examples every run, so a parallel run is deterministic too) with the profile's 100 examples,
deep and all are random with 1000. A test that sets its own count writes tests._tier.examples(n),
which scales n the same way.

Marks: deep (a wide sweep or an expensive reference), external (runs R or a Fortran
compiler), serial (must not share the machine with other test processes: GIL, thread
counts, timing, memory limits; tools/test.py runs these alone after the parallel part).
The modules in SERIAL get the serial mark here, whole; single tests carry @pytest.mark.serial
themselves (pytest -m serial --co lists both).

--match PATTERN (repeatable; bench/cli.py's matching: a glob, or a substring without
wildcards) keeps the tests whose id (tests/<area>/test_<topic>.py::test_<name>[...]) matches one pattern.
"""

import os

import pytest
from hypothesis import settings

from bench import cli

from . import _tier

# Thread-count tests (array/frame test_threads, linalg/test_parallel) set the pool of their own
# process only and time nothing, so they run in parallel like the rest. A few tests elsewhere
# bound a call's time generously (50 ms or more) to catch hangs; they passed at 4-8 workers.
SERIAL = {
    "tests/test_gil_release.py",  # GIL release measured by wall time (failed at 8 workers)
    "tests/random/test_alloc_errors.py",  # allocations near the physical memory
}

settings.register_profile("quick", deadline=None, derandomize=True)
settings.register_profile("deep", deadline=None, max_examples=100 * _tier.DEEP_FACTOR)
settings.register_profile("all", deadline=None, max_examples=100 * _tier.DEEP_FACTOR)


def pytest_addoption(parser):
    parser.addoption("--tier", choices=_tier.TIERS, help="quick (default), deep or all; else SD_TIER")
    parser.addoption(
        "--match",
        action="append",
        default=[],
        metavar="PATTERN",
        help="keep the tests whose id matches (glob; plain text matches as a substring); repeatable",
    )


def pytest_configure(config):
    tier = config.getoption("--tier") or os.environ.get("SD_TIER", "quick")
    if tier not in _tier.TIERS:
        raise pytest.UsageError(f"SD_TIER must be one of {', '.join(_tier.TIERS)}, not {tier!r}")
    _tier.tier = tier
    settings.load_profile(tier)
    config.addinivalue_line("markers", "deep: a wide sweep or an expensive reference (SD_TIER=deep)")
    config.addinivalue_line("markers", "external: runs R or a Fortran compiler (SD_TIER=all)")
    config.addinivalue_line("markers", "serial: runs alone, never next to other test processes")


def pytest_report_header(config):
    return f"stardust test tier: {_tier.tier}"


def pytest_collection_modifyitems(config, items):
    root = config.rootpath
    skip = {"quick": ("deep", "external"), "deep": ("external",), "all": ()}[_tier.tier]
    patterns = config.getoption("--match")
    keep, dropped = [], []
    for item in items:
        if str(item.path.relative_to(root)) in SERIAL:
            item.add_marker(pytest.mark.serial)
        out = any(item.get_closest_marker(m) for m in skip) or not cli.matches(patterns, item.nodeid)
        (dropped if out else keep).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep


@pytest.fixture
def restore_threads():
    """For a test that calls sd.set_num_threads: the pool's thread count after it is the one
    before it. A module whose every test changes it uses it through
    `pytestmark = pytest.mark.usefixtures("restore_threads")`."""
    import stardust as sd

    before = sd.get_num_threads()
    yield
    sd.set_num_threads(before)
