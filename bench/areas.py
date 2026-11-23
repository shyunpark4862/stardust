"""The library's areas, in the one order every tool lists them (README.md's table).

The golden outputs, the speed and accuracy measurements, the coverage report and the test
tools all take their area names and their default order from here. A tool that does not
cover an area yet says so where it filters this tuple; it never keeps a list of its own.
"""

AREAS = (
    "array",
    "linalg",
    "frame",
    "special",
    "stats",
    "regression",
    "random",
    "optimize",
    "htest",
    "sparse",
    "integrate",
    "differentiate",
    "spatial",
    "graph",
    "interpolate",
    "fft",
    "smooth",
    "plot",
)


def check(names, available=AREAS):
    """`names` in AREAS order, or ValueError naming the first unknown one. Empty means all of
    `available`."""
    if not names:
        return tuple(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"unknown area {unknown[0]!r} (choose from {', '.join(available)})")
    return tuple(a for a in available if a in names)
