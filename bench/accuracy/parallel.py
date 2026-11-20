"""Measuring sweeps in worker processes (python -m bench.accuracy run --jobs N).

Each worker rebuilds the sweep list of the areas it is asked about once and measures by id, so
the result of a sweep does not depend on the number of processes (every measure is a function
of the sweep's own inputs; stardust's results do not depend on its thread count, which tools/
golden.py checks). Workers run one stardust thread each (the processes are the parallelism)
and are replaced after a few sweeps (maxtasksperchild) to keep memory flat.
"""

from __future__ import annotations

import multiprocessing as mp
import os

_SWEEPS: dict = {}


def _init():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    import stardust as sd

    sd.set_num_threads(1)


def _measure(job):
    sid, compare = job
    from . import sources

    area = sid.split("/", 1)[0]
    if area not in _SWEEPS:
        _SWEEPS[area] = {s.id: s for s in sources.sweeps([area])}
    return sources.measure(_SWEEPS[area][sid], compare=compare)


def measure_all(sweeps, compare=False, jobs=10, progress=None) -> list:
    """The entries of the sweeps, in their order."""
    ids = [s.id for s in sweeps]
    # The comparators run in forked children (common.isolated); macOS's Objective-C runtime aborts
    # a forked child when another thread was initializing a class, unless told not to check.
    # It reads this at process start: so even --jobs 1 measures in one spawned worker.
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    done = {}
    pool = mp.get_context("spawn").Pool(max(1, min(jobs, 10)), initializer=_init, maxtasksperchild=40)
    try:
        for i, r in enumerate(pool.imap_unordered(_measure, [(sid, compare) for sid in ids], chunksize=1), 1):
            done[r["id"]] = r
            if progress:
                progress(i, len(ids), r)
        pool.close()  # not terminate: the workers exit normally and close their R sessions (atexit)
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    return [done[sid] for sid in ids]
