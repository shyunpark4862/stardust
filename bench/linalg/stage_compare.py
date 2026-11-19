"""M11: one build's times on the baseline rows, for comparing milestones.

bench/stage_compare.sh builds the end commit of M7, M8, M9 and M10 into
build/stage/<tag>/pkg and runs this script against each build in turn,
several rounds, so that the machine's drift over the session (seen in M10:
up to 2x on the same code) hits every build alike. The rows are
baseline.py's (same shapes, same seed); rows a build does not have (kron,
cross, expm before M10) are recorded as missing.

Usage: python bench/stage_compare.py TAG OUT.json [--threads N] [--ref]
  --ref : time numpy/scipy instead of sdla (once is enough)
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import baseline  # noqa: E402  (imports stardust_linalg from PYTHONPATH: the build under test)
import numpy as np  # noqa: E402

import stardust_numpy as sdnp  # noqa: E402


def main():
    tag, out = sys.argv[1], pathlib.Path(sys.argv[2])
    if "--threads" in sys.argv:
        sdnp.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]))
    ref = "--ref" in sys.argv
    np.seterr(all="ignore")
    rng = np.random.default_rng(0)
    rows = {}

    def run(name, shape, f_sd, r):
        key = f"{name} | {shape}"
        if ref:
            rows[key] = None if r is None else {"us": baseline.best_us(r[1]), "lib": r[0]}
            return
        try:
            rows[key] = baseline.best_us(f_sd)
        except AttributeError:  # the function is not in this build
            rows[key] = None

    for shape in baseline.MATRIX_SHAPES:
        for name, f_sd, r in baseline.matrix_rows(shape, rng):
            run(name, str(shape).replace(" ", ""), f_sd, r)
    for name, shape, f_sd, r in baseline.long_rows(rng):
        run(name, shape, f_sd, r)
    old = json.loads(out.read_text()) if out.exists() else {"tag": tag, "rounds": 0, "rows": {}}
    # keep each row's best over the rounds
    for k, v in rows.items():
        if ref:
            old["rows"][k] = v
        else:
            prev = old["rows"].get(k)
            old["rows"][k] = v if prev is None or v is None else min(prev, v)
    old["rounds"] += 1
    old["threads"] = sdnp.get_num_threads()
    out.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
