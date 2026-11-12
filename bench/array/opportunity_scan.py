"""Where is there still something to gain? Every operation family, 1 thread
and 10 threads, against numpy (single-threaded except BLAS).

Run:  .venv/bin/python bench/opportunity_scan.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp

N = 1_000_000


def data():
    r = np.random.default_rng(0)
    f = r.random(N) + 0.5
    return {
        "f": f, "g": r.random(N) + 0.5, "i": r.integers(-1000, 1000, N), "j": r.integers(1, 1000, N),
        "c": f + 1j * r.random(N), "d": r.random(N) + 1j * r.random(N), "b": r.random(N) > 0.5,
        "m": r.random((1000, 1000)), "t3": r.random((100, 100, 100)), "rows": r.integers(0, 1000, 1000),
        "iv": r.integers(0, 1000, N), "bm": r.random((1000, 8, 8)), "v": r.random(1000),
        "s1": np.array([1.0, 2.0, 3.0]), "s2": np.array([4.0, 5.0, 6.0]), "sm": np.arange(12.0).reshape(3, 4),
    }


CASES = [
    # elementwise
    ("i // j", "i // j", 10), ("i % j", "i % j", 10), ("i * j", "i * j", 10), ("i ** 3", "i ** 3", 5),
    ("f < g", "f < g", 10), ("b & b", "b & b", 10), ("i >> 3", "i >> 3", 10), ("sqrt(f)", "M.sqrt(f)", 10),
    ("round(f)", "M.round(f)", 10), ("isnan(f)", "M.isnan(f)", 10), ("abs(c)", "abs(c)", 5),
    ("c * d", "c * d", 10), ("c / d", "c / d", 5), ("exp(c)", "M.exp(c)", 2), ("where(b, f, g)", "M.where(b, f, g)", 10),
    ("clip(f)", "M.clip(f, min=0.7, max=1.2)", 10), ("astype f->i", "M.astype(f, M.int64)", 10),
    ("astype i->f", "M.astype(i, M.float64)", 10), ("expm1(f)", "M.expm1(f)", 3), ("cos(f)", "M.cos(f)", 3),
    # reductions
    ("prod(f)", "M.prod(f)", 10), ("sum(c)", "M.sum(c)", 10), ("min(i)", "M.min(i)", 10), ("any(b)", "M.any(b)", 10),
    ("argmax(m, axis=1)", "M.argmax(m, axis=1)", 10), ("mean(m, axis=0)", "M.mean(m, axis=0)", 10),
    ("std(f)", "M.std(f)", 10), ("cumulative_sum(f)", "M.cumulative_sum(f)", 10),
    ("sum(t3, axis=(0,2))", "M.sum(t3, axis=(0, 2))", 10),
    # shape and copies
    ("concat axis 0", "M.concat([m, m], axis=0)", 10), ("concat axis 1", "M.concat([m, m], axis=1)", 10),
    ("stack axis 2", "M.stack([m, m], axis=2)", 10), ("ravel(m.T)", "M.reshape(m.T, (-1,))", 10),
    ("t3 permuted copy", "M.permute_dims(t3, (2, 0, 1)).copy()", 10),
    # indexing
    ("x[mask]", "f[b]", 10), ("m[rows]", "m[rows]", 10), ("m[:, rows]", "m[:, rows]", 10),
    ("x[::2] = y[::2]", "g[::2] = f[::2]", 10), ("m[:] = v", "m[:] = v", 10),
    # sorting, sets
    ("argsort(f)", "M.argsort(f)", 2), ("sort(i)", "M.sort(i)", 2), ("sort(f, descending)", "M.sort(f, descending=True)", 2),
    ("unique_counts(iv)", "M.unique_counts(iv)", 2), ("unique_all(iv)", "M.unique_all(iv)", 2),
    # matmul
    ("batch 1000 x 8x8", "bm @ bm", 10), ("m @ v", "m @ v", 20),
    # creation
    ("arange", "M.arange(1000000)", 10), ("linspace", "M.linspace(0.0, 1.0, 1000000)", 10),
    ("zeros", "M.zeros(1000000)", 10), ("ones", "M.ones(1000000)", 10), ("eye(1000)", "M.eye(1000)", 10),
    # fixed cost
    ("tiny a + b", "s1 + s2", 20000), ("tiny a + 1.0", "s1 + 1.0", 20000), ("tiny sqrt", "M.sqrt(s1)", 20000),
    ("tiny x[1, 2]", "sm[1, 2]", 20000), ("tiny x[1]", "sm[1]", 20000), ("tiny x.T", "sm.T", 20000),
    ("tiny sum", "M.sum(s1)", 20000), ("tiny asarray(list)", "M.asarray([1.0, 2.0, 3.0])", 20000),
    ("tiny zeros(3)", "M.zeros(3)", 20000), ("tiny reshape", "M.reshape(sm, (4, 3))", 20000),
]

NP_NAMES = {"M.astype(f, M.int64)": "f.astype(np.int64)", "M.astype(i, M.float64)": "i.astype(np.float64)",
            "M.permute_dims(t3, (2, 0, 1)).copy()": "np.transpose(t3, (2, 0, 1)).copy()",
            "M.sort(f, descending=True)": "-np.sort(-f)", "M.unique_counts(iv)": "np.unique_counts(iv)",
            "M.unique_all(iv)": "np.unique_all(iv)", "M.clip(f, min=0.7, max=1.2)": "np.clip(f, 0.7, 1.2)",
            "M.cumulative_sum(f)": "np.cumulative_sum(f)"}


def main():
    arrays = data()
    rows = []
    for label, stmt, number in CASES:
        res = []
        for threads in (1, 10):
            sdnp.set_num_threads(threads)
            ns = {k: sdnp.asarray(v.copy()) for k, v in arrays.items()} | {"M": sdnp}
            res.append(min(timeit.repeat(stmt, globals=ns, number=number, repeat=5)) / number)
        np_stmt = NP_NAMES.get(stmt, stmt).replace("M.", "np.")
        ns = {k: v.copy() for k, v in arrays.items()} | {"np": np}
        t_np = min(timeit.repeat(np_stmt, globals=ns, number=number, repeat=5)) / number
        rows.append((label, res[0], res[1], t_np))
    sdnp.set_num_threads(10)
    fmt = lambda t: f"{t * 1e9:.0f}ns" if t < 1e-6 else (f"{t * 1e6:.0f}µs" if t < 1e-3 else f"{t * 1e3:.2f}ms")  # noqa: E731
    print("| case | 1 thread | 10 threads | numpy | 1t / numpy | 10t speedup |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, a, b, n in rows:
        print(f"| {label} | {fmt(a)} | {fmt(b)} | {fmt(n)} | {a / n:.2f}x | {a / b:.1f}x |")


if __name__ == "__main__":
    main()
