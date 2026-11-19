"""M6: advanced indexing, sorting and matmul -- the last baseline before M7.

Run:  .venv/bin/python bench/m6_index_sort_matmul.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp


def best_ms(stmt, ns, number=5):
    return min(timeit.repeat(stmt, globals=ns, number=number, repeat=5)) / number * 1e3


def table(title, cases, arrays, number=5):
    print(title)
    print(f"{'':44} {'stardust':>9} {'numpy':>9} {'ratio':>7}")
    for label, stmt in cases:
        t_sd = best_ms(stmt, {k: sdnp.asarray(v) for k, v in arrays.items()} | {"M": sdnp}, number)
        t_np = best_ms(stmt, dict(arrays) | {"M": np}, number)
        print(f"{label:44} {t_sd:7.2f}ms {t_np:7.2f}ms {t_sd / t_np:6.1f}x")
    print()


def main():
    rng = np.random.default_rng(0)
    n = 1_000_000
    arrays = {
        "x": rng.standard_normal(n),
        "idx": rng.integers(0, n, n),
        "mask": rng.random(n) > 0.5,
        "m": rng.standard_normal((1000, 1000)),
        "rows": rng.integers(0, 1000, 1000),
        "y": rng.standard_normal(n),
        "iv": rng.integers(0, 1000, n),
    }
    table(
        "1) advanced indexing (1M elements)",
        [
            ("x[idx]           random gather", "x[idx]"),
            ("x[mask]          boolean mask", "x[mask]"),
            ("m[rows]          1000 whole rows", "m[rows]"),
            ("m[:, rows]       1000 columns", "m[:, rows]"),
            ("y[idx] = 1.0     scatter", "y[idx] = 1.0"),
        ],
        arrays,
    )

    table(
        "2) sorting (1M elements)",
        [
            ("sort(x)          stable (merge sort)", "M.sort(x)"),
            ("sort(x, stable=False)  introsort", "M.sort(x, stable=False)"),
            ("argsort(x)       stable", "M.argsort(x, stable=True)"),
            ("sort(iv)         int64, many ties", "M.sort(iv)"),
            ("unique_values(iv)", "M.unique_values(iv)"),
        ],
        arrays,
        number=2,
    )

    mats = {
        "a": rng.standard_normal((1000, 1000)),
        "b": rng.standard_normal((1000, 1000)),
        "c": rng.standard_normal((500, 500)) + 1j * rng.standard_normal((500, 500)),
        "ia": rng.integers(-100, 100, (300, 300)),
        "v": rng.standard_normal(1000),
        "batch": rng.standard_normal((1000, 8, 8)),
    }
    table(
        "3) matmul",
        [
            ("a @ b            1000^3 float64 (dgemm)", "a @ b"),
            ("a.T @ b          transposed: same call, Trans flag", "a.T @ b"),
            ("a[:, ::2] @ b[::2]  strided: packed first", "a[:, ::2] @ b[::2]"),
            ("a @ v            gemv", "a @ v"),
            ("c @ c            500^3 complex128 (zgemm)", "c @ c"),
            ("ia @ ia          300^3 int64 (own loop)", "ia @ ia"),
            ("batch @ batch    1000 x (8x8)", "batch @ batch"),
        ],
        mats,
        number=3,
    )


if __name__ == "__main__":
    main()
