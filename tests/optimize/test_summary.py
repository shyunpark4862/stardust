"""OptimizeResult.summary() (O119): the method, convergence, status and message, the objective
(fun, or cost for least squares) and the counts, then x (cut when long; with f(x) for a system's
root); a batched scalar function one row per element (its bracket's ends)."""

import pytest

import stardust as sd

O = sd.optimize


def _quad(x):
    return float(sd.sum((x - 1.0) ** 2))


def _rows(text):
    lines = text.splitlines()
    i = lines.index("-" * max(len(line) for line in lines if set(line) == {"-"}))
    return [line.split() for line in lines[i + 1 : -1]]


def test_minimize_header_and_x():
    r = O.minimize(_quad, sd.asarray([3.0, -1.0, 0.5]), method="bfgs")
    s = r.summary()
    assert s.splitlines()[0].strip() == "Optimization Results"
    assert "Method:                           bfgs" in s and "Converged:                        True" in s
    assert f"{r.fun:#.6g}" in s and f"No. Iterations:{r.n_iter:>23}" in s and "Jacobian Evals:" in s
    message = next(line for line in s.splitlines() if line.startswith("Message:"))
    assert message.endswith(r.message) and len(message) == 78
    rows = _rows(s)
    assert [row[0] for row in rows] == ["x[0]", "x[1]", "x[2]"]
    assert [float(row[1]) for row in rows] == pytest.approx(r.x.tolist(), abs=5e-5)


def test_long_x_is_cut():
    s = O.minimize(_quad, sd.zeros((30,)), method="bfgs").summary()
    stubs = [row[0] for row in _rows(s)]
    assert stubs == ["x[0]", "x[1]", "x[2]", "x[3]", "x[4]", "...", "x[25]", "x[26]", "x[27]", "x[28]", "x[29]"]


def test_least_squares_cost_and_root_residuals():
    ls = O.least_squares(lambda x: sd.asarray([x[0] - 1.0, 10.0 * (x[1] - x[0] ** 2)]), sd.asarray([2.0, 2.0]))
    s = ls.summary()
    assert "Cost:" in s and "Optimality:" in s and "Function value" not in s
    assert s.splitlines()[s.splitlines().index("-" * 78) - 1].split() == ["x"]
    rt = O.root(lambda x: sd.asarray([x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]]),
                sd.asarray([0.0, 0.0]))  # fmt: skip
    t = rt.summary()
    assert t.splitlines()[t.splitlines().index("-" * 78) - 1].split() == ["x", "fun"]


def test_scalar_x_and_long_message():
    r = O.root_scalar(lambda x: x**3 - 2.0, bracket=(0.0, 2.0))
    s = r.summary()
    assert _rows(s) == [["x", f"{r.x:.4f}"]]
    made = O.OptimizeResult(x=sd.asarray([1.0]), converged=False, message="word " * 25, nfev=3).summary()
    assert all(len(line) <= 78 for line in made.splitlines())
    assert "Method:" not in made and "Function Evals:                      3" in made


def test_batched_one_row_per_element():
    c = sd.asarray([1.0, 2.0, 3.0])
    r = O.find_root(lambda x, c: x * x - c, (sd.zeros((3,)), sd.full((3,), 5.0)), args=(c,))
    s = r.summary()
    assert "Batch shape:                      (3,)  Converged:                      3 of 3" in s
    rows = _rows(s)
    assert [row[0] for row in rows] == ["[0]", "[1]", "[2]"]
    assert [float(row[1]) for row in rows] == pytest.approx([1.0, 2.0**0.5, 3.0**0.5], abs=5e-5)
    b = O.bracket_root(lambda x, c: x * x - c, sd.full((3,), 0.5), args=(c,)).summary()
    head = b.splitlines()[b.splitlines().index("-" * 78) - 1].split()
    assert head[:2] == ["bracket[0]", "bracket[1]"] and "converged" in head


def test_summary_takes_no_alpha():
    with pytest.raises(TypeError, match=r"^summary\(\): unexpected keyword argument 'alpha'$"):
        O.minimize(_quad, sd.zeros((2,))).summary(alpha=0.05)
