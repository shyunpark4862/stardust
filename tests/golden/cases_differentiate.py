"""Golden cases for sd.differentiate (tools/golden.py has the contract).

sd.differentiate's only public name so far is its result class (milestone C0 prepared the area:
docs/design/integrate.md §8): the class's fields, read-only rules and repr. Milestone C2 adds the
cases of derivative, jacobian and hessian on the problems of tests/differentiate/problems.py.
"""


def cases(L):
    sd = L.sd
    R = sd.differentiate.DifferentiateResult
    yield (
        "DifferentiateResult fields",
        lambda: R(df=2.7182818284590127, error=4.5e-14, converged=True, status=0, n_iter=2, nfev=11, x=1.0),
    )
    yield (
        "DifferentiateResult batch",
        lambda: R(batch=(2,), df=sd.asarray([1.0, 2.0]), error=sd.zeros(2), converged=sd.asarray([True, False]),
                  status=sd.asarray([0, -2]), n_iter=sd.asarray([2, 10]), nfev=sd.asarray([11, 27])),
    )  # fmt: skip
    yield (
        "DifferentiateResult hessian-like",
        lambda: R(ddf=sd.eye(2), error=sd.zeros((2, 2)), converged=sd.ones((2, 2)) > 0),
    )
    yield "DifferentiateResult repr", lambda: repr(R(df=float("inf"), converged=False, status=-3))
    yield "err DifferentiateResult positional", lambda: R(1)
    yield "err DifferentiateResult unknown field", lambda: R(nit=3)
    yield "err DifferentiateResult absent field", lambda: R(df=1.0).ddf
    yield "err DifferentiateResult read-only", lambda: setattr(R(df=1.0), "df", 2.0)
    yield "err DifferentiateResult unpack", lambda: tuple(R(df=1.0, error=0.0))
