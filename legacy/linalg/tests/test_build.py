"""The package imports and exposes what API.md lists for the milestones done so far."""

import stardust_linalg as sdla


def test_import_and_surface():
    assert sdla.__version__ == "0.0.1"
    assert set(sdla.__all__) == {
        "LinAlgError", "det", "inv", "slogdet", "solve",
        "vecdot", "outer", "trace", "diagonal", "matrix_power", "vector_norm", "matrix_norm",
        "solve_triangular", "cholesky", "cholesky_solve", "lu", "lu_solve",
        "qr", "qr_pivoted", "lstsq",
    }
    for name in sdla.__all__:
        assert hasattr(sdla, name)
    assert issubclass(sdla.LinAlgError, ValueError)
