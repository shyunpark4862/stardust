"""The package imports (its surface is checked against API.md in test_api_surface.py, M11)."""

import stardust_linalg as sdla


def test_import_and_surface():
    assert sdla.__version__ == "0.0.1"
    for name in sdla.__all__:
        assert hasattr(sdla, name)
    assert issubclass(sdla.LinAlgError, ValueError)
