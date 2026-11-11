"""The package imports (its surface is checked against docs/api/linalg.md in test_api_surface.py)."""

import stardust as sd


def test_import_and_surface():
    assert sd.linalg.__version__ == "0.0.1"
    for name in sd.linalg.__all__:
        assert hasattr(sd.linalg, name)
    assert issubclass(sd.linalg.LinAlgError, ValueError)
