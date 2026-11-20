"""The package imports (its surface is checked against docs/api/linalg.md in test_api_surface.py)."""

import stardust as sd


def test_import_and_surface():
    assert not hasattr(sd.linalg, "__version__")  # one package, no version of its own
    for name in sd.linalg.__all__:
        assert hasattr(sd.linalg, name)
    assert issubclass(sd.linalg.LinAlgError, ValueError)
