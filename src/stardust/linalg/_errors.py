"""Exceptions of sd.linalg (docs/api/linalg.md §8; docs/design/linalg.md §2.2)."""


class LinAlgError(ValueError):
    """A numerical failure: a singular matrix, a matrix that is not positive
    definite, or an algorithm that did not converge."""


# Shown as stardust.linalg.LinAlgError in tracebacks: the public path, not this module's.
LinAlgError.__module__ = "stardust.linalg"
