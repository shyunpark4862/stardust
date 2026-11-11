"""Exceptions of sd.linalg (docs/api/linalg.md §8; docs/design/linalg.md §2.2)."""


class LinAlgError(ValueError):
    """A numerical failure: a singular matrix, a matrix that is not positive
    definite, or an algorithm that did not converge."""
