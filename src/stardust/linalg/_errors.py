"""Exceptions of stardust-linalg (API.md §7, DESIGN.md §2.4)."""


class LinAlgError(ValueError):
    """A numerical failure: a singular matrix, a matrix that is not positive
    definite, or an algorithm that did not converge."""
