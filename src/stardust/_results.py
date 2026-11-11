"""Named result tuples of the unique_* functions, with Array API names
(docs/api/array.md §8.3)."""

from collections import namedtuple

UniqueAllResult = namedtuple("UniqueAllResult", ["values", "indices", "inverse_indices", "counts"])
UniqueCountsResult = namedtuple("UniqueCountsResult", ["values", "counts"])
UniqueInverseResult = namedtuple("UniqueInverseResult", ["values", "inverse_indices"])
