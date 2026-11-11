"""Named result tuples (API.md §0.3). Field names follow the Array API."""

from collections import namedtuple

SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
