"""df.groupby(by): grouped aggregation (API.md §16).

factorize gives each row the code of its key tuple; a C++ Groups lays the
rows out group by group and reduces each group. The result is a DataFrame
whose first columns are the keys (single key or several alike, API.md §4)
over a RangeIndex. Groups come out sorted by key (sort=True) or in order of
first appearance, with NA-key groups dropped (dropna=True) or last.
"""

from . import _core, _unique
from ._reduce import Reductions
from .index import RangeIndex

AGG_FUNCS = ("sum", "mean", "min", "max", "count", "std", "var", "first", "last")


class _Grouping:
    """What every view of one groupby shares: the codes, the row layout and
    the output order of the groups, computed once."""

    __slots__ = ("keys", "groups", "first", "order")

    def __init__(self, df, by, sort, dropna):
        key_cols = [df._cols[k] for k in by]
        codes, first, ngroups = _unique.codes(key_cols)
        self.groups = _core.Groups(codes, ngroups)
        self.first = first  # each group's first row
        reps = [c.take(first) for c in key_cols]  # one row per group, in code order
        order = _core.sort_groups(reps) if sort else _core.arange(0, 1, ngroups)
        if dropna:  # groups whose key holds NA are left out (NaN is a value)
            keep = None
            for r in reps:
                valid = r.notna()
                keep = valid if keep is None else _core.binary_op("and", keep, valid, 0)
            if keep is not None and len(keep.nonzero()) != ngroups:
                order = order.take(keep.take(order).nonzero())
        self.order = order
        self.keys = {k: rep.take(order) for k, rep in zip(by, reps)}


class GroupBy(Reductions):
    __slots__ = ("_df", "_by", "_values", "_grouping", "_sort", "_dropna")

    def __init__(self, df, by, *, sort=True, dropna=True, values=None, grouping=None):
        by = [by] if isinstance(by, str) else list(by)
        if not by:
            raise ValueError("groupby needs at least one key column")
        if len(set(by)) != len(by):
            raise ValueError("key columns must be unique")
        for k in by:
            df._column(k)  # KeyError for an unknown column
        if grouping is None:
            # The columns as they are now (M12+ review): a later df["k"] = ...
            # replaces a dict entry, and the grouping, computed on first use,
            # must not meet the new columns. Columns are values sharing their
            # Buffers, so this copies no data; writes into rows copy the
            # Buffer first (CoW), so they do not show here either.
            from .frame import DataFrame

            df = DataFrame._new(dict(df._cols), df._index)
        self._df, self._by, self._sort, self._dropna = df, by, sort, dropna
        self._values = [k for k in df._cols if k not in by] if values is None else values
        self._grouping = grouping

    def _state(self):
        if self._grouping is None:
            self._grouping = _Grouping(self._df, self._by, self._sort, self._dropna)
        return self._grouping

    def __getitem__(self, key):
        """g["c"] or g[["c", "d"]]: aggregate only those value columns."""
        names = [key] if isinstance(key, str) else list(key)
        for k in names:
            self._df._column(k)
            if k in self._by:
                raise ValueError(f"{k!r} is a key column, not a value column")
        if len(set(names)) != len(names):
            raise ValueError("value columns must be unique")
        return GroupBy(self._df, self._by, sort=self._sort, dropna=self._dropna, values=names,
                       grouping=self._grouping)

    def _result(self, columns):
        """The keys, then `columns` ({name: per-group Column in code order})."""
        from .frame import DataFrame

        state = self._state()
        cols = dict(state.keys)
        for name, col in columns.items():
            if name in cols:
                raise ValueError(f"result column {name!r} would repeat a key column")
            cols[name] = col.take(state.order)
        return DataFrame._new(cols, RangeIndex(len(state.order)))

    def _apply(self, op, name, skipna=True, ddof=1):
        if op not in AGG_FUNCS:
            raise ValueError(f"unknown aggregation {op!r}; use one of {', '.join(AGG_FUNCS)}")
        return self._state().groups.reduce(op, self._df._cols[name], skipna, ddof)

    def _reduce(self, op, skipna, ddof):  # the Reductions methods (API.md §11 C3)
        return self._result({k: self._apply(op, k, skipna, ddof) for k in self._values})

    def first(self, *, skipna=True):
        """Each group's first value that is not NA (skipna=False: its first row's value)."""
        return self._reduce("first", skipna, 1)

    def last(self, *, skipna=True):
        return self._reduce("last", skipna, 1)

    def size(self):
        """How many rows each group has, in a column named "size"."""
        return self._result({"size": self._state().groups.sizes()})

    def agg(self, func=None, /, **named):
        """agg("sum") for every value column, agg({"c": "sum"}) per column, or
        agg(out=("c", "sum")) with chosen result names (API.md §16)."""
        if (func is None) == (not named):
            raise TypeError("pass either one function name / dict, or named aggregations (name=(column, function))")
        if isinstance(func, str):
            return self._reduce(func, True, 1) if func in AGG_FUNCS else self._apply(func, None)
        if isinstance(func, dict):
            plan = {}
            for col, f in func.items():
                if not isinstance(f, str):
                    raise TypeError(f"one function name per column; for several, use named aggregations (got {f!r})")
                plan[col] = (col, f)
        elif func is not None:
            raise TypeError("a list of functions is not supported (no MultiIndex); use named aggregations")
        else:
            plan = {}
            for out, spec in named.items():
                if not (isinstance(spec, tuple) and len(spec) == 2 and all(isinstance(x, str) for x in spec)):
                    raise TypeError(f"a named aggregation is name=(column, function), got {out}={spec!r}")
                plan[out] = spec
        for col, _ in plan.values():
            if col not in self._values:
                raise KeyError(f"{col!r} is not a value column of this groupby")
        return self._result({out: self._apply(f, col) for out, (col, f) in plan.items()})
