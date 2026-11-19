"""df.groupby(by): grouped aggregation (docs/api/frame.md §7).

factorize gives each row the code of its key tuple; a C++ Groups lays the
rows out group by group and reduces each group. The result is a DataFrame
whose first columns are the keys (single key or several alike, docs/api/frame.md §7.2)
over a RangeIndex. Groups come out sorted by key (sort=True) or in order of
first appearance, with NA-key groups dropped (dropna=True) or last.
"""

from . import _core, _unique
from ._errors import check_bool, name_list, type_name
from ._index import RangeIndex
from ._reduce import Reductions, own_reductions

AGG_FUNCS = ("sum", "mean", "min", "max", "count", "std", "var", "first", "last", "size")


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


@own_reductions
class GroupBy(Reductions):
    __slots__ = ("_df", "_by", "_values", "_grouping", "_sort", "_dropna")

    def __init__(self, df, by, *, sort=True, dropna=True, values=None, grouping=None):
        # Made by df.groupby(by): the messages name groupby().
        by = name_list(by, "by", "groupby")
        check_bool(sort, "sort", "groupby")
        check_bool(dropna, "dropna", "groupby")
        if not by:
            raise ValueError("groupby(): 'by' must name at least one column")
        if len(set(by)) != len(by):
            raise ValueError("groupby(): key columns must be unique")
        for k in by:
            df._column(k, "groupby")  # KeyError for an unknown column
        if grouping is None:
            # The columns as they are now (docs/design/frame.md §5.3): a later df["k"] = ...
            # replaces a dict entry, and the grouping, computed on first use,
            # must not meet the new columns. Columns are values sharing their
            # Buffers, so this copies no data; writes into rows copy the
            # Buffer first (CoW), so they do not show here either.
            from ._frame import DataFrame

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
        if isinstance(key, str):
            names = [key]
        elif isinstance(key, list):  # as df[[...]]
            names = key
        else:
            raise TypeError(f"column names must be str, not {type_name(key)}")
        for k in names:
            self._df._column(k)
            if k in self._by:
                raise ValueError(f"{k!r} is a key column, not a value column")
        if len(set(names)) != len(names):
            raise ValueError("value columns must be unique")
        return GroupBy(self._df, self._by, sort=self._sort, dropna=self._dropna, values=names, grouping=self._grouping)

    # No iteration over groups (docs/api/frame.md §7): without this, Python
    # would iterate through __getitem__ with 0, 1, ...
    __iter__ = None

    def _result(self, columns, func):
        """The keys, then `columns` ({name: per-group Column in code order})."""
        from ._frame import DataFrame

        state = self._state()
        cols = dict(state.keys)
        for name, col in columns.items():
            if name in cols:
                raise ValueError(f"{func}(): result column {name!r} would repeat a key column")
            cols[name] = col.take(state.order)
        return DataFrame._new(cols, RangeIndex(len(state.order)))

    def _apply(self, op, name, func, skipna=True, correction=0):
        if op not in AGG_FUNCS:
            raise ValueError(f"{func}(): unknown aggregation {op!r}; use one of {', '.join(AGG_FUNCS)}")
        if op == "size":  # the group sizes, whatever the column (as size())
            return self._state().groups.sizes()
        return self._state().groups.reduce(op, self._df._cols[name], skipna, correction, func=func)

    def _reduce(self, op, skipna, correction, func=None):  # the Reductions methods (docs/api/frame.md §7.3)
        func = func or op  # sum() is named sum; agg("sum") names agg
        return self._result({k: self._apply(op, k, func, skipna, correction) for k in self._values}, func)

    def first(self, *, skipna=True):
        """Each group's first value that is not NA (skipna=False: its first row's value)."""
        return self._reduce("first", check_bool(skipna, "skipna", "first"), 1)

    def last(self, *, skipna=True):
        return self._reduce("last", check_bool(skipna, "skipna", "last"), 1)

    def size(self):
        """How many rows each group has, in a column named "size"."""
        return self._result({"size": self._state().groups.sizes()}, "size")

    def agg(self, func=None, /, **named):
        """agg("sum") for every value column, agg({"c": "sum"}) per column, or
        agg(out=("c", "sum")) with chosen result names (docs/api/frame.md §7.4)."""
        if (func is None) == (not named):
            raise TypeError(
                "agg(): pass either one function name / dict, or named aggregations (name=(column, function))"
            )
        if isinstance(func, str):
            if func == "size":  # agg("size") is size(): one "size" column (docs/api/frame.md §7.4)
                return self._result({"size": self._apply("size", None, "agg")}, "agg")
            return self._reduce(func, True, 0, "agg") if func in AGG_FUNCS else self._apply(func, None, "agg")
        if isinstance(func, dict):
            plan = {}
            for col, f in func.items():
                if not isinstance(f, str):
                    raise TypeError(
                        f"agg(): one function name per column; for several, use named aggregations (got {f!r})"
                    )
                plan[col] = (col, f)
        elif isinstance(func, (list, tuple)):
            raise TypeError("agg(): a list of functions is not supported (no MultiIndex); use named aggregations")
        elif func is not None:
            what = "a callable is not supported" if callable(func) else f"{type_name(func)} is not supported"
            raise TypeError(
                f"agg(): {what}; pass a function name (a str such as 'sum'), a dict of "
                "{column: function name}, or named aggregations (name=(column, function name))"
            )
        else:
            plan = {}
            for out, spec in named.items():
                if not (isinstance(spec, tuple) and len(spec) == 2 and all(isinstance(x, str) for x in spec)):
                    raise TypeError(f"agg(): a named aggregation is name=(column, function), got {out}={spec!r}")
                plan[out] = spec
        for col, _ in plan.values():
            if col not in self._values:
                raise KeyError(f"agg(): {col!r} is not a value column of this groupby")
        return self._result({out: self._apply(f, col, "agg") for out, (col, f) in plan.items()}, "agg")
