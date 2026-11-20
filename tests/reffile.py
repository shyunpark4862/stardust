"""The one layout of every npz reference file (tests/<area>/ref/*.npz): read and write.

A file holds points, their inputs, and the reference values of one or more methods at them:

    layout                   1 (this layout's version)
    arb                      what computed the references: "python-flint <version>" (Arb), and
                             any other exact method in words after a "; "
    target_bits              each reference is within 2^-target_bits of its value (relative, or
                             of the area's own scale: max(1, |value|), the vector's largest entry)
    tool                     the generator, tools/<area>_ref.py
    n_requested              the points the generator was asked for (-1: not recorded)
    inputs                   the names of the point inputs, in argument order
    input/<name>             one per name: axis 0 is the point (N points)
    group                    int64 (N,): the group of each point, 0 .. ngroups - 1
    ngroups                  the number of groups (1 when the points are not grouped)
    group_inputs             the names of the per-group inputs (the parameters a group shares)
    group_input/<name>       axis 0 is the group
    methods                  the methods with references, in order
    method/<m>/over          "point" or "group": what the method's values are attached to
    method/<m>/index         int64: the point (or group) of each value; absent when the values
                             are every point (group) in order
    method/<m>/hi, lo        the reference as a double-double, axis 0 aligned with index
    method/<m>/dropped       int64: the points (or groups) the generator dropped because their
                             reference did not settle within its precision and iteration caps
                             (empty, with n_requested -1, where that was not recorded)
    method/<m>/<field>       further arrays aligned with index: guaranteed, scale, kmin, ...

An array whose items differ in shape (matrices of several sizes) is ragged: the key holds the
items raveled in C order one after another, and "<key>.shape" (n, D) their shapes, each row
padded with -1 to the largest number of dimensions D. Reading gives a list of arrays for a
ragged key and the array itself otherwise; both index by item the same way.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

LAYOUT = 1


@dataclasses.dataclass
class Method:
    over: str
    index: np.ndarray
    hi: object
    lo: object
    dropped: np.ndarray
    fields: dict


@dataclasses.dataclass
class Ref:
    meta: dict  # arb, target_bits, tool, n_requested
    inputs: dict  # name -> array (or list of arrays, ragged), in argument order
    group: np.ndarray
    ngroups: int
    group_inputs: dict
    methods: dict  # name -> Method

    @property
    def points(self) -> int:
        return int(self.group.size)

    def args(self, method=None):
        """The point inputs in argument order; with method, at that method's points."""
        out = list(self.inputs.values())
        if method is None:
            return tuple(out)
        idx = self.methods[method].index
        return tuple(take(v, idx) for v in out)


def take(v, idx):
    """Items idx of an array or a ragged list."""
    if isinstance(v, list):
        return [v[int(i)] for i in idx]
    return v[idx]


def _put(out, key, value):
    if isinstance(value, list):
        arrs = [np.asarray(v) for v in value]
        shapes = {a.shape for a in arrs}
        if len(shapes) <= 1 and arrs:
            out[key] = np.stack(arrs)
            return
        dims = max((a.ndim for a in arrs), default=1) or 1
        table = np.full((len(arrs), dims), -1, dtype=np.int64)
        for i, a in enumerate(arrs):
            table[i, : a.ndim] = a.shape
        flat = [a.ravel() for a in arrs]
        out[key] = np.concatenate(flat) if flat else np.zeros(0)
        out[f"{key}.shape"] = table
    else:
        out[key] = np.asarray(value)


def _get(z, key):
    v = z[key]
    sk = f"{key}.shape"
    if sk not in z.files:
        return v
    items, at = [], 0
    for row in z[sk]:
        shape = tuple(int(s) for s in row if s >= 0)
        n = int(np.prod(shape)) if shape else 1
        items.append(v[at : at + n].reshape(shape))
        at += n
    if at != v.size:
        raise ValueError(f"{key}: the shapes cover {at} of {v.size} values")
    return items


def _names(z, key):
    return [str(s) for s in z[key].tolist()] if key in z.files else []


def write(
    path,
    *,
    arb: str,
    target_bits: int,
    tool: str,
    inputs: dict,
    methods: dict,
    group=None,
    ngroups: int | None = None,
    group_inputs: dict | None = None,
    n_requested: int = -1,
) -> None:
    """Write a reference file. inputs: {name: array or list}, in argument order; methods:
    {name: {"over": "point" | "group", "index", "hi", "lo", ["dropped"], further fields}};
    group: the points' groups (default all 0)."""
    n = None
    for v in inputs.values():
        n = len(v)
        break
    if n is None:  # no point inputs: the methods' indices say how many points there are
        n = max((int(np.max(m["index"])) + 1 for m in methods.values() if len(m["index"])), default=0)
    group = np.zeros(n, dtype=np.int64) if group is None else np.asarray(group, dtype=np.int64)
    group_inputs = group_inputs or {}
    if ngroups is None:
        ngroups = next((len(v) for v in group_inputs.values()), int(group.max()) + 1 if group.size else 1)
    out = {
        "layout": np.int64(LAYOUT),
        "arb": np.str_(arb),
        "target_bits": np.int64(target_bits),
        "tool": np.str_(tool),
        "n_requested": np.int64(n_requested),
        "inputs": np.array(list(inputs), dtype=str),
        "group": group,
        "ngroups": np.int64(ngroups),
        "group_inputs": np.array(list(group_inputs), dtype=str),
        "methods": np.array(list(methods), dtype=str),
    }
    for name, v in inputs.items():
        if len(v) != n:
            raise ValueError(f"input {name}: {len(v)} points, not {n}")
        _put(out, f"input/{name}", v)
    for name, v in group_inputs.items():
        if len(v) != ngroups:
            raise ValueError(f"group input {name}: {len(v)} groups, not {ngroups}")
        _put(out, f"group_input/{name}", v)
    for name, m in methods.items():
        m = dict(m)
        over = m.pop("over")
        if over not in ("point", "group"):
            raise ValueError(f"method {name}: over {over!r}")
        index = np.asarray(m.pop("index"), dtype=np.int64)
        out[f"method/{name}/over"] = np.str_(over)
        every = n if over == "point" else ngroups
        if not (index.size == every and (index == np.arange(every)).all()):
            out[f"method/{name}/index"] = index
        out[f"method/{name}/dropped"] = np.asarray(m.pop("dropped", np.zeros(0, np.int64)), dtype=np.int64)
        for field in ("hi", "lo"):
            if field not in m:
                raise ValueError(f"method {name}: no {field}")
        for field, v in m.items():
            if len(v) != index.size:
                raise ValueError(f"method {name}: {field} has {len(v)} values for {index.size} indices")
            _put(out, f"method/{name}/{field}", v)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)


def read(path) -> Ref:
    with np.load(path) as z:
        if "layout" not in z.files or int(z["layout"]) != LAYOUT:
            raise ValueError(f"{path}: not a reference file of layout {LAYOUT}")
        meta = {
            "arb": str(z["arb"]),
            "target_bits": int(z["target_bits"]),
            "tool": str(z["tool"]),
            "n_requested": int(z["n_requested"]),
        }
        inputs = {k: _get(z, f"input/{k}") for k in _names(z, "inputs")}
        group_inputs = {k: _get(z, f"group_input/{k}") for k in _names(z, "group_inputs")}
        group = z["group"]
        ngroups = int(z["ngroups"])
        methods = {}
        for m in _names(z, "methods"):
            pre = f"method/{m}/"
            fields = {}
            for key in z.files:
                if key.startswith(pre) and not key.endswith(".shape"):
                    f = key[len(pre) :]
                    if f not in ("over", "index", "dropped", "hi", "lo"):
                        fields[f] = _get(z, key)
            over = str(z[pre + "over"])
            if pre + "index" in z.files:
                index = z[pre + "index"]
            else:
                index = np.arange(group.size if over == "point" else ngroups, dtype=np.int64)
            methods[m] = Method(over, index, _get(z, pre + "hi"), _get(z, pre + "lo"), z[pre + "dropped"], fields)
        return Ref(meta, inputs, group, ngroups, group_inputs, methods)


def provenance(path) -> dict:
    """{"arb", "target_bits", "tool", "n_requested"} of a reference file, without its arrays."""
    with np.load(path) as z:
        return {
            "arb": str(z["arb"]),
            "target_bits": int(z["target_bits"]),
            "tool": str(z["tool"]),
            "n_requested": int(z["n_requested"]),
        }
