"""Python's call-shape errors in the library's form (API.md §2.1).

A wrong call shape (too many positional arguments, a missing or an unexpected argument, a
positional-only one by keyword) raises Python's own TypeError, "minimize() got an unexpected
keyword argument 'foo'", "Series.__init__() takes ...". The heavy entry points -- module
functions, constructors, read_csv, merge, concat, groupby, the regression and optimize
functions, the descriptive statistics -- report it as every other error of the library:
"minimize(): unexpected keyword argument 'foo'", "Series(): takes ...". Small methods keep
Python's form: a wrapper costs about 60 ns a call, too much on calls of a few hundred.

Only the call's own binding error is converted; a TypeError raised inside the wrapped
function passes unchanged.

The forms:
    function(fn)         a module function, named by its __name__
    method(name)         a method or an __init__ named `name`; Python's counts include self,
                         which the caller never passes
    methods(cls)         every public method defined in cls, each named by itself
    keyword_init(cls)    a constructor with keyword-only parameters, named by the class called
"""

import functools
import inspect
import re

_TAKES = re.compile(
    r"takes (from (\d+) to )?(\d+) positional arguments? but (\d+)( positional arguments?)?(.*) (was|were) given"
)


def message(name, rest, self_counted=False):
    """Python's call-shape error `rest` (the text after "f() ") as "name(): ...":
    "name(): takes 2 positional arguments but 3 were given", "name(): unexpected keyword
    argument 'foo'". `self_counted`: Python's numbers count self, which are reported without
    it ("summary(): takes 0 positional arguments but 1 was given")."""
    m = _TAKES.match(rest)
    if m:
        d = 1 if self_counted else 0
        lo = int(m.group(2)) - d if m.group(2) else None
        hi, got = int(m.group(3)) - d, int(m.group(4)) - d
        span = f"from {lo} to {hi}" if lo is not None else f"{hi}"
        noun = "argument" if hi == 1 and lo is None else "arguments"
        was = "was" if got == 1 else "were"
        tail = m.group(6) or ""
        if m.group(5):
            tail = f" positional argument{'' if got == 1 else 's'}{tail}"
        rest = f"takes {span} positional {noun} but {got}{tail} {was} given"
    elif rest.startswith("got an unexpected"):
        rest = rest[len("got an ") :]
    return f"{name}(): {rest}"


def _converted(e, head, name, self_counted):
    """The library's form of e when e is the binding error of the function named by `head` (no
    frame of the function beneath the wrapper's), else None."""
    msg = str(e)
    if not msg.startswith(head) or e.__traceback__.tb_next is not None:
        return None
    return TypeError(message(name, msg[len(head) :], self_counted))


def _wrap(fn, name, self_counted):
    head = f"{fn.__qualname__}() "

    @functools.wraps(fn)
    def call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            c = _converted(e, head, name, self_counted)
            if c is None:
                raise
            raise c from None

    return call


def function(fn):
    """fn (a module function) with its call-shape errors as "fn(): ..."."""
    return _wrap(fn, fn.__name__, False)


def method(name):
    """A decorator for a method or an __init__: its call-shape errors as "name(): ...", with
    self left out of the counts ("Series(): ...", "groupby(): ...")."""
    return lambda fn: _wrap(fn, name, True)


def methods(cls):
    """Wraps every public method defined in cls (not one wrapped already, by a base class) so
    that its call-shape errors say the method's name: "summary(): takes 0 positional arguments
    but 1 was given", not "_LinearResult.summary() takes 1 ...". Returns cls."""
    for name, fn in list(vars(cls).items()):
        if name.startswith("_") or not inspect.isfunction(fn) or getattr(fn, "_calls_wrapped", False):
            continue
        setattr(cls, name, _bound(fn, name))
    return cls


def _bound(fn, name):
    """methods()' wrapper: self is its own parameter, so a call without self or with self= is
    Python's error at the wrapper, not converted."""
    head = f"{fn.__qualname__}() "

    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except TypeError as e:
            c = _converted(e, head, name, True)
            if c is None:
                raise
            raise c from None

    call._calls_wrapped = True
    return call


def keyword_init(cls):
    """cls.__init__ (keyword-only parameters) with its call-shape errors named by the class
    called, which may be a subclass sharing the __init__: "Normal(): unexpected keyword argument
    'cool'", "Gamma(): missing 1 required keyword-only argument: 'a'", "Normal(): takes no
    positional arguments but 1 was given", not "Normal.__init__() got ..."."""
    init = cls.__dict__.get("__init__")
    if init is None or getattr(init, "_calls_wrapped", False):
        return
    head = f"{init.__qualname__}() "

    def __init__(self, *args, **kwargs):
        if args:
            n = len(args)
            raise TypeError(
                f"{type(self).__name__}(): takes no positional arguments but {n} {'was' if n == 1 else 'were'} given"
            )
        try:
            init(self, **kwargs)
        except TypeError as e:
            c = _converted(e, head, type(self).__name__, True)
            if c is None:
                raise
            raise c from None

    __init__.__wrapped__ = init  # (inspect.signature shows the keyword-only parameters)
    __init__.__doc__ = init.__doc__
    __init__.__qualname__ = init.__qualname__
    __init__._calls_wrapped = True
    cls.__init__ = __init__
