"""Cases whose results M6 changes on purpose (DESIGN.md §9), per area:
(regex on the case name, reason). golden.py check reports them per reason and
flags patterns that match nothing. Message-only differences of a listed case
are accepted too.
"""

EXPECTED = {
    "array": [
        (r"^asarray dtype string err$", "§9.2: a dtype argument takes the name too (dtype='float64')"),
        (r"dtype (numpy|python type|python) err$", "§9.2: the dtype message names the accepted names (message)"),
    ],
    "linalg": [
        (r"^trace/err numpy dtype$", "§9.2: the dtype message names the accepted names (message)"),
    ],
    "frame": [
        (r"^ctor dtype (float from int|string)$", "§9.2: Python types (float, str) are not dtypes: TypeError"),
        (r"^(prop (big )?df dtypes|repr dtypes)$", "§9.2: DataFrame.dtypes is {column: dtype object}"),
        (r"^err (ctor dtype unknown|astype bad dtype|csv dtype bad)$", "§9.2: the dtype message names the accepted spellings (message)"),
        (r"^err (arith \w+ b,[bi]|unary (neg|abs) b|df neg string)$", "§9.1 #2: bool arithmetic is int64 arithmetic"),
        (r"^cmp \w+ (exact|exact scalar|int vs float scalar 2\*\*63)$", "§9.1 #3: int vs float compares as float64"),
        (r"^(df set loc lossless|set loc slice scalar lossless|fillna i float lossless"
         r"|err (set|df set|fillna i) lossy|err set (bool into int|int into bool|string into int)"
         r"|err fillna [bs] int|err df fillna scalar mixed)$", "§9.1 #4: assignment and fillna convert by dtype only"),
        (r"^(err ctor (dtype (bool|string) from int|dtype float inexact|float to int lossy|nan to int|int\+float inexact)"
         r"|err to_numpy dtype lossy)$", "§9.1 #5: dtype= is an explicit conversion; int with float rounds"),
        (r"^err (big )?astype f->int", "§9.1 #6: astype float -> int saturates, NaN -> 0"),
        (r"^((big )?(df )?reduce (std|var)(?!.*ddof=).*|(big )?groupby .* (std|var)|groupby agg named|big groupby agg dict)$",
         "§9.1 #7: std/var default correction=0"),
        (r"^err reduce (std ddof negative|var ddof float)$", "§9.1 #7: correction takes any int or float, as in the array area"),
        (r"^err (arith add i,2\*\*64|ctor int too big)$", "§9.1 #12: the array's message for an int outside int64 (message)"),
        (r"^err threads (float|keyword)$", "M5: the thread setter is sd.set_num_threads (API.md §3)"),
    ],
}
