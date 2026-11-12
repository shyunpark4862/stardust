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
        (r"^err threads (float|keyword)$", "M5: the thread setter is sd.set_num_threads (API.md §3)"),
    ],
}
