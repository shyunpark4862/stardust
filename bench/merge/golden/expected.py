"""Cases whose results M6 changes on purpose (DESIGN.md §9), per area:
(regex on the case name, reason). golden.py check reports them per reason and
flags patterns that match nothing. Message-only differences of a listed case
are accepted too.
"""

EXPECTED = {
    "array": [
        (r"^asarray dtype string err$", "§9.2: a dtype argument takes the name too (dtype='float64')"),
        (r"dtype (numpy|python type|python) err$", "§9.2: the dtype message names the accepted names (message)"),
        (r"^(arg)?sort (n|f special) axis=-?\d desc=True |^(arg)?sort large n desc$",
         "§9.1 #9: NaN is the largest value: first when descending"),
        (r"^unique_(values|counts|inverse|all) (n|nans|f special|c|large n)$",
         "§9.1 #10: all NaNs are one value, sorted last"),
        (r"^op (eq|ne) (list|numpy|None)", "§9.1 #13: == and != with an operand the other operators reject is a TypeError"),
        (r"^full_like ((int|bool) 1\.5|int with 1\.5)$",
         "§9.5: full_like without dtype= converts implicitly: 1.5 into an int or bool array is a TypeError"),
        (r"^full_like (int|float) with 1j err$", "§9.5: full_like without dtype= converts implicitly: the assignment message (message)"),
        (r"^ufunc (divide|atan2) big int$", "§9.5: a Python int outside int64 is an OverflowError in divide and atan2 too"),
        (r"^arange (\(nan,\)|\(0, inf\) err)$", "§9.5: nan or inf in arange is a ValueError"),
        (r"^(permute_dims (list axes|3d \(0, 1, 2\.0\)) err|expand_dims 3d 2\.0 err|red \w+ axis (float|list) err)$",
         "§9.5: an axis argument's message says an int or a tuple of ints (message)"),
        (r"^setattr (shape|dtype) err$", "§9.5: assigning an attribute names it (message)"),
        (r"^(ufunc (real|imag) pyscalar [bifc] err|pyscalar real complex err)$", "§9.5: real and imag take Python scalars, like conj"),
        (r"^ufunc (real|imag) (list|numpy) err$", "§9.5: real and imag take Python scalars: the message lists them (message)"),
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
        (r"^((big )?reduce mean (huge )?i( skipna=False)?|df reduce mean)$", "§9.1 #11: int mean is the array area's float64 sum over the count"),
        (r"^err ctor scalar$", "§4.3: a Series can be made from an sd.ndarray (message lists it)"),
        (r"^err threads (float|keyword)$", "M5: the thread setter is sd.set_num_threads (API.md §3)"),
        (r"^sig keyword (head|astype|fillna|reindex|df sort_values|groupby|duplicated|merge|concat|Index|Series|equals)$",
         "§9.3: main arguments are positional-only: passing one by keyword is a TypeError"),
        (r"^err (sort_values ascending int|df sort_values ascending length)$",
         "§9.3: ascending -> descending: the message names descending (message)"),
        (r"^err sig tolist$", "§9.4: to_list -> tolist, no alias: the absent spelling is to_list (message)"),
        (r"^(big )?unique ", "§9.4: unique() -> unique_values(): sorted order, NaN once, NA last"),
        (r"^(err logic (and|or|xor) (i,i|b,i|b,1)|err unary invert i|err df and int)$",
         "§9.4: int64 bitwise operators & | ^ ~ (bool with int64 is int64); a float column still raises"),
    ],
}
