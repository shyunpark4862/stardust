"""numpy 2.5.5 on the integer edge cases of Table tab:int-edges (run with uv: --with numpy)."""
import numpy as np, warnings
warnings.simplefilter("ignore")
M=np.iinfo(np.int64).min
def show(name, f):
    try:
        with np.errstate(all='ignore'):
            r=f()
        print(f"{name:32s} -> {r!r}")
    except Exception as e:
        print(f"{name:32s} -> {type(e).__name__}: {e}")
a=lambda *v: np.array(v,dtype=np.int64)
show("INT_MIN // -1", lambda: a(M)//a(-1))
show("INT_MIN % -1", lambda: a(M)%a(-1))
show("7 // 0", lambda: a(7)//a(0))
show("7 % 0", lambda: a(7)%a(0))
show("-7 // 2", lambda: a(-7)//a(2))
show("-7 % 2", lambda: a(-7)%a(2))
show("7 % -2", lambda: a(7)%a(-2))
show("abs(INT_MIN)", lambda: np.abs(a(M)))
show("-INT_MIN", lambda: -a(M))
show("1 << 64", lambda: np.left_shift(a(1),a(64)))
show("1 << 63", lambda: np.left_shift(a(1),a(63)))
show("1 << -1", lambda: np.left_shift(a(1),a(-1)))
show("-8 >> 64", lambda: np.right_shift(a(-8),a(64)))
show("-8 >> 1", lambda: np.right_shift(a(-8),a(1)))
show("-8 >> -1", lambda: np.right_shift(a(-8),a(-1)))
show("2 ** -1", lambda: a(2)**a(-1))
show("3 ** 41", lambda: a(3)**a(41))
show("python 3**41 mod", lambda: ((3**41 + 2**63) % 2**64) - 2**63)
show("INT_MAX + 1", lambda: a(2**63-1)+a(1))
print(np.__version__)
