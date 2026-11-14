"""Copy the Boost.Math headers sd.special uses into third_party/boost_math/ (user decision
2026-11-14: Boost where it is no loss, our code elsewhere; only the headers needed, in the
repository).

    .venv/bin/python tools/vendor_boost_math.py <boostorg/math checkout>

The header set is what clang's -MM reports for the entry headers below in standalone mode
(BOOST_MATH_STANDALONE: no other Boost library). Adding a Boost function to sd.special means
adding its header here and running the tool again.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "third_party" / "boost_math"
ENTRY = [
    "boost/math/policies/policy.hpp",
    "boost/math/special_functions/airy.hpp",
    "boost/math/special_functions/bessel.hpp",
    "boost/math/special_functions/beta.hpp",
    "boost/math/special_functions/digamma.hpp",
    "boost/math/special_functions/expint.hpp",
    "boost/math/special_functions/gamma.hpp",
    "boost/math/special_functions/hypergeometric_1F1.hpp",
    "boost/math/special_functions/lambert_w.hpp",
    "boost/math/special_functions/zeta.hpp",
]


def main(src):
    src = pathlib.Path(src).resolve()
    inc = src / "include"
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        tu = pathlib.Path(tmp) / "need.cpp"
        tu.write_text("#define BOOST_MATH_STANDALONE\n" + "".join(f"#include <{h}>\n" for h in ENTRY))
        deps = subprocess.run(["clang++", "-std=c++20", f"-I{inc}", "-MM", str(tu)], capture_output=True,
                              text=True, check=True).stdout
    files = sorted({pathlib.Path(p) for p in deps.replace("\\\n", " ").split() if p.startswith(str(inc))})
    if (OUT / "include").exists():
        shutil.rmtree(OUT / "include")
    for f in files:
        dst = OUT / "include" / f.relative_to(inc)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
    shutil.copy2(src / "LICENSE", OUT / "LICENSE")
    size = sum(f.stat().st_size for f in files)
    (OUT / "README.md").write_text(f"""# Boost.Math (부분)

sd.special이 부르는 Boost.Math 헤더만 옮겨 둔 것이다 (docs/design/special.md §5.14).

- 출처: https://github.com/boostorg/math, 커밋 {commit}
- 라이선스: Boost Software License 1.0 (`LICENSE`)
- 헤더 {len(files)}개, {size / 1e6:.1f} MB. standalone 모드(`BOOST_MATH_STANDALONE`)라 다른 Boost 라이브러리가 필요 없다.
- 고치지 않는다. 다시 만들려면 `tools/vendor_boost_math.py <checkout>`
""")
    print(f"{len(files)} headers, {size / 1e6:.1f} MB, commit {commit} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1])
