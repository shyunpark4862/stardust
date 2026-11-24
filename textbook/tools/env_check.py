"""Compare protected LaTeX environments (theorem-like, proof, algorithm) between two files.
Usage: python3 tools/env_check.py BEFORE AFTER [--fix OUT]
With --fix, writes AFTER with each differing block replaced by BEFORE's block (same order)."""
import re, sys
ENVS = "theorem|lemma|proposition|corollary|definition|assumption|proof|algorithm|algorithmic"
pat = re.compile(r"\\begin\{(" + ENVS + r")\}.*?\\end\{\1\}", re.S)
b = open(sys.argv[1]).read(); a = open(sys.argv[2]).read()
bb = [m for m in pat.finditer(b)]; aa = [m for m in pat.finditer(a)]
print(f"blocks before {len(bb)} after {len(aa)}")
diff = [i for i, (x, y) in enumerate(zip(bb, aa)) if x.group(0) != y.group(0)]
print("differing:", diff)
if "--fix" in sys.argv and len(bb) == len(aa):
    out = []; pos = 0
    for x, y in zip(bb, aa):
        out.append(a[pos:y.start()]); out.append(x.group(0)); pos = y.end()
    out.append(a[pos:])
    open(sys.argv[sys.argv.index("--fix") + 1], "w").write("".join(out))
    print("fixed")
