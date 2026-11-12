Machine: arm64 26.6.2, python 3.15.0, 14 cpus. ns per call, best of 7 alternating rounds.

| call | before S0 (4 operands) | S0 (5 operands) | S0 / before |
|---|---:|---:|---:|
| add(1.0, 2.0) scalars | 166 | 166 | 1.001 |
| add size 1 | 168 | 166 | 0.992 |
| add size 16 | 169 | 186 | 1.098 |
| x16 + y16 (operator) | 182 | 175 | 0.960 |
| add size 1000 | 275 | 284 | 1.034 |
| add int64 + float64 size 16 (buffered) | 184 | 186 | 1.013 |
| exp size 16 | 161 | 164 | 1.018 |
| where size 16 | 201 | 201 | 0.999 |
| clip size 16 | 259 | 262 | 1.013 |
| add (8,8,8) transposed + C | 471 | 469 | 0.996 |
| sum size 1000 | 188 | 189 | 1.004 |
| add 10^6, 10 threads | 36315 | 37428 | 1.031 |

Rerun of the size-16 rows (`--rounds 9 --only 16`), same sitting:

| call | before S0 (4 operands) | S0 (5 operands) | S0 / before |
|---|---:|---:|---:|
| add size 16 | 172 | 181 | 1.055 |
| x16 + y16 (operator) | 178 | 186 | 1.045 |
| add int64 + float64 size 16 (buffered) | 181 | 187 | 1.036 |
| exp size 16 | 166 | 163 | 0.981 |
| where size 16 | 200 | 209 | 1.042 |
| clip size 16 | 253 | 256 | 1.013 |
