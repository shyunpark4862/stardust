// The matchings and the assignment (docs/design/graph.md §2.4, §3.5; milestone GR3).
//
//   _bipartite_matching(indptr, indices, rows, cols) -> col_of_row (int64 (rows,), -1 for a free row)
//       The largest matching of the bipartite graph whose rows are the CSR's rows and whose columns are
//       its column indices (the pattern; a stored 0 is an edge): Hopcroft-Karp. Each phase lays out the
//       breadth-first levels from the free rows (ascending) and then searches depth first from each free
//       row (ascending; the neighbours ascending, on current arcs) for vertex-disjoint shortest augmenting
//       paths, until a phase finds none.
//   _assign_dense(cost, maximize) -> (status, row_ind, col_ind, u, v)
//       linear_sum_assignment of a C-contiguous float64 (r, c) matrix: every entry is a pair, +inf (-inf
//       with maximize) one that cannot be chosen. status 0: solved; 1: no assignment of the smaller side
//       (row_ind and the rest None); 2: an entry that is NaN, or -inf (+inf with maximize).
//   _assign_sparse(indptr, indices, data, rows, cols, maximize) -> (status, row_ind, col_ind, u, v)
//       min_weight_full_bipartite_matching: the stored entries are the pairs (finite weights: the Python
//       layer checks them). status 0 or 1.
//
// The two assignments share one solver, Jonker and Volgenant's (LAPJV), with dual variables u (rows) and
// v (columns). When rows > cols the problem is transposed first (scipy's convention). The start: a square
// problem's column reduction and reduction transfer, then (any shape) two passes of the augmenting row
// reduction, which leave few rows free. Each free row, ascending, then gets a shortest augmenting path:
// Dijkstra's on the reduced costs c(i, j) - u(i) - v(j) >= 0 with Crouse's dual update (the form scipy's
// linear_sum_assignment follows). The search takes the columns in the order (distance, free first, index),
// a total order, so the dense scan and the sparse heap take the same columns, and every step of the start
// scans a row in ascending column order: for the same pairs the dense and the sparse solver give the same
// assignment, the one this fixed order picks among ties (decision GR16). u, v are returned for the tests:
// with integer costs every quantity is an integer below 2^53 and exact, so u(i) + v(j) <= c(i, j) on every
// pair, with equality on the chosen ones (and v <= 0 on a free column), proves the optimum exactly.
#include "array/ndarray.hpp"
#include "bind/pyutil.hpp"
#include "graph/graph.hpp"
#include "graph/graph_mod.hpp"
#include "parallel/pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace nb = nanobind;

namespace sd::graph {

namespace {

using std::int64_t;

constexpr double kInf = std::numeric_limits<double>::infinity();
// The steps of a pass of the augmenting row reduction, per row (decision GR45): past them a freed row
// waits for the next pass or the searches. Measured on dense 1000 x 1000 and sparse 5000 x 5000 costs.
constexpr int64_t kReduceSteps = 4;

nb::object vec1(DType t, int64_t n) {
    const int64_t shape[] = {n};
    return wrap(new_array(t, shape));
}

template <class T>
T* ptr(nb::object& o) {
    return reinterpret_cast<T*>(as_ndarray(o).data);
}

[[noreturn]] void wrong(const char* func, const char* what) {
    throw nb::type_error((std::string(func) + "(): wrong " + what).c_str());
}

const NDArray& vec_arg(nb::handle o, const char* func, DType dtype, const char* what) {
    if (!is_ndarray(o)) wrong(func, what);
    const NDArray& a = as_ndarray(o);
    if (a.ndim != 1 || !a.c_contiguous || a.dtype != dtype) wrong(func, what);
    return a;
}

// A rectangular CSR read in place.
struct Bip {
    int64_t rows = 0, cols = 0;
    const int64_t* ip = nullptr;
    const int64_t* ix = nullptr;
    const double* w = nullptr;
};

Bip bip_arg(nb::handle ip, nb::handle ix, nb::handle dh, int64_t rows, int64_t cols, const char* func) {
    const NDArray& p = vec_arg(ip, func, DType::Int64, "indptr");
    const NDArray& i = vec_arg(ix, func, DType::Int64, "indices");
    if (rows < 0 || cols < 0 || p.shape[0] != rows + 1) wrong(func, "indptr");
    Bip b{rows, cols, reinterpret_cast<const int64_t*>(p.data), reinterpret_cast<const int64_t*>(i.data), nullptr};
    if (b.ip[0] != 0 || b.ip[rows] != i.shape[0]) wrong(func, "indptr");
    if (!dh.is_none()) {
        const NDArray& d = vec_arg(dh, func, DType::Float64, "data");
        if (d.shape[0] != i.shape[0]) wrong(func, "data");
        b.w = reinterpret_cast<const double*>(d.data);
    }
    return b;
}

// ---- Hopcroft-Karp

void hopcroft_karp(const Bip& g, std::vector<int64_t>& col_of_row) {
    const int64_t R = g.rows, Cn = g.cols;
    constexpr int64_t kFar = INT64_MAX;
    col_of_row.assign(static_cast<size_t>(R), -1);
    std::vector<int64_t> row_of_col(static_cast<size_t>(Cn), -1), dist(static_cast<size_t>(R)),
        queue(static_cast<size_t>(R));
    std::vector<int64_t> arc(static_cast<size_t>(R)), stack;
    // a greedy start: each row (ascending) takes its smallest free column
    for (int64_t i = 0; i < R; ++i) {
        for (int64_t k = g.ip[i]; k < g.ip[i + 1]; ++k) {
            const int64_t j = g.ix[k];
            if (row_of_col[static_cast<size_t>(j)] < 0) {
                row_of_col[static_cast<size_t>(j)] = i;
                col_of_row[static_cast<size_t>(i)] = j;
                break;
            }
        }
    }
    for (;;) {
        int64_t head = 0, tail = 0, limit = kFar;
        for (int64_t i = 0; i < R; ++i) {
            if (col_of_row[static_cast<size_t>(i)] < 0) {
                dist[static_cast<size_t>(i)] = 0;
                queue[static_cast<size_t>(tail++)] = i;
            } else {
                dist[static_cast<size_t>(i)] = kFar;
            }
        }
        while (head < tail) {
            const int64_t i = queue[static_cast<size_t>(head++)];
            const int64_t d = dist[static_cast<size_t>(i)];
            if (d >= limit) break;
            for (int64_t k = g.ip[i]; k < g.ip[i + 1]; ++k) {
                const int64_t w = row_of_col[static_cast<size_t>(g.ix[k])];
                if (w < 0) {
                    limit = std::min(limit, d + 1);
                } else if (dist[static_cast<size_t>(w)] == kFar) {
                    dist[static_cast<size_t>(w)] = d + 1;
                    queue[static_cast<size_t>(tail++)] = w;
                }
            }
        }
        if (limit == kFar) return;
        for (int64_t i = 0; i < R; ++i) arc[static_cast<size_t>(i)] = g.ip[i];
        for (int64_t root = 0; root < R; ++root) {
            if (col_of_row[static_cast<size_t>(root)] >= 0 || dist[static_cast<size_t>(root)] != 0) continue;
            // depth first from root over rows: stack holds the rows of the alternating path
            stack.clear();
            stack.push_back(root);
            int64_t found = -1;  // the free column that ends the path
            while (!stack.empty() && found < 0) {
                const int64_t i = stack.back();
                const int64_t di = dist[static_cast<size_t>(i)];
                int64_t& k = arc[static_cast<size_t>(i)];
                bool moved = false;
                for (; k < g.ip[i + 1]; ++k) {
                    const int64_t j = g.ix[k];
                    const int64_t w = row_of_col[static_cast<size_t>(j)];
                    if (w < 0) {
                        if (di + 1 == limit) {
                            found = j;
                            moved = true;
                            break;
                        }
                    } else if (dist[static_cast<size_t>(w)] == di + 1) {
                        stack.push_back(w);
                        moved = true;
                        break;
                    }
                }
                if (!moved) {  // a dead end
                    dist[static_cast<size_t>(i)] = kFar;
                    stack.pop_back();
                    if (!stack.empty()) ++arc[static_cast<size_t>(stack.back())];
                }
            }
            if (found < 0) continue;
            // augment: the rows of the stack take the columns their current arcs point at
            int64_t j = found;
            for (size_t q = stack.size(); q-- > 0;) {
                const int64_t i = stack[q];
                const int64_t prev = col_of_row[static_cast<size_t>(i)];
                col_of_row[static_cast<size_t>(i)] = j;
                row_of_col[static_cast<size_t>(j)] = i;
                dist[static_cast<size_t>(i)] = kFar;  // a row of an augmenting path is used up in this phase
                j = prev;
            }
        }
    }
}

nb::object bipartite_matching(nb::handle ip, nb::handle ix, int64_t rows, int64_t cols) {
    const Bip g = bip_arg(ip, ix, nb::none(), rows, cols, "_bipartite_matching");
    std::vector<int64_t> col_of_row;
    {
        const parallel::Unlocked unlocked(g.ip[rows] >= kParallelWork);
        hopcroft_karp(g, col_of_row);
    }
    nb::object out = vec1(DType::Int64, rows);
    std::copy(col_of_row.begin(), col_of_row.end(), ptr<int64_t>(out));
    return out;
}

// ---- the shortest augmenting path

// The costs of one row of the assignment: dense (a row of the matrix, +inf for no pair) or sparse.
struct DenseRows {
    const double* c;
    int64_t cols;
    double sign;
    template <class F>
    void each(int64_t i, F&& f) const {  // +inf entries too: they never come first
        const double* row = c + i * cols;
        for (int64_t j = 0; j < cols; ++j) f(j, sign * row[j]);
    }
};

struct SparseRows {
    const int64_t* ip;
    const int64_t* ix;
    const double* w;
    double sign;
    template <class F>
    void each(int64_t i, F&& f) const {
        for (int64_t k = ip[i]; k < ip[i + 1]; ++k) f(ix[k], sign * w[k]);
    }
};

struct Solver {
    int64_t R, Cn;
    std::vector<int64_t> col4row, row4col, path, sr, sc;
    std::vector<double> u, v, shortest;
    std::vector<uint8_t> in_sc, touched;
    std::vector<int64_t> touched_list;
    // the indexed binary heap of the sparse search
    std::vector<int64_t> heap, pos;

    Solver(int64_t r, int64_t c)
        : R(r),
          Cn(c),
          col4row(static_cast<size_t>(r), -1),
          row4col(static_cast<size_t>(c), -1),
          path(static_cast<size_t>(c), -1),
          u(static_cast<size_t>(r), 0.0),
          v(static_cast<size_t>(c), 0.0),
          shortest(static_cast<size_t>(c), kInf),
          in_sc(static_cast<size_t>(c), 0),
          touched(static_cast<size_t>(c), 0),
          pos(static_cast<size_t>(c), -1) {}

    // Whether column a comes before column b in the order of the search: distance, free first, index.
    bool before(int64_t a, int64_t b) const {
        const double x = shortest[static_cast<size_t>(a)], y = shortest[static_cast<size_t>(b)];
        if (x != y) return x < y;
        const bool fa = row4col[static_cast<size_t>(a)] < 0, fb = row4col[static_cast<size_t>(b)] < 0;
        if (fa != fb) return fa;
        return a < b;
    }

    // The dual update and the augmentation once the search from cur reached the free column sink at
    // distance min_val (Crouse's update).
    void finish(int64_t cur, int64_t sink, double min_val) {
        u[static_cast<size_t>(cur)] += min_val;
        for (int64_t i : sr)
            if (i != cur)
                u[static_cast<size_t>(i)] += min_val - shortest[static_cast<size_t>(col4row[static_cast<size_t>(i)])];
        for (int64_t j : sc) v[static_cast<size_t>(j)] -= min_val - shortest[static_cast<size_t>(j)];
        int64_t j = sink;
        for (;;) {
            const int64_t i = path[static_cast<size_t>(j)];
            row4col[static_cast<size_t>(j)] = i;
            std::swap(col4row[static_cast<size_t>(i)], j);
            if (i == cur) break;
        }
    }

    // ---- the start of a square problem: column reduction and reduction transfer (Jonker-Volgenant)
    //
    // v(j) is the least cost of column j (u = 0: every reduced cost >= 0), and the columns, from the last
    // to the first, each take the first row attaining their least cost when that row is still free. A row
    // taken by exactly one column then lowers that column's v by its second least value, so the column
    // stays least in its row and is tied with the second. v may be positive: a square problem assigns
    // every column, so v has no sign to keep. false: a row or a column without a pair.
    template <class Rows>
    bool reduce_columns(const Rows& m, std::vector<int64_t>& free) {
        std::vector<int64_t> argrow(static_cast<size_t>(Cn), -1), matches(static_cast<size_t>(R), 0);
        std::fill(v.begin(), v.end(), kInf);
        for (int64_t i = 0; i < R; ++i) {
            m.each(i, [&](int64_t j, double x) {
                if (x < v[static_cast<size_t>(j)]) {
                    v[static_cast<size_t>(j)] = x;
                    argrow[static_cast<size_t>(j)] = i;
                }
            });
        }
        for (int64_t j = Cn - 1; j >= 0; --j) {
            const int64_t i = argrow[static_cast<size_t>(j)];
            if (i < 0) return false;
            if (++matches[static_cast<size_t>(i)] == 1) {
                col4row[static_cast<size_t>(i)] = j;
                row4col[static_cast<size_t>(j)] = i;
            }
        }
        free.clear();
        for (int64_t i = 0; i < R; ++i) {
            if (matches[static_cast<size_t>(i)] == 0) {
                free.push_back(i);
            } else if (matches[static_cast<size_t>(i)] == 1) {
                const int64_t j1 = col4row[static_cast<size_t>(i)];
                double low = kInf;
                m.each(i, [&](int64_t j, double x) {
                    if (j != j1) low = std::min(low, x - v[static_cast<size_t>(j)]);
                });
                if (low < kInf) v[static_cast<size_t>(j1)] -= low;
            }
        }
        return true;
    }

    // ---- the start: augmenting row reduction (Jonker-Volgenant), two passes over the free rows
    //
    // A square problem starts from the column reduction above, any other shape from v = 0 with every row
    // free. v only decreases here and a column once taken stays taken, so in a rectangular problem v <= 0
    // and every free column keeps v = 0 (its dual constraints). The invariant: an assigned row's
    // column attains the least c(i, k) - v(k) of its row (lowering another column's v only raises that
    // column's value). A free row i takes its least column j1 (value v1; the second least v2 at j2) and
    // lowers v(j1) by v2 - v1, so j1 stays least and tied with j2; when v1 = v2 and j1 is taken it takes j2
    // instead. The row j1 held is freed: searched again at once when v(j1) dropped, else on the next pass.
    // A row with one pair only (no j2) keeps v. A pass takes at most kReduceSteps steps per row (and
    // Jonker-Volgenant's own bound). Afterwards u(i) is the value of i's column (tight) or the
    // least value of a free row, so the searches below start from feasible duals and tight pairs.
    // false: a row without a pair.
    template <class Rows>
    bool reduce_rows(const Rows& m, std::vector<int64_t>& free) {
        if (R == Cn) {
            if (!reduce_columns(m, free)) return false;
        } else {
            free.resize(static_cast<size_t>(R));
            std::iota(free.begin(), free.end(), int64_t{0});
        }
        for (int pass = 0; pass < 2 && !free.empty(); ++pass) {
            int64_t current = 0, kept = 0, steps = 0;
            const int64_t nfree = static_cast<int64_t>(free.size());
            while (current < nfree) {
                ++steps;
                const int64_t i = free[static_cast<size_t>(current++)];
                double v1 = kInf, v2 = kInf;
                int64_t j1 = -1, j2 = -1;
                m.each(i, [&](int64_t j, double x) {
                    const double c = x - v[static_cast<size_t>(j)];
                    if (c < v1) {
                        v2 = v1;
                        j2 = j1;
                        v1 = c;
                        j1 = j;
                    } else if (c < v2) {
                        v2 = c;
                        j2 = j;
                    }
                });
                if (j1 < 0) return false;
                int64_t i0 = row4col[static_cast<size_t>(j1)];
                const double vj = v[static_cast<size_t>(j1)];
                const double lowered = j2 >= 0 ? vj - (v2 - v1) : vj;
                const bool lowers = lowered < vj;
                if (steps < current * R && steps < kReduceSteps * R) {
                    if (lowers) {
                        v[static_cast<size_t>(j1)] = lowered;
                    } else if (i0 >= 0 && j2 >= 0) {
                        j1 = j2;
                        i0 = row4col[static_cast<size_t>(j2)];
                    }
                    if (i0 >= 0) {
                        if (lowers) free[static_cast<size_t>(--current)] = i0;
                        else free[static_cast<size_t>(kept++)] = i0;
                    }
                } else if (i0 >= 0) {
                    free[static_cast<size_t>(kept++)] = i0;
                }
                if (i0 >= 0) col4row[static_cast<size_t>(i0)] = -1;
                col4row[static_cast<size_t>(i)] = j1;
                row4col[static_cast<size_t>(j1)] = i;
            }
            free.resize(static_cast<size_t>(kept));
        }
        std::sort(free.begin(), free.end());
        for (int64_t i = 0; i < R; ++i) {
            const int64_t ji = col4row[static_cast<size_t>(i)];
            double low = kInf;
            m.each(i, [&](int64_t j, double x) {
                if (ji < 0 || j == ji) low = std::min(low, x - v[static_cast<size_t>(j)]);
            });
            u[static_cast<size_t>(i)] = low;
        }
        return true;
    }

    // ---- dense: every column is scanned at each step
    bool run_dense(const DenseRows& m) {
        std::vector<int64_t> todo;
        if (!reduce_rows(m, todo)) return false;
        std::vector<int64_t> remaining(static_cast<size_t>(Cn));
        for (int64_t cur : todo) {
            std::iota(remaining.begin(), remaining.end(), int64_t{0});
            int64_t nrem = Cn;
            std::fill(shortest.begin(), shortest.end(), kInf);
            sr.clear();
            sc.clear();
            int64_t i = cur, sink = -1;
            double min_val = 0.0;
            while (sink < 0) {
                sr.push_back(i);
                const double* row = m.c + i * m.cols;
                const double ui = u[static_cast<size_t>(i)];
                int64_t best = -1, at = -1;
                for (int64_t q = 0; q < nrem; ++q) {
                    const int64_t j = remaining[static_cast<size_t>(q)];
                    const double r = min_val + m.sign * row[j] - ui - v[static_cast<size_t>(j)];
                    if (r < shortest[static_cast<size_t>(j)]) {
                        path[static_cast<size_t>(j)] = i;
                        shortest[static_cast<size_t>(j)] = r;
                    }
                    if (best < 0 || before(j, best)) {
                        best = j;
                        at = q;
                    }
                }
                if (best < 0 || !(shortest[static_cast<size_t>(best)] < kInf)) return false;
                min_val = shortest[static_cast<size_t>(best)];
                sc.push_back(best);
                remaining[static_cast<size_t>(at)] = remaining[static_cast<size_t>(--nrem)];
                if (row4col[static_cast<size_t>(best)] < 0) sink = best;
                else i = row4col[static_cast<size_t>(best)];
            }
            finish(cur, sink, min_val);
        }
        return true;
    }

    // ---- sparse: the adjacent columns are relaxed and a heap holds the reached ones
    void heap_up(int64_t q) {
        const int64_t x = heap[static_cast<size_t>(q)];
        while (q > 0) {
            const int64_t p = (q - 1) / 2;
            if (!before(x, heap[static_cast<size_t>(p)])) break;
            heap[static_cast<size_t>(q)] = heap[static_cast<size_t>(p)];
            pos[static_cast<size_t>(heap[static_cast<size_t>(q)])] = q;
            q = p;
        }
        heap[static_cast<size_t>(q)] = x;
        pos[static_cast<size_t>(x)] = q;
    }

    int64_t heap_pop() {
        const int64_t top = heap[0];
        pos[static_cast<size_t>(top)] = -1;
        const int64_t x = heap.back();
        heap.pop_back();
        const int64_t m = static_cast<int64_t>(heap.size());
        if (m == 0) return top;
        int64_t q = 0;
        for (;;) {
            int64_t c = 2 * q + 1;
            if (c >= m) break;
            if (c + 1 < m && before(heap[static_cast<size_t>(c + 1)], heap[static_cast<size_t>(c)])) ++c;
            if (!before(heap[static_cast<size_t>(c)], x)) break;
            heap[static_cast<size_t>(q)] = heap[static_cast<size_t>(c)];
            pos[static_cast<size_t>(heap[static_cast<size_t>(q)])] = q;
            q = c;
        }
        heap[static_cast<size_t>(q)] = x;
        pos[static_cast<size_t>(x)] = q;
        return top;
    }

    bool run_sparse(const SparseRows& m) {
        std::vector<int64_t> todo;
        if (!reduce_rows(m, todo)) return false;
        for (int64_t cur : todo) {
            sr.clear();
            sc.clear();
            heap.clear();
            int64_t i = cur, sink = -1;
            double min_val = 0.0;
            bool ok = true;
            while (sink < 0) {
                sr.push_back(i);
                const double ui = u[static_cast<size_t>(i)];
                for (int64_t k = m.ip[i]; k < m.ip[i + 1]; ++k) {
                    const int64_t j = m.ix[k];
                    if (in_sc[static_cast<size_t>(j)]) continue;
                    const double r = min_val + m.sign * m.w[k] - ui - v[static_cast<size_t>(j)];
                    if (!touched[static_cast<size_t>(j)]) {
                        touched[static_cast<size_t>(j)] = 1;
                        touched_list.push_back(j);
                    }
                    if (r < shortest[static_cast<size_t>(j)]) {
                        path[static_cast<size_t>(j)] = i;
                        shortest[static_cast<size_t>(j)] = r;
                        if (pos[static_cast<size_t>(j)] < 0) {
                            heap.push_back(j);
                            heap_up(static_cast<int64_t>(heap.size()) - 1);
                        } else {
                            heap_up(pos[static_cast<size_t>(j)]);
                        }
                    }
                }
                if (heap.empty()) {
                    ok = false;
                    break;
                }
                const int64_t best = heap_pop();
                min_val = shortest[static_cast<size_t>(best)];
                sc.push_back(best);
                in_sc[static_cast<size_t>(best)] = 1;
                if (row4col[static_cast<size_t>(best)] < 0) sink = best;
                else i = row4col[static_cast<size_t>(best)];
            }
            if (ok) finish(cur, sink, min_val);
            for (int64_t j : touched_list) {
                touched[static_cast<size_t>(j)] = 0;
                in_sc[static_cast<size_t>(j)] = 0;
                shortest[static_cast<size_t>(j)] = kInf;
                pos[static_cast<size_t>(j)] = -1;
            }
            touched_list.clear();
            if (!ok) return false;
        }
        return true;
    }
};

// (row_ind, col_ind, u, v) of a solved problem (transposed: the solver's rows are the columns).
nb::object result(const Solver& s, bool transposed) {
    const int64_t m = s.R;
    nb::object ri = vec1(DType::Int64, m), ci = vec1(DType::Int64, m);
    int64_t* RI = ptr<int64_t>(ri);
    int64_t* CI = ptr<int64_t>(ci);
    if (!transposed) {
        for (int64_t i = 0; i < m; ++i) {
            RI[i] = i;
            CI[i] = s.col4row[static_cast<size_t>(i)];
        }
    } else {  // the pairs (col4row[j], j) by ascending row
        std::vector<int64_t> order(static_cast<size_t>(m));
        std::iota(order.begin(), order.end(), int64_t{0});
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return s.col4row[static_cast<size_t>(a)] < s.col4row[static_cast<size_t>(b)];
        });
        for (int64_t q = 0; q < m; ++q) {
            RI[q] = s.col4row[static_cast<size_t>(order[static_cast<size_t>(q)])];
            CI[q] = order[static_cast<size_t>(q)];
        }
    }
    nb::object ou = vec1(DType::Float64, s.R), ov = vec1(DType::Float64, s.Cn);
    std::copy(s.u.begin(), s.u.end(), ptr<double>(ou));
    std::copy(s.v.begin(), s.v.end(), ptr<double>(ov));
    return nb::make_tuple(0, ri, ci, ou, ov);
}

nb::object failed(int status) { return nb::make_tuple(status, nb::none(), nb::none(), nb::none(), nb::none()); }

nb::object assign_dense(nb::handle ah, bool maximize) {
    const char* func = "_assign_dense";
    if (!is_ndarray(ah)) wrong(func, "cost");
    const NDArray& a = as_ndarray(ah);
    if (a.ndim != 2 || !a.c_contiguous || a.dtype != DType::Float64) wrong(func, "cost");
    const int64_t r = a.shape[0], c = a.shape[1];
    const double* A = reinterpret_cast<const double*>(a.data);
    const double sign = maximize ? -1.0 : 1.0;
    const bool big = r * c >= kParallelWork;
    bool bad = false;
    {
        const parallel::Unlocked unlocked(big);  // no Python object in here
        for (int64_t k = 0, e = r * c; k < e; ++k) {
            const double x = sign * A[k];
            bad |= (x != x) | (x == -kInf);
        }
    }
    if (bad) return failed(2);
    const bool transposed = r > c;
    const int64_t R = transposed ? c : r, Cn = transposed ? r : c;
    Solver s(R, Cn);
    bool ok;
    {
        const parallel::Unlocked unlocked(big);
        if (transposed) {
            std::vector<double> t(static_cast<size_t>(r * c));
            for (int64_t i = 0; i < r; ++i)
                for (int64_t j = 0; j < c; ++j) t[static_cast<size_t>(j * r + i)] = A[i * c + j];
            ok = s.run_dense(DenseRows{t.data(), r, sign});
        } else {
            ok = s.run_dense(DenseRows{A, c, sign});
        }
    }
    if (!ok) return failed(1);
    return result(s, transposed);
}

nb::object assign_sparse(nb::handle ip, nb::handle ix, nb::handle dh, int64_t rows, int64_t cols, bool maximize) {
    const Bip g = bip_arg(ip, ix, dh, rows, cols, "_assign_sparse");
    if (!g.w) wrong("_assign_sparse", "data");
    const int64_t nnz = g.ip[rows];
    const double sign = maximize ? -1.0 : 1.0;
    const bool transposed = rows > cols;
    const int64_t R = transposed ? cols : rows, Cn = transposed ? rows : cols;
    Solver s(R, Cn);
    bool ok;
    {
        const parallel::Unlocked unlocked(nnz >= kParallelWork);
        if (transposed) {  // the transpose by a counting sort (rows of the transpose ascending)
            std::vector<int64_t> tp(static_cast<size_t>(cols + 1), 0), ti(static_cast<size_t>(nnz));
            std::vector<double> tw(static_cast<size_t>(nnz));
            for (int64_t k = 0; k < nnz; ++k) ++tp[static_cast<size_t>(g.ix[k] + 1)];
            for (int64_t j = 0; j < cols; ++j) tp[static_cast<size_t>(j + 1)] += tp[static_cast<size_t>(j)];
            std::vector<int64_t> next(tp.begin(), tp.end() - 1);
            for (int64_t i = 0; i < rows; ++i) {
                for (int64_t k = g.ip[i]; k < g.ip[i + 1]; ++k) {
                    const size_t at = static_cast<size_t>(next[static_cast<size_t>(g.ix[k])]++);
                    ti[at] = i;
                    tw[at] = g.w[k];
                }
            }
            ok = s.run_sparse(SparseRows{tp.data(), ti.data(), tw.data(), sign});
        } else {
            ok = s.run_sparse(SparseRows{g.ip, g.ix, g.w, sign});
        }
    }
    if (!ok) return failed(1);
    return result(s, transposed);
}

}  // namespace

void bind_matching(nb::module_& m) {
    m.def(
        "_bipartite_matching", &bipartite_matching, nb::arg("indptr"), nb::arg("indices"), nb::arg("rows"),
        nb::arg("cols")
    );
    m.def("_assign_dense", &assign_dense, nb::arg("cost"), nb::arg("maximize"));
    m.def(
        "_assign_sparse", &assign_sparse, nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("rows"),
        nb::arg("cols"), nb::arg("maximize")
    );
}

}  // namespace sd::graph
