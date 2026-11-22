"""The definitions of sd.graph's exact answers, written out plainly in Python (docs/design/graph.md §4, §7.2:
the structural functions are checked against these, not against reference files). Small graphs only.

A graph here is (n, edges) with edges [(i, j, w)] the stored entries i -> j (w a Python number). The
undirected reading has the edge i - j when either direction is stored, with the smaller weight of the two
when both are (NaN when either is NaN, -0.0 of 0.0 and -0.0: docs/design/graph.md §2.1).
"""

from __future__ import annotations

import heapq
import math
import sys


def undirected_weight(a: float, b: float) -> float:
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if a != b:
        return min(a, b)
    return a if math.copysign(1.0, a) < 0 else b


def weights(n: int, edges, directed: bool) -> dict:
    """{(u, v): weight} of the edges a search may take (both directions when undirected)."""
    out = {}
    for i, j, w in edges:
        w = float(w)
        pairs = ((i, j),) if directed else ((i, j), (j, i))
        for a, b in pairs:
            out[(a, b)] = w if (a, b) not in out else undirected_weight(out[(a, b)], w)
    return out


def neighbours(n: int, edges, directed: bool) -> list[list[int]]:
    """The neighbours of every node, ascending and distinct."""
    adj = [set() for _ in range(n)]
    for a, b in weights(n, edges, directed):
        adj[a].add(b)
    return [sorted(s) for s in adj]


def bfs(n, edges, s, directed):
    """(order, predecessors) of the breadth-first search from s: a queue, neighbours ascending."""
    adj = neighbours(n, edges, directed)
    pred = [-9999] * n
    seen = [False] * n
    seen[s] = True
    order = [s]
    head = 0
    while head < len(order):
        v = order[head]
        head += 1
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                pred[w] = v
                order.append(w)
    return order, pred


def dfs(n, edges, s, directed):
    """(order, predecessors) of the recursive depth-first search from s (preorder, neighbours ascending)."""
    adj = neighbours(n, edges, directed)
    pred = [-9999] * n
    seen = [False] * n
    order = []
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 4 * n + 100))

    def visit(v):
        seen[v] = True
        order.append(v)
        for w in adj[v]:
            if not seen[w]:
                pred[w] = v
                visit(w)

    try:
        visit(s)
    finally:
        sys.setrecursionlimit(limit)
    return order, pred


def tree(n, edges, s, directed, depth):
    """{(parent, child): weight} of the search tree from s: one edge per node reached, the weight of the
    edge it was found by."""
    _, pred = (dfs if depth else bfs)(n, edges, s, directed)
    w = weights(n, edges, directed)
    return {(pred[v], v): w[(pred[v], v)] for v in range(n) if pred[v] >= 0}


def reachable(n, edges, s, directed=True):
    return set(bfs(n, edges, s, directed)[0])


def labels_by_smallest_node(groups: list[int]) -> tuple[int, list[int]]:
    """Renumber any labelling so that each component is numbered by the order of its smallest node."""
    renum = {}
    out = []
    for g in groups:
        if g not in renum:
            renum[g] = len(renum)
        out.append(renum[g])
    return len(renum), out


def components(n, edges, strong):
    """(count, labels) by the definition: weak, nodes joined by an undirected path; strong, nodes that
    reach each other. Labels by the order of each component's smallest node."""
    if not strong:
        reach = [reachable(n, edges, v, False) for v in range(n)]
        return labels_by_smallest_node([min(reach[v]) for v in range(n)])
    reach = [reachable(n, edges, v) for v in range(n)]
    return labels_by_smallest_node([min(u for u in reach[v] if v in reach[u]) for v in range(n)])


def topological_sort(n, edges):
    """The lexicographically smallest topological order (the smallest node with no remaining in-edge
    first), or None when there is a cycle (a self loop too)."""
    indeg = [0] * n
    out = [[] for _ in range(n)]
    for i, j, _ in edges:
        out[i].append(j)
        indeg[j] += 1
    ready = [v for v in range(n) if indeg[v] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        v = heapq.heappop(ready)
        order.append(v)
        for w in out[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(ready, w)
    return order if len(order) == n else None


def is_topological(n, edges, order) -> bool:
    pos = {v: k for k, v in enumerate(order)}
    return sorted(order) == list(range(n)) and all(pos[i] < pos[j] for i, j, _ in edges)


# ---- shortest paths (milestone GR2; docs/design/graph.md §3.2)


def exact_weights(n, edges, directed, unweighted=False):
    """[[(v, Fraction w)]] per node, ascending, self loops left out; and the set of nodes with a negative
    self loop (a negative cycle by itself)."""
    from fractions import Fraction

    adj = [[] for _ in range(n)]
    loops = set()
    for (a, b), w in sorted(weights(n, edges, directed).items()):
        w = Fraction(1) if unweighted else Fraction(w)
        if a == b:
            if w < 0:
                loops.add(a)
            continue
        adj[a].append((b, w))
    return adj, loops


def shortest_exact(adj, loops, sources):
    """The exact shortest distances from the sources together (Fraction, None where unreachable) by a
    queue Bellman-Ford, or None when a negative cycle (a negative self loop too) is reachable."""
    from collections import deque
    from fractions import Fraction

    n = len(adj)
    d = [None] * n
    q = deque()
    for s in sources:
        if d[s] is None:
            d[s] = Fraction(0)
            q.append(s)
    inq = [d[v] is not None for v in range(n)]
    count = [0] * n
    while q:
        u = q.popleft()
        inq[u] = False
        count[u] += 1
        if count[u] > n + 1:
            return None
        for v, w in adj[u]:
            t = d[u] + w
            if d[v] is None or t < d[v]:
                d[v] = t
                if not inq[v]:
                    inq[v] = True
                    q.append(v)
    if any(d[v] is not None for v in loops):
        return None
    return d


def predecessors_exact(adj, d, sources):
    """The rule of docs/design/graph.md §3.2 on exact distances: (pred, edge count h, source of the chain)."""
    n = len(adj)
    h = [-1] * n
    order = []
    for s in sources:
        if h[s] < 0:
            h[s] = 0
            order.append(s)
    pred = [-9999] * n
    k = 0
    while k < len(order):
        u = order[k]
        k += 1
        for v, w in adj[u]:
            if d[v] is None or d[u] + w != d[v]:
                continue
            if h[v] < 0:
                h[v], pred[v] = h[u] + 1, u
                order.append(v)
            elif h[v] == h[u] + 1 and u < pred[v]:
                pred[v] = u
    src = [-9999] * n
    for v in order:
        src[v] = v if pred[v] == -9999 else src[pred[v]]
    return pred, h, src


def loopless_paths(adj, s, t):
    """Every loopless s-t path as (exact length, edge count, nodes), sorted (yen's order)."""
    from fractions import Fraction

    if s == t:
        return [(Fraction(0), 0, (s,))]
    found = []
    stack = [(s, (s,), Fraction(0))]
    while stack:
        u, path, length = stack.pop()
        for v, w in adj[u]:
            if v in path:
                continue
            if v == t:
                found.append((length + w, len(path), path + (v,)))
            else:
                stack.append((v, path + (v,), length + w))
    return sorted(found)


def spanning_forest(n, edges):
    """The minimum spanning forest of the undirected reading under the key (w, min(i, j), max(i, j)) (GR17):
    {(i, j): w} of the stored places of the chosen edges (the entry that gave the weight; the upper
    triangle when both directions have the same bits), by Kruskal on that key. Weights are compared
    exactly (Fraction), -0.0 equal to 0.0."""
    import struct
    from fractions import Fraction

    stored = {}  # (i, j) -> w, i != j
    for i, j, w in edges:
        if i != j:
            stored[(i, j)] = float(w)
    best = {}  # (a, b), a < b -> (weight, place)
    for (i, j), w in stored.items():
        a, b = min(i, j), max(i, j)
        if (a, b) in best:
            continue
        up, lo = stored.get((a, b)), stored.get((b, a))
        if up is None or lo is None:
            best[(a, b)] = (w, (i, j))
            continue
        u = undirected_weight(up, lo)
        upper = struct.pack("<d", u) == struct.pack("<d", up)  # also when both have the same bits
        best[(a, b)] = (u, (a, b) if upper else (b, a))
    order = sorted(best, key=lambda e: (Fraction(best[e][0]), e[0], e[1]))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    out = {}
    for a, b in order:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
            out[best[(a, b)][1]] = best[(a, b)][0]
    return out


def max_flow_value(n, caps, s, t):
    """The maximum flow value of {(u, v): capacity} (self loops ignored) by plain augmenting paths on Python
    integers (no overflow)."""
    from collections import deque

    res = {}
    for (u, v), c in caps.items():
        if u != v:
            res[(u, v)] = res.get((u, v), 0) + int(c)
            res.setdefault((v, u), 0)
    adj = [[] for _ in range(n)]
    for u, v in res:
        adj[u].append(v)
    total = 0
    while True:
        pred = {s: None}
        q = deque([s])
        while q and t not in pred:
            u = q.popleft()
            for v in adj[u]:
                if v not in pred and res[(u, v)] > 0:
                    pred[v] = u
                    q.append(v)
        if t not in pred:
            return total
        d, v = None, t
        while pred[v] is not None:
            d = res[(pred[v], v)] if d is None else min(d, res[(pred[v], v)])
            v = pred[v]
        v = t
        while pred[v] is not None:
            res[(pred[v], v)] -= d
            res[(v, pred[v])] += d
            v = pred[v]
        total += d


def matching_size(rows, cols, pairs):
    """The size of a largest matching of a bipartite graph ({(i, j)} the edges) by augmenting paths."""
    adj = [[] for _ in range(rows)]
    for i, j in sorted(pairs):
        adj[i].append(j)
    owner = [-1] * cols

    def augment(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] < 0 or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 4 * (rows + cols) + 100))
    try:
        return sum(augment(i, set()) for i in range(rows))
    finally:
        sys.setrecursionlimit(limit)


def assignment_exact(cost):
    """(exact optimal cost, col_ind) of the assignment of the rows of a (r, c) cost matrix, r <= c, of least
    total cost: a list of rows of Python numbers, None for a pair that cannot be chosen. The shortest
    augmenting path method in exact rationals (Fraction), so the optimum is exact for float costs too;
    None when no assignment exists. Small matrices only."""
    from fractions import Fraction

    r = len(cost)
    c = len(cost[0]) if r else 0
    C = [[None if x is None else Fraction(x) for x in row] for row in cost]
    u, v = [Fraction(0)] * r, [Fraction(0)] * c
    col4row, row4col = [-1] * r, [-1] * c
    for cur in range(r):
        shortest, path = [None] * c, [-1] * c
        done_c, rows = [False] * c, []
        i, low, sink = cur, Fraction(0), -1
        while sink < 0:
            rows.append(i)
            for j in range(c):
                if done_c[j] or C[i][j] is None:
                    continue
                d = low + C[i][j] - u[i] - v[j]
                if shortest[j] is None or d < shortest[j]:
                    shortest[j], path[j] = d, i
            cand = [j for j in range(c) if not done_c[j] and shortest[j] is not None]
            if not cand:
                return None
            j = min(cand, key=lambda j: (shortest[j], row4col[j] >= 0, j))
            low = shortest[j]
            done_c[j] = True
            if row4col[j] < 0:
                sink = j
            else:
                i = row4col[j]
        u[cur] += low
        for i in rows:
            if i != cur:
                u[i] += low - shortest[col4row[i]]
        for j in range(c):
            if done_c[j]:
                v[j] -= low - shortest[j]
        j = sink
        while True:
            i = path[j]
            row4col[j] = i
            col4row[i], j = j, col4row[i]
            if i == cur:
                break
    return sum(C[i][col4row[i]] for i in range(r)), col4row


def reverse_cuthill_mckee(n, pairs, symmetric_mode, seed_order=None):
    """The reverse Cuthill-McKee order of a pattern ({(i, j)} the stored entries; docs/design/graph.md §3.6,
    decision GR52) as scipy's _reverse_cuthill_mckee runs it: the pattern read is the pairs (symmetric_mode)
    or the pairs and their transposes; a node's degree is its row length plus 1 for a stored diagonal; the
    components are searched breadth first from the first unvisited node of seed_order(degrees) (by
    default the nodes by (degree, index); scipy's is numpy's unstable argsort of the int32 degrees), a
    node's new neighbours queued by (degree, index); the order is reversed."""
    rows = [set() for _ in range(n)]
    for i, j in pairs:
        rows[i].add(j)
        if not symmetric_mode:
            rows[j].add(i)
    degree = [len(r) + (i in r) for i, r in enumerate(rows)]
    seeds = sorted(range(n), key=lambda i: (degree[i], i)) if seed_order is None else list(seed_order(degree))
    seen, order = [False] * n, []
    for s in seeds:
        if seen[s]:
            continue
        seen[s] = True
        order.append(s)
        head = len(order) - 1
        while head < len(order):
            i = order[head]
            head += 1
            new = [j for j in sorted(rows[i]) if not seen[j]]
            for j in new:
                seen[j] = True
            order.extend(sorted(new, key=lambda j: (degree[j], j)))
    return order[::-1]


# ---- the undirected structure (milestone GR5) ----------------------------------------------------------


def simple_edges(n, pairs):
    """The edges {(i, j): i < j} of the undirected reading of a pattern without its diagonal."""
    return sorted({(min(i, j), max(i, j)) for i, j in pairs if i != j})


def _labels_without(n, edges, node=None, edge=None):
    """Connected-component labels of the undirected graph without one node (label -1) or one edge."""
    adj = [[] for _ in range(n)]
    for e in edges:
        if e != edge and node not in e:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
    label = [-1] * n
    c = 0
    for s in range(n):
        if s == node or label[s] >= 0:
            continue
        label[s] = c
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if label[v] < 0:
                    label[v] = c
                    stack.append(v)
        c += 1
    return label


def articulation_points(n, pairs):
    """The nodes whose removal splits their component: the other nodes of v's component fall into two
    components or more."""
    edges = simple_edges(n, pairs)
    whole = _labels_without(n, edges)
    out = []
    for v in range(n):
        without = _labels_without(n, edges, node=v)
        mine = {without[u] for u in range(n) if u != v and whole[u] == whole[v]}
        if len(mine) >= 2:
            out.append(v)
    return out


def bridges(n, pairs):
    """The edges whose removal disconnects their two ends, as sorted (i, j), i < j."""
    edges = simple_edges(n, pairs)
    out = []
    for e in edges:
        label = _labels_without(n, edges, edge=e)
        if label[e[0]] != label[e[1]]:
            out.append(e)
    return out


def biconnected_components(n, pairs):
    """(edges, labels): the edges sorted and each edge's block, numbered by the blocks' smallest edges. Two
    edges are in one block when they are in one component and no node x separates them: in the graph
    without x an edge lies in the component of its end other than x (of both ends when neither is x), and
    the two edges' components agree for every x."""
    edges = simple_edges(n, pairs)
    whole = _labels_without(n, edges)
    keys = [[whole[e[0]]] for e in edges]
    for x in range(n):
        label = _labels_without(n, edges, node=x)
        for k, (a, b) in enumerate(edges):
            if a == x and b == x:
                continue
            keys[k].append(label[b] if a == x else label[a])
    block, labels = {}, []
    for k in range(len(edges)):
        labels.append(block.setdefault(tuple(keys[k]), len(block)))
    return edges, labels


def core_number(n, pairs):
    """The core number of every node by the definition: the largest k such that the node survives the
    repeated removal of the nodes of degree below k."""
    edges = simple_edges(n, pairs)
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    core = [0] * n
    k = 1
    while True:
        alive = set(range(n))
        changed = True
        while changed:
            changed = False
            for v in list(alive):
                if len(adj[v] & alive) < k:
                    alive.discard(v)
                    changed = True
        if not alive:
            return core
        for v in alive:
            core[v] = k
        k += 1


# ---- betweenness on the computed shortest-path DAG (milestone GR5) ---------------------------------------


def float_distances(adj, s):
    """The least fixed point of d(v) = min fl(d(u) + w) from s (docs/design/graph.md §3.2) for positive
    weights, by Dijkstra (fl(d + w) >= d keeps a taken node final). adj: [[(v, w)]] with float weights."""
    n = len(adj)
    d = [math.inf] * n
    d[s] = 0.0
    done = [False] * n
    heap = [(0.0, s)]
    while heap:
        du, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for v, w in adj[u]:
            t = du + w
            if t < d[v]:
                d[v] = t
                heapq.heappush(heap, (t, v))
    return d


def betweenness_dag(n, edges, directed, unweighted):
    """Brandes over the shortest-path DAG of the computed distances (decision GR58): per source the float
    distances, the DAG edges u -> v with fl(d(u) + w) == d(v) and (d(u), h(u)) < (d(v), h(v)) (h the
    breadth-first edge count over the tight edges from s: it orders absorbed steps), and the exact path
    counts and dependencies (Python integers and Fractions). Returns (raw, stats): raw[v] the exact sum over
    the ordered pairs (no scaling), with endpoints the separate sums ends[v]; stats the contract's
    quantities: L (the longest DAG path in edges), dout and din (the largest DAG out- and in-degree) and
    sigma (the largest path count)."""
    from fractions import Fraction

    adj = [[] for _ in range(n)]
    for (u, v), x in sorted(weights(n, edges, directed).items()):
        if u != v:  # a self loop is never a DAG edge
            adj[u].append((v, 1.0 if unweighted else float(x)))
    raw = [Fraction(0)] * n
    ends = [0] * n
    stats = {"L": 0, "dout": 0, "din": 0, "sigma": 1}
    for s in range(n):
        d = float_distances(adj, s)
        h = {s: 0}  # the breadth-first edge count over the tight edges
        queue = [s]
        for u in queue:
            for v, x in adj[u]:
                if v not in h and d[v] != math.inf and d[u] + x == d[v]:
                    h[v] = h[u] + 1
                    queue.append(v)
        order = sorted((v for v in range(n) if d[v] != math.inf), key=lambda v: (d[v], h[v], v))
        dag = {v: [] for v in order}
        for u in order:
            for v, x in adj[u]:
                if d[v] != math.inf and d[u] + x == d[v] and (d[u], h[u]) < (d[v], h[v]):
                    dag[u].append(v)
        preds = {v: [] for v in order}
        for u in order:
            for v in dag[u]:
                preds[v].append(u)
        sigma = {s: 1}
        depth = {s: 0}
        for v in order[1:]:
            sigma[v] = sum(sigma[u] for u in preds[v])
            depth[v] = max(depth[u] for u in preds[v]) + 1
        delta = {}
        for u in reversed(order):
            delta[u] = sum((Fraction(sigma[u], sigma[x]) * (1 + delta[x]) for x in dag[u]), Fraction(0))
        for v in order[1:]:
            raw[v] += delta[v]
            ends[v] += 1
        ends[s] += len(order) - 1
        stats["L"] = max(stats["L"], max(depth.values()))
        stats["dout"] = max([stats["dout"]] + [len(x) for x in dag.values()])
        stats["din"] = max([stats["din"]] + [len(x) for x in preds.values()])
        stats["sigma"] = max(stats["sigma"], max(sigma.values()))
    return raw, ends, stats


def betweenness_scaled(n, raw, ends, directed, normalized, endpoints):
    """networkx's endpoints and _rescale over the exact sums of betweenness_dag (Fractions)."""
    from fractions import Fraction

    bc = [r + (ends[v] if endpoints else 0) for v, r in enumerate(raw)]
    N = n if endpoints else n - 1
    if N < 2:
        return bc
    if normalized:
        scale = Fraction(1, N * (N - 1))
    elif not directed:
        scale = Fraction(1, 2)
    else:
        return bc
    return [x * scale for x in bc]
