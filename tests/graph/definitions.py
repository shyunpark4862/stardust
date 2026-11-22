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
