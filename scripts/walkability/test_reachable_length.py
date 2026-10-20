"""Reference test: the JIT reachable-length kernel vs an independent networkx
implementation of the partial-edge rule.

Both implementations run on the same collapsed, self-loop-free undirected graph,
so the only thing under test is the kernel's edge accounting. Exits nonzero on
any mismatch.

Run from the repo root:
    .venv/bin/python scripts/walkability/test_reachable_length.py
"""

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from walkability.compute_reachable_length import dijkstra_reachable_length  # noqa: E402


def collapse(raw):
    """Collapse parallel edges to min-time and drop self-loops (mirrors build_csr)."""
    best = {}
    for a, b, t, l in raw:
        if a == b:
            continue
        key = (a, b) if a <= b else (b, a)
        if key not in best or t < best[key][0]:
            best[key] = (t, l)
    return [(a, b, t, l) for (a, b), (t, l) in best.items()]


def build_csr(n, edges):
    src = np.array([e[0] for e in edges], np.int64)
    dst = np.array([e[1] for e in edges], np.int64)
    wt = np.array([e[2] for e in edges], np.float64)
    wl = np.array([e[3] for e in edges], np.float64)
    m = len(src)
    head = np.zeros(n + 1, np.int64)
    for a, b in zip(src, dst):
        head[a + 1] += 1
        head[b + 1] += 1
    head = np.cumsum(head)
    pos = head[:-1].copy()
    to = np.empty(2 * m, np.int64)
    wtarr = np.empty(2 * m, np.float64)
    wlarr = np.empty(2 * m, np.float64)
    for a, b, t, l in zip(src, dst, wt, wl):
        to[pos[a]] = b
        wtarr[pos[a]] = t
        wlarr[pos[a]] = l
        pos[a] += 1
        to[pos[b]] = a
        wtarr[pos[b]] = t
        wlarr[pos[b]] = l
        pos[b] += 1
    return head, to, wtarr, wlarr


def reference(s, G, cutoff):
    """Independent partial-edge reference on the same graph, via networkx."""
    dist = nx.single_source_dijkstra_path_length(G, s, weight="time")
    total = 0.0
    for u, v, d in G.edges(data=True):
        w = d["time"]
        l = d["length"]
        du = dist.get(u, np.inf)
        dv = dist.get(v, np.inf)
        fu = 0.0 if du == np.inf else max(0.0, min(1.0, (cutoff - du) / w))
        fv = 0.0 if dv == np.inf else max(0.0, min(1.0, (cutoff - dv) / w))
        total += l * min(1.0, fu + fv)
    return total


def kernel_values(n, head, to, wtarr, wlarr, cutoff, sources):
    out = np.empty(len(sources))
    dt = np.zeros(n)
    stamp = np.zeros(n, np.int64)
    done = np.zeros(n, np.int64)
    ht = np.empty(to.shape[0] + 1)
    hv = np.empty(to.shape[0] + 1, np.int64)
    for i, s in enumerate(sources):
        out[i] = dijkstra_reachable_length(
            s, head, to, wtarr, wlarr, cutoff, i + 1, dt, stamp, done, ht, hv
        )
    return out


def check(n, raw, cutoff, seed):
    edges = collapse(raw)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, t, l in edges:
        G.add_edge(a, b, time=t, length=l)
    head, to, wtarr, wlarr = build_csr(n, edges)
    sources = np.arange(n, dtype=np.int64)
    got = kernel_values(n, head, to, wtarr, wlarr, cutoff, sources)
    want = np.array([reference(s, G, cutoff) for s in range(n)])
    err = float(np.max(np.abs(got - want)))
    rel = err / max(1.0, float(np.max(np.abs(want))))
    assert rel < 1e-9, f"seed {seed} cutoff {cutoff}: max abs err {err:.6f} (rel {rel:.2e})"
    return err


def main():
    cases = [
        # a single long edge: a dead-end node reaches only its partial share
        (2, [(0, 1, 400.0, 500.0)], "single edge"),
        # a chain of equal edges, so the frontier is exercised at every node
        (6, [(i, i + 1, 300.0, 375.0) for i in range(5)], "chain"),
        # a triangle with a long edge whose middle is not covered
        (3, [(0, 1, 200.0, 250.0), (1, 2, 200.0, 250.0), (0, 2, 800.0, 1000.0)], "triangle"),
        # parallel edges + a self-loop, to exercise collapse/drop
        (4, [(0, 1, 100.0, 120.0), (0, 1, 90.0, 110.0), (2, 2, 50.0, 60.0), (1, 2, 300.0, 375.0)], "collapse"),
    ]
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(3, 30))
        m = int(rng.integers(1, 4 * n))
        raw = []
        for _ in range(m):
            a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
            t = float(rng.uniform(50.0, 900.0))
            raw.append((a, b, t, t * rng.uniform(1.0, 2.0)))
        cases.append((n, raw, f"random-{_ + 1}"))

    for n, raw, name in cases:
        for cutoff in (100.0, 600.0, 1200.0):
            err = check(n, raw, cutoff, name)
            print(f"  ok {name:16s} cutoff={cutoff:6.0f}  max abs err={err:.3e}")

    print("all reference checks passed")


if __name__ == "__main__":
    main()
