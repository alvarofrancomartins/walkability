"""Compute the reachable-network-length metric on the 10-minute walking isochrone.

For every junction, ``reachable_length(v)`` is the total walkable network length
lying inside its 10-minute isochrone — the set of nodes reachable from ``v``
within ``CUTOFF`` seconds at a single modeled walking speed. It reads as "how
much network this junction can reach on foot", and peaks where a junction sits
in a dense, well-connected, fine-grained part of the network.

One Dijkstra per node (weighted by travel time, out to ``CUTOFF``) yields the
metric. Each edge is credited a partial share equal to the fraction of its
length that lies inside the isochrone, so the measure is invariant to how
finely the network is digitized: an edge crossing the isochrone boundary
contributes only its reachable portion, and a long edge whose middle lies
outside the isochrone is not overcounted.

JIT-compiled with numba and parallelised across cores; each worker reuses its
scratch arrays across nodes, so a search costs time proportional to the nodes it
touches rather than to the whole graph. Results are cached to ``outputs/{city}/``
keyed to the cutoff and the segment cache, so re-runs are a no-op unless the
inputs change.
"""

import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(__file__))
from citynet import (  # noqa: E402
    CITY_CONFIGS,
    CUTOFF_S,
    build_graph,
    load_boundary,
    load_segments,
    parse_city,
    reach_cache_key,
)

CUTOFF = CUTOFF_S  # 10 minutes, shared with the plot's title/labels
NPROC = os.cpu_count() or 4


@njit
def dijkstra_reachable_length(src, head, to, wt, wl, cutoff, run, dt, stamp, done, ht, hv):
    """Reachable network length for one source, within ``cutoff`` seconds.

    Returns the total walkable length reachable within ``cutoff``, crediting a
    partial share of each edge. An edge (u, v) of time ``w`` and length ``wl``
    contributes ``wl * min(1, f_u + f_v)`` where ``f_x = max(0, (cutoff -
    dist(x)) / w)`` is the fraction of the edge reachable from endpoint ``x``.
    This makes the measure invariant to how finely a road is split into
    segments (a subdivision that keeps the geometry identical leaves the metric
    unchanged).

    Implemented inside a single Dijkstra: when a node ``u`` is finalized it
    credits ``min(1, rem_u/w)`` of each incident edge whose far end is not yet
    finalized, and — once the far end ``v`` is later finalized — adds
    ``min(1, (rem_u + rem_v)/w) - min(1, rem_v/w)``, so the edge ends up with
    exactly ``min(1, f_u + f_v)`` and is never double-counted. An edge whose far
    end is never reached keeps only its one-sided share.

    The scratch arrays are owned by the caller and reused across calls, so the
    search is O(touched nodes) rather than O(n): ``stamp[v] == run`` marks a
    valid ``dt[v]``, and ``done[v] == run`` marks a finalized node. Both are
    compared against ``run``, which the caller must increment for every call.
    """
    dt[src] = 0.0
    stamp[src] = run
    ht[0] = 0.0
    hv[0] = src
    size = 1
    total = 0.0
    inf = np.inf

    while size > 0:
        t = ht[0]
        u = hv[0]
        size -= 1
        if size > 0:
            ht[0] = ht[size]
            hv[0] = hv[size]
            i = 0
            while True:
                l = 2 * i + 1
                r = 2 * i + 2
                k = i
                if l < size and ht[l] < ht[k]:
                    k = l
                if r < size and ht[r] < ht[k]:
                    k = r
                if k == i:
                    break
                ht[i], ht[k] = ht[k], ht[i]
                hv[i], hv[k] = hv[k], hv[i]
                i = k

        if done[u] == run:
            continue
        done[u] = run

        rem = cutoff - t  # remaining budget at u (>= 0: we only push nt <= cutoff)

        for i in range(head[u], head[u + 1]):
            v = to[i]
            w = wt[i]
            if done[v] == run:
                # v was finalized earlier, so its side already contributed
                # min(1, rem_v/w). The correct total is min(1, f_u + f_v) ==
                # min(1, (rem_u + rem_v)/w); add the difference, which is what
                # u's side adds on top of v's share.
                remv = cutoff - dt[v]
                a = 1.0 if w <= 0.0 else min(1.0, (rem + remv) / w)
                b = 1.0 if w <= 0.0 else min(1.0, remv / w)
                total += wl[i] * (a - b)
                continue
            # v not yet finalized: credit the fraction of this edge reachable
            # from u's end. If v is finalized later, its side's extra share is
            # added then via the branch above.
            total += wl[i] * (1.0 if w <= 0.0 else min(1.0, rem / w))
            nt = t + w
            if nt > cutoff:
                continue
            # strict '<': on equal travel time the first-found path wins
            cur = dt[v] if stamp[v] == run else inf
            if nt < cur:
                dt[v] = nt
                stamp[v] = run
                j = size
                ht[j] = nt
                hv[j] = v
                size += 1
                while j > 0:
                    p = (j - 1) // 2
                    if ht[p] <= ht[j]:
                        break
                    ht[p], ht[j] = ht[j], ht[p]
                    hv[p], hv[j] = hv[j], hv[p]
                    j = p

    return total


@njit
def _component_ids(n, src, dst):
    """Union-find over the collapsed edges; returns each node's component id."""
    parent = np.arange(n)
    for k in range(len(src)):
        x = src[k]
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        ra = x
        x = dst[k]
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        rb = x
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        x = i
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        parent[i] = x
    return parent


def build_csr(nodes, edges):
    """Collapse parallel edges to min-time and build undirected CSR arrays.

    Self-loops (segments whose two endpoints snapped to the same node) are
    dropped: they connect a node to itself, so they cannot affect reachability
    or travel times, and keeping them would double-count their length in the
    reachable-network-length metric.
    """
    n = len(nodes)
    assert (nodes.index.to_numpy() == np.arange(n)).all(), "node ids must be 0..n-1"

    u = edges.index.get_level_values(0).to_numpy(np.int64)
    v = edges.index.get_level_values(1).to_numpy(np.int64)
    assert u.min() >= 0 and u.max() < n, "edge from-node out of range"
    assert v.min() >= 0 and v.max() < n, "edge to-node out of range"

    wt = edges["travel_time_s"].to_numpy(np.float64)
    wl = edges["length_m"].to_numpy(np.float64)

    best = {}
    for a, b, t, l in zip(u, v, wt, wl):
        if a == b:
            continue
        key = (a, b) if a <= b else (b, a)
        if key not in best or t < best[key][0]:
            best[key] = (t, l)

    src = np.fromiter((k[0] for k in best), np.int64)
    dst = np.fromiter((k[1] for k in best), np.int64)
    wtime = np.fromiter((x[0] for x in best.values()), np.float64)
    wlen = np.fromiter((x[1] for x in best.values()), np.float64)

    e = len(src)
    to = np.empty(2 * e, np.int64)
    wtarr = np.empty(2 * e, np.float64)
    wlarr = np.empty(2 * e, np.float64)

    deg = np.zeros(n, np.int64)
    for a, b in zip(src, dst):
        deg[a] += 1
        deg[b] += 1
    head = np.zeros(n + 1, np.int64)
    head[1:] = np.cumsum(deg)
    pos = head[:-1].copy()

    for a, b, t, l in zip(src, dst, wtime, wlen):
        to[pos[a]] = b
        wtarr[pos[a]] = t
        wlarr[pos[a]] = l
        pos[a] += 1
        to[pos[b]] = a
        wtarr[pos[b]] = t
        wlarr[pos[b]] = l
        pos[b] += 1

    comp = _component_ids(n, src, dst)
    return head, to, wtarr, wlarr, comp


def _print_stats(L, label):
    d = L[~np.isnan(L)]
    if len(d) == 0:
        print(f"  {label}: no nodes")
        return
    lo, hi = np.percentile(d, 5), np.percentile(d, 95)
    print(
        f"  {label}: min {d.min()/1000:.2f}  mean {d.mean()/1000:.2f}  "
        f"max {d.max()/1000:.2f} km  (90% {lo/1000:.1f}-{hi/1000:.1f} km, n={len(d):,})"
    )


def main():
    city = parse_city()
    cache = f"outputs/{city}/reachable_length.npy"
    cache_meta = f"outputs/{city}/reachable_length.npy.meta"
    display = CITY_CONFIGS[city]["display"]

    seg = load_segments(city)
    nodes, edges = build_graph(seg)
    head, to, wt, wl, comp = build_csr(nodes, edges)
    n = len(nodes)

    counts = np.bincount(comp)
    largest = int(counts.argmax())
    main_mask = comp == largest
    n_main = int(main_mask.sum())
    n_dropped = n - n_main
    n_components = int((counts > 0).sum())
    print(
        f"Largest connected component: {n_main:,}/{n:,} nodes "
        f"({n_dropped:,} dropped across {n_components - 1:,} other components)"
    )

    key = reach_cache_key(city)
    cached = os.path.exists(cache) and os.path.exists(cache_meta)
    if cached:
        with open(cache_meta) as f:
            cached = f.read().strip() == key

    if cached:
        L = np.load(cache)
        print(f"Loaded cached metric from {cache}")
    else:
        print(
            f"Computing reachable network length for {n_main:,} nodes "
            f"on {NPROC} cores ..."
        )
        L = np.full(n, np.nan)
        sources = np.flatnonzero(main_mask)
        L[main_mask] = _compute_parallel(head, to, wt, wl, sources)
        os.makedirs(f"outputs/{city}", exist_ok=True)
        np.save(cache, L)
        with open(cache_meta, "w") as f:
            f.write(key)
        print(f"Saved metric to {cache}")

    inside = nodes.geometry.within(load_boundary(city).geometry.iloc[0]).to_numpy()
    print("Reachable network length (km):")
    for label, mask in (("buffered", slice(None)), (display, inside)):
        _print_stats(L[mask], label)


_HEAD = _TO = _WT = _WL = None


def _new_buffers(n, nedges):
    """Fresh scratch arrays for one worker (reused across that worker's nodes)."""
    return (
        np.zeros(n),  # dt: shortest travel time, valid iff stamp[v] == run
        np.zeros(n, np.int64),  # stamp: which run wrote dt[v]
        np.zeros(n, np.int64),  # done: which run finalized v
        np.empty(nedges + 1),  # ht: binary heap, times
        np.empty(nedges + 1, np.int64),  # hv: binary heap, nodes
    )


def _compute_parallel(head, to, wt, wl, sources):
    global _HEAD, _TO, _WT, _WL
    _HEAD, _TO, _WT, _WL = head, to, wt, wl

    n = head.shape[0] - 1
    nedges = to.shape[0]
    _ = dijkstra_reachable_length(  # JIT warm-up
        int(sources[0]), head, to, wt, wl, CUTOFF, 1, *_new_buffers(n, nedges)
    )

    chunk_size = 256
    chunks = [sources[i : i + chunk_size] for i in range(0, len(sources), chunk_size)]

    import multiprocessing as mp

    ctx = mp.get_context("fork")
    L = np.empty(len(sources))
    pos = 0
    t0 = time.time()

    with ctx.Pool(NPROC) as pool:
        for c, out in zip(chunks, pool.imap(_worker, chunks)):
            L[pos : pos + len(c)] = out
            pos += len(c)
            if pos % 5120 == 0 or pos == len(sources):
                rate = pos / (time.time() - t0)
                print(
                    f"  {pos:,}/{len(sources):,} nodes  ({pos/len(sources)*100:.0f}%)  "
                    f"{rate:.0f} nodes/s  ETA {(len(sources)-pos)/rate/60:.1f} min",
                    flush=True,
                )
    return L


def _worker(chunk):
    bufs = _new_buffers(_HEAD.shape[0] - 1, _TO.shape[0])
    out = np.empty(len(chunk))
    for i, s in enumerate(chunk):
        out[i] = dijkstra_reachable_length(
            s, _HEAD, _TO, _WT, _WL, CUTOFF, i + 1, *bufs
        )
    return out


if __name__ == "__main__":
    main()
