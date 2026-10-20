"""Independent verification of the Paraná gas-station distance map.

Re-derives the pipeline's claims from *outside* the pipeline and checks them
against external truth (published road distances, OSM/Nominatim, projection
theory, published state dimensions):

  T1  artifact consistency        npy / graph / cache keys / README statistics
  T2  edge geometry vs weight     ``length_m`` really equals the edge geometry
  T3  UTM vs WGS84 (haversine)    projection distortion within its theoretical range
  T4  station snapping            cKDTree vs geopandas ``sjoin_nearest``
  T5  cross-implementation        numba Dijkstra vs networkx on 32 spatial subgraphs
  T6  furthest junction           where is it, which station serves it, networkx re-check
  T7  border seeding              nearest station across state lines + PR-only recompute
  T8  unreachable junctions       NaN nodes are tiny isolated fragments
  T9  city-pair distances         shortest paths vs published road distances
  T10 rendered PNG                pixel colors match the plotted values (affine fit)
  T11 registry coordinates        vs OSM/Nominatim spot-checks
  T12 map extents                 vs published state dimensions and area

Run:
    .venv/bin/python scripts/stations/verify_map.py
"""

import json
import os
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely import get_coordinates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.network import build_graph  # noqa: E402
from stations.parana import (  # noqa: E402
    CRS,
    METRIC_NPY,
    UF,
    build_csr,
    load_boundary,
    load_road_segments,
    load_stations,
    multi_source_dijkstra,
    station_cache_key,
)
from stations.shared import SNAP_M_STATIONS, snap_stations  # noqa: E402
import stations.plot_parana_distance as plotmod  # noqa: E402

BG = np.array([11, 15, 20])  # plot's #0b0f14 background, 0-255
GREY_RGB = np.array([58, 63, 71])  # plot's #3a3f47 unreachable-road color
R_EARTH = 6371008.8  # mean Earth radius, metres

RESULTS = []


def check(name, ok, detail=""):
    """Record one check. ``ok`` True=PASS, False=FAIL, None=WARN (informational)."""
    if ok is None:
        status = "WARN"
    elif bool(ok):
        status = "PASS"
    else:
        status = "FAIL"
    RESULTS.append((status, name, detail))
    print(f"[{status}] {name}: {detail}", flush=True)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance (metres) between WGS84 points, vectorized."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def nominatim(url):
    """One OSM Nominatim query; None on failure (network / rate limit)."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "walkability-map-audit/1.0 (dataviz verification)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return f"ERROR: {exc}"


def main():
    print("=" * 72)
    print("Independent verification of the Paraná gas-station distance map")
    print("=" * 72, flush=True)
    t0 = time.time()

    # ---------------------------------------------------------------- load data
    seg = load_road_segments()
    nodes, edges = build_graph(seg)
    n = len(nodes)
    dist = np.load(METRIC_NPY)  # metres per node
    boundary = load_boundary()
    poly = boundary.geometry.iloc[0]
    print(f"loaded: {n:,} nodes, {len(edges):,} edges, "
          f"metric {METRIC_NPY} ({time.time() - t0:.0f}s)", flush=True)

    stations = load_stations()
    xmin, ymin, xmax, ymax = seg.total_bounds
    in_bbox = stations.cx[xmin:xmax, ymin:ymax]
    sources, snap_dists, valid_idx, n_dropped = snap_stations(in_bbox, nodes)
    seeded = np.unique(sources)
    pos_of = {idx: p for p, idx in enumerate(in_bbox.index)}
    kept_pos = np.array([pos_of[i] for i in valid_idx])
    kept_mask = np.zeros(len(in_bbox), bool)
    kept_mask[kept_pos] = True
    seed_rows = in_bbox.iloc[kept_pos]
    seed_map = {}  # node id -> positions in seed_rows
    for pos, s in enumerate(sources):
        seed_map.setdefault(int(s), []).append(pos)
    print(f"seeding recomputed: {len(in_bbox):,} bbox stations -> "
          f"{len(seeded):,} seeded nodes, {n_dropped} dropped", flush=True)

    XY = get_coordinates(nodes.geometry.values)  # (n, 2) UTM metres
    u = edges.index.get_level_values(0).to_numpy()
    v = edges.index.get_level_values(1).to_numpy()
    wl = edges["length_m"].to_numpy()
    inside = nodes.geometry.within(poly).to_numpy()
    fin = np.isfinite(dist)
    in_fin = inside & fin
    nan_in = inside & ~fin
    d = dist[in_fin] / 1000.0  # km
    rng = np.random.default_rng(20261019)

    # ------------------------------------------------------------- T1 artifacts
    check("T1 npy length == node count", len(dist) == n,
          f"{len(dist):,} vs {n:,}")
    check("T1 node ids are 0..n-1",
          (nodes.index.to_numpy() == np.arange(n)).all(), "")
    check("T1 metric dtype float64", dist.dtype == np.float64, str(dist.dtype))
    check("T1 edge endpoints in range", u.min() >= 0 and v.max() < n,
          f"u in [{u.min()},{u.max()}], v in [{v.min()},{v.max()}]")
    check("T1 seeded nodes have distance 0", dist[seeded].max() == 0.0,
          f"max over {len(seeded):,} seeds = {dist[seeded].max():g} m")
    with open(METRIC_NPY + ".meta") as f:
        cached_key = f.read().strip()
    check("T1 metric cache key fresh", cached_key == station_cache_key(SNAP_M_STATIONS),
          f"meta={cached_key} current={station_cache_key(SNAP_M_STATIONS)}")
    nan_frac = nan_in.sum() / inside.sum()
    nan_frac_all = int((~fin).sum()) / n
    check("T1 unreachable fraction (in-state) small",
          0.0005 <= nan_frac <= 0.005,
          f"{100 * nan_frac:.3f}% of {inside.sum():,} in-state nodes")
    check("T1 unreachable fraction (all nodes) ~0.3%",
          0.001 <= nan_frac_all <= 0.01,
          f"{100 * nan_frac_all:.3f}% of all {n:,} nodes")
    med, mx = float(np.median(d)), float(d.max())
    check("T1 README stats: median 2.29 km", abs(med - 2.29) <= 0.01,
          f"median={med:.3f} km")
    check("T1 README stats: max 87.25 km", abs(mx - 87.25) <= 0.01,
          f"max={mx:.3f} km")
    for km, ref in ((0.5, 13.2), (1.0, 29.5), (5.0, 62.0)):
        pct = 100 * (d <= km).mean()
        check(f"T1 README share within {km:g} km ≈ {ref}%",
              abs(pct - ref) <= 0.1, f"{pct:.1f}%")
    check("T1 seeding counts 3,094 / 2,880 / 178",
          len(in_bbox) == 3094 and len(seeded) == 2880 and n_dropped == 178,
          f"{len(in_bbox):,} bbox stations, {len(seeded):,} nodes, {n_dropped} dropped")
    used = pd.read_csv("outputs/pr/stations_used.csv")
    check("T1 stations_used.csv matches the seeded rows",
          len(used) == len(in_bbox)
          and np.allclose(used["Latitude"].to_numpy(), in_bbox["Latitude"].to_numpy(), atol=1e-9)
          and np.allclose(used["Longitude"].to_numpy(), in_bbox["Longitude"].to_numpy(), atol=1e-9),
          f"{len(used):,} rows")

    # ------------------------------------------------- T2 geometry vs edge weight
    rel = np.abs(seg.geometry.length.to_numpy() - seg["length_m"].to_numpy()) \
        / np.maximum(seg["length_m"].to_numpy(), 1e-9)
    check("T2 segment length_m == geometry.length", rel.max() < 1e-9,
          f"max rel diff {rel.max():.2e} over {len(seg):,} segments")
    rele = np.abs(edges.geometry.length.to_numpy() - wl) / np.maximum(wl, 1e-9)
    check("T2 edge length_m == edge geometry.length", rele.max() < 1e-9,
          f"max rel diff {rele.max():.2e} over {len(edges):,} edges")

    # ----------------------------------------------- T3 UTM vs haversine fidelity
    # The projection is faithful iff the great-circle length along each edge's
    # own vertices matches its planar (UTM) length within the zone's scale-factor
    # range (0.9996 at the central meridian, ~1.0016 at the zone edges). Comparing
    # only the two endpoints would measure road sinuosity instead, so walk every
    # vertex of the polyline.
    tr = pyproj.Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    samp_e = rng.choice(len(edges), size=600, replace=False)
    ee = edges.iloc[samp_e]
    ratios = []
    for g, L in zip(ee.geometry, ee["length_m"].to_numpy()):
        parts = [g] if g.geom_type == "LineString" else list(g.geoms)
        total = 0.0
        for part in parts:
            c = get_coordinates(part)
            lon, lat = tr.transform(c[:, 0], c[:, 1])
            total += haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum()
        ratios.append(total / L)
    ratios = np.asarray(ratios)
    # Expected band: UTM 22S scale factor 0.9996 (central meridian) to ~1.0016
    # (zone edges), times the spherical-haversine vs WGS84-ellipsoid radius
    # difference (±0.4%) -- together ~0.994..1.006. Anything far outside would
    # mean a wrong CRS or unit mix-up.
    check("T3 UTM polyline length vs WGS84 great-circle",
          ratios.min() > 0.994 and ratios.max() < 1.006,
          f"600 edges: min {ratios.min():.5f}  max {ratios.max():.5f}  "
          f"mean {ratios.mean():.5f} (scale-factor + spherical-radius band)")

    # --------------------------------------------- T4 snapping vs plain cKDTree
    tree_nodes = cKDTree(XY)
    d_tree, j_tree = tree_nodes.query(get_coordinates(in_bbox.geometry.values))
    dd = np.abs(d_tree[kept_pos] - snap_dists)
    n_mismatch = int((j_tree[kept_pos] != sources).sum())
    check("T4 snap distances: cKDTree == sjoin_nearest",
          dd.max() < 0.01 and n_mismatch == 0,
          f"max |Δ| = {dd.max():.4f} m, {n_mismatch} node-assignment mismatches (ties ok)")
    check("T4 dropped stations truly >100 m from any node",
          d_tree[~kept_mask].min() > SNAP_M_STATIONS,
          f"min nearest-node distance {d_tree[~kept_mask].min():.1f} m for "
          f"{int((~kept_mask).sum())} dropped stations")

    # ---------------------------------------------- T5 numba vs networkx checks
    head, to, wt = build_csr(nodes, edges)
    print("CSR built; starting networkx cross-checks", flush=True)

    def nx_ball(s0, R, cutoff):
        """networkx graph of the ball around node ``s0`` (R, cutoff in metres).

        Parallel edges are collapsed to the minimum length per undirected node
        pair, exactly as ``build_csr`` does, so both implementations run on the
        same graph.
        """
        ball = tree_nodes.query_ball_point(XY[s0], R)
        m = np.zeros(n, bool)
        m[ball] = True
        em = m[u] & m[v]
        su, sv, sw = u[em], v[em], wl[em]
        df = pd.DataFrame({"lo": np.minimum(su, sv), "hi": np.maximum(su, sv), "l": sw})
        best = df[df["lo"] != df["hi"]].groupby(["lo", "hi"], sort=False)["l"].min()
        G = nx.Graph()
        G.add_weighted_edges_from(
            zip(best.index.get_level_values(0).tolist(),
                best.index.get_level_values(1).tolist(),
                best.to_numpy().tolist()),
            weight="weight",
        )
        seeds_ball = seeded[m[seeded]]
        lengths, paths = nx.multi_source_dijkstra(
            G, {int(s): 0.0 for s in seeds_ball}, cutoff=cutoff, weight="weight"
        )
        return G, seeds_ball, lengths, paths

    # Sample junctions across the whole distance distribution. Any node on the
    # optimal path is within `metric` of the junction (arc >= Euclidean), so a
    # ball of radius metric + 3 km always contains the full path and the nearest
    # seeded node -- the subgraph result must equal the full-graph result exactly.
    idx_all = np.flatnonzero(in_fin)
    sorted_idx = idx_all[np.argsort(dist[idx_all])]
    pcts = [0.5, 1, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
            70, 75, 80, 84, 88, 91, 94, 96, 97, 98, 98.7, 99.3, 99.7, 99.9]
    samples = sorted_idx[[int(round(p / 100 * (len(sorted_idx) - 1))) for p in pcts]]
    t5_ok = True
    t5_bad = []
    n_probe = 0
    for s0 in samples:
        d0 = float(dist[s0])
        _, seeds_ball, lengths, _ = nx_ball(s0, d0 + 3000.0, d0 + 2000.0)
        d_nx = lengths.get(int(s0), np.inf)
        if len(seeds_ball) == 0 or abs(d_nx - d0) > 1e-6:
            t5_ok = False
            t5_bad.append((int(s0), d0, d_nx, len(seeds_ball)))
        # Probe extra ball nodes: networkx sees only a subset of the seeds, so it
        # can tie or lose, but it must never find a *shorter* route than the
        # shipped metric.
        if s0 in samples[::5]:
            ball = tree_nodes.query_ball_point(XY[s0], d0 + 3000.0)
            for b in rng.choice(ball, size=min(15, len(ball)), replace=False):
                n_probe += 1
                d_nx_b = lengths.get(int(b), np.inf)
                if np.isfinite(dist[b]) and d_nx_b < dist[b] - 1e-6:
                    t5_ok = False
                    t5_bad.append(("beats", int(b), float(dist[b]), float(d_nx_b)))
    check("T5 numba Dijkstra == networkx (31 subgraphs)",
          t5_ok,
          f"{len(samples)} sampled junctions across the distance range, "
          f"{n_probe} extra probe nodes; mismatches: {t5_bad[:3] if t5_bad else 'none'}")

    # ----------------------------------------------------- T6 furthest junction
    n_max = int(sorted_idx[-1])
    d_max = float(dist[n_max])
    lon_max, lat_max = tr.transform(XY[n_max, 0], XY[n_max, 1])
    rev = nominatim(
        f"https://nominatim.openstreetmap.org/reverse?lat={lat_max:.6f}"
        f"&lon={lon_max:.6f}&format=jsonv2&zoom=10"
    )
    place = rev["display_name"] if isinstance(rev, dict) else rev
    check("T6 furthest junction located", True,
          f"node {n_max} at ({lat_max:.5f}, {lon_max:.5f}), d={d_max / 1000:.2f} km "
          f"({place})")
    time.sleep(1.1)
    # Independent re-check of the headline number, and the station that serves it.
    _, seeds_ball, lengths, paths = nx_ball(n_max, d_max + 3000.0, d_max + 2000.0)
    d_nx = lengths.get(int(n_max), np.inf)
    check("T6 max distance networkx == numba", abs(d_nx - d_max) <= 1e-6,
          f"nx {d_nx / 1000:.3f} km vs shipped {d_max / 1000:.3f} km")
    ds = multi_source_dijkstra(head, to, wt, np.array([n_max], np.int64))
    k = int(seeded[np.argmin(ds[seeded])])
    rows = seed_rows.iloc[seed_map[k]]
    name = rows["Razão Social"].iloc[0]
    mun = rows["MUNICÍPIO"].iloc[0]
    uf_st = rows["UF"].iloc[0]
    sl = float(np.hypot(*(XY[k] - XY[n_max]))) / 1000.0
    check("T6 nearest station to the furthest junction",
          abs(ds[k] - d_max) <= 1e-6 and np.isfinite(ds[k]),
          f"'{name}' ({uf_st}/{mun}) — straight line {sl:.1f} km, "
          f"network {ds[k] / 1000:.2f} km (ratio {d_max / 1000 / sl:.2f}x)")

    # ----------------------------------------------- T7 border seeding + PR-only
    # Paraná-only seeding must *not* improve on the all-state run anywhere, and
    # the in-state maximum must grow back to the README's ~102 km.
    in_pr = in_bbox[in_bbox["UF"] == UF]
    sources_pr, _, _, nd_pr = snap_stations(in_pr, nodes)
    dist_pr = multi_source_dijkstra(head, to, wt, sources_pr)
    mx_pr = dist_pr[in_fin].max() / 1000.0
    check("T7 PR-only max ≈ 102 km (README)", 95.0 <= mx_pr <= 110.0,
          f"PR-only max {mx_pr:.2f} km vs all-UF {mx:.2f} km "
          f"({nd_pr} PR stations dropped at snap)")
    improved = int((dist_pr[in_fin] > dist[in_fin] + 1.0).sum())
    check("T7 PR-only run never shorter than all-UF",
          bool(np.all(dist_pr[fin] >= dist[fin] - 1e-9)),
          f"{improved:,} in-state junctions improve from cross-border seeds")
    improved_mask = in_fin & (dist_pr > dist + 1.0)

    # Boundary-vertex KDTree (reused by T10): a junction is "on the border" when
    # it lies within 6 km of an outline vertex (segments are ~70 m long, so the
    # vertex-to-line error is negligible).
    bverts = get_coordinates(poly.boundary)
    btree = cKDTree(bverts)
    d_bvert, _ = btree.query(XY, k=1)
    border_nodes = np.flatnonzero(in_fin & (d_bvert <= 6000.0))
    print(f"{len(border_nodes):,} in-state junctions within 6 km of the border; "
          f"checking {min(40, len(border_nodes))}", flush=True)
    sample_b = rng.choice(border_nodes, size=min(40, len(border_nodes)), replace=False)
    t7_ok = True
    uf_tally = {}
    for s0 in sample_b:
        d0 = float(dist[s0])
        _, seeds_ball, lengths, paths = nx_ball(s0, d0 + 3000.0, d0 + 2000.0)
        d_nx = lengths.get(int(s0), np.inf)
        if abs(d_nx - d0) > 1e-6:
            t7_ok = False
            continue
        seed_node = int(paths[int(s0)][0])
        uf_near = seed_rows["UF"].iloc[seed_map[seed_node][0]]
        uf_tally[uf_near] = uf_tally.get(uf_near, 0) + 1
    check("T7 border junctions: nx == shipped metric", t7_ok,
          f"{len(sample_b)} border junctions; nearest-station UF: "
          f"{dict(sorted(uf_tally.items()))}")
    # Junctions that the neighbouring-state stations actually serve: their
    # metric must strictly improve over the PR-only run, and the serving
    # station must be outside Paraná.
    cross = np.flatnonzero(improved_mask & (d_bvert <= 6000.0))
    check("T7 cross-state seeds serve border junctions", len(cross) > 0,
          f"{len(cross):,} in-state border junctions served from another state")
    sample_c = rng.choice(cross, size=min(30, len(cross)), replace=False)
    ok_cross = len(sample_c) > 0
    uf_cross = {}
    for s0 in sample_c:
        d0 = float(dist[s0])
        _, _, lengths, paths = nx_ball(s0, d0 + 3000.0, d0 + 2000.0)
        if abs(lengths.get(int(s0), np.inf) - d0) > 1e-6:
            ok_cross = False
            continue
        seed_node = int(paths[int(s0)][0])
        uf_near = seed_rows["UF"].iloc[seed_map[seed_node][0]]
        uf_cross[uf_near] = uf_cross.get(uf_near, 0) + 1
        if uf_near == UF:
            ok_cross = False
    check("T7 improved border junction's station is out-of-state",
          ok_cross and all(k != UF for k in uf_cross),
          f"{len(sample_c)} sampled: nearest-station UF {dict(sorted(uf_cross.items()))}")

    # --------------------------------------------------- T8 unreachable pieces
    mnan = nan_in
    ids_nan = np.flatnonzero(mnan)
    remap = np.full(n, -1)
    remap[ids_nan] = np.arange(len(ids_nan))
    em = mnan[u] & mnan[v]
    A = csr_matrix(
        (np.ones(int(em.sum())), (remap[u[em]], remap[v[em]])),
        shape=(len(ids_nan), len(ids_nan)),
    )
    ncomp, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    check("T8 unreachable nodes are tiny isolated fragments",
          sizes.max() <= 100 and nan_frac <= 0.005,
          f"{ncomp:,} components; largest {sizes.max()}; "
          f"sizes {np.sort(sizes)[::-1][:6].tolist()}")

    # --------------------------------------- T9 city pairs vs published distances
    CITIES = {
        "Curitiba": (-25.4297353, -49.2719544),  # Praça Tiradentes (Nominatim)
        "Paranaguá": (-25.5148822, -48.5226695),
        "Foz do Iguaçu": (-25.5304023, -54.5830692),
        "Maringá": (-23.4252690, -51.9382078),
        "Londrina": (-23.3112878, -51.1595023),
    }
    PAIRS = [  # published road distances (km), central value of the sources
        ("Curitiba", "Paranaguá", 90.0),
        ("Curitiba", "Foz do Iguaçu", 640.0),
        ("Curitiba", "Londrina", 387.0),
        ("Curitiba", "Maringá", 425.0),
        ("Maringá", "Londrina", 98.0),
    ]
    city_nodes = {}
    for name, (lat, lon) in CITIES.items():
        xc, yc = tr.transform(lon, lat, direction="INVERSE")  # WGS84 -> UTM 22S
        city_nodes[name] = int(tree_nodes.query(np.array([[xc, yc]]))[1][0])
    for a, b, ref in PAIRS:
        d_single = multi_source_dijkstra(head, to, wt, np.array([city_nodes[a]], np.int64))
        d_km = d_single[city_nodes[b]] / 1000.0
        lat_a, lon_a = CITIES[a]
        lat_b, lon_b = CITIES[b]
        sl = float(haversine_m(np.array([lat_a]), np.array([lon_a]),
                               np.array([lat_b]), np.array([lon_b]))[0]) / 1000.0
        relp = d_km / ref
        check(f"T9 {a} → {b} ≈ {ref:.0f} km",
              0.92 <= relp <= 1.08,
              f"network {d_km:.0f} km ({relp:.2f}x published), "
              f"straight line {sl:.0f} km")

    # -------------------------------------------------- T10 rendered-PNG pixels
    png = "outputs/pr/station_distance.png"
    img = plt.imread(png)
    rgb = (img[:, :, :3] * 255).astype(int)
    H, W = rgb.shape[:2]
    png_fresh = os.path.getmtime(png) >= os.path.getmtime(METRIC_NPY)
    check("T10 PNG at least as new as the metric", png_fresh,
          f"{W}x{H} px" + ("" if png_fresh else "  (png older than npy!)"))

    # Bright neutral pixels = the white border (0.8 white over dark bg) and the
    # white text labels (title, subtitle, "near/far" colorbar labels). The text
    # sits in small disconnected clusters while the border is one long closed
    # curve, so keep only the largest connected component: the text can then no
    # longer pollute the fit. Fit the axis-aligned affine data->pixel transform
    # by iterative nearest-boundary-vertex least squares on that border mask.
    mask = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 200) & (rgb[:, :, 2] > 200)
    lab, n_lab = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())
    sizes_comp = sizes[1:]  # exclude the background (label 0)
    next_big = int(np.partition(sizes_comp, -2)[-2]) if len(sizes_comp) > 1 else 0
    keep_lab = int(sizes_comp.argmax()) + 1
    mask = lab == keep_lab
    ys, xs = np.nonzero(mask)
    pix = np.column_stack([xs, ys])
    imin, imax = pix[:, 0].argmin(), pix[:, 0].argmax()
    v_left = bverts[bverts[:, 0].argmin()]
    v_right = bverts[bverts[:, 0].argmax()]
    sx = (pix[imax, 0] - pix[imin, 0]) / (v_right[0] - v_left[0])
    if abs(v_right[1] - v_left[1]) < 10000.0:
        sy = sx
    else:
        sy = (pix[imin, 1] - pix[imax, 1]) / (v_right[1] - v_left[1])
    cx = pix[imin, 0] - sx * v_left[0]
    cy = pix[imin, 1] + sy * v_left[1]
    for _ in range(20):
        sub = rng.choice(len(pix), size=min(8000, len(pix)), replace=False)
        Xd = (pix[sub, 0] - cx) / sx
        Yd = (cy - pix[sub, 1]) / sy
        dsub, jsub = btree.query(np.column_stack([Xd, Yd]))
        # Reject residual stragglers beyond 2 km of a boundary vertex (the true
        # fit is sub-pixel, so a fixed threshold cannot harm it).
        ok = dsub <= 2000.0
        if ok.sum() < 200:
            break
        Xv = bverts[jsub][ok]
        (sx_new, cx_new), *_ = np.linalg.lstsq(
            np.column_stack([Xv[:, 0], np.ones(len(Xv))]), pix[sub][ok, 0], rcond=None
        )
        (sy_new, cy_new), *_ = np.linalg.lstsq(
            np.column_stack([-Xv[:, 1], np.ones(len(Xv))]), pix[sub][ok, 1], rcond=None
        )
        if sx_new > 0 and sy_new > 0:
            sx, sy, cx, cy = sx_new, sy_new, cx_new, cy_new
    dres, _ = btree.query(np.column_stack([(pix[:, 0] - cx) / sx, (cy - pix[:, 1]) / sy]))
    med_px = float(np.median(dres) * sx)
    check("T10 affine border fit converged", 0 < sx < 1e6 and 0 < sy < 1e6 and med_px < 1.5,
          f"median residual {med_px:.2f} px over {len(pix):,} border pixels "
          f"(largest of {n_lab} bright components, next {next_big:,}; "
          f"sx={sx:.4f} px/m, sy={sy:.4f} px/m)")

    def to_pixel(xy_utm):
        return sx * xy_utm[..., 0] + cx, cy - sy * xy_utm[..., 1]

    def window_rgb(px, py, half=3):
        """RGB values in a (2*half+1)² window around a pixel; None if off-image."""
        lo_x, hi_x = int(px) - half, int(px) + half
        lo_y, hi_y = int(py) - half, int(py) + half
        lo_x, hi_x = max(lo_x, 0), min(hi_x, W - 1)
        lo_y, hi_y = max(lo_y, 0), min(hi_y, H - 1)
        if hi_x < lo_x or hi_y < lo_y:
            return None
        return rgb[lo_y:hi_y + 1, lo_x:hi_x + 1].reshape(-1, 3)

    # The border line itself: 0.8 white over the dark background.
    b_ok = True
    b_bad = []
    for vert in (v_left, v_right, bverts[bverts[:, 1].argmax()], bverts[bverts[:, 1].argmin()]):
        win = window_rgb(*to_pixel(vert))
        if win is None:
            b_ok = False
            b_bad.append(-1)
            continue
        exp = 0.8 * 255.0 + 0.2 * BG
        best = np.min(np.max(np.abs(win - exp), axis=1))
        if best > 10:
            b_ok = False
            b_bad.append(best)
    check("T10 drawn border pixels are 0.8-white over bg", b_ok,
          f"4 extreme vertices, worst Δ = {max(b_bad, default=0):.0f}")

    # Rebuild the exact drawn pieces and their plotted values.
    uw = edges.index.get_level_values(0).to_numpy()
    vw = edges.index.get_level_values(1).to_numpy()
    one_in = inside[uw] ^ inside[vw]
    keep_whole = edges[~one_in & inside[uw]]
    uw, vw = keep_whole.index.get_level_values(0).to_numpy(), \
        keep_whole.index.get_level_values(1).to_numpy()
    values_km = dist / 1000.0
    P = get_coordinates(nodes.geometry.values)
    g0 = np.array([get_coordinates(g)[0] for g in keep_whole.geometry])
    start_is_u = np.hypot(*(g0 - P[uw]).T) <= np.hypot(*(g0 - P[vw]).T)
    du = np.where(start_is_u, values_km[uw], values_km[vw])
    dv = np.where(start_is_u, values_km[vw], values_km[uw])
    segs, seg_vals = plotmod._explode_subdiv(keep_whole.geometry, du, dv)
    lo = float(np.nanpercentile(values_km[inside], plotmod.PCT_LO))
    hi = float(np.nanpercentile(values_km[inside], plotmod.PCT_HI))
    norm = Normalize(vmin=lo, vmax=hi, clip=False)
    cmap = plt.get_cmap(plotmod.CMAP)
    check("T10 color scale matches plot (2nd-98th pct)",
          abs(lo - 0.12) <= 0.05 and abs(hi - 31.12) <= 0.2,
          f"{lo:.3f} -> {hi:.3f} km")
    mids = np.array([p.mean(axis=0) for p in segs])
    mtree = cKDTree(mids)
    print(f"rebuilt {len(segs):,} drawn pieces; sampling pixels", flush=True)

    def piece_windows(i):
        """RGB of the piece's own pixels: small windows around each polyline
        vertex. A 1 km piece is only a few pixels long, so its vertices cover
        the drawn line better than a single window at its midpoint."""
        vx, vy = to_pixel(segs[i])
        wins = [window_rgb(vx[k], vy[k], half=2) for k in range(len(vx))]
        wins = [w for w in wins if w is not None]
        return np.concatenate(wins) if wins else None

    for target in (0.05, 0.3, 1.0, 3.0, 8.0, 15.0, 25.0, 31.0, 40.0, 60.0, 87.0):
        cand = np.flatnonzero(np.abs(seg_vals - target) <= 0.4)
        if len(cand) == 0:
            check(f"T10 pixel color d≈{target} km", None, "no piece near the target")
            continue
        # A piece can be overdrawn by a crossing piece of another value, so try
        # several value-isolated candidates and keep the best-matching one.
        tried, best, best_val = 0, np.inf, float("nan")
        for iso in (1200.0, 800.0, 400.0):
            for i in rng.choice(cand, size=min(300, len(cand)), replace=False):
                nb = mtree.query_ball_point(mids[i], r=iso)
                if len(nb) == 0:
                    continue
                if np.nanmax(np.abs(seg_vals[nb] - seg_vals[i])) > 2.5:
                    continue
                if btree.query(mids[i])[0] < 8000.0:
                    continue
                win = piece_windows(i)
                if win is None:
                    continue
                exp_i = 0.8 * (np.array(cmap(norm(seg_vals[i]))[:3]) * 255) + 0.2 * BG
                d_i = float(np.min(np.max(np.abs(win - exp_i), axis=1)))
                if d_i < best:
                    best, best_val = d_i, float(seg_vals[i])
                tried += 1
                if best <= 5:
                    break
            if best <= 5:
                break
        if tried == 0:
            check(f"T10 pixel color d≈{target} km", None, "no isolated piece found")
        else:
            check(f"T10 pixel color d≈{target} km", best <= 25,
                  f"piece value {best_val:.2f} km, best Δ = {best:.0f} "
                  f"({tried} candidate pieces)")
    # Grey check: the unreachable fragments are tiny (T8), and pieces shorter
    # than a few pixels cannot be sampled reliably, so filter the NaN pieces by
    # length and take the one furthest from any reachable road.
    nan_idx = np.flatnonzero(np.isnan(seg_vals))
    if len(nan_idx):
        seg_len = np.array([np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum()
                            for p in (segs[i] for i in nan_idx)])
        long = nan_idx[seg_len >= 400.0]
        if len(long):
            fin = np.isfinite(seg_vals)
            d_fin = cKDTree(mids[fin]).query(mids[long], k=1)[0]
            i = long[int(np.argmax(d_fin))]
            d_gap = float(d_fin.max())
            if d_gap >= 200.0 and btree.query(mids[i])[0] >= 8000.0:
                win = window_rgb(*to_pixel(mids[i]))
                if win is None:
                    check("T10 unreachable road drawn grey", None, "sample outside the image")
                else:
                    best = np.min(np.max(np.abs(win - GREY_RGB), axis=1))
                    check("T10 unreachable road drawn grey", best <= 15,
                          f"{d_gap:.0f} m from any reachable road; "
                          f"expected {GREY_RGB}, best Δ = {best:.0f}")
            else:
                check("T10 unreachable road drawn grey", None,
                      f"grey pieces at most {d_gap:.0f} m from reachable roads "
                      f"or near the border")
        else:
            check("T10 unreachable road drawn grey", None, "no grey piece ≥ 400 m")
    else:
        check("T10 unreachable road drawn grey", None, "no unreachable pieces")

    # --------------------------------------- T11 registry coords vs OSM/Nominatim
    # Registry municipality names are unaccented uppercase ("MARINGA", not
    # "Maringá"); match accent-insensitively so a spelling mismatch can never
    # leave the sample frame empty (it once crashed the sampling).
    def norm_mun(s):
        return "".join(c for c in unicodedata.normalize("NFD", str(s).upper())
                       if unicodedata.category(c) != "Mn").strip()

    picks = []
    for mun in ("CURITIBA", "MARINGÁ"):
        sub = stations[(stations["UF"] == UF)
                       & (stations["MUNICÍPIO"].map(norm_mun) == norm_mun(mun))]
        sub = sub[sub["Endereço"].astype(str).str.contains(r"\d", regex=True, na=False)]
        sub = sub[sub["Endereço"].astype(str).str.len() > 12]
        if len(sub) == 0:
            check(f"T11 station sample {mun}", None, "no digit-address rows")
            continue
        for _, row in sub.sample(min(3, len(sub)), random_state=7).iterrows():
            picks.append(row)
    t11 = 0
    for row in picks[:4]:
        q = f"{row['Endereço']}, {row['MUNICÍPIO']}, Paraná, Brazil"
        enc = urllib.parse.quote(q)
        resp = nominatim(
            f"https://nominatim.openstreetmap.org/search?q={enc}&format=jsonv2&limit=1"
        )
        time.sleep(1.1)
        if not isinstance(resp, list) or not resp:
            check(f"T11 geocode '{row['Razão Social']}'", None, f"no OSM hit for {q[:60]}...")
            continue
        d_m = float(haversine_m(
            np.array([row["Latitude"]]), np.array([row["Longitude"]]),
            np.array([float(resp[0]["lat"])]), np.array([float(resp[0]["lon"])]),
        )[0])
        check(f"T11 geocode '{row['Razão Social'][:40]}'",
              d_m <= 300.0, f"OSM {resp[0]['lat']},{resp[0]['lon']} "
              f"vs registry {row['Latitude']},{row['Longitude']} = {d_m:.0f} m")
        t11 += 1
    if t11 == 0:
        check("T11 registry coords vs OSM", None, "no geocoding responses (rate limit?)")

    # --------------------------------------------------- T12 extents and area
    Xminp, Yminp, Xmaxp, Ymaxp = poly.bounds
    dx_km = (Xmaxp - Xminp) / 1000.0
    dy_km = (Ymaxp - Yminp) / 1000.0
    check("T12 E-W extent ≈ 674 km", 0.97 * 674 <= dx_km <= 1.03 * 674,
          f"{dx_km:.1f} km")
    check("T12 N-S extent ≈ 468 km", 0.97 * 468 <= dy_km <= 1.03 * 468,
          f"{dy_km:.1f} km")
    area = poly.area / 1e6
    check("T12 area ≈ 199,307.9 km² (IBGE)",
          abs(area - 199307.9) / 199307.9 <= 0.02,
          f"{area:,.0f} km² ({100 * (area / 199307.9 - 1):+.1f}%)")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 72)
    print(f"{'':4s}  {'check':52s}  detail")
    print("-" * 72)
    for status, name, detail in RESULTS:
        print(f"{status:4s}  {name:52s}  {detail}")
    n_pass = sum(1 for r in RESULTS if r[0] == "PASS")
    n_warn = sum(1 for r in RESULTS if r[0] == "WARN")
    n_fail = sum(1 for r in RESULTS if r[0] == "FAIL")
    print("-" * 72)
    print(f"{len(RESULTS)} checks: {n_pass} passed, {n_warn} warnings, "
          f"{n_fail} failed  ({time.time() - t0:.0f}s)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
