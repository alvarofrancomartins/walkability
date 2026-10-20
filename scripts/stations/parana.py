"""The Paraná state-wide gas-station distance pipeline.

Reads the Overture ``segment``/``connector`` GeoParquet files downloaded for the
state bbox, keeps the drivable road network (every road class except the
pedestrian/cycle-only classes), splits at connectors and snaps endpoints, and
caches the processed segments to ``data/pr/road_segments.parquet``.

The distance metric is a single multi-source Dijkstra seeded with every snapped
gas station, JIT-compiled with numba, over the road network weighted by
``length_m``. Station loading/snapping is shared via :mod:`stations.shared`.
"""

import hashlib
import json
import os
import sys

import numpy as np
from numba import njit

import city2graph as c2g
import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.network import NON_ROAD_CLASSES  # noqa: E402
from stations.cleaning import CLEAN_VERSION, clean_stations, cleaning_summary  # noqa: E402

UF = "PR"
DISPLAY = "Paraná"
CRS = "EPSG:32722"  # UTM 22S

STATIONS_XLSX = "data/all_gas_stations.xlsx"
SEG_PARQUET = "data/pr/segment.parquet"
CONN_PARQUET = "data/pr/connector.parquet"
BOUNDARY_GEOJSON = "data/pr/boundary.geojson"
SEG_CACHE = "data/pr/road_segments.parquet"
METRIC_NPY = "outputs/pr/station_distance.npy"

SNAP_M = 5.0  # endpoint clustering threshold (matches common.network.SNAP_M)

# Bump when the state network's filtering/graph logic changes, so the segment
# cache is rebuilt instead of silently reused.
CODE_VERSION = 1


def load_boundary(crs=CRS):
    """Paraná boundary (projected), cached at data/pr/boundary.geojson."""
    gdf = gpd.read_file(BOUNDARY_GEOJSON)
    return gdf.to_crs(crs)


def _norm_connectors(value):
    """Parquet stores the ``connectors`` column as numpy arrays of dicts, but
    ``process_overture_segments`` expects a plain list. Convert so the splitter
    sees connectors the same way as the GeoJSON-loaded data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _seg_cache_key():
    """Hash of source-file fingerprints + params; invalidates the segment cache."""
    fp = {}
    for name, path in (("segment", SEG_PARQUET), ("connector", CONN_PARQUET)):
        st = os.stat(path)
        fp[name] = (st.st_size, st.st_mtime_ns)
    return hashlib.sha1(
        json.dumps({"code_version": CODE_VERSION, "snap_m": SNAP_M, "sources": fp}, sort_keys=True).encode()
    ).hexdigest()[:12]


def station_cache_key(snap_m, lat_col="Latitude", lon_col="Longitude", scope="all_uf"):
    """Hash of the station file + params + segment cache; the ``.npy`` cache key.

    ``scope`` records which stations seed the Dijkstra — ``"all_uf"`` seeds every
    station with coordinates (any state), so a border junction can reach a station
    just across the state line. Changing it invalidates the cached metric.
    """
    st = os.stat(STATIONS_XLSX)
    fp = {"size": st.st_size, "mtime": st.st_mtime_ns}
    seg_key = open(SEG_CACHE + ".meta").read().strip()
    return hashlib.sha1(
        json.dumps(
            {
                "metric": "parana_station_distance",
                "snap_m": snap_m,
                "lat_col": lat_col,
                "lon_col": lon_col,
                "station_scope": scope,
                "clean_version": CLEAN_VERSION,
                "stations": fp,
                "segments": seg_key,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]


def load_road_segments():
    """Drivable road segments for Paraná, split + snapped (cached).

    Returns the processed GeoDataFrame (projected, with ``length_m``).
    """
    meta_path = SEG_CACHE + ".meta"
    key = _seg_cache_key()
    if os.path.exists(SEG_CACHE) and os.path.exists(meta_path):
        with open(meta_path) as f:
            if f.read().strip() == key:
                print(f"Loading cached road segments from {SEG_CACHE}")
                return gpd.read_parquet(SEG_CACHE)

    print("Reading segment parquet ...", flush=True)
    seg = gpd.read_parquet(
        SEG_PARQUET, columns=["subtype", "class", "connectors", "level_rules", "geometry"]
    )
    seg["connectors"] = seg["connectors"].map(_norm_connectors)

    n_raw = len(seg)
    seg = seg[seg["subtype"] == "road"]
    n_road = len(seg)
    seg = seg[~seg["class"].isin(NON_ROAD_CLASSES)]
    n_drive = len(seg)
    print(
        f"Filtered {n_raw:,} raw bbox segments -> {n_drive:,} drivable roads "
        f"(dropped {n_raw - n_road:,} rail/water, {n_road - n_drive:,} non-drivable class)",
        flush=True,
    )

    # Project the whole bbox (no pre-clip: clipping would split some roads into
    # MultiLineStrings, which the connector splitter cannot handle). Out-of-state
    # roads from the bbox are real drivable links and are clipped at plot time.
    seg = seg.to_crs(CRS)
    connectors = gpd.read_parquet(CONN_PARQUET, columns=["id", "geometry"]).to_crs(CRS)

    print("Splitting at connectors and snapping endpoints (long step) ...", flush=True)
    seg = c2g.process_overture_segments(seg, connectors_gdf=connectors, threshold=SNAP_M, get_barriers=False)
    seg = seg[seg.geometry.notna() & ~seg.geometry.is_empty].copy()
    seg["length_m"] = seg.geometry.length
    # The split/snap is done; these raw Overture columns are no longer needed and
    # (holding numpy arrays / lists of dicts) can't be written back to parquet.
    seg = seg.drop(columns=["connectors", "level_rules"], errors="ignore")

    print(
        f"Split/snap done: {len(seg):,} road segments "
        f"(from {n_drive:,} pre-split drivable roads)",
        flush=True,
    )

    os.makedirs(os.path.dirname(SEG_CACHE), exist_ok=True)
    seg.to_parquet(SEG_CACHE)
    with open(meta_path, "w") as f:
        f.write(key)
    print(f"Cached road segments to {SEG_CACHE}")
    return seg


def load_stations():
    """Gas stations with coordinates, from any state, as a GeoDataFrame (projected).

    Seeding the Dijkstra with every station — not just ``UF == "PR"`` — lets a
    junction near the state line reach a station just across it, so border areas
    are not painted artificially far. Stations far from the network bbox are
    dropped at snap time anyway.

    The registry is cleaned first (see :mod:`stations.cleaning`): rows missing
    coordinates, aviation-fuel suppliers, floating (boat-fuel) stations, and
    stations on a bogus shared coordinate are excluded, so the map reads
    "distance to the nearest *car-fuel* station".
    """
    df = pd.read_excel(STATIONS_XLSX, sheet_name="Sheet1")
    df, dropped = clean_stations(df)
    print(cleaning_summary(dropped, "Gas stations"))
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="EPSG:4326"
    ).to_crs(CRS)


def build_csr(nodes, edges):
    """Collapse parallel edges to min-length and build undirected CSR arrays.

    Returns ``(head, to, wt)`` where ``wt`` is the edge length (metres)."""
    n = len(nodes)
    assert (nodes.index.to_numpy() == np.arange(n)).all(), "node ids must be 0..n-1"

    u = edges.index.get_level_values(0).to_numpy(np.int64)
    v = edges.index.get_level_values(1).to_numpy(np.int64)
    wl = edges["length_m"].to_numpy(np.float64)

    # Collapse parallel edges (and drop self-loops) to the minimum length per
    # undirected node pair — vectorized instead of a Python loop over the edges.
    df = pd.DataFrame({"lo": np.minimum(u, v), "hi": np.maximum(u, v), "l": wl})
    best = df[df["lo"] != df["hi"]].groupby(["lo", "hi"], sort=False)["l"].min()
    src = best.index.get_level_values(0).to_numpy(np.int64)
    dst = best.index.get_level_values(1).to_numpy(np.int64)
    w = best.to_numpy(np.float64)

    # Undirected: each collapsed pair (a, b) becomes directed edges a->b and b->a.
    # Lay out the CSR adjacency with bincount + argsort instead of Python loops.
    s2 = np.concatenate([src, dst])
    t2 = np.concatenate([dst, src])
    w2 = np.concatenate([w, w])
    order = np.argsort(s2)
    to = t2[order]
    wt = w2[order]

    deg = np.bincount(s2, minlength=n)
    head = np.zeros(n + 1, np.int64)
    head[1:] = np.cumsum(deg)
    return head, to, wt


@njit
def multi_source_dijkstra(head, to, wt, sources):
    """Nearest-source distance (metres) for every node, all sources at once.

    A single Dijkstra seeded with every source at distance 0; each node's
    distance is then the shortest path to any source. Unreachable nodes keep
    ``inf`` (converted to ``NaN`` by the caller).
    """
    n = head.shape[0] - 1
    m = to.shape[0]
    dist = np.full(n, np.inf)
    ht = np.empty(m + sources.size + 1)
    hv = np.empty(m + sources.size + 1, np.int64)
    size = 0

    for s in sources:
        if dist[s] > 0.0:
            dist[s] = 0.0
            ht[size] = 0.0
            hv[size] = s
            j = size
            size += 1
            while j > 0:
                p = (j - 1) // 2
                if ht[p] <= ht[j]:
                    break
                ht[p], ht[j] = ht[j], ht[p]
                hv[p], hv[j] = hv[j], hv[p]
                j = p

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
        if t > dist[u]:
            continue
        for i in range(head[u], head[u + 1]):
            v = to[i]
            nd = t + wt[i]
            if nd < dist[v]:
                dist[v] = nd
                ht[size] = nd
                hv[size] = v
                j = size
                size += 1
                while j > 0:
                    p = (j - 1) // 2
                    if ht[p] <= ht[j]:
                        break
                    ht[p], ht[j] = ht[j], ht[p]
                    hv[p], hv[j] = hv[j], hv[p]
                    j = p

    return dist
