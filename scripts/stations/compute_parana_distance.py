"""Compute the road-network distance from every junction to the nearest gas
station, for the whole state of Paraná.

Every station with coordinates (any state) that lies inside the network bbox is
snapped to its nearest network node and a single multi-source Dijkstra
(numba-JIT, weighted by ``length_m``) yields the nearest-station distance for
every node. Seeding stations from neighbouring states too (not just
``UF == "PR"``) keeps junctions near the state line from being painted
artificially far — their real nearest station is often just across the border.
The network is the drivable road network — every road class except the
pedestrian/cycle-only classes, so expressways and service roads (driveways,
parking aisles) are kept. Results are cached to
``outputs/pr/station_distance.npy`` keyed to the station file, the snap
threshold, the seeding scope and the segment cache.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.network import build_graph  # noqa: E402
from stations.parana import (  # noqa: E402
    DISPLAY,
    METRIC_NPY,
    UF,
    build_csr,
    load_boundary,
    load_road_segments,
    load_stations,
    multi_source_dijkstra,
    station_cache_key,
)
from stations.shared import SNAP_M_STATIONS, print_stats, snap_stations  # noqa: E402


def main():
    seg = load_road_segments()
    nodes, edges = build_graph(seg)
    n = len(nodes)
    print(f"Road network: {n:,} nodes, {len(edges):,} edges")

    cache = METRIC_NPY
    cache_meta = cache + ".meta"
    key = station_cache_key(SNAP_M_STATIONS)

    cached = os.path.exists(cache) and os.path.exists(cache_meta)
    if cached:
        with open(cache_meta) as f:
            cached = f.read().strip() == key

    if cached:
        dist = np.load(cache)
        print(f"Loaded cached metric from {cache}")
    else:
        stations = load_stations()  # every station with coordinates, any state
        xmin, ymin, xmax, ymax = seg.total_bounds
        in_bbox = stations.cx[xmin:xmax, ymin:ymax]
        n_pr = int((in_bbox["UF"] == UF).sum())
        print(
            f"{len(in_bbox):,} stations inside the network bbox "
            f"({n_pr:,} in {DISPLAY}, {len(in_bbox) - n_pr:,} in neighbouring states)"
        )
        # Audit trail: the cleaned stations that feed this map (WGS 84 lat/lon).
        cols = ["UF", "MUNICÍPIO", "Razão Social", "Latitude", "Longitude",
                "Vinculação a Distribuidor"]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        in_bbox[cols].to_csv("outputs/pr/stations_used.csv", index=False)
        print(f"Wrote cleaned station list to outputs/pr/stations_used.csv")
        sources, _snap_dists, _valid_idx, n_dropped = snap_stations(in_bbox, nodes)
        print(
            f"-> {len(np.unique(sources)):,} unique nodes seeded "
            f"({n_dropped} dropped beyond {SNAP_M_STATIONS:.0f} m from any node)"
        )
        head, to, wt = build_csr(nodes, edges)
        print(f"Computing nearest-station distance for {n:,} nodes ...")
        dist = multi_source_dijkstra(head, to, wt, sources)
        dist[~np.isfinite(dist)] = np.nan
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, dist)
        with open(cache_meta, "w") as f:
            f.write(key)
        print(f"Saved metric to {cache}")

    inside = nodes.geometry.within(load_boundary().geometry.iloc[0]).to_numpy()
    print("Distance to nearest gas station (km):")
    for label, mask in (("buffered", slice(None)), ("Paraná", inside)):
        print_stats(dist[mask], label)


if __name__ == "__main__":
    main()
