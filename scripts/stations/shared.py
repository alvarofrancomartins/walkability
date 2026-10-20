"""Shared gas-station helpers for the Paraná state pipeline.

The state pipeline (``parana`` / ``compute_parana_distance`` /
``plot_parana_distance``) needs to snap each station to its nearest network node
and print the resulting distance statistics; those pieces live here once.
"""

import geopandas as gpd
import numpy as np

# Stations snap to their nearest network node; points farther than this (metres)
# are treated as not on the network and dropped. Generous because a station may
# sit mid-block, some distance from the nearest junction (node), yet still be
# right beside the road.
SNAP_M_STATIONS = 100.0


def snap_stations(stations, nodes, snap_m=SNAP_M_STATIONS):
    """Snap each station to its nearest network node (within ``snap_m``).

    Returns ``(sources, snap_dists, valid_idx, n_dropped)`` where ``sources`` are
    the node ids each kept station snapped to, ``snap_dists`` the snap distances
    (metres), ``valid_idx`` the position of each kept station in ``stations``,
    and ``n_dropped`` the number of stations beyond the threshold.
    """
    joined = gpd.sjoin_nearest(
        stations[["geometry"]],
        nodes[["geometry"]],
        how="left",
        max_distance=snap_m,
        distance_col="snap_m",
    )
    # sjoin_nearest returns one row per tie, so a station exactly equidistant from
    # two nodes would seed both; keep one row per station (the left index).
    joined = joined[~joined.index.duplicated(keep="first")]
    # The right-index column is named after the nodes index ('index_right' when
    # unnamed, e.g. 'node_id' when the graph builder names it); detect it rather
    # than hardcoding, so this keeps working across geopandas versions.
    id_col = next(c for c in joined.columns if c not in ("geometry", "snap_m"))
    valid = joined[id_col].notna()
    sources = joined.loc[valid, id_col].astype(np.int64).to_numpy()
    snap_dists = joined.loc[valid, "snap_m"].to_numpy()
    valid_idx = joined.index.to_numpy()[valid.to_numpy()]  # input-frame index values
    return sources, snap_dists, valid_idx, int((~valid).sum())


def print_stats(dist, label):
    """Print min/median/max nearest-station distance for a labelled node set."""
    d = dist[~np.isnan(dist)]
    if len(d) == 0:
        print(f"  {label}: no nodes")
        return
    print(
        f"  {label}: min {d.min()/1000:.2f}  median {np.median(d)/1000:.2f}  "
        f"max {d.max()/1000:.2f} km  (n={len(d):,})"
    )
