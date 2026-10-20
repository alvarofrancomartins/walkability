"""Network-colored map of reachable walkable network length (sequential viridis).

Colors each segment by its reachable network length — the total walkable network
length inside the 10-minute isochrone — taken as the mean of its two endpoints,
clipped to the 2nd-98th percentiles with extend arrows, and sorted ascending so
higher values draw on top. Segments with no value (in dropped fragments) are
drawn faintly grey. The metric is computed on the buffered network (which
extends beyond the municipality); the drawing clips to the municipal boundary so
only the city itself is shown.

The drawing itself is shared with the station-distance map via
:mod:`common.plotting`; only the colormap, titles and colorbar labels differ
here.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.network import (  # noqa: E402
    CITY_CONFIGS,
    build_graph,
    load_boundary,
    load_segments,
    parse_city,
)
from common.plotting import BG, DPI, attach, draw_map, node_mask, parse_cmap  # noqa: E402
from walkability.metric import CUTOFF_S, reach_cache_key  # noqa: E402

CMAP = "viridis"
CUTOFF_MIN = CUTOFF_S / 60.0  # minutes, matches the metric's cutoff


def load_network(city="curitiba"):
    """The walking network of a city plus its boundary."""
    nodes, edges = build_graph(load_segments(city))
    return nodes, edges, load_boundary(city)


def main():
    city = parse_city()
    cmap = parse_cmap(CMAP)
    display = CITY_CONFIGS[city]["display"]
    metric_path = f"outputs/{city}/reachable_length.npy"
    suffix = "" if cmap == CMAP else f"_{cmap}"
    plot_path = f"outputs/{city}/reachable_length{suffix}.png"

    key = reach_cache_key(city)
    meta_path = metric_path + ".meta"
    if not os.path.exists(metric_path):
        sys.exit(
            f"{metric_path} is missing — run "
            f"scripts/walkability/compute_reachable_length.py --city {city} first."
        )
    if not os.path.exists(meta_path):
        sys.exit(
            f"{meta_path} is missing — re-run "
            f"scripts/walkability/compute_reachable_length.py --city {city}."
        )
    with open(meta_path) as f:
        if f.read().strip() != key:
            sys.exit(
                f"{metric_path} is stale (parameters changed) — re-run "
                f"scripts/walkability/compute_reachable_length.py --city {city}."
            )

    values = np.load(metric_path) / 1000.0  # km
    nodes, edges, boundary = load_network(city)
    ec = attach(edges, boundary, values, "reachable_km")
    scale = values[node_mask(nodes, boundary)]

    fig, ax = plt.subplots(figsize=(9, 13))
    fig.patch.set_facecolor(BG)
    lo, hi = draw_map(
        fig,
        ax,
        ec,
        "reachable_km",
        display,
        f"Reachable walkable network length\n{CUTOFF_MIN:g}-min isochrone",
        scale,
        cmap,
        ("sparse", "dense"),
    )
    boundary.boundary.plot(ax=ax, color="white", linewidth=0.5, alpha=0.6, zorder=3)

    os.makedirs(f"outputs/{city}", exist_ok=True)
    fig.savefig(plot_path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    print(f"Saved to {plot_path}")
    print(f"color scale: {lo:.0f} km (low) -> {hi:.0f} km (high)")


if __name__ == "__main__":
    main()
