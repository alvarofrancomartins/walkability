"""Network-colored map of reachable walkable network length (sequential viridis).

Colors each segment by its reachable network length — the total walkable network
length inside the 10-minute walking isochrone — taken as the mean of its two
endpoints, clipped to the 2nd-98th percentiles with extend arrows, and sorted
ascending so higher values draw on top. Segments with no value (in dropped
fragments) are drawn faintly grey. The metric is computed on the buffered
network (which extends beyond the municipality); the drawing clips to the
municipal boundary so only the city itself is shown.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

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

CMAP = "viridis"
CUTOFF_MIN = CUTOFF_S / 60.0  # minutes, matches the metric's cutoff
PCT_LO = 2.0
PCT_HI = 98.0
BG = "#0b0f14"


def load_network(city="curitiba"):
    """The walking network of a city plus its boundary."""
    nodes, edges = build_graph(load_segments(city))
    return nodes, edges, load_boundary(city)


def attach(edges, boundary, values_km, column):
    """Clip the network to the municipality with the metric attached per segment.

    Each segment takes the mean of its two endpoints' values (a linear
    approximation to the variation along the road). Clipping uses
    keep_geom_type=True: without it, segments that merely graze the clip edge
    (in the narrow NW tip) come back as Points, which plot() draws as marker
    dots that ignore linewidth.
    """
    u = edges.index.get_level_values(0).to_numpy()
    v = edges.index.get_level_values(1).to_numpy()
    assert u.max() < len(values_km) and v.max() < len(values_km), "node out of range"
    edges[column] = (values_km[u] + values_km[v]) / 2.0

    ec = edges.clip(boundary, keep_geom_type=True)
    ec = ec[
        ec.geometry.notna()
        & ~ec.geometry.is_empty
        & ec.geom_type.isin(["LineString", "MultiLineString"])
    ]
    return ec.sort_values(column, na_position="first")


def node_mask(nodes, boundary):
    """Boolean mask of nodes inside the municipal boundary (the color basis)."""
    return nodes.geometry.within(boundary.geometry.iloc[0]).to_numpy()


def draw_map(fig, ax, edges_cur, column, title, subtitle, scale_values):
    """Draw the metric map; returns the (lo, hi) of its color scale.

    The color scale is taken from `scale_values` — the per-node metric values of
    the junctions inside the municipal boundary — not from the per-segment
    values actually drawn, so every junction carries equal weight regardless of
    how many segments meet there.
    """
    v = edges_cur[column].to_numpy()
    lo = float(np.nanpercentile(scale_values, PCT_LO))
    hi = float(np.nanpercentile(scale_values, PCT_HI))
    norm = Normalize(vmin=lo, vmax=hi, clip=False)

    ax.set_facecolor(BG)
    edges_cur.plot(
        ax=ax,
        column=column,
        cmap=CMAP,
        norm=norm,
        linewidth=0.5,
        alpha=0.9,
        zorder=2,
        # segments in dropped fragments have no value; draw them faintly
        # instead of leaving them invisible against the background
        missing_kwds={"color": "#3a3f47", "linewidth": 0.5},
    )

    # Big city name, with a subtitle just beneath it.
    ax.set_title(title, color="white", fontsize=34, fontweight="bold", pad=70)
    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#c4c9d0",
        fontsize=20,
    )

    # Minimal gradient legend: a "sparse → dense" bar in the bottom-left corner,
    # with its labels just above the bar (not on it, where white text would clash
    # with the yellow end of the palette).
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array(v)
    cax = fig.add_axes([0.570, 0.13, 0.33, 0.02])  # bottom-right, raised toward the map
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    x0, y0 = cax.get_position().x0, cax.get_position().y0
    w, h = cax.get_position().width, cax.get_position().height
    gap = 8 / (fig.get_size_inches()[1] * 72)  # ~8 pt gap above the bar
    fig.text(x0, y0 + h + gap, "sparse", ha="left", va="bottom", color="white", fontsize=16)
    fig.text(x0 + w, y0 + h + gap, "dense", ha="right", va="bottom", color="white", fontsize=16)

    ax.set_axis_off()
    return lo, hi


def main():
    city = parse_city()
    display = CITY_CONFIGS[city]["display"]
    metric_path = f"outputs/{city}/reachable_length.npy"
    plot_path = f"outputs/{city}/reachable_length.png"

    key = reach_cache_key(city)
    meta_path = metric_path + ".meta"
    if not os.path.exists(metric_path):
        sys.exit(
            f"{metric_path} is missing — run "
            f"scripts/compute_reachable_length.py --city {city} first."
        )
    if not os.path.exists(meta_path):
        sys.exit(
            f"{meta_path} is missing — re-run "
            f"scripts/compute_reachable_length.py --city {city}."
        )
    with open(meta_path) as f:
        if f.read().strip() != key:
            sys.exit(
                f"{metric_path} is stale (parameters changed) — re-run "
                f"scripts/compute_reachable_length.py --city {city}."
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
        f"Reachable walkable network length\n{CUTOFF_MIN:g}-min walking isochrone",
        scale,
    )
    boundary.boundary.plot(ax=ax, color="white", linewidth=0.5, alpha=0.6, zorder=3)

    os.makedirs(f"outputs/{city}", exist_ok=True)
    fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Saved to {plot_path}")
    print(f"color scale: {lo:.0f} km (low) -> {hi:.0f} km (high)")


if __name__ == "__main__":
    main()
