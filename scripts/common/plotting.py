"""Shared drawing helpers for the walkability city map.

``plot_reachable_length.py`` attaches a per-segment value as the mean of its
endpoints, clips to the municipal boundary, applies the 2nd–98th percentile
color scale, and lays out the "low -> high" gradient legend; those pieces live
here once.
"""

import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

# Shared map style: dark background, the "no value" grey, the percentile clip
# and the render dpi.
BG = "#0b0f14"
GREY = "#3a3f47"
PCT_LO = 2.0
PCT_HI = 98.0
DPI = 600


def attach(edges, boundary, values_km, column):
    """Clip the network to the municipality with the metric attached per segment.

    Each segment takes the mean of its two endpoints' values (a linear
    approximation to the variation along the road). Clipping uses
    ``keep_geom_type=True``: without it, segments that merely graze the clip
    edge come back as Points, which ``plot()`` draws as marker dots that ignore
    linewidth.
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


def draw_map(fig, ax, edges_cur, column, title, subtitle, scale_values, cmap, bar_labels):
    """Draw the metric map; returns the ``(lo, hi)`` of its color scale.

    The color scale is taken from ``scale_values`` — the per-node metric values
    of the junctions inside the municipal boundary — not from the per-segment
    values actually drawn, so every junction carries equal weight regardless of
    how many segments meet there. ``bar_labels`` is a ``(low, high)`` pair of
    words for the two ends of the gradient legend, e.g. ``("sparse", "dense")``.
    """
    v = edges_cur[column].to_numpy()
    lo = float(np.nanpercentile(scale_values, PCT_LO))
    hi = float(np.nanpercentile(scale_values, PCT_HI))
    norm = Normalize(vmin=lo, vmax=hi, clip=False)

    ax.set_facecolor(BG)
    edges_cur.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        norm=norm,
        linewidth=0.5,
        alpha=0.9,
        zorder=2,
        # segments in dropped fragments have no value; draw them faintly
        # instead of leaving them invisible against the background
        missing_kwds={"color": GREY, "linewidth": 0.5},
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

    # Minimal gradient legend: a "low -> high" bar in the bottom-right corner,
    # with its labels just above the bar (not on it, where white text would
    # clash with the bright end of the palette).
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(v)
    cax = fig.add_axes([0.570, 0.13, 0.33, 0.02])  # bottom-right, raised toward the map
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    x0, y0 = cax.get_position().x0, cax.get_position().y0
    w, h = cax.get_position().width, cax.get_position().height
    gap = 8 / (fig.get_size_inches()[1] * 72)  # ~8 pt gap above the bar
    fig.text(x0, y0 + h + gap, bar_labels[0], ha="left", va="bottom", color="white", fontsize=16)
    fig.text(x0 + w, y0 + h + gap, bar_labels[1], ha="right", va="bottom", color="white", fontsize=16)

    ax.set_axis_off()
    return lo, hi


def parse_cmap(default):
    """Parse ``--cmap`` from the command line; falls back to ``default``."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cmap", default=default)
    return parser.parse_known_args()[0].cmap
