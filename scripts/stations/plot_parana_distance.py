"""Map of distance to the nearest gas station, for the state of Paraná.

Each road segment is colored by the true distance from its location to the
nearest gas station — ``min(d_u + x, d_v + (L - x))`` at arc distance ``x``
along the segment, not the endpoint mean (which overstates the middle of a long
rural road by up to ``L/2``). Segments longer than 1 km are subdivided so the
color follows that mid-segment peak. The scale is clipped to the 2nd-98th
percentiles, segments are sorted ascending so farther ones draw on top, and
segments unreachable from any station are drawn faintly grey. Drawn with a
single ``LineCollection`` so the ~1.5M state segments render in seconds rather
than the per-row plotting the city map uses.

The color scale is a linear ``Normalize`` between the 2nd and 98th percentiles
of in-state junction distances; the legend labels both ends with their km values.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from shapely import get_coordinates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.network import build_graph  # noqa: E402
from stations.parana import (  # noqa: E402
    CRS,
    DISPLAY,
    METRIC_NPY,
    load_boundary,
    load_road_segments,
    station_cache_key,
)
from stations.shared import SNAP_M_STATIONS  # noqa: E402

CMAP = "viridis_r"
PCT_LO = 2.0
PCT_HI = 98.0
BG = "#0b0f14"
GREY = "#3a3f47"
DPI = 300


def _explode(geometries, values):
    """Flatten LineString/MultiLineString into coordinate arrays + aligned values."""
    segs = []
    vals = []
    for g, val in zip(geometries, values):
        if g.geom_type == "LineString":
            segs.append(np.asarray(g.coords))
            vals.append(val)
        else:  # MultiLineString
            for part in g.geoms:
                segs.append(np.asarray(part.coords))
                vals.append(val)
    return segs, np.asarray(vals)


def _explode_subdiv(geometries, du, dv, max_len=1000.0):
    """Flatten whole segments, coloring each piece by its true arc-length distance.

    For a point at arc distance ``x`` (metres) from the start of a segment of
    length ``L`` with start/end distances ``du``/``dv`` (km), the nearest-station
    distance is ``min(du + x/1000, dv + (L - x)/1000)`` — leaving through whichever
    end is closer. Segments longer than ``max_len`` are split into straight
    sub-pieces so the color follows the mid-segment peak the endpoint mean misses;
    shorter segments are kept whole and colored at their midpoint. At whole-state
    scale a straight chord per piece is indistinguishable from the true curve.
    """
    segs = []
    vals = []
    for g, a, b in zip(geometries, du, dv):
        parts = [g] if g.geom_type == "LineString" else list(g.geoms)
        for part in parts:
            coords = np.asarray(part.coords)
            if len(coords) < 2:
                continue
            dl = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
            s = np.concatenate([[0.0], np.cumsum(dl)])
            L = s[-1]
            if L <= max_len:
                smid = L / 2.0
                segs.append(coords)
                vals.append(float(np.minimum(a + smid / 1000.0, b + (L - smid) / 1000.0)))
                continue
            # Sample arc positions every max_len (plus the true end) and
            # interpolate coordinates so each piece is a two-point chord.
            cuts = np.arange(max_len, L, max_len)
            xs = np.concatenate([[0.0], cuts, [L]])
            pts = np.column_stack(
                [np.interp(xs, s, coords[:, 0]), np.interp(xs, s, coords[:, 1])]
            )
            for i in range(len(xs) - 1):
                smid = (xs[i] + xs[i + 1]) / 2.0
                segs.append(pts[i : i + 2])
                vals.append(float(np.minimum(a + smid / 1000.0, b + (L - smid) / 1000.0)))
    return segs, np.asarray(vals)


def main():
    seg = load_road_segments()
    nodes, edges = build_graph(seg)

    values_km = np.load(METRIC_NPY) / 1000.0  # km
    if len(values_km) != len(nodes):
        raise SystemExit(
            f"Stale metric: {METRIC_NPY} has {len(values_km):,} values but the "
            f"graph has {len(nodes):,} nodes. Re-run "
            f"scripts/stations/compute_parana_distance.py."
        )
    key = station_cache_key(SNAP_M_STATIONS)
    with open(METRIC_NPY + ".meta") as f:
        if f.read().strip() != key:
            raise SystemExit(
                f"Stale metric: {METRIC_NPY} was computed with different data or "
                f"parameters. Re-run scripts/stations/compute_parana_distance.py."
            )
    assert (nodes.index.to_numpy() == np.arange(len(nodes))).all(), (
        "node ids must be 0..n-1 so values_km[node_id] indexes correctly"
    )

    boundary = load_boundary(CRS)

    # Full-detail boundary: keep the raw ~37k-vertex outline so the state border
    # and the segment trimming are pixel-exact.
    poly = boundary.geometry.iloc[0]
    boundary = gpd.GeoDataFrame(geometry=[poly], crs=CRS)

    # Trim the road network to the state outline.
    #
    # The distance metric is already computed (the cached .npy); this step is only
    # the drawing's geometry work — trim roads to the state outline and hide the
    # out-of-state roads the rectangular bbox pulled in. Clipping all ~1.5M
    # segments against the full-detail boundary is what used to take tens of
    # minutes, so classify each segment by its endpoints and clip only the small
    # fraction that actually cross the border:
    #   * both endpoints inside  -> keep whole (no clip)
    #   * one in, one out        -> crosses the border; clip to trim it
    #   * both endpoints outside -> drop
    # (A segment that dips outside a concave notch with both ends inside stays
    # whole rather than trimmed — a sliver invisible at this scale; the boundary
    # outline itself keeps its full detail.)
    u = edges.index.get_level_values(0).to_numpy()
    v = edges.index.get_level_values(1).to_numpy()
    inside = nodes.geometry.within(poly).to_numpy()
    one_in = inside[u] ^ inside[v]

    keep_whole = edges[~one_in & inside[u]]
    clipped = edges[one_in].clip(boundary, keep_geom_type=True)
    clipped = clipped[
        clipped.geometry.notna() & ~clipped.geometry.is_empty
        & clipped.geom_type.isin(["LineString", "MultiLineString"])
    ]
    print(
        f"Clipped to state: {len(keep_whole) + len(clipped):,} segments "
        f"({len(keep_whole):,} kept whole, {len(clipped):,} border segments clipped)",
        flush=True,
    )

    # Whole segments keep their exact node endpoints, so a point along the road
    # has the true distance min(d_u + x, d_v + (L - x)); subdivide long segments
    # so the color follows the mid-segment peak instead of a single mean.
    uw = keep_whole.index.get_level_values(0).to_numpy()
    vw = keep_whole.index.get_level_values(1).to_numpy()
    # segments_to_graph does not guarantee the geometry runs from u to v (~1/3 of
    # edges are stored reversed), so align the endpoint distances with the
    # geometry's own direction before subdividing — otherwise the gradient flips
    # along the reversed long edges.
    P = get_coordinates(nodes.geometry.values)  # (n_nodes, 2), in node-id order
    # get_coordinates(g)[0] (not g.coords[0]) also handles a MultiLineString edge,
    # which .coords does not support; edges are LineStrings today, so this is the
    # same result, just without the latent crash.
    g0 = np.array([get_coordinates(g)[0] for g in keep_whole.geometry])
    start_is_u = np.hypot(*(g0 - P[uw]).T) <= np.hypot(*(g0 - P[vw]).T)
    du = np.where(start_is_u, values_km[uw], values_km[vw])
    dv = np.where(start_is_u, values_km[vw], values_km[uw])
    segs, seg_vals = _explode_subdiv(keep_whole.geometry, du, dv)

    # Border-crossing segments are trimmed by the clip, so a fragment's position
    # along its original edge is lost; fall back to the endpoint mean (few, short).
    uc = clipped.index.get_level_values(0).to_numpy()
    vc = clipped.index.get_level_values(1).to_numpy()
    segs_c, vals_c = _explode(clipped.geometry, (values_km[uc] + values_km[vc]) / 2.0)
    segs += segs_c
    seg_vals = np.concatenate([seg_vals, vals_c])

    valid = ~np.isnan(seg_vals)
    missing = ~valid

    scale = values_km[inside]
    finite = scale[~np.isnan(scale)]
    for km in (0.5, 1.0, 5.0):
        print(f"  {100 * (finite <= km).mean():.1f}% of in-state junctions within {km:g} km")
    lo = float(np.nanpercentile(scale, PCT_LO))
    hi = float(np.nanpercentile(scale, PCT_HI))
    norm = Normalize(vmin=lo, vmax=hi, clip=False)
    cmap = plt.get_cmap(CMAP)

    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    if missing.any():
        ax.add_collection(
            LineCollection([segs[i] for i in np.flatnonzero(missing)],
                           colors=GREY, linewidths=0.4, zorder=1)
        )

    order = np.argsort(seg_vals[valid])  # ascending: far (dark) draws on top
    idx = np.flatnonzero(valid)[order]
    ax.add_collection(
        LineCollection(
            [segs[i] for i in idx],
            colors=cmap(norm(seg_vals[idx])),
            linewidths=0.4,
            alpha=0.8,
            zorder=2,
        )
    )

    boundary.boundary.plot(ax=ax, color="white", linewidth=0.6, alpha=0.8, zorder=3)

    ax.set_title("Distance to the nearest gas station", color="white",
                 fontsize=34, fontweight="bold", pad=48)
    ax.text(0.5, 1.02, DISPLAY, transform=ax.transAxes, ha="center",
            va="bottom", color="#c4c9d0", fontsize=28)

    sm = plt.cm.ScalarMappable(cmap=cmap.reversed(), norm=norm)
    sm.set_array(seg_vals)
    cax = fig.add_axes([0.570, 0.06, 0.33, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    x0, y0 = cax.get_position().x0, cax.get_position().y0
    w, h = cax.get_position().width, cax.get_position().height
    gap = 8 / (fig.get_size_inches()[1] * 72)
    fig.text(x0, y0 + h + gap, "far", ha="left", va="bottom", color="white", fontsize=16)
    fig.text(x0 + w, y0 + h + gap, "near", ha="right", va="bottom", color="white", fontsize=16)

    ax.set_axis_off()
    ax.margins(0)

    os.makedirs(os.path.dirname(METRIC_NPY), exist_ok=True)
    path = "outputs/pr/station_distance.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    print(f"Saved to {path}")
    print(f"color scale: {lo:.2f} km (near) -> {hi:.2f} km (far)")


if __name__ == "__main__":
    main()
