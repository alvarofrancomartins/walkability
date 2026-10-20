"""Shared Overture road-network prep, used by both analyses (with cache).

Loads raw Overture segments + connectors for a city and a surrounding buffer,
filters them into one of two views of the network, splits at connectors and
snaps endpoints, and returns the processed segments plus the graph's node/edge
tables:

* ``load_segments(city, "walk")`` — the pedestrian network (analysis 1:
  walkability). Every road class except motorways/trunk, minus segments
  explicitly closed to foot traffic, and minus sidewalk/crosswalk/driveway/
  parking subclasses that duplicate a road or are private vehicle access.
* ``load_segments(city, "road")`` — the drivable network (analysis 2: gas-station
  accessibility). Every road class except the pedestrian/cycle-only classes.

A single modeled walking speed is assigned so the walkable view carries a travel
time. The city is chosen with ``--city`` (default ``curitiba``). Processed
segments are cached to GeoParquet keyed to the pipeline parameters and source
files, so a parameter or data change invalidates the cache automatically.

The walkability metric's own definition (cutoff, metric version, ``.npy`` cache
key) lives in :mod:`walkability.metric`, not here.
"""

import ast
import hashlib
import json
import os

import city2graph as c2g
import geopandas as gpd

SNAP_M = 5.0

# Bump when the filtering/graph logic changes so the parquet and reach
# caches (which key off this module) are rebuilt instead of silently reused.
CODE_VERSION = 5

# Walking speed model: a single constant. 4.5 km/h (1.25 m/s) is the standard
# pedestrian design speed used in walkability/accessibility literature, so the
# 10-minute budget is ~750 m of travel.
WALK_SPEED_KMH = 4.5

# Road classes a pedestrian cannot walk on. Motorways are controlled-access
# highways with no pedestrian access. ``trunk`` (expressways, rodovias,
# marginais) is also dropped: ~99% of its segments carry no foot-access tag, so
# keeping them treats expressway centerlines as walkable shortcuts across
# barriers (see the trunk note in *Assumptions & limitations*). Rail segments
# (class ``standard_gauge``) are excluded separately by the ``subtype == "road"``
# filter. Every other class — primary, residential, footway, path, cycleway,
# steps, track, service, living_street, ... — is kept; segments genuinely
# closed to foot traffic are removed via the ``access_restrictions`` foot
# denials rather than a blanket class drop.
NON_WALK_CLASSES = {"motorway", "trunk"}

# Road classes a motor vehicle cannot drive on. These are the pedestrian- and
# cycle-only classes — footway, path, steps, cycleway, pedestrian. Every other
# road class (motorway, trunk, primary, secondary, tertiary, residential,
# service, track, living_street, unclassified, ...) is drivable, so the driving
# network is the complement of the walkable filter: expressways are kept and
# the non-drivable classes are dropped. Used for the road-network metric
# (distance to the nearest gas station).
NON_ROAD_CLASSES = {"footway", "path", "cycleway", "steps", "pedestrian"}

# Overture ``mode`` tokens that mean a denial closes the segment to pedestrians.
# A denial restricted to any other mode (car, bicycle, hgv, bus, ...) does not
# close it to walking.
FOOT_MODES = {"foot", "pedestrian"}

# Subclasses dropped because they are not part of the walkable street network.
# ``sidewalk`` / ``crosswalk`` / ``cycle_crossing`` are separate geometries that
# duplicate a road centerline the metric already treats as walkable; keeping
# them double-counts and rewards areas digitized in more detail. ``driveway`` /
# ``parking_aisle`` are private vehicle access, not pedestrian network.
DROP_SUBCLASS = {
    "footway": {"sidewalk", "crosswalk"},
    "cycleway": {"cycle_crossing"},
    "service": {"driveway", "parking_aisle"},
}

# One registry entry per city: geocoding name, display name, CRS, and the file
# paths the pipeline reads/writes. Each city's files live under data/{city}/.
# Curitiba, Maringá and Marília lie in UTM 22S; São Paulo in UTM 23S.
CITY_CONFIGS = {
    "curitiba": {
        "geocode": "Curitiba, Brazil",
        "display": "Curitiba",
        "crs": "EPSG:32722",
        "source_segment": "data/curitiba/segment.geojson",
        "source_connector": "data/curitiba/connector.geojson",
        "cache": "data/curitiba/walk_segments.parquet",
        "road_cache": "data/curitiba/road_segments.parquet",
        "boundary": "data/curitiba/boundary.geojson",
    },
    "maringa": {
        "geocode": "Maringá, Paraná, Brazil",
        "display": "Maringá",
        "crs": "EPSG:32722",
        "source_segment": "data/maringa/segment.geojson",
        "source_connector": "data/maringa/connector.geojson",
        "cache": "data/maringa/walk_segments.parquet",
        "road_cache": "data/maringa/road_segments.parquet",
        "boundary": "data/maringa/boundary.geojson",
    },
    "marilia": {
        "geocode": "Marília, São Paulo, Brazil",
        "display": "Marília",
        "crs": "EPSG:32722",
        "source_segment": "data/marilia/segment.geojson",
        "source_connector": "data/marilia/connector.geojson",
        "cache": "data/marilia/walk_segments.parquet",
        "road_cache": "data/marilia/road_segments.parquet",
        "boundary": "data/marilia/boundary.geojson",
    },
    "sao_paulo": {
        "geocode": "São Paulo, São Paulo, Brazil",
        "display": "São Paulo",
        "crs": "EPSG:32723",
        "source_segment": "data/sao_paulo/segment.geojson",
        "source_connector": "data/sao_paulo/connector.geojson",
        "cache": "data/sao_paulo/walk_segments.parquet",
        "road_cache": "data/sao_paulo/road_segments.parquet",
        "boundary": "data/sao_paulo/boundary.geojson",
    },
}


def _fully_denied(value):
    """True if a segment is closed to pedestrians along its whole length.

    Only two cases count as a closure:

    * an unconditional denial (no ``when`` qualifiers at all), or
    * a denial whose ``when.mode`` explicitly covers pedestrians (``foot``,
      ``pedestrian``).

    A directional denial (``heading`` set) is a one-way restriction, which does
    not bind pedestrians, so it is ignored. A denial restricted to other modes
    (``car``, ``bicycle``, ``hgv``, ``motorcycle``, ``bus``) does not close the
    segment to walking. A restriction covering only part of the segment
    (``between`` narrower than [0, 1]) leaves the rest walkable, so it is
    ignored too.
    """
    if not isinstance(value, str) or not value:
        return False
    try:
        records = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return False
    if not isinstance(records, list):
        return False
    for r in records:
        if not isinstance(r, dict) or r.get("access_type") != "denied":
            continue
        when = r.get("when") or {}
        if when.get("heading") is not None:
            continue
        mode = when.get("mode")
        if isinstance(mode, str):
            mode = [mode]  # Overture emits a list; guard against a lone string
        if mode is not None and not (set(mode) & FOOT_MODES):
            continue
        between = r.get("between")
        if between is None or between == [0.0, 1.0]:
            return True
    return False


def cache_key(city="curitiba", network="walk"):
    """Hash of pipeline params + source-file fingerprints; invalidates the cache.

    ``network`` selects which filter produced the segments (``"walk"`` or
    ``"road"``). The walk key is unchanged from before the road network existed;
    the road key adds its own fields so the two caches never share a key.
    """
    cfg = CITY_CONFIGS[city]
    params = {
        "code_version": CODE_VERSION,
        "snap_m": SNAP_M,
        "walk_speed_kmh": WALK_SPEED_KMH,
        "non_walk_classes": sorted(NON_WALK_CLASSES),
        "foot_modes": sorted(FOOT_MODES),
        "drop_subclass": {k: sorted(v) for k, v in DROP_SUBCLASS.items()},
    }
    if network == "road":
        params["network"] = "road"
        params["non_road_classes"] = sorted(NON_ROAD_CLASSES)
    # Fingerprints keyed by field name rather than path, so relocating or
    # renaming the data directory does not invalidate the cache.
    sources = {}
    for name, path in (
        ("segment", cfg["source_segment"]),
        ("connector", cfg["source_connector"]),
    ):
        st = os.stat(path)
        sources[name] = (st.st_size, st.st_mtime_ns)
    blob = json.dumps({"params": params, "sources": sources}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def load_segments(city="curitiba", network="walk"):
    """Return the processed segments GeoDataFrame (cached) for a network.

    ``network="walk"`` is the pedestrian network: every road class except
    motorway/trunk, minus segments closed to foot traffic and the
    sidewalk/crosswalk/driveway/parking subclasses that duplicate a road or are
    private vehicle access. ``network="road"`` is the drivable road network:
    every road class except the pedestrian/cycle-only classes (footway, path,
    cycleway, steps, pedestrian) — so expressways are kept, and no foot-access
    or subclass filter is applied (a gas station is reached by car, driveways
    and parking aisles included).
    """
    cfg = CITY_CONFIGS[city]
    cache = cfg["road_cache"] if network == "road" else cfg["cache"]
    cache_meta = cache + ".meta"
    key = cache_key(city, network)
    if os.path.exists(cache) and os.path.exists(cache_meta):
        with open(cache_meta) as f:
            if f.read().strip() == key:
                print(f"Loading cached processed segments from {cache}")
                return gpd.read_parquet(cache)

    print("Processing raw segments (this may take a minute) ...")
    seg = gpd.read_file(cfg["source_segment"])
    connectors = gpd.read_file(cfg["source_connector"])

    n_raw = len(seg)
    seg = seg[seg["subtype"] == "road"]
    n_road = len(seg)

    non_classes = NON_ROAD_CLASSES if network == "road" else NON_WALK_CLASSES
    seg = seg[~seg["class"].isin(non_classes)]
    n_class = len(seg)

    if network == "walk":
        seg = seg[~seg["access_restrictions"].map(_fully_denied)]
    n_access = len(seg)

    if network == "walk":
        drop = None
        for cls, subs in DROP_SUBCLASS.items():
            m = (seg["class"] == cls) & (seg["subclass"].astype(str).isin(subs))
            drop = m if drop is None else (drop | m)
        n_sub = int(drop.sum()) if drop is not None else 0
        seg = seg[~drop] if drop is not None else seg
        print(
            f"Filtered {n_raw:,} raw segments -> {len(seg):,}: "
            f"-{n_raw - n_road:,} non-road (rail), "
            f"-{n_road - n_class:,} non-walkable class (motorway/trunk), "
            f"-{n_class - n_access:,} closed to pedestrians, "
            f"-{n_sub:,} non-network subclass (sidewalk/crosswalk/driveway/parking)"
        )
    else:
        print(
            f"Filtered {n_raw:,} raw segments -> {len(seg):,}: "
            f"-{n_raw - n_road:,} non-road (rail), "
            f"-{n_road - n_class:,} non-drivable class "
            f"(footway/path/cycleway/steps/pedestrian)"
        )

    seg = seg.to_crs(cfg["crs"])
    connectors = connectors.to_crs(cfg["crs"])

    seg = c2g.process_overture_segments(
        seg, connectors_gdf=connectors, threshold=SNAP_M, get_barriers=False
    )
    seg = seg[seg.geometry.notna() & ~seg.geometry.is_empty].copy()

    seg["length_m"] = seg.geometry.length
    seg["speed_kmh"] = WALK_SPEED_KMH
    seg["travel_time_s"] = seg["length_m"] / (WALK_SPEED_KMH / 3.6)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    seg.to_parquet(cache)
    with open(cache_meta, "w") as f:
        f.write(key)
    print(f"Cached processed segments to {cache} ({len(seg):,} segments)")
    return seg


def build_graph(seg):
    """Return (nodes, edges) GeoDataFrames for the walking network."""
    return c2g.segments_to_graph(seg, directed=False)


def load_boundary(city="curitiba"):
    """Municipal boundary (projected), cached to avoid repeat geocoding."""
    cfg = CITY_CONFIGS[city]
    path = cfg["boundary"]
    if os.path.exists(path):
        return gpd.read_file(path).to_crs(cfg["crs"])
    boundary = c2g.get_boundaries(cfg["geocode"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    boundary.to_file(path, driver="GeoJSON")
    return boundary.to_crs(cfg["crs"])


def parse_city(default="curitiba"):
    """Parse ``--city`` from the command line; shared by the entry-point scripts."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--city", default=default, choices=list(CITY_CONFIGS))
    return parser.parse_known_args()[0].city
