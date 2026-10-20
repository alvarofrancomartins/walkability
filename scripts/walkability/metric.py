"""The walkability metric's definition: a 10-minute reachable-length isochrone.

Holds the constants that define the reachable-network-length metric and the
cache key that ties the computed ``reachable_length.npy`` to those constants
plus the segment cache. The compute and plot scripts both import from here so
they always agree on what "the metric" is.

The metric itself: for a junction ``v``, sum the length of the walkable network
inside ``v``'s 10-minute isochrone, crediting each edge for the fraction of its
length that lies inside (the partial-edge rule, so digitization detail cannot
inflate the value).
"""

import hashlib
import json

from common.network import cache_key

# Travel-time budget for the reachable-network-length metric (seconds). At the
# single modeled walking speed (common.network.WALK_SPEED_KMH) this is ~750 m of
# travel. Shared with the plot script, which renders it in minutes.
CUTOFF_S = 600.0

# Bump when the metric's *definition* changes — e.g. switching between an
# all-or-nothing edge rule and partial-edge credit — even though the segments
# and their travel times are unchanged. The segment cache keys off
# CODE_VERSION (in common.network); the .npy cache keys off this, so a bump here
# rebuilds the metric on its own.
METRIC_VERSION = 1


def reach_cache_key(city="curitiba"):
    """Hash of the metric's parameters + segment-cache key; the .npy cache key.

    Shared by the compute and plot scripts so the plot can refuse a stale
    ``reachable_length.npy`` instead of silently pairing fresh labels with old
    values.
    """
    return hashlib.sha1(
        json.dumps(
            {
                "metric_version": METRIC_VERSION,
                "cutoff": CUTOFF_S,
                "segments": cache_key(city),
            }
        ).encode()
    ).hexdigest()[:12]
