"""Walkability analysis: reachable walkable network length within 10 minutes.

For every junction in a city's walkable network, ``reachable_length(v)`` is the
total walkable length inside its 10-minute isochrone. It reads as "how much
network can you actually reach from here on foot" and peaks in dense,
well-connected, fine-grained districts.

Pipeline: ``compute_reachable_length.py`` -> ``plot_reachable_length.py``
(reference-checked by ``test_reachable_length.py``).
"""
