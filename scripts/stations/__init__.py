"""Gas-station accessibility analysis: distance to the nearest gas station.

For every junction in the drivable road network of the whole state of Paraná,
``station_distance(v)`` is the shortest distance along the roads to the nearest
gas station (snapped to its nearest network node). It reads as "how far is the
nearest posto de gasolina from here, along the roads". The pipeline is
``parana.py`` / ``compute_parana_distance.py`` / ``plot_parana_distance.py``.
"""
