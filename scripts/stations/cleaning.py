"""Cleaning rules for the ANP gas-station registry.

The registry is the ANP *retail automotive fuel* dataset, but it still contains
a small number of entries that are not street-corner car-fuel stations, and a
larger number with missing or clearly bogus coordinates. This module encodes the
rules that decide which rows are usable for the "distance to the nearest car-fuel
station" map.

``clean_stations(df)`` returns ``(kept, dropped)``; ``dropped`` carries a
``reason`` column so every exclusion is auditable. The rules are deliberately
conservative: they only drop a row when the evidence is strong.

Bump ``CLEAN_VERSION`` whenever a rule changes, so the cached metric is
recomputed instead of silently reused.
"""

import pandas as pd

# Bump when any rule below changes; keyed into the station metric cache.
CLEAN_VERSION = 1

# A coordinate shared by this many or more stations is a geocoding/centroid
# artifact. Real stations are never within a metre of each other, so >=5 stations
# on one exact (6-decimal) point is a bogus default coordinate or a municipality
# centroid. This catches the nationwide default at (-22.778, -43.337) (148
# stations across 21 states) and the five-way municipal-centroid cluster in
# western Paraná.
BOGUS_COORD_MIN = 5

# Razão Social tokens that mark an aviation-fuel supplier or aviation service:
# these sell avgas/Jet-A to aircraft, not street fuel to cars. Matched after the
# car-posto guard below, so "Auto Posto Aviação"/"Aeroposto" street stations
# survive.
_AVIATION_NAME = (
    r"AERONAVE|AERON[ÁA]UTIC|AVIA[ÇC][ÃA]O|T[ÁA]XI A[ÉE]REO|TAXI A[ÉE]REO"
    r"|JETFLY|JET SERVICE|HELIPARK|FLYWAY|AEROFOTOGR?A|AEROPREST|AEROTANGARA"
    r"|SKY JET|COMBUST[IÍ]VE[IÍ]S? DE AVIA|COMBUST[IÍ]VEL DE AVIA"
    r"|PRODUTOS?.{0,20}AVIA|SERVI[ÇC]OS?.{0,20}A[ÉE]REO|SERVI[ÇC]OS AEROPORTU"
)

# Street postos that happen to be *named* after an airport or aviation (e.g.
# "Auto Posto Aeroporto", "Aeroposto"). These ARE normal car stations, so the
# aviation-name rule must not drop them.
_CAR_POSTO = r"AUTO ?POSTO|AUTOPOSTO|AEROPOSTO|AERO ?POSTO"

_FLOATING = r"\bFLUTUANTE\b"


def _first_match(mask, label, reasons):
    """Assign ``label`` to rows where ``mask`` holds and no label is set yet."""
    reasons.loc[mask & (reasons == "")] = label


def clean_stations(df):
    """Apply the cleaning rules; return ``(kept, dropped)`` with drop reasons.

    ``df`` is the raw registry as read from the spreadsheet (WGS 84 lat/lon, all
    original columns). The rules are, in order (first match wins):

    1. ``missing_coordinates``      -- no Latitude or Longitude.
    2. ``aviation_fuel_air_bp``     -- distributor is AIR BP (BP's aviation arm).
    3. ``floating``                 -- "FLUTUANTE" in the name (boat fuel).
    4. ``aviation_name``            -- aviation-fuel/service tokens, unless it is
                                       a street posto named after an airport.
    5. ``bogus_shared_coordinate``  -- coordinate shared by >=BOGUS_COORD_MIN
                                       stations (geocoding/centroid artifact).
    """
    reasons = pd.Series("", index=df.index, dtype=object)

    _first_match(df["Latitude"].isna() | df["Longitude"].isna(), "missing_coordinates", reasons)

    _first_match(df["Vinculação a Distribuidor"] == "AIR BP", "aviation_fuel_air_bp", reasons)

    name = df["Razão Social"].astype(str).str.upper()
    _first_match(name.str.contains(_FLOATING, regex=True, na=False), "floating", reasons)

    is_aviation_name = name.str.contains(_AVIATION_NAME, regex=True, na=False)
    is_car_posto = name.str.contains(_CAR_POSTO, regex=True, na=False)
    _first_match(is_aviation_name & ~is_car_posto, "aviation_name", reasons)

    key = df["Latitude"].round(6).astype(str) + "," + df["Longitude"].round(6).astype(str)
    shared = key.value_counts()
    bogus = df["Latitude"].notna() & key.isin(shared[shared >= BOGUS_COORD_MIN].index)
    _first_match(bogus, "bogus_shared_coordinate", reasons)

    dropped = df[reasons != ""].copy()
    dropped["reason"] = reasons[reasons != ""]
    kept = df[reasons == ""].copy()
    return kept, dropped


def cleaning_summary(dropped, label=""):
    """One-line count of kept/dropped and a per-reason breakdown, for logging."""
    lines = [f"{label} dropped {len(dropped):,} stations:"]
    for reason, n in dropped["reason"].value_counts().items():
        lines.append(f"    - {reason}: {n:,}")
    return "\n".join(lines)
