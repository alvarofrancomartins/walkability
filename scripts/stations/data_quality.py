"""Data-quality audit for the ANP gas-station registry.

Runs the checks that matter for the "distance to the nearest gas station" map
and for the "are all of these truly car-fuel stations?" question, prints a
report, and writes two files for auditability:

* ``data/all_gas_stations_cleaned.csv``  -- rows the cleaning rules keep.
* ``data/all_gas_stations_dropped.csv``  -- rows dropped, with a ``reason`` column.

Run:
    .venv/bin/python scripts/stations/data_quality.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stations.cleaning import clean_stations  # noqa: E402

STATIONS_XLSX = "data/all_gas_stations.xlsx"
CLEANED_CSV = "data/all_gas_stations_cleaned.csv"
DROPPED_CSV = "data/all_gas_stations_dropped.csv"

# Brazil's land bbox, widened for the oceanic islands (Fernando de Noronha etc.).
LAT_MIN, LAT_MAX = -33.80, 5.30
LON_MIN, LON_MAX = -74.00, -32.00


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    df = pd.read_excel(STATIONS_XLSX, sheet_name="Sheet1")
    print(f"File: {STATIONS_XLSX}  ({len(df):,} rows x {df.shape[1]} cols)")

    section("1. Missing / blank values per column")
    for c in df.columns:
        if df[c].dtype == object:
            n = df[c].isna().sum() + (df[c].astype(str).str.strip() == "").sum()
        else:
            n = df[c].isna().sum()
        flag = "  <-- used by the map" if c in ("Latitude", "Longitude", "UF") else ""
        print(f"  {c:36s} {n:7d} ({100 * n / len(df):.2f}%){flag}")

    section("2. Latitude / Longitude validity")
    lat, lon = df["Latitude"], df["Longitude"]
    has = lat.notna() & lon.notna()
    print(f"  rows with coordinates: {has.sum():,}")
    print(f"  zero lat/lon: {int((lat == 0).sum())} / {int((lon == 0).sum())}")
    in_br = has & lat.between(LAT_MIN, LAT_MAX) & lon.between(LON_MIN, LON_MAX)
    print(f"  inside Brazil bbox: {in_br.sum():,}")
    print(f"  outside Brazil bbox: {(has & ~in_br).sum():,} (oceanic islands expected)")
    for nm, s in (("lat", lat), ("lon", lon)):
        dec = s.dropna().astype(str).str.split(".").str[1].str.len()
        print(f"  {nm} decimal places: {dec.value_counts().sort_index().to_dict()}")

    section("3. Exact shared coordinates (geocoding / centroid artifacts)")
    key = df["Latitude"].round(6).astype(str) + "," + df["Longitude"].round(6).astype(str)
    vc = key.value_counts()
    clusters = vc[vc >= 2]
    print(f"  distinct coordinate pairs: {key[has].nunique():,} of {has.sum():,} rows")
    print(f"  pairs shared by >=2 stations: {len(clusters):,} ({clusters.sum():,} rows)")
    print(f"  pairs shared by >=5 stations (bogus): {(vc >= 5).sum():,} ({vc[vc >= 5].sum():,} rows)")
    print("\n  clusters shared by >=3 stations:")
    for k, c in clusters[clusters >= 3].items():
        rows = df[key == k]
        n_uf = rows["UF"].nunique()
        n_mun = rows["MUNICÍPIO"].nunique()
        print(f"    {k}  x{c}  distinct UF={n_uf}  distinct MUNICÍPIO={n_mun}")

    section("4. Identifier / key-column integrity")
    cnpj = df["CNPJ"].astype(str)
    print(f"  CNPJ (stored as int, leading zeros lost): "
          f"{cnpj.str.len().value_counts().sort_index().to_dict()}")
    cep = df["CEP"].astype(str)
    print(f"  CEP  (stored as int, leading zeros lost): "
          f"{cep.str.len().value_counts().sort_index().to_dict()}")
    print(f"  'Código Instalação i-Simp' unique: {df['Código Instalação i-Simp'].nunique():,} / {len(df):,}")
    print(f"  'Nº Autorizacao' duplicated numbers: "
          f"{(df['Nº Autorizacao'].value_counts() > 1).sum():,} "
          f"({df['Nº Autorizacao'].value_counts()[df['Nº Autorizacao'].value_counts() > 1].sum():,} rows)")

    section("5. Categorical anomalies")
    print("  Origem (coordinate source):")
    print(df["Origem"].value_counts(dropna=False).to_string().replace("\n", "\n  "))
    print("\n  'Vinculação a Distribuidor' = 'SP' (not a distributor):", (df["Vinculação a Distribuidor"] == "SP").sum())
    sp = df[df["Vinculação a Distribuidor"] == "SP"]
    print("    UFs of those rows:", sp["UF"].value_counts().to_dict())
    print(f"  Status PMQC values: {df['Status PMQC'].unique().tolist()}")
    print(f"  Delivery = SIM: {(df['Delivery'] == 'SIM').sum():,}")

    section("6. Non-automotive stations (the 'truly gas?' question)")
    kept, dropped = clean_stations(df)
    print(f"  kept: {len(kept):,}   dropped: {len(dropped):,}")
    print("  dropped by reason:")
    for reason, n in dropped["reason"].value_counts().items():
        print(f"    {reason:28s} {n:7,d}")
    pr = dropped[dropped["UF"] == "PR"]
    print(f"\n  dropped in Paraná: {len(pr):,} (by reason):")
    print(pr["reason"].value_counts().to_string().replace("\n", "\n  "))
    interesting = pr[pr["reason"] != "missing_coordinates"]
    print(f"\n  Paraná drops other than missing coordinates: {len(interesting):,}")
    print(interesting[["MUNICÍPIO", "Razão Social", "reason"]].to_string())

    section("7. Write cleaned / dropped lists")
    kept.to_csv(CLEANED_CSV, index=False)
    dropped.to_csv(DROPPED_CSV, index=False)
    print(f"  wrote {CLEANED_CSV} ({len(kept):,} rows)")
    print(f"  wrote {DROPPED_CSV} ({len(dropped):,} rows)")


if __name__ == "__main__":
    main()
