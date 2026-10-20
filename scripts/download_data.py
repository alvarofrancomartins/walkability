"""Download Overture segments + connectors for a city and a surrounding buffer.

The buffer ensures the 10-minute walking reach (~750 m) of border nodes is not
truncated at the municipal boundary. Writes data/<city>/segment.geojson and
data/<city>/connector.geojson. Choose the city with ``--city``.

Run with the venv on PATH so the ``overturemaps`` CLI is found:
    PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/download_data.py --city maringa
"""

import argparse
import os
import sys

import city2graph as c2g

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.network import CITY_CONFIGS, load_boundary  # noqa: E402

BUFFER_M = 2_000  # 2 km past the municipal boundary (>= ~750 m 10-min walk reach)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="curitiba", choices=list(CITY_CONFIGS))
    city = parser.parse_args().city
    cfg = CITY_CONFIGS[city]

    boundary = load_boundary(city)  # projected; cached if already computed
    buffered = boundary.buffer(BUFFER_M).to_crs("EPSG:4326")
    area = buffered.iloc[0]
    print(f"[{cfg['display']}] Buffered area bbox: {area.bounds}")

    c2g.load_overture_data(
        area=area,
        types=["segment", "connector"],
        output_dir=f"data/{city}",
        prefix="",
        save_to_file=True,
        return_data=False,
    )
    print(f"[{cfg['display']}] Download complete.")


if __name__ == "__main__":
    main()
