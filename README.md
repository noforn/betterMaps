# betterMaps

Generate interactive geographic opportunity maps for advisors.

## Usage

```bash
git clone https://github.com/noforn/betterMaps.git
cd betterMaps
bash samMap.sh
```
## Configuration

Modify the `CONFIG` dict in `betterSam.py` to adjust clustering, geocoding, and map settings.

## Features

- Load advisor and location data from CSV files
- Geocode addresses with SQLite-based caching for speed
- Cluster advisors using K‑Means for territory insights
- Produce interactive HTML maps with Folium and MarkerCluster
- Easily customize via the CONFIG dictionary (rate limits, clusters, map view)