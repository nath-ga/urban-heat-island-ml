from pathlib import Path
import pandas as pd
import geopandas as gpd

import osmnx as ox
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

BRAZIL_CSV = RAW_DIR / "Sample_Brazil_uhi_data.csv"
OUT_GPKG = PROCESSED_DIR / "osm_roads_brazil_bbox.gpkg"

def get_lat_lon_columns(df: pd.DataFrame) -> tuple[str, str]:
    lat_candidates = ["Latitude", "latitude", "LAT", "lat"]
    lon_candidates = ["Longitude", "longitude", "LON", "lon"]
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    if not lat_col or not lon_col:
        raise ValueError(f"Could not find lat/lon columns. Found columns: {list(df.columns)}")
    return lat_col, lon_col


def main() -> None:
    df = pd.read_csv(BRAZIL_CSV)
    lat_col, lon_col = get_lat_lon_columns(df)

    # Bounding box from all points (or use a sample if file is huge)
    north = df[lat_col].max()
    south = df[lat_col].min()
    east = df[lon_col].max()
    west = df[lon_col].min()

    # Add a small margin (in degrees) so we don't clip edges
    margin = 0.01
    north += margin
    south -= margin
    east += margin
    west -= margin

    print("Downloading OSM roads for bbox:")
    print("north, south, east, west =", north, south, east, west)

    # Build graph and convert to edges GeoDataFrame
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # Keep only geometry + a few useful columns if present
    keep_cols = [c for c in ["geometry", "highway", "name", "length"] if c in edges.columns]
    edges = edges[keep_cols].copy()

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    edges.to_file(OUT_GPKG, layer="roads", driver="GPKG")
    print("Saved:", OUT_GPKG)


if __name__ == "__main__":
    main()
