from pathlib import Path
import pandas as pd
import geopandas as gpd

# --- PATHS ---
BRAZIL_CSV = Path(r"C:\Users\Nathalie\neue projekte\urban-heat-island-ml\data\raw\Sample_Brazil_uhi_data.csv")
BRAZIL_BUILDINGS_FOLDER = Path(r"C:\Users\Nathalie\neue projekte\urban-heat-island-ml\data\raw\Brazil Building Footprints")

# --- SETTINGS ---
N_POINTS = 200          # keep small for first run
RADIUS_M = 100          # buffer radius in meters

def find_first_shapefile(folder: Path) -> Path:
    shp_files = list(folder.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp files found in {folder}")
    return shp_files[0]

def main() -> None:
    # 1) Load training points
    df = pd.read_csv(BRAZIL_CSV)
    # Adjust column names if needed
    lat_col = "Latitude" if "Latitude" in df.columns else "latitude"
    lon_col = "Longitude" if "Longitude" in df.columns else "longitude"

    df_small = df.head(N_POINTS).copy()

    points = gpd.GeoDataFrame(
        df_small,
        geometry=gpd.points_from_xy(df_small[lon_col], df_small[lat_col]),
        crs="EPSG:4326",
    )

    # 2) Load building footprints
    shp_path = find_first_shapefile(BRAZIL_BUILDINGS_FOLDER)
    buildings = gpd.read_file(shp_path)

    # buildings is EPSG:4326 (from your output). For metric buffers we must project.
    # Use Web Mercator for a first baseline (good enough for small radii).
    points_m = points.to_crs(epsg=3857)
    buildings_m = buildings.to_crs(epsg=3857)

    # 3) Buffer points and count buildings in each buffer
    buffers = points_m.copy()
    buffers["geometry"] = points_m.geometry.buffer(RADIUS_M)

    # spatial join: which buildings fall inside which buffer polygon
    joined = gpd.sjoin(buildings_m, buffers[["geometry"]], predicate="within", how="inner")

    # joined has index_right = row index of buffer/point
    counts = joined.groupby("index_right").size()

    # 4) Attach counts back to points (default 0)
    points_m["bldg_count_100m"] = 0
    points_m.loc[counts.index, "bldg_count_100m"] = counts.values

    # 5) Return to EPSG:4326 (optional) and save a tiny output table
    out = points_m.drop(columns="geometry").copy()
    out_path = Path("outputs") / "tables"
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"brazil_points_{N_POINTS}_bldgcount_{RADIUS_M}m.csv"
    out.to_csv(out_file, index=False)

    print("Saved:", out_file)
    print(out[["UHI_Class", "bldg_count_100m"]].head(10))
    print("Counts summary:", out["bldg_count_100m"].describe().to_dict())

    # --- A) Mean building count per UHI class ---
    summary = (
        out.groupby("UHI_Class")["bldg_count_100m"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    print("\nBuilding density summary by UHI class (100m radius):")
    print(summary)

if __name__ == "__main__":
    main()
