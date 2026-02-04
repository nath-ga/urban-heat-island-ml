import sys
from pathlib import Path

import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import geopandas as gpd
except Exception as e:
    print("GeoPandas import FAILED:", e)
    sys.exit(1)

# Shapely check (GeoPandas depends on it)
try:
    import shapely  # noqa: F401
    shapely_ok = True
except Exception:
    shapely_ok = False

print("Python:", sys.version.split()[0])
print("GeoPandas:", gpd.__version__)
print("Shapely available:", shapely_ok)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

BRAZIL_CSV = RAW_DIR / "Sample_Brazil_uhi_data.csv"
BRAZIL_BUILDINGS_FOLDER = RAW_DIR / "Brazil Building Footprints"

print("\nBrazil CSV:", BRAZIL_CSV)
print("Exists:", BRAZIL_CSV.exists())

print("\nBuildings folder:", BRAZIL_BUILDINGS_FOLDER)
print("Exists:", BRAZIL_BUILDINGS_FOLDER.exists())

# --- Load CSV ---
if BRAZIL_CSV.exists():
    df = pd.read_csv(BRAZIL_CSV)
    print("\nBrazil CSV loaded:", df.shape)
    print("Columns:", list(df.columns))
    if "UHI_Class" in df.columns:
        print("UHI_Class values:", df["UHI_Class"].value_counts().to_dict())
else:
    print("\nBrazil CSV not found. Please check filename/path.")
    sys.exit(1)

# --- Find & load one shapefile ---
if BRAZIL_BUILDINGS_FOLDER.exists():
    shp_files = list(BRAZIL_BUILDINGS_FOLDER.rglob("*.shp"))
    print("\nShapefiles found:", len(shp_files))
    if not shp_files:
        print("No .shp files found inside buildings folder.")
        sys.exit(1)

    shp_path = shp_files[0]
    print("Loading shapefile:", shp_path)

    gdf = gpd.read_file(shp_path)
    print("Buildings loaded:", gdf.shape)
    print("CRS:", gdf.crs)
    print("Columns (first 40):", list(gdf.columns)[:40])

    # Guess a height column name
    candidates = [c for c in gdf.columns if "height" in c.lower() or "hgt" in c.lower() or "elev" in c.lower()]
    print("Possible height columns:", candidates)
else:
    print("\nBuildings folder not found. Please check folder name/path.")
    sys.exit(1)
