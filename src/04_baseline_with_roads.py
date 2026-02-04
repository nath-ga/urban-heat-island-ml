from pathlib import Path
import time

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- PATHS (Windows) ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

BRAZIL_CSV = RAW_DIR / "Sample_Brazil_uhi_data.csv"
BRAZIL_BUILDINGS_FOLDER = RAW_DIR / "Brazil Building Footprints"

# created by 03_fetch_roads_osm.py
ROADS_GPKG = Path("data") / "processed" / "osm_roads_brazil_bbox.gpkg"
ROADS_LAYER = "roads"

# --- SETTINGS ---
N_POINTS = 3000
RADII_M = [50, 100, 200]
RANDOM_STATE = 42


def find_first_shapefile(folder: Path) -> Path:
    shp_files = list(folder.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp files found in {folder}")
    return shp_files[0]


def get_lat_lon_columns(df: pd.DataFrame) -> tuple[str, str]:
    lat_candidates = ["Latitude", "latitude", "LAT", "lat"]
    lon_candidates = ["Longitude", "longitude", "LON", "lon"]

    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)

    if not lat_col or not lon_col:
        raise ValueError(f"Could not find lat/lon columns. Found columns: {list(df.columns)}")
    return lat_col, lon_col


def add_building_features(
    df_points: pd.DataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    radii_m: list[int],
) -> pd.DataFrame:
    """Compute building count and total building area (m²) within each radius around each point."""
    points = gpd.GeoDataFrame(
        df_points.copy(),
        geometry=gpd.points_from_xy(df_points[lon_col], df_points[lat_col]),
        crs="EPSG:4326",
    )

    points_m = points.to_crs(epsg=3857)
    buildings_m = buildings_gdf.to_crs(epsg=3857).copy()
    buildings_m["bldg_area_m2"] = buildings_m.geometry.area

    out = pd.DataFrame(points_m.drop(columns="geometry"))

    for r in radii_m:
        buffers = points_m[["geometry"]].copy()
        buffers["geometry"] = points_m.geometry.buffer(r)

        joined = gpd.sjoin(
            buildings_m[["geometry", "bldg_area_m2"]],
            buffers,
            predicate="within",
            how="inner",
        )

        counts = joined.groupby("index_right").size()
        area_sum = joined.groupby("index_right")["bldg_area_m2"].sum()

        out[f"bldg_count_{r}m"] = 0
        out.loc[counts.index, f"bldg_count_{r}m"] = counts.values

        out[f"bldg_area_m2_{r}m"] = 0.0
        out.loc[area_sum.index, f"bldg_area_m2_{r}m"] = area_sum.values

    return out


def add_road_length_features_mvp(
    df_points: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    radii_m: list[int],
) -> pd.DataFrame:
    """
    MVP: Sum full lengths of road segments that intersect each buffer.
    Faster but overcounts when long segments barely touch the buffer.
    """
    points = gpd.GeoDataFrame(
        df_points.copy(),
        geometry=gpd.points_from_xy(df_points[lon_col], df_points[lat_col]),
        crs="EPSG:4326",
    )

    points_m = points.to_crs(epsg=3857)
    roads_m = roads_gdf.to_crs(epsg=3857).copy()
    roads_m["road_len_m"] = roads_m.geometry.length

    out = pd.DataFrame(points_m.drop(columns="geometry"))

    for r in radii_m:
        buffers = points_m[["geometry"]].copy()
        buffers["geometry"] = points_m.geometry.buffer(r)

        joined = gpd.sjoin(
            roads_m[["geometry", "road_len_m"]],
            buffers,
            predicate="intersects",
            how="inner",
        )

        road_sum = joined.groupby("index_right")["road_len_m"].sum()

        col = f"road_len_m_{r}m"
        out[col] = 0.0
        out.loc[road_sum.index, col] = road_sum.values

    return out

def add_road_length_200m_clipped(
    df_points: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
) -> pd.Series:
    """
    Compute clipped road length (meters) within 200m buffer for each point.
    """
    points = gpd.GeoDataFrame(
        df_points.copy(),
        geometry=gpd.points_from_xy(df_points[lon_col], df_points[lat_col]),
        crs="EPSG:4326",
    )

    points_m = points.to_crs(epsg=3857)
    roads_m = roads_gdf.to_crs(epsg=3857)

    buffers = gpd.GeoDataFrame(
        index=points_m.index,
        geometry=points_m.geometry.buffer(200),
        crs=points_m.crs,
    )

    joined = gpd.sjoin(
        roads_m[["geometry"]],
        buffers[["geometry"]],
        predicate="intersects",
        how="inner",
    )

    joined = joined.join(
        buffers.rename(columns={"geometry": "buffer_geom"}),
        on="index_right",
    )

    inter = joined.geometry.intersection(joined["buffer_geom"])
    inter_len = inter.length

    road_sum = inter_len.groupby(joined["index_right"]).sum()

    # return as Series aligned to points
    out = pd.Series(0.0, index=points_m.index, name="road_len_m_200m_clipped")
    out.loc[road_sum.index] = road_sum.values
    return out

def main() -> None:
    t0 = time.time()

    # 1) Load training points
    df = pd.read_csv(BRAZIL_CSV)
    if "UHI_Class" not in df.columns:
        raise ValueError("Expected column 'UHI_Class' in training data.")
    lat_col, lon_col = get_lat_lon_columns(df)

    # 2) Sample points
    df_s = df.sample(n=min(N_POINTS, len(df)), random_state=RANDOM_STATE).copy()

    # 3) Load buildings + roads
    shp_path = find_first_shapefile(BRAZIL_BUILDINGS_FOLDER)
    buildings = gpd.read_file(shp_path)

    if not ROADS_GPKG.exists():
        raise FileNotFoundError(f"Roads GeoPackage not found: {ROADS_GPKG}")

    roads = gpd.read_file(ROADS_GPKG, layer=ROADS_LAYER)

    # 4) Build features
    df_bldg = add_building_features(df_s, buildings, lat_col, lon_col, RADII_M)
    df_roads = add_road_length_features_mvp(df_s, roads, lat_col, lon_col, RADII_M)

    road_200m_clipped = add_road_length_200m_clipped(
        df_s, roads, lat_col, lon_col
    )

    df_roads["road_len_m_200m"] = road_200m_clipped.values

    # Join by index (same sampled rows)
    road_cols = [f"road_len_m_{r}m" for r in RADII_M]
    df_feat = df_bldg.join(df_roads[road_cols])

    # Feature columns: building count + building area + road length per radius
    feature_cols: list[str] = []
    for r in RADII_M:
        feature_cols.append(f"bldg_count_{r}m")
        feature_cols.append(f"bldg_area_m2_{r}m")
        feature_cols.append(f"road_len_m_{r}m")

    print("Using features:", feature_cols)
    print("Columns available:", list(df_feat.columns))

    # 5) Prepare ML arrays
    X = df_feat[feature_cols].astype(float).values
    y_text = df_feat["UHI_Class"].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    class_counts = pd.Series(y_text).value_counts().to_dict()
    print("Class distribution in sample:", class_counts)
    print("Classes:", list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # 6) Model (scaled logistic regression baseline)
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # 7) Evaluation
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # 8) Save artifacts (these are ignored by git, that's fine)
    out_dir = Path("outputs") / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = "bldgcount_bldgarea_roadlen_" + "_".join([f"{r}m" for r in RADII_M])
    df_feat.to_csv(out_dir / f"brazil_sample_{len(df_s)}_{tag}.csv", index=False)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in le.classes_],
        columns=[f"pred_{c}" for c in le.classes_],
    )
    cm_df.to_csv(out_dir / f"confusion_matrix_{len(df_s)}_{tag}.csv", index=True)

    print(f"\nSaved feature sample + confusion matrix to {out_dir}")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
