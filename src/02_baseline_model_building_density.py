from pathlib import Path
import time

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- PATHS (Windows) ---
BRAZIL_CSV = Path(r"C:\Users\Nathalie\neue projekte\urban-heat-island-ml\data\raw\Sample_Brazil_uhi_data.csv")
BRAZIL_BUILDINGS_FOLDER = Path(r"C:\Users\Nathalie\neue projekte\urban-heat-island-ml\data\raw\Brazil Building Footprints")

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

    # Project to meters
    points_m = points.to_crs(epsg=3857)
    buildings_m = buildings_gdf.to_crs(epsg=3857).copy()

    # Building area in m² (because CRS is meters)
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

        # Count buildings per point
        counts = joined.groupby("index_right").size()

        # Sum building areas per point
        area_sum = joined.groupby("index_right")["bldg_area_m2"].sum()

        out[f"bldg_count_{r}m"] = 0
        out.loc[counts.index, f"bldg_count_{r}m"] = counts.values

        out[f"bldg_area_m2_{r}m"] = 0.0
        out.loc[area_sum.index, f"bldg_area_m2_{r}m"] = area_sum.values

    return out


def main() -> None:
    t0 = time.time()

    # 1) Load data
    df = pd.read_csv(BRAZIL_CSV)
    if "UHI_Class" not in df.columns:
        raise ValueError("Expected column 'UHI_Class' in training data.")

    lat_col, lon_col = get_lat_lon_columns(df)

    # 2) Sample points
    df_s = df.sample(n=min(N_POINTS, len(df)), random_state=RANDOM_STATE).copy()

    # 3) Load buildings
    shp_path = find_first_shapefile(BRAZIL_BUILDINGS_FOLDER)
    buildings = gpd.read_file(shp_path)

    # 4) Build features
    df_feat = add_building_features(df_s, buildings, lat_col, lon_col, RADII_M)

    feature_cols = []
    for r in RADII_M:
        feature_cols.append(f"bldg_count_{r}m")
        feature_cols.append(f"bldg_area_m2_{r}m")

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

    # 6) Baseline model
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

    # 8) Save artifacts
    out_dir = Path("outputs") / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = "_".join([f"{r}m" for r in RADII_M]) + "_count_area"
    df_feat.to_csv(out_dir / f"brazil_sample_{len(df_s)}_bldgcount_{tag}.csv", index=False)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in le.classes_],
        columns=[f"pred_{c}" for c in le.classes_],
    )
    cm_df.to_csv(out_dir / f"confusion_matrix_{len(df_s)}_bldgcount_{tag}.csv", index=True)

    print(f"\nSaved feature sample + confusion matrix to {out_dir}")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
