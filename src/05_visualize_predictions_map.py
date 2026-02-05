from pathlib import Path
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import folium

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# --- Project paths (relative to repo root) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

BRAZIL_CSV = RAW_DIR / "Sample_Brazil_uhi_data.csv"
BRAZIL_BUILDINGS_FOLDER = RAW_DIR / "Brazil Building Footprints"
ROADS_GPKG = PROCESSED_DIR / "osm_roads_brazil_bbox.gpkg"
ROADS_LAYER = "roads"


# --- SETTINGS ---
N_POINTS = 3000
RADII_M = [50, 100, 200]
RANDOM_STATE = 42
TEST_SIZE = 0.25


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
    """MVP: sum full lengths of road segments that intersect each buffer."""
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
        buffers = gpd.GeoDataFrame(
            index=points_m.index,
            geometry=points_m.geometry.buffer(r),
            crs=points_m.crs,
        )

        joined = gpd.sjoin(
            roads_m[["geometry", "road_len_m"]],
            buffers[["geometry"]],
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
    """Clipped: sum only road length inside 200m buffer."""
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

    out = pd.Series(0.0, index=points_m.index, name="road_len_m_200m")
    out.loc[road_sum.index] = road_sum.values
    return out


def make_map_html(df_pred: pd.DataFrame, out_html: Path) -> None:
    """Create a Folium map colored by predicted class, with True vs Pred popups."""
    center_lat = float(df_pred["Latitude"].mean())
    center_lon = float(df_pred["Longitude"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    # colors for predicted class
    color_map = {"Low": "blue", "Medium": "orange", "High": "red"}

    for _, r in df_pred.iterrows():
        pred = r["pred_class"]
        true = r["UHI_Class"]
        folium.CircleMarker(
            location=[float(r["Latitude"]), float(r["Longitude"])],
            radius=3,
            color=color_map.get(pred, "gray"),
            fill=True,
            fill_opacity=0.75,
            popup=f"True: {true} | Pred: {pred}",
        ).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(out_html)
    print("Saved map:", out_html)


def main() -> None:
    t0 = time.time()

    if not BRAZIL_CSV.exists():
        raise FileNotFoundError(f"Missing: {BRAZIL_CSV}")
    if not BRAZIL_BUILDINGS_FOLDER.exists():
        raise FileNotFoundError(f"Missing: {BRAZIL_BUILDINGS_FOLDER}")
    if not ROADS_GPKG.exists():
        raise FileNotFoundError(f"Missing: {ROADS_GPKG} (run src/03_fetch_roads_osm.py first)")

    df = pd.read_csv(BRAZIL_CSV)
    if "UHI_Class" not in df.columns:
        raise ValueError("Expected column 'UHI_Class' in training data.")

    lat_col, lon_col = get_lat_lon_columns(df)

    # sample for speed + reproducibility
    df_s = df.sample(n=min(N_POINTS, len(df)), random_state=RANDOM_STATE).copy()

    # load geometries
    shp_path = find_first_shapefile(BRAZIL_BUILDINGS_FOLDER)
    buildings = gpd.read_file(shp_path)
    roads = gpd.read_file(ROADS_GPKG, layer=ROADS_LAYER)

    # features
    df_bldg = add_building_features(df_s, buildings, lat_col, lon_col, RADII_M)
    df_roads = add_road_length_features_mvp(df_s, roads, lat_col, lon_col, RADII_M)

    # replace 200m with clipped version
    df_roads["road_len_m_200m"] = add_road_length_200m_clipped(df_s, roads, lat_col, lon_col).values

    road_cols = [f"road_len_m_{r}m" for r in RADII_M]
    df_feat = df_bldg.join(df_roads[road_cols])

    feature_cols: list[str] = []
    for r in RADII_M:
        feature_cols.append(f"bldg_count_{r}m")
        feature_cols.append(f"bldg_area_m2_{r}m")
        feature_cols.append(f"road_len_m_{r}m")

    X = df_feat[feature_cols].astype(float).values
    y_text = df_feat["UHI_Class"].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    # split by index so we can map predictions back to coordinates
    idx = df_feat.index.to_numpy()
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    X_train = df_feat.loc[idx_train, feature_cols].astype(float).values
    X_test = df_feat.loc[idx_test, feature_cols].astype(float).values

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    pred_class = le.inverse_transform(y_pred)

    # prediction table for mapping
    df_pred = df_feat.loc[idx_test, ["Longitude", "Latitude", "UHI_Class"]].copy()
    df_pred["pred_class"] = pred_class

    # save predictions table (optional)
    out_tables = OUTPUTS_DIR / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    pred_csv = out_tables / "predictions_testset.csv"
    df_pred.to_csv(pred_csv, index=False)
    print("Saved predictions:", pred_csv)

    # map
    out_html = OUTPUTS_DIR / "maps" / "uhi_predictions_testset.html"
    make_map_html(df_pred, out_html)

    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
