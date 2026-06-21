from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.spatial import cKDTree


SEVERITY_MAP = {
    1: 8,  # fatal
    2: 4,  # serious
    3: 1,  # slight
}


def accidents_to_gdf_metric(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Transformă accidentele în GeoDataFrame metric, folosind doar coloanele necesare.

    Nu folosește cache și nu citește fișiere intermediare. Reducerea coloanelor
    scade memoria folosită la Secțiunea 5 și poate accelera reproiecția.
    """
    keep_cols = [
        "latitude",
        "longitude",
        "collision_severity",
        "urban_or_rural_area",
        "severity_score",
        "severe_flag",
    ]
    temp = df[[c for c in keep_cols if c in df.columns]].copy()

    if "severity_score" not in temp.columns:
        temp["severity_score"] = temp["collision_severity"].map(SEVERITY_MAP).fillna(1)
    if "severe_flag" not in temp.columns:
        temp["severe_flag"] = temp["collision_severity"].isin([1, 2]).astype(int)

    geometry = [Point(lon, lat) for lon, lat in zip(temp["longitude"], temp["latitude"])]
    gdf = gpd.GeoDataFrame(temp, geometry=geometry, crs="EPSG:4326")
    return gdf.to_crs(epsg=3857)


def assign_accidents_to_intersections(
    accidents_metric: gpd.GeoDataFrame,
    intersections_metric: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Asociază fiecare accident la cea mai apropiată intersecție,
    DAR doar dacă intră în raza adaptivă a acelei intersecții.

    Returnează:
    - accidents_with_assignments
    - intersection_stats
    """
    accidents_metric = accidents_metric.copy()
    intersections_metric = intersections_metric.copy()

    inter_coords = list(zip(intersections_metric.geometry.x, intersections_metric.geometry.y))
    tree = cKDTree(inter_coords)

    accident_coords = list(zip(accidents_metric.geometry.x, accidents_metric.geometry.y))
    distances, idxs = tree.query(accident_coords, k=1)

    accidents_metric["nearest_intersection_idx"] = idxs
    accidents_metric["distance_to_nearest_intersection_m"] = distances

    intersections_lookup = intersections_metric.reset_index(drop=True)[
        ["intersection_id", "adaptive_radius_m", "geometry"]
    ].copy()

    accidents_metric["assigned_intersection_id"] = None
    accidents_metric["assigned_to_intersection"] = 0

    for i in range(len(accidents_metric)):
        nearest_idx = int(accidents_metric.iloc[i]["nearest_intersection_idx"])
        d = float(accidents_metric.iloc[i]["distance_to_nearest_intersection_m"])
        radius = float(intersections_lookup.iloc[nearest_idx]["adaptive_radius_m"])
        intersection_id = intersections_lookup.iloc[nearest_idx]["intersection_id"]

        if d <= radius:
            accidents_metric.at[accidents_metric.index[i], "assigned_intersection_id"] = intersection_id
            accidents_metric.at[accidents_metric.index[i], "assigned_to_intersection"] = 1

    assigned = accidents_metric[accidents_metric["assigned_to_intersection"] == 1].copy()

    if assigned.empty:
        intersection_stats = intersections_metric.copy()
        intersection_stats["accidents_count"] = 0
        intersection_stats["avg_severity"] = 0.0
        intersection_stats["severe_share"] = 0.0
        intersection_stats["severity_sum"] = 0.0
        intersection_stats["composite_score"] = 0.0
        return accidents_metric, intersection_stats

    grouped = assigned.groupby("assigned_intersection_id", as_index=False).agg(
        accidents_count=("collision_severity", "size"),
        avg_severity=("severity_score", "mean"),
        severe_share=("severe_flag", "mean"),
        severity_sum=("severity_score", "sum"),
    )

    grouped["severe_share"] = grouped["severe_share"] * 100.0

    max_acc = grouped["accidents_count"].max() if len(grouped) else 1
    max_sev_sum = grouped["severity_sum"].max() if len(grouped) else 1
    max_severe_share = grouped["severe_share"].max() if len(grouped) else 1

    grouped["composite_score"] = (
        0.35 * (grouped["accidents_count"] / max(max_acc, 1)) +
        0.40 * (grouped["severity_sum"] / max(max_sev_sum, 1)) +
        0.25 * (grouped["severe_share"] / max(max_severe_share, 1))
    ) * 100.0

    intersection_stats = intersections_metric.merge(
        grouped,
        left_on="intersection_id",
        right_on="assigned_intersection_id",
        how="left",
    )

    for col in ["accidents_count", "avg_severity", "severe_share", "severity_sum", "composite_score"]:
        intersection_stats[col] = intersection_stats[col].fillna(0)

    return accidents_metric, intersection_stats


def detect_rural_segment_hotspots(
    accidents_metric: gpd.GeoDataFrame,
    min_accidents: int = 3,
    cluster_radius_m: float = 50.0,
) -> gpd.GeoDataFrame:
    """
    Detectează hotspot-uri rurale din accidentele NEASOCIATE intersecțiilor,
    folosind rază fixă de 50m.

    Regula:
    - luăm doar accidente rurale
    - luăm doar accidente neasociate intersecțiilor
    - pentru fiecare punct, căutăm vecinii pe 50m
    - formăm clustere simple prin componente conexe
    """
    remaining = accidents_metric[
        (accidents_metric["assigned_to_intersection"] == 0)
    ].copy()

    if "urban_or_rural_area" in remaining.columns:
        remaining = remaining[remaining["urban_or_rural_area"] == 2].copy()

    if remaining.empty:
        return gpd.GeoDataFrame(columns=[
            "hotspot_id",
            "accidents_count",
            "avg_severity",
            "severe_share",
            "severity_sum",
            "composite_score",
            "geometry",
        ], geometry="geometry", crs="EPSG:3857")

    coords = list(zip(remaining.geometry.x, remaining.geometry.y))
    tree = cKDTree(coords)

    neighbors_list = tree.query_ball_point(coords, r=cluster_radius_m)

    visited = set()
    clusters = []

    for i in range(len(coords)):
        if i in visited:
            continue

        component = set()
        stack = [i]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)

            for nb in neighbors_list[node]:
                if nb not in visited:
                    stack.append(nb)

        if len(component) >= min_accidents:
            clusters.append(sorted(component))

    if not clusters:
        return gpd.GeoDataFrame(columns=[
            "hotspot_id",
            "accidents_count",
            "avg_severity",
            "severe_share",
            "severity_sum",
            "composite_score",
            "geometry",
        ], geometry="geometry", crs="EPSG:3857")

    rows = []

    for cluster_id, indices in enumerate(clusters, start=1):
        cluster_df = remaining.iloc[indices].copy()

        centroid = cluster_df.geometry.unary_union.centroid
        accidents_count = len(cluster_df)
        avg_severity = float(cluster_df["severity_score"].mean())
        severe_share = float(cluster_df["severe_flag"].mean() * 100.0)
        severity_sum = float(cluster_df["severity_score"].sum())

        rows.append({
            "hotspot_id": cluster_id,
            "accidents_count": accidents_count,
            "avg_severity": avg_severity,
            "severe_share": severe_share,
            "severity_sum": severity_sum,
            "geometry": centroid,
        })

    hotspots = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")

    max_acc = hotspots["accidents_count"].max() if len(hotspots) else 1
    max_sev_sum = hotspots["severity_sum"].max() if len(hotspots) else 1
    max_severe_share = hotspots["severe_share"].max() if len(hotspots) else 1

    hotspots["composite_score"] = (
        0.35 * (hotspots["accidents_count"] / max(max_acc, 1)) +
        0.40 * (hotspots["severity_sum"] / max(max_sev_sum, 1)) +
        0.25 * (hotspots["severe_share"] / max(max_severe_share, 1))
    ) * 100.0

    return hotspots


def to_wgs84(gdf_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf_metric.empty:
        return gdf_metric
    return gdf_metric.to_crs(epsg=4326)