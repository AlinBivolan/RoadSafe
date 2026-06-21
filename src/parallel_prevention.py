from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from shapely.geometry import Point

from .prevention import accidents_to_gdf_metric

Backend = Literal["joblib", "dask"]


@dataclass(frozen=True)
class PreventionResult:
    backend: str
    workers: int
    accidents_assigned: gpd.GeoDataFrame
    intersection_stats: gpd.GeoDataFrame
    rural_hotspots: gpd.GeoDataFrame
    timings: dict[str, float]


def _query_kdtree_nearest(tree: cKDTree, points: np.ndarray, workers: int):
    try:
        return tree.query(points, k=1, workers=max(1, int(workers)))
    except TypeError:
        return tree.query(points, k=1)


def _query_ball_point(tree: cKDTree, points: np.ndarray, radius_m: float, workers: int):
    try:
        return tree.query_ball_point(points, r=float(radius_m), workers=max(1, int(workers)))
    except TypeError:
        return tree.query_ball_point(points, r=float(radius_m))


def _empty_rural_hotspots(crs="EPSG:3857") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        columns=[
            "hotspot_id", "accidents_count", "avg_severity", "severe_share",
            "severity_sum", "composite_score", "geometry",
        ],
        geometry="geometry",
        crs=crs,
    )


def _severity_interpretation(value: float) -> str:
    try:
        v = float(value)
    except Exception:
        return "Necunoscut"
    if v < 2.0:
        return "scăzută / preponderent ușoară"
    if v < 5.5:
        return "medie / influență serious"
    return "ridicată / influență fatală"


def make_synthetic_intersections(df: pd.DataFrame, target_points: int = 1200) -> gpd.GeoDataFrame:
    """Creează intersecții sintetice pentru benchmark offline.

    Nu înlocuiește OSM în aplicația principală. Este util doar pentru testarea
    paralelizării când nu vrei să descarci o rețea reală.
    """
    sample = df.dropna(subset=["latitude", "longitude"]).copy()
    if sample.empty:
        return gpd.GeoDataFrame(columns=["intersection_id", "adaptive_radius_m", "geometry"], geometry="geometry", crs="EPSG:3857")

    lat_min, lat_max = sample["latitude"].quantile([0.02, 0.98])
    lon_min, lon_max = sample["longitude"].quantile([0.02, 0.98])
    side = max(2, int(math.sqrt(max(4, int(target_points)))))
    lats = np.linspace(float(lat_min), float(lat_max), side)
    lons = np.linspace(float(lon_min), float(lon_max), side)
    points = [Point(lon, lat) for lat in lats for lon in lons]
    gdf = gpd.GeoDataFrame(
        {"intersection_id": [f"synthetic_{i}" for i in range(1, len(points) + 1)]},
        geometry=points,
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=min(2, len(coords)))
    if len(coords) > 1:
        nearest = distances[:, 1]
    else:
        nearest = np.array([100.0])
    gdf["nearest_intersection_distance_m"] = nearest
    # Rază standard 50 m; pentru intersecții mai apropiate de 100 m, folosim jumătate din distanță.
    gdf["adaptive_radius_m"] = np.minimum(50.0, nearest / 2.0)
    return gdf


def assign_accidents_vectorized(
    accidents_metric: gpd.GeoDataFrame,
    intersections_metric: gpd.GeoDataFrame,
    workers: int = 1,
) -> gpd.GeoDataFrame:
    accidents_metric = accidents_metric.copy()
    intersections_lookup = intersections_metric.reset_index(drop=True)[
        ["intersection_id", "adaptive_radius_m", "geometry"]
    ].copy()

    if accidents_metric.empty or intersections_lookup.empty:
        accidents_metric["nearest_intersection_idx"] = []
        accidents_metric["distance_to_nearest_intersection_m"] = []
        accidents_metric["assigned_intersection_id"] = None
        accidents_metric["assigned_to_intersection"] = 0
        return accidents_metric

    inter_coords = np.column_stack([
        intersections_lookup.geometry.x.to_numpy(),
        intersections_lookup.geometry.y.to_numpy(),
    ])
    accident_coords = np.column_stack([
        accidents_metric.geometry.x.to_numpy(),
        accidents_metric.geometry.y.to_numpy(),
    ])

    tree = cKDTree(inter_coords)
    distances, idxs = _query_kdtree_nearest(tree, accident_coords, workers=workers)
    radii = intersections_lookup["adaptive_radius_m"].to_numpy(dtype=float)[idxs]
    intersection_ids = intersections_lookup["intersection_id"].to_numpy(dtype=object)[idxs]
    assigned_mask = distances <= radii

    accidents_metric["nearest_intersection_idx"] = idxs.astype(int)
    accidents_metric["distance_to_nearest_intersection_m"] = distances.astype(float)
    accidents_metric["assigned_to_intersection"] = assigned_mask.astype(int)
    accidents_metric["assigned_intersection_id"] = np.where(assigned_mask, intersection_ids, None)
    return accidents_metric


def aggregate_intersection_stats(accidents_assigned: pd.DataFrame, intersections_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    intersections_metric = intersections_metric.copy()
    assigned = accidents_assigned[accidents_assigned["assigned_to_intersection"] == 1].copy()
    if assigned.empty:
        out = intersections_metric.copy()
        out["accidents_count"] = 0
        out["avg_severity"] = 0.0
        out["severe_share"] = 0.0
        out["severity_sum"] = 0.0
        out["composite_score"] = 0.0
        out["severity_interpretation"] = "fără accidente asociate"
        return out

    grouped = assigned.groupby("assigned_intersection_id", as_index=False).agg(
        accidents_count=("collision_severity", "size"),
        avg_severity=("severity_score", "mean"),
        severe_share=("severe_flag", "mean"),
        severity_sum=("severity_score", "sum"),
    )
    grouped["severe_share"] = grouped["severe_share"] * 100.0
    grouped["severity_interpretation"] = grouped["avg_severity"].apply(_severity_interpretation)

    max_acc = grouped["accidents_count"].max() if len(grouped) else 1
    max_sev_sum = grouped["severity_sum"].max() if len(grouped) else 1
    max_severe_share = grouped["severe_share"].max() if len(grouped) else 1
    grouped["composite_score"] = (
        0.35 * (grouped["accidents_count"] / max(max_acc, 1))
        + 0.40 * (grouped["severity_sum"] / max(max_sev_sum, 1))
        + 0.25 * (grouped["severe_share"] / max(max_severe_share, 1))
    ) * 100.0

    out = intersections_metric.merge(
        grouped,
        left_on="intersection_id",
        right_on="assigned_intersection_id",
        how="left",
    )
    for col in ["accidents_count", "avg_severity", "severe_share", "severity_sum", "composite_score"]:
        out[col] = out[col].fillna(0)
    out["severity_interpretation"] = out["severity_interpretation"].fillna("fără accidente asociate")
    return out


def detect_rural_hotspots_fast(
    accidents_assigned: gpd.GeoDataFrame,
    min_accidents: int = 3,
    cluster_radius_m: float = 50.0,
    workers: int = 1,
) -> gpd.GeoDataFrame:
    remaining = accidents_assigned[accidents_assigned["assigned_to_intersection"] == 0].copy()
    if "urban_or_rural_area" in remaining.columns:
        remaining = remaining[remaining["urban_or_rural_area"] == 2].copy()
    if remaining.empty:
        return _empty_rural_hotspots(crs=accidents_assigned.crs)

    coords = np.column_stack([remaining.geometry.x.to_numpy(), remaining.geometry.y.to_numpy()])
    tree = cKDTree(coords)
    neighbors_list = _query_ball_point(tree, coords, radius_m=cluster_radius_m, workers=workers)

    visited: set[int] = set()
    clusters: list[list[int]] = []
    for i in range(len(coords)):
        if i in visited:
            continue
        component: set[int] = set()
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for nb in neighbors_list[node]:
                if nb not in visited:
                    stack.append(int(nb))
        if len(component) >= int(min_accidents):
            clusters.append(sorted(component))

    if not clusters:
        return _empty_rural_hotspots(crs=accidents_assigned.crs)

    rows = []
    for cluster_id, indices in enumerate(clusters, start=1):
        cluster_df = remaining.iloc[indices].copy()
        centroid = cluster_df.geometry.unary_union.centroid
        avg_severity = float(cluster_df["severity_score"].mean())
        rows.append({
            "hotspot_id": cluster_id,
            "accidents_count": int(len(cluster_df)),
            "avg_severity": avg_severity,
            "severe_share": float(cluster_df["severe_flag"].mean() * 100.0),
            "severity_sum": float(cluster_df["severity_score"].sum()),
            "geometry": centroid,
        })

    hotspots = gpd.GeoDataFrame(rows, geometry="geometry", crs=accidents_assigned.crs)
    max_acc = hotspots["accidents_count"].max() if len(hotspots) else 1
    max_sev_sum = hotspots["severity_sum"].max() if len(hotspots) else 1
    max_severe_share = hotspots["severe_share"].max() if len(hotspots) else 1
    hotspots["composite_score"] = (
        0.35 * (hotspots["accidents_count"] / max(max_acc, 1))
        + 0.40 * (hotspots["severity_sum"] / max(max_sev_sum, 1))
        + 0.25 * (hotspots["severe_share"] / max(max_severe_share, 1))
    ) * 100.0
    return hotspots


def _split_dataframe(df: pd.DataFrame, parts: int) -> list[pd.DataFrame]:
    return [chunk.copy() for chunk in np.array_split(df, max(1, int(parts))) if len(chunk) > 0]


def run_prevention_joblib(
    accidents_df: pd.DataFrame,
    intersections_metric: gpd.GeoDataFrame,
    min_rural_accidents: int = 3,
    rural_radius_m: float = 50.0,
    workers: int = 1,
) -> PreventionResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    accidents_metric = accidents_to_gdf_metric(accidents_df)
    timings["conversie_crs_s"] = time.perf_counter() - t0

    t = time.perf_counter()
    # cKDTree poate folosi intern workeri; nu copiem GeoDataFrame-ul pe procese separate.
    accidents_assigned = assign_accidents_vectorized(accidents_metric, intersections_metric, workers=workers)
    timings["asociere_kdtree_s"] = time.perf_counter() - t

    t = time.perf_counter()
    intersection_stats = aggregate_intersection_stats(accidents_assigned, intersections_metric)
    timings["agregare_intersectii_s"] = time.perf_counter() - t

    # Hotspot-urile rurale au fost eliminate din secțiunea 5.
    rural_hotspots = _empty_rural_hotspots(crs=intersections_metric.crs)
    timings["total_s"] = time.perf_counter() - t0

    return PreventionResult("joblib", int(workers), accidents_assigned, intersection_stats, rural_hotspots, timings)


def run_prevention_dask(
    accidents_df: pd.DataFrame,
    intersections_metric: gpd.GeoDataFrame,
    min_rural_accidents: int = 3,
    rural_radius_m: float = 50.0,
    workers: int = 1,
) -> PreventionResult:
    """Variantă demonstrativă cu Dask delayed.

    Dask este folosit ca scheduler distribuit/local pentru bucăți de dataframe.
    Pentru corectitudine, agregarea finală pe intersecții se face după concatenare.
    """
    try:
        from dask import delayed, compute
    except Exception as exc:
        raise RuntimeError("Dask nu este instalat. Rulează: pip install dask") from exc

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    chunks = _split_dataframe(accidents_df, max(1, int(workers)))

    @delayed
    def convert_and_assign(chunk: pd.DataFrame):
        metric = accidents_to_gdf_metric(chunk)
        return assign_accidents_vectorized(metric, intersections_metric, workers=1)

    t = time.perf_counter()
    delayed_parts = [convert_and_assign(chunk) for chunk in chunks]
    parts = compute(*delayed_parts, scheduler="threads", num_workers=max(1, int(workers)))
    accidents_assigned = pd.concat(parts, ignore_index=False) if parts else accidents_to_gdf_metric(accidents_df.iloc[0:0])
    timings["conversie_plus_asociere_dask_s"] = time.perf_counter() - t

    t = time.perf_counter()
    intersection_stats = aggregate_intersection_stats(accidents_assigned, intersections_metric)
    timings["agregare_intersectii_s"] = time.perf_counter() - t

    # Hotspot-urile rurale au fost eliminate din secțiunea 5.
    rural_hotspots = _empty_rural_hotspots(crs=intersections_metric.crs)
    timings["total_s"] = time.perf_counter() - t0

    return PreventionResult("dask", int(workers), accidents_assigned, intersection_stats, rural_hotspots, timings)
