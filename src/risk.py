from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import substring


# =========================================================
# CONSTANTE SECȚIUNEA 6
# =========================================================
ROUTE_BUFFER_M = 40.0
ROUTE_MAX_SEGMENTS = 10
ROUTE_MIN_SEGMENT_LENGTH_KM = 5.0
ROUTE_FREQUENCY_SCALE_PER_KM = 2.50
ROUTE_FREQUENCY_WEIGHT = 0.45
ROUTE_SEVERITY_WEIGHT = 0.55


def _to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")
    return gdf.to_crs(epsg=3857)


def _route_to_metric(line: LineString) -> LineString:
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs="EPSG:4326").to_crs(epsg=3857)
    return gdf.geometry.iloc[0]


def auto_segment_count(
    route_length_km: float,
    max_segments: int = ROUTE_MAX_SEGMENTS,
    min_segment_length_km: float = ROUTE_MIN_SEGMENT_LENGTH_KM,
) -> int:
    """Maximum 10 segmente, dar fără segmente sub 5 km dacă traseul permite."""
    try:
        route_length_km = float(route_length_km)
    except Exception:
        return 1

    if route_length_km <= 0:
        return 1

    if route_length_km < min_segment_length_km:
        return 1

    return max(1, min(int(max_segments), int(route_length_km // min_segment_length_km)))


def route_frequency_score(freq_per_km: float) -> float:
    """Transformă accidente/km în scor 0–100."""
    value = max(0.0, float(freq_per_km))
    return float(min(100.0, 100.0 * (1.0 - math.exp(-value / ROUTE_FREQUENCY_SCALE_PER_KM))))


def route_severity_score(avg_severity_display: float) -> float:
    """Transformă severitatea medie 1–3 în scor 0–100."""
    value = float(avg_severity_display)
    return float(min(100.0, max(0.0, (value - 1.0) / 2.0 * 100.0)))


def route_composite_score(frequency_score: float, severity_score: float) -> float:
    """Scor final: 45% frecvență + 55% severitate."""
    return float(min(100.0, max(0.0,
        ROUTE_FREQUENCY_WEIGHT * float(frequency_score)
        + ROUTE_SEVERITY_WEIGHT * float(severity_score)
    )))


def calculate_route_risk(
    accidents_df: pd.DataFrame,
    route_line: LineString,
    buffer_m: float = ROUTE_BUFFER_M,
) -> dict:
    accidents_gdf = _to_gdf(accidents_df)
    route_metric = _route_to_metric(route_line)

    route_length_km = max(route_metric.length / 1000, 0.001)
    nearby = accidents_gdf[accidents_gdf.geometry.within(route_metric.buffer(buffer_m))].copy()

    if nearby.empty:
        return {
            "route_length_km": float(route_length_km),
            "accidents_count": 0,
            "frequency_score": 0.0,
            "severity_score": 0.0,
            "avg_severity": 0.0,
            "avg_severity_display": 0.0,
            "composite_score": 0.0,
        }

    accidents_count = int(len(nearby))
    freq_per_km = accidents_count / route_length_km

    # Scala simplă pentru afișare: Slight=1, Serious=2, Fatal=3.
    if "severity_display" in nearby.columns:
        avg_severity_display = float(nearby["severity_display"].mean())
    else:
        display_map = {1: 3, 2: 2, 3: 1}
        avg_severity_display = float(nearby["collision_severity"].map(display_map).mean())

    if "severity_score" in nearby.columns:
        avg_severity = float(nearby["severity_score"].mean())
    else:
        avg_severity = avg_severity_display

    frequency_score = route_frequency_score(freq_per_km)
    severity_score = route_severity_score(avg_severity_display)
    composite_score = route_composite_score(frequency_score, severity_score)

    return {
        "route_length_km": float(route_length_km),
        "accidents_count": accidents_count,
        "frequency_score": float(frequency_score),
        "severity_score": float(severity_score),
        "avg_severity": float(avg_severity),
        "avg_severity_display": float(avg_severity_display),
        "composite_score": float(composite_score),
    }


def split_line_into_n_segments(line_metric: LineString, n_segments: int) -> list[LineString]:
    if n_segments <= 1:
        return [line_metric]

    total_length = line_metric.length
    out: list[LineString] = []
    for i in range(n_segments):
        start_dist = total_length * i / n_segments
        end_dist = total_length * (i + 1) / n_segments
        out.append(substring(line_metric, start_dist, end_dist))
    return out


def split_line_by_length(line_metric: LineString, segment_len_m: float) -> list[LineString]:
    n_segments = max(1, math.ceil(line_metric.length / max(segment_len_m, 1.0)))
    return split_line_into_n_segments(line_metric, n_segments)


def _segment_metrics(seg_metric: LineString, accidents_gdf: gpd.GeoDataFrame, buffer_m: float, idx: int) -> dict:
    seg_length_km = max(seg_metric.length / 1000, 0.001)
    nearby = accidents_gdf[accidents_gdf.geometry.within(seg_metric.buffer(buffer_m))].copy()

    if nearby.empty:
        acc_count = 0
        avg_sev = 0.0
        avg_sev_display = 0.0
        freq_score = 0.0
        sev_score = 0.0
        composite = 0.0
    else:
        acc_count = int(len(nearby))
        freq_per_km = acc_count / seg_length_km

        if "severity_display" in nearby.columns:
            avg_sev_display = float(nearby["severity_display"].mean())
        else:
            display_map = {1: 3, 2: 2, 3: 1}
            avg_sev_display = float(nearby["collision_severity"].map(display_map).mean())

        if "severity_score" in nearby.columns:
            avg_sev = float(nearby["severity_score"].mean())
        else:
            avg_sev = avg_sev_display

        freq_score = route_frequency_score(freq_per_km)
        sev_score = route_severity_score(avg_sev_display)
        composite = route_composite_score(freq_score, sev_score)

    seg_wgs84 = gpd.GeoSeries([seg_metric], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
    return {
        "segment_id": idx,
        "segment_length_km": float(seg_length_km),
        "accidents_count": acc_count,
        "frequency_score": float(freq_score),
        "severity_score": float(sev_score),
        "avg_severity": float(avg_sev),
        "avg_severity_display": float(avg_sev_display),
        "composite_score": float(composite),
        "geometry": seg_wgs84,
    }


def calculate_segment_risks(
    accidents_df: pd.DataFrame,
    route_line: LineString,
    buffer_m: float = ROUTE_BUFFER_M,
    mode: str = "auto",
    segment_value: int | float | None = None,
    n_jobs: int = 1,
) -> list[dict]:
    accidents_gdf = _to_gdf(accidents_df)
    route_metric = _route_to_metric(route_line)
    route_length_km = max(route_metric.length / 1000, 0.001)

    if mode == "count":
        n_segments = int(segment_value or auto_segment_count(route_length_km))
        segments = split_line_into_n_segments(route_metric, n_segments)
    elif mode == "length":
        segments = split_line_by_length(route_metric, float(segment_value))
    else:
        n_segments = auto_segment_count(route_length_km)
        segments = split_line_into_n_segments(route_metric, n_segments)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_segment_metrics)(seg, accidents_gdf, buffer_m, idx)
        for idx, seg in enumerate(segments, start=1)
    )
    return results


def sample_points_along_line(line_metric: LineString, step_m: float) -> list[Point]:
    total = line_metric.length
    if total <= 0:
        return []
    distances = list(range(0, int(total), int(max(step_m, 1))))
    if not distances or distances[-1] != int(total):
        distances.append(int(total))
    return [line_metric.interpolate(d) for d in distances]


def _point_attention_metric(
    pt: Point,
    idx: int,
    route_metric: LineString,
    accidents_gdf: gpd.GeoDataFrame,
    search_radius_m: float,
    min_score_threshold: float,
) -> dict | None:
    nearby = accidents_gdf[accidents_gdf.geometry.within(pt.buffer(search_radius_m))].copy()
    if nearby.empty:
        return None

    local_frequency = float(len(nearby))
    if "severity_display" in nearby.columns:
        local_avg_severity = float(nearby["severity_display"].mean())
    else:
        display_map = {1: 3, 2: 2, 3: 1}
        local_avg_severity = float(nearby["collision_severity"].map(display_map).mean())

    local_severity_score = route_severity_score(local_avg_severity)

    if local_severity_score < min_score_threshold:
        return None

    return {
        "candidate_id": idx,
        "point_metric": pt,
        "distance_from_start_km": float(route_metric.project(pt) / 1000.0),
        "local_severity_score": float(local_severity_score),
        "local_frequency_score": float(local_frequency),
        "local_avg_severity": float(local_avg_severity),
        "accidents_count": int(len(nearby)),
    }


def detect_attention_points(
    accidents_df: pd.DataFrame,
    route_line: LineString,
    sample_step_m: float = 200.0,
    search_radius_m: float = 150.0,
    min_score_threshold: float = 8.0,
    merge_distance_m: float = 250.0,
    top_n: int = 5,
    n_jobs: int = 1,
) -> list[dict]:
    accidents_gdf = _to_gdf(accidents_df)
    route_metric = _route_to_metric(route_line)
    points = sample_points_along_line(route_metric, sample_step_m)

    raw = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_point_attention_metric)(
            pt, idx, route_metric, accidents_gdf, search_radius_m, min_score_threshold
        )
        for idx, pt in enumerate(points, start=1)
    )
    raw = [x for x in raw if x is not None]

    if not raw:
        return []

    raw.sort(key=lambda x: x["distance_from_start_km"])
    groups = []
    current = [raw[0]]

    for cand in raw[1:]:
        if cand["point_metric"].distance(current[-1]["point_metric"]) <= merge_distance_m:
            current.append(cand)
        else:
            groups.append(current)
            current = [cand]
    groups.append(current)

    points_out = []
    for gid, group in enumerate(groups, start=1):
        best = max(group, key=lambda x: x["local_severity_score"])
        centroid_metric = MultiPoint([g["point_metric"] for g in group]).centroid
        centroid_wgs84 = gpd.GeoSeries([centroid_metric], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]

        points_out.append({
            "point_id": gid,
            "latitude": float(centroid_wgs84.y),
            "longitude": float(centroid_wgs84.x),
            "distance_from_start_km": float(best["distance_from_start_km"]),
            "local_severity_score": float(best["local_severity_score"]),
            "local_frequency_score": float(best["local_frequency_score"]),
            "local_avg_severity": float(best["local_avg_severity"]),
            "accidents_count": int(best["accidents_count"]),
        })

    points_out.sort(key=lambda x: x["local_severity_score"], reverse=True)
    return points_out[:top_n]
