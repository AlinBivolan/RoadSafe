from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from joblib import Parallel, delayed
from shapely.geometry import LineString

from .risk import (
    _segment_metrics,
    _to_gdf,
    _route_to_metric,
    split_line_by_length,
    split_line_into_n_segments,
    detect_attention_points,
)

Backend = Literal["joblib", "dask"]


@dataclass(frozen=True)
class RouteBenchmarkResult:
    backend: str
    workers: int
    segment_results: list[dict]
    attention_points: list[dict]
    timings: dict[str, float]


def make_synthetic_route(df: pd.DataFrame) -> LineString:
    """Traseu offline prin zona datasetului, util pentru benchmark fără OSRM."""
    lat_min = float(df["latitude"].quantile(0.05))
    lat_max = float(df["latitude"].quantile(0.95))
    lon_min = float(df["longitude"].quantile(0.05))
    lon_max = float(df["longitude"].quantile(0.95))
    return LineString([
        (lon_min, lat_min),
        ((lon_min + lon_max) / 2, (lat_min + lat_max) / 2),
        (lon_max, lat_max),
    ])


def _segments_for_route(route_line: LineString, mode: str, segment_value: int | float):
    route_metric = _route_to_metric(route_line)
    if mode == "length":
        segments = split_line_by_length(route_metric, float(segment_value))
    else:
        segments = split_line_into_n_segments(route_metric, int(segment_value))
    return route_metric, segments


def run_route_joblib(
    accidents_df: pd.DataFrame,
    route_line: LineString,
    buffer_m: float = 100.0,
    segment_mode: str = "count",
    segment_value: int | float = 12,
    sample_step_m: float = 200.0,
    attention_radius_m: float = 150.0,
    min_attention_score: float = 8.0,
    top_points: int = 5,
    workers: int = 1,
) -> RouteBenchmarkResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    t = time.perf_counter()
    accidents_gdf = _to_gdf(accidents_df)
    route_metric, segments = _segments_for_route(route_line, segment_mode, segment_value)
    timings["pregatire_geometrie_s"] = time.perf_counter() - t

    t = time.perf_counter()
    segment_results = Parallel(n_jobs=max(1, int(workers)), prefer="threads")(
        delayed(_segment_metrics)(seg, accidents_gdf, buffer_m, idx)
        for idx, seg in enumerate(segments, start=1)
    )
    timings["segmente_joblib_s"] = time.perf_counter() - t

    t = time.perf_counter()
    attention_points = detect_attention_points(
        accidents_df,
        route_line,
        sample_step_m=sample_step_m,
        search_radius_m=attention_radius_m,
        min_score_threshold=min_attention_score,
        merge_distance_m=max(sample_step_m, attention_radius_m),
        top_n=top_points,
        n_jobs=max(1, int(workers)),
    )
    timings["puncte_critice_joblib_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - t0
    return RouteBenchmarkResult("joblib", int(workers), segment_results, attention_points, timings)


def run_route_dask(
    accidents_df: pd.DataFrame,
    route_line: LineString,
    buffer_m: float = 100.0,
    segment_mode: str = "count",
    segment_value: int | float = 12,
    sample_step_m: float = 200.0,
    attention_radius_m: float = 150.0,
    min_attention_score: float = 8.0,
    top_points: int = 5,
    workers: int = 1,
) -> RouteBenchmarkResult:
    try:
        from dask import delayed, compute
    except Exception as exc:
        raise RuntimeError("Dask nu este instalat. Rulează: pip install dask") from exc

    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    t = time.perf_counter()
    accidents_gdf = _to_gdf(accidents_df)
    route_metric, segments = _segments_for_route(route_line, segment_mode, segment_value)
    timings["pregatire_geometrie_s"] = time.perf_counter() - t

    t = time.perf_counter()
    tasks = [
        delayed(_segment_metrics)(seg, accidents_gdf, buffer_m, idx)
        for idx, seg in enumerate(segments, start=1)
    ]
    segment_results = list(compute(*tasks, scheduler="threads", num_workers=max(1, int(workers)))) if tasks else []
    timings["segmente_dask_s"] = time.perf_counter() - t

    # Pentru puncte critice păstrăm implementarea Joblib existentă, ca benchmark-ul să arate clar
    # că unele părți sunt mai potrivite pentru Joblib decât pentru Dask.
    t = time.perf_counter()
    attention_points = detect_attention_points(
        accidents_df,
        route_line,
        sample_step_m=sample_step_m,
        search_radius_m=attention_radius_m,
        min_score_threshold=min_attention_score,
        merge_distance_m=max(sample_step_m, attention_radius_m),
        top_n=top_points,
        n_jobs=max(1, int(workers)),
    )
    timings["puncte_critice_joblib_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - t0
    return RouteBenchmarkResult("dask", int(workers), segment_results, attention_points, timings)
