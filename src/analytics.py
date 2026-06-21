from __future__ import annotations

import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _chunk_dataframe(df: pd.DataFrame, n_chunks: int) -> list[pd.DataFrame]:
    n_chunks = max(1, n_chunks)
    chunk_size = math.ceil(len(df) / n_chunks)
    return [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]


def _minmax(series: pd.Series) -> pd.Series:
    smin = series.min()
    smax = series.max()
    if smax == smin:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - smin) / (smax - smin)


def _grid_chunk(chunk: pd.DataFrame, cell_size: float) -> pd.DataFrame:
    temp = chunk.copy()
    temp["lat_bin"] = np.floor(temp["latitude"] / cell_size) * cell_size
    temp["lon_bin"] = np.floor(temp["longitude"] / cell_size) * cell_size

    agg = temp.groupby(["lat_bin", "lon_bin"], as_index=False).agg(
        accidents_count=("collision_severity", "size"),
        severity_sum=("severity_score", "sum"),
        severe_count=("severe_flag", "sum"),
        avg_speed=("speed_limit", "mean"),
        avg_severity=("severity_display", "mean"),
    )
    return agg


def build_grid_metrics(
    df: pd.DataFrame,
    cell_size: float = 0.02,
    n_jobs: int = 1,
    use_dask: bool = False,
) -> pd.DataFrame:
    agg = None

    if use_dask:
        try:
            import dask.dataframe as dd

            temp = df.copy()
            temp["lat_bin"] = np.floor(temp["latitude"] / cell_size) * cell_size
            temp["lon_bin"] = np.floor(temp["longitude"] / cell_size) * cell_size

            ddf = dd.from_pandas(temp, npartitions=max(2, n_jobs if n_jobs > 0 else 4))
            dask_agg = ddf.groupby(["lat_bin", "lon_bin"]).agg({
                "collision_severity": "count",
                "severity_score": "sum",
                "severe_flag": "sum",
                "speed_limit": "mean",
                "severity_display": "mean",
            }).compute()

            dask_agg.columns = [
                "accidents_count",
                "severity_sum",
                "severe_count",
                "avg_speed",
                "avg_severity",
            ]
            agg = dask_agg.reset_index()
        except Exception:
            agg = None

    if agg is None:
        chunks = _chunk_dataframe(df, abs(n_jobs) if n_jobs and n_jobs > 0 else 4)
        parts = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_grid_chunk)(chunk, cell_size) for chunk in chunks
        )
        agg = pd.concat(parts, ignore_index=True).groupby(["lat_bin", "lon_bin"], as_index=False).agg(
            accidents_count=("accidents_count", "sum"),
            severity_sum=("severity_sum", "sum"),
            severe_count=("severe_count", "sum"),
            avg_speed=("avg_speed", "mean"),
            avg_severity=("avg_severity", "mean"),
        )

    agg["severe_share"] = agg["severe_count"] / agg["accidents_count"].clip(lower=1)
    agg["lat_center"] = agg["lat_bin"] + cell_size / 2
    agg["lon_center"] = agg["lon_bin"] + cell_size / 2

    agg["composite_score"] = (
        0.35 * _minmax(agg["accidents_count"]) +
        0.40 * _minmax(agg["severity_sum"]) +
        0.25 * _minmax(agg["severe_share"])
    ) * 100

    return agg


def yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("collision_year", as_index=False).size().rename(columns={"size": "accidents"})


def hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("hour", as_index=False).size().rename(columns={"size": "accidents"})


def hourly_severity_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("hour", as_index=False).agg(
        accidents=("collision_severity", "size"),
        avg_severity_raw=("severity_display", "mean"),
        severe_share=("severe_flag", "mean"),
    )
    out["severe_share"] = out["severe_share"] * 100
    return out


def weekday_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("weekday", as_index=False, observed=False).size().rename(columns={"size": "accidents"})
    return out.sort_values("weekday")


def severity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("severity_label", as_index=False).size().rename(columns={"size": "count"})


def weather_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "weather_label" not in df.columns:
        return pd.DataFrame(columns=["weather_label", "count"])
    out = df.groupby("weather_label", as_index=False).size().rename(columns={"size": "count"})
    return out.sort_values("count", ascending=False)


def light_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "light_label" not in df.columns:
        return pd.DataFrame(columns=["light_label", "count"])
    out = df.groupby("light_label", as_index=False).size().rename(columns={"size": "count"})
    return out.sort_values("count", ascending=False)


def urban_rural_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("area_type", as_index=False).agg(
        accidents=("collision_severity", "size"),
        avg_severity=("severity_display", "mean"),
        severe_share=("severe_flag", "mean"),
        avg_speed=("speed_limit", "mean"),
    )
    out["severe_share"] = out["severe_share"] * 100
    return out


def top_hotspots(grid_df: pd.DataFrame, top_n: int = 10, by: str = "composite_score") -> pd.DataFrame:
    cols = ["lat_center", "lon_center", "accidents_count", "avg_severity", "severe_share", "composite_score"]
    return grid_df.sort_values(by, ascending=False)[cols].head(top_n).reset_index(drop=True)


def frequency_vs_severity_points(grid_df: pd.DataFrame) -> pd.DataFrame:
    out = grid_df[["accidents_count", "avg_severity", "severe_share", "composite_score"]].copy()
    out["severe_share"] = out["severe_share"] * 100
    return out


def filter_df(
    df: pd.DataFrame,
    years: tuple[int, int] | None = None,
    area_type: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if years:
        out = out[(out["collision_year"] >= years[0]) & (out["collision_year"] <= years[1])]
    if area_type and area_type != "All":
        out = out[out["area_type"] == area_type]
    return out