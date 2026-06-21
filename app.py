from __future__ import annotations

import os
import time
import platform
from statistics import mean, stdev
import hashlib
import pickle
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from joblib import Parallel, delayed
import streamlit as st
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import substring
import math

from src.analytics import (
    build_grid_metrics,
    filter_df,
    frequency_vs_severity_points,
    hourly_counts,
    hourly_severity_summary,
    light_distribution,
    severity_distribution,
    top_hotspots,
    urban_rural_summary,
    weather_distribution,
    weekday_counts,
    yearly_counts,
)
from src.data_loader import dataset_overview, load_accidents
from src.maps import (
    COLORS,
    bar_chart,
    line_chart,
    map_grid,
    map_hotspots,
    map_metric_density,
    map_metric_frequency,
    map_metric_severity,
    map_points,
    prevention_map,
    route_map,
    scatter_frequency_vs_severity,
)
from src.prevention import (
    accidents_to_gdf_metric,
    assign_accidents_to_intersections,
    detect_rural_segment_hotspots,
    to_wgs84,
)
from src.road_network import (
    compute_adaptive_radii,
    download_drive_network,
    extract_intersections_from_graph,
    intersections_back_to_wgs84,
    intersections_to_metric,
)
from src.routing import geocode_location, get_routes
from src.risk import (
    calculate_route_risk,
    calculate_segment_risks,
)
from src.theme import info_card, load_theme, section_header
from src.performance import auto_runtime_config


st.set_page_config(page_title="RoadSafe Clean", layout="wide", page_icon="🛣️")
load_theme()

# Scroll-snap ajută pagina să se oprească natural la începutul fiecărei secțiuni.
# În Streamlit, comportamentul exact depinde de browser, dar pe Chrome/Edge/Firefox
# oferă efectul de scroll fluid între capitole fără sidebar.
st.markdown(
    """
    <style>
    html {
        scroll-behavior: smooth;
        scroll-snap-type: y proximity;
    }
    .snap-anchor {
        scroll-snap-align: start;
        scroll-margin-top: 0.75rem;
        height: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_anchor(anchor_id: str) -> None:
    st.markdown(f'<div id="{anchor_id}" class="snap-anchor"></div>', unsafe_allow_html=True)

DATA_PATH = "data/accidents.csv"
ROAD_PLACES_PATH = Path("data/uk_places_official.csv")


# =========================================================
# HELPERS
# =========================================================
def get_data() -> pd.DataFrame:
    return load_accidents(DATA_PATH)


def get_grid(df: pd.DataFrame, cell_size: float = 0.02, n_jobs: int = 4, use_dask: bool = False) -> pd.DataFrame:
    return build_grid_metrics(df, cell_size=cell_size, n_jobs=n_jobs, use_dask=use_dask)


def _safe_place_slug(place_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", place_name.strip().lower()).strip("_")
    digest = hashlib.md5(place_name.strip().lower().encode("utf-8")).hexdigest()[:8]
    return f"{slug[:55]}_{digest}" if slug else f"place_{digest}"


def load_place_catalog() -> pd.DataFrame:
    """Catalog oficial generat din OpenStreetMap/Overpass, nu listă scrisă manual."""
    columns = ["label", "place_name", "cache_key", "admin_level", "osm_type", "osm_id"]
    if not ROAD_PLACES_PATH.exists():
        return pd.DataFrame(columns=columns)

    catalog = pd.read_csv(ROAD_PLACES_PATH)
    for col in columns:
        if col not in catalog.columns:
            catalog[col] = ""

    catalog["label"] = catalog["label"].fillna(catalog["place_name"]).astype(str).str.strip()
    catalog["place_name"] = catalog["place_name"].fillna(catalog["label"]).astype(str).str.strip()
    catalog["cache_key"] = catalog["cache_key"].fillna("").astype(str).str.strip()
    catalog = catalog[catalog["label"].ne("") & catalog["cache_key"].ne("")].copy()
    return catalog.drop_duplicates("cache_key").sort_values("label").reset_index(drop=True)



def _normalize_search_text(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _place_suggestions(catalog: pd.DataFrame, query: str, limit: int = 8) -> list[str]:
    """Autocomplete local, stil Google Maps: text liber -> cele mai apropiate zone din catalog."""
    if catalog.empty:
        return []

    labels = catalog["label"].astype(str).tolist()
    q = _normalize_search_text(query)
    if not q:
        return labels[:limit]

    scored: list[tuple[float, str]] = []
    tokens = q.split()

    for label in labels:
        norm = _normalize_search_text(label)
        token_bonus = sum(1 for token in tokens if token in norm) / max(len(tokens), 1)
        prefix_bonus = 1.0 if norm.startswith(q) else 0.0
        similarity = SequenceMatcher(None, q, norm).ratio()
        score = 0.55 * token_bonus + 0.25 * prefix_bonus + 0.20 * similarity
        if token_bonus > 0 or similarity >= 0.35:
            scored.append((score, label))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [label for _, label in scored[:limit]]


def get_intersections_for_place(place_name: str):
    """Descarcă rețeaua rutieră OSM pentru zona aleasă și calculează intersecțiile.

    Nu salvează cache pe disc și nu încarcă rezultate pre-calculate.
    """
    G = download_drive_network(place_name)
    intersections = extract_intersections_from_graph(G)
    intersections_metric = intersections_to_metric(intersections)
    intersections_metric = compute_adaptive_radii(intersections_metric)
    return intersections_metric


def animated_counter(label: str, value: float | int | str, suffix: str = "", decimals: int = 0, color: str = "#FF7A00"):
    """Card metric static.

    Versiunile anterioare foloseau JavaScript pentru animație, dar Streamlit poate bloca
    execuția scripturilor inserate prin markdown. De aceea unele valori rămâneau la 0.
    Aici afișăm direct valoarea calculată, fără JS, ca să fie stabil.
    """
    if isinstance(value, str):
        display_value = value
    else:
        try:
            if pd.isna(value):
                value = 0
        except Exception:
            pass
        if decimals == 0:
            display_value = f"{int(round(float(value))):,}".replace(",", ".")
        else:
            display_value = f"{float(value):.{decimals}f}"
        display_value = f"{display_value}{suffix}"

    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
            min-height: 92px;
        ">
            <div style="font-size: 0.92rem; color: #A8B4C3; margin-bottom: 8px;">{label}</div>
            <div style="font-size: 1.85rem; font-weight: 850; color: {color}; line-height: 1.15;">
                {display_value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_scroll(fig: go.Figure, height: int | None = None):
    if height is not None:
        fig.update_layout(height=height)

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
                "autoScale2d",
            ],
        },
    )





def _gdf_bounds_center(gdf) -> tuple[float, float] | tuple[None, None]:
    """Returnează centrul aproximativ al unui GeoDataFrame WGS84 ca (lat, lon)."""
    try:
        if gdf is None or gdf.empty or gdf.geometry.is_empty.all():
            return None, None
        minx, miny, maxx, maxy = gdf.total_bounds
        return float((miny + maxy) / 2.0), float((minx + maxx) / 2.0)
    except Exception:
        return None, None


def _point_center_from_row(row) -> tuple[float | None, float | None]:
    """Extrage centrul unei geometrii punctuale ca (lat, lon)."""
    try:
        geom = row.geometry
        return float(geom.y), float(geom.x)
    except Exception:
        return None, None



def _record_runtime_event(
    section: str,
    operation: str,
    elapsed_s: float,
    workers: int,
    rows: int,
    details: dict | None = None,
) -> None:
    """Salvează în sesiune timpul real al rulărilor făcute de utilizator."""
    if "runtime_log" not in st.session_state:
        st.session_state["runtime_log"] = []

    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "section": section,
        "operation": operation,
        "elapsed_s": round(float(elapsed_s), 3),
        "workers": int(workers),
        "rows": int(rows),
    }
    if details:
        event.update(details)

    st.session_state["runtime_log"].append(event)
    # păstrăm jurnalul compact
    st.session_state["runtime_log"] = st.session_state["runtime_log"][-30:]


def _severity_interpretation_from_weighted_score(value: float) -> str:
    """Interpretare pentru scorul ponderat folosit în prevenție: 1=Slight, 4=Serious, 8=Fatal."""
    try:
        v = float(value)
    except Exception:
        return "Necunoscut"
    if v < 2.0:
        return "scăzută / preponderent ușoară"
    if v < 5.5:
        return "medie / influență serious"
    return "ridicată / influență fatală"


def _empty_rural_hotspots_like(crs="EPSG:3857") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        columns=[
            "hotspot_id", "accidents_count", "avg_severity", "severe_share",
            "severity_sum", "composite_score", "severity_interpretation", "geometry",
        ],
        geometry="geometry",
        crs=crs,
    )


def _query_kdtree_nearest(tree: cKDTree, points: np.ndarray, n_jobs: int):
    """Compatibilitate SciPy: versiunile noi acceptă workers, cele vechi nu."""
    try:
        return tree.query(points, k=1, workers=max(1, int(n_jobs)))
    except TypeError:
        return tree.query(points, k=1)


def _query_ball_point(tree: cKDTree, points: np.ndarray, radius_m: float, n_jobs: int):
    """Compatibilitate SciPy pentru query_ball_point(..., workers=...)."""
    try:
        return tree.query_ball_point(points, r=float(radius_m), workers=max(1, int(n_jobs)))
    except TypeError:
        return tree.query_ball_point(points, r=float(radius_m))


def _assign_accidents_to_intersections_fast(
    accidents_metric: gpd.GeoDataFrame,
    intersections_metric: gpd.GeoDataFrame,
    n_jobs: int = 1,
) -> gpd.GeoDataFrame:
    """Asociere vectorizată accidente -> cea mai apropiată intersecție.

    Varianta inițială parcurgea accidentele într-un for Python. Aici:
    1. construim KDTree o singură dată pentru toate intersecțiile;
    2. întrebăm simultan pentru toate accidentele care este cea mai apropiată intersecție;
    3. aplicăm vectorizat regula d <= adaptive_radius_m.
    """
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
    distances, idxs = _query_kdtree_nearest(tree, accident_coords, n_jobs=n_jobs)

    radii = intersections_lookup["adaptive_radius_m"].to_numpy(dtype=float)[idxs]
    intersection_ids = intersections_lookup["intersection_id"].to_numpy(dtype=object)[idxs]
    assigned_mask = distances <= radii

    accidents_metric["nearest_intersection_idx"] = idxs.astype(int)
    accidents_metric["distance_to_nearest_intersection_m"] = distances.astype(float)
    accidents_metric["assigned_to_intersection"] = assigned_mask.astype(int)
    accidents_metric["assigned_intersection_id"] = np.where(assigned_mask, intersection_ids, None)

    return accidents_metric


def _detect_rural_segment_hotspots_fast(
    accidents_metric: gpd.GeoDataFrame,
    min_accidents: int = 3,
    cluster_radius_m: float = 50.0,
    n_jobs: int = 1,
) -> gpd.GeoDataFrame:
    """Hotspot-uri rurale pentru accidentele neasociate intersecțiilor.

    Căutarea vecinilor se face cu KDTree. Componenta de conectivitate rămâne logică
    de graf, dar evităm calculele brute de distanță între toate perechile de puncte.
    """
    remaining = accidents_metric[accidents_metric["assigned_to_intersection"] == 0].copy()

    if "urban_or_rural_area" in remaining.columns:
        remaining = remaining[remaining["urban_or_rural_area"] == 2].copy()

    if remaining.empty:
        return _empty_rural_hotspots_like(crs=accidents_metric.crs)

    coords = np.column_stack([remaining.geometry.x.to_numpy(), remaining.geometry.y.to_numpy()])
    tree = cKDTree(coords)
    neighbors_list = _query_ball_point(tree, coords, radius_m=cluster_radius_m, n_jobs=n_jobs)

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
        return _empty_rural_hotspots_like(crs=accidents_metric.crs)

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
            "severity_interpretation": _severity_interpretation_from_weighted_score(avg_severity),
            "geometry": centroid,
        })

    hotspots = gpd.GeoDataFrame(rows, geometry="geometry", crs=accidents_metric.crs)
    max_acc = hotspots["accidents_count"].max() if len(hotspots) else 1
    max_sev_sum = hotspots["severity_sum"].max() if len(hotspots) else 1
    max_severe_share = hotspots["severe_share"].max() if len(hotspots) else 1
    hotspots["composite_score"] = (
        0.35 * (hotspots["accidents_count"] / max(max_acc, 1))
        + 0.40 * (hotspots["severity_sum"] / max(max_sev_sum, 1))
        + 0.25 * (hotspots["severe_share"] / max(max_severe_share, 1))
    ) * 100.0
    return hotspots


def _aggregate_intersection_stats_from_assignments(accidents_assigned: pd.DataFrame, intersections_metric):
    intersections_metric = intersections_metric.copy()
    assigned = accidents_assigned[accidents_assigned["assigned_to_intersection"] == 1].copy()

    if assigned.empty:
        intersection_stats = intersections_metric.copy()
        intersection_stats["accidents_count"] = 0
        intersection_stats["avg_severity"] = 0.0
        intersection_stats["severe_share"] = 0.0
        intersection_stats["severity_sum"] = 0.0
        intersection_stats["composite_score"] = 0.0
        intersection_stats["severity_interpretation"] = "fără accidente asociate"
        return intersection_stats

    grouped = assigned.groupby("assigned_intersection_id", as_index=False).agg(
        accidents_count=("collision_severity", "size"),
        avg_severity=("severity_score", "mean"),
        severe_share=("severe_flag", "mean"),
        severity_sum=("severity_score", "sum"),
    )
    grouped["severe_share"] = grouped["severe_share"] * 100.0
    grouped["severity_interpretation"] = grouped["avg_severity"].apply(_severity_interpretation_from_weighted_score)

    max_acc = grouped["accidents_count"].max() if len(grouped) else 1
    max_sev_sum = grouped["severity_sum"].max() if len(grouped) else 1
    max_severe_share = grouped["severe_share"].max() if len(grouped) else 1
    grouped["composite_score"] = (
        0.35 * (grouped["accidents_count"] / max(max_acc, 1))
        + 0.40 * (grouped["severity_sum"] / max(max_sev_sum, 1))
        + 0.25 * (grouped["severe_share"] / max(max_severe_share, 1))
    ) * 100.0

    intersection_stats = intersections_metric.merge(
        grouped,
        left_on="intersection_id",
        right_on="assigned_intersection_id",
        how="left",
    )
    for col in ["accidents_count", "avg_severity", "severe_share", "severity_sum", "composite_score"]:
        intersection_stats[col] = intersection_stats[col].fillna(0)
    intersection_stats["severity_interpretation"] = intersection_stats["severity_interpretation"].fillna("fără accidente asociate")
    return intersection_stats


def _parallel_prevention_pipeline(
    full_df: pd.DataFrame,
    intersections_metric,
    n_jobs: int,
):
    """Secțiunea 5 optimizată: asociere accidente -> intersecții.

    Notă: download-ul OSM nu poate fi accelerat cu workeri locali. Partea optimizată este
    calculul local: conversia accidentelor, asocierea la intersecții și agregarea statisticilor.
    """
    n_jobs = max(1, int(n_jobs))
    accidents_metric = accidents_to_gdf_metric(full_df)
    accidents_assigned = _assign_accidents_to_intersections_fast(
        accidents_metric,
        intersections_metric,
        n_jobs=n_jobs,
    )
    intersection_stats_metric = _aggregate_intersection_stats_from_assignments(
        accidents_assigned,
        intersections_metric,
    )
    return accidents_assigned, intersection_stats_metric


def _parallel_prevention_pipeline_timed(
    full_df: pd.DataFrame,
    intersections_metric,
    n_jobs: int,
):
    """Rulează secțiunea 5 și întoarce timpii pe pași.

    Important: nu toate fazele pot fi accelerate cu workeri. Workerii ajută cel mai mult
    la interogările KDTree, dar conversia CRS, groupby-ul pandas și conversia pentru hartă
    pot rămâne aproape constante.
    """
    timings: dict[str, float] = {}

    t = time.perf_counter()
    accidents_metric = accidents_to_gdf_metric(full_df)
    timings["1_conversie_accidente_crs_s"] = time.perf_counter() - t

    t = time.perf_counter()
    accidents_assigned = _assign_accidents_to_intersections_fast(
        accidents_metric,
        intersections_metric,
        n_jobs=n_jobs,
    )
    timings["2_asociere_kdtree_s"] = time.perf_counter() - t

    t = time.perf_counter()
    intersection_stats_metric = _aggregate_intersection_stats_from_assignments(
        accidents_assigned,
        intersections_metric,
    )
    timings["3_agregare_intersectii_s"] = time.perf_counter() - t

    timings["total_calcul_local_s"] = sum(timings.values())
    return accidents_assigned, intersection_stats_metric, timings


def _timings_to_dataframe(timings: dict[str, float]) -> pd.DataFrame:
    labels = {
        "0_incarcare_retea_osm_s": "Încărcare/descărcare rețea OSM + intersecții",
        "1_conversie_accidente_crs_s": "Conversie accidente în sistem metric",
        "2_asociere_kdtree_s": "Asociere accidente la intersecții cu KDTree",
        "3_agregare_intersectii_s": "Agregare statistici pe intersecții",
        "5_conversie_harta_s": "Conversie rezultate pentru hartă",
        "total_calcul_local_s": "Total calcul local",
        "total_rulare_s": "Total rulare secțiunea 5",
    }
    rows = []
    for key, value in timings.items():
        rows.append({
            "Pas": labels.get(key, key),
            "Timp (s)": round(float(value), 3),
        })
    return pd.DataFrame(rows)

def _memory_mb() -> float | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _timed_benchmark_run(task_name: str, func, repeats: int) -> dict:
    """Rulează o funcție de mai multe ori, fără cache Streamlit, pentru măsurare corectă."""
    times: list[float] = []
    rows_out = None
    extra = ""

    # Warm-up separat: prinde inițializări lazy, dar nu intră în medie.
    try:
        warm = func()
        if isinstance(warm, tuple):
            rows_out, extra = warm
        elif hasattr(warm, "__len__"):
            rows_out = len(warm)
    except Exception as exc:
        return {
            "task": task_name,
            "status": "eroare",
            "workers": None,
            "engine": None,
            "repeats": 0,
            "mean_s": None,
            "std_s": None,
            "min_s": None,
            "max_s": None,
            "rows_out": None,
            "memory_mb": round(_memory_mb(), 2) if _memory_mb() is not None else None,
            "extra": "",
            "error": repr(exc),
        }

    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        try:
            result = func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            if isinstance(result, tuple):
                rows_out, extra = result
            elif hasattr(result, "__len__"):
                rows_out = len(result)
        except Exception as exc:
            return {
                "task": task_name,
                "status": "eroare",
                "workers": None,
                "engine": None,
                "repeats": len(times),
                "mean_s": round(mean(times), 4) if times else None,
                "std_s": round(stdev(times), 4) if len(times) > 1 else 0.0 if times else None,
                "min_s": round(min(times), 4) if times else None,
                "max_s": round(max(times), 4) if times else None,
                "rows_out": rows_out,
                "memory_mb": round(_memory_mb(), 2) if _memory_mb() is not None else None,
                "extra": extra,
                "error": repr(exc),
            }

    return {
        "task": task_name,
        "status": "ok",
        "workers": None,
        "engine": None,
        "repeats": len(times),
        "mean_s": round(mean(times), 4),
        "std_s": round(stdev(times), 4) if len(times) > 1 else 0.0,
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
        "rows_out": rows_out,
        "memory_mb": round(_memory_mb(), 2) if _memory_mb() is not None else None,
        "extra": extra,
        "error": "",
    }


def _make_synthetic_route(df_in: pd.DataFrame) -> LineString:
    """Traseu artificial prin bounding box; nu folosește internet/geocoding."""
    lat_min = float(df_in["latitude"].quantile(0.05))
    lat_max = float(df_in["latitude"].quantile(0.95))
    lon_min = float(df_in["longitude"].quantile(0.05))
    lon_max = float(df_in["longitude"].quantile(0.95))
    return LineString([
        (lon_min, lat_min),
        ((lon_min + lon_max) / 2, (lat_min + lat_max) / 2),
        (lon_max, lat_max),
    ])


def _split_dataframe_evenly(df_in: pd.DataFrame, n_parts: int) -> list[pd.DataFrame]:
    n_parts = max(1, int(n_parts))
    return [chunk.copy() for chunk in np.array_split(df_in, n_parts) if len(chunk) > 0]


def _aggregate_intersection_stats_from_assignments_legacy(accidents_assigned: pd.DataFrame, intersections_metric):
    intersections_metric = intersections_metric.copy()
    assigned = accidents_assigned[accidents_assigned["assigned_to_intersection"] == 1].copy()

    if assigned.empty:
        intersection_stats = intersections_metric.copy()
        intersection_stats["accidents_count"] = 0
        intersection_stats["avg_severity"] = 0.0
        intersection_stats["severe_share"] = 0.0
        intersection_stats["severity_sum"] = 0.0
        intersection_stats["composite_score"] = 0.0
        return intersection_stats

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
        0.35 * (grouped["accidents_count"] / max(max_acc, 1))
        + 0.40 * (grouped["severity_sum"] / max(max_sev_sum, 1))
        + 0.25 * (grouped["severe_share"] / max(max_severe_share, 1))
    ) * 100.0

    intersection_stats = intersections_metric.merge(
        grouped,
        left_on="intersection_id",
        right_on="assigned_intersection_id",
        how="left",
    )
    for col in ["accidents_count", "avg_severity", "severe_share", "severity_sum", "composite_score"]:
        intersection_stats[col] = intersection_stats[col].fillna(0)
    return intersection_stats


def _prevention_pipeline_for_benchmark(
    full_df: pd.DataFrame,
    intersections_metric,
    min_rural_accidents: int,
    rural_radius: float,
    n_jobs: int,
):
    """Rulează calculele grele din secțiunea 5 pe tot datasetul.

    Descărcarea rețelei OSM este exclusă din timp, fiindcă nu depinde de numărul de workeri.
    Testăm partea care chiar poate fi paralelizată: conversie accidente + asociere la intersecții pe bucăți.
    """
    chunks = _split_dataframe_evenly(full_df, n_jobs)

    def assign_one_chunk(chunk: pd.DataFrame):
        acc_metric = accidents_to_gdf_metric(chunk)
        assigned_chunk, _ = assign_accidents_to_intersections(acc_metric, intersections_metric)
        return assigned_chunk

    if n_jobs <= 1:
        assigned_parts = [assign_one_chunk(chunks[0])] if chunks else []
    else:
        assigned_parts = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(assign_one_chunk)(chunk) for chunk in chunks
        )

    accidents_assigned = pd.concat(assigned_parts, ignore_index=False) if assigned_parts else accidents_to_gdf_metric(full_df.iloc[0:0])
    intersection_stats_metric = _aggregate_intersection_stats_from_assignments(accidents_assigned, intersections_metric)
    rural_hotspots_metric = detect_rural_segment_hotspots(
        accidents_assigned,
        min_accidents=int(min_rural_accidents),
        cluster_radius_m=float(rural_radius),
    )
    return (
        len(intersection_stats_metric) + len(rural_hotspots_metric),
        f"intersections={len(intersection_stats_metric)}; rural_hotspots={len(rural_hotspots_metric)}",
    )


def run_sections_5_6_benchmark(
    full_df: pd.DataFrame,
    intersections_metric,
    route_line: LineString,
    workers_to_test: list[int],
    min_rural_accidents: int,
    rural_radius: float,
    buffer_m: float,
    segmentation_mode: str,
    segment_value: int | float,
    sample_step_m: float,
    attention_radius_m: float,
    min_attention_score: float,
    top_points: int,
    repeats: int = 1,
) -> pd.DataFrame:
    results: list[dict] = []
    workers_to_test = sorted(set(max(1, int(w)) for w in workers_to_test))

    for w in workers_to_test:
        r = _timed_benchmark_run(
            "Secțiunea 5 - prevenție",
            lambda w=w: _prevention_pipeline_for_benchmark(
                full_df,
                intersections_metric,
                min_rural_accidents=min_rural_accidents,
                rural_radius=rural_radius,
                n_jobs=w,
            ),
            repeats,
        )
        r.update({"workers": w, "engine": "Joblib"})
        results.append(r)

        r = _timed_benchmark_run(
            "Secțiunea 6 - risc pe segmente",
            lambda w=w: calculate_segment_risks(
                full_df,
                route_line,
                buffer_m=ROUTE_BUFFER_M,
                mode="count",
                segment_value=ROUTE_MAX_SEGMENTS,
                n_jobs=w,
            ),
            repeats,
        )
        r.update({"workers": w, "engine": "Joblib"})
        results.append(r)

        r = _timed_benchmark_run(
            "Secțiunea 6 - puncte critice",
            lambda w=w: detect_attention_points(
                full_df,
                route_line,
                sample_step_m=ATTENTION_SAMPLE_STEP_M,
                search_radius_m=ATTENTION_RADIUS_M,
                min_score_threshold=MIN_ATTENTION_SCORE,
                merge_distance_m=max(ATTENTION_SAMPLE_STEP_M, ATTENTION_RADIUS_M),
                top_n=TOP_ATTENTION_POINTS,
                n_jobs=w,
            ),
            repeats,
        )
        r.update({"workers": w, "engine": "Joblib"})
        results.append(r)

    out = pd.DataFrame(results)
    ok = out[out["status"] == "ok"].copy()
    out["speedup_vs_1_worker"] = None
    out["efficiency_pct"] = None

    if ok.empty:
        return out

    for task, group in ok.groupby("task"):
        baseline_rows = group[group["workers"] == 1]
        baseline_time = float(baseline_rows.iloc[0]["mean_s"]) if not baseline_rows.empty else float(group.sort_values("workers").iloc[0]["mean_s"])
        baseline_workers = 1 if not baseline_rows.empty else int(group.sort_values("workers").iloc[0]["workers"])
        mask = (out["task"] == task) & out["mean_s"].notna()
        out.loc[mask, "speedup_vs_1_worker"] = out.loc[mask, "mean_s"].apply(
            lambda t: round(baseline_time / float(t), 3) if float(t) > 0 else None
        )
        out.loc[mask, "efficiency_pct"] = out.loc[mask].apply(
            lambda row: round((float(row["speedup_vs_1_worker"]) / max(int(row["workers"]) / baseline_workers, 1)) * 100, 2)
            if row["speedup_vs_1_worker"] is not None else None,
            axis=1,
        )

    return out


def make_benchmark_markdown(results_df: pd.DataFrame, metadata: dict) -> str:
    lines = ["# Raport benchmark RoadSafe - secțiunile 5 și 6", ""]
    lines.append("## Configurație")
    for key, value in metadata.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Rezultate")
    display_cols = [
        "task", "engine", "workers", "status", "repeats", "mean_s", "std_s",
        "min_s", "max_s", "speedup_vs_1_worker", "efficiency_pct", "rows_out", "memory_mb", "error"
    ]
    lines.append(results_df[display_cols].to_markdown(index=False))
    lines.append("")
    ok = results_df[results_df["status"] == "ok"].copy()
    if not ok.empty:
        best_by_task = ok.sort_values("mean_s").groupby("task", as_index=False).first()
        lines.append("## Interpretare")
        for _, row in best_by_task.iterrows():
            lines.append(f"- **{row['task']}**: cel mai bun rezultat a fost cu **{int(row['workers'])} workeri**, media **{row['mean_s']} secunde**, speedup **{row['speedup_vs_1_worker']}x**.")
    lines.append("")
    lines.append("Notă: descărcarea rețelei OSM și obținerea traseului sunt făcute o singură dată înainte de benchmark. Testul măsoară calculele locale care pot fi influențate de numărul de workeri.")
    return "\n".join(lines)

def build_hourly_severity_figure(df: pd.DataFrame) -> go.Figure:
    hourly = hourly_severity_summary(df)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["accidents"],
            mode="lines+markers",
            name="Accidente",
            line=dict(color=COLORS["orange"], width=4),
            marker=dict(size=7),
            yaxis="y",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["avg_severity_raw"],
            mode="lines+markers",
            name="Severitate medie",
            line=dict(color=COLORS["cyan"], width=3),
            marker=dict(size=6),
            yaxis="y2",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Accidente pe ore și severitate medie",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111822",
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.14,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(title="Ora"),
        yaxis=dict(title="Nr. accidente"),
        yaxis2=dict(
            title="Severitate medie brută",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        height=440,
    )

    return fig


def build_area_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = urban_rural_summary(df).copy()
    if out.empty:
        return out

    out["accidents"] = out["accidents"].round(0).astype(int)
    out["avg_severity"] = out["avg_severity"].round(2)
    out["severe_share"] = out["severe_share"].round(2)
    out["avg_speed"] = out["avg_speed"].round(2)
    return out


# =========================================================
# LOAD DATA
# =========================================================
try:
    df = get_data()
except Exception as exc:
    st.error(f"Eroare la încărcarea datelor: {exc}")
    st.stop()

# =========================================================
# PAGE HEADER
# =========================================================
st.markdown(
    """
    <div style="text-align:center; padding-top: 8px; padding-bottom: 10px;">
        <h1 style="
            margin-bottom: 0.35rem;
            font-size: 2.5rem;
            font-weight: 900;
            letter-spacing: 0.5px;
        ">
            RoadSafe — Dashboard operațional
        </h1>
        <div style="
            color:#A7B7C8;
            font-size: 1.05rem;
            max-width: 980px;
            margin: 0 auto;
            line-height: 1.6;
        ">
            Dashboard integrat pentru analiză exploratorie, hotspot-uri, prevenție și risc pe traseu. Benchmark-urile sunt separate în aplicații dedicate.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONFIGURARE AUTOMATĂ A RULĂRII
# =========================================================
min_year = int(df["collision_year"].min())
max_year = int(df["collision_year"].max())

# Dashboardul principal pornește fără panou de setări globale.
# Folosim toate datele și o granularitate stabilă pentru harta grid.
selected_years = (min_year, max_year)
selected_area = "All"
cell_size = 0.02
use_dask = False

max_cpu = min(8, os.cpu_count() or 4)
auto_cfg = auto_runtime_config(len(df), task="generic", user_cap=max_cpu, prefer_distributed=False)
parallel_jobs = auto_cfg.n_jobs

st.caption(f"Aplicația folosește Joblib cu {parallel_jobs} workeri estimați.")

filtered_df = filter_df(df, years=selected_years, area_type=selected_area)
overview = dataset_overview(filtered_df)
grid_df = get_grid(filtered_df, cell_size=cell_size, n_jobs=parallel_jobs, use_dask=use_dask)

st.markdown("---")

# =========================================================
# 1. DESCRIEREA DATELOR
# =========================================================
section_anchor("sec-1")
section_header(
    "1. Descrierea datelor",
    "Privire de ansamblu asupra setului de date: volum, severitate, ani acoperiți și distribuție geografică."
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    animated_counter("Număr accidente", overview["rows"], color=COLORS["orange"])
with m2:
    animated_counter("An început", overview["years_min"], color=COLORS["cyan"])
with m3:
    animated_counter("An final", overview["years_max"], color=COLORS["lime"])
with m4:
    animated_counter("Pondere accidente grave", overview["severe_share"] * 100, suffix="%", decimals=2, color=COLORS["pink"])

m5, m6 = st.columns(2)
with m5:
    animated_counter("Severitate medie", overview["avg_severity"], decimals=2, color="#FFC857")
with m6:
    animated_counter("Victime medii / accident", overview["avg_casualties"], decimals=2, color="#00D1FF")

r1c1, r1c2 = st.columns(2)

with r1c1:
    sev_df = severity_distribution(filtered_df)
    plotly_scroll(
        bar_chart(
            sev_df,
            "severity_label",
            "count",
            COLORS["orange"],
            "Distribuția severității"
        ),
        height=400,
    )

with r1c2:
    yr_df = yearly_counts(filtered_df)
    plotly_scroll(
        line_chart(
            yr_df,
            "collision_year",
            "accidents",
            COLORS["cyan"],
            "Accidente pe ani"
        ),
        height=400,
    )

plotly_scroll(
    map_points(filtered_df, "Hartă generală a accidentelor", sample_n=8000, height=620),
)

info_card(
    "Ce observi aici",
    "Secțiunea introduce baza de date: volum, severitate, distribuție anuală și localizarea generală a accidentelor pe hartă."
)

st.markdown("---")

# =========================================================
# 2. ANALIZĂ TEMPORALĂ ȘI CONTEXTUALĂ
# =========================================================
section_anchor("sec-2")
section_header(
    "2. Analiza temporală și contextuală",
    "Aici vedem când apar accidentele și cum se modifică severitatea lor pe parcursul zilei."
)

hourly_df = hourly_counts(filtered_df)
hourly_sev_df = hourly_severity_summary(filtered_df)
weekday_df = weekday_counts(filtered_df)
weather_df = weather_distribution(filtered_df)
light_df = light_distribution(filtered_df)

hourly_peak = hourly_df.sort_values("accidents", ascending=False).iloc[0]
weekday_peak = weekday_df.sort_values("accidents", ascending=False).iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    animated_counter("Ora de vârf", int(hourly_peak["hour"]), color=COLORS["orange"])
with c2:
    st.markdown(
        f"""
        <div style="padding-top: 10px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 8px;">
            <div style="font-size: 0.95rem; color: #A8B4C3; margin-bottom: 6px;">Ziua cea mai încărcată</div>
            <div style="font-size: 2rem; font-weight: 800; color: {COLORS["cyan"]}; line-height: 1.1;">
                {weekday_peak["weekday"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    animated_counter("Severitate medie brută", hourly_sev_df["avg_severity_raw"].mean(), decimals=2, color=COLORS["lime"])
with c4:
    animated_counter("Limită medie viteză", filtered_df["speed_limit"].mean(), decimals=1, color=COLORS["pink"])

r2c1, r2c2 = st.columns(2)
with r2c1:
    plotly_scroll(build_hourly_severity_figure(filtered_df), height=440)

with r2c2:
    plotly_scroll(
        bar_chart(
            weekday_df,
            "weekday",
            "accidents",
            COLORS["cyan"],
            "Accidente pe zile"
        ),
        height=440,
    )

r2c3, r2c4 = st.columns(2)

with r2c3:
    if not weather_df.empty:
        fig_weather = go.Figure(
            go.Bar(
                x=weather_df["count"],
                y=weather_df["weather_label"],
                orientation="h",
                marker_color=COLORS["pink"],
            )
        )
        fig_weather.update_layout(
            template="plotly_dark",
            title="Condiții meteo",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111822",
            margin=dict(l=20, r=20, t=50, b=20),
            height=430,
            yaxis=dict(autorange="reversed"),
        )
        plotly_scroll(fig_weather, height=430)
    else:
        info_card("Condiții meteo", "Coloana meteo nu este disponibilă.")

with r2c4:
    if not light_df.empty:
        fig_light = go.Figure(
            go.Bar(
                x=light_df["count"],
                y=light_df["light_label"],
                orientation="h",
                marker_color=COLORS["lime"],
            )
        )
        fig_light.update_layout(
            template="plotly_dark",
            title="Condiții de lumină",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111822",
            margin=dict(l=20, r=20, t=50, b=20),
            height=430,
            yaxis=dict(autorange="reversed"),
        )
        plotly_scroll(fig_light, height=430)
    else:
        info_card("Condiții de lumină", "Coloana lumină nu este disponibilă.")

info_card(
    "Ce arată această secțiune",
    "Graficul principal arată atât dinamica accidentelor pe ore, cât și severitatea medie brută. Astfel poți observa dacă intervalele cu multe accidente sunt și cele în care accidentele tind să fie mai grave."
)

st.markdown("---")

# =========================================================
# 3. FRECVENȚĂ VS SEVERITATE
# =========================================================
section_anchor("sec-3")
section_header(
    "3. Frecvență vs severitate",
    "Aici comparăm explicit frecvența accidentelor cu severitatea lor, în special între mediul urban și cel rural."
)

urban_df = build_area_summary_table(filtered_df)
urban_df = urban_df[urban_df["area_type"] != "Unknown"].copy()

if urban_df.empty:
    st.warning("Nu există suficiente date pentru comparația urban/rural.")
else:
    urban_row = urban_df[urban_df["area_type"] == "Urban"]
    rural_row = urban_df[urban_df["area_type"] == "Rural"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        animated_counter("Accidente Urban", int(urban_row["accidents"].iloc[0]) if not urban_row.empty else 0, color=COLORS["orange"])
    with c2:
        animated_counter("Severitate Urban", urban_row["avg_severity"].iloc[0] if not urban_row.empty else 0, decimals=2, color=COLORS["cyan"])
    with c3:
        animated_counter("Accidente Rural", int(rural_row["accidents"].iloc[0]) if not rural_row.empty else 0, color=COLORS["lime"])
    with c4:
        animated_counter("Severitate Rural", rural_row["avg_severity"].iloc[0] if not rural_row.empty else 0, decimals=2, color=COLORS["pink"])

    left, right = st.columns(2)

    with left:
        fig_freq = go.Figure()
        fig_freq.add_trace(
            go.Bar(
                x=urban_df["area_type"],
                y=urban_df["accidents"],
                marker_color=[COLORS["orange"], COLORS["lime"]],
                name="Număr accidente",
            )
        )
        fig_freq.update_layout(
            template="plotly_dark",
            title="Număr accidente: urban vs rural",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111822",
            margin=dict(l=20, r=20, t=45, b=20),
            height=400,
        )
        plotly_scroll(fig_freq, height=400)

    with right:
        fig_sev = go.Figure()
        fig_sev.add_trace(
            go.Bar(
                x=urban_df["area_type"],
                y=urban_df["avg_severity"],
                name="Severitate medie",
                marker_color=COLORS["cyan"],
            )
        )
        fig_sev.add_trace(
            go.Scatter(
                x=urban_df["area_type"],
                y=urban_df["severe_share"],
                name="Pondere grave (%)",
                mode="lines+markers",
                line=dict(color=COLORS["pink"], width=3),
                yaxis="y2",
            )
        )
        fig_sev.update_layout(
            template="plotly_dark",
            title="Severitate medie și pondere accidente grave",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111822",
            margin=dict(l=20, r=20, t=45, b=20),
            height=400,
            yaxis=dict(title="Severitate medie"),
            yaxis2=dict(
                title="Pondere grave (%)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )
        plotly_scroll(fig_sev, height=400)

    pts = frequency_vs_severity_points(grid_df)
    plotly_scroll(
        scatter_frequency_vs_severity(pts),
        height=470,
    )

    map_left, map_right = st.columns(2)
    with map_left:
        plotly_scroll(
            map_metric_frequency(grid_df, "Hartă frecvență", height=520),
        )
    with map_right:
        plotly_scroll(
            map_metric_severity(grid_df, "Hartă severitate", height=520),
        )

    st.dataframe(
        urban_df,
        width="stretch",
        column_config={
            "area_type": "Tip zonă",
            "accidents": st.column_config.NumberColumn("Nr. accidente"),
            "avg_severity": st.column_config.NumberColumn("Severitate medie", format="%.2f"),
            "severe_share": st.column_config.NumberColumn("Pondere grave (%)", format="%.2f"),
            "avg_speed": st.column_config.NumberColumn("Limită medie viteză", format="%.2f"),
        },
        hide_index=True,
    )


st.markdown("---")

# =========================================================
# 4. HOTSPOT-URI
# =========================================================
section_anchor("sec-4")
section_header(
    "4. Analiza spațială și hotspot-uri",
    "Zonele periculoase sunt identificate prin frecvență, severitate și scor compozit."
)

top_df = top_hotspots(grid_df, top_n=12, by="composite_score").copy()
top_df["avg_severity"] = top_df["avg_severity"].round(2)
top_df["severe_share"] = (top_df["severe_share"] * 100).round(2)
top_df["composite_score"] = top_df["composite_score"].round(2)

c1, c2, c3, c4 = st.columns(4)
with c1:
    animated_counter("Celule analizate", len(grid_df), color=COLORS["orange"])
with c2:
    animated_counter("Scor maxim hotspot", grid_df["composite_score"].max(), decimals=1, color=COLORS["cyan"])
with c3:
    animated_counter("Severitate medie maximă", grid_df["avg_severity"].max(), decimals=2, color=COLORS["lime"])
with c4:
    animated_counter("Pondere max. grave", grid_df["severe_share"].max() * 100, decimals=2, suffix="%", color=COLORS["pink"])

left, right = st.columns([1.5, 1.0])

with left:
    plotly_scroll(
        map_hotspots(grid_df, "composite_score", "Hartă hotspot-uri", height=760),
    )

with right:
    st.markdown("#### Top hotspot-uri")
    st.dataframe(
        top_df,
        width="stretch",
        column_config={
            "lat_center": st.column_config.NumberColumn("Lat", format="%.4f"),
            "lon_center": st.column_config.NumberColumn("Lon", format="%.4f"),
            "accidents_count": st.column_config.NumberColumn("Accidente"),
            "avg_severity": st.column_config.NumberColumn("Severitate medie", format="%.2f"),
            "severe_share": st.column_config.NumberColumn("Pondere grave (%)", format="%.2f"),
            "composite_score": st.column_config.ProgressColumn("Scor compozit", min_value=0, max_value=100, format="%.2f"),
        },
        hide_index=True,
    )

    plotly_scroll(
        bar_chart(
            top_df.reset_index().rename(columns={"index": "rank"}),
            "rank",
            "composite_score",
            COLORS["pink"],
            "Top hotspot-uri după scor compozit",
            height=320,
        ),
        height=320,
    )


st.markdown("---")

# =========================================================
# 5. PREVENȚIE
# =========================================================
section_anchor("sec-5")
section_header(
    "5. Prevenție: intersecții cu risc",
    "Intersecțiile sunt detectate din rețeaua rutieră reală, iar accidentele sunt asociate folosind raze de 50 m ajustate automat când intersecțiile sunt foarte apropiate."
)

place_name = st.text_input(
    "Zonă pentru rețeaua rutieră",
    value=st.session_state.get("prevention_place_name", "County Durham, England"),
    placeholder="Ex.: County Durham, England / London, England / Manchester, England",
    help="Momentan aplicația descarcă rețeaua rutieră din OpenStreetMap la prima rulare pentru zona scrisă aici. Cache-ul offline complet îl facem într-un update separat.",
    key="prevention_place_name",
)


prevention_df = df  # Secțiunea 5 folosește mereu tot datasetul.


run_prevention = st.button("Rulează analiza pentru zona aleasă")

if run_prevention:
    with st.spinner("Se pregătește analiza de prevenție..."):
        started_s5 = time.perf_counter()
        try:
            phase_timings = {}

            t_phase = time.perf_counter()
            intersections_metric = get_intersections_for_place(place_name)
            phase_timings["0_incarcare_retea_osm_s"] = time.perf_counter() - t_phase

            accidents_assigned, intersection_stats_metric, local_timings = _parallel_prevention_pipeline_timed(
                prevention_df,
                intersections_metric,
                n_jobs=parallel_jobs,
            )
            phase_timings.update(local_timings)

            t_phase = time.perf_counter()
            intersection_stats_wgs84 = intersections_back_to_wgs84(intersection_stats_metric)
            place_center_lat, place_center_lon = _gdf_bounds_center(intersection_stats_wgs84)
            phase_timings["5_conversie_harta_s"] = time.perf_counter() - t_phase

            elapsed_s5 = time.perf_counter() - started_s5
            phase_timings["total_rulare_s"] = elapsed_s5

            _record_runtime_event(
                section="5",
                operation="Prevenție: intersecții cu risc",
                elapsed_s=elapsed_s5,
                workers=parallel_jobs,
                rows=len(prevention_df),
                details={
                    "place": place_name,
                    "intersections": int(len(intersection_stats_metric)),
                    "osm_network_s": round(phase_timings.get("0_incarcare_retea_osm_s", 0.0), 3),
                    "local_compute_s": round(phase_timings.get("total_calcul_local_s", 0.0), 3),
                    "kdtree_assignment_s": round(phase_timings.get("2_asociere_kdtree_s", 0.0), 3),
                    "method": "KDTree vectorizat + workers; timpi separați pe pași",
                },
            )

            st.session_state["prevention_results"] = {
                "intersection_stats_wgs84": intersection_stats_wgs84,
                "accidents_assigned": accidents_assigned,
                "runtime_s": elapsed_s5,
                "workers": parallel_jobs,
                "phase_timings": phase_timings,
                "place_name": place_name,
                "place_center_lat": place_center_lat,
                "place_center_lon": place_center_lon,
                "map_focus_lat": place_center_lat,
                "map_focus_lon": place_center_lon,
                "map_focus_zoom": 10,
            }
            st.success("Analiza de prevenție a fost finalizată.")
        except Exception as exc:
            st.error(f"Eroare la construirea modelului de prevenție: {exc}")

if "prevention_results" in st.session_state:
    pr = st.session_state["prevention_results"]

    inter_df = pr["intersection_stats_wgs84"]
    accidents_assigned = pr["accidents_assigned"]

    total_intersections_flagged = int((inter_df["accidents_count"] > 0).sum()) if not inter_df.empty else 0
    assigned_share = float(accidents_assigned["assigned_to_intersection"].mean() * 100) if not accidents_assigned.empty else 0.0
    unassigned_share = max(0.0, 100.0 - assigned_share)

    c1, c2, c3 = st.columns(3)
    with c1:
        animated_counter("Intersecții problematice", total_intersections_flagged, color=COLORS["orange"])
    with c2:
        animated_counter("Accidente asociate intersecțiilor", assigned_share, suffix="%", decimals=2, color=COLORS["pink"])
    with c3:
        animated_counter("Accidente neasociate", unassigned_share, suffix="%", decimals=2, color=COLORS["cyan"])

    focus_lat = pr.get("map_focus_lat")
    focus_lon = pr.get("map_focus_lon")
    focus_zoom = pr.get("map_focus_zoom", 10)

    fig_prevention = prevention_map(
        inter_df,
        None,
        center_lat=focus_lat,
        center_lon=focus_lon,
        height=720,
    )
    fig_prevention.update_layout(mapbox_zoom=focus_zoom)
    plotly_scroll(fig_prevention)

    st.markdown("#### Top intersecții cu risc")
    inter_full = inter_df[inter_df["accidents_count"] > 0].copy().sort_values(
        "composite_score", ascending=False
    ).head(20)

    if inter_full.empty:
        st.info("Nu există intersecții cu accidente asociate pentru zona aleasă.")
    else:
        inter_full["lat"] = inter_full.geometry.y
        inter_full["lon"] = inter_full.geometry.x
        inter_top = inter_full[
            [
                "intersection_id",
                "accidents_count",
                "avg_severity",
                "severity_interpretation",
                "adaptive_radius_m",
                "composite_score",
                "lat",
                "lon",
            ]
        ].reset_index(drop=True)


        selected_row_idx = None
        try:
            table_event = st.dataframe(
                inter_top,
                width="stretch",
                column_config={
                    "intersection_id": "ID intersecție",
                    "accidents_count": st.column_config.NumberColumn("Accidente"),
                    "avg_severity": st.column_config.NumberColumn("Severitate", format="%.2f"),
                    "severity_interpretation": "Interpretare",
                    "adaptive_radius_m": st.column_config.NumberColumn("Rază asociere (m)", format="%.1f"),
                    "composite_score": st.column_config.ProgressColumn("Scor", min_value=0, max_value=100, format="%.2f"),
                    "lat": st.column_config.NumberColumn("Lat", format="%.5f"),
                    "lon": st.column_config.NumberColumn("Lon", format="%.5f"),
                },
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="intersection_select_table",
            )
            selected_rows = table_event.selection.rows if hasattr(table_event, "selection") else []
            if selected_rows:
                selected_row_idx = int(selected_rows[0])
        except TypeError:
            st.dataframe(
                inter_top,
                width="stretch",
                column_config={
                    "intersection_id": "ID intersecție",
                    "accidents_count": st.column_config.NumberColumn("Accidente"),
                    "avg_severity": st.column_config.NumberColumn("Severitate", format="%.2f"),
                    "severity_interpretation": "Interpretare",
                    "adaptive_radius_m": st.column_config.NumberColumn("Rază asociere (m)", format="%.1f"),
                    "composite_score": st.column_config.ProgressColumn("Scor", min_value=0, max_value=100, format="%.2f"),
                    "lat": st.column_config.NumberColumn("Lat", format="%.5f"),
                    "lon": st.column_config.NumberColumn("Lon", format="%.5f"),
                },
                hide_index=True,
            )

        labels = [
            f"#{i + 1} | ID {row['intersection_id']} | scor {row['composite_score']:.1f} | accidente {int(row['accidents_count'])}"
            for i, row in inter_top.iterrows()
        ]
        selected_label = st.selectbox(
            "Mută harta pe intersecția:",
            options=["—"] + labels,
            index=0,
            key="intersection_focus_selectbox",
        )
        if selected_label != "—":
            selected_row_idx = labels.index(selected_label)

        if selected_row_idx is not None:
            selected = inter_top.iloc[selected_row_idx]
            new_lat = float(selected["lat"])
            new_lon = float(selected["lon"])
            old_lat = st.session_state["prevention_results"].get("map_focus_lat")
            old_lon = st.session_state["prevention_results"].get("map_focus_lon")
            st.session_state["prevention_results"]["map_focus_lat"] = new_lat
            st.session_state["prevention_results"]["map_focus_lon"] = new_lon
            st.session_state["prevention_results"]["map_focus_zoom"] = 14
            st.success(
                f"Harta este focalizată pe intersecția {selected['intersection_id']} "
                f"({selected['lat']:.5f}, {selected['lon']:.5f})."
            )
            if old_lat != new_lat or old_lon != new_lon:
                st.rerun()


st.markdown("---")

# =========================================================
# 6. RISC PE TRASEU
# =========================================================
section_anchor("sec-6")
section_header(
    "6. Aplicația de risc pe traseu",
    "Risc pe segmente pentru ruta selectată."
)

# Setări metodologice fixe pentru traseu. Le ținem în cod ca interfața să rămână simplă.
ROUTE_BUFFER_M = 40.0
ROUTE_MAX_SEGMENTS = 10
ROUTE_MIN_SEGMENT_LENGTH_KM = 5.0
ROUTE_COLOR_MODE = "composite_score"
route_df = df  # Secțiunea 6 folosește mereu tot datasetul pentru consistență.
route_parallel_jobs = parallel_jobs


def _risk_level(score: float) -> str:
    try:
        score = float(score)
    except Exception:
        return "necunoscut"
    if score < 30:
        return "scăzut"
    if score < 60:
        return "mediu"
    return "ridicat"


def _severity_route_label(value: float) -> str:
    try:
        value = float(value)
    except Exception:
        return "necunoscut"
    if value < 1.5:
        return "preponderent ușoară"
    if value < 2.3:
        return "mixtă / serious influențează scorul"
    return "foarte gravă / fatal influențează scorul"


# =========================================================
# RISC PE TRASEU - variantă strictă pe segmente
# =========================================================
# Problema observată: dacă severitatea se calculează ca sumă pe km,
# segmentele urbane pot primi scor mare doar pentru că au multe accidente.
# Corecție: severitatea este calculată ca MEDIE PONDERATĂ a accidentelor
# apropiate de traseu, iar frecvența rămâne separată.
#
# Folosim o rază strictă de 40 m în jurul traseului.
# Accidentele din această rază intră în calcul cu aceeași pondere;
# nu mai folosim factor de apropiere față de traseu.
# Formula calibrată pentru traseu/segmente:
# 45% frecvență și 55% severitate medie.
ROUTE_FREQUENCY_WEIGHT = 0.45
ROUTE_SEVERITY_WEIGHT = 0.55
ROUTE_SEVERE_SHARE_WEIGHT = 0.00
ROUTE_FREQUENCY_SCALE_PER_KM = 2.50


def _to_metric_accidents(df_in: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = [Point(lon, lat) for lon, lat in zip(df_in["longitude"], df_in["latitude"])]
    gdf = gpd.GeoDataFrame(df_in.copy(), geometry=geometry, crs="EPSG:4326")
    return gdf.to_crs(epsg=3857)


def _route_to_metric_strict(line: LineString) -> LineString:
    return gpd.GeoSeries([line], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]


def _distance_weight_strict(distance_m: float, max_distance_m: float) -> float:
    """Compatibilitate pentru funcții vechi: în modelul actual nu mai ponderăm după distanță."""
    return 1.0 if float(distance_m) <= float(max_distance_m) else 0.0


def _context_from_nearby(nearby: pd.DataFrame) -> dict:
    if nearby.empty or "area_type" not in nearby.columns:
        return {"urban_share": 0.0, "rural_share": 0.0, "dominant_area": "Unknown"}
    urban_share = float((nearby["area_type"] == "Urban").mean() * 100)
    rural_share = float((nearby["area_type"] == "Rural").mean() * 100)
    if urban_share >= 60:
        dominant = "Urban"
    elif rural_share >= 60:
        dominant = "Rural"
    else:
        dominant = "Mixt"
    return {"urban_share": urban_share, "rural_share": rural_share, "dominant_area": dominant}


def _strict_segment_scores(
    nearby: gpd.GeoDataFrame,
    length_km: float,
    distance_col: str,
    max_distance_m: float,
) -> dict:
    """
    Calculează scoruri fără să lase frecvența să crească artificial severitatea.

    - frequency_score: câte accidente sunt în raza de 40 m, raportat la km.
    - severity_score: media severității accidentelor din raza de 40 m.
    - severe_share: procent de accidente Serious/Fatal.
    - composite_score: scor final = 55% frecvență + 30% severitate + 15% accidente grave.
    """
    if nearby.empty:
        ctx = _context_from_nearby(nearby)
        return {
            "accidents_count": 0,
            "frequency_score": 0.0,
            "severity_score": 0.0,
            "avg_severity": 0.0,
            "avg_severity_display": 0.0,
            "severe_share": 0.0,
            "composite_score": 0.0,
            "urban_share": ctx["urban_share"],
            "rural_share": ctx["rural_share"],
            "dominant_area": ctx["dominant_area"],
        }

    nearby = nearby.copy()
    if distance_col in nearby.columns:
        nearby = nearby[nearby[distance_col] <= max_distance_m].copy()

    if nearby.empty:
        return _strict_segment_scores(nearby, length_km, distance_col, max_distance_m)

    accidents_count = int(len(nearby))
    freq_per_km = accidents_count / max(length_km, 0.001)

    # Severitatea este medie simplă pentru accidentele intrate în raza de 40 m.
    # Folosim scala afișată: Slight=1, Serious=2, Fatal=3.
    avg_severity_display = float(nearby["severity_display"].mean())
    severe_share = float(nearby["severe_flag"].mean() * 100)

    # Frecvența/densitatea este componenta principală.
    # Formula crește odată cu accidentele/km și se plafonează spre 100.
    frequency_score = 100.0 * (1.0 - math.exp(-freq_per_km / ROUTE_FREQUENCY_SCALE_PER_KM))
    frequency_score = min(100.0, max(0.0, frequency_score))

    # Severitatea este medie simplă, nu sumă.
    # Scala afișată: Slight=1 -> 0, Serious=2 -> 50, Fatal=3 -> 100.
    severity_score = (avg_severity_display - 1.0) / 2.0 * 100.0
    severity_score = min(100.0, max(0.0, severity_score))

    composite_score = (
        ROUTE_FREQUENCY_WEIGHT * frequency_score
        + ROUTE_SEVERITY_WEIGHT * severity_score
    )

    ctx = _context_from_nearby(nearby)
    return {
        "accidents_count": int(len(nearby)),
        "frequency_score": float(frequency_score),
        "severity_score": float(severity_score),
        "avg_severity": float(avg_severity_display),
        "avg_severity_display": float(avg_severity_display),
        "severe_share": float(severe_share),
        "composite_score": float(min(100.0, composite_score)),
        "urban_share": ctx["urban_share"],
        "rural_share": ctx["rural_share"],
        "dominant_area": ctx["dominant_area"],
    }


def calculate_route_risk(accidents_df: pd.DataFrame, route_line: LineString, buffer_m: float = 40.0) -> dict:
    accidents_gdf = _to_metric_accidents(accidents_df)
    route_metric = _route_to_metric_strict(route_line)
    route_length_km = max(route_metric.length / 1000, 0.001)

    nearby = accidents_gdf[accidents_gdf.geometry.within(route_metric.buffer(buffer_m))].copy()
    if not nearby.empty:
        nearby["distance_to_route"] = nearby.geometry.distance(route_metric)

    scores = _strict_segment_scores(nearby, route_length_km, "distance_to_route", buffer_m)
    scores["route_length_km"] = float(route_length_km)
    scores["risk_model"] = "strict_segment_severity_mean"
    return scores


def _auto_segment_count(
    route_length_m: float,
    max_segments: int = ROUTE_MAX_SEGMENTS,
    min_segment_length_km: float = ROUTE_MIN_SEGMENT_LENGTH_KM,
) -> int:
    """Alege automat numărul de segmente pentru traseu.

    Regula metodologică:
    - încercăm să folosim maximum 10 segmente;
    - niciun segment nu trebuie să fie sub 5 km, dacă traseul are cel puțin 5 km;
    - pentru trasee mai scurte de 5 km, păstrăm un singur segment.

    Exemple:
    - 35 km -> 7 segmente de 5 km;
    - 52 km -> 10 segmente de 5.2 km;
    - 4 km  -> 1 segment de 4 km.
    """
    max_segments = max(1, int(max_segments))
    min_segment_length_m = max(float(min_segment_length_km) * 1000.0, 1.0)
    if route_length_m <= min_segment_length_m:
        return 1
    return max(1, min(max_segments, int(route_length_m // min_segment_length_m)))


def split_line_into_n_segments_strict(line_metric: LineString, n_segments: int) -> list[LineString]:
    if n_segments <= 1:
        return [line_metric]
    total_length = line_metric.length
    return [
        substring(line_metric, total_length * i / n_segments, total_length * (i + 1) / n_segments)
        for i in range(n_segments)
    ]


def _segment_metrics_strict(seg_metric: LineString, accidents_gdf: gpd.GeoDataFrame, buffer_m: float, idx: int) -> dict:
    seg_length_km = max(seg_metric.length / 1000, 0.001)
    nearby = accidents_gdf[accidents_gdf.geometry.within(seg_metric.buffer(buffer_m))].copy()
    if not nearby.empty:
        nearby["dist"] = nearby.geometry.distance(seg_metric)
    scores = _strict_segment_scores(nearby, seg_length_km, "dist", buffer_m)
    seg_wgs84 = gpd.GeoSeries([seg_metric], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
    scores.update({
        "segment_id": idx,
        "segment_length_km": float(seg_length_km),
        "geometry": seg_wgs84,
    })
    return scores


def calculate_segment_risks(
    accidents_df: pd.DataFrame,
    route_line: LineString,
    buffer_m: float = 40.0,
    mode: str = "auto",
    segment_value: int | float = ROUTE_MAX_SEGMENTS,
    min_segment_length_km: float = ROUTE_MIN_SEGMENT_LENGTH_KM,
    n_jobs: int = 1,
) -> list[dict]:
    accidents_gdf = _to_metric_accidents(accidents_df)
    route_metric = _route_to_metric_strict(route_line)

    if mode == "count":
        n_segments = int(segment_value)
    else:
        safe_segment_value = ROUTE_MAX_SEGMENTS if segment_value is None else segment_value
        n_segments = _auto_segment_count(
            route_metric.length,
            max_segments=int(safe_segment_value),
            min_segment_length_km=float(min_segment_length_km),
        )

    segments = split_line_into_n_segments_strict(route_metric, n_segments)
    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_segment_metrics_strict)(seg, accidents_gdf, buffer_m, idx)
        for idx, seg in enumerate(segments, start=1)
    )


input_col1, input_col2 = st.columns(2)
with input_col1:
    start_location = st.text_input("Locație plecare", "Durham")
with input_col2:
    end_location = st.text_input("Locație destinație", "Sunderland")

if st.button("Calculează traseul și riscul"):
    with st.spinner("Se pregătește analiza traseului..."):
        total_started_s6 = time.perf_counter()

        t0 = time.perf_counter()
        start_coords = geocode_location(start_location)
        end_coords = geocode_location(end_location)
        geocoding_s = time.perf_counter() - t0

        if not start_coords or not end_coords:
            st.error("Nu s-au putut geocoda locațiile. Verifică numele introduse.")
            st.stop()

        t0 = time.perf_counter()
        routes = get_routes(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
        routing_s = time.perf_counter() - t0
        if not routes:
            st.error("Nu s-a putut obține traseul.")
            st.stop()

        selected_route = routes[0]
        route_line = selected_route["line"]

        t0 = time.perf_counter()
        global_risk = calculate_route_risk(route_df, route_line, buffer_m=ROUTE_BUFFER_M)
        global_risk_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        segment_results = calculate_segment_risks(
            route_df,
            route_line,
            buffer_m=ROUTE_BUFFER_M,
            mode="auto",
            segment_value=ROUTE_MAX_SEGMENTS,
            n_jobs=route_parallel_jobs,
        )
        segments_s = time.perf_counter() - t0

        attention_points = []
        attention_s = 0.0

        total_elapsed_s6 = time.perf_counter() - total_started_s6
        local_calc_s6 = global_risk_s + segments_s

        timing_details_s6 = {
            "geocoding_s": round(float(geocoding_s), 3),
            "routing_s": round(float(routing_s), 3),
            "global_risk_s": round(float(global_risk_s), 3),
            "segments_s": round(float(segments_s), 3),
            "local_calc_s": round(float(local_calc_s6), 3),
            "total_s": round(float(total_elapsed_s6), 3),
        }

        _record_runtime_event(
            section="6",
            operation="Risc pe traseu",
            elapsed_s=total_elapsed_s6,
            workers=route_parallel_jobs,
            rows=len(route_df),
            details={
                "route": f"{start_location} -> {end_location}",
                "segments": int(len(segment_results)),
                "attention_points": 0,
                "local_calc_s": round(float(local_calc_s6), 3),
                "method": "global + segmente",
                "includes_external": "geocoding + OSRM routing",
            },
        )

        st.session_state["route_results"] = {
            "start_coords": start_coords,
            "end_coords": end_coords,
            "route_line": route_line,
            "global_risk": global_risk,
            "segment_results": segment_results,
            "attention_points": attention_points,
            "color_mode": ROUTE_COLOR_MODE,
            "runtime_s": total_elapsed_s6,
            "local_calc_s": local_calc_s6,
            "workers": route_parallel_jobs,
            "timing_details": timing_details_s6,
        }
        st.success("Analiza traseului a fost finalizată.")

if "route_results" in st.session_state:
    rr = st.session_state["route_results"]
    gr = rr["global_risk"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distanță traseu", f"{gr['route_length_km']:.2f} km")
    c2.metric("Accidente apropiate", int(gr["accidents_count"]))
    c3.metric("Scor risc", f"{gr['composite_score']:.1f}/100")
    c4.metric("Nivel risc", _risk_level(gr["composite_score"]).upper())


    plotly_scroll(
        route_map(
            rr["start_coords"],
            rr["end_coords"],
            rr["route_line"],
            segments=rr["segment_results"],
            attention_points=None,
            color_mode=rr["color_mode"],
            height=760,
        ),
    )

    st.markdown("### Segmente traseu")
    seg_df = pd.DataFrame([
        {
            "segment_id": s["segment_id"],
            "segment_length_km": round(s["segment_length_km"], 2),
            "accidents_count": int(s["accidents_count"]),
            "avg_severity_display": round(s.get("avg_severity_display", 0), 2),
            "composite_score": round(s["composite_score"], 1),
            "nivel_risc": _risk_level(s["composite_score"]),
        }
        for s in rr["segment_results"]
    ])

    st.dataframe(
        seg_df,
        width="stretch",
        column_config={
            "segment_id": "Segment",
            "segment_length_km": st.column_config.NumberColumn("Lungime (km)", format="%.2f"),
            "accidents_count": st.column_config.NumberColumn("Accidente apropiate"),
            "avg_severity_display": st.column_config.NumberColumn("Severitate", format="%.2f", help="Scală simplă: 1 = ușor, 2 = grav, 3 = fatal"),
            "composite_score": st.column_config.ProgressColumn("Scor risc", min_value=0, max_value=100, format="%.1f"),
            "nivel_risc": "Nivel risc",
        },
        hide_index=True,
    )


