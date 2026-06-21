from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Compatibilitate NumPy / Pandas
# -------------------------------------------------------------------
# În unele versiuni noi de NumPy, apelul np.array_split(df, n)
# pe un DataFrame poate declanșa FutureWarning legat de swapaxes.
# Modulul src.parallel_prevention folosește împărțirea pe bucăți,
# iar aici reparăm problema global, fără să schimbăm logica benchmark-ului.
_ORIGINAL_ARRAY_SPLIT = np.array_split


def _safe_array_split(ary, indices_or_sections, axis=0):
    if isinstance(ary, (pd.DataFrame, pd.Series)):
        index_chunks = _ORIGINAL_ARRAY_SPLIT(np.arange(len(ary)), indices_or_sections)
        return [ary.iloc[idx].copy() for idx in index_chunks]
    return _ORIGINAL_ARRAY_SPLIT(ary, indices_or_sections, axis=axis)


np.array_split = _safe_array_split

warnings.filterwarnings(
    "ignore",
    message=".*DataFrame.swapaxes.*deprecated.*",
    category=FutureWarning,
)

from src.data_loader import load_accidents
from src.performance import candidate_worker_list, auto_runtime_config
from src.parallel_prevention import (
    make_synthetic_intersections,
    run_prevention_dask,
    run_prevention_joblib,
)
from src.road_network import (
    compute_adaptive_radii,
    download_drive_network,
    extract_intersections_from_graph,
    intersections_to_metric,
)
from src.theme import load_theme, section_header, info_card

st.set_page_config(page_title="RoadSafe Benchmark Secțiunea 5", layout="wide", page_icon="🧪")
load_theme()

DATA_PATH = "data/accidents.csv"


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_accidents(DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_real_intersections(place_name: str):
    graph = download_drive_network(place_name)
    intersections = extract_intersections_from_graph(graph)
    intersections_metric = intersections_to_metric(intersections)
    return compute_adaptive_radii(intersections_metric)


def timed_repeats(label: str, func, repeats: int) -> dict:
    times = []
    last = None
    error = ""
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        try:
            last = func()
            times.append(time.perf_counter() - start)
        except Exception as exc:
            error = repr(exc)
            break
    return {
        "task": "Secțiunea 5 - prevenție",
        "backend": label,
        "status": "ok" if not error else "eroare",
        "repeats": len(times),
        "mean_s": round(mean(times), 4) if times else None,
        "std_s": round(stdev(times), 4) if len(times) > 1 else 0.0 if times else None,
        "min_s": round(min(times), 4) if times else None,
        "max_s": round(max(times), 4) if times else None,
        "intersections": len(last.intersection_stats) if last is not None and hasattr(last, "intersection_stats") else None,
        "assigned_accidents": int(last.accidents_assigned["assigned_to_intersection"].sum())
        if last is not None and hasattr(last, "accidents_assigned") and "assigned_to_intersection" in last.accidents_assigned
        else None,
        "error": error,
    }


def add_speedup(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["speedup_vs_baseline"] = None
    out["efficiency_pct"] = None
    ok = out[out["status"] == "ok"].copy()
    if ok.empty:
        return out

    for backend, group in ok.groupby("backend"):
        group = group.sort_values("workers")
        base = group[group["workers"] == 1]
        base = base.iloc[0] if not base.empty else group.iloc[0]
        base_time = float(base["mean_s"])
        base_workers = int(base["workers"])

        mask = (out["backend"] == backend) & out["mean_s"].notna()
        out.loc[mask, "speedup_vs_baseline"] = out.loc[mask].apply(
            lambda row: round(base_time / float(row["mean_s"]), 3) if float(row["mean_s"]) > 0 else None,
            axis=1,
        )
        out.loc[mask, "efficiency_pct"] = out.loc[mask].apply(
            lambda row: round(float(row["speedup_vs_baseline"]) / max(int(row["workers"]) / base_workers, 1) * 100, 2)
            if row["speedup_vs_baseline"] is not None
            else None,
            axis=1,
        )
    return out


st.title("RoadSafe — Benchmark Secțiunea 5")
st.caption(
    "Benchmark separat pentru prevenție: asociere accidente → intersecții cu risc. "
    "Hotspot-urile rurale au fost eliminate, ca în aplicația principală."
)

if not Path(DATA_PATH).exists():
    st.error("Nu există data/accidents.csv. Copiază datasetul în folderul data și rulează din nou.")
    st.stop()

df = get_data()

section_header("1. Configurare test", "Alege volumul de date, sursa intersecțiilor și backend-urile comparate.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    max_rows = st.slider("Rânduri folosite", 5_000, max(5_000, len(df)), min(len(df), 100_000), step=5_000)
with c2:
    repeats = st.slider("Repetări / configurație", 1, 5, 1)
with c3:
    max_cpu = min(8, os.cpu_count() or 4)
    user_cap = max_cpu
    st.metric("Limită workeri", f"{user_cap}")
with c4:
    cfg = auto_runtime_config(len(df), task="generic", user_cap=user_cap, prefer_distributed=False)
    st.metric("Estimare automată", f"{cfg.n_jobs} workeri")

source = st.radio(
    "Sursă intersecții",
    ["Sintetic offline", "OpenStreetMap real"],
    horizontal=True,
    help="Sintetic offline este recomandat pentru demonstrație rapidă. OSM real depinde de internet și de mărimea zonei.",
)

if source == "OpenStreetMap real":
    place = st.text_input("Zonă OSM", "Leeds, West Yorkshire, England")
else:
    place = "intersecții sintetice"

b1, b2, b3 = st.columns(3)
with b1:
    test_joblib = st.checkbox("Testează Joblib", value=True)
with b2:
    test_dask = st.checkbox("Testează Dask", value=True)
with b3:
    workers = st.multiselect(
        "Workeri testați",
        options=list(range(1, user_cap + 1)),
        default=candidate_worker_list(max_rows, user_cap=user_cap),
    )

sample_df = df.sample(min(max_rows, len(df)), random_state=42).copy()

section_header("2. Ce se măsoară", "Benchmark-ul măsoară doar calculul local, nu desenarea hărții.")
info_card(
    "Pași incluși",
    "Conversie coordonate în metri, asociere accidente → intersecții cu KDTree și agregare statistici pe intersecții. "
    "Pentru OSM real, descărcarea rețelei este separată și cache-uită de Streamlit.",
)

if st.button("Rulează benchmark secțiunea 5", type="primary"):
    if not workers:
        st.warning("Alege cel puțin un număr de workeri înainte să rulezi benchmark-ul.")
    else:
        with st.spinner("Pregătesc intersecțiile..."):
            if source == "OpenStreetMap real":
                intersections_metric = get_real_intersections(place)
            else:
                intersections_metric = make_synthetic_intersections(sample_df, target_points=1600)

        st.write(f"Intersecții folosite: **{len(intersections_metric):,}**")
        rows = []
        for w in sorted(set(int(x) for x in workers)):
            if test_joblib:
                rows.append(
                    {
                        **timed_repeats(
                            "Joblib",
                            lambda w=w: run_prevention_joblib(sample_df, intersections_metric, workers=w),
                            repeats,
                        ),
                        "workers": w,
                    }
                )
            if test_dask:
                rows.append(
                    {
                        **timed_repeats(
                            "Dask",
                            lambda w=w: run_prevention_dask(sample_df, intersections_metric, workers=w),
                            repeats,
                        ),
                        "workers": w,
                    }
                )

        results = add_speedup(pd.DataFrame(rows))
        st.session_state["section5_benchmark_results"] = results

if "section5_benchmark_results" in st.session_state:
    results = st.session_state["section5_benchmark_results"]
    section_header("3. Rezultate", "Comparație între backend-uri și numărul de workeri.")
    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button(
        "Descarcă rezultate CSV",
        results.to_csv(index=False).encode("utf-8"),
        "benchmark_section5_prevention.csv",
        "text/csv",
    )

section_header("4. Explicație optimizări", "Ce înseamnă fiecare mod testat.")
st.markdown(
    """
**Joblib** rulează aceeași operație pe mai multe thread-uri sau procese. În acest proiect este potrivit pentru calcule geospațiale locale, mai ales când folosim `cKDTree`, `GeoPandas` și `Shapely`. Pentru secțiunea 5, Joblib este opțiunea recomandată în aplicația principală.

**Dask** este un framework pentru calcule pe bucăți de date. Poate lucra local sau distribuit pe cluster. În acest benchmark este folosit cu `delayed`, adică împarte dataframe-ul în bucăți și programează fiecare bucată ca task separat. Este util pentru demonstrație și pentru dataseturi mari, dar poate avea overhead mai mare decât Joblib pe calcule geospațiale.

**Workeri** înseamnă fire de execuție sau procese care lucrează în paralel. Mai mulți workeri pot reduce timpul, dar pot și încetini dacă datasetul este mic, dacă memoria este limitată sau dacă se copiază multă geometrie între task-uri.

**KDTree** accelerează căutarea celui mai apropiat punct. Fără KDTree, ai compara fiecare accident cu fiecare intersecție. Cu KDTree, căutarea devine mult mai rapidă.

**Vectorizarea** înseamnă să evităm bucle Python rând-cu-rând și să lucrăm pe array-uri NumPy/Pandas. Este una dintre cele mai importante optimizări din secțiunea 5.

**Cache-ul Streamlit** evită refacerea unor operații scumpe, precum încărcarea datelor sau descărcarea intersecțiilor OSM, la fiecare rerulare a paginii.
"""
)
