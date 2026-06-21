from __future__ import annotations

import os
import time
from pathlib import Path
from statistics import mean, stdev

import pandas as pd
import streamlit as st

from src.data_loader import load_accidents
from src.performance import candidate_worker_list, auto_runtime_config
from src.parallel_risk import make_synthetic_route, run_route_dask, run_route_joblib
from src.routing import geocode_location, get_routes
from src.theme import load_theme, section_header, info_card

st.set_page_config(page_title="RoadSafe Benchmark Secțiunea 6", layout="wide", page_icon="🧪")
load_theme()

DATA_PATH = "data/accidents.csv"


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_accidents(DATA_PATH)


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
        "task": "Secțiunea 6 - risc traseu",
        "backend": label,
        "status": "ok" if not error else "eroare",
        "repeats": len(times),
        "mean_s": round(mean(times), 4) if times else None,
        "std_s": round(stdev(times), 4) if len(times) > 1 else 0.0 if times else None,
        "min_s": round(min(times), 4) if times else None,
        "max_s": round(max(times), 4) if times else None,
        "segments": len(last.segment_results) if last else None,
        "attention_points": len(last.attention_points) if last else None,
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
            if row["speedup_vs_baseline"] is not None else None,
            axis=1,
        )
    return out


st.title("RoadSafe — Benchmark Secțiunea 6")
st.caption("Benchmark separat pentru risc pe traseu: risc pe segmente și puncte critice.")

if not Path(DATA_PATH).exists():
    st.error("Nu există data/accidents.csv. Copiază datasetul în folderul data și rulează din nou.")
    st.stop()

df = get_data()

section_header("1. Configurare test", "Alege traseul, volumul de date și backend-urile comparate.")

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
    # Aceeași estimare ca în aplicația principală: dataset complet, Joblib, cap 8.
    cfg = auto_runtime_config(len(df), task="generic", user_cap=user_cap, prefer_distributed=False)
    st.metric("Estimare automată", f"{cfg.n_jobs} workeri")

route_source = st.radio(
    "Sursă traseu",
    ["Sintetic offline", "OSRM real"],
    horizontal=True,
    help="Sintetic offline este recomandat pentru benchmark rapid. OSRM real depinde de internet.",
)

if route_source == "OSRM real":
    r1, r2 = st.columns(2)
    with r1:
        start_name = st.text_input("Start", "Leeds")
    with r2:
        end_name = st.text_input("Destinație", "Bradford")
else:
    start_name = end_name = "traseu sintetic"

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

p1, p2, p3 = st.columns(3)
with p1:
    buffer_m = 40.0
st.info("Benchmark-ul folosește aceeași rază ca aplicația principală: 40 m în jurul traseului, fără factor de apropiere.")
with p2:
    segment_value = st.slider("Număr segmente", 4, 80, 16)
with p3:
    sample_step_m = st.slider("Pas puncte critice (m)", 50, 500, 200, step=50)

attention_radius_m = st.slider("Rază punct critic (m)", 50, 500, 150, step=50)
min_attention_score = st.slider("Scor minim punct critic", 1.0, 30.0, 8.0, step=1.0)
top_points = st.slider("Top puncte critice", 1, 20, 5)

sample_df = df.sample(min(max_rows, len(df)), random_state=42).copy()

section_header("2. Ce se măsoară", "Benchmark-ul măsoară calculul local al riscului pe traseu.")
info_card(
    "Pași incluși",
    "Conversie geometrii, împărțire traseu în segmente, calcul risc pe segment și detectare puncte critice. Pentru OSRM real, obținerea traseului este făcută înainte de testul de calcul.",
)

if st.button("Rulează benchmark secțiunea 6", type="primary"):
    with st.spinner("Pregătesc traseul..."):
        if route_source == "OSRM real":
            start = geocode_location(start_name)
            end = geocode_location(end_name)
            if start is None or end is None:
                st.error("Nu am putut geocoda startul sau destinația.")
                st.stop()
            routes = get_routes(start[0], start[1], end[0], end[1])
            if not routes:
                st.error("OSRM nu a returnat niciun traseu.")
                st.stop()
            route_line = routes[0]["line"]
        else:
            route_line = make_synthetic_route(sample_df)

    rows = []
    for w in sorted(set(int(x) for x in workers)):
        if test_joblib:
            rows.append({
                **timed_repeats(
                    "Joblib",
                    lambda w=w: run_route_joblib(
                        sample_df, route_line, buffer_m=buffer_m, segment_mode="count", segment_value=segment_value,
                        sample_step_m=sample_step_m, attention_radius_m=attention_radius_m,
                        min_attention_score=min_attention_score, top_points=top_points, workers=w,
                    ),
                    repeats,
                ),
                "workers": w,
            })
        if test_dask:
            rows.append({
                **timed_repeats(
                    "Dask",
                    lambda w=w: run_route_dask(
                        sample_df, route_line, buffer_m=buffer_m, segment_mode="count", segment_value=segment_value,
                        sample_step_m=sample_step_m, attention_radius_m=attention_radius_m,
                        min_attention_score=min_attention_score, top_points=top_points, workers=w,
                    ),
                    repeats,
                ),
                "workers": w,
            })

    results = add_speedup(pd.DataFrame(rows))
    st.session_state["section6_benchmark_results"] = results

if "section6_benchmark_results" in st.session_state:
    results = st.session_state["section6_benchmark_results"]
    section_header("3. Rezultate", "Comparație între backend-uri și numărul de workeri.")
    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button(
        "Descarcă rezultate CSV",
        results.to_csv(index=False).encode("utf-8"),
        "benchmark_section6_route.csv",
        "text/csv",
    )

section_header("4. Explicație optimizări", "Ce înseamnă fiecare mod testat.")
st.markdown(
    """
**Joblib** paralelizează calculul riscului pe segmente. Fiecare segment poate fi analizat separat, deci este o potrivire naturală pentru secțiunea 6. În aplicația principală, Joblib rămâne alegerea de bază.

**Dask** programează segmentele ca task-uri independente. Este bun pentru demonstrarea ideii de scheduler și execuție distribuită, dar pentru segmente puține poate fi mai lent decât Joblib din cauza overhead-ului.

**Paralelism pe segmente** înseamnă că ruta este împărțită în bucăți, iar fiecare bucată calculează accidentele apropiate, severitatea și scorul de risc independent de celelalte.

**Punctele critice** sunt eșantionate de-a lungul traseului. La fiecare punct se caută accidente într-o rază locală și se păstrează doar zonele care depășesc pragul minim de severitate.

**Raza traseului** este fixă: 40 m. Benchmark-ul folosește aceeași logică precum aplicația principală: accidentele intră sau nu intră în calcul, fără factor de apropiere.

**Numărul de workeri** trebuie testat. Pentru 8 segmente, 16 workeri nu au sens. Pentru 80 de segmente și dataset mare, mai mulți workeri pot ajuta.
"""
)
