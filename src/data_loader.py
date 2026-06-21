from __future__ import annotations

from pathlib import Path
import pandas as pd

# Codurile standard din dataset:
# 1 = Fatal, 2 = Serious, 3 = Slight
SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight"}
SEVERITY_SCORE_MAP = {1: 8, 2: 4, 3: 1}
SEVERITY_DISPLAY_MAP = {3: 1, 2: 2, 1: 3}
URBAN_RURAL_LABELS = {1: "Urban", 2: "Rural"}

WEATHER_MAP = {
    1: "Fine, no high winds",
    2: "Raining, no high winds",
    3: "Snowing, no high winds",
    4: "Fine, high winds",
    5: "Raining, high winds",
    6: "Snowing, high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown",
    -1: "Missing/Unknown",
}

LIGHT_MAP = {
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
    -1: "Missing/Unknown",
}

# Singurele coloane din accidents.csv folosite de proiect.
RELEVANT_COLUMNS = [
    "latitude",
    "longitude",
    "collision_severity",
    "number_of_casualties",
    "speed_limit",
    "date",
    "time",
    "urban_or_rural_area",
    "weather_conditions",
    "light_conditions",
]


def _read_only_relevant_columns(csv_path: str | Path) -> pd.DataFrame:
    """Citește direct accidents.csv, doar cu coloanele relevante dacă există.

    Nu folosește cache, Parquet sau CSV intermediar.
    """
    csv_path = Path(csv_path)
    header = pd.read_csv(csv_path, nrows=0)
    available_cols = [c.strip() for c in header.columns]
    usecols = [c for c in RELEVANT_COLUMNS if c in available_cols]
    return pd.read_csv(csv_path, usecols=usecols)


def optimize_accident_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memoria ocupată de dataframe după citirea CSV-ului."""
    df = df.copy()

    for c in ["latitude", "longitude", "severity_score", "severity_display", "frequency_component", "severity_component"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    for c in ["collision_year", "number_of_casualties", "speed_limit"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int16")

    for c in [
        "collision_severity",
        "month",
        "hour",
        "weekday_num",
        "severe_flag",
        "urban_or_rural_area",
        "weather_conditions",
        "light_conditions",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype("int8")

    for c in ["weekday", "severity_label", "area_type", "weather_label", "light_label"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df


def load_accidents(csv_path: str) -> pd.DataFrame:
    """Încarcă strict data/accidents.csv și creează coloanele derivate.

    Important: funcția nu caută accidents_clean.csv, nu folosește cache și nu creează fișiere.
    """
    df = _read_only_relevant_columns(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = [
        "latitude",
        "longitude",
        "collision_severity",
        "number_of_casualties",
        "speed_limit",
        "date",
        "time",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloanele: {missing}")

    numeric_cols = [
        "latitude",
        "longitude",
        "collision_severity",
        "number_of_casualties",
        "speed_limit",
        "urban_or_rural_area",
        "weather_conditions",
        "light_conditions",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "collision_severity", "date", "time"]).copy()
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)].copy()

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).copy()

    df["collision_year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["weekday_num"] = df["datetime"].dt.weekday
    df["weekday"] = df["datetime"].dt.day_name()

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["weekday"] = pd.Categorical(df["weekday"], categories=weekday_order, ordered=True)

    df["collision_severity"] = df["collision_severity"].fillna(-1).astype(int)
    df["severity_label"] = df["collision_severity"].map(SEVERITY_LABELS).fillna("Unknown")
    df["severity_score"] = df["collision_severity"].map(SEVERITY_SCORE_MAP).fillna(1)
    df["severity_display"] = df["collision_severity"].map(SEVERITY_DISPLAY_MAP).fillna(0)
    df["severe_flag"] = df["collision_severity"].isin([1, 2]).astype(int)

    if "urban_or_rural_area" in df.columns:
        df["urban_or_rural_area"] = df["urban_or_rural_area"].fillna(-1).astype(int)
        df["area_type"] = df["urban_or_rural_area"].map(URBAN_RURAL_LABELS).fillna("Unknown")
    else:
        df["urban_or_rural_area"] = -1
        df["area_type"] = "Unknown"

    df["number_of_casualties"] = df["number_of_casualties"].fillna(0)
    df["speed_limit"] = df["speed_limit"].fillna(0)

    df["frequency_component"] = 1.0
    df["severity_component"] = df["severity_score"] + 0.2 * df["number_of_casualties"]

    if "weather_conditions" in df.columns:
        df["weather_conditions"] = df["weather_conditions"].fillna(-1).astype(int)
        df["weather_label"] = df["weather_conditions"].map(WEATHER_MAP).fillna("Unknown")
    else:
        df["weather_conditions"] = -1
        df["weather_label"] = "Unknown"

    if "light_conditions" in df.columns:
        df["light_conditions"] = df["light_conditions"].fillna(-1).astype(int)
        df["light_label"] = df["light_conditions"].map(LIGHT_MAP).fillna("Unknown")
    else:
        df["light_conditions"] = -1
        df["light_label"] = "Unknown"

    keep_columns = [
        "latitude",
        "longitude",
        "collision_severity",
        "number_of_casualties",
        "speed_limit",
        "date",
        "time",
        "urban_or_rural_area",
        "weather_conditions",
        "light_conditions",
        "datetime",
        "collision_year",
        "month",
        "hour",
        "weekday_num",
        "weekday",
        "severity_label",
        "severity_score",
        "severity_display",
        "severe_flag",
        "area_type",
        "frequency_component",
        "severity_component",
        "weather_label",
        "light_label",
    ]
    df = df[[c for c in keep_columns if c in df.columns]].copy()
    return optimize_accident_dtypes(df)


def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "years_min": int(df["collision_year"].min()),
        "years_max": int(df["collision_year"].max()),
        "avg_casualties": float(df["number_of_casualties"].mean()),
        "avg_speed_limit": float(df["speed_limit"].mean()),
        "severe_share": float(df["severe_flag"].mean()),
        "avg_severity": float(df["severity_display"].mean()),
    }
