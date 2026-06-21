from __future__ import annotations

import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import LineString

PLOTLY_TEMPLATE = "plotly_dark"

COLORS = {
    "orange": "#FF7A00",
    "cyan": "#00D1FF",
    "lime": "#B7FF00",
    "pink": "#FF4DA6",
    "red": "#FF3B3B",
    "amber": "#FFC857",
    "slate": "#95A4B8",
    "black": "#111111",
}


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111822",
        margin=dict(l=18, r=18, t=40, b=18),
        font=dict(color="#E8EEF5"),
    )
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str, height: int = 420) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color_discrete_sequence=[color], title=title)
    return style_fig(fig, height)


def line_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str, height: int = 420) -> go.Figure:
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_traces(line_color=color, marker_color=color)
    return style_fig(fig, height)


def grouped_bar(df: pd.DataFrame, x: str, y_cols: list[str], title: str, height: int = 440) -> go.Figure:
    fig = go.Figure()
    palette = [COLORS["orange"], COLORS["cyan"], COLORS["pink"], COLORS["lime"]]
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Bar(name=col, x=df[x], y=df[col], marker_color=palette[i % len(palette)]))
    fig.update_layout(barmode="group", title=title)
    return style_fig(fig, height)


def scatter_frequency_vs_severity(df: pd.DataFrame, height: int = 500) -> go.Figure:
    fig = px.scatter(
        df,
        x="accidents_count",
        y="avg_severity",
        size="composite_score",
        color="severe_share",
        color_continuous_scale="Viridis",
        title="Frecvență vs severitate medie",
        labels={
            "accidents_count": "Frecvență locală",
            "avg_severity": "Severitate medie",
            "severe_share": "Pondere accidente grave",
        },
    )
    return style_fig(fig, height)


def _map_center(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"lat": 54.7, "lon": -1.3}
    return {
        "lat": float(df["lat_center"].mean()),
        "lon": float(df["lon_center"].mean()),
    }


def _scale_sizes(values: pd.Series, min_size: float = 5, max_size: float = 24) -> list[float]:
    values = values.fillna(0).astype(float)
    vmax = float(values.max()) if len(values) else 0.0
    if vmax <= 0:
        return [min_size for _ in values]
    return (min_size + (values / vmax) * (max_size - min_size)).tolist()


def map_grid(grid_df: pd.DataFrame, metric: str, title: str, zoom: float = 5.2, height: int = 560) -> go.Figure:
    fig = px.scatter_mapbox(
        grid_df,
        lat="lat_center",
        lon="lon_center",
        color=metric,
        size="accidents_count",
        hover_data={
            "accidents_count": True,
            "avg_severity": ":.2f",
            "severe_share": ":.2f",
            "composite_score": ":.2f",
            "lat_center": False,
            "lon_center": False,
        },
        color_continuous_scale="Viridis",
        zoom=zoom,
        height=height,
        title=title,
    )
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_colorbar=dict(title=metric),
    )
    return fig


def map_metric_frequency(grid_df: pd.DataFrame, title: str = "Hartă frecvență", zoom: float = 5.2, height: int = 560) -> go.Figure:
    df = grid_df.copy()
    fig = go.Figure()
    if not df.empty:
        values = df["accidents_count"].fillna(0).astype(float)
        fig.add_trace(go.Scattermapbox(
            lat=df["lat_center"],
            lon=df["lon_center"],
            mode="markers",
            marker=dict(
                size=_scale_sizes(values, 5, 20),
                color=values,
                colorscale="YlGnBu",
                opacity=0.72,
                showscale=True,
                colorbar=dict(title="accidente"),
                cmin=0,
                cmax=max(float(values.quantile(0.95)), 1.0),
            ),
            customdata=df[["accidents_count", "avg_severity", "composite_score"]].values,
            hovertemplate=(
                "Accidente: %{customdata[0]}<br>"
                "Severitate medie: %{customdata[1]:.2f}<br>"
                "Scor compozit: %{customdata[2]:.2f}<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.update_layout(
        template="plotly_dark",
        mapbox_style="carto-darkmatter",
        mapbox_zoom=zoom,
        mapbox_center=_map_center(df),
        height=height,
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def map_metric_severity(grid_df: pd.DataFrame, title: str = "Hartă severitate", zoom: float = 5.2, height: int = 560) -> go.Figure:
    df = grid_df.copy()
    fig = go.Figure()
    if not df.empty:
        severity = df["avg_severity"].fillna(0).astype(float)
        sizes = _scale_sizes(df["accidents_count"], 6, 18)
        fig.add_trace(go.Scattermapbox(
            lat=df["lat_center"],
            lon=df["lon_center"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=severity,
                colorscale="Cividis",
                opacity=0.78,
                showscale=True,
                colorbar=dict(title="severitate"),
                cmin=1.0,
                cmax=3.0,
            ),
            customdata=df[["accidents_count", "avg_severity", "severe_share"]].values,
            hovertemplate=(
                "Accidente: %{customdata[0]}<br>"
                "Severitate medie: %{customdata[1]:.2f}<br>"
                "Pondere grave: %{customdata[2]:.2f}<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.update_layout(
        template="plotly_dark",
        mapbox_style="carto-darkmatter",
        mapbox_zoom=zoom,
        mapbox_center=_map_center(df),
        height=height,
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def map_metric_density(grid_df: pd.DataFrame, metric: str, title: str, zoom: float = 5.2, height: int = 560) -> go.Figure:
    if metric == "avg_severity":
        return map_metric_severity(grid_df, title=title, zoom=zoom, height=height)
    return map_metric_frequency(grid_df, title=title, zoom=zoom, height=height)


def map_hotspots(grid_df: pd.DataFrame, metric: str = "composite_score", title: str = "Hartă hotspot-uri", zoom: float = 5.2, height: int = 700) -> go.Figure:
    df = grid_df.copy()
    fig = go.Figure()
    if not df.empty:
        score = df[metric].fillna(0).astype(float)
        sizes = _scale_sizes(score, 8, 30)

        # Contur compatibil: marker negru mai mare sub markerul colorat.
        fig.add_trace(go.Scattermapbox(
            lat=df["lat_center"],
            lon=df["lon_center"],
            mode="markers",
            marker=dict(
                size=[s + 3 for s in sizes],
                color="#050608",
                opacity=0.55,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scattermapbox(
            lat=df["lat_center"],
            lon=df["lon_center"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=score,
                colorscale="Turbo",
                opacity=0.86,
                showscale=True,
                colorbar=dict(title="scor"),
                cmin=0,
                cmax=max(float(score.max()), 1.0),
            ),
            customdata=df[["accidents_count", "avg_severity", "severe_share", "composite_score"]].values,
            hovertemplate=(
                "Accidente: %{customdata[0]}<br>"
                "Severitate medie: %{customdata[1]:.2f}<br>"
                "Pondere grave: %{customdata[2]:.2f}<br>"
                "Scor compozit: %{customdata[3]:.2f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        template="plotly_dark",
        mapbox_style="carto-darkmatter",
        mapbox_zoom=zoom,
        mapbox_center=_map_center(df),
        height=height,
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def map_points(df: pd.DataFrame, title: str, sample_n: int = 6000, height: int = 560) -> go.Figure:
    sample = df.sample(min(sample_n, len(df)), random_state=42).copy()
    fig = px.scatter_mapbox(
        sample,
        lat="latitude",
        lon="longitude",
        color="severity_label",
        color_discrete_map={
            "Fatal": COLORS["red"],
            "Serious": COLORS["orange"],
            "Slight": COLORS["cyan"],
        },
        zoom=5.2,
        height=height,
        title=title,
        opacity=0.55,
    )
    fig.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _separator_trace_at_boundary(prev_coords: list[tuple[float, float]], next_coords: list[tuple[float, float]], width_px: int = 12) -> go.Scattermapbox | None:
    if len(prev_coords) < 2 or len(next_coords) < 2:
        return None
    bx, by = prev_coords[-1]
    px, py = prev_coords[-2]
    nx, ny = next_coords[1]

    vx = (bx - px) + (nx - bx)
    vy = (by - py) + (ny - by)
    norm = math.hypot(vx, vy)
    if norm == 0:
        return None
    vx /= norm
    vy /= norm
    perp_x, perp_y = -vy, vx

    half_len = 0.0028
    lon0 = bx - perp_x * half_len
    lon1 = bx + perp_x * half_len
    lat0 = by - perp_y * half_len
    lat1 = by + perp_y * half_len
    return go.Scattermapbox(
        lat=[lat0, lat1],
        lon=[lon0, lon1],
        mode="lines",
        line=dict(width=width_px, color="#020202"),
        hoverinfo="skip",
        showlegend=False,
    )


def route_map(
    start: tuple[float, float],
    end: tuple[float, float],
    route_line: LineString,
    segments: list[dict] | None = None,
    attention_points: list[dict] | None = None,
    color_mode: str = "composite_score",
    height: int = 650,
) -> go.Figure:
    fig = go.Figure()

    if segments:
        segment_coord_lists: list[list[tuple[float, float]]] = []
        for seg in segments:
            score = seg.get(color_mode, seg.get("composite_score", 0))
            if score < 30:
                color = "#00D1FF"
            elif score < 60:
                color = "#FFC857"
            else:
                color = "#FF3B3B"

            coords = list(seg["geometry"].coords)
            segment_coord_lists.append(coords)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]

            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=11, color="#000000"),
                hoverinfo="skip",
                showlegend=False,
            ))
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=7, color=color),
                name=f"Segment {seg['segment_id']}",
                hovertext=(
                    f"Segment {seg['segment_id']}<br>"
                    f"Frecvență: {seg.get('frequency_score', 0):.1f}<br>"
                    f"Severitate: {seg.get('severity_score', 0):.1f}<br>"
                    f"Scor: {seg.get('composite_score', 0):.1f}"
                ),
                hoverinfo="text",
                showlegend=False,
            ))

        for i in range(len(segment_coord_lists) - 1):
            sep = _separator_trace_at_boundary(segment_coord_lists[i], segment_coord_lists[i + 1], width_px=13)
            if sep is not None:
                fig.add_trace(sep)
    else:
        coords = list(route_line.coords)
        fig.add_trace(go.Scattermapbox(
            lat=[c[1] for c in coords],
            lon=[c[0] for c in coords],
            mode="lines",
            line=dict(width=7, color=COLORS["orange"]),
            name="Route",
            showlegend=False,
        ))

    fig.add_trace(go.Scattermapbox(
        lat=[start[0], end[0]],
        lon=[start[1], end[1]],
        mode="markers",
        marker=dict(size=14, color=[COLORS["lime"], COLORS["pink"]]),
        text=["Start", "Destinație"],
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        mapbox_style="carto-darkmatter",
        mapbox_zoom=7,
        mapbox_center={"lat": (start[0] + end[0]) / 2, "lon": (start[1] + end[1]) / 2},
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def prevention_map(
    intersections_wgs84: gpd.GeoDataFrame,
    rural_hotspots_wgs84: gpd.GeoDataFrame | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    height: int = 700,
) -> go.Figure:
    fig = go.Figure()

    if intersections_wgs84 is not None and not intersections_wgs84.empty:
        filtered_intersections = intersections_wgs84[intersections_wgs84["accidents_count"] > 0].copy()

        if not filtered_intersections.empty:
            score = filtered_intersections["composite_score"].fillna(0).astype(float)
            marker_sizes = _scale_sizes(score, 9, 26)

            fig.add_trace(go.Scattermapbox(
                lat=filtered_intersections.geometry.y,
                lon=filtered_intersections.geometry.x,
                mode="markers",
                marker=dict(size=[s + 3 for s in marker_sizes], color="#050608", opacity=0.55),
                hoverinfo="skip",
                showlegend=False,
            ))
            fig.add_trace(go.Scattermapbox(
                lat=filtered_intersections.geometry.y,
                lon=filtered_intersections.geometry.x,
                mode="markers",
                marker=dict(
                    size=marker_sizes,
                    color=score,
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(title="scor"),
                    opacity=0.86,
                    cmin=0,
                    cmax=max(float(score.max()), 1.0),
                ),
                text=[
                    (
                        f"Intersecție {row['intersection_id']}<br>"
                        f"Accidente: {row['accidents_count']}<br>"
                        f"Severitate medie: {row['avg_severity']:.2f}<br>"
                        f"Pondere grave: {row['severe_share']:.2f}%<br>"
                        f"Rază: {row['adaptive_radius_m']:.1f} m<br>"
                        f"Scor: {row['composite_score']:.2f}"
                    )
                    for _, row in filtered_intersections.iterrows()
                ],
                hoverinfo="text",
                name="Intersecții",
            ))

    if center_lat is None or center_lon is None:
        if intersections_wgs84 is not None and not intersections_wgs84.empty:
            center_lat = float(intersections_wgs84.geometry.y.mean())
            center_lon = float(intersections_wgs84.geometry.x.mean())
        else:
            center_lat, center_lon = 54.7, -1.3

    fig.update_layout(
        template="plotly_dark",
        mapbox_style="carto-darkmatter",
        mapbox_zoom=9,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )

    return fig
