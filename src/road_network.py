from __future__ import annotations

import osmnx as ox
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree


def download_drive_network(place_name: str):
    """
    Descarcă rețeaua rutieră de tip drive din OpenStreetMap.
    """
    G = ox.graph_from_place(place_name, network_type="drive")
    return G


def extract_intersections_from_graph(G) -> gpd.GeoDataFrame:
    """
    Extrage nodurile care pot fi considerate intersecții reale.
    street_count >= 3 este un prag bun de început.
    """
    nodes, _ = ox.graph_to_gdfs(G)

    if "street_count" not in nodes.columns:
        # fallback defensiv
        intersections = nodes.copy()
    else:
        intersections = nodes[nodes["street_count"] >= 3].copy()

    intersections = intersections.reset_index().rename(columns={"osmid": "intersection_id"})
    if "intersection_id" not in intersections.columns:
        intersections["intersection_id"] = intersections.index.astype(str)

    return intersections


def intersections_to_metric(intersections_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Transformă intersecțiile în CRS metric pentru calcule în metri.
    """
    if intersections_gdf.crs is None:
        intersections_gdf = intersections_gdf.set_crs(epsg=4326)
    return intersections_gdf.to_crs(epsg=3857)


def build_intersection_kdtree(intersections_metric: gpd.GeoDataFrame):
    """
    Construiește KDTree pentru intersecții.
    """
    coords = list(zip(intersections_metric.geometry.x, intersections_metric.geometry.y))
    tree = cKDTree(coords)
    return tree, coords


def compute_adaptive_radii(intersections_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculează raza de asociere pentru fiecare intersecție.

    Regula folosită:
    - raza standard este 50 m;
    - dacă cea mai apropiată intersecție este la mai puțin de 100 m,
      raza devine jumătate din distanța până la acea intersecție.

    Echivalent matematic: r = min(50, distanța_către_cea_mai_apropiată_intersecție / 2).
    Astfel, zonele de asociere ale două intersecții foarte apropiate nu se suprapun excesiv.
    """
    intersections_metric = intersections_metric.copy()
    tree, coords = build_intersection_kdtree(intersections_metric)

    # query cu k=2: primul vecin este el însuși, al doilea este cel mai apropiat vecin real
    distances, indices = tree.query(coords, k=2)

    intersections_metric["nearest_intersection_distance_m"] = distances[:, 1]
    intersections_metric["adaptive_radius_m"] = intersections_metric[
        "nearest_intersection_distance_m"
    ].apply(lambda d: min(50.0, float(d) / 2.0))

    return intersections_metric


def intersections_back_to_wgs84(intersections_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Reproiectează înapoi pentru afișare pe hartă.
    """
    return intersections_metric.to_crs(epsg=4326)