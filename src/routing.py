from __future__ import annotations

import requests
from geopy.geocoders import Nominatim
from shapely.geometry import LineString

OSRM_URL = (
    "https://router.project-osrm.org/route/v1/driving/"
    "{lon1},{lat1};{lon2},{lat2}"
    "?overview=full&geometries=geojson&alternatives=true"
)


def geocode_location(location_name: str) -> tuple[float, float] | None:
    geolocator = Nominatim(user_agent="roadsafe_route_app")
    result = geolocator.geocode(location_name)
    if result is None:
        return None
    return float(result.latitude), float(result.longitude)


def get_routes(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> list[dict]:
    url = OSRM_URL.format(
        lon1=start_lon,
        lat1=start_lat,
        lon2=end_lon,
        lat2=end_lat,
    )
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    data = response.json()

    routes = []
    for idx, route in enumerate(data.get("routes", []), start=1):
        coords = route["geometry"]["coordinates"]  # [lon, lat]
        routes.append({
            "route_id": idx,
            "distance_m": float(route["distance"]),
            "duration_s": float(route["duration"]),
            "coordinates": coords,
            "line": LineString(coords),
        })
    return routes