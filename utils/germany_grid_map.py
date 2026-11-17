"""
germany_grid_map.py
===============
Generate a dictionary of evenly spaced points across Germany.
 
Usage
-----
    from utils.germany_grid_map import generate_germany_grid, visualize_grid
 
    points = generate_germany_grid(n_points=500)
    # {1: [longitude, latitude, "Bayern"], 2: [...], ...}
    visualize_grid(points, output_path="germany_grid.html")

How it works
------------
1. Download GeoJSON boundaries for Germany's 16 federal states.
2. Build a hexagonal grid in the EPSG:3035 metric projection.
3. Keep points inside Germany and convert them to WGS84 coordinates.
4. Adjust the grid spacing with a binary search until the number of points
   is close to n_points.
"""
 
from __future__ import annotations
 
import math
import json
import urllib.request
from typing import Optional
 
from shapely.geometry import shape, Point, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, transform as shapely_transform
import pyproj

import folium
from branca.element import Figure
 
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
_STATES_URL = (
    "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/"
    "main/2_bundeslaender/4_niedrig.geo.json"
)
 
# Equal-area projection for Europe (coordinates in meters).
_PROJ_METRIC = pyproj.CRS("EPSG:3035")
_PROJ_WGS84  = pyproj.CRS("EPSG:4326")
 
_transformer_to_metric  = pyproj.Transformer.from_crs(_PROJ_WGS84, _PROJ_METRIC, always_xy=True)
_transformer_to_wgs84   = pyproj.Transformer.from_crs(_PROJ_METRIC, _PROJ_WGS84, always_xy=True)
 
 
# ---------------------------------------------------------------------------
# Load boundary data
# ---------------------------------------------------------------------------
 
def _load_states_geojson(url: str = _STATES_URL) -> dict:
    """Download the federal state boundaries as GeoJSON."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())
 
 
def _build_state_index(geojson: dict) -> tuple[BaseGeometry, list[tuple[str, BaseGeometry]]]:
    """
    Return the combined Germany polygon and a list of state polygons.

    All polygons use the metric projection.
    """
    state_polys: list[tuple[str, BaseGeometry]] = []
 
    def to_metric(geom):
        return shapely_transform(
            lambda x, y, z=None: _transformer_to_metric.transform(x, y),
            geom,
        )
 
    for feature in geojson["features"]:
        name = feature["properties"]["name"]
        geom = shape(feature["geometry"])
        geom_m = to_metric(geom)
        state_polys.append((name, geom_m))
 
    germany_poly = unary_union([g for _, g in state_polys])
    return germany_poly, state_polys
 
 
# ---------------------------------------------------------------------------
# Build the grid
# ---------------------------------------------------------------------------
 
def _hex_grid_points(germany_poly, spacing_m: float) -> list[tuple[float, float]]:
    """
    Return hexagonal grid points inside germany_poly.

    spacing_m is the horizontal distance between points, in meters.
    Staggered rows provide more even coverage than a square grid.
    """
    minx, miny, maxx, maxy = germany_poly.bounds
 
    row_height = spacing_m * math.sqrt(3) / 2  # Vertical distance between rows.
 
    points = []
    row = 0
    y = miny
    while y <= maxy + row_height:
        # Shift every second row by half the horizontal spacing.
        x_offset = 0.5 * spacing_m if (row % 2 == 1) else 0.0
        x = minx + x_offset
        while x <= maxx + spacing_m:
            pt = Point(x, y)
            if germany_poly.contains(pt):
                points.append((x, y))
            x += spacing_m
        y += row_height
        row += 1
 
    return points
 
 
def _spacing_for_n_points(germany_poly, n_points: int,
                           tol: float = 0.02,
                           max_iter: int = 40) -> float:
    """
    Find grid spacing, in meters, that gives about n_points points.

    The target range is within tol * 100 percent of n_points.
    """
    # Estimate spacing from Germany's area.
    area_m2 = germany_poly.area          # About 357,000 km².
    spacing_est = math.sqrt(area_m2 / n_points) * 1.05
 
    lo, hi = spacing_est * 0.2, spacing_est * 5.0
 
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        pts = _hex_grid_points(germany_poly, mid)
        got = len(pts)
 
        if got == 0:
            hi = mid
            continue
 
        if abs(got - n_points) / n_points <= tol:
            return mid
 
        if got > n_points:
            lo = mid   # Too many points: increase the spacing.
        else:
            hi = mid   # Too few points: decrease the spacing.
 
    return (lo + hi) / 2
 
 
# ---------------------------------------------------------------------------
# Assign each point to a state
# ---------------------------------------------------------------------------
 
def _assign_states(
    metric_points: list[tuple[float, float]],
    state_polys: list[tuple[str, BaseGeometry]],
) -> list[Optional[str]]:
    """
    Return the federal state name for each point in metric coordinates.

    A point on a shared border can be assigned to either neighboring state.
    """
    labels = []
    for x, y in metric_points:
        pt = Point(x, y)
        label = None
        for name, poly in state_polys:
            if poly.contains(pt) or poly.touches(pt):
                label = name
                break
        # Use the nearest state if no polygon contains or touches the point.
        if label is None:
            min_dist = float("inf")
            for name, poly in state_polys:
                d = poly.distance(pt)
                if d < min_dist:
                    min_dist = d
                    label = name
        labels.append(label)
    return labels
 
 
# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
 
def generate_germany_grid(
    n_points: int = 200,
    tol: float = 0.03,
    states_url: str = _STATES_URL,
) -> dict[int, list]:
    """
    Generate a dictionary of evenly spaced points across Germany.
 
    Parameters
    ----------
    n_points : int
        Target number of points. The result may differ because grid sizes
        change in discrete steps.
    tol : float
        Allowed relative difference from n_points (default: 3%).
    states_url : str
        URL of the federal state boundaries in GeoJSON format.
 
    Returns
    -------
    dict
        {
            1: [longitude, latitude, "Bayern"],
            2: [longitude, latitude, "Nordrhein-Westfalen"],
            ...
        }
        Coordinates use WGS84 decimal degrees. State names are in German.
    """
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
 
    print("[germany_grid] Loading federal state boundaries...")
    geojson = _load_states_geojson(states_url)
    germany_poly, state_polys = _build_state_index(geojson)
 
    print(f"[germany_grid] Finding grid spacing for about {n_points} points...")
    spacing_m = _spacing_for_n_points(germany_poly, n_points, tol=tol)
    spacing_km = spacing_m / 1000
    print(f"[germany_grid] Grid spacing: {spacing_km:.2f} km")
 
    metric_pts = _hex_grid_points(germany_poly, spacing_m)
    print(f"[germany_grid] Points inside Germany: {len(metric_pts)}")
 
    print("[germany_grid] Assigning points to federal states...")
    labels = _assign_states(metric_pts, state_polys)
 
    # Convert coordinates back to WGS84.
    result: dict[int, list] = {}
    for idx, ((x, y), label) in enumerate(zip(metric_pts, labels), start=1):
        lon, lat = _transformer_to_wgs84.transform(x, y)
        result[idx] = [round(lon, 6), round(lat, 6), label]
 
    print(f"[germany_grid] Done. Total points: {len(result)}")
    return result
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    pts = generate_germany_grid(n_points=n)
    # Show the first 10 points for a quick check.
    for k, v in list(pts.items())[:10]:
        print(f"  {k}: lon={v[0]}, lat={v[1]}, state={v[2]}")
    print(f"  ... {len(pts)} points in total")


    



def visualize_grid(
    points: dict[int, list],
    output_path: str = "germany_grid.html",
    open_browser: bool = False,
) -> str:
    """
    Save an interactive HTML map of the grid points and Europe.

    Parameters
    ----------
    points : dict
        Dictionary of {id: [lon, lat, state_name]} from generate_germany_grid().
    output_path : str
        Path for the output HTML file.
    open_browser : bool
        Open the saved file in a browser after creating it.

    Returns
    -------
    str
        Absolute path to the saved HTML file.
    """
    import colorsys, hashlib, webbrowser
    from pathlib import Path

    STATE_COLORS: dict[str, str] = {
        "Baden-Württemberg":      "#E63946",
        "Bayern":                 "#F4A261",
        "Berlin":                 "#2A9D8F",
        "Brandenburg":            "#E9C46A",
        "Bremen":                 "#264653",
        "Hamburg":                "#6A0572",
        "Hessen":                 "#457B9D",
        "Mecklenburg-Vorpommern": "#1D3557",
        "Niedersachsen":          "#A8DADC",
        "Nordrhein-Westfalen":    "#E76F51",
        "Rheinland-Pfalz":        "#52B788",
        "Saarland":               "#B5838D",
        "Sachsen":                "#6D6875",
        "Sachsen-Anhalt":         "#C77DFF",
        "Schleswig-Holstein":     "#023E8A",
        "Thüringen":              "#D62828",
    }

    def _color(state: str | None) -> str:
        if state and state in STATE_COLORS:
            return STATE_COLORS[state]
        if state:
            h = int(hashlib.md5(state.encode()).hexdigest()[:6], 16)
            r, g, b = colorsys.hsv_to_rgb((h % 360) / 360, 0.7, 0.85)
            return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        return "#888888"

    # Embed state boundaries in the HTML to avoid external tile servers.
    # Source: _STATES_URL (Unlicense).
    states_path = Path(__file__).resolve().parent / "germany_states_low.geojson"
    with states_path.open(encoding="utf-8") as f:
        states_geojson = json.load(f)
    # Natural Earth 1:50m country boundaries for Europe and nearby countries.
    # Source: https://github.com/nvkelso/natural-earth-vector (public domain).
    countries_path = Path(__file__).resolve().parent / "europe_countries.geojson"
    with countries_path.open(encoding="utf-8") as f:
        countries_geojson = json.load(f)

    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6,
                   tiles=None, control_scale=True)
    root = m.get_root()
    if not isinstance(root, Figure):
        raise TypeError("Expected a Folium Figure as the map root")
    root.header.add_child(folium.Element(
        "<style>.leaflet-container { background: #edf3f8; }"
        "path.leaflet-interactive:focus { outline: none !important; }"
        ".country-label { color: #52616b; font: 600 13px Arial, sans-serif; "
        "text-align: center; white-space: nowrap; text-shadow: 0 1px white, "
        "1px 0 white, 0 -1px white, -1px 0 white; }</style>"
    ))
    folium.GeoJson(
        countries_geojson,
        name="European countries",
        style_function=lambda _: {
            "fillColor": "#e4e9dd", "fillOpacity": 1,
            "color": "#aab6aa", "weight": 1.5,
        },
        tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Country:"]),
    ).add_to(m)
    folium.GeoJson(
        states_geojson,
        name="State boundaries",
        style_function=lambda _: {
            "fillColor": "#fffdf8", "fillOpacity": 1,
            "color": "#8c9aa5", "weight": 1,
        },
        tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["State:"]),
    ).add_to(m)
    country_labels = {
        "Ireland": (53.2, -8.0),
        "United Kingdom": (54.0, -3.5),
        "Portugal": (39.5, -8.0),
        "Spain": (40.2, -4.0),
        "France": (46.8, 2.3),
        "Netherlands": (52.2, 5.3),
        "Belgium": (50.5, 4.7),
        "Denmark": (56.2, 9.8),
        "Norway": (62.5, 10.0),
        "Sweden": (62.0, 16.0),
        "Finland": (63.5, 26.0),
        "Poland": (52.2, 19.0),
        "Czechia": (49.8, 15.4),
        "Austria": (47.7, 14.2),
        "Switzerland": (46.8, 8.3),
        "Italy": (42.8, 12.5),
        "Romania": (45.9, 25.0),
        "Greece": (39.5, 22.0),
        "Turkey": (39.5, 33.0),
    }
    label_layer = folium.FeatureGroup(name="Country names", show=True)
    for name, location in country_labels.items():
        folium.Marker(
            location=location,
            icon=folium.DivIcon(
                icon_size=(100, 20), icon_anchor=(50, 10),
                html=f'<div class="country-label">{name}</div>',
            ),
        ).add_to(label_layer)
    label_layer.add_to(m)
    m.fit_bounds([[33.5, -12.0], [72.0, 45.0]], padding=(20, 20))

    # Create a separate layer for each state.
    state_names = sorted({v[2] for v in points.values() if v[2]})
    layers: dict[str, folium.FeatureGroup] = {}
    for state in state_names:
        fg = folium.FeatureGroup(name=state, show=True)
        m.add_child(fg)
        layers[state] = fg

    for key, (lon, lat, state) in points.items():
        color = _color(state)
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1,
            tooltip=folium.Tooltip(
                f"<b>#{key}</b><br>State: {state or '—'}<br>"
                f"Lon: {lon:.5f}<br>Lat: {lat:.5f}"
            ),
        ).add_to(layers.get(state, m))

    # Build the legend.
    state_counts = {}
    for v in points.values():
        s = v[2] or "—"
        state_counts[s] = state_counts.get(s, 0) + 1

    rows = "".join(
        f'<tr><td><span style="display:inline-block;width:11px;height:11px;'
        f'border-radius:50%;background:{_color(s)};margin-right:5px"></span>{s}</td>'
        f'<td style="padding-left:8px;text-align:right">{c}</td></tr>'
        for s, c in sorted(state_counts.items(), key=lambda x: -x[1])
    )
    legend = f"""
    <div style="position:fixed;bottom:30px;right:10px;z-index:9999;background:white;
                padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.25);
                font-family:Arial,sans-serif;font-size:12px;max-height:400px;overflow-y:auto">
        <b>Germany · {len(points)} points</b>
        <table style="margin-top:8px">{rows}</table>
    </div>"""
    root.html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)

    out = Path(output_path).resolve()
    m.save(str(out))
    print(f"[visualize_grid] Saved: {out}  ({len(points)} points)")

    if open_browser:
        webbrowser.open(out.as_uri())

    return str(out)
