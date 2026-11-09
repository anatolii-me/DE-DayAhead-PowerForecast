"""
germany_grid_map.py
===============
Генерирует словарь равноудалённых географических точек, покрывающих всю
территорию Германии.
 
Использование
-------------
    from germany_grid import generate_germany_grid
 
    points = generate_germany_grid(n_points=500)
    # {1: [longitude, latitude, "Bavaria"], 2: [...], ...}
 
Алгоритм
--------
1. Скачивается GeoJSON границ 16 федеральных земель.
2. Строится гексагональная сетка с шагом spacing_km в метрической проекции
   EPSG:3035 (ETRS89-LAEA Europe — равновеликая проекция для Европы,
   расстояния в метрах без искажений).
3. Точки внутри полигона Германии отбираются и приводятся обратно к WGS84.
4. Если точек оказалось больше/меньше n_points, шаг пересчитывается и
   итерация повторяется (бинарный поиск) — до тех пор, пока не будет
   достигнута нужная плотность.
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
 
 
# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
 
_STATES_URL = (
    "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/"
    "main/2_bundeslaender/4_niedrig.geo.json"
)
 
# Равновеликая проекция Европы (единицы — метры)
_PROJ_METRIC = pyproj.CRS("EPSG:3035")
_PROJ_WGS84  = pyproj.CRS("EPSG:4326")
 
_transformer_to_metric  = pyproj.Transformer.from_crs(_PROJ_WGS84, _PROJ_METRIC, always_xy=True)
_transformer_to_wgs84   = pyproj.Transformer.from_crs(_PROJ_METRIC, _PROJ_WGS84, always_xy=True)
 
 
# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------
 
def _load_states_geojson(url: str = _STATES_URL) -> dict:
    """Скачивает (или берёт из кеша в памяти) GeoJSON федеральных земель."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())
 
 
def _build_state_index(geojson: dict) -> tuple[BaseGeometry, list[tuple[str, BaseGeometry]]]:
    """
    Возвращает:
      - germany_poly  : объединённый полигон Германии (метрическая проекция)
      - state_polys   : список (name, polygon_in_metric)
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
# Генерация сетки
# ---------------------------------------------------------------------------
 
def _hex_grid_points(germany_poly, spacing_m: float) -> list[tuple[float, float]]:
    """
    Строит гексагональную сетку с шагом spacing_m (в метрах) и возвращает
    список (x, y) точек внутри germany_poly.
    Гексагональная укладка даёт более равномерное покрытие, чем прямоугольная.
    """
    minx, miny, maxx, maxy = germany_poly.bounds
 
    row_height = spacing_m * math.sqrt(3) / 2  # вертикальный шаг гексагона
 
    points = []
    row = 0
    y = miny
    while y <= maxy + row_height:
        # чётные/нечётные ряды смещены на половину шага
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
    Бинарным поиском находит шаг сетки (в метрах), при котором количество
    точек внутри Германии близко к n_points (±tol*100 %).
    """
    # Грубая начальная оценка через площадь
    area_m2 = germany_poly.area          # ~357 000 км²
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
            lo = mid   # шаг маловат → увеличиваем нижнюю границу
        else:
            hi = mid   # шаг велик   → уменьшаем верхнюю границу
 
    return (lo + hi) / 2
 
 
# ---------------------------------------------------------------------------
# Определение принадлежности к земле
# ---------------------------------------------------------------------------
 
def _assign_states(
    metric_points: list[tuple[float, float]],
    state_polys: list[tuple[str, BaseGeometry]],
) -> list[Optional[str]]:
    """
    Для каждой точки (x, y) в метрической проекции возвращает название
    федеральной земли или None (не должно происходить после фильтрации по
    germany_poly, но на стыках границ возможно).
    """
    labels = []
    for x, y in metric_points:
        pt = Point(x, y)
        label = None
        for name, poly in state_polys:
            if poly.contains(pt) or poly.touches(pt):
                label = name
                break
        # запасной вариант: ближайший полигон
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
# Публичная функция
# ---------------------------------------------------------------------------
 
def generate_germany_grid(
    n_points: int = 200,
    tol: float = 0.03,
    states_url: str = _STATES_URL,
) -> dict[int, list]:
    """
    Генерирует словарь равноудалённых точек, покрывающих всю Германию.
 
    Параметры
    ---------
    n_points : int
        Желаемое количество точек. Итоговое число может отличаться в пределах
        ±(tol*100) % из-за дискретности сетки.
    tol : float
        Допустимое относительное отклонение от n_points (по умолчанию 3 %).
    states_url : str
        URL GeoJSON федеральных земель (можно передать локальный путь).
 
    Возвращает
    ----------
    dict
        {
            1: [longitude, latitude, "Bavaria"],
            2: [longitude, latitude, "North Rhine-Westphalia"],
            ...
        }
        Координаты в WGS84 (десятичные градусы).
    """
    if n_points < 1:
        raise ValueError("n_points должен быть >= 1")
 
    print(f"[germany_grid] Загрузка границ федеральных земель...")
    geojson = _load_states_geojson(states_url)
    germany_poly, state_polys = _build_state_index(geojson)
 
    print(f"[germany_grid] Подбор шага сетки для ~{n_points} точек...")
    spacing_m = _spacing_for_n_points(germany_poly, n_points, tol=tol)
    spacing_km = spacing_m / 1000
    print(f"[germany_grid] Шаг сетки: {spacing_km:.2f} км")
 
    metric_pts = _hex_grid_points(germany_poly, spacing_m)
    print(f"[germany_grid] Сгенерировано точек внутри Германии: {len(metric_pts)}")
 
    print(f"[germany_grid] Определение федеральных земель...")
    labels = _assign_states(metric_pts, state_polys)
 
    # Перевод обратно в WGS84
    result: dict[int, list] = {}
    for idx, ((x, y), label) in enumerate(zip(metric_pts, labels), start=1):
        lon, lat = _transformer_to_wgs84.transform(x, y)
        result[idx] = [round(lon, 6), round(lat, 6), label]
 
    print(f"[germany_grid] Готово. Итого точек: {len(result)}")
    return result
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    pts = generate_germany_grid(n_points=n)
    # Вывод первых 10 для проверки
    for k, v in list(pts.items())[:10]:
        print(f"  {k}: lon={v[0]}, lat={v[1]}, state={v[2]}")
    print(f"  ... всего {len(pts)} точек")


    



def visualize_grid(
    points: dict[int, list],
    output_path: str = "germany_grid.html",
    open_browser: bool = False,
) -> str:
    """
    Визуализирует словарь точек на интерактивной карте и сохраняет HTML.

    Параметры
    ---------
    points : dict
        Словарь вида {id: [lon, lat, state_name]} из generate_germany_grid().
    output_path : str
        Путь для сохранения HTML-файла.
    open_browser : bool
        Открыть браузер автоматически после генерации.

    Возвращает
    ----------
    str : абсолютный путь к сохранённому файлу.
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

    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6,
                   tiles="CartoDB positron", control_scale=True)

    # Слои по землям
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
                f"<b>#{key}</b><br>Земля: {state or '—'}<br>"
                f"Lon: {lon:.5f}<br>Lat: {lat:.5f}"
            ),
        ).add_to(layers.get(state, m))

    # Легенда
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
        <b>Германия · {len(points)} точек</b>
        <table style="margin-top:8px">{rows}</table>
    </div>"""
    root = m.get_root()
    assert hasattr(root, 'html'), "Folium root не имеет html"
    root.html.add_child(folium.Element(legend))  # type: ignore[union-attr]
    folium.LayerControl(collapsed=False).add_to(m)

    out = Path(output_path).resolve()
    m.save(str(out))
    print(f"[visualize_grid] Сохранено: {out}  ({len(points)} точек)")

    if open_browser:
        webbrowser.open(out.as_uri())

    return str(out)