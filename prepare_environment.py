import json
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

SCENE_JSON = "scene_coordinates_fixed_barrier23.json"
OUTPUT_JSON = "processed_environment.json"

SCALE = 0.05
BARRIER_WIDTH_M = 0.30
CONNECTOR_WIDTH_M = 0.07
AUTO_CONNECTOR_WIDTH_M = 0.07

MOVABLE_BARRIER_IDS = {
    12, 13, 14, 15, 16,
    17, 18, 19, 20,
    21, 22, 2301, 2302, 24
}

# Manual connectors in original image pixel coordinates
CONNECTOR_LINES_PX = [
    [(620, 120), (720, 260)],
    [(640, 330), (750, 470)],
    [(1840, 360), (1840, 740)],
    [(900, 250), (900, 430)],
    [(1200, 430), (1200, 650)],
]

# Exact remaining connectors in meter coordinates from diagnosis
AUTO_CONNECTORS_M = [
    [(0.50, 8.91), (0.44, 8.72)],
    [(68.35, 19.18), (68.06, 19.70)],
    [(49.04, 23.17), (49.17, 22.59)],
    [(43.00, 18.74), (42.57, 18.74)],
    [(0.48, 17.07), (0.43, 16.94)],
]


def px_to_m_point(p, image_height):
    x, y = p
    return (x * SCALE, (image_height - y) * SCALE)


def polygon_from_points(points, image_height):
    pts = [px_to_m_point(p, image_height) for p in points]
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = make_valid(poly)
    return poly


def safe_union(items):
    return unary_union(items) if items else Polygon()


def count_components(geom):
    if geom.is_empty:
        return 0
    if geom.geom_type == "Polygon":
        return 1
    if geom.geom_type == "MultiPolygon":
        return len(geom.geoms)
    if hasattr(geom, "geoms"):
        return len([g for g in geom.geoms if g.geom_type == "Polygon"])
    return 0

def remove_small_components(geom, min_area=1.0):
    if geom.geom_type == "Polygon":
        return geom

    if geom.geom_type == "MultiPolygon":
        keep = [g for g in geom.geoms if g.area >= min_area]
        return unary_union(keep)

    if hasattr(geom, "geoms"):
        keep = [g for g in geom.geoms if g.geom_type == "Polygon" and g.area >= min_area]
        return unary_union(keep)

    return geom

with open(SCENE_JSON, "r", encoding="utf-8") as f:
    scene = json.load(f)

if "image" in scene:
    image_height = scene["image"]["height"]
else:
    image_height = scene["images"][0]["height"]

walkable_polys = []
work_polys = []
obstacle_polys = []
fixed_barrier_polys = []
movable_barrier_polys = []
manual_connector_polys = []
auto_connector_polys = []

for obj in scene["objects"]:
    label = obj["label"].strip()
    obj_type = obj["type"]
    points = obj["points"]
    obj_id = obj["id"]

    if obj_type == "polygon" and label == "walkable_area":
        walkable_polys.append(polygon_from_points(points, image_height))

    elif obj_type == "polygon" and label == "work_space":
        work_polys.append(polygon_from_points(points, image_height))

    elif obj_type == "polygon" and label == "building_or_tent":
        obstacle_polys.append(polygon_from_points(points, image_height))

    elif obj_type == "polyline" and label == "barrier":
        line_pts = [px_to_m_point(p, image_height) for p in points]
        barrier_poly = LineString(line_pts).buffer(
            BARRIER_WIDTH_M,
            cap_style=2,
            join_style=2,
        )

        if obj_id in MOVABLE_BARRIER_IDS:
            movable_barrier_polys.append(barrier_poly)
        else:
            fixed_barrier_polys.append(barrier_poly)

# Manual connectors from blue-circled gaps
for line_px in CONNECTOR_LINES_PX:
    line_m = [px_to_m_point(p, image_height) for p in line_px]
    connector = LineString(line_m).buffer(
        CONNECTOR_WIDTH_M,
        cap_style=2,
        join_style=2,
    )
    manual_connector_polys.append(connector)

# Auto connectors from exact nearest-gap diagnosis
for line_m in AUTO_CONNECTORS_M:
    connector = LineString(line_m).buffer(
        AUTO_CONNECTOR_WIDTH_M,
        cap_style=2,
        join_style=2,
    )
    auto_connector_polys.append(connector)

walkable = safe_union(walkable_polys)
work_space = safe_union(work_polys)
obstacles = safe_union(obstacle_polys)
fixed_barriers = safe_union(fixed_barrier_polys)
movable_barriers = safe_union(movable_barrier_polys)
manual_connectors = safe_union(manual_connector_polys)
auto_connectors = safe_union(auto_connector_polys)

movement_area = unary_union([walkable, work_space])
all_barriers = unary_union([fixed_barriers, movable_barriers])

# True annotated environment before artificial connectors
full_geometry = movement_area.difference(obstacles).difference(all_barriers)
full_geometry = make_valid(full_geometry)

# Simulation-ready connected environment
connected_geometry = unary_union([
    full_geometry,
    manual_connectors,
    auto_connectors,
])

connected_geometry = connected_geometry.difference(obstacles)
connected_geometry = remove_small_components(connected_geometry, min_area=1.0)
connected_geometry = make_valid(connected_geometry)

output = {
    "metadata": {
        "source_json": SCENE_JSON,
        "scale_m_per_pixel": SCALE,
        "barrier_width_m": BARRIER_WIDTH_M,
        "connector_width_m": CONNECTOR_WIDTH_M,
        "auto_connector_width_m": AUTO_CONNECTOR_WIDTH_M,
        "coordinate_system": "meters, origin bottom-left after y-axis flip",
        "movable_barrier_ids": sorted(list(MOVABLE_BARRIER_IDS)),
        "connector_lines_px": CONNECTOR_LINES_PX,
        "auto_connectors_m": AUTO_CONNECTORS_M,
    },
    "areas": {
        "walkable_m2": walkable.area,
        "work_space_m2": work_space.area,
        "movement_area_m2": movement_area.area,
        "obstacles_m2": obstacles.area,
        "fixed_barriers_m2": fixed_barriers.area,
        "movable_barriers_m2": movable_barriers.area,
        "manual_connectors_m2": manual_connectors.area,
        "auto_connectors_m2": auto_connectors.area,
        "full_geometry_m2": full_geometry.area,
        "connected_geometry_m2": connected_geometry.area,
    },
    "geometry_wkt": {
        "walkable": walkable.wkt,
        "work_space": work_space.wkt,
        "movement_area": movement_area.wkt,
        "obstacles": obstacles.wkt,
        "fixed_barriers": fixed_barriers.wkt,
        "movable_barriers": movable_barriers.wkt,
        "manual_connectors": manual_connectors.wkt,
        "auto_connectors": auto_connectors.wkt,
        "full_geometry": full_geometry.wkt,
        "connected_geometry": connected_geometry.wkt,
        "simulation_geometry": connected_geometry.wkt,
    },
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("Saved:", OUTPUT_JSON)
print("Full geometry type:", full_geometry.geom_type)
print("Full components:", count_components(full_geometry))
print("Connected geometry type:", connected_geometry.geom_type)
print("Connected components:", count_components(connected_geometry))
print("Full geometry area:", round(full_geometry.area, 2))
print("Connected geometry area:", round(connected_geometry.area, 2))

if connected_geometry.geom_type == "MultiPolygon":
    comps = sorted(connected_geometry.geoms, key=lambda g: g.area, reverse=True)
    print("\nStill disconnected components:")
    for i, g in enumerate(comps, start=1):
        print(
            f"Component {i}: area={g.area:.2f}, "
            f"bounds={tuple(round(v, 2) for v in g.bounds)}"
        )