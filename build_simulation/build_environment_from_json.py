import json
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from shapely.validation import make_valid

SCENE_JSON = "scene_coordinates_fixed_barrier23.json"
OUTPUT_PNG = "jupedsim_environment_preview.png"

BARRIER_WIDTH = 6.0

# Put movable barrier IDs here later
MOVABLE_BARRIER_IDS = {
    12, 13, 14, 15, 16,
    17, 18, 19, 20,
    21, 22, 2301, 2302, 24
}


def polygon_from_points(points):
    poly = Polygon(points)
    if not poly.is_valid:
        poly = make_valid(poly)
    return poly


def plot_geom(ax, geom, facecolor="none", edgecolor="black", alpha=0.4, linewidth=1):
    if geom.is_empty:
        return

    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor,
                alpha=alpha, linewidth=linewidth)

        for hole in geom.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, facecolor="white", edgecolor="black",
                    alpha=1.0, linewidth=linewidth)

    elif geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            plot_geom(ax, g, facecolor, edgecolor, alpha, linewidth)


with open(SCENE_JSON, "r", encoding="utf-8") as f:
    scene = json.load(f)

walkable_polys = []
work_polys = []
obstacle_polys = []
fixed_barrier_polys = []
movable_barrier_polys = []

barrier_objects = []

for obj in scene["objects"]:
    label = obj["label"].strip()
    obj_type = obj["type"]
    points = obj["points"]
    obj_id = obj["id"]

    if obj_type == "polygon" and label == "walkable_area":
        walkable_polys.append(polygon_from_points(points))

    elif obj_type == "polygon" and label == "work_space":
        work_polys.append(polygon_from_points(points))

    elif obj_type == "polygon" and label == "building_or_tent":
        obstacle_polys.append(polygon_from_points(points))

    elif obj_type == "polyline" and label == "barrier":
        line = LineString(points)
        barrier_poly = line.buffer(
            BARRIER_WIDTH / 2.0,
            cap_style=2,
            join_style=2
        )

        barrier_objects.append((obj_id, line))

        if obj_id in MOVABLE_BARRIER_IDS:
            movable_barrier_polys.append(barrier_poly)
        else:
            fixed_barrier_polys.append(barrier_poly)

walkable = unary_union(walkable_polys)
work_space = unary_union(work_polys)
obstacles = unary_union(obstacle_polys)
fixed_barriers = unary_union(fixed_barrier_polys)
movable_barriers = unary_union(movable_barrier_polys)

movement_area = unary_union([walkable, work_space])
all_barriers = unary_union([fixed_barriers, movable_barriers])

simulation_area = movement_area.difference(obstacles).difference(all_barriers)

fig, ax = plt.subplots(figsize=(16, 8))

plot_geom(ax, walkable, facecolor="lightblue", edgecolor="blue", alpha=0.35)
plot_geom(ax, work_space, facecolor="lightcyan", edgecolor="cyan", alpha=0.65)
plot_geom(ax, obstacles, facecolor="lightyellow", edgecolor="gold", alpha=0.75)

plot_geom(ax, fixed_barriers, facecolor="none", edgecolor="red", alpha=1.0, linewidth=2)
plot_geom(ax, movable_barriers, facecolor="none", edgecolor="orange", alpha=1.0, linewidth=3)

# Draw barrier IDs
for obj_id, line in barrier_objects:
    cx, cy = line.interpolate(0.5, normalized=True).coords[0]
    ax.text(
        cx, cy, str(obj_id),
        fontsize=8,
        color="black",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_title("Environment preview: movement areas, obstacles, fixed barriers, movable barriers")
ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)
plt.show()

print("Saved preview:", OUTPUT_PNG)
print("Walkable polygons:", len(walkable_polys))
print("Work-space polygons:", len(work_polys))
print("Obstacle polygons:", len(obstacle_polys))
print("Fixed barriers:", len(fixed_barrier_polys))
print("Movable barriers:", len(movable_barrier_polys))