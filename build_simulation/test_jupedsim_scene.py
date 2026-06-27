import json
import pathlib
import random
import os

import jupedsim as jps
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, box, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from jupedsim.internal.notebook_utils import animate, read_sqlite_file


SCENE_JSON = "scene_coordinates.json"
TRAJECTORY_FILE = "../draft_html/scene_test.sqlite"

SCALE = 0.05
BARRIER_WIDTH_M = 0.30
SAFE_DISTANCE = 0.40
N_AGENTS = 15
MAX_ITERATIONS = 3000

# Barriers you want to move/change later.
# For now, they are still treated as obstacles.
MOVABLE_BARRIER_IDS = {
    12, 13, 14, 15, 16,
    17, 18, 19, 20,
    21, 22, 23, 24
}


def px_to_m_point(p, image_height):
    x, y = p
    return (x * SCALE, (image_height - y) * SCALE)


def polygon_from_points(points, image_height):
    pts = [px_to_m_point(p, image_height) for p in points]
    return make_valid(Polygon(pts))


def largest_polygon(geom):
    if geom.is_empty:
        raise RuntimeError("Geometry is empty.")

    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)

    if hasattr(geom, "geoms"):
        polygons = [g for g in geom.geoms if isinstance(g, Polygon)]
        if polygons:
            return max(polygons, key=lambda g: g.area)

    raise ValueError(f"Unsupported geometry type: {geom.geom_type}")


def build_environment(scene_json):
    with open(scene_json, "r", encoding="utf-8") as f:
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

    walkable = unary_union(walkable_polys) if walkable_polys else Polygon()
    work_space = unary_union(work_polys) if work_polys else Polygon()
    obstacles = unary_union(obstacle_polys) if obstacle_polys else Polygon()
    fixed_barriers = unary_union(fixed_barrier_polys) if fixed_barrier_polys else Polygon()
    movable_barriers = unary_union(movable_barrier_polys) if movable_barrier_polys else Polygon()

    movement_area = unary_union([walkable, work_space])
    all_barriers = unary_union([fixed_barriers, movable_barriers])

    full_geometry = movement_area.difference(obstacles).difference(all_barriers)
    full_geometry = make_valid(full_geometry)

    layers = {
        "walkable": walkable,
        "work_space": work_space,
        "obstacles": obstacles,
        "fixed_barriers": fixed_barriers,
        "movable_barriers": movable_barriers,
        "movement_area": movement_area,
        "full_geometry": full_geometry,
    }

    return full_geometry, layers


def plot_geometry(ax, geom, facecolor="lightblue", edgecolor="black", alpha=0.25, label=None, linewidth=1):
    if geom.is_empty:
        return

    if geom.geom_type == "Polygon":
        polygons = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    else:
        polygons = [g for g in geom.geoms if g.geom_type == "Polygon"]

    first = True

    for poly in polygons:
        x, y = poly.exterior.xy
        ax.fill(
            x, y,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            label=label if first else None,
        )

        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, facecolor="white", edgecolor="black", alpha=1.0)

        first = False


def random_points_in_polygon(poly, n, min_distance=0.7):
    minx, miny, maxx, maxy = poly.bounds
    points = []

    attempts = 0
    max_attempts = n * 3000
    min_dist_sq = min_distance ** 2

    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))

        if not poly.contains(p):
            continue

        too_close = False
        for qx, qy in points:
            if (p.x - qx) ** 2 + (p.y - qy) ** 2 < min_dist_sq:
                too_close = True
                break

        if not too_close:
            points.append((p.x, p.y))

    if len(points) < n:
        print(f"Warning: only generated {len(points)} / {n} points.")

    return points


def make_start_zone(geometry):
    minx, miny, maxx, maxy = geometry.bounds

    safe_geometry = geometry.buffer(-SAFE_DISTANCE)

    if safe_geometry.is_empty:
        raise RuntimeError("Safe geometry is empty. Reduce SAFE_DISTANCE.")

    candidate = safe_geometry.intersection(
        box(
            minx,
            miny,
            minx + 0.30 * (maxx - minx),
            maxy,
        )
    )

    if not candidate.is_empty:
        return largest_polygon(candidate)

    return largest_polygon(safe_geometry)


def make_convex_goal_inside_geometry(geometry):
    safe_geometry = geometry.buffer(-SAFE_DISTANCE)

    if safe_geometry.is_empty:
        safe_geometry = geometry

    largest_safe = largest_polygon(safe_geometry)
    minx, miny, maxx, maxy = largest_safe.bounds

    candidate_centers = [
        Point(maxx - 1.0, (miny + maxy) / 2),
        Point(maxx - 3.0, (miny + maxy) / 2),
        Point((minx + maxx) / 2, (miny + maxy) / 2),
        largest_safe.representative_point(),
    ]

    sizes = [
        (0.8, 0.8),
        (0.5, 0.5),
        (0.3, 0.3),
    ]

    for center in candidate_centers:
        if not largest_safe.contains(center):
            continue

        cx, cy = center.x, center.y

        for gw, gh in sizes:
            goal = box(
                cx - gw / 2,
                cy - gh / 2,
                cx + gw / 2,
                cy + gh / 2,
            )

            if largest_safe.contains(goal):
                return goal

    p = largest_safe.representative_point()
    cx, cy = p.x, p.y
    return box(cx - 0.1, cy - 0.1, cx + 0.1, cy + 0.1)


if os.path.exists(TRAJECTORY_FILE):
    os.remove(TRAJECTORY_FILE)

full_geometry, layers = build_environment(SCENE_JSON)

# JuPedSim requires one connected accessible area.
geometry = largest_polygon(full_geometry)

start_zone = make_start_zone(geometry)
goal_area = make_convex_goal_inside_geometry(geometry)
positions = random_points_in_polygon(start_zone, N_AGENTS)

print("Full geometry type:", full_geometry.geom_type)
print("Simulation geometry type:", geometry.geom_type)
print("Full geometry area:", full_geometry.area)
print("Simulation geometry area:", geometry.area)
print("Start zone area:", start_zone.area)
print("Goal area:", goal_area.area)
print("Agents:", len(positions))

simulation = jps.Simulation(
    model=jps.AnticipationVelocityModel(),
    geometry=geometry,
    dt=0.01,
    trajectory_writer=jps.SqliteTrajectoryWriter(
        output_file=pathlib.Path(TRAJECTORY_FILE),
        every_nth_frame=5,
    ),
)

exit_id = simulation.add_exit_stage(goal_area)
journey = jps.JourneyDescription([exit_id])
journey_id = simulation.add_journey(journey)

for pos in positions:
    simulation.add_agent(
        jps.AnticipationVelocityModelAgentParameters(
            journey_id=journey_id,
            stage_id=exit_id,
            position=pos,
            desired_speed=random.uniform(1.0, 1.4),
            anticipation_time=1.0,
            reaction_time=0.3,
        )
    )

fig, ax = plt.subplots(figsize=(14, 7))

plot_geometry(ax, layers["walkable"], facecolor="lightblue", edgecolor="blue", alpha=0.25, label="walkable")
plot_geometry(ax, layers["work_space"], facecolor="lightcyan", edgecolor="cyan", alpha=0.45, label="work_space")
plot_geometry(ax, layers["obstacles"], facecolor="lightyellow", edgecolor="gold", alpha=0.75, label="fixed obstacles")
plot_geometry(ax, layers["fixed_barriers"], facecolor="none", edgecolor="red", alpha=1.0, linewidth=2, label="fixed barriers")
plot_geometry(ax, layers["movable_barriers"], facecolor="none", edgecolor="orange", alpha=1.0, linewidth=3, label="movable barriers")
plot_geometry(ax, geometry, facecolor="none", edgecolor="black", alpha=1.0, linewidth=2, label="simulated connected area")

sx, sy = start_zone.exterior.xy
ax.plot(sx, sy, linewidth=2, label="start zone")

gx, gy = goal_area.exterior.xy
ax.fill(gx, gy, alpha=0.7, label="goal")

ax.scatter(
    [p[0] for p in positions],
    [p[1] for p in positions],
    s=12,
    label="agents",
)

ax.set_aspect("equal")
ax.legend()
ax.set_title("JuPedSim scene test configuration")
plt.show()

while simulation.agent_count() > 0 and simulation.iteration_count() < MAX_ITERATIONS:
    simulation.iterate()

simulation._writer.close()

print("Finished.")
print("Elapsed time:", simulation.elapsed_time())
print("Remaining agents:", simulation.agent_count())

trajectories, walkable_area = read_sqlite_file(TRAJECTORY_FILE)

fig = animate(
    trajectories,
    walkable_area,
    title_note="Scene test",
    every_nth_frame=5,
    width=900,
    height=600,
)

fig.write_html("scene_test_animation.html")
print("Animation saved: scene_test_animation.html")