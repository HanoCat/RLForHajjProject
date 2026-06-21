import json
import pathlib
import random
import os

import jupedsim as jps
from shapely import wkt
from shapely.geometry import Point, box, Polygon, MultiPolygon
from shapely.validation import make_valid
from jupedsim.internal.notebook_utils import animate, read_sqlite_file


ENV_JSON = "processed_environment.json"
TRAJECTORY_FILE = "whole_scene_test.sqlite"
HTML_FILE = "whole_scene_animation.html"

N_AGENTS = 5
MAX_ITERATIONS = 1000
SAFE_DISTANCE = 0.4
MIN_AGENT_DISTANCE = 0.7


def largest_polygon(geom):
    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)

    if hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        if polys:
            return max(polys, key=lambda g: g.area)

    raise RuntimeError(f"Unsupported geometry: {geom.geom_type}")


def random_points(poly, n):
    minx, miny, maxx, maxy = poly.bounds
    points = []
    attempts = 0
    max_attempts = n * 5000

    while len(points) < n and attempts < max_attempts:
        attempts += 1

        p = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy),
        )

        if not poly.contains(p):
            continue

        too_close = False
        for qx, qy in points:
            if (p.x - qx) ** 2 + (p.y - qy) ** 2 < MIN_AGENT_DISTANCE ** 2:
                too_close = True
                break

        if not too_close:
            points.append((p.x, p.y))

    if len(points) < n:
        print(f"Warning: only generated {len(points)} / {n} agents")

    return points


def make_start_zone(geometry):
    safe = geometry.buffer(-SAFE_DISTANCE)
    if safe.is_empty:
        safe = geometry

    minx, miny, maxx, maxy = safe.bounds

    start_zone = safe.intersection(
        box(
            minx,
            miny,
            minx + 0.25 * (maxx - minx),
            maxy,
        )
    )

    if start_zone.is_empty:
        return largest_polygon(safe)

    return largest_polygon(start_zone)


def make_goal(geometry):
    safe = geometry.buffer(-SAFE_DISTANCE)
    if safe.is_empty:
        safe = geometry

    minx, miny, maxx, maxy = safe.bounds

    candidate_centers = [
        Point(maxx - 1.0, (miny + maxy) / 2),
        Point(maxx - 2.0, (miny + maxy) / 2),
        Point(maxx - 3.0, (miny + maxy) / 2),
        safe.representative_point(),
    ]

    for center in candidate_centers:
        if not safe.contains(center):
            continue

        for size in [0.8, 0.5, 0.3, 0.2]:
            goal = box(
                center.x - size / 2,
                center.y - size / 2,
                center.x + size / 2,
                center.y + size / 2,
            )

            if safe.contains(goal):
                return goal

    p = safe.representative_point()
    return box(p.x - 0.1, p.y - 0.1, p.x + 0.1, p.y + 0.1)


with open(ENV_JSON, "r", encoding="utf-8") as f:
    env = json.load(f)

geometry = wkt.loads(env["geometry_wkt"]["connected_geometry"])
geometry = make_valid(geometry)

print(env["metadata"].get("connector_lines_px"))
print("Loaded key: connected_geometry")
print("Loaded geometry type:", geometry.geom_type)
print("Loaded geometry area:", round(geometry.area, 2))

if geometry.geom_type == "MultiPolygon":
    comps = sorted(list(geometry.geoms), key=lambda g: g.area, reverse=True)

    print("\nStill disconnected after loading:")
    for i, g in enumerate(comps, start=1):
        print(
            f"Component {i}: area={g.area:.2f}, "
            f"bounds={tuple(round(v, 2) for v in g.bounds)}"
        )

    raise RuntimeError(
        "connected_geometry is still MultiPolygon. The issue is in prepare_environment.py, not this test script."
    )

start_zone = make_start_zone(geometry)
goal_area = make_goal(geometry)

positions = random_points(start_zone, N_AGENTS)

print("Start zone area:", round(start_zone.area, 2))
print("Goal area:", round(goal_area.area, 2))
print("Agents:", len(positions))

if os.path.exists(TRAJECTORY_FILE):
    os.remove(TRAJECTORY_FILE)

simulation = jps.Simulation(
    model=jps.AnticipationVelocityModel(),
    geometry=geometry,
    dt=0.05,
    trajectory_writer=jps.SqliteTrajectoryWriter(
        output_file=pathlib.Path(TRAJECTORY_FILE),
        every_nth_frame=5,
    ),
)

exit_id = simulation.add_exit_stage(goal_area)
journey = jps.JourneyDescription([exit_id])
journey_id = simulation.add_journey(journey)

for pos in positions:
    try:
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
    except RuntimeError as e:
        print("Skipped one agent:", e)

while simulation.agent_count() > 0 and simulation.iteration_count() < MAX_ITERATIONS:
    simulation.iterate()

simulation._writer.close()

print("Finished.")
print("Elapsed:", round(simulation.elapsed_time(), 2))
print("Remaining:", simulation.agent_count())

trajectories, walkable_area = read_sqlite_file(TRAJECTORY_FILE)

fig = animate(
    trajectories,
    walkable_area,
    title_note="Whole scene connected test",
    every_nth_frame=5,
    width=1200,
    height=700,
)

fig.write_html(HTML_FILE)
print("Saved:", HTML_FILE)