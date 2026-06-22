import json
import os
import pathlib
import random

import jupedsim as jps
from shapely import wkt
from shapely.geometry import Point, box, Polygon, MultiPolygon
from shapely.validation import make_valid
from jupedsim.internal.notebook_utils import animate, read_sqlite_file


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


def load_environment(env_json, geometry_key="connected_geometry"):
    with open(env_json, "r", encoding="utf-8") as f:
        env = json.load(f)

    geometry = wkt.loads(env["geometry_wkt"][geometry_key])
    geometry = make_valid(geometry)

    if geometry.geom_type != "Polygon":
        raise RuntimeError(
            f"{geometry_key} is {geometry.geom_type}, but JuPedSim needs one connected Polygon."
        )

    return geometry, env


def box_from_fraction(geometry, frac_box):
    minx, miny, maxx, maxy = geometry.bounds
    fx1, fy1, fx2, fy2 = frac_box

    return box(
        minx + fx1 * (maxx - minx),
        miny + fy1 * (maxy - miny),
        minx + fx2 * (maxx - minx),
        miny + fy2 * (maxy - miny),
    )


def make_zone_from_fraction(geometry, frac_box, safe_distance=0.4):
    safe = geometry.buffer(-safe_distance)

    if safe.is_empty:
        safe = geometry

    zone = safe.intersection(box_from_fraction(geometry, frac_box))

    if zone.is_empty:
        raise RuntimeError(f"Zone is empty for fraction box: {frac_box}")

    return largest_polygon(zone)


def make_convex_goal_from_zone(goal_zone, size=0.4, safe_distance=0.0):
    minx, miny, maxx, maxy = goal_zone.bounds

    # Exit strip at the LEFT boundary of the goal zone
    exit_width = 0.6

    goal = box(
        minx - exit_width,
        miny,
        minx + exit_width,
        maxy,
    )

    return goal

def random_points(poly, n, min_distance=0.7):
    minx, miny, maxx, maxy = poly.bounds
    points = []

    attempts = 0
    max_attempts = n * 5000
    min_dist_sq = min_distance ** 2

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
            if (p.x - qx) ** 2 + (p.y - qy) ** 2 < min_dist_sq:
                too_close = True
                break

        if not too_close:
            points.append((p.x, p.y))

    if len(points) < n:
        print(f"Warning: generated {len(points)} / {n} agents.")

    return points


def create_simulation(geometry, trajectory_file, dt=0.05):
    if os.path.exists(trajectory_file):
        os.remove(trajectory_file)

    return jps.Simulation(
        model=jps.CollisionFreeSpeedModelV2(),
        geometry=geometry,
        dt=dt,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(trajectory_file),
            every_nth_frame=5,
        ),
    )


def add_agents(simulation, positions, goal_area, speed_min=1.0, speed_max=1.4):
    exit_id = simulation.add_exit_stage(goal_area)
    journey = jps.JourneyDescription([exit_id])
    journey_id = simulation.add_journey(journey)

    added = 0

    for agent_id, pos in enumerate(positions, start=1):
        try:
            simulation.add_agent(
                jps.CollisionFreeSpeedModelV2AgentParameters(
                    journey_id=journey_id,
                    stage_id=exit_id,
                    position=pos,
                    desired_speed=random.uniform(speed_min, speed_max),
                    radius=0.12,
                    #time_gap = 0.8,
                    range_neighbor_repulsion = 0.5,
                    #strength_neighbor_repulsion = 1.0.imag,
                )

            )
            added += 1

        except RuntimeError as e:
            print(f"Skipped agent {agent_id} at {pos}: {e}")

    return added

def add_agents_with_mid_goal(
    simulation,
    positions,
    goal_area,
    mid_points=None,
    mid_distance=1.0,
    speed_min=1.0,
    speed_max=1.4,
):
    exit_id = simulation.add_exit_stage(goal_area)
    added = 0

    for agent_id, pos in enumerate(positions, start=1):
        try:
            if mid_points is not None:
                mid_point = mid_points[agent_id - 1]
                waypoint_id = simulation.add_waypoint_stage(mid_point, mid_distance)

                journey = jps.JourneyDescription([waypoint_id, exit_id])
                journey.set_transition_for_stage(
                    waypoint_id,
                    jps.Transition.create_fixed_transition(exit_id),
                )

                stage_id = waypoint_id

            else:
                journey = jps.JourneyDescription([exit_id])
                stage_id = exit_id

            journey_id = simulation.add_journey(journey)

            simulation.add_agent(
                jps.CollisionFreeSpeedModelV2AgentParameters(
                    journey_id=journey_id,
                    stage_id=stage_id,
                    position=pos,
                    desired_speed=random.uniform(speed_min, speed_max),
                    radius=0.12,
                )
            )
            added += 1

        except RuntimeError as e:
            print(f"Skipped agent {agent_id} at {pos}: {e}")

    return added
def run_simulation(simulation, max_iterations):
    while simulation.agent_count() > 0 and simulation.iteration_count() < max_iterations:
        simulation.iterate()

    simulation._writer.close()

    return {
        "elapsed_time": simulation.elapsed_time(),
        "remaining_agents": simulation.agent_count(),
        "iterations": simulation.iteration_count(),
    }


def save_animation(every_nth_frame_n, trajectory_file, html_file, title="Simulation"):
    trajectories, walkable_area = read_sqlite_file(trajectory_file)

    fig = animate(
        trajectories,
        walkable_area,
        title_note=title,
        every_nth_frame=every_nth_frame_n,
        width=1200,
        height=700,
    )

    fig.write_html(html_file)