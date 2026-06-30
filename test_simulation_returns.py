import os
import time
import random
import pathlib
import pandas as pd

from shapely.geometry import Point, box

import pedpy
from jupedsim.internal.notebook_utils import read_sqlite_file

from scenario_config_orginal import SCENARIO
from barrier_control import apply_barrier_pair_states
from simulation_utils import *


def create_geometry(env, barrier_pair_states):
    return apply_barrier_pair_states(
        env,
        barrier_pair_states,
        SCENARIO["barrier_pairs"],
        SCENARIO["barrier_pose_config"],
    )


def load_agents(base_geometry):
    agent_groups = []
    p2pnet_positions = []

    if "p2pnet_points_file" in SCENARIO:
        p2pnet_positions = load_p2pnet_points(
            SCENARIO["p2pnet_points_file"],
            base_geometry,
            min_score=SCENARIO.get("p2pnet_min_score", 0.0),
            max_agents=SCENARIO.get("p2pnet_max_agents"),
        )

    for group in SCENARIO["agent_groups"]:
        start_zone = make_zone_from_fraction(
            base_geometry,
            group["start_box_frac"],
            safe_distance=SCENARIO["safe_distance"],
        )

        goal_zone = make_zone_from_fraction(
            base_geometry,
            group["goal_box_frac"],
            safe_distance=SCENARIO["safe_distance"],
        )

        goal_area = make_convex_goal_from_zone(
            goal_zone,
            size=0.4,
            safe_distance=SCENARIO["safe_distance"],
        )

        random_positions = random_points(
            start_zone,
            group["count"],
            min_distance=SCENARIO["min_agent_distance"],
        )

        p2pnet_group_positions = [
            pos for pos in p2pnet_positions
            if start_zone.covers(Point(pos))
        ]

        positions = random_positions + p2pnet_group_positions

        agent_groups.append({
            "group_id": group["group_id"],
            "positions": positions,
            "goal_area": goal_area,
        })

    return agent_groups


def select_agent_subset(agent_groups, max_agents=None, shuffle=True):
    if max_agents is None:
        return agent_groups

    all_agents = []

    for group in agent_groups:
        for pos in group["positions"]:
            all_agents.append({
                "group_id": group["group_id"],
                "position": pos,
                "goal_area": group["goal_area"],
            })

    if shuffle:
        random.shuffle(all_agents)

    selected_agents = all_agents[:max_agents]

    grouped = {}

    for agent in selected_agents:
        group_id = agent["group_id"]

        if group_id not in grouped:
            grouped[group_id] = {
                "group_id": group_id,
                "positions": [],
                "goal_area": agent["goal_area"],
            }

        grouped[group_id]["positions"].append(agent["position"])

    return list(grouped.values())


def add_valid_agents_to_simulation(simulation, agent_groups, geometry):
    total_added = 0
    total_skipped = 0

    for group in agent_groups:
        valid_positions = [
            pos for pos in group["positions"]
            if geometry.covers(Point(pos))
        ]

        skipped = len(group["positions"]) - len(valid_positions)

        if valid_positions:
            added = add_agents(
                simulation,
                valid_positions,
                group["goal_area"],
                speed_min=SCENARIO["speed_min"],
                speed_max=SCENARIO["speed_max"],
            )
            total_added += added

        total_skipped += skipped

    return total_added, total_skipped


def make_measurement_area_from_geometry(geometry, margin=0.0):
    """
    For first diagnostic test, use the full environment bounds as measurement area.
    Later we can define bottleneck-specific areas.
    """
    minx, miny, maxx, maxy = geometry.bounds

    area_poly = box(
        minx + margin,
        miny + margin,
        maxx - margin,
        maxy - margin,
    )

    return pedpy.MeasurementArea(list(area_poly.exterior.coords))


def analyze_with_pedpy(trajectory_file, measurement_area, speed_frame_step=5):
    """
    Reads JuPedSim SQLite trajectory and computes:
    - classic density over time
    - individual speed
    - mean speed per frame in measurement area

    PedPy classic density returns frame + density in 1/m².
    PedPy individual speed returns id/frame/speed, and mean speed gives frame + speed.
    """
    metrics = {}

    t0 = time.time()
    traj_data, walkable_area = read_sqlite_file(trajectory_file)
    metrics["read_sqlite_seconds"] = time.time() - t0

    metrics["trajectory_rows"] = len(traj_data.data)
    metrics["trajectory_frames"] = traj_data.data["frame"].nunique()
    metrics["trajectory_agents"] = traj_data.data["id"].nunique()

    t0 = time.time()
    density = pedpy.compute_classic_density(
        traj_data=traj_data,
        measurement_area=measurement_area,
    )
    metrics["density_compute_seconds"] = time.time() - t0

    metrics["mean_density"] = float(density["density"].mean())
    metrics["max_density"] = float(density["density"].max())
    metrics["last_density"] = float(density["density"].iloc[-1])

    t0 = time.time()
    individual_speed = pedpy.compute_individual_speed(
        traj_data=traj_data,
        frame_step=speed_frame_step,
    )
    metrics["individual_speed_compute_seconds"] = time.time() - t0

    if len(individual_speed) > 0:
        metrics["mean_individual_speed"] = float(individual_speed["speed"].mean())
        metrics["max_individual_speed"] = float(individual_speed["speed"].max())
        metrics["min_individual_speed"] = float(individual_speed["speed"].min())
    else:
        metrics["mean_individual_speed"] = None
        metrics["max_individual_speed"] = None
        metrics["min_individual_speed"] = None

    t0 = time.time()
    individual_speed = pedpy.compute_individual_speed(
        traj_data=traj_data,
        frame_step=speed_frame_step,
    )

    metrics["individual_speed_compute_seconds"] = time.time() - t0

    if len(individual_speed) > 0:
        metrics["mean_individual_speed"] = float(individual_speed["speed"].mean())
        metrics["max_individual_speed"] = float(individual_speed["speed"].max())
        metrics["min_individual_speed"] = float(individual_speed["speed"].min())
    else:
        metrics["mean_individual_speed"] = None
        metrics["max_individual_speed"] = None
        metrics["min_individual_speed"] = None

    return metrics, density, individual_speed, None


def add_derived_simulation_metrics(result):
    initial = max(result["initial_agents"], 1)
    remaining = result["remaining_agents"]
    evacuated = initial - remaining
    elapsed = max(result["elapsed_time"], 1e-6)

    result["evacuated_agents"] = evacuated
    result["evacuation_ratio"] = evacuated / initial
    result["remaining_ratio"] = remaining / initial
    result["throughput_agents_per_second"] = evacuated / elapsed

    return result


def run_one_test(
    env,
    agent_groups,
    num_agents,
    max_iterations,
    dt,
    every_nth_frame,
    barrier_states,
    test_id,
):
    geometry = create_geometry(env, barrier_states)

    trajectory_file = f"diagnostic_test_{test_id}.sqlite"

    simulation_pram = {
        "dt": dt,
        "write_trajectory": True,
        "every_nth_frame": every_nth_frame,
    }

    simulation = create_simulation(
        geometry,
        trajectory_file=trajectory_file,
        simulation_pram=simulation_pram,
    )

    episode_agent_groups = select_agent_subset(
        agent_groups,
        max_agents=num_agents,
        shuffle=True,
    )

    total_added, total_skipped = add_valid_agents_to_simulation(
        simulation,
        episode_agent_groups,
        geometry,
    )

    t0 = time.time()
    result = run_simulation(
        simulation,
        max_iterations,
    )
    simulation_seconds = time.time() - t0

    result = add_derived_simulation_metrics(result)

    measurement_area = make_measurement_area_from_geometry(geometry)

    try:
        pedpy_metrics, density, individual_speed, mean_speed = analyze_with_pedpy(
            trajectory_file=trajectory_file,
            measurement_area=measurement_area,
            speed_frame_step=5,
        )
    except Exception as e:
        pedpy_metrics = {"pedpy_error": str(e)}
        density = pd.DataFrame()
        individual_speed = pd.DataFrame()
        mean_speed = pd.DataFrame()

    row = {
        "test_id": test_id,
        "requested_agents": num_agents,
        "added_agents": total_added,
        "skipped_agents": total_skipped,
        "max_iterations": max_iterations,
        "dt": dt,
        "every_nth_frame": every_nth_frame,
        "simulation_seconds": simulation_seconds,
        **result,
        **pedpy_metrics,
    }

    print("\n" + "=" * 70)
    print(f"TEST {test_id}")
    print(f"requested_agents: {num_agents}")
    print(f"added_agents: {total_added}")
    print(f"skipped_agents: {total_skipped}")
    print(f"max_iterations: {max_iterations}")
    print(f"dt: {dt}")
    print(f"every_nth_frame: {every_nth_frame}")
    print(f"simulation_seconds: {simulation_seconds:.3f}")
    print("simulation result:", result)
    print("pedpy metrics:", pedpy_metrics)

    return row


def main():
    os.makedirs("diagnostic_results", exist_ok=True)

    _, env = load_environment(SCENARIO["env_json"])

    base_geometry = create_geometry(
        env,
        SCENARIO["barrier_pair_states"],
    )

    agent_groups = load_agents(base_geometry)

    tests = [
        {"num_agents": 50, "max_iterations": 500, "dt": 0.05, "every_nth_frame": 10},
        {"num_agents": 50, "max_iterations": 1000, "dt": 0.05, "every_nth_frame": 10},
        {"num_agents": 100, "max_iterations": 500, "dt": 0.05, "every_nth_frame": 10},
        {"num_agents": 200, "max_iterations": 500, "dt": 0.05, "every_nth_frame": 10},

        # test cost of denser writing
        {"num_agents": 100, "max_iterations": 500, "dt": 0.05, "every_nth_frame": 1},
    ]

    rows = []

    for i, test in enumerate(tests, start=1):
        row = run_one_test(
            env=env,
            agent_groups=agent_groups,
            num_agents=test["num_agents"],
            max_iterations=test["max_iterations"],
            dt=test["dt"],
            every_nth_frame=test["every_nth_frame"],
            barrier_states=SCENARIO["barrier_pair_states"],
            test_id=i,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv = "diagnostic_results/simulation_pedpy_metrics.csv"
    df.to_csv(output_csv, index=False)

    print("\nSUMMARY")
    print(df)
    print(f"\nSaved diagnostic summary to: {output_csv}")


if __name__ == "__main__":
    main()