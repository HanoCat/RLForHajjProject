# runners/run_scenario.py

from pathlib import Path

from shapely.geometry import Point

from config.scenario_config import TRAINING_CONFIG
from utils.barrier_control import apply_barrier_pair_states
from utils.simulation_utils import (
    load_environment,
    make_zone_from_fraction,
    make_convex_goal_from_zone,
    random_points,
    load_p2pnet_points,
    create_simulation,
    add_agents,
    run_simulation,
    save_animation,
    plot_zones_agents,
)



def build_geometry(config):
    _, env = load_environment(config["env_json"])

    geometry = apply_barrier_pair_states(
        env,
        config["barrier_pair_states"],
        config["barrier_pairs"],
        config["barrier_pose_config"],
    )

    return geometry


def load_detection_points(config, geometry):
    if "p2pnet_points_file" not in config:
        return []

    return load_p2pnet_points(
        config["p2pnet_points_file"],
        geometry,
        min_score=config.get("p2pnet_min_score", 0.0),
        max_agents=config.get("p2pnet_max_agents"),
    )


def build_agent_groups(config, geometry, p2pnet_positions):
    agent_groups = []
    total_agents = 0

    for group in config["agent_groups"]:
        start_zone = make_zone_from_fraction(
            geometry,
            group["start_box_frac"],
            safe_distance=config["safe_distance"],
        )

        goal_zone = make_zone_from_fraction(
            geometry,
            group["goal_box_frac"],
            safe_distance=config["safe_distance"],
        )

        goal_area = make_convex_goal_from_zone(
            goal_zone,
            size=0.4,
            safe_distance=config["safe_distance"],
        )

        random_positions = random_points(
            start_zone,
            group["count"],
            min_distance=config["min_agent_distance"],
        )

        p2pnet_group_positions = [
            pos for pos in p2pnet_positions
            if start_zone.covers(Point(pos))
        ]

        positions = random_positions + p2pnet_group_positions

        agent_groups.append({
            "group_id": group["group_id"],
            "positions": positions,
            "start_zone": start_zone,
            "goal_zone": goal_zone,
            "goal_area": goal_area,
        })

        total_agents += len(positions)

    return agent_groups, total_agents


def run_scenario(log_dir):
    config = dict(TRAINING_CONFIG)
    config["trajectory_file"] = str(log_dir / "trajectory.sqlite")
    config["html_file"] = str(log_dir / "animation.html")
    config["zones_plot_file"] = str(log_dir / "zones_agents.png")


    print(f"Running scenario: {config['name']}")

    geometry = build_geometry(config)

    p2pnet_positions = load_detection_points(config, geometry)
    agent_groups, total_agents = build_agent_groups(
        config,
        geometry,
        p2pnet_positions,
    )

    print("Geometry area:", round(geometry.area, 2))
    print("Groups:", len(agent_groups))
    print("Total agents:", total_agents)

    for group in agent_groups:
        print(
            group["group_id"],
            "| agents:", len(group["positions"]),
            "| start area:", round(group["start_zone"].area, 2),
            "| goal area:", round(group["goal_area"].area, 2),
        )

    plot_zones_agents(
        geometry,
        agent_groups,
        config
    )

    simulation = create_simulation(
        geometry,
        config["trajectory_file"],
        config["simulation"]
    )

    total_added = 0

    for group in agent_groups:
        added = add_agents(
            simulation,
            group["positions"],
            group["goal_area"],
            speed_min=config["speed_min"],
            speed_max=config["speed_max"],
        )

        print(f"Added agents for {group['group_id']}:", added)
        total_added += added

    print("Total added agents:", total_added)

    result = run_simulation(
        simulation,
        config["max_iterations"],
    )

    print("Result:", result)

    save_animation(
        config["every_nth_frame_n"],
        config["trajectory_file"],
        config["html_file"],
        title=config["name"],
    )

    print("Scenario outputs saved in:", config["output_dir"])
    print("Animation saved:", config["html_file"])

    return {
        "result": result,
        "geometry": geometry,
        "agent_groups": agent_groups,
        "total_agents": total_agents,
        "output_dir": config["output_dir"],
        "trajectory_file": config["trajectory_file"],
        "html_file": config["html_file"],
        "zones_plot_file": config["zones_plot_file"],
    }