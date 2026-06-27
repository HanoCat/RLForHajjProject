import os
from absl import app
from tqdm import tqdm, trange
from shapely.geometry import Point

from scenario_config import SCENARIO
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
    total_agents = 0
    p2pnet_positions = []

    if "p2pnet_points_file" in SCENARIO:
        p2pnet_positions = load_p2pnet_points(
            SCENARIO["p2pnet_points_file"],
            base_geometry,
            min_score=SCENARIO.get("p2pnet_min_score", 0.0),
            max_agents=SCENARIO.get("p2pnet_max_agents"),
        )

    with tqdm(total=len(SCENARIO["agent_groups"]), desc="Loading groups") as pbar:
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

            total_agents += len(positions)

            pbar.set_postfix(
                total_agents=total_agents,
                group_agents=len(positions),
            )
            pbar.update(1)

    print(f"Total candidate agents loaded once: {total_agents}")
    return agent_groups


def is_position_valid(position, geometry):
    point = Point(position)

    if hasattr(geometry, "walkable_area"):
        return geometry.walkable_area.covers(point)

    if hasattr(geometry, "walkable_areas"):
        return any(area.covers(point) for area in geometry.walkable_areas)

    return True


def add_valid_agents_to_simulation(simulation, agent_groups, geometry):
    total_added = 0
    total_skipped = 0

    for group in agent_groups:
        valid_positions = [
            pos for pos in group["positions"]
            if is_position_valid(pos, geometry)
        ]

        skipped = len(group["positions"]) - len(valid_positions)

        if valid_positions:
            add_agents(
                simulation,
                valid_positions,
                group["goal_area"],
                speed_min=SCENARIO["speed_min"],
                speed_max=SCENARIO["speed_max"],
            )

        total_added += len(valid_positions)
        total_skipped += skipped

    return total_added, total_skipped


def train_RL():
    num_episodes = 1

    env_json = SCENARIO["env_json"]
    _, env = load_environment(env_json)

    print(f"Environment loaded: {env_json}")

    base_geometry = create_geometry(
        env,
        SCENARIO["barrier_pair_states"],
    )

    agent_groups = load_agents(base_geometry)

    for episode in trange(SCENARIO["num_episodes"], desc="Training episodes"):

        for step in range(SCENARIO["num_steps"]):

            # Later, your RL action will update this variable.
            barrier_pair_states = SCENARIO["barrier_pair_states"]

            geometry = create_geometry(
                env,
                barrier_pair_states,
            )

            root, ext = os.path.splitext(SCENARIO["trajectory_file"])
            trajectory_file = f"{root}_episode_{episode + 1}{ext}"

            simulation = create_simulation(
                geometry,
                trajectory_file,
                dt=0.05,
            )

            total_added, total_skipped = add_valid_agents_to_simulation(
                simulation,
                agent_groups,
                geometry,
            )

            #print(f"Agents added: {total_added}")
            #print(f"Agents skipped this episode: {total_skipped}")


            result = run_simulation(
                simulation,
                SCENARIO["max_iterations"],
            )

            print("Result:", result)

            if num_episodes == 1:
                save_animation(
                    SCENARIO["every_nth_frame_n"],
                    trajectory_file,
                    SCENARIO["html_file"],
                    title=SCENARIO["name"],
                )

                print("Animation saved:", SCENARIO["html_file"])


def main(argv):
    train_RL()


if __name__ == "__main__":
    app.run(main)