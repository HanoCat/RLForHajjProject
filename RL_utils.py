
import os
from absl import app
from tqdm import tqdm, trange
from shapely.geometry import Point

from scenario_config import SCENARIO
from barrier_control import apply_barrier_pair_states
from simulation_utils import *
import random
import numpy as np
import pickle


def save_training_plots(history, save_dir="logs/plots"):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    # Reward
    plt.figure()
    plt.plot(df["episode"], df["reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per episode")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_per_episode.png"))
    plt.close()

    # Number of agents
    plt.figure()
    plt.plot(df["episode"], df["num_agents"])
    plt.xlabel("Episode")
    plt.ylabel("Number of agents")
    plt.title("Agents per episode")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "agents_per_episode.png"))
    plt.close()

    # Remaining agents
    plt.figure()
    plt.plot(df["episode"], df["remaining_agents"])
    plt.xlabel("Episode")
    plt.ylabel("Remaining agents")
    plt.title("Remaining agents per episode")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "remaining_agents_per_episode.png"))
    plt.close()

    # Actor loss
    if "actor_loss" in df and df["actor_loss"].notna().any():
        plt.figure()
        plt.plot(df["episode"], df["actor_loss"])
        plt.xlabel("Episode")
        plt.ylabel("Actor loss")
        plt.title("Actor loss")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "actor_loss.png"))
        plt.close()

    # Critic losses
    if df["critic_1_loss"].notna().any():
        plt.figure()
        plt.plot(df["episode"], df["critic_1_loss"], label="Critic 1")
        plt.plot(df["episode"], df["critic_2_loss"], label="Critic 2")
        plt.xlabel("Episode")
        plt.ylabel("Critic loss")
        plt.title("Critic losses")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "critic_losses.png"))
        plt.close()

    print(f"Training plots saved to: {save_dir}")
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


def action_to_barrier_pair_states(action):
    pair_names = list(SCENARIO["barrier_pairs"].keys())

    return {
        pair_name: float(np.clip(action[i], 0.0, 1.0))
        for i, pair_name in enumerate(pair_names)
    }


def density_one_hot(num_agents):
    if num_agents < 100:
        return [1.0, 0.0, 0.0]   # low
    elif num_agents < 500:
        return [0.0, 1.0, 0.0]   # medium
    else:
        return [0.0, 0.0, 1.0]   # heavy


def build_state(num_agents, barrier_pair_states):
    pair_names = list(SCENARIO["barrier_pairs"].keys())

    return np.array(
        [num_agents / 1000.0]
        + density_one_hot(num_agents)
        + [float(barrier_pair_states[p]) for p in pair_names],
        dtype=np.float32,
    )


def compute_reward(result, debug=False):
    initial = max(result["initial_agents"], 1)
    remaining = result["remaining_agents"]
    evacuated = initial - remaining

    remaining_ratio = remaining / initial
    evacuated_ratio = evacuated / initial
    time_ratio = result["iterations"] / SCENARIO["max_iterations"]

    reward = evacuated_ratio - 0.1 * time_ratio

    if debug:
        print(
            f"Reward debug | initial={initial}, remaining={remaining}, "
            f"evacuated={evacuated}, evacuated_ratio={evacuated_ratio:.3f}, "
            f"time_ratio={time_ratio:.3f}, reward={reward:.3f}"
        )

    return float(reward)
def get_training_stage(episode):
    if episode < 50:
        return {
            "stage_name": "stage_1_fixed_small",
            "num_agents": 50,
            "randomize": False,
        }

    elif episode < 150:
        return {
            "stage_name": "stage_2_random_small_medium",
            "num_agents": random.randint(50, 200),
            "randomize": True,
        }

    elif episode < 300:
        return {
            "stage_name": "stage_3_random_medium",
            "num_agents": random.randint(200, 500),
            "randomize": True,
        }

    else:
        return {
            "stage_name": "stage_4_heavy",
            "num_agents": random.randint(500, 1000),
            "randomize": True,
        }

def get_training_stage_test(episode):
    if episode < 3:
        return {"stage_name": "stage_1_fixed_small", "num_agents": 30, "randomize": False}
    elif episode < 6:
        return {"stage_name": "stage_2_random_small", "num_agents": random.randint(30, 80), "randomize": True}
    elif episode < 9:
        return {"stage_name": "stage_3_random_medium", "num_agents": random.randint(80, 150), "randomize": True}
    else:
        return {"stage_name": "stage_4_heavy_test", "num_agents": random.randint(150, 250), "randomize": True}

def reset_training_case(episode):
    stage = get_training_stage(episode)

    if stage["randomize"]:
        initial_barrier_states = {
            pair_name: random.uniform(0.0, 1.0)
            for pair_name in SCENARIO["barrier_pairs"].keys()
        }
    else:
        initial_barrier_states = SCENARIO["barrier_pair_states"]

    state = build_state(
        num_agents=stage["num_agents"],
        barrier_pair_states=initial_barrier_states,
    )

    return {
        "stage_name": stage["stage_name"],
        "num_agents": stage["num_agents"],
        "initial_barrier_states": initial_barrier_states,
        "state": state,
    }


def save_policy_checkpoint(policy, episode, reward, stage_name, save_dir="logs"):
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(
        save_dir,
        f"policy_episode_{episode + 1}_{stage_name}_reward_{reward:.3f}.pickle"
    )

    print("------Now Save Models!------")
    print(f"Saving policy to: {ckpt_path}")

    with open(ckpt_path, "wb") as f:
        pickle.dump(policy, f)

    return ckpt_path

