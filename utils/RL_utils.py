from tqdm import tqdm

from config.trainig_config import SCENARIO
from utils.barrier_control import apply_barrier_pair_states
from utils.simulation_utils import *
import random
import numpy as np
import pickle
import pedpy
from shapely.geometry import box
from jupedsim.internal.notebook_utils import read_sqlite_file



class RunningNormalizer:
    def __init__(self, enabled=True, clip_value=5.0):
        self.enabled = enabled
        self.clip_value = clip_value
        self.mean = 0.0
        self.var_sum = 0.0
        self.count = 0

    def update(self, x):
        if not self.enabled:
            return float(x)

        x = float(x)
        self.count += 1

        if self.count == 1:
            self.mean = x
            self.var_sum = 0.0
            return 0.0

        delta = x - self.mean
        self.mean += delta / self.count
        self.var_sum += delta * (x - self.mean)

        std = self.std
        z = (x - self.mean) / (std + 1e-8)

        return float(np.clip(z, -self.clip_value, self.clip_value))

    @property
    def std(self):
        if self.count < 2:
            return 1.0
        return float(np.sqrt(self.var_sum / (self.count - 1)))

def get_stage_epsilon(
        episode,
        stage_start_episode,
        stage_length,
        epsilon_start=1.0,
        epsilon_end=0.05,
):
    progress = (
        episode - stage_start_episode
    ) / max(stage_length - 1, 1)

    progress = min(1.0, max(0.0, progress))

    epsilon = epsilon_start - (
        epsilon_start - epsilon_end
    ) * progress

    epsilon = min(epsilon_start, epsilon)
    epsilon = max(epsilon_end, epsilon)

    return epsilon


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

    return geometry.covers(point)


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

    return np.array([num_agents / 1000.0]
        + density_one_hot(num_agents)
        + [float(barrier_pair_states[p]) for p in pair_names],
        dtype=np.float32,
    )


def make_measurement_area_from_geometry(geometry, margin=0.0):
    minx, miny, maxx, maxy = geometry.bounds

    area_poly = box(
        minx + margin,
        miny + margin,
        maxx - margin,
        maxy - margin,
    )

    return pedpy.MeasurementArea(list(area_poly.exterior.coords))


def compute_pedpy_metrics(trajectory_file, geometry):
    measurement_area = make_measurement_area_from_geometry(geometry)

    traj_data, walkable_area = read_sqlite_file(trajectory_file)

    classic_density = pedpy.compute_classic_density(
        traj_data=traj_data,
        measurement_area=measurement_area,
    )

    individual_voronoi = pedpy.compute_individual_voronoi_polygons(
        traj_data=traj_data,
        walkable_area=walkable_area,
    )

    voronoi_density, intersecting = pedpy.compute_voronoi_density(
        individual_voronoi_data=individual_voronoi,
        measurement_area=measurement_area,
    )
    individual_speed = pedpy.compute_individual_speed(
        traj_data=traj_data,
        frame_step=5,
    )

    metrics = {
        "classic_mean_density": float(classic_density["density"].mean()),
        "classic_max_density": float(classic_density["density"].max()),

        "voronoi_mean_density": float(voronoi_density["density"].mean()),
        "voronoi_max_density": float(voronoi_density["density"].max()),
        "voronoi_95_density": float(voronoi_density["density"].quantile(0.95)),
    }

    metrics.update({
        "mean_speed": float(individual_speed["speed"].mean()),
        "min_speed": float(individual_speed["speed"].min()),
        "speed_05": float(individual_speed["speed"].quantile(0.05)),
        "speed_10": float(individual_speed["speed"].quantile(0.10)),
        "stopped_ratio": float((individual_speed["speed"] < 0.2).mean()),
    })

    return metrics

def compute_reward(
        result,
        trajectory_file=None,
        geometry=None,
        reward_normalizer=None,
        use_normalized_reward=False,
        debug=False,
):
    initial = max(result["initial_agents"], 1)
    remaining = result["remaining_agents"]
    evacuated = initial - remaining
    elapsed = max(result["elapsed_time"], 1e-6)

    evacuation_ratio = evacuated / initial
    throughput_agents_per_second = evacuated / elapsed

    classic_mean_density = 0.0
    classic_max_density = 0.0
    voronoi_mean_density = 0.0
    voronoi_max_density = 0.0
    voronoi_95_density = 0.0

    mean_speed = 0.0
    min_speed = 0.0
    speed_05 = 0.0
    speed_10 = 0.0
    stopped_ratio = 1.0

    if trajectory_file is not None and geometry is not None:
        pedpy_metrics = compute_pedpy_metrics(
            trajectory_file=trajectory_file,
            geometry=geometry,
        )

        classic_mean_density = pedpy_metrics["classic_mean_density"]
        classic_max_density = pedpy_metrics["classic_max_density"]
        voronoi_mean_density = pedpy_metrics["voronoi_mean_density"]
        voronoi_max_density = pedpy_metrics["voronoi_max_density"]
        voronoi_95_density = pedpy_metrics["voronoi_95_density"]

        mean_speed = pedpy_metrics["mean_speed"]
        min_speed = pedpy_metrics["min_speed"]
        speed_05 = pedpy_metrics["speed_05"]
        speed_10 = pedpy_metrics["speed_10"]
        stopped_ratio = pedpy_metrics["stopped_ratio"]

    speed_score = mean_speed / SCENARIO["speed_max"]
    speed_score = np.clip(speed_score, 0.0, 1.0)
    speed_loss = 1.0 - speed_score

    raw_cost = (
        0.6 * stopped_ratio
        + 0.4 * speed_loss
    )

    raw_reward = -float(np.clip(raw_cost, 0.0, 1.0))

    use_exp_reward = SCENARIO.get("use_exp_reward", False)

    if use_exp_reward:
        alpha_exp = SCENARIO.get("reward_alpha_exp", 8.0)
        reward_scale = SCENARIO.get("reward_scale", 1.0)

        reward = -reward_scale * (
                np.exp(alpha_exp * abs(raw_reward)) - 1.0
        )
    else:
        reward = raw_reward

    if use_normalized_reward and reward_normalizer is not None:
        normalized_cost = reward_normalizer.update(raw_cost)
        reward = -float(normalized_cost)
    else:
        normalized_cost = raw_cost

    if debug:
        print(
            f"Reward debug | reward={reward:.3f}, "
            f"raw_reward={raw_reward:.3f}, "
            f"raw_cost={raw_cost:.3f}, "
            f"mean_speed={mean_speed:.3f}, "
            f"stopped_ratio={stopped_ratio:.3f}, "
            f"speed_loss={speed_loss:.3f}, "
            f"evacuated_ratio={evacuation_ratio:.3f}, "
            f"throughput={throughput_agents_per_second:.3f}"
        )

    return float(reward), {
        "reward": reward,
        "raw_reward": raw_reward,
        "raw_cost": raw_cost,
        "normalized_cost": normalized_cost,

        "speed_loss": speed_loss,
        "mean_speed": mean_speed,
        "min_speed": min_speed,
        "speed_05": speed_05,
        "speed_10": speed_10,
        "stopped_ratio": stopped_ratio,

        "evacuation_ratio": evacuation_ratio,
        "throughput_agents_per_second": throughput_agents_per_second,

        "classic_mean_density": classic_mean_density,
        "classic_max_density": classic_max_density,

        "voronoi_mean_density": voronoi_mean_density,
        "voronoi_max_density": voronoi_max_density,

        "voronoi_95_density": voronoi_95_density,
    }
def get_training_stage(episode):

    # Rare early exposure to heavy crowd cases
    if episode < SCENARIO["early_heavy_until_episode"]:
        if random.random() < SCENARIO["early_heavy_probability"]:
            return {
                "stage_name": "early_heavy_probe",
                "num_agents": random.randint(500, 800),
                "randomize": True,
                "stage_start": episode,
                "stage_length": 1,
                "fixed_epsilon": 0.50,
            }

    # normal curriculum continues here
    if episode < SCENARIO["stages_test"][0][1]:
        return {"stage_name": "stage_1_fixed_small", "num_agents": 50, "randomize": False, "stage_start": SCENARIO["stages_test"][0][0], "stage_length": SCENARIO["stages_test"][0][1]}
    elif episode < SCENARIO["stages_test"][1][1]:
        return {"stage_name": "stage_2_random_small", "num_agents": random.randint(50, 200), "randomize": True, "stage_start": SCENARIO["stages_test"][1][0], "stage_length": SCENARIO["stages_test"][1][1]}
    elif episode < SCENARIO["stages_test"][2][1]:
        return {"stage_name": "stage_3_random_medium", "num_agents": random.randint(200, 500), "randomize": True, "stage_start": SCENARIO["stages_test"][2][0], "stage_length": SCENARIO["stages_test"][2][1]}
    else:
        return {"stage_name": "stage_4_heavy", "num_agents": random.randint(500, 1000), "randomize": True, "stage_start": SCENARIO["stages_test"][3][0], "stage_length": SCENARIO["stages_test"][3][1]}



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

    case = {
        "stage_name": stage["stage_name"],
        "num_agents": stage["num_agents"],
        "initial_barrier_states": initial_barrier_states,
        "state": state,
        "stage_start": stage["stage_start"],
        "stage_length": stage["stage_length"],
    }

    if "fixed_epsilon" in stage:
        case["fixed_epsilon"] = stage["fixed_epsilon"]

    return case


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

