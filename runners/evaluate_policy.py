import pandas as pd
import torch
from utils.RL_utils import *

import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
        return super().find_class(module, name)


POLICY_PATH = "../models/RL_model.pickle"
OUTPUT_CSV = "outputs/evaluation_results.csv"

SEEDS = [101, 202, 303, 404, 505]
NUM_AGENTS_LIST = [300, 500, 700, 900]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_one_case(policy, env, agent_groups, method, seed, num_agents):
    set_seed(seed)

    pair_names = list(SCENARIO["barrier_pairs"].keys())

    if method == "all_open":
        barrier_pair_states = {p: 1.0 for p in pair_names}

    elif method == "all_closed":
        barrier_pair_states = {p: 0.0 for p in pair_names}

    elif method == "random":
        action = np.random.uniform(0.0, 1.0, size=7).astype(np.float32)
        barrier_pair_states = action_to_barrier_pair_states(action)

    elif method == "rl_policy":
        initial_states = {p: 1.0 for p in pair_names}
        state = build_state(num_agents, initial_states)
        action = policy.select_action(state, evaluate=True).astype(np.float32)
        barrier_pair_states = action_to_barrier_pair_states(action)

    else:
        raise ValueError(f"Unknown method: {method}")

    geometry = create_geometry(env, barrier_pair_states)

    sim_param = dict(SCENARIO["simulation_mode_training"])
    sim_param["write_trajectory"] = True
    sim_param["save_animation"] = False

    traj_dir = "../logs/eval_trajectories"
    os.makedirs(traj_dir, exist_ok=True)

    trajectory_file = os.path.join(
        traj_dir,
        f"{method}_seed{seed}_agents{num_agents}.sqlite"
    )

    simulation = create_simulation(
        geometry,
        trajectory_file,
        sim_param,
    )

    set_seed(seed)
    selected_groups = select_agent_subset(
        agent_groups,
        max_agents=num_agents,
        shuffle=True,
    )

    added, skipped = add_valid_agents_to_simulation(
        simulation,
        selected_groups,
        geometry,
    )

    result = run_simulation(
        simulation,
        SCENARIO["max_iterations"],
    )

    reward, metrics = compute_reward(
        result,
        trajectory_file=trajectory_file,
        geometry=geometry,
        debug=False,
    )

    row = {
        "method": method,
        "seed": seed,
        "num_agents_requested": num_agents,
        "added_agents": added,
        "skipped_agents": skipped,
        "reward": reward,
        "raw_reward": metrics["raw_reward"],
        "raw_cost": metrics["raw_cost"],
        "remaining_agents": result["remaining_agents"],
        "iterations": result["iterations"],
        "elapsed_time": result["elapsed_time"],
        "evacuation_ratio": metrics["evacuation_ratio"],
        "throughput_agents_per_second": metrics["throughput_agents_per_second"],
        "mean_speed": metrics["mean_speed"],
        "speed_loss": metrics["speed_loss"],
        "stopped_ratio": metrics["stopped_ratio"],
        "mean_density": metrics["voronoi_mean_density"],
        "max_density": metrics["voronoi_max_density"],
    }

    for p in pair_names:
        row[p] = barrier_pair_states[p]

    return row


def main():
    os.makedirs("../logs", exist_ok=True)

    with open(POLICY_PATH, "rb") as f:
        policy = CPU_Unpickler(f).load()

    cpu_device = torch.device("cpu")
    policy.device = cpu_device

    for name in ["actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2"]:
        if hasattr(policy, name):
            getattr(policy, name).to(cpu_device)

    for name in ["actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2"]:
        if hasattr(policy, name):
            getattr(policy, name).to(cpu_device)

    _, env = load_environment(SCENARIO["env_json"])

    base_geometry = create_geometry(
        env,
        SCENARIO["barrier_pair_states"],
    )

    agent_groups = load_agents(base_geometry)

    methods = [
        "all_open",
        "all_closed",
        "random",
        "rl_policy",
    ]

    rows = []

    for seed in SEEDS:
        for num_agents in NUM_AGENTS_LIST:
            for method in methods:
                print(f"Running {method} | seed={seed} | agents={num_agents}")

                row = run_one_case(
                    policy=policy,
                    env=env,
                    agent_groups=agent_groups,
                    method=method,
                    seed=seed,
                    num_agents=num_agents,
                )

                rows.append(row)

                pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved evaluation to:", OUTPUT_CSV)
    print("\nMean results:")
    print(
        df.groupby("method")[
            [
                "reward",
                "raw_reward",
                "evacuation_ratio",
                "throughput_agents_per_second",
                "mean_speed",
                "stopped_ratio",
                "remaining_agents",
            ]
        ].mean()
    )

    try:
        from utils.plot_evaluation import make_evaluation_plots
        make_evaluation_plots(OUTPUT_CSV)
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()