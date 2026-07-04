import os
import io
import pickle
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from scenario_config import SCENARIO
from simulation_utils import *
from RL_utils import *


POLICY_PATH = "RL_model.pickle"
OUTPUT_CSV = "outputs_parallel/evaluation_results_parallel_unseen_size.csv"

SEEDS = [101, 202, 303, 404, 505]
NUM_AGENTS_LIST = [5, 10, 1000, 1500]
METHODS = ["all_open", "all_closed", "random", "rl_policy"]

NUM_EVAL_WORKERS = 10


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
        return super().find_class(module, name)


_WORKER_ENV = None
_WORKER_AGENT_GROUPS = None
_WORKER_POLICY = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_policy_cpu(policy_path):
    with open(policy_path, "rb") as f:
        policy = CPU_Unpickler(f).load()

    cpu_device = torch.device("cpu")
    policy.device = cpu_device

    for name in ["actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2"]:
        if hasattr(policy, name):
            getattr(policy, name).to(cpu_device)

    return policy


def init_worker(env, agent_groups, policy_path):
    global _WORKER_ENV, _WORKER_AGENT_GROUPS, _WORKER_POLICY

    _WORKER_ENV = env
    _WORKER_AGENT_GROUPS = agent_groups
    _WORKER_POLICY = load_policy_cpu(policy_path)


def run_eval_job(job):
    try:
        method = job["method"]
        seed = job["seed"]
        num_agents = job["num_agents"]

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
            action = _WORKER_POLICY.select_action(state, evaluate=True).astype(np.float32)
            barrier_pair_states = action_to_barrier_pair_states(action)

        else:
            raise ValueError(f"Unknown method: {method}")

        geometry = create_geometry(_WORKER_ENV, barrier_pair_states)

        sim_param = dict(SCENARIO["simulation_mode_training"])
        sim_param["write_trajectory"] = True
        sim_param["save_animation"] = False

        traj_dir = "logs/eval_trajectories_parallel"
        os.makedirs(traj_dir, exist_ok=True)

        trajectory_file = os.path.join(
            traj_dir,
            f"{method}_seed{seed}_agents{num_agents}_pid{os.getpid()}.sqlite",
        )

        simulation = create_simulation(
            geometry,
            trajectory_file,
            sim_param,
        )

        set_seed(seed)

        selected_groups = select_agent_subset(
            _WORKER_AGENT_GROUPS,
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

        return {"ok": True, "row": row}

    except Exception as e:
        return {
            "ok": False,
            "job": job,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


def main():
    os.makedirs("logs", exist_ok=True)

    _, env = load_environment(SCENARIO["env_json"])

    base_geometry = create_geometry(
        env,
        SCENARIO["barrier_pair_states"],
    )

    agent_groups = load_agents(base_geometry)

    jobs = []
    for seed in SEEDS:
        for num_agents in NUM_AGENTS_LIST:
            for method in METHODS:
                jobs.append({
                    "seed": seed,
                    "num_agents": num_agents,
                    "method": method,
                })

    print(f"Total evaluation jobs: {len(jobs)}")
    print(f"Evaluation workers: {NUM_EVAL_WORKERS}")

    rows = []

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=NUM_EVAL_WORKERS,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(env, agent_groups, POLICY_PATH),
    ) as executor:

        futures = [executor.submit(run_eval_job, job) for job in jobs]

        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()

            if result["ok"]:
                row = result["row"]
                rows.append(row)

                print(
                    f"[{i}/{len(jobs)}] done | "
                    f"{row['method']} | seed={row['seed']} | "
                    f"agents={row['num_agents_requested']} | "
                    f"raw_reward={row['raw_reward']:.3f} | "
                    f"remaining={row['remaining_agents']}"
                )

                pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

            else:
                print(f"[{i}/{len(jobs)}] FAILED")
                print(result["job"])
                print(result["error"])
                print(result["traceback"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["seed", "num_agents_requested", "method"])
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
                "mean_density",
                "max_density",
            ]
        ].mean()
    )

    try:
        from plot_evaluation import make_evaluation_plots
        make_evaluation_plots(OUTPUT_CSV)
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()