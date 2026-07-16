# runners/evaluate.py

import io
import os
import pickle
import random
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from shapely import wkt
from tqdm import tqdm

from config.eval_config import EVALUATION_CONFIG
from utils.RL_utils import *
from utils.barrier_control import clean_geom
import torch
import network.sac_agent


_WORKER_ENV = None
_WORKER_AGENT_GROUPS = None
_WORKER_POLICY = None
_WORKER_TRAJECTORIES_DIR = None
_WORKER_SIMULATION_PARAM = None
_WORKER_KEEP_TRAJECTORIES = False


class CPUUnpickler(pickle.Unpickler):
    """
    Load older SAC checkpoints on CPU.

    Older checkpoints were saved when sac_agent.py was imported as:
        sac_agent

    The current project structure uses:
        network.sac_agent
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(
                io.BytesIO(data),
                map_location=torch.device("cpu"),
                weights_only=False,
            )

        # Remap old checkpoint module paths.
        module_aliases = {
            "sac_agent": "network.sac_agent",
        }

        module = module_aliases.get(module, module)

        return super().find_class(module, name)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_policy_cpu(policy_path):
    with open(policy_path, "rb") as file:
        policy = CPUUnpickler(file).load()

    device = torch.device("cpu")
    policy.device = device

    for name in [
        "actor",
        "critic_1",
        "critic_2",
        "target_critic_1",
        "target_critic_2",
    ]:
        if hasattr(policy, name):
            getattr(policy, name).to(device)

    return policy


def get_original_geometry(env):
    geometry_wkt = env["geometry_wkt"]

    if "simulation_geometry" in geometry_wkt:
        return clean_geom(wkt.loads(geometry_wkt["simulation_geometry"]))

    return clean_geom(wkt.loads(geometry_wkt["connected_geometry"]))


def get_rule_based_states(num_agents, pair_names):
    rules = EVALUATION_CONFIG.get(
        "rule_based_openings",
        [
            {"max_agents": 300, "opening": 0.45},
            {"max_agents": 500, "opening": 0.60},
            {"max_agents": 700, "opening": 0.70},
            {"max_agents": None, "opening": 0.80},
        ],
    )

    opening = rules[-1]["opening"]

    for rule in rules:
        max_agents = rule.get("max_agents")
        if max_agents is None or num_agents <= max_agents:
            opening = rule["opening"]
            break

    return {pair: float(opening) for pair in pair_names}


def build_barrier_states(method, seed, num_agents, pair_names):
    if method == "all_open":
        return {pair: 1.0 for pair in pair_names}

    if method == "all_closed":
        return {pair: 0.0 for pair in pair_names}

    if method == "random":
        set_seed(seed)
        action = np.random.uniform(0.0, 1.0, size=len(pair_names)).astype(np.float32)
        return action_to_barrier_pair_states(action)

    if method == "rule_based":
        return get_rule_based_states(num_agents, pair_names)

    raise ValueError(f"Unknown evaluation method: {method}")


def init_worker(
    env,
    agent_groups,
    policy_path,
    trajectories_dir,
    simulation_param,
    keep_trajectories,
):
    global _WORKER_ENV
    global _WORKER_AGENT_GROUPS
    global _WORKER_POLICY
    global _WORKER_TRAJECTORIES_DIR
    global _WORKER_SIMULATION_PARAM
    global _WORKER_KEEP_TRAJECTORIES

    _WORKER_ENV = env
    _WORKER_AGENT_GROUPS = agent_groups
    _WORKER_TRAJECTORIES_DIR = Path(trajectories_dir)
    _WORKER_SIMULATION_PARAM = simulation_param
    _WORKER_KEEP_TRAJECTORIES = keep_trajectories
    _WORKER_TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

    if "rl_policy" in EVALUATION_CONFIG["methods"]:
        _WORKER_POLICY = load_policy_cpu(policy_path)


def run_evaluation_job(job):
    trajectory_file = None

    try:
        method = job["method"]
        seed = job["seed"]
        num_agents = job["num_agents"]
        set_seed(seed)

        pair_names = list(EVALUATION_CONFIG["barrier_pairs"].keys())

        if method == "gt_original":
            geometry = get_original_geometry(_WORKER_ENV)
            barrier_states = {pair: np.nan for pair in pair_names}

        elif method == "rl_policy":
            if EVALUATION_CONFIG["random_policy_initial_state"]:
                initial_states = {
                    pair: np.random.uniform(0.0, 1.0)
                    for pair in pair_names
                }
            else:
                initial_states = EVALUATION_CONFIG["policy_initial_barrier_states"]

            state = build_state(
                num_agents=num_agents,
                barrier_pair_states=initial_states,
            )
            action = _WORKER_POLICY.select_action(state, evaluate=True).astype(np.float32)
            barrier_states = action_to_barrier_pair_states(action)
            geometry = create_geometry(_WORKER_ENV, barrier_states)

        else:
            barrier_states = build_barrier_states(
                method,
                seed,
                num_agents,
                pair_names,
            )
            geometry = create_geometry(_WORKER_ENV, barrier_states)

        simulation_param = dict(_WORKER_SIMULATION_PARAM)
        simulation_param["write_trajectory"] = True
        simulation_param["save_animation"] = False

        trajectory_path = _WORKER_TRAJECTORIES_DIR / (
            f"{method}_seed_{seed}_agents_{num_agents}_pid_{os.getpid()}.sqlite"
        )
        trajectory_file = str(trajectory_path)

        simulation = create_simulation(
            geometry,
            trajectory_file,
            simulation_param,
        )

        set_seed(seed)
        selected_groups = select_agent_subset(
            _WORKER_AGENT_GROUPS,
            max_agents=num_agents,
            shuffle=EVALUATION_CONFIG.get("shuffle_agents", True),
        )

        added, skipped = add_valid_agents_to_simulation(
            simulation,
            selected_groups,
            geometry,
        )

        result = run_simulation(
            simulation,
            EVALUATION_CONFIG["max_iterations"],
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
            "raw_reward": metrics.get("raw_reward"),
            "raw_cost": metrics.get("raw_cost"),
            "remaining_agents": result["remaining_agents"],
            "iterations": result["iterations"],
            "elapsed_time": result["elapsed_time"],
            "evacuation_ratio": metrics.get("evacuation_ratio"),
            "throughput_agents_per_second": metrics.get("throughput_agents_per_second"),
            "mean_speed": metrics.get("mean_speed"),
            "speed_loss": metrics.get("speed_loss"),
            "stopped_ratio": metrics.get("stopped_ratio"),
            "mean_density": metrics.get("voronoi_mean_density"),
            "max_density": metrics.get("voronoi_max_density"),
            "classic_mean_density": metrics.get("classic_mean_density"),
            "classic_max_density": metrics.get("classic_max_density"),
        }

        for pair in pair_names:
            row[pair] = barrier_states[pair]

        if trajectory_file and not _WORKER_KEEP_TRAJECTORIES:
            try:
                os.remove(trajectory_file)
            except OSError:
                pass

        return {"ok": True, "row": row}

    except Exception as error:
        if trajectory_file and not _WORKER_KEEP_TRAJECTORIES:
            try:
                os.remove(trajectory_file)
            except OSError:
                pass

        return {
            "ok": False,
            "job": job,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def aggregate_results(results_df):
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [column for column in numeric_columns if column != "seed"]

    return (
        results_df
        .groupby(["method", "num_agents_requested"], as_index=False)[numeric_columns]
        .mean()
    )


def make_evaluation_plot(summary_df, output_file):
    display_names = EVALUATION_CONFIG.get("method_display_names", {})
    method_order = [
        method
        for method in EVALUATION_CONFIG["methods"]
        if method in summary_df["method"].unique()
    ]

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    plot_specs = [
        (axes[0, 0], "remaining_agents", "(a) Remaining agents vs crowd size", "Remaining agents"),
        (axes[0, 1], "raw_reward", "(b) Reward vs crowd size", "Raw reward"),
        (axes[1, 0], "stopped_ratio", "(c) Stopped ratio vs crowd size", "Stopped ratio"),
        (axes[1, 1], "throughput_agents_per_second", "(d) Throughput vs crowd size", "Agents / second"),
    ]

    for axis, metric, title, ylabel in plot_specs:
        if metric not in summary_df.columns or summary_df[metric].isna().all():
            if metric == "raw_reward":
                metric = "reward"

        for method in method_order:
            method_df = (
                summary_df[summary_df["method"] == method]
                .sort_values("num_agents_requested")
            )
            axis.plot(
                method_df["num_agents_requested"],
                method_df[metric],
                marker="o",
                label=display_names.get(method, method),
            )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend()

    axes[1, 0].set_xlabel("Number of agents")
    axes[1, 1].set_xlabel("Number of agents")

    figure.tight_layout()
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate(log_dir):
    """Run all configured evaluation cases and save results under log_dir."""
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    trajectories_dir = log_dir / "trajectories"
    worker_errors_dir = log_dir / "worker_errors"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    worker_errors_dir.mkdir(parents=True, exist_ok=True)

    results_csv = log_dir / "evaluation_results.csv"
    summary_csv = log_dir / "evaluation_summary.csv"
    plot_file = log_dir / "evaluation_comparison.png"

    env_json = Path(EVALUATION_CONFIG["env_json"])
    if not env_json.is_absolute():
        env_json = (Path.cwd() / env_json).resolve()

    policy_path = Path(EVALUATION_CONFIG["policy_path"])
    if not policy_path.is_absolute():
        policy_path = (Path.cwd() / policy_path).resolve()

    methods = EVALUATION_CONFIG["methods"]

    if "rl_policy" in methods and not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    _, env = load_environment(str(env_json))
    print(f"Environment loaded: {env_json}")

    if "gt_original" in methods:
        base_geometry = get_original_geometry(env)
    else:
        base_geometry = create_geometry(
            env,
            EVALUATION_CONFIG["barrier_pair_states"],
        )

    agent_groups = load_agents(base_geometry)

    jobs = [
        {"seed": seed, "num_agents": num_agents, "method": method}
        for seed in EVALUATION_CONFIG["seeds"]
        for num_agents in EVALUATION_CONFIG["num_agents_list"]
        for method in methods
    ]

    available_cpus = os.cpu_count() or 1
    num_workers = min(
        int(EVALUATION_CONFIG.get("num_workers", available_cpus)),
        available_cpus,
    )

    if num_workers < 1:
        raise ValueError("'num_workers' must be at least 1.")

    simulation_param = dict(EVALUATION_CONFIG["simulation"])
    keep_trajectories = bool(EVALUATION_CONFIG.get("keep_trajectories", False))

    print(f"Methods: {methods}")
    print(f"Seeds: {EVALUATION_CONFIG['seeds']}")
    print(f"Crowd sizes: {EVALUATION_CONFIG['num_agents_list']}")
    print(f"Total evaluation jobs: {len(jobs)}")
    print(f"CPU workers: {num_workers}")
    print(f"Output directory: {log_dir}")

    rows = []
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
        initializer=init_worker,
        initargs=(
            env,
            agent_groups,
            str(policy_path),
            str(trajectories_dir),
            simulation_param,
            keep_trajectories,
        ),
    ) as executor:
        future_to_job = {
            executor.submit(run_evaluation_job, job): job
            for job in jobs
        }

        progress = tqdm(total=len(jobs), desc="Evaluation jobs")

        for future in as_completed(future_to_job):
            job = future_to_job[future]

            try:
                result = future.result()
            except Exception as error:
                result = {
                    "ok": False,
                    "job": job,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }

            if result["ok"]:
                rows.append(result["row"])
                pd.DataFrame(rows).to_csv(results_csv, index=False)
                progress.set_postfix(
                    method=result["row"]["method"],
                    agents=result["row"]["num_agents_requested"],
                    seed=result["row"]["seed"],
                )
            else:
                failed_job = result["job"]
                error_file = worker_errors_dir / (
                    f"{failed_job['method']}_seed_{failed_job['seed']}"
                    f"_agents_{failed_job['num_agents']}.txt"
                )
                error_file.write_text(
                    (
                        f"Job: {failed_job}\n\n"
                        f"Error: {result['error']}\n\n"
                        f"{result['traceback']}"
                    ),
                    encoding="utf-8",
                )
                print(f"\nEvaluation failed: {failed_job}")
                print(f"Error saved: {error_file}")

            progress.update(1)

        progress.close()

    if not rows:
        raise RuntimeError(
            "All evaluation jobs failed. "
            f"Check: {worker_errors_dir}"
        )

    results_df = pd.DataFrame(rows).sort_values(
        ["seed", "num_agents_requested", "method"]
    )
    results_df.to_csv(results_csv, index=False)

    summary_df = aggregate_results(results_df)
    summary_df.to_csv(summary_csv, index=False)
    make_evaluation_plot(summary_df, plot_file)

    print("\nEvaluation completed.")
    print("Detailed results:", results_csv)
    print("Mean results:", summary_csv)
    print("Comparison plot:", plot_file)

    return {
        "results": results_df,
        "summary": summary_df,
        "log_dir": str(log_dir),
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "plot_file": str(plot_file),
        "trajectories_dir": str(trajectories_dir),
        "worker_errors_dir": str(worker_errors_dir),
    }
