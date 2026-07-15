# runners/train_parallel.py

import os

# Keep each CPU worker from silently using many internal threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

from config.trainig_config import TRAINING_CONFIG
from network.sac_agent import SACAgent, ReplayBuffer
from utils.RL_utils import *


# These values are initialized once inside every worker process.
_WORKER_ENV = None
_WORKER_AGENT_GROUPS = None
_WORKER_SIMULATION_PARAM = None
_WORKER_TRAJECTORIES_DIR = None
_WORKER_KEEP_TRAJECTORIES = False


@contextmanager
def working_directory(path):
    """
    Temporarily run code from a specific directory.

    This preserves compatibility with existing helper functions such as
    save_policy_checkpoint() and save_training_plots() when they save files
    using relative paths.
    """
    previous_directory = Path.cwd()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous_directory)


def init_worker(
    env,
    agent_groups,
    simulation_param,
    trajectories_dir,
    keep_trajectories,
):
    """
    Initialize data shared by each CPU worker process.
    """
    global _WORKER_ENV
    global _WORKER_AGENT_GROUPS
    global _WORKER_SIMULATION_PARAM
    global _WORKER_TRAJECTORIES_DIR
    global _WORKER_KEEP_TRAJECTORIES

    _WORKER_ENV = env
    _WORKER_AGENT_GROUPS = agent_groups
    _WORKER_SIMULATION_PARAM = simulation_param
    _WORKER_TRAJECTORIES_DIR = Path(trajectories_dir)
    _WORKER_KEEP_TRAJECTORIES = keep_trajectories

    _WORKER_TRAJECTORIES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Give every worker an independent random stream.
    seed = (
        os.getpid()
        + int(time.time() * 1000)
    ) % (2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)


def run_simulation_job(job):
    """
    Run one JuPedSim rollout in a CPU worker.

    The worker:
    1. creates the barrier geometry,
    2. creates and runs the JuPedSim simulation,
    3. computes the reward and metrics,
    4. returns one SAC transition to the main process.

    The RL policy remains in the main process so the GPU is not shared
    across worker processes.
    """
    trajectory_file = None

    try:
        episode = job["episode"]
        step = job["step"]
        num_agents = job["num_agents"]
        stage_name = job["stage_name"]

        state = np.asarray(
            job["state"],
            dtype=np.float32,
        )

        action = np.asarray(
            job["action"],
            dtype=np.float32,
        )

        barrier_pair_states = action_to_barrier_pair_states(
            action
        )

        geometry = create_geometry(
            _WORKER_ENV,
            barrier_pair_states,
        )

        if _WORKER_SIMULATION_PARAM.get(
            "write_trajectory",
            False,
        ):
            trajectory_path = _WORKER_TRAJECTORIES_DIR / (
                f"trajectory_episode_{episode + 1:04d}"
                f"_step_{step + 1:03d}"
                f"_pid_{os.getpid()}.sqlite"
            )

            trajectory_file = str(trajectory_path)

        simulation = create_simulation(
            geometry,
            trajectory_file,
            _WORKER_SIMULATION_PARAM,
        )

        episode_agent_groups = select_agent_subset(
            _WORKER_AGENT_GROUPS,
            max_agents=num_agents,
            shuffle=_WORKER_SIMULATION_PARAM.get(
                "shuffle_agents_each_episode",
                False,
            ),
        )

        total_added, total_skipped = (
            add_valid_agents_to_simulation(
                simulation,
                episode_agent_groups,
                geometry,
            )
        )

        result = run_simulation(
            simulation,
            TRAINING_CONFIG["max_iterations"],
        )

        reward, reward_metrics = compute_reward(
            result,
            trajectory_file=trajectory_file,
            geometry=geometry,
            debug=False,
        )

        next_state = build_state(
            num_agents=num_agents,
            barrier_pair_states=barrier_pair_states,
        )

        mean_speed = reward_metrics["mean_speed"]

        replay_state = state.copy()
        replay_state[0] = mean_speed

        next_state = next_state.copy()
        next_state[0] = mean_speed

        saved_trajectory_file = (
            trajectory_file
            if _WORKER_KEEP_TRAJECTORIES
            else None
        )

        if (
            trajectory_file is not None
            and not _WORKER_KEEP_TRAJECTORIES
        ):
            try:
                os.remove(trajectory_file)
            except OSError:
                pass

        return {
            "ok": True,
            "episode": episode,
            "step": step,
            "stage": stage_name,
            "num_agents": num_agents,
            "epsilon": job["epsilon"],
            "state": replay_state,
            "action": action,
            "reward": float(reward),
            "next_state": next_state,
            "done": 1.0,
            "result": result,
            "reward_metrics": reward_metrics,
            "added_agents": total_added,
            "skipped_agents": total_skipped,
            "trajectory_file": saved_trajectory_file,
        }

    except Exception as error:
        # Remove incomplete trajectory files created by failed workers.
        if (
            trajectory_file is not None
            and not _WORKER_KEEP_TRAJECTORIES
        ):
            try:
                os.remove(trajectory_file)
            except OSError:
                pass

        return {
            "ok": False,
            "episode": job.get("episode"),
            "step": job.get("step"),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def make_jobs_for_episode(policy, episode):
    """
    Build independent rollout jobs for one parallel training episode.
    """
    jobs = []

    for step in range(TRAINING_CONFIG["num_steps"]):
        # Each parallel rollout receives its own curriculum sample.
        curriculum_index = (
            episode * TRAINING_CONFIG["num_steps"]
            + step
        )

        case = reset_training_case(curriculum_index)

        if "fixed_epsilon" in case:
            epsilon = case["fixed_epsilon"]
        else:
            epsilon = get_stage_epsilon(
                episode=episode,
                stage_start_episode=case["stage_start"],
                stage_length=case["stage_length"],
            )

        state = case["state"]

        if np.random.rand() < epsilon:
            action = np.random.uniform(
                0.0,
                1.0,
                size=7,
            ).astype(np.float32)
        else:
            action = policy.select_action(
                state,
                evaluate=False,
            ).astype(np.float32)

        jobs.append({
            "episode": episode,
            "step": step,
            "stage_name": case["stage_name"],
            "num_agents": case["num_agents"],
            "state": state,
            "action": action,
            "epsilon": epsilon,
        })

    return jobs


def _average_metric(batch_results, key):
    """
    Safely average one numeric reward metric across successful workers.
    """
    values = [
        result["reward_metrics"].get(key)
        for result in batch_results
    ]

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(np.mean(values))


def train_parallel(log_dir):
    """
    Train SAC using parallel JuPedSim CPU rollouts.

    All generated outputs are stored under log_dir:
        log_dir/
        ├── trajectories/
        ├── checkpoints/
        ├── plots/
        ├── worker_errors/
        └── training_history.csv
    """
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    trajectories_dir = log_dir / "trajectories"
    checkpoints_dir = log_dir / "checkpoints"
    plots_dir = log_dir / "plots"
    worker_errors_dir = log_dir / "worker_errors"

    for directory in (
        trajectories_dir,
        checkpoints_dir,
        plots_dir,
        worker_errors_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    history_file = log_dir / "training_history.csv"

    print(f"Parallel-training outputs: {log_dir}")

    # Resolve this before helper functions temporarily change directories.
    env_json = Path(TRAINING_CONFIG["env_json"])

    if not env_json.is_absolute():
        env_json = (Path.cwd() / env_json).resolve()

    _, env = load_environment(str(env_json))
    print(f"Environment loaded: {env_json}")

    base_geometry = create_geometry(
        env,
        TRAINING_CONFIG["barrier_pair_states"],
    )

    agent_groups = load_agents(base_geometry)

    if TRAINING_CONFIG["training"]:
        simulation_param = dict(
            TRAINING_CONFIG["simulation_mode_training"]
        )
    else:
        simulation_param = dict(
            TRAINING_CONFIG["simulation_mode_vis"]
        )

    policy = SACAgent(
        state_dim=11,
        action_dim=7,
    )

    print(f"SAC device: {policy.device}")

    replay_buffer = ReplayBuffer(
        max_size=TRAINING_CONFIG.get(
            "replay_buffer_size",
            100000,
        )
    )

    history = []

    available_cpus = os.cpu_count() or 1

    num_workers = int(
        TRAINING_CONFIG.get(
            "num_parallel_workers",
            min(8, available_cpus),
        )
    )

    if num_workers < 1:
        raise ValueError(
            "'num_parallel_workers' must be at least 1."
        )

    num_workers = min(
        num_workers,
        available_cpus,
    )

    num_rollouts = int(
        TRAINING_CONFIG["num_steps"]
    )

    if num_rollouts < 1:
        raise ValueError(
            "'num_steps' must be at least 1."
        )

    train_updates_per_batch = int(
        TRAINING_CONFIG.get(
            "train_updates_per_batch",
            num_rollouts,
        )
    )

    if train_updates_per_batch < 1:
        raise ValueError(
            "'train_updates_per_batch' must be at least 1."
        )

    keep_trajectories = bool(
        TRAINING_CONFIG.get(
            "keep_worker_trajectories",
            False,
        )
    )

    print("Start parallel training...")
    print(f"Episodes: {TRAINING_CONFIG['num_episodes']}")
    print(f"Parallel rollouts per episode: {num_rollouts}")
    print(f"Available CPU cores: {available_cpus}")
    print(f"CPU workers used: {num_workers}")
    print(f"Train updates per batch: {train_updates_per_batch}")
    print(f"Keep worker trajectories: {keep_trajectories}")

    # Spawn avoids CUDA/fork problems because the main process owns the GPU.
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
        initializer=init_worker,
        initargs=(
            env,
            agent_groups,
            simulation_param,
            str(trajectories_dir),
            keep_trajectories,
        ),
    ) as executor:
        for episode in trange(
            TRAINING_CONFIG["num_episodes"],
            desc="Parallel training episodes",
        ):
            episode_number = episode + 1

            jobs = make_jobs_for_episode(
                policy,
                episode,
            )

            future_to_job = {
                executor.submit(
                    run_simulation_job,
                    job,
                ): job
                for job in jobs
            }

            batch_results = []

            for future in as_completed(future_to_job):
                job = future_to_job[future]

                try:
                    worker_result = future.result()
                except Exception as error:
                    worker_result = {
                        "ok": False,
                        "episode": job["episode"],
                        "step": job["step"],
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }

                if not worker_result["ok"]:
                    error_file = worker_errors_dir / (
                        f"episode_{episode_number:04d}"
                        f"_step_{job['step'] + 1:03d}.txt"
                    )

                    error_file.write_text(
                        (
                            f"Error: {worker_result['error']}\n\n"
                            f"{worker_result['traceback']}"
                        ),
                        encoding="utf-8",
                    )

                    print(
                        "Worker failed:",
                        worker_result["error"],
                    )
                    print(
                        "Worker traceback saved:",
                        error_file,
                    )
                    continue

                batch_results.append(worker_result)

            if not batch_results:
                print(
                    f"Episode {episode_number}: "
                    "all worker jobs failed; skipping."
                )
                continue

            # Sort results so action columns are deterministic.
            batch_results.sort(
                key=lambda result: result["step"]
            )

            episode_reward = 0.0
            last_losses = None

            for result in batch_results:
                replay_buffer.add(
                    result["state"],
                    result["action"],
                    result["reward"],
                    result["next_state"],
                    result["done"],
                )

                episode_reward += result["reward"]

            for _ in range(train_updates_per_batch):
                losses = policy.train(
                    replay_buffer,
                    batch_size=TRAINING_CONFIG[
                        "batch_size_rl"
                    ],
                )

                if losses is not None:
                    last_losses = losses

            mean_reward = (
                episode_reward
                / len(batch_results)
            )

            mean_agents = float(
                np.mean([
                    result["num_agents"]
                    for result in batch_results
                ])
            )

            mean_added = float(
                np.mean([
                    result["added_agents"]
                    for result in batch_results
                ])
            )

            mean_skipped = float(
                np.mean([
                    result["skipped_agents"]
                    for result in batch_results
                ])
            )

            mean_remaining = float(
                np.mean([
                    result["result"]["remaining_agents"]
                    for result in batch_results
                ])
            )

            mean_iterations = float(
                np.mean([
                    result["result"]["iterations"]
                    for result in batch_results
                ])
            )

            mean_elapsed = float(
                np.mean([
                    result["result"]["elapsed_time"]
                    for result in batch_results
                ])
            )

            mean_epsilon = float(
                np.mean([
                    result["epsilon"]
                    for result in batch_results
                ])
            )

            # Store the first rollout's action only as a representative
            # action for this episode. The reward metrics are batch averages.
            representative_action = batch_results[0]["action"]

            history.append({
                "episode": episode_number,
                "stage": batch_results[0]["stage"],
                "successful_rollouts": len(batch_results),
                "failed_rollouts": (
                    len(jobs) - len(batch_results)
                ),
                "epsilon": mean_epsilon,
                "num_agents": mean_agents,
                "added_agents": mean_added,
                "skipped_agents": mean_skipped,
                "reward": mean_reward,

                "speed_loss": _average_metric(
                    batch_results,
                    "speed_loss",
                ),
                "mean_speed": _average_metric(
                    batch_results,
                    "mean_speed",
                ),
                "min_speed": _average_metric(
                    batch_results,
                    "min_speed",
                ),
                "speed_05": _average_metric(
                    batch_results,
                    "speed_05",
                ),
                "speed_10": _average_metric(
                    batch_results,
                    "speed_10",
                ),
                "stopped_ratio": _average_metric(
                    batch_results,
                    "stopped_ratio",
                ),

                "evacuation_ratio": _average_metric(
                    batch_results,
                    "evacuation_ratio",
                ),
                "throughput_agents_per_second": (
                    _average_metric(
                        batch_results,
                        "throughput_agents_per_second",
                    )
                ),

                "mean_density": _average_metric(
                    batch_results,
                    "voronoi_mean_density",
                ),
                "max_density": _average_metric(
                    batch_results,
                    "voronoi_max_density",
                ),

                "classic_mean_density": _average_metric(
                    batch_results,
                    "classic_mean_density",
                ),
                "classic_max_density": _average_metric(
                    batch_results,
                    "classic_max_density",
                ),

                "remaining_agents": mean_remaining,
                "iterations": mean_iterations,
                "elapsed_time": mean_elapsed,

                "actor_loss": (
                    None
                    if last_losses is None
                    else last_losses["actor_loss"]
                ),
                "critic_1_loss": (
                    None
                    if last_losses is None
                    else last_losses["critic_1_loss"]
                ),
                "critic_2_loss": (
                    None
                    if last_losses is None
                    else last_losses["critic_2_loss"]
                ),

                "pair_1_action": representative_action[0],
                "pair_2_action": representative_action[1],
                "pair_3_action": representative_action[2],
                "pair_4_action": representative_action[3],
                "pair_5_action": representative_action[4],
                "pair_6_action": representative_action[5],
                "pair_7_action": representative_action[6],
            })

            # Save progress every episode so interrupted runs keep their data.
            pd.DataFrame(history).to_csv(
                history_file,
                index=False,
            )

            print(
                f"Episode={episode_number}/"
                f"{TRAINING_CONFIG['num_episodes']} | "
                f"rollouts={len(batch_results)}/{len(jobs)} | "
                f"reward_mean={mean_reward:.3f} | "
                f"buffer={len(replay_buffer)} | "
                f"losses={last_losses}"
            )

            should_save_stage = (
                episode_number
                % TRAINING_CONFIG["save_every_episodes"]
                == 0
            )

            should_save_best = (
                mean_reward
                >= TRAINING_CONFIG["best_reward_threshold"]
            )

            should_save_eval = (
                episode_number
                % TRAINING_CONFIG["eval_freq_rl"]
                == 0
            )

            if (
                should_save_stage
                or should_save_best
                or should_save_eval
            ):
                with working_directory(checkpoints_dir):
                    save_policy_checkpoint(
                        policy=policy,
                        episode=episode,
                        reward=mean_reward,
                        stage_name=batch_results[0]["stage"],
                    )

                with working_directory(plots_dir):
                    save_training_plots(history)

    final_reward = (
        history[-1]["reward"]
        if history
        else 0.0
    )

    with working_directory(checkpoints_dir):
        save_policy_checkpoint(
            policy=policy,
            episode=TRAINING_CONFIG["num_episodes"] - 1,
            reward=final_reward,
            stage_name="final_parallel",
        )

    if history:
        with working_directory(plots_dir):
            save_training_plots(history)

        pd.DataFrame(history).to_csv(
            history_file,
            index=False,
        )

    print("Parallel training completed.")
    print("Training history saved:", history_file)
    print("Checkpoints directory:", checkpoints_dir)
    print("Plots directory:", plots_dir)
    print("Trajectories directory:", trajectories_dir)
    print("Worker errors directory:", worker_errors_dir)

    return {
        "policy": policy,
        "history": history,
        "log_dir": str(log_dir),
        "history_file": str(history_file),
        "checkpoints_dir": str(checkpoints_dir),
        "plots_dir": str(plots_dir),
        "trajectories_dir": str(trajectories_dir),
        "worker_errors_dir": str(worker_errors_dir),
    }
