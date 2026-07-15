# runners/train_sequential.py

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

from config.trainig_config import TRAINING_CONFIG
from network.sac_agent import SACAgent, ReplayBuffer
from utils.RL_utils import *


@contextmanager
def working_directory(path):
    """
    Temporarily run code from a specific directory.

    This keeps compatibility with existing helper functions such as
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


def train_sequential(log_dir):
    """
    Train the SAC policy sequentially.

    All generated outputs are stored under log_dir:
        log_dir/
        ├── trajectories/
        ├── animations/
        ├── checkpoints/
        ├── plots/
        └── training_history.csv
    """
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    trajectories_dir = log_dir / "trajectories"
    animations_dir = log_dir / "animations"
    checkpoints_dir = log_dir / "checkpoints"
    plots_dir = log_dir / "plots"

    for directory in (
        trajectories_dir,
        animations_dir,
        checkpoints_dir,
        plots_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    history_file = log_dir / "training_history.csv"

    print(f"Sequential-training outputs: {log_dir}")

    # Resolve the environment path before using a temporary working directory.
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
        simulation_pram = dict(
            TRAINING_CONFIG["simulation_mode_training"]
        )
    else:
        simulation_pram = dict(
            TRAINING_CONFIG["simulation_mode_vis"]
        )

    policy = SACAgent(
        state_dim=11,
        action_dim=7,
    )

    replay_buffer = ReplayBuffer(
        max_size=TRAINING_CONFIG.get(
            "replay_buffer_size",
            100000,
        )
    )

    history = []

    num_episodes = TRAINING_CONFIG["num_episodes"]
    num_steps = TRAINING_CONFIG["num_steps"]

    if num_steps < 1:
        raise ValueError(
            "'num_steps' must be at least 1 for sequential training."
        )

    print("Start training episodes...")

    for episode in trange(
        num_episodes,
        desc="Training episodes",
    ):
        episode_number = episode + 1

        case = reset_training_case(episode)

        if "fixed_epsilon" in case:
            epsilon = case["fixed_epsilon"]
        else:
            epsilon = get_stage_epsilon(
                episode=episode,
                stage_start_episode=case["stage_start"],
                stage_length=case["stage_length"],
            )

        num_agents = case["num_agents"]
        state = case["state"]
        initial_barrier_states = case["initial_barrier_states"]
        stage_name = case["stage_name"]

        print("Initial barrier states:", initial_barrier_states)
        print("Reset state:", state)

        episode_reward = 0.0

        print(
            f"Episode={episode_number}, "
            f"stage={stage_name}, "
            f"epsilon={epsilon:.3f}"
        )

        # These values are updated during each step and the final step values
        # are stored in the episode-level history row.
        total_added = 0
        total_skipped = 0
        result = None
        reward_metrics = None
        losses = None
        action = None

        for step in range(num_steps):
            step_number = step + 1

            if np.random.rand() < epsilon:
                action = np.random.uniform(
                    0.0,
                    1.0,
                    size=7,
                )
            else:
                action = policy.select_action(
                    state,
                    evaluate=False,
                )

            barrier_pair_states = action_to_barrier_pair_states(
                action
            )

            print(
                "Action states:",
                {
                    key: round(value, 2)
                    for key, value in barrier_pair_states.items()
                },
            )

            geometry = create_geometry(
                env,
                barrier_pair_states,
            )

            trajectory_file = None

            if simulation_pram.get("write_trajectory", False):
                trajectory_file = trajectories_dir / (
                    f"trajectory_episode_{episode_number:04d}"
                    f"_step_{step_number:03d}.sqlite"
                )
                trajectory_file = str(trajectory_file)

            simulation = create_simulation(
                geometry,
                trajectory_file,
                simulation_pram,
            )

            episode_agent_groups = select_agent_subset(
                agent_groups,
                max_agents=num_agents,
                shuffle=simulation_pram.get(
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
                debug=True,
            )

            episode_reward += reward

            next_state = build_state(
                num_agents=num_agents,
                barrier_pair_states=barrier_pair_states,
            )

            # Each simulation rollout is treated as a complete transition.
            done = 1.0

            # Add the measured mean speed to the current state before saving
            # the transition in the replay buffer.
            state[0] = reward_metrics["mean_speed"]

            print("State before replay buffer:", state)

            replay_buffer.add(
                state,
                action,
                reward,
                next_state,
                done,
            )

            losses = policy.train(
                replay_buffer,
                batch_size=TRAINING_CONFIG["batch_size_rl"],
            )

            print(
                f"Episode={episode_number}, "
                f"step={step_number}, "
                f"stage={stage_name}, "
                f"agents={num_agents}, "
                f"added={total_added}, "
                f"skipped={total_skipped}, "
                f"reward={reward:.3f}, "
                f"action={np.round(action, 2)}, "
                f"losses={losses}"
            )

            state = next_state

            if (
                simulation_pram.get("save_animation", False)
                and simulation_pram.get("write_trajectory", False)
                and trajectory_file is not None
            ):
                html_file = animations_dir / (
                    f"animation_episode_{episode_number:04d}"
                    f"_step_{step_number:03d}.html"
                )

                save_animation(
                    simulation_pram["every_nth_frame"],
                    trajectory_file,
                    str(html_file),
                    title=(
                        f"{TRAINING_CONFIG['name']} — "
                        f"Episode {episode_number}, "
                        f"Step {step_number}"
                    ),
                )

                print("Animation saved:", html_file)

        if result is None or reward_metrics is None or action is None:
            raise RuntimeError(
                "The episode finished without producing a simulation result."
            )

        should_save_stage = (
            episode_number
            % TRAINING_CONFIG["save_every_episodes"]
            == 0
        )

        should_save_best = (
            episode_reward
            >= TRAINING_CONFIG["best_reward_threshold"]
        )

        should_evaluate = (
            episode_number
            % TRAINING_CONFIG["eval_freq_rl"]
            == 0
        )

        history.append({
            "episode": episode_number,
            "stage": stage_name,
            "num_agents": num_agents,
            "added_agents": total_added,
            "skipped_agents": total_skipped,
            "reward": episode_reward,

            "speed_loss": reward_metrics["speed_loss"],
            "mean_speed": reward_metrics["mean_speed"],
            "min_speed": reward_metrics["min_speed"],
            "speed_05": reward_metrics["speed_05"],
            "speed_10": reward_metrics["speed_10"],
            "stopped_ratio": reward_metrics["stopped_ratio"],

            "evacuation_ratio": reward_metrics["evacuation_ratio"],
            "throughput_agents_per_second": (
                reward_metrics["throughput_agents_per_second"]
            ),

            "mean_density": reward_metrics["voronoi_mean_density"],
            "max_density": reward_metrics["voronoi_max_density"],

            "classic_mean_density": (
                reward_metrics["classic_mean_density"]
            ),
            "classic_max_density": (
                reward_metrics["classic_max_density"]
            ),

            "remaining_agents": result["remaining_agents"],
            "iterations": result["iterations"],
            "elapsed_time": result["elapsed_time"],

            "actor_loss": (
                None
                if losses is None
                else losses["actor_loss"]
            ),
            "critic_1_loss": (
                None
                if losses is None
                else losses["critic_1_loss"]
            ),
            "critic_2_loss": (
                None
                if losses is None
                else losses["critic_2_loss"]
            ),

            "pair_1_action": action[0],
            "pair_2_action": action[1],
            "pair_3_action": action[2],
            "pair_4_action": action[3],
            "pair_5_action": action[4],
            "pair_6_action": action[5],
            "pair_7_action": action[6],
        })

        # Save after every episode so progress is preserved if training stops.
        pd.DataFrame(history).to_csv(
            history_file,
            index=False,
        )

        if should_save_stage or should_save_best or should_evaluate:
            # These helper functions currently use their original signatures.
            # Running them from their target folders keeps relative outputs
            # under this training run's log directory.
            with working_directory(checkpoints_dir):
                save_policy_checkpoint(
                    policy=policy,
                    episode=episode,
                    reward=episode_reward,
                    stage_name=stage_name,
                )

            with working_directory(plots_dir):
                save_training_plots(history)

    print("Sequential training completed.")
    print("Training history saved:", history_file)
    print("Checkpoints directory:", checkpoints_dir)
    print("Plots directory:", plots_dir)
    print("Trajectories directory:", trajectories_dir)
    print("Animations directory:", animations_dir)

    return {
        "policy": policy,
        "history": history,
        "log_dir": str(log_dir),
        "history_file": str(history_file),
        "checkpoints_dir": str(checkpoints_dir),
        "plots_dir": str(plots_dir),
        "trajectories_dir": str(trajectories_dir),
        "animations_dir": str(animations_dir),
    }