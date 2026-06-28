import os
from absl import app
from tqdm import tqdm, trange
from shapely.geometry import Point

from scenario_config import SCENARIO
from barrier_control import apply_barrier_pair_states
from simulation_utils import *
import random
from sac_agent import SACAgent, ReplayBuffer
import numpy as np
import pickle
from RL_utils import *


def train_RL():

    env_json = SCENARIO["env_json"]
    _, env = load_environment(env_json)

    print(f"Environment loaded: {env_json}")

    base_geometry = create_geometry(
        env,
        SCENARIO["barrier_pair_states"],
    )

    agent_groups = load_agents(base_geometry)

    if SCENARIO["training"]:
        simulation_pram = SCENARIO["simulation_mode_training"]
    else:
        simulation_pram = SCENARIO["simulation_mode_vis"]

    policy = SACAgent(state_dim=11, action_dim=7)
    replay_buffer = ReplayBuffer(max_size=100000)
    history = []

    print('Start Training Episodes ....')

    for episode in trange(SCENARIO["num_episodes"], desc="Training episodes"):

        case = reset_training_case(episode)

        num_agents = case["num_agents"]
        state = case["state"]
        initial_barrier_states = case["initial_barrier_states"]
        stage_name = case["stage_name"]
        print('initial_barrier_states: ', initial_barrier_states)

        episode_reward = 0.0

        for step in range(SCENARIO["num_steps"]):

            if episode < SCENARIO["start_random_episodes"]:
                action = np.random.uniform(0.0, 1.0, size=7)
            else:
                action = policy.select_action(state, evaluate=False)

            barrier_pair_states = action_to_barrier_pair_states(action)
            print("Action states:", {k: round(v, 2) for k, v in barrier_pair_states.items()})

            geometry = create_geometry(
                env,
                barrier_pair_states,
            )

            trajectory_file = None
            if simulation_pram.get("write_trajectory", False):
                root, ext = os.path.splitext(SCENARIO["trajectory_file"])
                trajectory_file = f"{root}_episode_{episode + 1}_step_{step + 1}{ext}"

            simulation = create_simulation(
                geometry,
                trajectory_file,
                simulation_pram,
            )

            episode_agent_groups = select_agent_subset(
                agent_groups,
                max_agents=num_agents,
                shuffle=simulation_pram["shuffle_agents_each_episode"],
            )

            total_added, total_skipped = add_valid_agents_to_simulation(
                simulation,
                episode_agent_groups,
                geometry,
            )

            result = run_simulation(
                simulation,
                SCENARIO["max_iterations"],
            )

            reward = compute_reward(result, debug=True)
            episode_reward += reward

            next_state = build_state(
                num_agents=num_agents,
                barrier_pair_states=barrier_pair_states,
            )

            done = 1.0

            replay_buffer.add(state, action, reward, next_state, done)

            losses = policy.train(
                replay_buffer,
                batch_size=SCENARIO["batch_size_rl"],
            )

            print(
                f"Episode={episode}, stage={stage_name}, agents={num_agents}, "
                f"added={total_added}, skipped={total_skipped}, "
                f"reward={reward:.3f}, action={np.round(action, 2)}, losses={losses}"
            )

            state = next_state

            if simulation_pram.get("save_animation", False) and simulation_pram.get("write_trajectory", False):
                save_animation(
                    simulation_pram["every_nth_frame"],
                    trajectory_file,
                    SCENARIO["html_file"],
                    title=SCENARIO["name"],
                )

                print("Animation saved:", SCENARIO["html_file"])

        should_save_stage = (episode + 1) % SCENARIO["save_every_episodes"] == 0
        should_save_best = episode_reward >= SCENARIO["best_reward_threshold"]

        if should_save_stage or should_save_best or ((episode + 1) % SCENARIO["eval_freq_rl"] == 0):
            save_policy_checkpoint(
                policy=policy,
                episode=episode,
                reward=episode_reward,
                stage_name=stage_name,
            )

        history.append({
            "episode": episode,
            "stage": stage_name,
            "num_agents": num_agents,
            "added_agents": total_added,
            "skipped_agents": total_skipped,
            "reward": episode_reward,
            "remaining_agents": result["remaining_agents"],
            "iterations": result["iterations"],
            "elapsed_time": result["elapsed_time"],
            "actor_loss": None if losses is None else losses["actor_loss"],
            "critic_1_loss": None if losses is None else losses["critic_1_loss"],
            "critic_2_loss": None if losses is None else losses["critic_2_loss"],
            "pair_1_action": action[0],
            "pair_2_action": action[1],
            "pair_3_action": action[2],
            "pair_4_action": action[3],
            "pair_5_action": action[4],
            "pair_6_action": action[5],
            "pair_7_action": action[6],
        })

    save_training_plots(history)





def main(argv):
    train_RL()



if __name__ == "__main__":
    app.run(main)

