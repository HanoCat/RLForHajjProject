import os

# Keep each CPU worker from secretly using many threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback

from utils.RL_utils import *
from network.sac_agent import SACAgent, ReplayBuffer
from config.scenario_config import SCENARIO


# These are filled once per worker process by init_worker().
_WORKER_ENV = None
_WORKER_AGENT_GROUPS = None
_WORKER_SIMULATION_PARAM = None


def init_worker(env, agent_groups, simulation_param):
    global _WORKER_ENV, _WORKER_AGENT_GROUPS, _WORKER_SIMULATION_PARAM
    _WORKER_ENV = env
    _WORKER_AGENT_GROUPS = agent_groups
    _WORKER_SIMULATION_PARAM = simulation_param

    # Different random stream in every worker.
    seed = (os.getpid() + int(time.time() * 1000)) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)


def run_simulation_job(job):
    """
    CPU-only worker job.
    It creates geometry, runs JuPedSim, writes the SQLite trajectory,
    computes PedPy reward, then returns one SAC transition.
    """
    try:
        episode = job["episode"]
        step = job["step"]
        num_agents = job["num_agents"]
        state = np.asarray(job["state"], dtype=np.float32)
        action = np.asarray(job["action"], dtype=np.float32)
        stage_name = job["stage_name"]

        barrier_pair_states = action_to_barrier_pair_states(action)
        geometry = create_geometry(_WORKER_ENV, barrier_pair_states)

        trajectory_file = None
        if _WORKER_SIMULATION_PARAM.get("write_trajectory", False):
            traj_dir = SCENARIO.get("parallel_trajectory_dir", "/tmp/rl_hajj_trajectories")
            os.makedirs(traj_dir, exist_ok=True)
            trajectory_file = os.path.join(
                traj_dir,
                f"traj_ep{episode + 1:04d}_step{step + 1:03d}_pid{os.getpid()}.sqlite",
            )

        simulation = create_simulation(
            geometry,
            trajectory_file,
            _WORKER_SIMULATION_PARAM,
        )

        episode_agent_groups = select_agent_subset(
            _WORKER_AGENT_GROUPS,
            max_agents=num_agents,
            shuffle=_WORKER_SIMULATION_PARAM["shuffle_agents_each_episode"],
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

        keep_trajectories = SCENARIO.get("keep_worker_trajectories", False)
        if trajectory_file is not None and not keep_trajectories:
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
            "state": replay_state,
            "action": action,
            "reward": float(reward),
            "next_state": next_state,
            "done": 1.0,
            "result": result,
            "reward_metrics": reward_metrics,
            "added_agents": total_added,
            "skipped_agents": total_skipped,
            "trajectory_file": trajectory_file if keep_trajectories else None,
        }

    except Exception as e:
        return {
            "ok": False,
            "episode": job.get("episode"),
            "step": job.get("step"),
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


def make_jobs_for_episode(policy, episode):
    jobs = []

    for step in range(SCENARIO["num_steps"]):
        # Use a different curriculum sample for each parallel step.
        # This gives 10 independent rollouts per episode when num_steps=10.
        case = reset_training_case(episode * SCENARIO["num_steps"] + step)

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
            action = np.random.uniform(0.0, 1.0, size=7).astype(np.float32)
        else:
            action = policy.select_action(state, evaluate=False).astype(np.float32)

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


def train_RL():
    os.makedirs("../logs", exist_ok=True)

    env_json = SCENARIO["env_json"]
    _, env = load_environment(env_json)
    print(f"Environment loaded: {env_json}")

    base_geometry = create_geometry(
        env,
        SCENARIO["barrier_pair_states"],
    )

    agent_groups = load_agents(base_geometry)

    if SCENARIO["training"]:
        simulation_param = SCENARIO["simulation_mode_training"]
    else:
        simulation_param = SCENARIO["simulation_mode_vis"]

    policy = SACAgent(state_dim=11, action_dim=7)
    print(f"SAC device: {policy.device}")

    replay_buffer = ReplayBuffer(max_size=SCENARIO.get("replay_buffer_size", 100000))
    history = []

    num_workers = int(SCENARIO.get("num_parallel_workers", min(8, os.cpu_count() or 1)))
    train_updates_per_batch = int(SCENARIO.get("train_updates_per_batch", SCENARIO["num_steps"]))

    print("Start Parallel Training ....")
    print(f"Episodes: {SCENARIO['num_episodes']}")
    print(f"Parallel rollouts per episode: {SCENARIO['num_steps']}")
    print(f"CPU workers: {num_workers}")
    print(f"Train updates per batch: {train_updates_per_batch}")

    # Spawn avoids CUDA/fork issues because the main process owns the GPU policy.
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(env, agent_groups, simulation_param),
    ) as executor:

        for episode in trange(SCENARIO["num_episodes"], desc="Parallel training episodes"):
            jobs = make_jobs_for_episode(policy, episode)

            futures = [executor.submit(run_simulation_job, job) for job in jobs]
            batch_results = []

            for future in as_completed(futures):
                r = future.result()
                if not r["ok"]:
                    print("Worker failed:", r["error"])
                    print(r["traceback"])
                    continue
                batch_results.append(r)

            if not batch_results:
                print(f"Episode {episode}: all workers failed, skipping.")
                continue

            episode_reward = 0.0
            last_losses = None

            for r in batch_results:
                replay_buffer.add(
                    r["state"],
                    r["action"],
                    r["reward"],
                    r["next_state"],
                    r["done"],
                )
                episode_reward += r["reward"]

            for _ in range(train_updates_per_batch):
                losses = policy.train(
                    replay_buffer,
                    batch_size=SCENARIO["batch_size_rl"],
                )
                if losses is not None:
                    last_losses = losses

            mean_reward = episode_reward / len(batch_results)
            mean_agents = float(np.mean([r["num_agents"] for r in batch_results]))
            mean_added = float(np.mean([r["added_agents"] for r in batch_results]))
            mean_skipped = float(np.mean([r["skipped_agents"] for r in batch_results]))
            mean_remaining = float(np.mean([r["result"]["remaining_agents"] for r in batch_results]))
            mean_iterations = float(np.mean([r["result"]["iterations"] for r in batch_results]))
            mean_elapsed = float(np.mean([r["result"]["elapsed_time"] for r in batch_results]))

            # Use the first result for detailed metric fields, and average the most important ones.
            metric_keys = batch_results[0]["reward_metrics"].keys()
            avg_metrics = {}
            for key in metric_keys:
                values = [r["reward_metrics"][key] for r in batch_results]
                avg_metrics[key] = float(np.mean(values))

            first_action = batch_results[0]["action"]

            history.append({
                "episode": episode,
                "stage": batch_results[0]["stage"],
                "num_agents": mean_agents,
                "added_agents": mean_added,
                "skipped_agents": mean_skipped,
                "reward": mean_reward,

                "speed_loss": avg_metrics["speed_loss"],
                "mean_speed": avg_metrics["mean_speed"],
                "min_speed": avg_metrics["min_speed"],
                "speed_05": avg_metrics["speed_05"],
                "speed_10": avg_metrics["speed_10"],
                "stopped_ratio": avg_metrics["stopped_ratio"],

                "evacuation_ratio": avg_metrics["evacuation_ratio"],
                "throughput_agents_per_second": avg_metrics["throughput_agents_per_second"],

                "mean_density": avg_metrics["voronoi_mean_density"],
                "max_density": avg_metrics["voronoi_max_density"],

                "classic_mean_density": avg_metrics["classic_mean_density"],
                "classic_max_density": avg_metrics["classic_max_density"],

                "remaining_agents": mean_remaining,
                "iterations": mean_iterations,
                "elapsed_time": mean_elapsed,

                "actor_loss": None if last_losses is None else last_losses["actor_loss"],
                "critic_1_loss": None if last_losses is None else last_losses["critic_1_loss"],
                "critic_2_loss": None if last_losses is None else last_losses["critic_2_loss"],
                "pair_1_action": first_action[0],
                "pair_2_action": first_action[1],
                "pair_3_action": first_action[2],
                "pair_4_action": first_action[3],
                "pair_5_action": first_action[4],
                "pair_6_action": first_action[5],
                "pair_7_action": first_action[6],
            })

            print(
                f"Episode={episode + 1}/{SCENARIO['num_episodes']} | "
                f"rollouts={len(batch_results)} | "
                f"reward_mean={mean_reward:.3f} | "
                f"buffer={len(replay_buffer)} | "
                f"losses={last_losses}"
            )

            should_save_stage = (episode + 1) % SCENARIO["save_every_episodes"] == 0
            should_save_best = mean_reward >= SCENARIO["best_reward_threshold"]
            should_save_eval = (episode + 1) % SCENARIO["eval_freq_rl"] == 0

            if should_save_stage or should_save_best or should_save_eval:
                save_policy_checkpoint(
                    policy=policy,
                    episode=episode,
                    reward=mean_reward,
                    stage_name=batch_results[0]["stage"],
                )
                save_training_plots(history)

    save_policy_checkpoint(
        policy=policy,
        episode=SCENARIO["num_episodes"] - 1,
        reward=history[-1]["reward"] if history else 0.0,
        stage_name="final_parallel",
    )
    save_training_plots(history)
    print("Parallel training finished.")


def main(argv):
    train_RL()


if __name__ == "__main__":
    app.run(main)
