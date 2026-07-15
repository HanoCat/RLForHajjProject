

from config.base_config import CONFIG

EVAL_CONFIG = {
    **CONFIG,

    # this is only for test_reward_sensitivity.py file
    "reward_sensitivity_agent_counts": [10, 50, 100, 200, 800],
    "every_nth_frame_n": 5,
    # simulation scenario settings
    "name": "./log/four_zones_precise_test",
    "trajectory_file": "./log/four_zones_test.sqlite",
    "html_file": "org.html",
    "max_iterations": 1500, # time stop for the simulation.

    "min_agent_distance": 0.4,
    "safe_distance": 0.2,
    "speed_min": 1.0,
    "speed_max": 1.4,

    # SAC train
    "use_exp_reward": True,
    "reward_alpha_exp": 3.0,
    "reward_scale": 1.0,

    "num_episodes": 100,
    "num_steps": 10,  # parallel rollouts per episode
    "stages_test": [[0,20],[20,50],[50,80],[80,100]],
    "stages_train": [[],[],[],[]],
    "start_random_episodes": 2,
    "batch_size_rl": 64,
    "eval_freq_rl": 25,
    "save_every_episodes": 25,
    "best_reward_threshold": 0.85,

    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 200,

    "early_heavy_until_episode": 20,
    "early_heavy_probability": 0.05,


    "training": True,

    # Parallel RL training settings
    "num_parallel_workers": 10,  # start safe on A4000x2/16 CPU; try 12 later if stable
    "train_updates_per_batch": 10,
    "replay_buffer_size": 100000,
    "parallel_trajectory_dir": "/tmp/rl_hajj_trajectories",
    "keep_worker_trajectories": False,

    "simulation_mode_training": {
        "dt": 0.05,
        "training_num_agents": 100,   # use None for all agents
        "shuffle_agents_each_episode": True,
        "every_nth_frame": 20,


        "write_trajectory": True,
        "save_animation": False,
    },

    "simulation_mode_vis": {
        "dt": 0.05,
        "training_num_agents": None,
        "shuffle_agents_each_episode": True,
        "every_nth_frame": 5,


        "write_trajectory": True,
        "save_animation": True,
    },

    # load file of built env
    "env_json": "../json/processed_environment.json",

    "barrier_pair_states": {
    "pair_1": 0.0,
    "pair_2": 1.0,
    "pair_3": 1.0,
    "pair_4": 1.0,
    "pair_5": 1.0,
    "pair_6": 1.0,
    "pair_7": 1.0,
    },



}
