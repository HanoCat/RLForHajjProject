from config.base_config import CONFIG


TRAINING_CONFIG = {
    **CONFIG,

    # ------------------------------------------------------------------
    # Experiment settings
    # ------------------------------------------------------------------

    # Name used in animation titles and experiment output descriptions.
    "name": "APBC-RL",

    # Maximum number of JuPedSim iterations for each rollout.
    "max_iterations": 1500,

    # True uses the faster training simulation settings.
    # False uses visualization settings with HTML animations enabled.
    "training": True,


    # ------------------------------------------------------------------
    # Agent placement and movement
    # ------------------------------------------------------------------

    # Minimum spacing between randomly generated agents.
    "min_agent_distance": 0.4,

    # Margin maintained between generated areas and geometry boundaries.
    "safe_distance": 0.2,

    # Desired pedestrian speed range.
    "speed_min": 1.0,
    "speed_max": 1.4,


    # ------------------------------------------------------------------
    # Reinforcement learning settings
    # ------------------------------------------------------------------

    # Total number of training episodes.
    "num_episodes": 100,

    # Sequential mode:
    #   Number of consecutive simulation rollouts per episode.
    #
    # Parallel mode:
    #   Number of simulation rollout jobs submitted per episode.
    "num_steps": 10,

    # Number of replay-buffer samples used for each SAC update.
    "batch_size_rl": 64,

    # Maximum number of transitions stored in the replay buffer.
    "replay_buffer_size": 100000,

    # Save checkpoints and plots every N episodes.
    "save_every_episodes": 25,

    # Also trigger checkpoint and plot saving every N episodes.
    "eval_freq_rl": 25,

    # Save a checkpoint when the episode reward reaches this value.
    "best_reward_threshold": 0.85,


    # ------------------------------------------------------------------
    # Curriculum settings
    # ------------------------------------------------------------------

    # Allow occasional heavy-crowd samples during early training.
    "early_heavy_until_episode": 20,

    # Probability of selecting an early heavy-crowd sample.
    "early_heavy_probability": 0.05,

    # Curriculum learning stages.
    "stages_test": [
        [0, 20],
        [20, 50],
        [50, 80],
        [80, 100],
    ],

    # ------------------------------------------------------------------
    # Parallel training settings
    # ------------------------------------------------------------------

    # Number of CPU worker processes running JuPedSim rollouts.
    "num_parallel_workers": 10,

    # Number of SAC gradient updates after each parallel rollout batch.
    "train_updates_per_batch": 10,

    # Keep worker SQLite trajectory files after metrics are calculated.
    # False saves disk space and is recommended during training.
    "keep_worker_trajectories": False,


    # ------------------------------------------------------------------
    # Simulation modes
    # ------------------------------------------------------------------

    # Fast settings used during RL training.
    "simulation_mode_training": {
        # Simulation time step in seconds.
        "dt": 0.05,

        # Randomize the selected agent subset for each rollout.
        "shuffle_agents_each_episode": True,

        # Store one trajectory frame every N simulation iterations.
        "every_nth_frame": 20,

        # Required for trajectory-based PedPy reward metrics.
        "write_trajectory": True,

        # Keep disabled during training to avoid animation overhead.
        "save_animation": False,
    },

    # Settings used for debugging and visual inspection.
    "simulation_mode_vis": {
        "dt": 0.05,
        "shuffle_agents_each_episode": True,

        # Smaller values generate smoother animations but larger files.
        "every_nth_frame": 5,

        "write_trajectory": True,
        "save_animation": True,
    },



    # ------------------------------------------------------------------
    # Initial barrier states
    # ------------------------------------------------------------------

    # Initial state used for fixed curriculum cases.
    #
    # 0.0 = fully closed
    # 1.0 = fully open
    # Values between 0.0 and 1.0 represent intermediate poses.
    "barrier_pair_states": {
        "pair_1": 0.0,
        "pair_2": 1.0,
        "pair_3": 1.0,
        "pair_4": 1.0,
        "pair_5": 1.0,
        "pair_6": 1.0,
        "pair_7": 1.0,
    },


    # ------------------------------------------------------------------
    # Agent initialization
    # ------------------------------------------------------------------

    # At least one source must be enabled.
    #
    # Both may be True to combine random and P2PNet positions.
    "p2pnet_load": False,
    "random_agents_load": True,
}