from config.base_config import CONFIG

EVALUATION_CONFIG = {
    **CONFIG,

    # ------------------------------------------------------------------
    # Trained RL policy
    # ------------------------------------------------------------------

    # Path to the trained SAC policy (.pickle) used for evaluation.
    "policy_path": "./models/RL_model.pickle",


    # ------------------------------------------------------------------
    # Evaluation settings
    # ------------------------------------------------------------------

    # Random seeds used for repeated evaluation.
    # The reported results are averaged across all seeds.
    "seeds": [101, 202, 303, 404, 505],

    # Number of agents evaluated for each seed.
    "num_agents_list": [5, 300, 500, 700, 900, 1000, 1500],

    # Evaluation methods.
    #
    # rl_policy:
    #     Adaptive barrier configuration predicted by the trained RL policy.
    #
    # gt_original:
    #     Original environment without adaptive barrier control.
    #
    # all_open:
    #     All barrier pairs fully open.
    #
    # all_closed:
    #     All barrier pairs fully closed.
    #
    # random:
    #     Random barrier configuration (optional baseline).
    #
    # rule_based:
    #     User-defined rule-based barrier configuration (optional baseline).
    "methods": [
        "rl_policy",
        "gt_original",
        "all_open",
        "all_closed",
    ],

    # Display names used in plots and result figures.
    "method_display_names": {
        "rl_policy": "RL policy",
        "gt_original": "Original layout",
        "all_open": "All open",
        "all_closed": "All closed",
        "random": "Random",
        "rule_based": "Rule based",
    },


    # ------------------------------------------------------------------
    # Parallel evaluation settings
    # ------------------------------------------------------------------

    # Number of CPU workers used to evaluate simulations in parallel.
    "num_workers": 10,

    # Keep trajectory (.sqlite) files after evaluation.
    # False is recommended unless trajectories are needed for debugging.
    "keep_trajectories": False,

    # Shuffle the selected agent subset before each evaluation run.
    "shuffle_agents": True,

    # Maximum simulation iterations before stopping.
    "max_iterations": 1500,


    # ------------------------------------------------------------------
    # Simulation settings
    # ------------------------------------------------------------------

    "simulation": {

        # Simulation time step (seconds).
        "dt": 0.05,

        # Randomize the selected agent subset for each evaluation.
        "shuffle_agents_each_episode": True,

        # Save one trajectory frame every N simulation iterations.
        "every_nth_frame": 20,

        # Required for PedPy metric computation.
        "write_trajectory": True,

        # HTML animations are disabled during evaluation for speed.
        "save_animation": False,
    },

    # ------------------------------------------------------------------
    # RL policy input state
    # ------------------------------------------------------------------

    # If True, the policy starts from a random barrier configuration.
    # If False, the fixed barrier states below are used.
    "random_policy_initial_state": False,

    # Fixed initial barrier configuration used when
    # random_policy_initial_state is False.
    "policy_initial_barrier_states": {
        "pair_1": 1.0,
        "pair_2": 1.0,
        "pair_3": 1.0,
        "pair_4": 1.0,
        "pair_5": 1.0,
        "pair_6": 1.0,
        "pair_7": 1.0,
    },


    # ------------------------------------------------------------------
    # Fixed barrier baseline
    # ------------------------------------------------------------------

    # Barrier configuration used for methods that require a predefined
    # adaptive layout (e.g., creating the candidate-agent pool).
    #
    # 0.0 = fully closed
    # 1.0 = fully open
    # Values between 0.0 and 1.0 represent intermediate barrier states.
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