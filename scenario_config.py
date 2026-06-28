SCENARIO = {


    # simulation scenario settings
    "name": "four_zones_precise_test",
    "trajectory_file": "four_zones_test.sqlite",
    "html_file": "add_real_agents.html",
    "max_iterations": 100, # time stop for the simulation.

    "min_agent_distance": 0.4,
    "safe_distance": 0.2,
    "speed_min": 1.0,
    "speed_max": 1.4,

    # SAC train
    "num_episodes": 12,
    "num_steps": 1,
    "start_random_episodes": 2,
    "batch_size_rl": 4,
    "eval_freq_rl": 500,
    "save_every_episodes": 4,
    "best_reward_threshold": 0.85,


    "training": True,

    "simulation_mode_training": {
        "dt": 0.05,
        "training_num_agents": 100,   # use None for all agents
        "shuffle_agents_each_episode": True,
        "every_nth_frame": 20,


        "write_trajectory": False,
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
    "env_json": "processed_environment.json",

    ### Movable barrier ###
    "movable_barrier_ids": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 2301, 2302],

    "barrier_pairs": {
        "pair_1": (12, 13),
        "pair_2": (14, 15),
        "pair_3": (16, 17),
        "pair_4": (18, 19),
        "pair_5": (20, 22),
        "pair_6": (21, 24),
        "pair_7": (2301, 2302),
    },

    "barrier_pose_config": {
        12: {"closed": {"dx": -0.20, "dy": -0.30, "angle": -1}, "open": {"dx": 0.0, "dy": 0.30, "angle": -30}},
        13: {"closed": {"dx": -0.30, "dy": 0.90, "angle": 10}, "open": {"dx": 0.0, "dy": -0.35, "angle": 30}},

        14: {"closed": {"dx": 0.35, "dy": -0.60, "angle": 1}, "open": {"dx": 0.0, "dy": 0.0, "angle": -25}},
        15: {"closed": {"dx": -0.30, "dy": 0.70, "angle": -1}, "open": {"dx": 0.0, "dy": 0.0, "angle": 30}},

        16: {"closed": {"dx": 0.0, "dy": -0.75, "angle": 15}, "open": {"dx": -0.25, "dy": 0.15, "angle": -30}},
        17: {"closed": {"dx": 0.0, "dy": 0.75, "angle": -10}, "open": {"dx": 0.30, "dy": -0.40, "angle": 35}},

        18: {"closed": {"dx": 0.0, "dy": 0.20, "angle": 0}, "open": {"dx": -0.20, "dy": -0.25, "angle": -50}},
        19: {"closed": {"dx": 1.50, "dy": -0.60, "angle": -10}, "open": {"dx": 0.25, "dy": 0.20, "angle": 30}},

        20: {"closed": {"dx": 0.0, "dy": 0.25, "angle": 0}, "open": {"dx": -0.30, "dy": 0.20, "angle": 30}},
        22: {"closed": {"dx": 0.0, "dy": -0.25, "angle": 0}, "open": {"dx": -0.20, "dy": 0.0, "angle": -30}},

        21: {"closed": {"dx": 0.0, "dy": -0.55, "angle": 0}, "open": {"dx": 0.25, "dy": 0.10, "angle": -25}},
        24: {"closed": {"dx": 0.0, "dy": 0.90, "angle": -10}, "open": {"dx": -0.25, "dy": -0.25, "angle": 30}},

        2301: {"closed": {"dx": 0.0, "dy": -0.80, "angle": 15}, "open": {"dx": -0.25, "dy": -0.15, "angle": -30}},
        2302: {"closed": {"dx": 0.0, "dy": 0.90, "angle": -20}, "open": {"dx": 0.25, "dy": -0.10, "angle": 30}},
    },

    "barrier_pair_states": {
    "pair_1": 1.0,
    "pair_2": 1.0,
    "pair_3": 1.0,
    "pair_4": 1.0,
    "pair_5": 1.0,
    "pair_6": 1.0,
    "pair_7": 1.0,
    },


    ### Agents Initialization ###
    # load agents from the head detection of the scene
    "p2pnet_points_file": "p2pnet_points.json",
    "p2pnet_min_score": 0.5,
    "p2pnet_max_agents": None,

    # add more synthetic agents randomly based on the different zones in the scene
    "agent_groups": [
        {
            "group_id": "ZONE1_1",
            "count": 5,
            "start_box_frac": (0.86, 0.82, 0.98, 0.98),
            "mid_box_frac":   (0.76, 0.82, 0.85, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {
            "group_id": "ZONE1_2",
            "count": 5,
            "start_box_frac": (0.76, 0.82, 0.85, 0.98),
            "mid_box_frac": (0.63, 0.82, 0.73, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {
            "group_id": "ZONE1_3",
            "count": 5,
            "start_box_frac": (0.63, 0.82, 0.73, 0.98),
            "mid_box_frac": (0.56, 0.82, 0.63, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {
            "group_id": "ZONE1_4",
            "count": 5,
            "start_box_frac": (0.56, 0.82, 0.63, 0.98),
            "mid_box_frac": (0.30, 0.82, 0.53, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {
            "group_id": "ZONE1_5",
            "count": 5,
            "start_box_frac": (0.30, 0.82, 0.53, 0.98),
            "mid_box_frac": (0.09, 0.82, 0.23, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {## final zone 1
            "group_id": "ZONE1_6",
            "count": 5,
            "start_box_frac": (0.09, 0.82, 0.23, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {
            "group_id": "ZONE2_1",
            "count": 5,
            "start_box_frac": (0.65, 0.26, 0.99, 0.42),
            "mid_box_frac": (0.60, 0.26, 0.65, 0.42),
            "goal_box_frac":   (0.0, 0.30, 0.04, 0.55),
        },
        {
            "group_id": "ZONE2_2",
            "count": 5,
            "start_box_frac": (0.60, 0.26, 0.65, 0.42),
            "mid_box_frac": (0.32, 0.26, 0.58, 0.50),
            "goal_box_frac":   (0.0, 0.30, 0.04, 0.55),
        },
        {
            "group_id": "ZONE2_3",
            "count": 5,
            "start_box_frac": (0.32, 0.26, 0.58, 0.50),
            "mid_box_frac": (0.25, 0.30, 0.32, 0.50),
            "goal_box_frac":   (0.0, 0.30, 0.04, 0.55),
        },
        {
            "group_id": "ZONE2_4",
            "count": 5,
            "start_box_frac": (0.25, 0.30, 0.32, 0.50),
            "mid_box_frac": (0.09, 0.35, 0.23, 0.50),
            "goal_box_frac":   (0.0, 0.30, 0.04, 0.55),
        },
        {## final zone 2
            "group_id": "ZONE2_5",
            "count": 5,
            "start_box_frac": (0.09, 0.35, 0.23, 0.50),
            "goal_box_frac":   (0.0, 0.30, 0.04, 0.55),
        },
        {
            "group_id": "ZONE3_1",
            "count": 5,
            "start_box_frac": (0.60, 0.00, 0.99, 0.14),
            "mid_box_frac": (0.55, 0.00, 0.60, 0.14),
            "goal_box_frac":  (0.00, 0.24, 0.07, 0.42),
        },
        {
            "group_id": "ZONE3_2",
            "count": 5,
            "start_box_frac": (0.55, 0.00, 0.60, 0.14),
            "mid_box_frac": (0.30, 0.00, 0.52, 0.25),
            "goal_box_frac":  (0.00, 0.24, 0.07, 0.42),
        },
        {
            "group_id": "ZONE3_3",
            "count": 5,
            "start_box_frac": (0.30, 0.00, 0.52, 0.25),
            "mid_box_frac": (0.22, 0.00, 0.30, 0.27),
            "goal_box_frac":  (0.00, 0.24, 0.07, 0.42),
        },
        {
            "group_id": "ZONE3_4",
            "count": 5,
            "start_box_frac": (0.22, 0.00, 0.30, 0.27),
            "mid_box_frac": (0.09, 0.00, 0.19, 0.30),
            "goal_box_frac":  (0.00, 0.24, 0.07, 0.42),
        },
        { ## final zone 3
            "group_id": "ZONE3_5",
            "count": 5,
            "start_box_frac": (0.09, 0.00, 0.19, 0.30),
            "goal_box_frac":  (0.00, 0.24, 0.07, 0.42),
        },
    ],


}
