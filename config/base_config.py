

CONFIG = {
    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    # Processed environment file used to build the JuPedSim geometry.
    #
    # This JSON should contain:
    # - the walkable environment geometry,
    # - barrier polygons,
    # - barrier IDs,
    # - any other geometry information required by the simulation.
    "env_json": "./json/processed_environment.json",


    # ------------------------------------------------------------------
    # P2PNet agent detections
    # ------------------------------------------------------------------

    # JSON file containing pedestrian positions detected from the
    # original crowd image using P2PNet.
    "p2pnet_points_file": "./json/p2pnet_points.json",

    # Minimum P2PNet confidence score required to keep a detection.
    #
    # Higher values keep only more confident detections.
    "p2pnet_min_score": 0.5,

    # Maximum number of P2PNet-detected agents to load.
    #
    # Use None to keep all valid detections.
    "p2pnet_max_agents": None,


    # ------------------------------------------------------------------
    # Movable barriers
    # ------------------------------------------------------------------

    # IDs of all barriers that can move during scenario execution,
    # training, and evaluation.
    "movable_barrier_ids": [
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        24,
        2301,
        2302,
    ],

    # Barriers are controlled in pairs.
    #
    # Each pair is represented by one continuous state:
    #
    # 0.0 = fully closed
    # 1.0 = fully open
    #
    # Intermediate values interpolate between the closed and open poses.
    "barrier_pairs": {
        "pair_1": (12, 13),
        "pair_2": (14, 15),
        "pair_3": (16, 17),
        "pair_4": (18, 19),
        "pair_5": (20, 22),
        "pair_6": (21, 24),
        "pair_7": (2301, 2302),
    },

    # ------------------------------------------------------------------
    # Barrier poses
    # ------------------------------------------------------------------

    # Closed and open poses for every movable barrier.
    #
    # dx:
    #     Horizontal translation relative to the barrier's original
    #     position.
    #
    # dy:
    #     Vertical translation relative to the barrier's original
    #     position.
    #
    # angle:
    #     Rotation angle in degrees relative to the original pose.
    #
    # Continuous barrier states between 0.0 and 1.0 are created by
    # interpolating between these closed and open values.
    #
    # These values are specific to the current environment and were chosen
    # to keep barriers inside the valid walkable geometry.
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


    # ------------------------------------------------------------------
    # Agent initialization
    # ------------------------------------------------------------------

    # At least one source must be enabled.
    #
    # Both may be True to combine random and P2PNet positions.
    "p2pnet_load": False,
    "random_agents_load": True,

    # ------------------------------------------------------------------
    # Agent groups and movement zones
    # ------------------------------------------------------------------

    # Each group defines where agents start and where they move.
    #
    # group_id:
    #     Unique readable name for the group.
    #
    # count:
    #     Number of randomly generated agents for this group.
    #     This value is used only when random_agents_load is True.
    #
    # start_box_frac:
    #     Fractional bounding box used to create the start zone.
    #
    # mid_box_frac:
    #     Optional intermediate movement zone.
    #     Some helper functions may use this to represent the next section
    #     of a pedestrian route.
    #
    # goal_box_frac:
    #     Fractional bounding box used to create the final goal zone.
    #
    # Fractional box format:
    #
    #     (xmin_fraction, ymin_fraction, xmax_fraction, ymax_fraction)
    #
    # Each value must normally be between 0.0 and 1.0 and is interpreted
    # relative to the full environment bounds.

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
            "count": 1,
            "start_box_frac": (0.76, 0.82, 0.85, 0.98),
            "mid_box_frac": (0.63, 0.82, 0.73, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {
            "group_id": "ZONE1_3",
            "count": 1,
            "start_box_frac": (0.63, 0.82, 0.73, 0.98),
            "mid_box_frac": (0.56, 0.82, 0.63, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {
            "group_id": "ZONE1_4",
            "count": 1,
            "start_box_frac": (0.56, 0.82, 0.63, 0.98),
            "mid_box_frac": (0.30, 0.82, 0.53, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {
            "group_id": "ZONE1_5",
            "count": 1,
            "start_box_frac": (0.30, 0.82, 0.53, 0.98),
            "mid_box_frac": (0.09, 0.82, 0.23, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {## final zone 1
            "group_id": "ZONE1_6",
            "count": 1,
            "start_box_frac": (0.09, 0.82, 0.23, 0.98),
            "goal_box_frac":  (0.00, 0.78, 0.04, 1.0),
        },
        {
            "group_id": "ZONE2_1",
            "count": 1,
            "start_box_frac": (0.65, 0.26, 0.99, 0.42),
            "mid_box_frac": (0.60, 0.26, 0.65, 0.42),
            "goal_box_frac":   (0.0, 0.30, 0.04, 0.55),
        },
        {
            "group_id": "ZONE2_2",
            "count": 1,
            "start_box_frac": (0.60, 0.26, 0.65, 0.42),
            "mid_box_frac": (0.32, 0.26, 0.58, 0.50),
            "goal_box_frac":   (0.0, 0.30, 0.04, 0.55),
        },
        {
            "group_id": "ZONE2_3",
            "count": 1,
            "start_box_frac": (0.32, 0.26, 0.58, 0.50),
            "mid_box_frac": (0.25, 0.30, 0.32, 0.50),
            "goal_box_frac":   (0.0, 0.30, 0.04, 0.55),
        },
        {
            "group_id": "ZONE2_4",
            "count": 1,
            "start_box_frac": (0.25, 0.30, 0.32, 0.50),
            "mid_box_frac": (0.09, 0.35, 0.23, 0.50),
            "goal_box_frac":   (0.0, 0.30, 0.04, 0.55),
        },
        {## final zone 2
            "group_id": "ZONE2_5",
            "count": 1,
            "start_box_frac": (0.09, 0.35, 0.23, 0.50),
            "goal_box_frac":   (0.0, 0.30, 0.04, 0.55),
        },
        {
            "group_id": "ZONE3_1",
            "count": 1,
            "start_box_frac": (0.60, 0.00, 0.99, 0.14),
            "mid_box_frac": (0.55, 0.00, 0.60, 0.14),
            "goal_box_frac":  (0.00, 0.24, 0.07, 0.42),
        },
        {
            "group_id": "ZONE3_2",
            "count": 1,
            "start_box_frac": (0.55, 0.00, 0.60, 0.14),
            "mid_box_frac": (0.30, 0.00, 0.52, 0.25),
            "goal_box_frac":  (0.00, 0.24, 0.07, 0.42),
        },
        {
            "group_id": "ZONE3_3",
            "count": 1,
            "start_box_frac": (0.30, 0.00, 0.52, 0.25),
            "mid_box_frac": (0.22, 0.00, 0.30, 0.27),
            "goal_box_frac":  (0.00, 0.24, 0.07, 0.42),
        },
        {
            "group_id": "ZONE3_4",
            "count": 1,
            "start_box_frac": (0.22, 0.00, 0.30, 0.27),
            "mid_box_frac": (0.09, 0.00, 0.19, 0.30),
            "goal_box_frac":  (0.00, 0.24, 0.07, 0.42),
        },
        { ## final zone 3
            "group_id": "ZONE3_5",
            "count": 1,
            "start_box_frac": (0.09, 0.00, 0.19, 0.30),
            "goal_box_frac":  (0.00, 0.24, 0.07, 0.42),
        },
    ],

}