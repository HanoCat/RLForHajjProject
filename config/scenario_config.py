from config.base_config import CONFIG


TRAINING_CONFIG = {
    **CONFIG,

    # ------------------------------------------------------------------
    # Scenario output settings
    # ------------------------------------------------------------------

    # A descriptive name for this scenario.
    # This name may appear in plots, logs, and generated HTML files.
    "name": "agents_initial_positions",

    # SQLite file used to store the agents' trajectories during simulation.
    "trajectory_file": "../logs/scenario/scenario.sqlite",

    # Directory where all scenario outputs will be saved.
    "output_dir": "../logs/scenario/",

    # HTML file containing the generated crowd animation and results table.
    "html_file": "../logs/scenario/org.html",

    # Maximum number of simulation iterations.
    # The simulation stops when either:
    # 1. all agents finish, or
    # 2. this maximum number of iterations is reached.
    "max_iterations": 1500,





    # ------------------------------------------------------------------
    # Simulation engine settings
    # ------------------------------------------------------------------

    "simulation": {
        # Simulation time step in seconds.
        # Smaller values may improve numerical accuracy but increase runtime.
        "dt": 0.05,

        # Save one trajectory frame every N simulation iterations.
        # For example, 5 means every fifth frame is stored.
        "every_nth_frame": 5,

        # Save the simulation trajectory to the SQLite file.
        # This must usually remain True if you want metrics or animations.
        "write_trajectory": True,

        # Generate and save an HTML animation after the simulation.
        "save_animation": True,
    },


    # ------------------------------------------------------------------
    # Agent placement and movement settings
    # ------------------------------------------------------------------

    # Minimum allowed distance between added agents.
    # Increasing this value spreads agents farther apart.
    "min_agent_distance": 0.4,

    # Safety margin used when creating start zones, goal zones,
    # and other areas close to environment boundaries.
    "safe_distance": 0.2,

    # Minimum desired walking speed assigned to an agent.
    "speed_min": 1.0,

    # Maximum desired walking speed assigned to an agent.
    # Each agent receives a speed sampled between speed_min and speed_max.
    "speed_max": 1.4,


    # ------------------------------------------------------------------
    # Barrier configuration
    # ------------------------------------------------------------------

    # Continuous state of each barrier pair.
    #
    # 0.0 = fully closed
    # 1.0 = fully open
    # Values between 0.0 and 1.0 create intermediate barrier poses.
    #
    # The exact movement of each pair is defined in barrier_pose_config.
    "barrier_pair_states": {
        "pair_1": 1.0,
        "pair_2": 1.0,
        "pair_3": 0.0,
        "pair_4": 1.0,
        "pair_5": 1.0,
        "pair_6": 0.0,
        "pair_7": 0.0,
    },
}