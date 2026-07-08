import os
import random
import numpy as np
from shapely import wkt

from config.scenario_config import SCENARIO
from utils.simulation_utils import (
    load_environment,
    create_simulation,
    run_simulation,
    save_animation,
)
from utils.RL_utils import load_agents, select_agent_subset, add_valid_agents_to_simulation
from barrier_control import clean_geom
from barrier_control import apply_barrier_pair_states

SEED = 101
OUTPUT_DIR = "../paper_figures"
TRAJECTORY_FILE = os.path.join(OUTPUT_DIR, "gt_original_one_run.sqlite")
HTML_FILE = os.path.join(OUTPUT_DIR, f"rl_agent_{SEED}_allclosed_1500.html")


NUM_AGENTS = 1500


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_original_gt_geometry(env):
    # This is the real GT/original simulation geometry from processed_environment.json
    if "simulation_geometry" in env["geometry_wkt"]:
        return clean_geom(wkt.loads(env["geometry_wkt"]["simulation_geometry"]))

    return clean_geom(wkt.loads(env["geometry_wkt"]["connected_geometry"]))

import pandas as pd

def get_states_from_csv(csv_file, method="rl_policy", seed=101, num_agents=300):
    df = pd.read_csv(csv_file)

    row = df[
        (df["method"] == method)
        & (df["seed"] == seed)
        & (df["num_agents_requested"] == num_agents)
    ]

    if row.empty:
        raise ValueError(
            f"No row found for method={method}, seed={seed}, num_agents={num_agents}"
        )

    row = row.iloc[0]

    return {
        "pair_1": float(row["pair_1"]),
        "pair_2": float(row["pair_2"]),
        "pair_3": float(row["pair_3"]),
        "pair_4": float(row["pair_4"]),
        "pair_5": float(row["pair_5"]),
        "pair_6": float(row["pair_6"]),
        "pair_7": float(row["pair_7"]),
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _, env = load_environment(SCENARIO["env_json"])
    data = "RL"  # "GT" or "RL"

    EVAL_CSV = "evaluation_all.csv"

    if data == "GT":
        geometry = get_original_gt_geometry(env)

    else:
        rl_pair_states = get_states_from_csv(
            EVAL_CSV,
            method="rl_policy",
            seed=SEED,
            num_agents=NUM_AGENTS,
        )

        print("Using RL pair states:")
        print(rl_pair_states)

        geometry = apply_barrier_pair_states(
            env,
            rl_pair_states,
            SCENARIO["barrier_pairs"],
            SCENARIO["barrier_pose_config"],
        )


    agent_groups = load_agents(geometry)

    selected_groups = select_agent_subset(
        agent_groups,
        max_agents=NUM_AGENTS,
        shuffle=True,
    )

    sim_param = dict(SCENARIO["simulation_mode_vis"])
    sim_param["write_trajectory"] = True
    sim_param["save_animation"] = True

    simulation = create_simulation(
        geometry,
        TRAJECTORY_FILE,
        sim_param,
    )

    added, skipped = add_valid_agents_to_simulation(
        simulation,
        selected_groups,
        geometry,
    )
    print("Added agents:", added)
    print("Skipped agents:", skipped)

    result = run_simulation(
        simulation,
        SCENARIO["max_iterations"],
    )

    print("Result:", result)

    save_animation(
        sim_param["every_nth_frame"],
        TRAJECTORY_FILE,
        HTML_FILE,
        title=f"{data} barrier configuration",
    )

    print("Saved HTML:", HTML_FILE)


if __name__ == "__main__":
    main()