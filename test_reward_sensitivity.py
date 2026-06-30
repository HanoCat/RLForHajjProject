import os
import random
import numpy as np
import pandas as pd

from scenario_config_orginal import SCENARIO
from simulation_utils import *
from RL_utils import *


def make_barrier_test_cases():
    pair_names = list(SCENARIO["barrier_pairs"].keys())

    cases = []

    cases.append({
        "name": "all_open",
        "states": {p: 1.0 for p in pair_names},
    })

    cases.append({
        "name": "all_closed",
        "states": {p: 0.0 for p in pair_names},
    })

    cases.append({
        "name": "half_open",
        "states": {p: 0.5 for p in pair_names},
    })

    cases.append({
        "name": "alternating_1",
        "states": {
            p: 1.0 if i % 2 == 0 else 0.0
            for i, p in enumerate(pair_names)
        },
    })

    cases.append({
        "name": "alternating_2",
        "states": {
            p: 0.0 if i % 2 == 0 else 1.0
            for i, p in enumerate(pair_names)
        },
    })

    for k in range(5):
        cases.append({
            "name": f"random_{k + 1}",
            "states": {
                p: random.uniform(0.0, 1.0)
                for p in pair_names
            },
        })

    return cases


def run_single_reward_test(
        env,
        fixed_agent_groups,
        barrier_pair_states,
        case_name,
        test_id,
):
    geometry = create_geometry(
        env,
        barrier_pair_states,
    )

    os.makedirs("reward_sensitivity_results", exist_ok=True)

    trajectory_file = os.path.join(
        "reward_sensitivity_results",
        f"{case_name}.sqlite",
    )

    simulation_pram = {
        "dt": SCENARIO["simulation_mode_training"]["dt"],
        "write_trajectory": True,
        "every_nth_frame": SCENARIO["simulation_mode_training"].get(
            "every_nth_frame",
            10,
        ),
    }

    simulation = create_simulation(
        geometry,
        trajectory_file,
        simulation_pram,
    )

    total_added, total_skipped = add_valid_agents_to_simulation(
        simulation,
        fixed_agent_groups,
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
        debug=True,
    )

    row = {
        "test_id": test_id,
        "case_name": case_name,
        "reward": reward,
        "added_agents": total_added,
        "skipped_agents": total_skipped,
        "remaining_agents": result["remaining_agents"],
        "iterations": result["iterations"],
        "elapsed_time": result["elapsed_time"],
    }

    for key, value in reward_metrics.items():
        row[key] = value

    for pair_name, value in barrier_pair_states.items():
        row[pair_name] = value

    print("\n" + "=" * 70)
    print(f"Case: {case_name}")
    print(f"Reward: {reward:.4f}")
    print(f"Added agents: {total_added}, skipped: {total_skipped}")
    print("Barrier states:", {
        k: round(v, 2)
        for k, v in barrier_pair_states.items()
    })

    return row


def main():
    random.seed(42)
    np.random.seed(42)

    env_json = SCENARIO["env_json"]
    _, env = load_environment(env_json)

    print(f"Environment loaded: {env_json}")

    base_geometry = create_geometry(
        env,
        SCENARIO["barrier_pair_states"],
    )

    agent_groups = load_agents(base_geometry)

    agent_counts = SCENARIO.get(
        "reward_sensitivity_agent_counts",
        [10, 50, 100, 200, 800],
    )

    test_cases = make_barrier_test_cases()

    rows = []

    for num_agents in agent_counts:
        fixed_agent_groups = select_agent_subset(
            agent_groups,
            max_agents=num_agents,
            shuffle=True,
        )

        print("\n" + "#" * 80)
        print(f"Testing fixed agent subset: {num_agents}")
        print("#" * 80)

        for test_id, case in enumerate(test_cases, start=1):
            row = run_single_reward_test(
                env=env,
                fixed_agent_groups=fixed_agent_groups,
                barrier_pair_states=case["states"],
                case_name=f"{num_agents}_{case['name']}",
                test_id=test_id,
            )

            row["num_agents_requested"] = num_agents
            rows.append(row)

    df = pd.DataFrame(rows)

    output_csv = os.path.join(
        "reward_sensitivity_results",
        "reward_sensitivity_summary_multi_agents.csv",
    )

    df.to_csv(output_csv, index=False)

    print("\nSummary:")
    print(df.sort_values(["num_agents_requested", "reward"], ascending=[True, False]))
    print(f"\nSaved to: {output_csv}")

if __name__ == "__main__":
    main()