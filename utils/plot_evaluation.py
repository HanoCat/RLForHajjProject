import os
import pandas as pd
import matplotlib.pyplot as plt


def make_evaluation_plots(csv_path="outputs_parallel/evaluation_all_2.csv"):
    df = pd.read_csv(csv_path)

    out_dir = "../logs/evaluation_plots"
    os.makedirs(out_dir, exist_ok=True)

    # Cleaner method names
    method_order = ["rl_policy", "gt_original", "all_open", "all_closed"]

    def save_bar(metric, ylabel, filename, title):
        summary = df.groupby("method")[metric].mean().reindex(method_order)

        plt.figure()
        summary.plot(kind="bar")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    # 1. Main performance plots
    save_bar(
        "raw_reward",
        "Raw reward",
        "mean_raw_reward_by_method.png",
        "Mean raw reward by method",
    )

    save_bar(
        "remaining_agents",
        "Remaining agents",
        "remaining_agents_by_method.png",
        "Mean remaining agents by method",
    )

    save_bar(
        "throughput_agents_per_second",
        "Agents / second",
        "throughput_by_method.png",
        "Mean throughput by method",
    )

    save_bar(
        "mean_speed",
        "Mean speed",
        "mean_speed_by_method.png",
        "Mean speed by method",
    )

    save_bar(
        "stopped_ratio",
        "Stopped ratio",
        "stopped_ratio_by_method.png",
        "Stopped ratio by method",
    )

    save_bar(
        "max_density",
        "Max density",
        "max_density_by_method.png",
        "Max density by method",
    )

    # 2. Performance vs crowd size
    for metric, ylabel, filename, title in [
        ("raw_reward", "Raw reward", "reward_vs_agents.png", "Reward vs crowd size"),
        ("remaining_agents", "Remaining agents", "remaining_vs_agents.png", "Remaining agents vs crowd size"),
        ("throughput_agents_per_second", "Agents / second", "throughput_vs_agents.png", "Throughput vs crowd size"),
        ("stopped_ratio", "Stopped ratio", "stopped_ratio_vs_agents.png", "Stopped ratio vs crowd size"),
    ]:
        pivot = df.groupby(["num_agents_requested", "method"])[metric].mean().reset_index()

        plt.figure()
        for method in method_order:
            sub = pivot[pivot["method"] == method]
            if not sub.empty:
                plt.plot(
                    sub["num_agents_requested"],
                    sub[metric],
                    marker="o",
                    label=method,
                )

        plt.xlabel("Requested number of agents")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    # 3. Density-throughput tradeoff
    plt.figure()
    for method in method_order:
        sub = df[df["method"] == method]
        if not sub.empty:
            plt.scatter(
                sub["max_density"],
                sub["throughput_agents_per_second"],
                label=method,
            )

    plt.xlabel("Max density")
    plt.ylabel("Throughput agents / second")
    plt.title("Density-throughput tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "density_throughput_tradeoff.png"))
    plt.close()

    # 4. RL learned barrier actions
    pair_cols = [c for c in df.columns if c.startswith("pair_")]
    rl_df = df[df["method"] == "rl_policy"]

    if not rl_df.empty and pair_cols:
        action_mean = rl_df[pair_cols].mean()

        plt.figure()
        action_mean.plot(kind="bar")
        plt.ylabel("Mean action value")
        plt.title("Mean learned RL barrier actions")
        plt.ylim(0, 1)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rl_mean_barrier_actions.png"))
        plt.close()

    print(f"Evaluation plots saved to: {out_dir}")


if __name__ == "__main__":
    make_evaluation_plots()