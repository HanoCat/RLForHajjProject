import pandas as pd
import os

files = [
    "outputs_parallel/evaluation_results_parallel.csv",
    "outputs_parallel/evaluation_gt_rule_parallel.csv",
    "outputs_parallel/evaluation_results_parallel_unseen_size.csv",
    "outputs_parallel/evaluation_gt_rule_parallel_unseen_size.csv",
]

dfs = []

for f in files:
    if os.path.exists(f):
        print("Loading:", f)
        dfs.append(pd.read_csv(f))
    else:
        print("Missing:", f)

df = pd.concat(dfs, ignore_index=True)

df = df.sort_values(
    ["num_agents_requested", "seed", "method"]
)

merged_csv = "outputs_parallel/evaluation_all.csv"

df.to_csv(
    merged_csv,
    index=False,
)

print("\nSaved merged file:")
print(merged_csv)

print("\nMethods:")
print(df["method"].unique())

print("\nCrowd sizes:")
print(sorted(df["num_agents_requested"].unique()))


# ---------------------------------------------
# Automatically generate plots
# ---------------------------------------------
try:
    from plot_evaluation import make_evaluation_plots

    print("\nGenerating plots...")
    make_evaluation_plots(merged_csv)

    print("Plots generated successfully.")

except Exception as e:
    print(f"Plot generation failed: {e}")