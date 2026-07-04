






try:
    from plot_evaluation import make_evaluation_plots

    make_evaluation_plots("outputs_parallel/evaluation_all_methods.csv")
except Exception as e:
    print(f"Plotting skipped: {e}")