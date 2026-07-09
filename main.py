import argparse
from pathlib import Path

from runners.run_scenario import run_scenario
from runners.train_sequential import train_sequential
from runners.train_parallel import train_parallel
from runners.evaluate import evaluate

from pathlib import Path


def prepare_log_dir(mode):
    root_log_dir = Path("logs")
    root_log_dir.mkdir(exist_ok=True)

    log_dirs = {
        "scenario": root_log_dir / "scenario",
        "train-seq": root_log_dir / "training_sequential",
        "train-parallel": root_log_dir / "training_parallel",
        "evaluate": root_log_dir / "evaluation",
    }

    log_dir = log_dirs[mode]
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

def main():
    parser = argparse.ArgumentParser(
        description="RLForHajj: adaptive crowd barrier simulation, training, and evaluation"
    )

    parser.add_argument(
        "mode",
        choices=["scenario", "train-seq", "train-parallel", "evaluate"],
        help="Choose what to run.",
    )

    args = parser.parse_args()

    log_dir = prepare_log_dir(args.mode)

    if args.mode == "scenario":
        run_scenario(log_dir)
    elif args.mode == "train-seq":
        train_sequential(log_dir)
    elif args.mode == "train-parallel":
        train_parallel(log_dir)
    elif args.mode == "evaluate":
        evaluate(log_dir)


if __name__ == "__main__":
    main()