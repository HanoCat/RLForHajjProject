import argparse

from runners.run_scenario import run_scenario
from runners.train_sequential import train_sequential
from runners.train_parallel import train_parallel
from runners.evaluate import evaluate


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

    if args.mode == "scenario":
        run_scenario()
    elif args.mode == "train-seq":
        train_sequential()
    elif args.mode == "train-parallel":
        train_parallel()
    elif args.mode == "evaluate":
        evaluate()


if __name__ == "__main__":
    main()