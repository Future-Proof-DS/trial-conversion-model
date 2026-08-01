import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="beam-trial")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("train", help="Train the model and save it to models/")
    args = parser.parse_args()

    if args.command == "train":
        from beam_trial.train import train

        metrics = train()
        print(metrics)
