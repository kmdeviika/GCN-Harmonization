from __future__ import annotations

import argparse

from .harmonize import harmonize_from_config
from .train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manuscript-matched GCN EEG harmonization."
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)

    harmonize_parser = subparsers.add_parser("harmonize")
    harmonize_parser.add_argument("--config", required=True)

    args = parser.parse_args()

    if args.command == "train":
        result = train_from_config(args.config)
        print(f"Checkpoint saved to: {result}")
    else:
        result = harmonize_from_config(args.config)
        print(f"Harmonized EEG saved to: {result}")


if __name__ == "__main__":
    main()
