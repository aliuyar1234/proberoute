from __future__ import annotations

import argparse

from src.core.config import load_config, validate_config
from src.train.probe_trainer import train_probe_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    run_dir = train_probe_run(config, dry_run=args.dry_run, force=args.force, resume=args.resume)
    print(run_dir)


if __name__ == "__main__":
    main()
