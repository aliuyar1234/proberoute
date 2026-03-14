from __future__ import annotations

import argparse
from pathlib import Path

from src.train.checkpointing import request_pause


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    path = request_pause(Path(args.run_dir).resolve())
    print(path)


if __name__ == "__main__":
    main()
