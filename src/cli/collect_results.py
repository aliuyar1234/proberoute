from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.assemble_registry import assemble_registries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", required=True)
    args = parser.parse_args()
    paths = assemble_registries(Path(args.outputs_root).resolve())
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()

