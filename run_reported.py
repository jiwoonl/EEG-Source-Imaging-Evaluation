"""Run an experiment with its reported settings."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def command(config, root):
    script = (root / config["script"]).resolve()
    if script.parent != (root / "experiments").resolve() or script.suffix != ".py":
        raise ValueError("Expected a Python experiment in experiments/.")
    if not script.is_file():
        raise FileNotFoundError(script)
    argv = [sys.executable, str(script)]
    for key, value in config["args"].items():
        if value is None or value is False:
            continue
        argv.append("--" + key.replace("_", "-"))
        if value is True:
            continue
        argv.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads(args.config.read_text())
    argv = command(config, root)
    print(shlex.join(argv), flush=True)
    if not args.dry_run:
        subprocess.run(argv, cwd=root, check=True)


if __name__ == "__main__":
    main()
