#!/usr/bin/env python3
"""Run the local, framework-independent refactor checks."""

from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    compile_roots = [root / "src", root / "models", root / "qwenvl"]
    failed = [path for path in compile_roots if path.exists() and not compileall.compile_dir(path, quiet=1)]
    if failed:
        for path in failed:
            print(f"compile failed: {path}", file=sys.stderr)
        return 1

    run([sys.executable, str(root / "scripts" / "secret_scan.py"), str(root)], root)
    if not args.no_tests:
        run([sys.executable, "-m", "pytest", "-q"], root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

