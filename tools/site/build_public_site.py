"""Stage only the static pages allowed on GitHub Pages."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PUBLIC_ENTRIES = (
    ".nojekyll",
    "index.html",
    "styles.css",
    "app.js",
    "dataset-page",
    "guide",
    "motionllm-page",
)


def build(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents:
        raise ValueError("output must be outside the source directory")
    if output.exists():
        raise FileExistsError(output)
    for name in PUBLIC_ENTRIES:
        item = source / name
        if not item.exists():
            raise FileNotFoundError(item)
        if item.is_symlink() or (
            item.is_dir() and any(child.is_symlink() for child in item.rglob("*"))
        ):
            raise ValueError(f"public entry contains a symlink: {name}")
    output.mkdir(parents=True)
    for name in PUBLIC_ENTRIES:
        item = source / name
        target = output / name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("online_page"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
