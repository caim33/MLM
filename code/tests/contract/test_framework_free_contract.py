from __future__ import annotations

import ast
from pathlib import Path


def test_contracts_and_data_do_not_import_model_frameworks() -> None:
    source_root = Path(__file__).parents[2] / "src" / "motionllm"
    forbidden = ("torch", "transformers", "swift")
    checked = 0
    for package in (source_root / "contracts", source_root / "data"):
        for source in package.glob("*.py"):
            checked += 1
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    names = [node.module]
                assert not any(
                    name == prefix or name.startswith(prefix + ".")
                    for name in names
                    for prefix in forbidden
                ), f"{source} imports a forbidden model framework"
    assert checked > 0
