from __future__ import annotations

import ast
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[2]
ACTIVE_PYTHON_ROOTS = (
    "src",
    "qwenvl",
    "models",
    "rubric_rl",
    "codex_remote_tools",
    "tools",
    "model_evaluation_agent/scripts",
)
PERSONAL_PATH_PATTERN = re.compile(
    r"/(?:home|wangbenyou-[^/]+)/|[A-Za-z]:\\Users\\",
    re.IGNORECASE,
)


def active_python_files() -> list[Path]:
    files: list[Path] = []
    for root_name in ACTIVE_PYTHON_ROOTS:
        root = REPO / root_name
        if root.exists():
            files.extend(root.rglob("*.py"))
    return sorted(files)


def test_active_python_never_imports_legacy_tree() -> None:
    violations: list[str] = []
    for path in active_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "legacy" or name.startswith("legacy.") for name in names):
                violations.append(str(path.relative_to(REPO)))
    assert not violations, f"active modules import legacy code: {violations}"


def test_active_python_has_no_personal_absolute_paths() -> None:
    violations: list[str] = []
    for path in active_python_files():
        text = path.read_text(encoding="utf-8")
        if PERSONAL_PATH_PATTERN.search(text):
            violations.append(str(path.relative_to(REPO)))
    assert not violations, f"personal absolute paths remain in active code: {violations}"


def test_active_configs_have_no_personal_absolute_paths() -> None:
    violations: list[str] = []
    config_root = REPO / "configs"
    for suffix in ("*.json", "*.yaml", "*.yml", "*.toml"):
        for path in config_root.rglob(suffix):
            if PERSONAL_PATH_PATTERN.search(path.read_text(encoding="utf-8")):
                violations.append(str(path.relative_to(REPO)))
    assert not violations, f"personal absolute paths remain in active configs: {violations}"


def test_legacy_is_documented_as_non_importable() -> None:
    readme = (REPO / "legacy" / "README.md").read_text(encoding="utf-8")
    assert "不得 import" in readme
    assert (REPO / "legacy" / "qwen_vl_original" / "SOURCE_MANIFEST.md").is_file()
