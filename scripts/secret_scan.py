#!/usr/bin/env python3
"""Fail on high-confidence credentials in the active source tree.

The scanner deliberately reports only file and line number.  It never echoes
the matching line, which keeps CI logs safe even when a finding is real.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_NAMES = {
    ".codex_ssh_key",
    "dev_env_connection.txt",
    "id_rsa",
    "id_ed25519",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

TEXT_SUFFIXES = {
    ".py",
    ".pyi",
    ".sh",
    ".ps1",
    ".mjs",
    ".js",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
}

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("wandb-token", re.compile(r"\bwandb_v1_[A-Za-z0-9_-]{24,}\b")),
    ("sshpass", re.compile(r"\bsshpass\s+-p\s+\S+", re.IGNORECASE)),
    (
        "literal-secret",
        re.compile(
            r"\b(?:password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key)\b"
            r"\s*(?:=|:)\s*[\"'][^\"'\n]{8,}[\"']",
            re.IGNORECASE,
        ),
    ),
)

SAFE_MARKERS = (
    "<redacted>",
    "replace-me",
    "set-in-environment",
    "process environment",
    "${",
)


@dataclass(frozen=True)
class Finding:
    kind: str
    path: Path
    line_number: int


def iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        yield path


def scan(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_files(root):
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in {".pem", ".key"}:
            findings.append(Finding("forbidden-file", path, 0))
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            findings.append(Finding("unreadable-file", path, 0))
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if any(marker in lowered for marker in SAFE_MARKERS):
                continue
            for kind, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(Finding(kind, path, number))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = Path(args.root).resolve()
    findings = scan(root)
    for finding in findings:
        relative = finding.path.resolve().relative_to(root)
        suffix = f":{finding.line_number}" if finding.line_number else ""
        print(f"{finding.kind}: {relative}{suffix}")
    if findings:
        print(f"secret scan failed: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print("secret scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

