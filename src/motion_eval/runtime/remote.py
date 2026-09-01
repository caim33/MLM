"""Pinned-host-key OpenSSH command construction; no trust-on-first-use path."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .process import CommandSpec

_HOST = re.compile(r"\A(?:[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?|\[[0-9A-Fa-f:]+\])\Z")
_USER = re.compile(r"\A[A-Za-z0-9._-]{1,64}\Z")


@dataclass(frozen=True)
class PinnedSshTarget:
    host: str
    port: int
    user: str
    known_hosts_file: str

    def __post_init__(self) -> None:
        if not _HOST.fullmatch(self.host):
            raise ValueError("invalid SSH host")
        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ValueError("SSH port must be 1-65535")
        if not _USER.fullmatch(self.user):
            raise ValueError("invalid SSH user")
        known = Path(self.known_hosts_file).resolve(strict=True)
        if not known.is_file() or known.stat().st_size == 0:
            raise ValueError("a non-empty pinned known_hosts file is required")
        object.__setattr__(self, "known_hosts_file", str(known))

    def command(self, remote_argv: Sequence[str], *, timeout_seconds: float = 60) -> CommandSpec:
        if not remote_argv or any(
            not isinstance(item, str) or not item or any(ord(char) < 32 for char in item)
            for item in remote_argv
        ):
            raise ValueError("remote argv must be non-empty and control-free")
        # OpenSSH invokes a remote shell. shlex.join preserves argv boundaries;
        # callers still must select an approved executable rather than user text.
        remote_command = shlex.join(list(remote_argv))
        argv = (
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self.known_hosts_file}",
            "-o", "PasswordAuthentication=no",
            "-p", str(self.port),
            "--", f"{self.user}@{self.host}", remote_command,
        )
        return CommandSpec(
            argv=argv,
            cwd=str(Path.cwd()),
            timeout_seconds=timeout_seconds,
            label="pinned-ssh",
        )
