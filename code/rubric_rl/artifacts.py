"""Crash-safe JSONL artifact writing for Rubric-RL command-line tools.

The final JSONL is replaced only after every row has been written and
validated.  Interrupted runs leave an append-only ``.partial`` work file that
can be resumed without modifying the last committed artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class ArtifactError(ValueError):
    """Raised when an artifact cannot be trusted or resumed safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_contract(value: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return an immutable canonical encoding and a detached JSON object."""

    if not isinstance(value, Mapping) or not value:
        raise ArtifactError("run_contract must be a non-empty mapping")
    try:
        encoded = _json_dumps(dict(value))
    except (TypeError, ValueError) as exc:
        raise ArtifactError("run_contract must contain only finite JSON values") from exc
    decoded = _decode_object(encoded, location="run_contract")
    return encoded, decoded


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_json_dumps(dict(value)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read artifact metadata: {path}") from exc
    return _decode_object(raw, location=str(path))


def _source_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ArtifactError(f"artifact source does not exist: {path}") from exc
        if not resolved.is_file():
            raise ArtifactError(f"artifact source is not a file: {resolved}")
        resolved_text = str(resolved)
        if resolved_text in seen:
            raise ArtifactError(f"duplicate artifact source path: {resolved}")
        seen.add(resolved_text)
        records.append(
            {
                "path": resolved_text,
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return records


def freeze_source_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Capture the exact ordered source bytes a producer intends to use.

    Pass the returned records back to :class:`AtomicJsonlArtifact` through
    ``expected_source_records``.  The artifact checks them before any work and
    again at commit, so a prompt/code/data change during model startup cannot
    be silently recorded as different bytes from those selected by the run
    contract.
    """

    return [dict(record) for record in _source_records(paths)]


def _validate_expected_source_records(
    value: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        record = dict(raw)
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise ArtifactError(
                f"expected_source_records[{index}] has an invalid schema"
            )
        path = record["path"]
        digest = record["sha256"]
        size = record["size_bytes"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ArtifactError(
                f"expected_source_records[{index}].path must be absolute"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ArtifactError(
                f"expected_source_records[{index}].sha256 must be lowercase SHA-256"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ArtifactError(
                f"expected_source_records[{index}].size_bytes must be non-negative"
            )
        records.append({"path": path, "sha256": digest, "size_bytes": size})
    if len({record["path"] for record in records}) != len(records):
        raise ArtifactError("expected_source_records contains duplicate paths")
    return records


def _identifier(row: Mapping[str, Any], id_key: str) -> str:
    value = row.get(id_key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ArtifactError(f"{id_key} must be a non-empty, trimmed string")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ArtifactError(f"{id_key} contains a control newline")
    return value


def _decode_object(raw: str | bytes, *, location: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ArtifactError(f"duplicate JSON key {key!r} at {location}")
            output[key] = value
        return output

    def reject_constant(value: str) -> Any:
        raise ArtifactError(f"non-finite JSON constant {value!r} at {location}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"invalid JSON at {location}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSONL row must be an object at {location}")
    return value


def _load_complete_rows(path: Path, *, id_key: str) -> tuple[list[dict[str, Any]], bool]:
    """Read valid newline-terminated rows.

    A single unterminated tail is the only corruption accepted from a crashed
    append.  It is ignored and rewritten when the session starts.  Invalid
    complete lines fail closed.
    """

    if not path.exists():
        return [], False
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    dropped_tail = bool(lines and not lines[-1].endswith((b"\n", b"\r")))
    if dropped_tail:
        lines = lines[:-1]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            raise ArtifactError(f"blank JSONL line at {path}:{line_number}")
        value = _decode_object(raw, location=f"{path}:{line_number}")
        sid = _identifier(value, id_key)
        if sid in seen:
            raise ArtifactError(f"duplicate {id_key} {sid!r} in {path}")
        seen.add(sid)
        rows.append(value)
    return rows, dropped_tail


def load_jsonl_strict(path: Path, *, id_key: str = "sample_id") -> Iterator[dict[str, Any]]:
    rows, dropped_tail = _load_complete_rows(path, id_key=id_key)
    if dropped_tail:
        raise ArtifactError(f"unterminated JSONL tail in committed artifact: {path}")
    yield from rows


def iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    """Read object-only JSONL without silently skipping blank or scalar rows."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.rstrip("\r\n")
            if not text.strip():
                raise ArtifactError(f"blank JSONL line at {path}:{line_number}")
            yield _decode_object(text, location=f"{path}:{line_number}")


class AtomicJsonlArtifact:
    """Append rows to a recoverable work file and atomically publish it."""

    def __init__(
        self,
        output: Path,
        *,
        resume: bool,
        id_key: str = "sample_id",
        rubric_version: str,
        run_contract: Mapping[str, Any],
        source_paths: Iterable[Path] = (),
        expected_source_records: Iterable[Mapping[str, Any]] | None = None,
    ) -> None:
        self.output = output
        self.partial = output.with_name(output.name + ".partial")
        self.partial_state = output.with_name(output.name + ".partial.state.json")
        self.inventory = output.with_name(output.name + ".inventory.json")
        self.resume = resume
        self.id_key = id_key
        self.rubric_version = rubric_version
        self.source_paths = tuple(Path(item) for item in source_paths)
        self._expected_sources = (
            None
            if expected_source_records is None
            else _validate_expected_source_records(expected_source_records)
        )
        self._run_contract_json, _ = _canonical_contract(run_contract)
        self._run_contract_sha256 = hashlib.sha256(
            self._run_contract_json.encode("utf-8")
        ).hexdigest()
        self._handle: Any = None
        self._ids: set[str] = set()
        self._rows = 0
        self._recovered_tail = False
        self._sources: list[dict[str, Any]] = []

    def _contract(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "rubric_version": self.rubric_version,
            "artifact_path": str(self.output.resolve()),
            "id_key": self.id_key,
            "run_contract": _decode_object(
                self._run_contract_json, location="frozen run_contract"
            ),
            "run_contract_sha256": self._run_contract_sha256,
            "sources": self._sources,
        }

    def _validate_resume_metadata(self, seed: Path) -> None:
        if seed == self.partial:
            metadata_path = self.partial_state
            if not metadata_path.is_file():
                raise ArtifactError(
                    f"cannot resume partial artifact without provenance state: {metadata_path}"
                )
            metadata = _read_json_object(metadata_path)
            expected = self._contract()
            if metadata != expected:
                raise ArtifactError("partial artifact provenance does not match this run")
            return

        if not self.inventory.is_file():
            raise ArtifactError(
                f"cannot resume committed artifact without inventory: {self.inventory}"
            )
        metadata = _read_json_object(self.inventory)
        expected = self._contract()
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ArtifactError(
                    f"committed artifact inventory field {key!r} does not match this run"
                )
        if metadata.get("artifact_sha256") != sha256_file(seed):
            raise ArtifactError("committed artifact SHA-256 does not match its inventory")

    @property
    def done_ids(self) -> frozenset[str]:
        return frozenset(self._ids)

    def __enter__(self) -> "AtomicJsonlArtifact":
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # Freeze provenance before any rows are consumed. Commit verifies that
        # every source still has the same bytes and size, preventing a mixed
        # artifact from being published when an input changes mid-run.
        self._sources = _source_records(self.source_paths)
        if (
            self._expected_sources is not None
            and self._sources != self._expected_sources
        ):
            raise ArtifactError(
                "artifact sources changed after their exact-byte snapshot"
            )
        seed_rows: list[dict[str, Any]] = []
        if self.resume:
            seed = self.partial if self.partial.exists() else self.output
            if seed.exists():
                self._validate_resume_metadata(seed)
                seed_rows, self._recovered_tail = _load_complete_rows(seed, id_key=self.id_key)

        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.partial.name}.", suffix=".tmp", dir=str(self.output.parent)
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                for row in seed_rows:
                    handle.write(_json_dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.partial)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

        self._ids = {_identifier(row, self.id_key) for row in seed_rows}
        self._rows = len(seed_rows)
        _atomic_json(self.partial_state, self._contract())
        self._handle = self.partial.open("a", encoding="utf-8", newline="\n")
        return self

    def append(self, row: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise ArtifactError("artifact session is not open")
        payload = dict(row)
        sid = _identifier(payload, self.id_key)
        if sid in self._ids:
            raise ArtifactError(f"duplicate {self.id_key} {sid!r}")
        encoded = _json_dumps(payload)
        self._handle.write(encoded + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._ids.add(sid)
        self._rows += 1

    def commit(self) -> dict[str, Any]:
        if self._handle is None:
            raise ArtifactError("artifact session is not open")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None

        # Re-read before publication so a duplicate or malformed row can never
        # be hidden behind a valid inventory.
        rows, dropped_tail = _load_complete_rows(self.partial, id_key=self.id_key)
        if dropped_tail or len(rows) != self._rows:
            raise ArtifactError("partial artifact changed or has an incomplete tail")
        digest = sha256_file(self.partial)
        if _source_records(self.source_paths) != self._sources:
            raise ArtifactError("an artifact source changed while the output was being built")
        os.replace(self.partial, self.output)
        inventory = {
            **self._contract(),
            "artifact_sha256": digest,
            "rows": self._rows,
            "unique_ids": len(self._ids),
            "recovered_unterminated_tail": self._recovered_tail,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(self.inventory, inventory)
        self.partial_state.unlink(missing_ok=True)
        return inventory

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del traceback
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
        return False


__all__ = [
    "ArtifactError",
    "AtomicJsonlArtifact",
    "freeze_source_records",
    "iter_jsonl_objects",
    "load_jsonl_strict",
    "sha256_file",
]
