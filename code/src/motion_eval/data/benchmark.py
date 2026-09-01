"""Canonical fixed-denominator benchmark loading and smoke selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .jsonio import StrictJsonError, load_jsonl_strict


CANONICAL_BENCHMARK_SIZE = 500
SMOKE_SIZES = frozenset({1, 8, 32})


def _identity(value: object, *, name: str, source: Path | None = None, row: int | None = None) -> str:
    prefix = ""
    if source is not None:
        prefix = f"{source}"
        if row is not None:
            prefix += f": line {row}"
        prefix += ": "
    if not isinstance(value, str) or not value or value != value.strip():
        raise StrictJsonError(f"{prefix}benchmark requires an explicit {name}")
    if any(
        ord(character) < 32
        or ord(character) == 127
        or 0xD800 <= ord(character) <= 0xDFFF
        for character in value
    ):
        raise StrictJsonError(f"{prefix}benchmark {name} contains a control character")
    return value


def _media_reference(value: object, *, name: str, source: Path, row: int) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise StrictJsonError(
            f"{source}: line {row}: benchmark {name} must be a canonical media reference object"
        )
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """Identity needed by evaluation plus optional frozen derivation evidence."""

    sample_id: str
    group_id: str
    gold: str
    question: str | None = None
    options: Mapping[str, Any] | None = None
    video: Mapping[str, Any] | None = None
    motion: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _identity(self.sample_id, name="sample_id")
        _identity(self.group_id, name="group_id")
        if self.gold not in {"A", "B", "C", "D"}:
            raise StrictJsonError("benchmark gold must be exactly A, B, C, or D")


def load_benchmark(path: str | Path) -> tuple[BenchmarkItem, ...]:
    """Load the one canonical 500-row benchmark in physical row order."""

    source = Path(path).resolve(strict=True)
    rows = load_jsonl_strict(source)
    if len(rows) != CANONICAL_BENCHMARK_SIZE:
        raise StrictJsonError(
            f"{source}: canonical benchmark must contain exactly {CANONICAL_BENCHMARK_SIZE} rows"
        )

    result: list[BenchmarkItem] = []
    seen: dict[str, int] = {}
    for line_number, row in enumerate(rows, start=1):
        sample_id = _identity(
            row.get("sample_id"), name="sample_id", source=source, row=line_number
        )
        group_id = _identity(
            row.get("group_id"), name="group_id", source=source, row=line_number
        )
        if sample_id in seen:
            raise StrictJsonError(
                f"{source}: line {line_number}: duplicate benchmark sample_id; "
                f"first appeared at line {seen[sample_id]}"
            )
        seen[sample_id] = line_number
        gold = row.get("gold")
        if gold not in {"A", "B", "C", "D"}:
            raise StrictJsonError(
                f"{source}: line {line_number}: benchmark gold must be exactly A, B, C, or D"
            )
        question = row.get("question")
        if question is not None and not isinstance(question, str):
            raise StrictJsonError(
                f"{source}: line {line_number}: benchmark question must be a string"
            )
        options = row.get("options")
        if options is not None and not isinstance(options, Mapping):
            raise StrictJsonError(
                f"{source}: line {line_number}: benchmark options must be an object"
            )
        result.append(
            BenchmarkItem(
                sample_id,
                group_id,
                gold,
                question=question,
                options=(MappingProxyType(dict(options)) if options is not None else None),
                video=_media_reference(
                    row.get("video"), name="video", source=source, row=line_number
                ),
                motion=_media_reference(
                    row.get("motion"), name="motion", source=source, row=line_number
                ),
            )
        )
    return tuple(result)


def smoke_items(
    benchmark: Sequence[BenchmarkItem], size: int
) -> tuple[BenchmarkItem, ...]:
    """Return the deterministic physical-prefix smoke slice for 1/8/32."""

    if isinstance(size, bool) or not isinstance(size, int) or size not in SMOKE_SIZES:
        raise ValueError("smoke size must be exactly 1, 8, or 32")
    if len(benchmark) != CANONICAL_BENCHMARK_SIZE:
        raise ValueError(
            f"smoke selection requires the complete {CANONICAL_BENCHMARK_SIZE}-row benchmark"
        )
    if any(not isinstance(item, BenchmarkItem) for item in benchmark):
        raise TypeError("benchmark must contain only BenchmarkItem values")
    return tuple(benchmark[:size])


__all__ = [
    "BenchmarkItem",
    "CANONICAL_BENCHMARK_SIZE",
    "SMOKE_SIZES",
    "load_benchmark",
    "smoke_items",
]
