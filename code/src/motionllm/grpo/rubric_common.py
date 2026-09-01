"""Framework-independent validation helpers shared by Rubric-RL variants."""

from __future__ import annotations

import json
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


class RubricValidationError(ValueError):
    """A rubric, judgment, or candidate violates its frozen contract."""


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def strict_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise RubricValidationError(
            f"{name} must match {_IDENTIFIER_RE.pattern!r}"
        )
    return value


def strict_text(value: Any, *, name: str, max_length: int = 16_384) -> str:
    if not isinstance(value, str):
        raise RubricValidationError(f"{name} must be a string")
    if value != value.strip() or not value:
        raise RubricValidationError(f"{name} must be non-empty and trimmed")
    if len(value) > max_length or any(character in value for character in ("\x00", "\r")):
        raise RubricValidationError(f"{name} is too long or contains forbidden control data")
    return value


def finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RubricValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RubricValidationError(f"{name} must be finite")
    return result


def require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RubricValidationError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise RubricValidationError(f"{name} keys must be strings")
    return value


def require_exact_keys(
    value: Mapping[str, Any],
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing or extra:
        raise RubricValidationError(
            f"{name} keys mismatch: missing={missing}, unexpected={extra}"
        )


def strict_id_list(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise RubricValidationError(f"{name} must be a list")
    output: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        identifier = strict_identifier(item, name=f"{name}[{index}]")
        if identifier in seen:
            raise RubricValidationError(f"{name} contains duplicate id {identifier!r}")
        seen.add(identifier)
        output.append(identifier)
    return output


def strict_json_object(text: Any, *, max_bytes: int = 1_048_576) -> dict[str, Any]:
    """Parse exactly one JSON object, rejecting duplicate keys and NaN/Infinity."""

    if not isinstance(text, str):
        raise RubricValidationError("judge output must be text")
    if len(text.encode("utf-8")) > max_bytes:
        raise RubricValidationError("judge output exceeds the size limit")
    if text != text.strip() or not text:
        raise RubricValidationError("judge output must be one trimmed JSON object")

    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RubricValidationError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise RubricValidationError(f"non-finite JSON constant is forbidden: {value}")

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except RubricValidationError:
        raise
    except Exception as exc:
        raise RubricValidationError("judge output is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RubricValidationError("judge output must be one JSON object")
    return parsed


def canonical_sha256(value: Any, *, name: str) -> str:
    """Hash a finite canonical JSON value for cross-process bindings."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RubricValidationError(f"{name} is not finite canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise RubricValidationError(f"{name} must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_judgment_binding(
    criteria: Mapping[str, Any],
    candidate_response: str,
    *,
    sample_id: str,
    nonce: str,
) -> dict[str, str]:
    """Build the exact identity a judgment is allowed to score."""

    return {
        "sample_id": strict_identifier(sample_id, name="judgment.binding.sample_id"),
        "criteria_sha256": canonical_sha256(criteria, name="criteria"),
        "candidate_sha256": text_sha256(
            candidate_response, name="candidate_response"
        ),
        "nonce": strict_identifier(nonce, name="judgment.binding.nonce"),
    }


def validate_judgment_binding(
    value: Any,
    criteria: Mapping[str, Any],
    candidate_response: str,
    *,
    sample_id: str,
    expected_nonce: str | None = None,
) -> dict[str, str]:
    raw = require_mapping(value, name="judgment.binding")
    require_exact_keys(
        raw,
        name="judgment.binding",
        required={"sample_id", "criteria_sha256", "candidate_sha256", "nonce"},
    )
    nonce = strict_identifier(raw.get("nonce"), name="judgment.binding.nonce")
    expected = build_judgment_binding(
        criteria,
        candidate_response,
        sample_id=sample_id,
        nonce=expected_nonce if expected_nonce is not None else nonce,
    )
    observed = {
        "sample_id": strict_identifier(
            raw.get("sample_id"), name="judgment.binding.sample_id"
        ),
        "criteria_sha256": raw.get("criteria_sha256"),
        "candidate_sha256": raw.get("candidate_sha256"),
        "nonce": nonce,
    }
    for key in ("criteria_sha256", "candidate_sha256"):
        if not isinstance(observed[key], str) or _SHA256_RE.fullmatch(observed[key]) is None:
            raise RubricValidationError(f"judgment.binding.{key} must be lowercase SHA-256")
    if observed != expected:
        raise RubricValidationError("judgment binding does not match sample/criteria/candidate/nonce")
    return expected


def require_partition(
    universe: set[str],
    groups: Mapping[str, list[str] | set[str]],
    *,
    name: str,
) -> dict[str, set[str]]:
    """Require named groups to be pairwise-disjoint and exhaustive."""

    normalized = {key: set(value) for key, value in groups.items()}
    keys = tuple(normalized)
    overlaps: list[str] = []
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            shared = sorted(normalized[left] & normalized[right])
            if shared:
                overlaps.append(f"{left}/{right}={shared}")
    covered = set().union(*normalized.values()) if normalized else set()
    missing = sorted(universe - covered)
    extra = sorted(covered - universe)
    if overlaps or missing or extra:
        raise RubricValidationError(
            f"{name} must be a disjoint exhaustive partition: "
            f"overlaps={overlaps}, missing={missing}, extra={extra}"
        )
    return normalized


def expand_column(value: Any, size: int, *, name: str) -> list[Any]:
    """Expand one scalar or exact-length sequence for a reward batch."""

    if size < 0:
        raise RubricValidationError("batch size must be non-negative")
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        if value is None:
            raise RubricValidationError(f"{name} is required")
        return [value] * size
    items = list(value)
    if len(items) != size:
        raise RubricValidationError(
            f"{name} length {len(items)} does not match completions {size}"
        )
    return items


__all__ = [
    "RubricValidationError",
    "build_judgment_binding",
    "canonical_sha256",
    "expand_column",
    "finite_number",
    "require_exact_keys",
    "require_mapping",
    "require_partition",
    "strict_id_list",
    "strict_identifier",
    "strict_json_object",
    "strict_text",
    "text_sha256",
    "validate_judgment_binding",
]
