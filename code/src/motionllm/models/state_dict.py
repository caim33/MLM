"""Strict, framework-free state-dict extraction and shape auditing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .errors import StateDictAuditError


_WRAPPER_KEYS = ("state_dict", "net", "model_state_dict")


def extract_state_dict(payload: Any) -> Mapping[str, Any]:
    """Extract one recognized checkpoint mapping without heuristic recursion."""

    if not isinstance(payload, Mapping):
        raise StateDictAuditError("checkpoint payload must be a mapping")
    selected: Mapping[Any, Any] = payload
    wrappers = [key for key in _WRAPPER_KEYS if key in payload]
    if len(wrappers) > 1:
        raise StateDictAuditError(
            f"checkpoint contains ambiguous state wrappers: {wrappers!r}"
        )
    if wrappers:
        value = payload[wrappers[0]]
        if not isinstance(value, Mapping):
            raise StateDictAuditError(
                f"checkpoint wrapper {wrappers[0]!r} must contain a mapping"
            )
        selected = value
    if not selected:
        raise StateDictAuditError("checkpoint state dict must not be empty")
    if any(not isinstance(key, str) or not key for key in selected):
        raise StateDictAuditError("checkpoint state keys must be non-empty strings")
    return selected  # type: ignore[return-value]


def normalize_state_dict_keys(
    state: Mapping[str, Any],
    *,
    strip_prefixes: Sequence[str] = ("module.", "vqvae."),
) -> dict[str, Any]:
    """Strip only leading legacy prefixes and reject key collisions."""

    if not isinstance(state, Mapping):
        raise StateDictAuditError("state must be a mapping")
    prefixes = tuple(strip_prefixes)
    if any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
        raise StateDictAuditError("strip prefixes must be non-empty strings")
    normalized: dict[str, Any] = {}
    for original, value in state.items():
        if not isinstance(original, str) or not original:
            raise StateDictAuditError("state keys must be non-empty strings")
        key = original
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        if not key:
            raise StateDictAuditError(f"state key {original!r} becomes empty")
        if key in normalized:
            raise StateDictAuditError(
                f"state key collision after prefix normalization: {original!r} -> {key!r}"
            )
        normalized[key] = value
    return normalized


def _shape(value: Any, *, key: str) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raise StateDictAuditError(f"state value {key!r} has no shape")
    try:
        shape = tuple(int(dimension) for dimension in raw)
    except (TypeError, ValueError) as exc:
        raise StateDictAuditError(f"state value {key!r} has an invalid shape") from exc
    if any(dimension < 0 for dimension in shape):
        raise StateDictAuditError(f"state value {key!r} has a negative dimension")
    return shape


@dataclass(frozen=True, slots=True)
class StateDictAudit:
    matched_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def ok(self) -> bool:
        return not (self.missing_keys or self.unexpected_keys or self.shape_mismatches)

    def require_clean(self) -> "StateDictAudit":
        if not self.ok:
            details: list[str] = []
            if self.missing_keys:
                details.append(f"missing={list(self.missing_keys)!r}")
            if self.unexpected_keys:
                details.append(f"unexpected={list(self.unexpected_keys)!r}")
            if self.shape_mismatches:
                details.append(f"shape_mismatches={list(self.shape_mismatches)!r}")
            raise StateDictAuditError("checkpoint state audit failed: " + "; ".join(details))
        return self


def audit_state_dict(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> StateDictAudit:
    """Compare exact keys and tensor-like shapes before framework loading."""

    if not isinstance(expected, Mapping) or not isinstance(candidate, Mapping):
        raise StateDictAuditError("expected and candidate states must be mappings")
    expected_keys = set(expected)
    candidate_keys = set(candidate)
    if any(not isinstance(key, str) or not key for key in expected_keys | candidate_keys):
        raise StateDictAuditError("state keys must be non-empty strings")
    common = expected_keys & candidate_keys
    mismatches = tuple(
        (key, _shape(expected[key], key=key), _shape(candidate[key], key=key))
        for key in sorted(common)
        if _shape(expected[key], key=key) != _shape(candidate[key], key=key)
    )
    bad = {item[0] for item in mismatches}
    return StateDictAudit(
        matched_keys=tuple(sorted(common - bad)),
        missing_keys=tuple(sorted(expected_keys - candidate_keys)),
        unexpected_keys=tuple(sorted(candidate_keys - expected_keys)),
        shape_mismatches=mismatches,
    )


@dataclass(frozen=True, slots=True)
class VQCheckpointSelection:
    """One unambiguous, strictly audited VQ checkpoint load target."""

    target: Literal["full_vqvae", "encoder_only"]
    state_dict: Mapping[str, Any]
    audit: StateDictAudit


def select_vq_checkpoint_state(
    *,
    full_expected: Mapping[str, Any],
    encoder_expected: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> VQCheckpointSelection:
    """Select one exact full-VQ or encoder-only checkpoint schema.

    Encoder-only checkpoints are commonly saved either from
    ``encoder.state_dict()`` (bare keys) or from the parent VQ module
    (``encoder.``-prefixed keys).  Both are accepted only when their keys and
    shapes exactly match the encoder.  Partial, mixed, and ambiguous payloads
    remain fatal.
    """

    if not isinstance(full_expected, Mapping) or not isinstance(
        encoder_expected, Mapping
    ) or not isinstance(candidate, Mapping):
        raise StateDictAuditError(
            "full, encoder, and candidate states must be mappings"
        )

    full_audit = audit_state_dict(full_expected, candidate)
    if full_audit.ok:
        return VQCheckpointSelection(
            target="full_vqvae",
            state_dict=candidate,
            audit=full_audit.require_clean(),
        )

    encoder_candidates: list[tuple[str, Mapping[str, Any], StateDictAudit]] = []
    bare_audit = audit_state_dict(encoder_expected, candidate)
    if bare_audit.ok:
        encoder_candidates.append(("bare", candidate, bare_audit))

    if candidate and all(key.startswith("encoder.") for key in candidate):
        prefixed = {key[len("encoder.") :]: value for key, value in candidate.items()}
        prefixed_audit = audit_state_dict(encoder_expected, prefixed)
        if prefixed_audit.ok:
            encoder_candidates.append(("encoder_prefixed", prefixed, prefixed_audit))

    if len(encoder_candidates) == 1:
        _, selected, audit = encoder_candidates[0]
        return VQCheckpointSelection(
            target="encoder_only",
            state_dict=selected,
            audit=audit.require_clean(),
        )
    if len(encoder_candidates) > 1:
        schemas = [name for name, _, _ in encoder_candidates]
        raise StateDictAuditError(
            f"checkpoint matches multiple encoder-only schemas: {schemas!r}"
        )

    details = [
        "checkpoint matches neither the full VQ-VAE nor an exact bare/encoder-prefixed encoder state",
        f"full_missing={list(full_audit.missing_keys)!r}",
        f"full_unexpected={list(full_audit.unexpected_keys)!r}",
        f"full_shape_mismatches={list(full_audit.shape_mismatches)!r}",
    ]
    raise StateDictAuditError("; ".join(details))
