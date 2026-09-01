"""Canonical input modality definitions."""

from __future__ import annotations

from enum import Enum

from .errors import ModalityContractError


class Modality(str, Enum):
    """The only modalities accepted by a canonical MotionLLM sample.

    ``parse`` is intentionally case-sensitive.  Legacy lower-case ``branch``
    values are supported through the explicitly named ``from_branch`` method,
    which prevents accidental normalization at canonical data boundaries.
    """

    VIDEO = "V"
    MOTION = "M"
    VIDEO_MOTION = "VM"
    TEXT = "T"

    @classmethod
    def parse(cls, value: object) -> "Modality":
        if not isinstance(value, str):
            raise ModalityContractError("modality must be one of V, M, VM, or T")
        try:
            return cls(value)
        except ValueError as exc:
            raise ModalityContractError(
                "modality must be exactly one of V, M, VM, or T"
            ) from exc

    @classmethod
    def from_branch(cls, value: object) -> "Modality":
        if not isinstance(value, str):
            raise ModalityContractError("branch must be one of v, m, vm, or t")
        branch_map = {
            "v": cls.VIDEO,
            "m": cls.MOTION,
            "vm": cls.VIDEO_MOTION,
            "t": cls.TEXT,
        }
        try:
            return branch_map[value]
        except KeyError as exc:
            raise ModalityContractError(
                "legacy branch must be exactly one of v, m, vm, or t"
            ) from exc

    @property
    def branch(self) -> str:
        """Return the compatibility branch name used by legacy metadata."""

        return self.value.lower()

    @property
    def requires_video(self) -> bool:
        return self in (Modality.VIDEO, Modality.VIDEO_MOTION)

    @property
    def requires_motion(self) -> bool:
        return self in (Modality.MOTION, Modality.VIDEO_MOTION)

    def __str__(self) -> str:
        return self.value
