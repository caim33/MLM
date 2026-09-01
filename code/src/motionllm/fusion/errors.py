"""Exceptions raised by framework-independent fusion contracts."""


class FusionContractError(ValueError):
    """Base class for an invalid fusion protocol or specification."""


class MotionPlaceholderError(FusionContractError):
    """Motion anchors, boundary tokens, or placeholder counts are invalid."""


class ProjectorSpecError(FusionContractError):
    """A motion projector configuration or input shape is invalid."""
