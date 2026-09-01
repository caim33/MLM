"""Framework-independent motion/text fusion contracts."""

from .errors import FusionContractError, MotionPlaceholderError, ProjectorSpecError
from .placeholders import (
    MotionSpan,
    MotionTextProtocol,
    MotionTokenIds,
    TextAnchor,
    find_motion_anchors,
    parse_and_validate_motion_spans,
    parse_motion_spans,
    render_motion_span,
    replace_motion_anchors,
    validate_placeholder_counts,
)
from .projector import (
    LinearLayerSpec,
    ProjectorSpec,
    build_projector_spec,
    infer_projector_output_shape,
)

__all__ = [
    "FusionContractError",
    "LinearLayerSpec",
    "MotionPlaceholderError",
    "MotionSpan",
    "MotionTextProtocol",
    "MotionTokenIds",
    "ProjectorSpec",
    "ProjectorSpecError",
    "TextAnchor",
    "build_projector_spec",
    "find_motion_anchors",
    "infer_projector_output_shape",
    "parse_and_validate_motion_spans",
    "parse_motion_spans",
    "render_motion_span",
    "replace_motion_anchors",
    "validate_placeholder_counts",
]
