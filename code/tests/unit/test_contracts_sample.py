from __future__ import annotations

from pathlib import Path

import pytest

from motionllm.contracts import (
    GoldAnswer,
    MediaContractError,
    MediaReferences,
    Modality,
    Option,
    OptionContractError,
    OptionLabel,
    Sample,
    SampleContractError,
)


def _options() -> tuple[Option, ...]:
    return tuple(Option(label, f"option {label.value}") for label in OptionLabel)


def _sample(
    modality: Modality,
    *,
    video: Path | None = None,
    motion: Path | None = None,
    **kwargs: object,
) -> Sample:
    values = {
        "sample_id": "sample-1",
        "group_id": "group-1",
        "modality": modality,
        "question": "Which option is correct?",
        "options": _options(),
        "gold": GoldAnswer.from_label(OptionLabel.A),
        "media": MediaReferences(video=video, motion=motion),
    }
    values.update(kwargs)
    return Sample(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("modality", "has_video", "has_motion"),
    [
        (Modality.VIDEO, True, False),
        (Modality.MOTION, False, True),
        (Modality.VIDEO_MOTION, True, True),
        (Modality.TEXT, False, False),
    ],
)
def test_valid_modality_media_matrix(
    tmp_path: Path,
    modality: Modality,
    has_video: bool,
    has_motion: bool,
) -> None:
    sample = _sample(
        modality,
        video=tmp_path / "clip.mp4" if has_video else None,
        motion=tmp_path / "motion.npy" if has_motion else None,
    )
    assert sample.branch == modality.branch
    assert (sample.video is not None) is has_video
    assert (sample.motion is not None) is has_motion


@pytest.mark.parametrize("modality", list(Modality))
@pytest.mark.parametrize(
    ("has_video", "has_motion"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_every_wrong_modality_media_combination_is_rejected(
    tmp_path: Path,
    modality: Modality,
    has_video: bool,
    has_motion: bool,
) -> None:
    expected = (modality.requires_video, modality.requires_motion)
    if (has_video, has_motion) == expected:
        return
    with pytest.raises(MediaContractError):
        _sample(
            modality,
            video=tmp_path / "clip.mp4" if has_video else None,
            motion=tmp_path / "motion.npy" if has_motion else None,
        )


def test_media_references_must_already_be_absolute() -> None:
    with pytest.raises(MediaContractError):
        MediaReferences(video=Path("relative.mp4"))
    with pytest.raises(MediaContractError):
        MediaReferences(motion="motion.npy")  # type: ignore[arg-type]


def test_media_references_must_be_normalized(tmp_path: Path) -> None:
    unnormalized = tmp_path / "nested" / ".." / "clip.mp4"
    with pytest.raises(MediaContractError, match="normalized"):
        MediaReferences(video=unnormalized)


def test_options_must_be_unique_and_in_abcd_order() -> None:
    wrong_order = tuple(reversed(_options()))
    with pytest.raises(OptionContractError):
        _sample(Modality.TEXT, options=wrong_order)

    duplicate_text = tuple(Option(label, "same") for label in OptionLabel)
    with pytest.raises(OptionContractError):
        _sample(Modality.TEXT, options=duplicate_text)


def test_compatibility_metadata_fields_are_typed(tmp_path: Path) -> None:
    sample = _sample(
        Modality.MOTION,
        motion=tmp_path / "motion.npy",
        rollout_id=3,
        request_id="request-1",
        motion_lengths=[12, 8],
        metadata={"nested": {"scores": [1, 2]}},
    )
    assert sample.motion_lengths == (12, 8)
    assert sample.rollout_id == 3
    assert sample.request_id == "request-1"
    assert sample.metadata["nested"]["scores"] == (1, 2)
    with pytest.raises(TypeError):
        sample.metadata["new"] = True  # type: ignore[index]


def test_motion_lengths_are_forbidden_without_motion(tmp_path: Path) -> None:
    with pytest.raises(MediaContractError):
        _sample(
            Modality.VIDEO,
            video=tmp_path / "clip.mp4",
            motion_lengths=(4,),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_id", ""),
        ("sample_id", " leading"),
        ("group_id", "group\x00bad"),
        ("question", ""),
        ("request_id", " trailing "),
        ("rollout_id", -1),
        ("rollout_id", True),
    ],
)
def test_sample_rejects_invalid_identity_fields(
    field: str, value: object
) -> None:
    with pytest.raises(SampleContractError):
        _sample(Modality.TEXT, **{field: value})


def test_to_dict_preserves_canonical_gold_and_compatibility_branch(
    tmp_path: Path,
) -> None:
    sample = _sample(Modality.VIDEO, video=tmp_path / "clip.mp4")
    payload = sample.to_dict()
    assert payload["modality"] == "V"
    assert payload["branch"] == "v"
    assert payload["gold"] == "<answer>A</answer>"
    assert payload["options"] == {
        "A": "option A",
        "B": "option B",
        "C": "option C",
        "D": "option D",
    }
