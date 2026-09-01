from __future__ import annotations

import pytest

from motionllm.models import (
    MotionInjectionError,
    enumerate_motion_compute_placements,
    validate_motion_compute_contract,
)


class FakeTensor:
    def __init__(self, dtype: str, device: str, *, payload: str = "") -> None:
        self.dtype = dtype
        self.device = device
        self.payload = payload


class FakeModule:
    def __init__(self, *, parameters=(), buffers=()) -> None:
        self._parameters = tuple(parameters)
        self._buffers = tuple(buffers)

    def named_parameters(self, recurse=True):
        assert recurse is True
        return iter(self._parameters)

    def named_buffers(self, recurse=True):
        assert recurse is True
        return iter(self._buffers)


GOOD = FakeTensor("bf16", "cuda:0")


def placements(
    *,
    encoder: FakeModule | None = None,
    prenorm: FakeModule | None = None,
    projector: FakeModule | None = None,
    postnorm: FakeModule | None = None,
    apply_postnorm: bool = True,
    boundary: FakeModule | None = None,
):
    return enumerate_motion_compute_placements(
        encoder_name="motion_encoder.encoder",
        encoder=encoder or FakeModule(parameters=(("stem.weight", GOOD),)),
        motion_prenorm=prenorm or FakeModule(),
        motion_proj=projector or FakeModule(parameters=(("0.weight", GOOD),)),
        motion_postnorm=postnorm or FakeModule(parameters=(("weight", GOOD),)),
        apply_postnorm=apply_postnorm,
        motion_boundary_embed=boundary
        or FakeModule(parameters=(("weight", GOOD),)),
    )


def validate(values):
    validate_motion_compute_contract(
        expected_dtype="bf16",
        expected_device="cuda:0",
        module_placements=values,
    )


def test_later_encoder_parameter_mismatch_is_not_hidden_by_correct_first_layer():
    secret = "sentinel-parameter-payload"
    values = placements(
        encoder=FakeModule(
            parameters=(
                ("stem.weight", GOOD),
                ("blocks.3.weight", FakeTensor("float32", "cuda:0", payload=secret)),
            )
        )
    )
    with pytest.raises(
        MotionInjectionError,
        match=r"motion_encoder\.encoder\.blocks\.3\.weight dtype",
    ) as captured:
        validate(values)
    assert secret not in str(captured.value)


def test_buffer_only_module_mismatch_reports_exact_recursive_buffer_path():
    values = placements(
        prenorm=FakeModule(
            buffers=(("statistics.running_scale", FakeTensor("bf16", "cuda:1")),)
        )
    )
    with pytest.raises(
        MotionInjectionError,
        match=r"motion_prenorm\.statistics\.running_scale device",
    ):
        validate(values)


def test_boundary_embedding_mismatch_is_part_of_precompute_contract():
    values = placements(
        boundary=FakeModule(
            parameters=(("weight", FakeTensor("float32", "cuda:0")),)
        )
    )
    with pytest.raises(
        MotionInjectionError,
        match=r"motion_boundary_embed\.weight dtype",
    ):
        validate(values)


def test_all_parameters_and_buffers_pass_and_parameterless_modules_are_supported():
    values = placements(
        encoder=FakeModule(
            parameters=(("stem.weight", GOOD), ("blocks.1.bias", GOOD)),
            buffers=(("blocks.1.scale", GOOD),),
        ),
        prenorm=FakeModule(),
        projector=FakeModule(
            parameters=(("0.weight", GOOD), ("2.weight", GOOD)),
            buffers=(("calibration", GOOD),),
        ),
        postnorm=FakeModule(buffers=(("running", GOOD),)),
        boundary=FakeModule(parameters=(("weight", GOOD),)),
    )
    validate(values)
    assert {name for name, _, _ in values} == {
        "motion_encoder.encoder.stem.weight",
        "motion_encoder.encoder.blocks.1.bias",
        "motion_encoder.encoder.blocks.1.scale",
        "motion_proj.0.weight",
        "motion_proj.2.weight",
        "motion_proj.calibration",
        "motion_postnorm.running",
        "motion_boundary_embed.weight",
    }
