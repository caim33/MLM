from __future__ import annotations

import pytest

from motionllm.models import (
    MotionInjectionError,
    is_generation_prefill,
    prefill_motion_payload,
)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def test_cache_position_is_authoritative_for_prefill_and_decode():
    assert is_generation_prefill(cache_position=[0], past_key_values=object())
    assert not is_generation_prefill(cache_position=(1,), past_key_values=None)
    assert not is_generation_prefill(cache_position=Scalar(8), past_key_values=None)


def test_old_transformers_fallback_uses_cache_presence():
    assert is_generation_prefill(cache_position=None, past_key_values=None)
    assert not is_generation_prefill(cache_position=None, past_key_values=object())


def test_decode_drops_motion_and_lengths_without_mutating_prefill():
    motion = object()
    lengths = ((8,),)
    assert prefill_motion_payload(
        motion, lengths, cache_position=[0]
    ) == (motion, lengths)
    assert prefill_motion_payload(
        motion, lengths, cache_position=[1]
    ) == (None, None)


@pytest.mark.parametrize("bad", [[], -1, True, "0", [False]])
def test_bad_cache_positions_fail_closed(bad):
    with pytest.raises(MotionInjectionError):
        is_generation_prefill(cache_position=bad)

