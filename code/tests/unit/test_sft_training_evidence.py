import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from motionllm.training.evidence import (
    OptimizerEvidenceTracker,
    completed_step_counts,
)


def _model_with_gradient(value: float = 1.0) -> torch.nn.Module:
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.tensor([[value]], dtype=model.weight.dtype)
    return model


class _DistributedSummaryReducer:
    class ReduceOp:
        MAX = object()

    def __init__(self, remote_summary):
        self.remote_summary = remote_summary
        self.calls = []

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def get_world_size():
        return 2

    def all_reduce(self, summary, *, op):
        self.calls.append((tuple(summary.shape), op))
        remote = torch.tensor(
            self.remote_summary,
            dtype=summary.dtype,
            device=summary.device,
        )
        summary.copy_(torch.maximum(summary, remote))


class _TorchWithDistributed:
    def __init__(self, distributed):
        self.distributed = distributed

    def __getattr__(self, name):
        return getattr(torch, name)


def test_successful_optimizer_event_requires_post_step_accelerate_signal():
    tracker = OptimizerEvidenceTracker(torch)
    tracker.record_pre_optimizer_step(_model_with_gradient())
    tracker.record_optimizer_step(
        SimpleNamespace(optimizer_step_was_skipped=False)
    )

    assert tracker.pre_optimizer_events == 1
    assert tracker.optimizer_steps == 1
    assert tracker.skipped_optimizer_steps == 0


def test_amp_overflow_skipped_step_is_rejected_fail_closed():
    tracker = OptimizerEvidenceTracker(torch)
    tracker.record_pre_optimizer_step(_model_with_gradient())

    with pytest.raises(RuntimeError, match="skipped after AMP overflow"):
        tracker.record_optimizer_step(
            SimpleNamespace(optimizer_step_was_skipped=True)
        )

    assert tracker.optimizer_steps == 0
    assert tracker.skipped_optimizer_steps == 1


def test_missing_skip_signal_and_callback_reordering_are_rejected():
    tracker = OptimizerEvidenceTracker(torch)
    with pytest.raises(RuntimeError, match="order/count"):
        tracker.record_optimizer_step(SimpleNamespace(optimizer_step_was_skipped=False))

    tracker.record_pre_optimizer_step(_model_with_gradient())
    with pytest.raises(RuntimeError, match="cannot verify"):
        tracker.record_optimizer_step(SimpleNamespace())


def test_global_step_cannot_mask_an_unconfirmed_optimizer_update():
    tracker = OptimizerEvidenceTracker(torch)
    tracker.record_pre_optimizer_step(_model_with_gradient())
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=1, max_steps=1))

    with pytest.raises(RuntimeError, match="optimizer callback count"):
        completed_step_counts(trainer, tracker, starting_global_step=0)


def test_device_summary_preserves_global_max_and_ignores_frozen_gradients():
    model = torch.nn.Module()
    model.register_parameter("small", torch.nn.Parameter(torch.zeros(2)))
    model.register_parameter("large", torch.nn.Parameter(torch.zeros(2)))
    model.register_parameter(
        "frozen",
        torch.nn.Parameter(torch.zeros(1), requires_grad=False),
    )
    model.small.grad = torch.tensor([-2.0, 0.5])
    model.large.grad = torch.tensor([7.5, -3.0])
    model.frozen.grad = torch.tensor([float("nan")])

    tracker = OptimizerEvidenceTracker(torch)
    tracker.record_pre_optimizer_step(model)

    assert tracker.pre_optimizer_events == 1
    assert tracker.nonzero_finite_gradient_steps == 1
    assert tracker.max_gradient == 7.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_any_nonfinite_trainable_gradient_is_rejected(value):
    tracker = OptimizerEvidenceTracker(torch)

    with pytest.raises(RuntimeError, match="non-finite trainable gradient"):
        tracker.record_pre_optimizer_step(_model_with_gradient(value))

    assert tracker.pre_optimizer_events == 0
    assert tracker.nonzero_finite_gradient_steps == 0


@pytest.mark.parametrize("gradient", [None, 0.0, -0.0])
def test_missing_or_all_zero_trainable_gradients_are_rejected(gradient):
    model = _model_with_gradient()
    model.weight.grad = (
        None
        if gradient is None
        else torch.tensor([[gradient]], dtype=model.weight.dtype)
    )
    tracker = OptimizerEvidenceTracker(torch)

    with pytest.raises(RuntimeError, match="no nonzero finite trainable gradient"):
        tracker.record_pre_optimizer_step(model)

    assert tracker.pre_optimizer_events == 1
    assert tracker.nonzero_finite_gradient_steps == 0


def test_distributed_summary_uses_one_max_collective_and_global_maximum():
    distributed = _DistributedSummaryReducer((1.0, 0.0, 11.0))
    tracker = OptimizerEvidenceTracker(_TorchWithDistributed(distributed))

    tracker.record_pre_optimizer_step(_model_with_gradient(3.0))

    assert distributed.calls == [((3,), distributed.ReduceOp.MAX)]
    assert tracker.max_gradient == 11.0
    assert tracker.nonzero_finite_gradient_steps == 1


def test_nonfinite_gradient_on_another_rank_is_rejected_after_collective():
    distributed = _DistributedSummaryReducer((1.0, 1.0, 0.0))
    tracker = OptimizerEvidenceTracker(_TorchWithDistributed(distributed))

    with pytest.raises(RuntimeError, match="non-finite trainable gradient"):
        tracker.record_pre_optimizer_step(_model_with_gradient(3.0))

    assert distributed.calls == [((3,), distributed.ReduceOp.MAX)]
    assert tracker.pre_optimizer_events == 0


def test_parameter_loop_contains_no_device_to_host_scalar_reads():
    source = textwrap.dedent(
        inspect.getsource(OptimizerEvidenceTracker.record_pre_optimizer_step)
    )
    tree = ast.parse(source)
    forbidden = {"cpu", "item", "numpy", "tolist"}
    loop_reads = [
        node.func.attr
        for loop in (node for node in ast.walk(tree) if isinstance(node, ast.For))
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden
    ]

    assert loop_reads == []
    assert source.count(".cpu().tolist()") == 1
