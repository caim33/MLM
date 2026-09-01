from __future__ import annotations

import json

import pytest

from motion_eval.__main__ import build_parser, main


@pytest.mark.parametrize(
    "argv",
    [
        ["registry", "validate", "--help"],
        ["batch", "create", "--help"],
        ["batch", "validate", "--help"],
        ["plan", "--help"],
        ["finetune", "barrier", "--help"],
        ["finetune", "attempt", "--help"],
        ["finetune", "run-attempt", "--help"],
        ["finetune", "complete", "--help"],
        ["finetune", "block", "--help"],
        ["finetune", "open-eval", "--help"],
        ["eval", "smoke", "--help"],
        ["eval", "attempt", "--help"],
        ["eval", "run-attempt", "--help"],
        ["eval", "open-full", "--help"],
        ["release", "build", "--help"],
        ["release", "verify", "--help"],
        ["gpu", "status", "--help"],
        ["gpu", "keepalive", "start", "--help"],
        ["gpu", "keepalive", "status", "--help"],
        ["gpu", "keepalive", "stop", "--help"],
        ["keepalive", "status", "--help"],
    ],
)
def test_required_cli_help_surfaces(argv):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 0


def test_registry_and_gpu_dry_runs(capsys):
    assert main(["registry", "validate", "--dry-run"]) == 0
    registry_output = capsys.readouterr().out
    assert '"model_count": 15' in registry_output
    assert main(["gpu", "status", "--dry-run"]) == 0
    gpu_output = capsys.readouterr().out
    assert "nvidia-smi" in gpu_output


def test_gpu_keepalive_parser_requires_one_start_selector_and_typed_timeouts():
    parser = build_parser()
    args = parser.parse_args(
        [
            "gpu",
            "keepalive",
            "start",
            "--root",
            "/tmp/motionllm-keepalive",
            "--owner",
            "motionllm-test",
            "--gpu-index",
            "2",
            "--heartbeat-interval-seconds",
            "3.5",
            "--ready-timeout-seconds",
            "7",
            "--wait-timeout-seconds",
            "9",
        ]
    )
    assert args.command == "gpu"
    assert args.gpu_command == "keepalive"
    assert args.keepalive_command == "start"
    assert args.gpu_uuid is None
    assert args.gpu_index == 2
    assert args.heartbeat_interval_seconds == 3.5
    assert args.ready_timeout_seconds == 7.0
    assert args.wait_timeout_seconds == 9.0

    common = ["gpu", "keepalive", "start", "--root", "/tmp/root"]
    with pytest.raises(SystemExit):
        parser.parse_args(common)
    with pytest.raises(SystemExit):
        parser.parse_args(common + ["--gpu-index", "0", "--gpu-uuid", "GPU-0"])
    with pytest.raises(SystemExit):
        parser.parse_args(common + ["--gpu-index", "0", "--wait-timeout-seconds", "nan"])


def test_gpu_keepalive_dispatches_public_controller_api_and_whitelists_json(
    monkeypatch, tmp_path, capsys
):
    calls = []
    secret = "DO-NOT-PRINT-KEEPALIVE-SECRET"

    class FakeKeepaliveController:
        def __init__(self, root, *, project_owner):
            calls.append(("init", str(root), project_owner))

        def start(self, selector, **kwargs):
            calls.append(("start", selector, kwargs))
            return {
                "gpu_uuid": "GPU-TEST",
                "gpu_index": 3,
                "pid": 101,
                "owner": "motionllm-test",
                "state": "active",
                "full_environment": {"PASSWORD": secret},
                "secret": secret,
            }

        def status(self, *, stale_after_seconds):
            calls.append(("status", stale_after_seconds))
            return [
                {
                    "gpu_uuid": "GPU-TEST",
                    "gpu_index": 3,
                    "pid": 101,
                    "owner": "motionllm-test",
                    "alive": True,
                    "environment": secret,
                }
            ]

        def stop(self, gpu_uuid, *, wait_timeout_seconds):
            calls.append(("stop", gpu_uuid, wait_timeout_seconds))
            return {
                "stopped": True,
                "gpu_uuid": gpu_uuid,
                "gpu_index": 3,
                "pid": 101,
                "owner": "motionllm-test",
                "environment": secret,
            }

    monkeypatch.setattr("motion_eval.__main__.KeepaliveController", FakeKeepaliveController)
    root = tmp_path / "keepalive"
    common = ["--root", str(root), "--owner", "motionllm-test"]

    assert main(
        [
            "gpu",
            "keepalive",
            "start",
            *common,
            "--gpu-index",
            "3",
            "--heartbeat-interval-seconds",
            "4",
            "--ready-timeout-seconds",
            "5",
            "--wait-timeout-seconds",
            "6",
        ]
    ) == 0
    start_output = capsys.readouterr().out
    assert json.loads(start_output)["record"]["state"] == "active"
    assert secret not in start_output

    assert main(
        [
            "gpu",
            "keepalive",
            "status",
            *common,
            "--stale-after-seconds",
            "17",
        ]
    ) == 0
    status_output = capsys.readouterr().out
    assert json.loads(status_output)["records"][0]["alive"] is True
    assert secret not in status_output

    assert main(
        [
            "gpu",
            "keepalive",
            "stop",
            *common,
            "--gpu-uuid",
            "GPU-TEST",
            "--wait-timeout-seconds",
            "8",
        ]
    ) == 0
    stop_output = capsys.readouterr().out
    assert json.loads(stop_output)["result"]["stopped"] is True
    assert secret not in stop_output

    assert calls == [
        ("init", str(root), "motionllm-test"),
        (
            "start",
            3,
            {
                "heartbeat_interval_seconds": 4.0,
                "ready_timeout_seconds": 5.0,
                "wait_timeout_seconds": 6.0,
            },
        ),
        ("init", str(root), "motionllm-test"),
        ("status", 17.0),
        ("init", str(root), "motionllm-test"),
        ("stop", "GPU-TEST", 8.0),
    ]


@pytest.mark.parametrize(
    "action_args",
    [
        ["start", "--gpu-uuid", "GPU-DRY"],
        ["status"],
        ["stop", "--gpu-uuid", "GPU-DRY"],
    ],
)
def test_gpu_keepalive_dry_run_never_constructs_controller_or_writes(
    action_args, monkeypatch, tmp_path, capsys
):
    class ForbiddenController:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry-run must not construct the mutating controller")

    monkeypatch.setattr("motion_eval.__main__.KeepaliveController", ForbiddenController)
    secret = "DRY-RUN-ENV-SECRET"
    monkeypatch.setenv("MOTION_EVAL_TEST_SECRET", secret)
    root = tmp_path / "must-not-exist"
    argv = [
        "gpu",
        "keepalive",
        *action_args,
        "--root",
        str(root),
        "--owner",
        "motionllm-test",
        "--dry-run",
    ]
    assert main(argv) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["dry_run"] is True
    assert payload["selected_gpu_role_mutex"] is True
    assert payload["real_cuda_remote_gate_required"] is True
    assert secret not in output
    assert not root.exists()


def test_legacy_keepalive_status_alias_is_deprecated_and_compatible(
    monkeypatch, tmp_path, capsys
):
    calls = []

    class FakeKeepaliveController:
        def __init__(self, root, *, project_owner):
            calls.append((str(root), project_owner))

        def status(self, *, stale_after_seconds):
            calls.append(stale_after_seconds)
            return []

    monkeypatch.setattr("motion_eval.__main__.KeepaliveController", FakeKeepaliveController)
    root = tmp_path / "legacy"
    assert main(
        [
            "keepalive",
            "status",
            "--root",
            str(root),
            "--owner",
            "legacy-owner",
            "--stale-after-seconds",
            "12",
        ]
    ) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["deprecated"] is True
    assert payload["replacement"] == "gpu keepalive status"
    assert "deprecated" in captured.err.lower()
    assert calls == [(str(root), "legacy-owner"), 12.0]


def test_gpu_keepalive_help_states_mutex_training_exclusion_and_real_cuda(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["gpu", "keepalive", "start", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out.lower()
    assert "role mutex" in help_text
    assert "finetune/eval" in help_text
    assert "real" in help_text and "cuda" in help_text
