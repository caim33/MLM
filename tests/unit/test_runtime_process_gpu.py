from __future__ import annotations

import json
import os
import sys
import venv
from pathlib import Path

import pytest

from motion_eval.contracts import EvaluationErrorCode
from motion_eval.core import hash_path, sha256_file
import motion_eval.runtime.process as process_module
from motion_eval.runtime import (
    CommandSpec,
    GPUDevice,
    GPUInventory,
    GPULeaseStore,
    GPUQueryError,
    KeepaliveStore,
    PinnedSshTarget,
    run_command,
    run_verified_python,
)


def test_subprocess_is_shell_free_and_command_metacharacters_are_data(tmp_path):
    marker = tmp_path / "must_not_exist"
    payload = f"; touch {marker}"
    spec = CommandSpec(
        argv=(sys.executable, "-c", "import sys; print(sys.argv[1])", payload),
        cwd=str(tmp_path),
        timeout_seconds=5,
    )
    result = run_command(spec)
    assert result.succeeded
    assert payload in result.stdout
    assert not marker.exists()


def test_secret_environment_value_is_redacted_from_all_output(tmp_path):
    secret = "unit-test-secret-value-723"
    spec = CommandSpec(
        argv=(
            sys.executable,
            "-c",
            "import os,sys; print(os.environ['HF_TOKEN']); print(os.environ['HF_TOKEN'], file=sys.stderr)",
        ),
        cwd=str(tmp_path),
        env={"HF_TOKEN": secret},
        timeout_seconds=5,
    )
    assert secret not in str(spec.receipt())
    result = run_command(spec)
    assert result.succeeded
    assert secret not in result.stdout + result.stderr
    assert "<redacted>" in result.stdout and "<redacted>" in result.stderr


def test_timeout_and_oom_are_classified(tmp_path):
    timeout = run_command(
        CommandSpec(
            argv=(sys.executable, "-c", "import time; time.sleep(2)"),
            cwd=str(tmp_path),
            timeout_seconds=0.05,
        )
    )
    assert timeout.error_code is EvaluationErrorCode.TIMEOUT
    oom = run_command(
        CommandSpec(
            argv=(sys.executable, "-c", "import sys; print('CUDA out of memory', file=sys.stderr); sys.exit(1)"),
            cwd=str(tmp_path),
            timeout_seconds=5,
        )
    )
    assert oom.error_code is EvaluationErrorCode.OOM


def test_verified_python_executes_frozen_bytes_with_script_semantics(tmp_path):
    interpreter = str(Path(sys.executable).resolve(strict=True))
    script = tmp_path / "worker.py"
    script.write_text(
        "import sys\nprint(__file__)\nprint('|'.join(sys.argv))\n",
        encoding="utf-8",
    )
    spec = CommandSpec(
        argv=(interpreter, str(script.resolve()), "alpha", "beta"),
        cwd=str(tmp_path),
        timeout_seconds=5,
    )
    result = run_verified_python(
        spec,
        expected_interpreter_sha256=sha256_file(interpreter),
        expected_script_sha256=sha256_file(script),
        import_root=str(tmp_path.resolve()),
        expected_import_root_receipt={
            "path": str(tmp_path.resolve()),
            **hash_path(tmp_path, symlink_policy="reject").to_dict(),
        },
    )
    assert result.succeeded
    assert str(script.resolve()) in result.stdout
    assert f"{script.resolve()}|alpha|beta" in result.stdout


def test_verified_python_ignores_ambient_pythonpath_sitecustomize(
    tmp_path, monkeypatch
):
    interpreter = str(Path(sys.executable).resolve(strict=True))
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    marker = tmp_path / "sitecustomize-ran.txt"
    (ambient / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    worker_root = tmp_path / "worker"
    worker_root.mkdir()
    script = worker_root / "worker.py"
    script.write_text("print('VERIFIED-WORKER')\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(ambient))

    result = run_verified_python(
        CommandSpec(argv=(interpreter, str(script.resolve())), cwd=str(worker_root)),
        expected_interpreter_sha256=sha256_file(interpreter),
        expected_script_sha256=sha256_file(script),
        import_root=str(worker_root.resolve()),
        expected_import_root_receipt={
            "path": str(worker_root.resolve()),
            **hash_path(worker_root, symlink_policy="reject").to_dict(),
        },
    )

    assert result.succeeded, result.stderr
    assert "VERIFIED-WORKER" in result.stdout
    assert not marker.exists()


def test_verified_python_preserves_symlinked_venv_launcher_and_site_packages(tmp_path):
    environment = tmp_path / "venv"
    try:
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create a symlink virtualenv on this platform: {exc}")
    launcher = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not launcher.is_symlink():
        pytest.skip("virtualenv did not create a symlink Python launcher")

    probe = run_command(
        CommandSpec(
            argv=(
                str(launcher),
                "-c",
                "import json,sys,sysconfig; "
                "print(json.dumps({'prefix':sys.prefix,'purelib':sysconfig.get_paths()['purelib']}))",
            ),
            cwd=str(tmp_path),
            timeout_seconds=10,
        )
    )
    assert probe.succeeded, probe.stderr
    environment_details = json.loads(probe.stdout)
    assert Path(environment_details["prefix"]).resolve() == environment.resolve()
    purelib = Path(environment_details["purelib"])
    assert purelib.resolve().is_relative_to(environment.resolve())
    (purelib / "venv_only_dependency.py").write_text(
        "VALUE = 'VENV-DEPENDENCY-FOUND'\n", encoding="utf-8"
    )

    worker_root = tmp_path / "worker"
    worker_root.mkdir()
    script = worker_root / "worker.py"
    script.write_text(
        "import json,sys,venv_only_dependency\n"
        "print(json.dumps({'prefix':sys.prefix,'value':venv_only_dependency.VALUE}))\n",
        encoding="utf-8",
    )
    target = launcher.resolve(strict=True)
    result = run_verified_python(
        CommandSpec(argv=(str(launcher), str(script)), cwd=str(worker_root)),
        expected_interpreter_sha256=sha256_file(target),
        expected_interpreter_target_path=str(target),
        expected_script_sha256=sha256_file(script),
        import_root=str(worker_root.resolve()),
        expected_import_root_receipt={
            "path": str(worker_root.resolve()),
            **hash_path(worker_root, symlink_policy="reject").to_dict(),
        },
    )
    assert result.succeeded, result.stderr
    executed = json.loads(result.stdout)
    assert Path(executed["prefix"]).resolve() == environment.resolve()
    assert executed["value"] == "VENV-DEPENDENCY-FOUND"


def test_verified_python_rejects_symlink_launcher_bound_to_another_target(tmp_path):
    launcher = tmp_path / ("python.exe" if os.name == "nt" else "python")
    try:
        launcher.symlink_to(Path(sys.executable).resolve(strict=True))
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create an interpreter symlink on this platform: {exc}")
    wrong_target = tmp_path / "not-the-interpreter"
    wrong_target.write_bytes(b"different regular file")
    worker_root = tmp_path / "worker"
    worker_root.mkdir()
    script = worker_root / "worker.py"
    script.write_text("print('must not run')\n", encoding="utf-8")
    result = run_verified_python(
        CommandSpec(argv=(str(launcher), str(script)), cwd=str(worker_root)),
        expected_interpreter_sha256=sha256_file(wrong_target),
        expected_interpreter_target_path=str(wrong_target.resolve()),
        expected_script_sha256=sha256_file(script),
        import_root=str(worker_root.resolve()),
        expected_import_root_receipt={
            "path": str(worker_root.resolve()),
            **hash_path(worker_root, symlink_policy="reject").to_dict(),
        },
    )
    assert not result.process_started
    assert result.error_code is EvaluationErrorCode.RUNTIME_ERROR
    assert "launcher differs from the frozen target" in result.stderr


def test_verified_python_rejects_source_changed_before_verified_read(tmp_path):
    interpreter = str(Path(sys.executable).resolve(strict=True))
    script = tmp_path / "worker.py"
    script.write_text("print('frozen')\n", encoding="utf-8")
    frozen_sha256 = sha256_file(script)
    frozen_import_root = {
        "path": str(tmp_path.resolve()),
        **hash_path(tmp_path, symlink_policy="reject").to_dict(),
    }
    script.write_text("print('changed')\n", encoding="utf-8")
    result = run_verified_python(
        CommandSpec(argv=(interpreter, str(script.resolve())), cwd=str(tmp_path)),
        expected_interpreter_sha256=sha256_file(interpreter),
        expected_script_sha256=frozen_sha256,
        import_root=str(tmp_path.resolve()),
        expected_import_root_receipt=frozen_import_root,
    )
    assert not result.process_started
    assert result.error_code is EvaluationErrorCode.RUNTIME_ERROR
    assert "pre-exec integrity failure" in result.stderr


def test_verified_python_path_swap_at_spawn_cannot_change_executed_bytes(
    tmp_path, monkeypatch
):
    interpreter = str(Path(sys.executable).resolve(strict=True))
    script = tmp_path / "worker.py"
    original_source = b"print('FROZEN-BYTES-RAN')\n"
    script.write_bytes(original_source)
    original_run = process_module.subprocess.run

    def swap_path_then_spawn(argv, **kwargs):
        # run_verified_python has already captured and verified stdin bytes.
        script.write_text("print('MALICIOUS-PATH-RAN')\n", encoding="utf-8")
        try:
            return original_run(argv, **kwargs)
        finally:
            script.write_bytes(original_source)

    monkeypatch.setattr(process_module.subprocess, "run", swap_path_then_spawn)
    import_root_receipt = {
        "path": str(tmp_path.resolve()),
        **hash_path(tmp_path, symlink_policy="reject").to_dict(),
    }
    result = run_verified_python(
        CommandSpec(argv=(interpreter, str(script.resolve())), cwd=str(tmp_path)),
        expected_interpreter_sha256=sha256_file(interpreter),
        expected_script_sha256=sha256_file(script),
        import_root=str(tmp_path.resolve()),
        expected_import_root_receipt=import_root_receipt,
    )
    assert result.succeeded
    assert "FROZEN-BYTES-RAN" in result.stdout
    assert "MALICIOUS-PATH-RAN" not in result.stdout


def test_verified_python_transitive_import_swap_executes_captured_bundle(
    tmp_path, monkeypatch
):
    interpreter = str(Path(sys.executable).resolve(strict=True))
    script = tmp_path / "worker.py"
    helper = tmp_path / "helper.py"
    marker = tmp_path / "transitive_attack_marker.txt"
    script.write_text("import helper\nprint(helper.VALUE)\n", encoding="utf-8")
    original_helper = b"VALUE = 'SAFE-HELPER'\n"
    helper.write_bytes(original_helper)
    frozen_root = {
        "path": str(tmp_path.resolve()),
        **hash_path(tmp_path, symlink_policy="reject").to_dict(),
    }
    original_run = process_module.subprocess.run

    def swap_helper_then_spawn(argv, **kwargs):
        assert argv[1:3] == ["-I", "-c"]
        helper.write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('ATTACKED')\n"
            "VALUE = 'MALICIOUS-HELPER'\n",
            encoding="utf-8",
        )
        try:
            return original_run(argv, **kwargs)
        finally:
            helper.write_bytes(original_helper)

    monkeypatch.setattr(process_module.subprocess, "run", swap_helper_then_spawn)
    result = run_verified_python(
        CommandSpec(argv=(interpreter, str(script.resolve())), cwd=str(tmp_path)),
        expected_interpreter_sha256=sha256_file(interpreter),
        expected_script_sha256=sha256_file(script),
        import_root=str(tmp_path.resolve()),
        expected_import_root_receipt=frozen_root,
    )
    assert result.succeeded
    assert "SAFE-HELPER" in result.stdout
    assert "MALICIOUS-HELPER" not in result.stdout
    assert not marker.exists()
    assert not (tmp_path / "__pycache__").exists()
    assert hash_path(tmp_path, symlink_policy="reject").digest == frozen_root["digest"]


def test_verified_python_does_not_import_module_added_after_bundle_capture(
    tmp_path, monkeypatch
):
    interpreter = str(Path(sys.executable).resolve(strict=True))
    script = tmp_path / "worker.py"
    late_helper = tmp_path / "late_helper.py"
    marker = tmp_path / "late_module_marker.txt"
    script.write_text("import late_helper\n", encoding="utf-8")
    frozen_root = {
        "path": str(tmp_path.resolve()),
        **hash_path(tmp_path, symlink_policy="reject").to_dict(),
    }
    original_run = process_module.subprocess.run

    def add_module_then_spawn(argv, **kwargs):
        late_helper.write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('EXECUTED')\n",
            encoding="utf-8",
        )
        try:
            return original_run(argv, **kwargs)
        finally:
            late_helper.unlink(missing_ok=True)

    monkeypatch.setattr(process_module.subprocess, "run", add_module_then_spawn)
    result = run_verified_python(
        CommandSpec(argv=(interpreter, str(script.resolve())), cwd=str(tmp_path)),
        expected_interpreter_sha256=sha256_file(interpreter),
        expected_script_sha256=sha256_file(script),
        import_root=str(tmp_path.resolve()),
        expected_import_root_receipt=frozen_root,
    )
    assert not result.succeeded
    assert "late_helper" in result.stderr
    assert not marker.exists()


def test_verified_python_package_path_cannot_fall_back_to_mutable_tree(
    tmp_path, monkeypatch
):
    interpreter = str(Path(sys.executable).resolve(strict=True))
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'FROZEN-PACKAGE'\n", encoding="utf-8")
    script = tmp_path / "worker.py"
    script.write_text("import pkg.late_added\n", encoding="utf-8")
    late_child = package / "late_added.py"
    marker = tmp_path / "package_fallback_marker.txt"
    frozen_root = {
        "path": str(tmp_path.resolve()),
        **hash_path(tmp_path, symlink_policy="reject").to_dict(),
    }
    original_run = process_module.subprocess.run

    def add_package_child_then_spawn(argv, **kwargs):
        late_child.write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('EXECUTED')\n",
            encoding="utf-8",
        )
        try:
            return original_run(argv, **kwargs)
        finally:
            late_child.unlink(missing_ok=True)

    monkeypatch.setattr(
        process_module.subprocess, "run", add_package_child_then_spawn
    )
    result = run_verified_python(
        CommandSpec(argv=(interpreter, str(script.resolve())), cwd=str(tmp_path)),
        expected_interpreter_sha256=sha256_file(interpreter),
        expected_script_sha256=sha256_file(script),
        import_root=str(tmp_path.resolve()),
        expected_import_root_receipt=frozen_root,
    )
    assert not result.succeeded
    assert "late_added" in result.stderr
    assert not marker.exists()
    assert not (package / "__pycache__").exists()


def test_verified_python_imports_frozen_package_child_without_bytecode(tmp_path):
    interpreter = str(Path(sys.executable).resolve(strict=True))
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "child.py").write_text("VALUE = 'FROZEN-CHILD'\n", encoding="utf-8")
    script = tmp_path / "worker.py"
    script.write_text("from pkg import child\nprint(child.VALUE)\n", encoding="utf-8")
    frozen_root = {
        "path": str(tmp_path.resolve()),
        **hash_path(tmp_path, symlink_policy="reject").to_dict(),
    }
    result = run_verified_python(
        CommandSpec(argv=(interpreter, str(script.resolve())), cwd=str(tmp_path)),
        expected_interpreter_sha256=sha256_file(interpreter),
        expected_script_sha256=sha256_file(script),
        import_root=str(tmp_path.resolve()),
        expected_import_root_receipt=frozen_root,
    )
    assert result.succeeded
    assert "FROZEN-CHILD" in result.stdout
    assert not (package / "__pycache__").exists()


class FailingProbe:
    def query(self):
        raise GPUQueryError("query unavailable")


def test_gpu_query_failure_never_counts_as_idle(tmp_path):
    store = KeepaliveStore(
        tmp_path,
        project_owner="motionllm-test",
        probe=FailingProbe(),
        pid_alive=lambda _pid: True,
    )
    with pytest.raises(GPUQueryError):
        store.register(gpu_uuid="GPU-1", pid=123, command_fingerprint="abc")
    assert store.status() == []


class IdleProbe:
    def query(self):
        return GPUInventory(
            devices=(GPUDevice(0, "GPU-1", "test", 1000, 0, 0),),
            processes=(),
        )


def test_gpu_lease_records_owner_pid_uuid_and_refuses_other_owner(tmp_path):
    store = GPULeaseStore(
        tmp_path,
        project_owner="motionllm-test",
        probe=IdleProbe(),
        pid_alive=lambda pid: pid == 123,
    )
    lease = store.acquire("GPU-1", pid=123, purpose="finetune:qwen")
    assert lease["gpu_uuid"] == "GPU-1"
    assert lease["pid"] == 123
    assert lease["owner"] == "motionllm-test"
    other = GPULeaseStore(
        tmp_path,
        project_owner="other-project",
        probe=IdleProbe(),
        pid_alive=lambda _pid: True,
    )
    with pytest.raises(Exception, match="refusing"):
        other.release("GPU-1", lease_id=lease["lease_id"], pid=123)
    store.release("GPU-1", lease_id=lease["lease_id"], pid=123)


def test_pinned_ssh_has_no_trust_on_first_use(tmp_path):
    known = tmp_path / "known_hosts"
    known.write_text("[example.invalid]:22 ssh-ed25519 AAAATEST\n", encoding="utf-8")
    target = PinnedSshTarget("example.invalid", 22, "runner", str(known))
    spec = target.command(("/bin/true",))
    joined = " ".join(spec.argv)
    assert "StrictHostKeyChecking=yes" in joined
    assert f"UserKnownHostsFile={known.resolve()}" in joined
    assert "PasswordAuthentication=no" in joined
    assert "AutoAddPolicy" not in joined


def test_pinned_ssh_requires_existing_nonempty_known_hosts(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        PinnedSshTarget("example.invalid", 22, "runner", str(tmp_path / "missing"))
