"""Shell-free subprocess execution with bounded environment and redaction."""

from __future__ import annotations

import os
import re
import subprocess
import time
import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from motion_eval.contracts import EvaluationErrorCode
from motion_eval.core import capture_directory_bytes

ALLOWED_ENV_KEYS = frozenset(
    {
        "PATH",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "NCCL_DEBUG",
        "WANDB_MODE",
        "WANDB_API_KEY",
        "HF_TOKEN",
    }
)
_SECRET_ENV_KEYS = frozenset({"WANDB_API_KEY", "HF_TOKEN"})
_SECRET_FLAG_NAMES = frozenset(
    {
        "password", "passwd", "secret", "token", "apikey", "privatekey",
        "authorization", "credential", "hftoken", "wandbapikey",
    }
)
_SECRET_ASSIGNMENT = re.compile(
    r"\A(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|authorization|credential)\s*=",
    re.IGNORECASE,
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class CommandValidationError(ValueError):
    pass


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if not argv:
        raise CommandValidationError("argv cannot be empty")
    result: list[str] = []
    expects_secret = False
    for index, value in enumerate(argv):
        if not isinstance(value, str) or not value or _CONTROL.search(value):
            raise CommandValidationError(f"argv[{index}] must be a non-empty control-free string")
        lowered = value.lower()
        if expects_secret:
            raise CommandValidationError("secret values must be passed via process environment, not argv")
        flag_name = re.sub(r"[-_]", "", value.split("=", 1)[0].lstrip("-").lower())
        if value.startswith("--") and flag_name in _SECRET_FLAG_NAMES:
            if "=" in value:
                raise CommandValidationError("secret values must not be embedded in argv")
            expects_secret = True
        if re.search(r"[A-Za-z][A-Za-z0-9+.-]*://[^/@\s]+:[^/@\s]+@", value):
            raise CommandValidationError("credentials embedded in a URI are forbidden")
        if _SECRET_ASSIGNMENT.search(value):
            raise CommandValidationError("secret-like key/value is forbidden in argv")
        result.append(value)
    if expects_secret:
        raise CommandValidationError("secret command-line flags are forbidden")
    return tuple(result)


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...] | Sequence[str]
    cwd: str
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 3600.0
    label: str = "worker"

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", _validate_argv(self.argv))
        if not isinstance(self.cwd, str) or not self.cwd or _CONTROL.search(self.cwd):
            raise CommandValidationError("cwd must be a non-empty control-free string")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise CommandValidationError("timeout_seconds must be positive")
        safe_env: dict[str, str] = {}
        for key, value in self.env.items():
            if key not in ALLOWED_ENV_KEYS:
                raise CommandValidationError(f"environment key is not allowlisted: {key}")
            if not isinstance(value, str) or _CONTROL.search(value):
                raise CommandValidationError(f"environment value for {key} is invalid")
            safe_env[key] = value
        object.__setattr__(self, "env", MappingProxyType(safe_env))
        if not isinstance(self.label, str) or not self.label.strip():
            raise CommandValidationError("label must be non-empty")

    def receipt(self) -> dict[str, object]:
        environment: dict[str, object] = {}
        for key, value in self.env.items():
            environment[key] = {"present": True, "redacted": True} if key in _SECRET_ENV_KEYS else value
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": environment,
            "timeout_seconds": self.timeout_seconds,
            "label": self.label,
            "shell": False,
        }

    def preview(self) -> str:
        return " ".join(_quote_preview(item) for item in self.argv)


def _quote_preview(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:\\=-]+", value):
        return value
    return repr(value)


def redact_text(text: str, env: Mapping[str, str]) -> str:
    result = text
    for key, value in env.items():
        if key in _SECRET_ENV_KEYS and value:
            result = result.replace(value, "<redacted>")
        result = re.sub(
            rf"(?i)({re.escape(key)}\s*[=:]\s*)[^\s,;]+",
            rf"\1<redacted>" if key in _SECRET_ENV_KEYS else rf"\1<value>",
            result,
        )
    result = re.sub(
        r"(?i)((?:password|passwd|secret|token|api[_-]?key|authorization)\s*[=:]\s*)[^\s,;]+",
        r"\1<redacted>",
        result,
    )
    return result


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    error_code: EvaluationErrorCode
    timed_out: bool = False
    dry_run: bool = False
    process_started: bool = True

    @property
    def succeeded(self) -> bool:
        return self.error_code is EvaluationErrorCode.NONE and self.returncode == 0


def _classify(returncode: int, stderr: str) -> EvaluationErrorCode:
    if returncode == 0:
        return EvaluationErrorCode.NONE
    lowered = stderr.lower()
    if "out of memory" in lowered or "cuda oom" in lowered or returncode in {137, -9}:
        return EvaluationErrorCode.OOM
    return EvaluationErrorCode.RUNTIME_ERROR


_VERIFIED_PYTHON_BOOTSTRAP = """\
import hashlib
import importlib.abc
import importlib.util
import io
import os
import sys
import zipfile

_motion_eval_payload = sys.stdin.buffer.read()
if len(_motion_eval_payload) < 8:
    raise RuntimeError("verified Python payload is truncated")
_motion_eval_source_size = int.from_bytes(_motion_eval_payload[:8], "big")
_motion_eval_source_end = 8 + _motion_eval_source_size
if _motion_eval_source_end > len(_motion_eval_payload):
    raise RuntimeError("verified Python source frame is invalid")
_motion_eval_source = _motion_eval_payload[8:_motion_eval_source_end]
_motion_eval_bundle = _motion_eval_payload[_motion_eval_source_end:]
_motion_eval_file = sys.argv[1]
_motion_eval_import_root = os.path.abspath(sys.argv[2])
_motion_eval_bundle_sha256 = sys.argv[3]
if hashlib.sha256(_motion_eval_bundle).hexdigest() != _motion_eval_bundle_sha256:
    raise RuntimeError("verified Python import bundle digest mismatch")
sys.argv = [_motion_eval_file, *sys.argv[4:]]
sys.dont_write_bytecode = True


class _MotionEvalBundleImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, payload, original_root, bundle_sha256):
        self._archive = zipfile.ZipFile(io.BytesIO(payload), "r")
        self._names = frozenset(self._archive.namelist())
        self._root = original_root
        self._root_package = os.path.basename(original_root.rstrip(os.sep))
        self._virtual_root = f"<motion-eval-verified:{bundle_sha256}>"
        for name in self._names:
            parts = name.split("/")
            if not name or name.startswith("/") or "\\\\" in name or any(
                part in {"", ".", ".."} for part in parts
            ):
                raise RuntimeError("verified Python import bundle contains an unsafe path")

    def _module_base(self, fullname):
        if fullname == self._root_package:
            return ""
        prefix = self._root_package + "."
        if fullname.startswith(prefix):
            fullname = fullname[len(prefix):]
        return fullname.replace(".", "/")

    def _lookup(self, fullname):
        base = self._module_base(fullname)
        package = (base + "/__init__.py") if base else "__init__.py"
        module = (base + ".py") if base else None
        if package in self._names:
            return package, True
        if module is not None and module in self._names:
            return module, False
        return None

    def find_spec(self, fullname, path=None, target=None):
        located = self._lookup(fullname)
        if located is not None:
            return importlib.util.spec_from_loader(
                fullname, self, is_package=located[1]
            )
        base = self._module_base(fullname)
        prefix = (base + "/") if base else ""
        if (
            (fullname == self._root_package and bool(self._names))
            or (prefix and any(name.startswith(prefix) for name in self._names))
        ):
            spec = importlib.util.spec_from_loader(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [
                self._virtual_root + "/" + base
            ]
            return spec
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        located = self._lookup(module.__name__)
        if located is None:
            raise ImportError(module.__name__)
        relative, is_package = located
        filename = os.path.join(self._root, *relative.split("/"))
        module.__file__ = filename
        if is_package:
            package_relative = os.path.dirname(relative).replace("\\\\", "/")
            module.__path__ = [
                self._virtual_root
                + (("/" + package_relative) if package_relative else "")
            ]
        source = self._archive.read(relative)
        exec(compile(source, filename, "exec"), module.__dict__)

    def get_filename(self, fullname):
        located = self._lookup(fullname)
        if located is None:
            raise ImportError(fullname)
        return os.path.join(self._root, *located[0].split("/"))

    def get_data(self, path):
        absolute = os.path.abspath(path)
        try:
            relative = os.path.relpath(absolute, self._root)
        except ValueError as exc:
            raise OSError(path) from exc
        normalized = relative.replace(os.sep, "/")
        if normalized.startswith("../") or normalized not in self._names:
            raise OSError(path)
        return self._archive.read(normalized)


_motion_eval_importer = _MotionEvalBundleImporter(
    _motion_eval_bundle, _motion_eval_import_root, _motion_eval_bundle_sha256
)
sys.meta_path.insert(0, _motion_eval_importer)
_motion_eval_virtual_path = _motion_eval_importer._virtual_root
_motion_eval_clean_path = []
for _motion_eval_entry in sys.path:
    if _motion_eval_entry == "":
        continue
    _motion_eval_resolved = os.path.abspath(_motion_eval_entry or os.getcwd())
    if os.path.normcase(_motion_eval_resolved) != os.path.normcase(_motion_eval_import_root):
        _motion_eval_clean_path.append(_motion_eval_entry)
sys.path[:] = [_motion_eval_virtual_path, *_motion_eval_clean_path]
_motion_eval_globals = {
    "__name__": "__main__",
    "__file__": _motion_eval_file,
    "__package__": None,
    "__cached__": None,
    "__loader__": None,
    "__spec__": None,
}
exec(compile(_motion_eval_source, _motion_eval_file, "exec"), _motion_eval_globals)
"""


def _verified_import_bundle(
    import_root: str,
    expected_receipt: Mapping[str, object],
) -> tuple[bytes, str]:
    required = {"path", "algorithm", "kind", "digest", "file_count", "total_bytes"}
    if set(expected_receipt) != required:
        raise CommandValidationError("frozen Python import-root receipt is invalid")
    candidate = Path(import_root).resolve(strict=True)
    if str(candidate) != expected_receipt.get("path"):
        raise CommandValidationError("Python import root differs from its frozen path")
    capture = capture_directory_bytes(candidate)
    if capture.receipt.to_dict() != {
        key: expected_receipt[key] for key in required if key != "path"
    }:
        raise CommandValidationError("Python import root differs from its frozen receipt")
    casefolded: set[str] = set()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for relative, payload in capture.files:
            normalized = relative.replace("\\", "/")
            folded = normalized.casefold()
            if folded in casefolded:
                raise CommandValidationError(
                    "Python import bundle has case-insensitive duplicate paths"
                )
            casefolded.add(folded)
            info = zipfile.ZipInfo(normalized, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def _decode_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


def _read_verified_regular_file(path: str, expected_sha256: str, *, label: str) -> bytes:
    """Read one exact regular-file generation and verify its frozen digest.

    The bytes returned by this function, rather than the path, are the trust
    boundary for Python source execution.  File identity is checked before and
    after the read so a concurrent replace is either detected or yields a
    complete generation whose digest must still match.
    """

    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise CommandValidationError(f"{label} frozen sha256 is invalid")
    candidate = Path(path)
    if candidate.is_symlink():
        raise CommandValidationError(f"{label} symlinks are forbidden")
    try:
        before = candidate.stat()
        with candidate.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            payload = handle.read()
            after_open = os.fstat(handle.fileno())
        after_path = candidate.stat()
    except OSError as exc:
        raise CommandValidationError(f"{label} cannot be read as a frozen regular file") from exc
    if not candidate.is_file():
        raise CommandValidationError(f"{label} must be a regular file")
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after_open):
        raise CommandValidationError(f"{label} changed while it was read")
    if identity(after_open) != identity(after_path):
        raise CommandValidationError(f"{label} path changed while it was read")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise CommandValidationError(f"{label} differs from the frozen sha256")
    return payload


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.fspath(left)) == os.path.normcase(os.fspath(right))


def _read_verified_interpreter(
    launcher: str,
    expected_sha256: str,
    *,
    expected_target_path: str | None,
) -> bytes:
    """Verify a launcher binding while hashing its explicit regular-file target.

    A POSIX virtual environment commonly exposes ``bin/python`` as a symlink.
    Executing the resolved base interpreter path changes Python's venv discovery,
    so the launcher remains in ``argv[0]``.  When a target is explicitly frozen,
    both pre- and post-read resolution must still name that exact canonical file;
    the target itself receives the same stable-stat and SHA-256 checks as before.
    """

    if expected_target_path is None:
        return _read_verified_regular_file(
            launcher, expected_sha256, label="Python interpreter"
        )
    launcher_path = Path(launcher)
    target_path = Path(expected_target_path)
    if not launcher_path.is_absolute() or not target_path.is_absolute():
        raise CommandValidationError(
            "Python interpreter launcher and frozen target must be absolute paths"
        )
    try:
        canonical_target = target_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CommandValidationError(
            "Python interpreter frozen target cannot be resolved"
        ) from exc
    if not _same_path(canonical_target, target_path):
        raise CommandValidationError(
            "Python interpreter frozen target must be a canonical regular-file path"
        )

    def verify_launcher_binding() -> None:
        try:
            resolved = launcher_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CommandValidationError(
                "Python interpreter launcher cannot be resolved"
            ) from exc
        if not _same_path(resolved, target_path):
            raise CommandValidationError(
                "Python interpreter launcher differs from the frozen target"
            )

    verify_launcher_binding()
    payload = _read_verified_regular_file(
        str(target_path), expected_sha256, label="Python interpreter target"
    )
    verify_launcher_binding()
    return payload


def _run_subprocess(
    spec: CommandSpec,
    *,
    argv: Sequence[str],
    stdin_bytes: bytes | None = None,
) -> ProcessResult:
    child_env: dict[str, str] = {}
    for key in ALLOWED_ENV_KEYS - _SECRET_ENV_KEYS:
        if key in os.environ:
            child_env[key] = os.environ[key]
    child_env.update(spec.env)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=spec.cwd,
            env=child_env,
            shell=False,
            capture_output=True,
            text=stdin_bytes is None,
            encoding="utf-8" if stdin_bytes is None else None,
            errors="replace" if stdin_bytes is None else None,
            input=stdin_bytes,
            timeout=spec.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_output(exc.stdout)
        stderr = _decode_output(exc.stderr)
        return ProcessResult(
            None,
            redact_text(stdout, spec.env),
            redact_text(stderr, spec.env),
            time.monotonic() - started,
            EvaluationErrorCode.TIMEOUT,
            timed_out=True,
            process_started=True,
        )
    except OSError as exc:
        return ProcessResult(
            None,
            "",
            redact_text(str(exc), spec.env),
            time.monotonic() - started,
            EvaluationErrorCode.RUNTIME_ERROR,
            process_started=False,
        )
    stdout = redact_text(_decode_output(completed.stdout), spec.env)
    stderr = redact_text(_decode_output(completed.stderr), spec.env)
    return ProcessResult(
        completed.returncode,
        stdout,
        stderr,
        time.monotonic() - started,
        _classify(completed.returncode, stderr),
    )


def run_command(spec: CommandSpec, *, dry_run: bool = False) -> ProcessResult:
    """Execute exactly ``argv`` with ``shell=False``; never inherit arbitrary env."""

    if dry_run:
        return ProcessResult(
            0,
            spec.preview(),
            "",
            0.0,
            EvaluationErrorCode.NONE,
            dry_run=True,
            process_started=False,
        )
    return _run_subprocess(spec, argv=spec.argv)


def run_verified_python(
    spec: CommandSpec,
    *,
    expected_interpreter_sha256: str,
    expected_interpreter_target_path: str | None = None,
    expected_script_sha256: str,
    import_root: str,
    expected_import_root_receipt: Mapping[str, object],
    dry_run: bool = False,
) -> ProcessResult:
    """Run the frozen Python source bytes, never a hash-then-open script path.

    ``spec.argv`` remains the canonical/auditable command.  The child receives
    both the verified top-level source and a stable, Merkle-checked import-tree
    bundle over stdin.  A fixed in-memory importer restores ``sys.argv`` and
    ``__file__`` without falling back to the mutable runner directory or writing
    bytecode.  The interpreter executable is digest-checked immediately before
    and after the child; protection against a malicious process with the same OS
    identity replacing and restoring that executable inside the final spawn
    window still requires OS-level ACL/isolation.
    """

    if len(spec.argv) < 2:
        raise CommandValidationError("verified Python command requires interpreter and script")
    if dry_run:
        return run_command(spec, dry_run=True)
    interpreter, script = spec.argv[:2]
    try:
        _read_verified_interpreter(
            interpreter,
            expected_interpreter_sha256,
            expected_target_path=expected_interpreter_target_path,
        )
        source = _read_verified_regular_file(
            script, expected_script_sha256, label="Python runner"
        )
        resolved_script = Path(script).resolve(strict=True)
        resolved_import_root = Path(import_root).resolve(strict=True)
        try:
            resolved_script.relative_to(resolved_import_root)
        except ValueError as exc:
            raise CommandValidationError(
                "Python runner must be inside the frozen import root"
            ) from exc
        import_bundle, import_bundle_sha256 = _verified_import_bundle(
            str(resolved_import_root), expected_import_root_receipt
        )
    except CommandValidationError as exc:
        return ProcessResult(
            None,
            "",
            redact_text(f"pre-exec integrity failure: {exc}", spec.env),
            0.0,
            EvaluationErrorCode.RUNTIME_ERROR,
            process_started=False,
        )
    # Isolated mode prevents ambient PYTHONPATH, the mutable working directory,
    # and user-site packages from running ``sitecustomize``/``usercustomize``
    # before the verified in-memory bootstrap gains control.
    runtime_argv = (
        interpreter,
        "-I",
        "-c",
        _VERIFIED_PYTHON_BOOTSTRAP,
        script,
        str(resolved_import_root),
        import_bundle_sha256,
        *spec.argv[2:],
    )
    framed_payload = len(source).to_bytes(8, "big") + source + import_bundle
    try:
        # Bundle capture may be comparatively expensive.  Recheck immediately
        # before spawn so the original launcher is still bound to the frozen,
        # content-verified target at the execution boundary.
        _read_verified_interpreter(
            interpreter,
            expected_interpreter_sha256,
            expected_target_path=expected_interpreter_target_path,
        )
    except CommandValidationError as exc:
        return ProcessResult(
            None,
            "",
            redact_text(f"pre-exec integrity failure: {exc}", spec.env),
            0.0,
            EvaluationErrorCode.RUNTIME_ERROR,
            process_started=False,
        )
    result = _run_subprocess(spec, argv=runtime_argv, stdin_bytes=framed_payload)
    try:
        _read_verified_interpreter(
            interpreter,
            expected_interpreter_sha256,
            expected_target_path=expected_interpreter_target_path,
        )
    except CommandValidationError as exc:
        return ProcessResult(
            result.returncode,
            result.stdout,
            redact_text(f"{result.stderr}\npost-exec integrity failure: {exc}".strip(), spec.env),
            result.duration_seconds,
            EvaluationErrorCode.RUNTIME_ERROR,
            timed_out=result.timed_out,
            dry_run=result.dry_run,
            process_started=result.process_started,
        )
    return result
