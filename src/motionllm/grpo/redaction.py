"""Secret-free logging and manifest guards."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_SENSITIVE_KEY = re.compile(
    r"(?:password|passwd|passphrase|secret|token|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|authorization|cookie|session)",
    flags=re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+[A-Za-z0-9._~+/=-]+)",
    flags=re.IGNORECASE,
)
# Any URI userinfo is credential-like.  This deliberately catches both
# ``user:password@host`` and token-only forms such as ``token@host``.
_URI_USERINFO = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^/@\s]+@")
_EMBEDDED_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?:password|passwd|passphrase|secret|token|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|authorization)\s*[\"']?\s*[:=]",
    flags=re.IGNORECASE,
)


class SecretMaterialError(ValueError):
    pass


def is_sensitive_key(key: Any) -> bool:
    return bool(_SENSITIVE_KEY.search(str(key)))


def redact_mapping_for_log(value: Mapping[str, Any]) -> dict[str, Any]:
    def redact_value(item: Any) -> Any:
        if isinstance(item, Mapping):
            return redact_mapping_for_log(item)
        if isinstance(item, (list, tuple)):
            return [redact_value(element) for element in item]
        if isinstance(item, str) and (
            _SECRET_VALUE.search(item)
            or _URI_USERINFO.search(item)
            or _EMBEDDED_SENSITIVE_ASSIGNMENT.search(item)
        ):
            return "<redacted>"
        if isinstance(item, str) or item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, BaseException):
            return f"<{type(item).__name__}>"
        # Never call arbitrary __str__/__repr__ implementations in a log sanitizer.
        return "<object>"

    result: dict[str, Any] = {}
    for key, item in value.items():
        name = key if isinstance(key, str) else f"<{type(key).__name__}>"
        if is_sensitive_key(name):
            result[name] = "<redacted>"
        else:
            result[name] = redact_value(item)
    return result


def describe_environment_overrides(value: Mapping[str, Any]) -> dict[str, str]:
    """Describe override keys without ever returning an environment value."""

    return {
        str(key): "<unset>" if item is None else ("<redacted>" if is_sensitive_key(key) else "<set>")
        for key, item in value.items()
    }


def redact_command_for_log(command: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    redact_next = False
    for raw in command:
        item = raw if isinstance(raw, str) else "<object>"
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        if item.startswith("--"):
            key, separator, _ = item.partition("=")
            if is_sensitive_key(key):
                if separator:
                    result.append(f"{key}=<redacted>")
                else:
                    result.append(key)
                    redact_next = True
                continue
        if (
            _SECRET_VALUE.search(item)
            or _URI_USERINFO.search(item)
            or _EMBEDDED_SENSITIVE_ASSIGNMENT.search(item)
        ):
            result.append("<redacted>")
            continue
        result.append(item)
    return tuple(result)


def assert_manifest_secret_free(value: Any, *, path: str = "$") -> None:
    """Reject credential-bearing keys or obvious private material in manifests."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            if is_sensitive_key(name):
                raise SecretMaterialError(f"credential-like manifest key is forbidden at {path}.{name}")
            assert_manifest_secret_free(item, path=f"{path}.{name}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_manifest_secret_free(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and (
        _SECRET_VALUE.search(value)
        or _URI_USERINFO.search(value)
        or _EMBEDDED_SENSITIVE_ASSIGNMENT.search(value)
    ):
        raise SecretMaterialError(f"credential-like manifest value is forbidden at {path}")
    if not (
        value is None
        or isinstance(value, (str, bool, int, float))
    ):
        raise SecretMaterialError(f"unsupported manifest value type is forbidden at {path}")
