"""Fail-closed JSON/HTTPS client for online Rubric-RL judges."""

from __future__ import annotations

import http.client
import ipaddress
import json
import queue
import secrets
import socket
import ssl
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Mapping

from .motion_rubric_v2 import (
    MOTION_RUBRIC_V2_VERSION,
    validate_motion_criteria_v2,
    validate_motion_judgment_v2,
)
from .qa_rubric import QA_RUBRIC_VERSION, validate_qa_criteria, validate_qa_judgment
from .rubric_common import (
    RubricValidationError,
    build_judgment_binding,
    finite_number,
    require_exact_keys,
    require_mapping,
    strict_json_object,
)


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise RubricValidationError("online judge exceeded its total deadline")
    return value


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if not address.is_global or any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_reserved,
            address.is_unspecified,
        )
    ):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return _is_public_address(address.ipv4_mapped)
        if address.sixtofour is not None or address.teredo is not None:
            return False
    return True


def _resolve_public_addresses(host: str, port: int, deadline: float) -> tuple[str, ...]:
    """Resolve once under the deadline and reject every non-global answer."""

    result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            result_queue.put(
                socket.getaddrinfo(
                    host,
                    port,
                    type=socket.SOCK_STREAM,
                    proto=socket.IPPROTO_TCP,
                )
            )
        except BaseException as exc:  # delivered without rendering host details
            result_queue.put(exc)

    threading.Thread(target=resolve, daemon=True).start()
    try:
        result = result_queue.get(timeout=_remaining(deadline))
    except queue.Empty as exc:
        raise RubricValidationError("online judge DNS resolution timed out") from exc
    if isinstance(result, BaseException):
        raise RubricValidationError(
            f"online judge DNS resolution failed ({type(result).__name__})"
        ) from result
    addresses: set[str] = set()
    for entry in result:
        address = str(entry[4][0]).split("%", 1)[0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise RubricValidationError("online judge DNS returned an invalid address") from exc
        if not _is_public_address(parsed):
            raise RubricValidationError(
                "online judge DNS resolved to a private, loopback, link-local, or special address"
            )
        addresses.add(parsed.compressed)
    if not addresses:
        raise RubricValidationError("online judge DNS returned no usable addresses")
    return tuple(sorted(addresses))


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to an audited IP while retaining hostname TLS verification."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_ip: str,
        deadline: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            timeout=_remaining(deadline),
            context=context,
        )
        self._connect_ip = connect_ip
        self._deadline = deadline

    def connect(self) -> None:
        parsed_ip = ipaddress.ip_address(self._connect_ip)
        family = socket.AF_INET6 if parsed_ip.version == 6 else socket.AF_INET
        raw = socket.socket(family, socket.SOCK_STREAM)
        try:
            # Publish the in-flight raw socket before connect/TLS so the
            # absolute-deadline watchdog can close and unblock either phase.
            self.sock = raw
            raw.settimeout(_remaining(self._deadline))
            if self.source_address is not None:
                raw.bind(self.source_address)
            destination: Any = (
                (self._connect_ip, self.port, 0, 0)
                if parsed_ip.version == 6
                else (self._connect_ip, self.port)
            )
            raw.connect(destination)
            # TCP connection and TLS negotiation are separate blocking phases.
            # Recompute the remaining budget so they cannot each consume the
            # original timeout and exceed the one wall-clock deadline.
            raw.settimeout(_remaining(self._deadline))
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
            self.sock.settimeout(_remaining(self._deadline))
        except BaseException:
            raw.close()
            raise


def _direct_https_post(
    endpoint: str,
    payload: bytes,
    headers: Mapping[str, str],
    *,
    timeout_seconds: float,
    max_response_bytes: int,
) -> bytes:
    """POST without proxies or redirects, pinned to one audited DNS result."""

    deadline = time.monotonic() + timeout_seconds
    parsed = urllib.parse.urlsplit(endpoint)
    assert parsed.hostname is not None  # validated by OnlineJudgeConfig
    port = parsed.port or 443
    addresses = _resolve_public_addresses(parsed.hostname, port, deadline)
    target = parsed.path or "/"
    context = ssl.create_default_context()
    last_error: BaseException | None = None
    for address in addresses:
        connection = _PinnedHTTPSConnection(
            parsed.hostname,
            port,
            connect_ip=address,
            deadline=deadline,
            context=context,
        )
        watchdog = threading.Timer(_remaining(deadline), connection.close)
        watchdog.daemon = True
        watchdog.start()
        try:
            # Connect explicitly.  ``HTTPConnection.request`` otherwise folds
            # TCP/TLS setup and upload into one call while reusing the initial
            # socket timeout for every phase.
            connection.connect()
            if connection.sock is not None:
                connection.sock.settimeout(_remaining(deadline))
            connection.request("POST", target, body=payload, headers=dict(headers))
            if connection.sock is not None:
                connection.sock.settimeout(_remaining(deadline))
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise RubricValidationError("online judge redirects are forbidden")
            if not 200 <= response.status < 300:
                raise RubricValidationError(
                    f"online judge returned HTTP status {response.status}"
                )
            if response.headers.get_content_type() != "application/json":
                raise RubricValidationError(
                    "online judge response must be application/json"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                if connection.sock is not None:
                    connection.sock.settimeout(_remaining(deadline))
                chunk = response.read(min(65_536, max_response_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_response_bytes:
                    raise RubricValidationError(
                        "online judge response exceeds the size limit"
                    )
            _remaining(deadline)
            return b"".join(chunks)
        except RubricValidationError:
            raise
        except (OSError, TimeoutError, ssl.SSLError, http.client.HTTPException) as exc:
            last_error = exc
        finally:
            watchdog.cancel()
            connection.close()
    assert last_error is not None
    raise RubricValidationError(
        f"online judge request failed ({type(last_error).__name__})"
    ) from last_error


@dataclass(frozen=True)
class OnlineJudgeConfig:
    endpoint: str
    timeout_seconds: float = 60.0
    bearer_token: str | None = field(default=None, repr=False)
    max_response_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme != "https":
            raise RubricValidationError("online judge endpoint must use HTTPS")
        if not parsed.hostname or any(character.isspace() for character in self.endpoint):
            raise RubricValidationError("online judge URL must contain one valid hostname")
        hostname = parsed.hostname.rstrip(".").casefold()
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise RubricValidationError("online judge URL must not target loopback")
        try:
            literal_address = ipaddress.ip_address(hostname.split("%", 1)[0])
        except ValueError:
            literal_address = None
        if literal_address is not None and not _is_public_address(literal_address):
            raise RubricValidationError(
                "online judge URL must not target a private, loopback, link-local, or special address"
            )
        try:
            port = parsed.port
        except ValueError as exc:
            raise RubricValidationError("online judge URL contains an invalid port") from exc
        if port is not None and not 1 <= port <= 65_535:
            raise RubricValidationError("online judge URL contains an invalid port")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RubricValidationError(
                "online judge URL must not contain credentials, a query, or a fragment"
            )
        timeout = finite_number(self.timeout_seconds, name="online judge timeout")
        if timeout <= 0 or timeout > 600:
            raise RubricValidationError("online judge timeout must be in (0, 600]")
        if isinstance(self.max_response_bytes, bool) or not isinstance(self.max_response_bytes, int):
            raise RubricValidationError("max_response_bytes must be an integer")
        if not 1 <= self.max_response_bytes <= 16 * 1024 * 1024:
            raise RubricValidationError("max_response_bytes is outside the allowed range")
        if self.bearer_token is not None and (
            not isinstance(self.bearer_token, str)
            or not self.bearer_token
            or any(character in self.bearer_token for character in "\r\n\x00")
        ):
            raise RubricValidationError("online judge bearer token is malformed")


class OnlineRubricJudge:
    """Call a frozen judge service without logging URL secrets or response text."""

    def __init__(self, config: OnlineJudgeConfig) -> None:
        self.config = config

    def _request(
        self,
        *,
        rubric_version: str,
        criteria: Mapping[str, Any],
        candidate_response: str,
        sample_id: str,
    ) -> Mapping[str, Any]:
        if not isinstance(candidate_response, str):
            raise RubricValidationError("candidate response must be a string")
        nonce = secrets.token_hex(32)
        binding = build_judgment_binding(
            criteria,
            candidate_response,
            sample_id=sample_id,
            nonce=nonce,
        )
        payload = json.dumps(
            {
                "rubric_version": rubric_version,
                "binding": binding,
                "criteria": criteria,
                "candidate_response": candidate_response,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"
        try:
            body = _direct_https_post(
                self.config.endpoint,
                payload,
                headers,
                timeout_seconds=self.config.timeout_seconds,
                max_response_bytes=self.config.max_response_bytes,
            )
        except RubricValidationError:
            raise
        except (TimeoutError, OSError) as exc:
            raise RubricValidationError(
                f"online judge request failed ({type(exc).__name__})"
            ) from exc
        try:
            text = body.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RubricValidationError("online judge response is not UTF-8") from exc
        wrapper = strict_json_object(text)
        require_exact_keys(
            wrapper,
            name="online judge response",
            required={"rubric_version", "binding", "judgment"},
        )
        if wrapper.get("rubric_version") != rubric_version:
            raise RubricValidationError("online judge returned the wrong rubric version")
        if wrapper.get("binding") != binding:
            raise RubricValidationError("online judge returned a stale or mismatched binding")
        judgment = dict(
            require_mapping(wrapper.get("judgment"), name="online judge judgment")
        )
        if "binding" in judgment:
            raise RubricValidationError("online judge judgment must not override transport binding")
        judgment["binding"] = binding
        return judgment

    def judge_qa(self, criteria: Mapping[str, Any], candidate_response: str) -> dict[str, Any]:
        checked = validate_qa_criteria(criteria)
        raw = self._request(
            rubric_version=QA_RUBRIC_VERSION,
            criteria=checked,
            candidate_response=candidate_response,
            sample_id=checked["benchmark_id"],
        )
        return validate_qa_judgment(
            raw,
            checked,
            candidate_response=candidate_response,
            expected_nonce=raw["binding"]["nonce"],
            reject_unknown_ids=True,
        )

    def judge_motion_v2(
        self,
        criteria: Mapping[str, Any],
        candidate_response: str,
        *,
        sample_id: str,
    ) -> dict[str, Any]:
        checked = validate_motion_criteria_v2(criteria)
        raw = self._request(
            rubric_version=MOTION_RUBRIC_V2_VERSION,
            criteria=checked,
            candidate_response=candidate_response,
            sample_id=sample_id,
        )
        return validate_motion_judgment_v2(
            raw,
            checked,
            candidate_response=candidate_response,
            sample_id=sample_id,
            expected_nonce=raw["binding"]["nonce"],
            reject_unknown_ids=True,
        )


__all__ = ["OnlineJudgeConfig", "OnlineRubricJudge"]
