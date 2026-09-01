from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from motionllm.grpo import (
    OnlineJudgeConfig,
    OnlineRubricJudge,
    RubricValidationError,
)
from motionllm.grpo import rubric_online
from tests.unit.test_qa_rubric import qa_criteria, qa_judgment


class _Headers:
    def get_content_type(self):
        return "application/json"


def _semantic_qa_judgment(criteria, candidate):
    judgment = qa_judgment(criteria, candidate)
    judgment.pop("binding")
    return judgment


def test_online_qa_judge_posts_bound_versioned_request_and_validates_response(
    monkeypatch,
):
    criteria = qa_criteria()
    candidate = "<think>x</think><answer>A</answer>"
    captured = {}

    def fake_post(endpoint, payload, headers, **kwargs):
        request_payload = json.loads(payload)
        captured.update(
            endpoint=endpoint,
            payload=request_payload,
            headers=headers,
            kwargs=kwargs,
        )
        return json.dumps(
            {
                "rubric_version": "qa_mc_rubric_v1",
                "binding": request_payload["binding"],
                "judgment": _semantic_qa_judgment(criteria, candidate),
            },
            separators=(",", ":"),
        ).encode()

    monkeypatch.setattr(rubric_online, "_direct_https_post", fake_post)
    client = OnlineRubricJudge(
        OnlineJudgeConfig(
            endpoint="https://judge.invalid/rubric",
            timeout_seconds=3,
            bearer_token="token-sentinel",
        )
    )
    parsed = client.judge_qa(criteria, candidate)
    assert parsed["language_conciseness_score"] == 5
    assert captured["payload"]["rubric_version"] == "qa_mc_rubric_v1"
    assert captured["payload"]["criteria"]["benchmark_id"] == "QA_000001"
    assert captured["payload"]["candidate_response"] == candidate
    assert captured["payload"]["binding"] == parsed["binding"]
    assert captured["headers"]["Authorization"] == "Bearer token-sentinel"
    assert captured["kwargs"]["timeout_seconds"] == 3


def test_online_judge_rejects_wrong_version_and_mismatched_binding(monkeypatch):
    criteria = qa_criteria()
    candidate = "candidate"

    def response_with(*, wrong_version=False, wrong_binding=False):
        def fake_post(endpoint, payload, headers, **kwargs):
            del endpoint, headers, kwargs
            request_payload = json.loads(payload)
            binding = dict(request_payload["binding"])
            if wrong_binding:
                binding["candidate_sha256"] = "0" * 64
            return json.dumps(
                {
                    "rubric_version": (
                        "temporal_caption" if wrong_version else "qa_mc_rubric_v1"
                    ),
                    "binding": binding,
                    "judgment": _semantic_qa_judgment(criteria, candidate),
                },
                separators=(",", ":"),
            ).encode()

        return fake_post

    client = OnlineRubricJudge(
        OnlineJudgeConfig(endpoint="https://judge.invalid/rubric")
    )
    monkeypatch.setattr(
        rubric_online, "_direct_https_post", response_with(wrong_version=True)
    )
    with pytest.raises(RubricValidationError, match="wrong rubric version"):
        client.judge_qa(criteria, candidate)
    monkeypatch.setattr(
        rubric_online, "_direct_https_post", response_with(wrong_binding=True)
    )
    with pytest.raises(RubricValidationError, match="mismatched binding"):
        client.judge_qa(criteria, candidate)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1/judge",
        "https://10.1.2.3/judge",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/judge",
        "https://[fe80::1]/judge",
        "https://192.0.2.1/judge",
        "https://224.0.0.1/judge",
        "https://[::ffff:127.0.0.1]/judge",
    ],
)
def test_online_config_rejects_loopback_private_link_local_and_special_ips(endpoint):
    with pytest.raises(RubricValidationError, match="private|loopback|link-local|special"):
        OnlineJudgeConfig(endpoint=endpoint)


def test_dns_resolution_rejects_any_non_global_answer(monkeypatch):
    def fake_getaddrinfo(*args, **kwargs):
        del args, kwargs
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 443)),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("169.254.169.254", 443),
            ),
        ]

    monkeypatch.setattr(rubric_online.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(RubricValidationError, match="private|link-local|special"):
        rubric_online._resolve_public_addresses(
            "rebind.invalid", 443, rubric_online.time.monotonic() + 1
        )


def test_direct_transport_rejects_redirect_without_following_or_forwarding_again(
    monkeypatch,
):
    requests = []

    class FakeSocket:
        def settimeout(self, timeout):
            del timeout

    class RedirectResponse:
        status = 302
        headers = _Headers()

    class FakeConnection:
        def __init__(self, host, port, **kwargs):
            requests.append((host, port, kwargs))
            self.sock = FakeSocket()

        def connect(self):
            return None

        def request(self, method, target, body, headers):
            requests[-1] += (method, target, body, headers)

        def getresponse(self):
            return RedirectResponse()

        def close(self):
            return None

    monkeypatch.setattr(
        rubric_online, "_resolve_public_addresses", lambda *args: ("8.8.8.8",)
    )
    monkeypatch.setattr(rubric_online, "_PinnedHTTPSConnection", FakeConnection)
    with pytest.raises(RubricValidationError, match="redirects are forbidden"):
        rubric_online._direct_https_post(
            "https://judge.invalid/rubric",
            b"{}",
            {"Authorization": "Bearer token-sentinel"},
            timeout_seconds=3,
            max_response_bytes=1024,
        )
    assert len(requests) == 1
    assert requests[0][-1]["Authorization"] == "Bearer token-sentinel"


def test_direct_transport_enforces_one_total_wall_clock_deadline(monkeypatch):
    clock = [0.0]

    class FakeSocket:
        def settimeout(self, timeout):
            assert timeout <= 1.0

    class SlowResponse:
        status = 200
        headers = _Headers()

        def read(self, size):
            del size
            clock[0] += 0.6
            return b"x"

    class FakeConnection:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.sock = FakeSocket()

        def connect(self):
            return None

        def request(self, *args, **kwargs):
            del args, kwargs

        def getresponse(self):
            return SlowResponse()

        def close(self):
            return None

    monkeypatch.setattr(rubric_online.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        rubric_online, "_resolve_public_addresses", lambda *args: ("8.8.8.8",)
    )
    monkeypatch.setattr(rubric_online, "_PinnedHTTPSConnection", FakeConnection)
    with pytest.raises(RubricValidationError, match="total deadline"):
        rubric_online._direct_https_post(
            "https://judge.invalid/rubric",
            b"{}",
            {},
            timeout_seconds=1.0,
            max_response_bytes=1024,
        )


def test_direct_transport_rebudgets_after_connect_before_request(monkeypatch):
    clock = [0.0]

    class BudgetSocket:
        timeout = None

        def settimeout(self, timeout):
            self.timeout = timeout

    class FakeConnection:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.sock = BudgetSocket()

        def connect(self):
            clock[0] += 0.6

        def request(self, *args, **kwargs):
            del args, kwargs
            assert self.sock.timeout == pytest.approx(0.4)
            clock[0] += self.sock.timeout
            raise TimeoutError("simulated upload timeout")

        def close(self):
            return None

    monkeypatch.setattr(rubric_online.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        rubric_online, "_resolve_public_addresses", lambda *args: ("8.8.8.8",)
    )
    monkeypatch.setattr(rubric_online, "_PinnedHTTPSConnection", FakeConnection)
    with pytest.raises(RubricValidationError, match="request failed"):
        rubric_online._direct_https_post(
            "https://judge.invalid/rubric",
            b"{}",
            {},
            timeout_seconds=1.0,
            max_response_bytes=1024,
        )
    assert clock[0] == pytest.approx(1.0)


def test_direct_transport_watchdog_interrupts_blocking_header_read(monkeypatch):
    released = threading.Event()

    class BlockingSocket:
        def settimeout(self, timeout):
            del timeout

        def close(self):
            released.set()

    class BlockingConnection:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.sock = BlockingSocket()

        def connect(self):
            return None

        def request(self, *args, **kwargs):
            del args, kwargs

        def getresponse(self):
            assert released.wait(timeout=1.0)
            raise OSError("closed by absolute-deadline watchdog")

        def close(self):
            self.sock.close()

    monkeypatch.setattr(
        rubric_online, "_resolve_public_addresses", lambda *args: ("8.8.8.8",)
    )
    monkeypatch.setattr(
        rubric_online, "_PinnedHTTPSConnection", BlockingConnection
    )
    started = time.monotonic()
    with pytest.raises(RubricValidationError, match="request failed"):
        rubric_online._direct_https_post(
            "https://judge.invalid/rubric",
            b"{}",
            {},
            timeout_seconds=0.2,
            max_response_bytes=1024,
        )
    assert released.is_set()
    assert time.monotonic() - started < 0.6
