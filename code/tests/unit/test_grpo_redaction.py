from __future__ import annotations

import pytest

from motionllm.grpo import (
    assert_manifest_secret_free,
    describe_environment_overrides,
    redact_command_for_log,
    redact_mapping_for_log,
)
from motionllm.grpo.redaction import SecretMaterialError


def test_environment_descriptions_never_contain_values():
    secret = "sentinel-super-secret-value"
    described = describe_environment_overrides(
        {"WANDB_API_KEY": secret, "CUDA_VISIBLE_DEVICES": "0", "REMOVE_ME": None}
    )
    assert secret not in repr(described)
    assert "0" not in repr(described)
    assert described == {
        "WANDB_API_KEY": "<redacted>",
        "CUDA_VISIBLE_DEVICES": "<set>",
        "REMOVE_ME": "<unset>",
    }


def test_nested_mapping_and_command_redact_by_key():
    secret = "sentinel-super-secret-value"
    redacted = redact_mapping_for_log(
        {"run": {"api_token": secret, "name": "safe"}, "password": secret}
    )
    assert secret not in repr(redacted)
    command = redact_command_for_log(
        ["swift", "--api_token", secret, "--password=another-secret", "--seed", "1"]
    )
    assert secret not in " ".join(command)
    assert "another-secret" not in " ".join(command)
    assert command[-2:] == ("--seed", "1")


def test_mapping_redaction_recurses_into_list_and_tuple_strings():
    value = redact_mapping_for_log(
        {
            "items": [
                "Bearer abc.def.ghi",
                ("https://user:password@example.invalid/x", {"safe": "ok"}),
            ]
        }
    )
    rendered = repr(value)
    assert "abc.def.ghi" not in rendered
    assert "password" not in rendered
    assert "ok" in rendered


def test_command_redacts_url_userinfo_and_nested_json_credentials():
    command = redact_command_for_log(
        [
            "swift",
            "--vllm_server_base_url",
            "https://user:password@example.invalid/v1",
            "--model_kwargs",
            '{"api_key":"nested-secret"}',
        ]
    )
    rendered = " ".join(command)
    assert "password" not in rendered
    assert "nested-secret" not in rendered


def test_token_only_uri_userinfo_is_redacted_and_rejected():
    secret_uri = "https://sentinel-token@example.invalid/v1"
    rendered = " ".join(
        redact_command_for_log(["swift", "--vllm_server_base_url", secret_uri])
    )
    assert "sentinel-token" not in rendered
    with pytest.raises(SecretMaterialError):
        assert_manifest_secret_free({"endpoint": secret_uri})


def test_exception_and_custom_objects_never_leak_through_repr():
    secret = "adversarial-secret-token"

    class Dangerous:
        def __repr__(self):
            raise AssertionError("sanitizer must not call arbitrary repr")

        def __str__(self):
            return secret

    value = redact_mapping_for_log(
        {
            "error": RuntimeError(f"Bearer {secret}"),
            "custom": Dangerous(),
        }
    )
    rendered = repr(value)
    assert secret not in rendered
    assert value == {"error": "<RuntimeError>", "custom": "<object>"}


@pytest.mark.parametrize(
    "manifest",
    [
        {"api_key": "x"},
        {"nested": {"password": "x"}},
        {"notes": "-----BEGIN " + "PRIVATE" + " KEY-----"},
        {"headers": ["Bearer abc.def.ghi"]},
        {"endpoint": "https://user:password@example.invalid/v1"},
        {"notes": "api_key=nested-secret"},
    ],
)
def test_manifest_rejects_credential_material(manifest):
    with pytest.raises(SecretMaterialError):
        assert_manifest_secret_free(manifest)


def test_manifest_accepts_normal_provenance():
    assert_manifest_secret_free(
        {"batch_id": "batch", "model_id": "model", "hash": "a" * 64}
    )
