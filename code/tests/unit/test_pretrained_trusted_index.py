from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import motion_eval.data.receipts as receipt_module
from motion_eval.data import BatchReceiptError


def _event_trust() -> dict[str, str]:
    return {
        "key_id": f"sha256:{'1' * 64}",
        "state_scope_id": "2" * 64,
    }


def _regular_index(
    tmp_path: Path, *, generated_at: datetime | None = None
) -> tuple[Path, Path, dict[str, list[dict[str, object]]], dict[str, object]]:
    root = tmp_path / "pretrained"
    component = root / "by_model" / "model" / "base.bin"
    component.parent.mkdir(parents=True)
    component.write_bytes(b"original-component")
    assets, _ = receipt_module._freeze_pretrained_assets(
        root,
        {
            "model": [
                {
                    "role": "base",
                    "path": "by_model/model/base.bin",
                    "kind": "checkpoint",
                    "expected_sha256": None,
                }
            ]
        },
    )
    index = receipt_module._build_trusted_pretrained_index(
        root,
        assets,
        event_trust=_event_trust(),
        generated_at=generated_at or datetime.now(timezone.utc),
    )
    return root, component, assets, index


def _verify_index(
    root: Path,
    assets: dict[str, list[dict[str, object]]],
    index: dict[str, object],
) -> None:
    receipt_module._verify_trusted_pretrained_index(
        index,
        assets=assets,
        pretrained_root=root,
        event_trust=_event_trust(),
    )


def test_trusted_pretrained_index_rejects_internal_tamper(tmp_path: Path) -> None:
    root, _, assets, index = _regular_index(tmp_path)
    tampered = deepcopy(index)
    tampered["path_bindings"][0]["logical_path"] = str(tmp_path / "exchanged")

    with pytest.raises(BatchReceiptError, match="hash/trust binding"):
        _verify_index(root, assets, tampered)


def test_schema_v2_index_has_no_ttl_and_accepts_an_old_freeze(tmp_path: Path) -> None:
    old = datetime(2001, 1, 1, tzinfo=timezone.utc)
    root, _, assets, index = _regular_index(tmp_path, generated_at=old)

    assert index["schema_version"] == "2.0"
    assert index["generated_at"] == old.isoformat()
    assert "expires_at" not in index
    _verify_index(root, assets, index)


def test_trusted_pretrained_index_rejects_pretrained_root_exchange(
    tmp_path: Path,
) -> None:
    root, _, assets, index = _regular_index(tmp_path)
    exchanged_root = tmp_path / "other_pretrained"
    exchanged_component = exchanged_root / "by_model" / "model" / "base.bin"
    exchanged_component.parent.mkdir(parents=True)
    exchanged_component.write_bytes(b"original-component")

    with pytest.raises(BatchReceiptError, match="root was exchanged"):
        receipt_module._verify_trusted_pretrained_index(
            index,
            assets=assets,
            pretrained_root=exchanged_root,
            event_trust=_event_trust(),
        )


def test_trusted_pretrained_index_rejects_symlink_target_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, logical, assets, _ = _regular_index(tmp_path)
    target_digest = ["3" * 64]
    real_is_link = receipt_module._is_link_or_reparse

    def simulated_link(path: Path) -> bool:
        return Path(path) == logical or real_is_link(Path(path))

    def simulated_link_hash(*args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs.get("symlink_policy") == "link"
        return SimpleNamespace(digest=target_digest[0])

    monkeypatch.setattr(receipt_module, "_is_link_or_reparse", simulated_link)
    monkeypatch.setattr(receipt_module, "hash_path", simulated_link_hash)
    index = receipt_module._build_trusted_pretrained_index(
        root,
        assets,
        event_trust=_event_trust(),
        generated_at=datetime.now(timezone.utc),
    )

    # Simulate the same logical link entry resolving to a different target.
    target_digest[0] = "4" * 64
    with pytest.raises(BatchReceiptError, match="target changed"):
        _verify_index(root, assets, index)


def test_normal_index_verification_does_not_rehash_underlying_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, component, assets, index = _regular_index(tmp_path)
    replacement = b"tampered-component"
    assert len(replacement) == component.stat().st_size
    component.write_bytes(replacement)

    def unexpected_content_hash(*args: object, **kwargs: object) -> object:
        raise AssertionError("normal trusted-index verification rehashed asset content")

    with monkeypatch.context() as patch:
        patch.setattr(receipt_module, "hash_path", unexpected_content_hash)
        _verify_index(root, assets, index)

    with pytest.raises(BatchReceiptError, match="changed after batch freeze"):
        receipt_module._verify_pretrained_assets(assets)
