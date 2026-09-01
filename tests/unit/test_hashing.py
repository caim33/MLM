import hashlib
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import motion_eval.core.hashing as hashing_module
from motion_eval.core import (
    FileChangedDuringHashError,
    PathOutsideRootError,
    SymlinkNotAllowedError,
    capture_directory_bytes,
    directory_merkle_sha256,
    hash_path,
    sha256_file,
    sha256_json,
)


def test_file_sha256_matches_standard_library(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc")

    assert sha256_file(artifact) == hashlib.sha256(b"abc").hexdigest()
    receipt = hash_path(artifact)
    assert receipt.algorithm == "sha256"
    assert receipt.digest == sha256_file(artifact)
    assert receipt.file_count == 1
    assert receipt.total_bytes == 3


def _create_tree(root: Path, reverse: bool = False):
    root.mkdir()
    ordinary = [
        lambda: (root / "z.txt").write_text("last", encoding="utf-8"),
        lambda: (root / "empty").mkdir(),
        lambda: (root / "nested").mkdir(),
        lambda: (root / "nested" / "a.bin").write_bytes(b"first"),
    ]
    alternate = [
        lambda: (root / "nested").mkdir(),
        lambda: (root / "nested" / "a.bin").write_bytes(b"first"),
        lambda: (root / "empty").mkdir(),
        lambda: (root / "z.txt").write_text("last", encoding="utf-8"),
    ]
    for action in alternate if reverse else ordinary:
        action()


def test_directory_merkle_is_independent_of_creation_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_tree(first)
    _create_tree(second, reverse=True)

    assert directory_merkle_sha256(first) == directory_merkle_sha256(second)
    receipt = hash_path(first)
    assert receipt.kind == "directory"
    assert receipt.file_count == 2
    assert receipt.total_bytes == len(b"lastfirst")


def test_directory_merkle_changes_for_content_path_and_empty_directory(tmp_path):
    root = tmp_path / "tree"
    _create_tree(root)
    original = directory_merkle_sha256(root)

    (root / "z.txt").write_text("changed", encoding="utf-8")
    content_changed = directory_merkle_sha256(root)
    assert content_changed != original

    (root / "z.txt").write_text("last", encoding="utf-8")
    (root / "empty").rename(root / "renamed-empty")
    assert directory_merkle_sha256(root) != original


def test_directory_capture_digest_is_computed_from_returned_exact_bytes(tmp_path):
    root = tmp_path / "capture"
    _create_tree(root)
    captured = capture_directory_bytes(root)

    assert captured.receipt.to_dict() == hash_path(root).to_dict()
    assert dict(captured.files) == {
        "nested/a.bin": b"first",
        "z.txt": b"last",
    }


def test_directory_capture_enforces_bounded_bundle_limits(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    (root / "a.py").write_bytes(b"1234")
    (root / "b.py").write_bytes(b"5678")

    with pytest.raises(Exception, match="file-count limit"):
        capture_directory_bytes(root, max_files=1)
    with pytest.raises(Exception, match="byte-size limit"):
        capture_directory_bytes(root, max_total_bytes=7)


def test_canonical_json_hash_rejects_nonfinite_numbers():
    with pytest.raises(ValueError, match="finite canonical JSON"):
        sha256_json({"score": float("nan")})


def test_symlink_is_rejected_by_default(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("data", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    with pytest.raises(SymlinkNotAllowedError):
        sha256_file(link)


def test_directory_symlink_follow_must_stay_in_allowed_root(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("secret", encoding="utf-8")
    link = tree / "escape"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable on this platform")

    with pytest.raises(ValueError, match="outside declared root"):
        directory_merkle_sha256(tree, symlink_policy="follow", allowed_root=tree)


def test_link_policy_rejects_simulated_reparse_entry_outside_allowed_root(
    tmp_path, monkeypatch
):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    entry = outside / "junction"
    entry.write_text("simulated reparse entry", encoding="utf-8")
    entry_info = entry.lstat()
    original_link_info = hashing_module._link_info

    def simulated_link_info(path):
        return entry_info if Path(path) == entry else original_link_info(path)

    def forbidden_readlink(path):
        raise AssertionError("readlink must not run before entry containment passes")

    monkeypatch.setattr(hashing_module, "_link_info", simulated_link_info)
    monkeypatch.setattr(hashing_module.os, "readlink", forbidden_readlink)

    with pytest.raises(PathOutsideRootError, match="outside declared root"):
        hash_path(entry, symlink_policy="link", allowed_root=allowed)


def test_link_policy_hashes_simulated_in_root_windows_reparse_entry(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    root.mkdir()
    entry = root / "junction"
    entry.write_text("simulated reparse entry", encoding="utf-8")
    entry_info = entry.lstat()
    target_text = r"\??\C:\canonical\target"
    original_link_info = hashing_module._link_info

    monkeypatch.setattr(
        hashing_module,
        "_link_info",
        lambda path: entry_info if Path(path) == entry else original_link_info(path),
    )
    monkeypatch.setattr(hashing_module.os, "readlink", lambda path: target_text)

    receipt = hash_path(entry, symlink_policy="link", allowed_root=root)
    target_bytes = os.fsencode(target_text)
    assert receipt.kind == "symlink"
    assert receipt.digest == hashlib.sha256(target_bytes).hexdigest()
    assert receipt.total_bytes == len(target_bytes)


def test_windows_reparse_attribute_is_classified_as_link_on_every_platform():
    simulated = SimpleNamespace(
        st_mode=stat.S_IFDIR,
        st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
    )
    assert hashing_module._is_link_stat(simulated)


def test_link_policy_rejects_identity_change_while_reading(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    entry = root / "link"
    entry.write_text("simulated link", encoding="utf-8")
    stable = entry.lstat()
    changed = SimpleNamespace(
        st_dev=stable.st_dev,
        st_ino=stable.st_ino,
        st_mode=stable.st_mode,
        st_size=stable.st_size,
        st_mtime_ns=stable.st_mtime_ns + 1,
        st_ctime_ns=stable.st_ctime_ns,
    )
    calls = 0
    original_link_info = hashing_module._link_info

    def delegated_link_info(path):
        nonlocal calls
        if Path(path) == entry:
            calls += 1
            return stable if calls < 3 else changed
        return original_link_info(path)

    monkeypatch.setattr(hashing_module, "_link_info", delegated_link_info)
    monkeypatch.setattr(hashing_module.os, "readlink", lambda path: "target")

    with pytest.raises(FileChangedDuringHashError, match="link changed"):
        hash_path(entry, symlink_policy="link", allowed_root=root)
