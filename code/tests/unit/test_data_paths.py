from __future__ import annotations

import os
from pathlib import Path

import pytest

from motionllm.data import (
    MediaNotFoundError,
    PathResolutionError,
    UnsafePathError,
    resolve_media_path,
    resolve_path_within_root,
)


def test_resolves_relative_and_absolute_paths_inside_root(tmp_path: Path) -> None:
    media = tmp_path / "media"
    nested = media / "nested"
    nested.mkdir(parents=True)
    clip = nested / "clip.mp4"
    clip.write_bytes(b"video")

    assert resolve_media_path(media, "nested/clip.mp4") == clip.resolve()
    assert resolve_media_path(media, clip.resolve()) == clip.resolve()


@pytest.mark.parametrize("reference", ["../outside.mp4", "nested/../../outside.mp4"])
def test_rejects_relative_traversal_even_when_target_is_missing(
    tmp_path: Path, reference: str
) -> None:
    media = tmp_path / "media"
    (media / "nested").mkdir(parents=True)
    with pytest.raises(UnsafePathError):
        resolve_media_path(media, reference)


def test_rejects_absolute_path_outside_root(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    with pytest.raises(UnsafePathError):
        resolve_media_path(media, outside)


def test_rejects_uri_null_and_empty_references(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    for reference in ("https://example.test/clip.mp4", "bad\x00name", ""):
        with pytest.raises(PathResolutionError):
            resolve_media_path(media, reference)


@pytest.mark.skipif(os.name != "nt", reason="NTFS path syntax is Windows-specific")
@pytest.mark.parametrize("reference", ["clip.mp4:secret", "NUL", r"\\.\NUL"])
def test_rejects_windows_device_and_alternate_stream_paths(
    tmp_path: Path, reference: str
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    with pytest.raises(UnsafePathError):
        resolve_media_path(media, reference)


def test_missing_media_is_explicit_and_can_only_be_opted_out(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    with pytest.raises(MediaNotFoundError):
        resolve_media_path(media, "missing.mp4")
    assert resolve_media_path(media, "missing.mp4", must_exist=False) == (
        media / "missing.mp4"
    ).resolve()


def test_deferred_existence_does_not_allow_an_existing_directory_as_media(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    (media / "directory").mkdir(parents=True)
    with pytest.raises(PathResolutionError, match="regular file"):
        resolve_media_path(media, "directory", must_exist=False)


@pytest.mark.skipif(os.name != "nt", reason="drive-relative paths are Windows-specific")
def test_rejects_drive_relative_path(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    with pytest.raises(UnsafePathError, match="drive-relative"):
        resolve_media_path(media, "C:clip.mp4", must_exist=False)


def test_expected_path_kind_is_checked(tmp_path: Path) -> None:
    media = tmp_path / "media"
    directory = media / "frames"
    directory.mkdir(parents=True)
    with pytest.raises(PathResolutionError):
        resolve_media_path(media, "frames")
    assert (
        resolve_path_within_root(media, "frames", expected_kind="directory")
        == directory.resolve()
    )


def test_symlink_cannot_escape_media_root(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    link = media / "link.mp4"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks is not permitted in this environment")
    with pytest.raises(UnsafePathError):
        resolve_media_path(media, link)


def test_root_must_exist_and_be_directory(tmp_path: Path) -> None:
    with pytest.raises(PathResolutionError):
        resolve_media_path(tmp_path / "missing", "clip.mp4")
    root_file = tmp_path / "root-file"
    root_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(PathResolutionError):
        resolve_media_path(root_file, "clip.mp4")
