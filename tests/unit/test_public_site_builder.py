from pathlib import Path

import pytest

from tools.site.build_public_site import PUBLIC_ENTRIES, build


def test_public_site_excludes_server_and_private_index(tmp_path: Path) -> None:
    source = tmp_path / "online_page"
    source.mkdir()
    for name in PUBLIC_ENTRIES:
        item = source / name
        if "." in name and name != ".nojekyll":
            item.write_text("public", encoding="utf-8")
        elif name == ".nojekyll":
            item.touch()
        else:
            item.mkdir()
            (item / "index.html").write_text("public", encoding="utf-8")
    private = source / "data_page"
    private.mkdir()
    (private / "server.py").write_text("private", encoding="utf-8")
    (private / "index.sqlite").write_text("private", encoding="utf-8")

    output = tmp_path / "artifact"
    build(source, output)

    assert (output / "index.html").is_file()
    assert (output / "dataset-page" / "index.html").is_file()
    assert not (output / "data_page").exists()


def test_public_site_rejects_symlink(tmp_path: Path) -> None:
    source = tmp_path / "online_page"
    source.mkdir()
    for name in PUBLIC_ENTRIES:
        item = source / name
        if name in ("dataset-page", "guide", "motionllm-page"):
            item.mkdir()
        else:
            item.touch()
    private = tmp_path / "private"
    private.write_text("secret", encoding="utf-8")
    (source / "dataset-page" / "leak").symlink_to(private)

    with pytest.raises(ValueError, match="symlink"):
        build(source, tmp_path / "artifact")
