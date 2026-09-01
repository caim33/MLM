#!/usr/bin/env python3
"""Check local href/src targets and HTML fragments in a static site."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.targets: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag in {"a", "link"} and values.get("href"):
            self.targets.append(("href", str(values["href"])))
        if tag in {"img", "script", "source"} and values.get("src"):
            self.targets.append(("src", str(values["src"])))


def parse_page(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    pages = sorted(root.rglob("*.html"))
    parsed = {page: parse_page(page) for page in pages}
    failures: list[str] = []
    checked = 0

    for page, content in parsed.items():
        for kind, raw_target in content.targets:
            split = urlsplit(raw_target)
            if split.scheme or split.netloc or raw_target.startswith(("mailto:", "data:", "javascript:")):
                continue
            relative = unquote(split.path)
            target = (root / relative.lstrip("/")) if relative.startswith("/") else (page.parent / relative)
            if not relative:
                target = page
            target = target.resolve()
            checked += 1
            try:
                target.relative_to(root)
            except ValueError:
                failures.append(f"{page.relative_to(root)}: {kind} escapes root: {raw_target}")
                continue
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                failures.append(f"{page.relative_to(root)}: missing {kind}: {raw_target}")
                continue
            if split.fragment and target.suffix.lower() == ".html":
                target_parser = parsed.get(target) or parse_page(target)
                if split.fragment not in target_parser.ids:
                    failures.append(
                        f"{page.relative_to(root)}: missing fragment #{split.fragment} in "
                        f"{target.relative_to(root)}"
                    )

    print(f"pages={len(pages)} checked_local_targets={checked} failures={len(failures)}")
    for failure in failures:
        print(f"FAIL\t{failure}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
