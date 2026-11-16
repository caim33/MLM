"""Check the public staging boundary for static learning reports."""

import tempfile
import unittest
from pathlib import Path

from build_public_site import PUBLIC_ENTRIES, build


class PublicReportStagingTest(unittest.TestCase):
    def test_report_assets_are_published_while_private_service_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "online_page"
            source.mkdir()
            directories = {
                "dataset-page", "guide", "motionllm-page", "marin-535b", "deepseek"
            }
            for name in PUBLIC_ENTRIES:
                target = source / name
                if name in directories:
                    target.mkdir()
                    (target / "index.html").write_text("public", encoding="utf-8")
                else:
                    target.write_text("public", encoding="utf-8")
            assets = source / "marin-535b" / "assets"
            assets.mkdir()
            (assets / "routing.mp4").write_bytes(b"animation-fixture")
            deepseek_assets = source / "deepseek" / "assets"
            deepseek_assets.mkdir()
            (deepseek_assets / "architecture.png").write_bytes(b"figure-fixture")
            (source / "deepseek" / "deepseek-study-report.pdf").write_bytes(
                b"report-fixture"
            )
            private = source / "data_page"
            private.mkdir()
            (private / "server.py").write_text("private-service", encoding="utf-8")
            (private / "index.sqlite").write_bytes(b"private-index")
            (source / ".env").write_text("PRIVATE_FIXTURE=exclude", encoding="utf-8")

            output = base / "public-site"
            build(source, output)

            self.assertTrue((output / "marin-535b" / "index.html").is_file())
            self.assertEqual(
                (output / "marin-535b" / "assets" / "routing.mp4").read_bytes(),
                b"animation-fixture",
            )
            self.assertFalse((output / "data_page").exists())
            self.assertFalse((output / ".env").exists())
            self.assertEqual(
                (output / "deepseek" / "assets" / "architecture.png").read_bytes(),
                b"figure-fixture",
            )
            self.assertEqual(
                (output / "deepseek" / "deepseek-study-report.pdf").read_bytes(),
                b"report-fixture",
            )


if __name__ == "__main__":
    unittest.main()
