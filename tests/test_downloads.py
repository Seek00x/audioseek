"""Download recovery never promotes incomplete or unverified model bytes."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from audioseek.downloads import ModelDownloadError, fetch_verified_file


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model.bin"
        self.payload = b"verified model fixture"
        self.item = {"filename": "model.bin", "bytes": len(self.payload),
                     "lfs_sha256": hashlib.sha256(self.payload).hexdigest(), "git_blob_sha1": None}
        self.messages = []

    def fetch(self, **kwargs):
        return fetch_verified_file(self.path, self.item, "https://example.invalid/model.bin",
                                   report=self.messages.append, **kwargs)

    def test_valid_cache_never_contacts_network(self):
        self.path.write_bytes(self.payload)
        with patch("audioseek.downloads.urlopen", side_effect=AssertionError("must use verified cache")):
            self.assertEqual(self.fetch(), self.item["lfs_sha256"])

    def test_download_shows_progress_and_promotes_verified_bytes(self):
        with patch("audioseek.downloads.urlopen", return_value=io.BytesIO(self.payload)):
            self.fetch()
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertTrue(any("100%" in message for message in self.messages))
        self.assertFalse(list(self.path.parent.glob("*.part")))

    def test_transient_network_failure_retries_current_file(self):
        with patch("audioseek.downloads.urlopen", side_effect=[URLError("temporary outage"), io.BytesIO(self.payload)]) as request:
            with patch("audioseek.downloads.time.sleep"):
                self.fetch()
        self.assertEqual(request.call_count, 2)
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_wrong_checksum_cannot_replace_existing_file(self):
        original = b"existing invalid cache"
        self.path.write_bytes(original)
        with patch("audioseek.downloads.urlopen", return_value=io.BytesIO(b"x" * len(self.payload))):
            with self.assertRaises(ModelDownloadError):
                self.fetch(attempts=1)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(list(self.path.parent.glob("*.part")))

    def test_truncated_and_oversized_files_are_rejected(self):
        for payload in [b"short", self.payload + b"extra"]:
            with self.subTest(payload=payload), patch("audioseek.downloads.urlopen", return_value=io.BytesIO(payload)):
                with self.assertRaises(ModelDownloadError):
                    self.fetch(attempts=1)
            self.assertFalse(self.path.exists())
            self.assertFalse(list(self.path.parent.glob("*.part")))

    def test_corrupt_cache_is_replaced_by_verified_download(self):
        self.path.write_bytes(b"x" * len(self.payload))
        with patch("audioseek.downloads.urlopen", return_value=io.BytesIO(self.payload)):
            self.fetch()
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_git_blob_hash_validates_small_upstream_files(self):
        self.item["lfs_sha256"] = None
        self.item["git_blob_sha1"] = hashlib.sha1(f"blob {len(self.payload)}\0".encode() + self.payload).hexdigest()
        with patch("audioseek.downloads.urlopen", return_value=io.BytesIO(self.payload)):
            self.fetch()
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_missing_revision_fails_without_repeated_requests(self):
        error = HTTPError("https://example.invalid/file", 404, "Missing", {}, None)
        with patch("audioseek.downloads.urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(ModelDownloadError, "HTTP 404"):
                self.fetch()
        self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
