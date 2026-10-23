"""CLI setup paths stay lightweight and report actionable model failures."""
import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from audioseek.downloads import ModelDownloadError
from audioseek.io import CONFIG_ROOT
from audioseek.ui import main


class StartupTests(unittest.TestCase):
    def test_installed_resources_are_present(self):
        for name in ["retrieval.json", "semantic.json", "upload.json", "asr_model_files.json", "semantic_model_files.json"]:
            self.assertTrue((CONFIG_ROOT / name).is_file(), name)

    def test_download_only_command_never_builds_interface(self):
        with patch("sys.argv", ["audioseek", "--download-models", "--data-dir", "cache-fixture"]):
            with patch("audioseek.downloads.download_models") as fetch:
                with patch("audioseek.ui.create_app", side_effect=AssertionError("must not load inference")):
                    main()
        self.assertEqual(fetch.call_args.args[0], Path("cache-fixture"))

    def test_download_failure_has_friendly_message_and_nonzero_exit(self):
        stderr = io.StringIO()
        with patch("sys.argv", ["audioseek", "--download-models"]):
            with patch("audioseek.downloads.download_models", side_effect=ModelDownloadError("Check access and retry.")):
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exit_result:
                    main()
        self.assertEqual(exit_result.exception.code, 1)
        self.assertIn("Model setup failed", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
