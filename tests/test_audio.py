"""Audio boundary and mapping checks, independent of downloading models."""
import tempfile
import unittest
import wave
from pathlib import Path
import sys
import numpy as np

from audioseek.base import BaseService
from audioseek.audio import chunk_time_ranges, decode_audio, extract_clip, playback_ranges, word_time_coordinates


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {"allowed_extensions": [".wav", ".mp3", ".m4a", ".flac", ".ogg"], "max_file_bytes": 1000000, "max_duration_seconds": 2, "sample_rate": 16000}

    def tearDown(self):
        self.temp.cleanup()

    def wav(self, seconds=1, channels=2, rate=8000):
        path = self.root / "input.wav"
        with wave.open(str(path), "wb") as output:
            output.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
            values = (np.sin(np.arange(int(seconds*rate))*2*np.pi*440/rate)*12000).astype("<i2")
            output.writeframes(np.repeat(values[:, None], channels, axis=1).tobytes())
        return path

    def test_resample_preserves_duration_and_mono_format(self):
        target = self.root / "normalized.wav"
        audit = decode_audio(self.wav(), target, self.config)
        self.assertAlmostEqual(audit["duration_seconds"], 1, places=3)
        with wave.open(str(target)) as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))

    def test_bad_format_empty_and_corrupt_are_rejected(self):
        for filename, data in [("bad.wav", b"not audio"), ("empty.wav", b""), ("bad.txt", b"text")]:
            path = self.root / filename; path.write_bytes(data)
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                decode_audio(path, self.root / "target.wav", self.config)
        self.assertFalse((self.root / "target.wav.part").exists())

    def test_duration_and_byte_limit_reject_without_partial_output(self):
        source = self.wav(seconds=3)
        target = self.root / "target.wav"
        with self.assertRaisesRegex(ValueError, "minutes"):
            decode_audio(source, target, self.config)
        self.assertFalse(target.exists())
        self.assertFalse((self.root / "target.wav.part").exists())
        with self.assertRaises(ValueError):
            decode_audio(source, target, {**self.config, "max_duration_seconds": 4, "max_file_bytes": 1})

    def test_missing_word_boundary_disables_whole_chunk_playback(self):
        segments = [{"text": "hello world", "speakerName": "Unknown", "timed_words": [{"char_start": 0, "char_end": 5, "start": .2, "end": .4}]}]
        times = word_time_coordinates(segments, 1)
        self.assertEqual(times, [{"start": .2, "end": .4}, None])
        self.assertIsNone(chunk_time_ranges({"word_start": 0, "word_end": 2}, times))

    def test_nonfinite_or_out_of_audio_times_cannot_be_played(self):
        for end in [float("nan"), 2, -.1]:
            segments = [{"text": "hello", "speakerName": "Unknown", "timed_words": [{"char_start": 0, "char_end": 5, "start": 0, "end": end}]}]
            self.assertEqual(word_time_coordinates(segments, 1), [None])

    def test_gaps_and_long_ranges_preserve_all_evidence(self):
        chunk = {"word_start": 0, "word_end": 3}
        times = [{"start": 1, "end": 2}, {"start": 2, "end": 3}, {"start": 20, "end": 200}]
        ranges = chunk_time_ranges(chunk, times)
        self.assertEqual(ranges, [[1, 3], [20, 200]])
        clipped = playback_ranges(ranges, 201)
        self.assertEqual(clipped, [[0, 4], [19, 109], [109, 199], [199, 201]])

    def test_player_uses_requested_frame_range(self):
        source = self.wav(channels=1, rate=16000)
        target = self.root / "clip.wav"
        extract_clip(source, target, .2, .7)
        with wave.open(str(target)) as clip:
            self.assertEqual(clip.getnframes(), 8000)
        with self.assertRaises(ValueError):
            extract_clip(source, target, .2, 1.1)

    def test_cleanup_is_scoped_to_one_valid_session(self):
        service = BaseService.__new__(BaseService)
        service.root = self.root; service.data_root = self.root / "data/user_uploads"; service.data_root.mkdir(parents=True)
        first, second = service.new_session(), service.new_session()
        for state in [first, second]:
            path = service.session_directory(state); path.mkdir(); (path / "keep.txt").write_text("test")
        service.clear_upload(first)
        self.assertFalse(service.session_directory(first).exists())
        self.assertTrue(service.session_directory(second).exists())
        with self.assertRaises(ValueError):
            service.clear_upload({"session_id": "../../"})


if __name__ == "__main__":
    unittest.main()
