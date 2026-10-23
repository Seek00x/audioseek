"""Capacity, block boundaries and recovery checks independent of model weights."""
import copy
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from audioseek.service import RecordingService, offset_segments
from audioseek.audio import UploadValidationError, decode_audio
from audioseek.base import BaseService


class ExtendedUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = json.loads((Path(__file__).resolve().parents[1]/"audioseek/config/upload.json").read_text("utf-8"))

    def tearDown(self):
        self.temp.cleanup()

    def wav(self, seconds=2, rate=8000):
        path = self.root / "source.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x01\x00" * int(seconds*rate))
        return path

    def test_exact_duration_allowed_and_overflow_leaves_no_partial(self):
        target = self.root / "decoded.wav"
        config = {**self.config, "max_duration_seconds": 2}
        audio = decode_audio(self.wav(), target, config)
        self.assertEqual(audio["duration_seconds"], 2)
        target.unlink()
        with self.assertRaises(UploadValidationError):
            decode_audio(self.wav(2.1), target, config)
        self.assertFalse(target.exists())
        self.assertFalse(target.with_suffix(".wav.part").exists())

    def test_file_cap_rejects_before_decode(self):
        source = self.wav()
        with patch("av.open", side_effect=AssertionError("must reject before decoding")):
            with self.assertRaises(UploadValidationError) as error:
                decode_audio(source, self.root/"decoded.wav", {**self.config, "max_file_bytes": 1024})
        self.assertIn("MiB", error.exception.display_message)

    def test_offset_preserves_character_coordinates_and_input(self):
        segment = [{"segment_start": .2, "segment_end": .9, "text": " hi", "timed_words":
                    [{"start": .2, "end": .9, "char_start": 0, "char_end": 3}]}]
        before = copy.deepcopy(segment)
        adjusted = offset_segments(segment, 3600)
        self.assertEqual(segment, before)
        self.assertEqual(adjusted[0]["timed_words"][0]["start"], 3600.2)
        self.assertEqual(adjusted[0]["timed_words"][0]["char_end"], 3)

    def service(self):
        service = RecordingService.__new__(RecordingService)
        service.config = {**self.config, "asr_block_seconds": 30}
        service.upload_identity = lambda source: {"asr_packages": {"test": "fixed"}}
        return service

    @staticmethod
    def fake_transcribe(service, path, duration, progress):
        with wave.open(str(path)) as audio:
            assert audio.getnframes()/audio.getframerate() == duration
        return [{"segment_start": .25, "segment_end": .75, "text": "hello", "timed_words":
                 [{"start": .25, "end": .75, "char_start": 0, "char_end": 5}]}], {
                 "duration_after_vad": duration, "language_probability": 1.0}

    def test_partial_failure_retries_only_unfinished_blocks_with_absolute_times(self):
        source, service = self.wav(65), self.service()
        calls = []
        def fail_second(obj, path, duration, progress):
            calls.append(duration)
            if len(calls) == 2:
                raise RuntimeError("injected interruption")
            return self.fake_transcribe(obj, path, duration, progress)
        with patch.object(BaseService, "transcribe", fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                service.transcribe(source, 65, lambda *args: None)
        checkpoints = self.root / "asr_blocks"
        self.assertEqual(len(list(checkpoints.glob("*.json"))), 1)
        self.assertFalse(list(checkpoints.glob("*.wav")))
        resumed = []
        def success(obj, path, duration, progress):
            resumed.append(duration)
            return self.fake_transcribe(obj, path, duration, progress)
        with patch.object(BaseService, "transcribe", success):
            segments, info = service.transcribe(source, 65, lambda *args: None)
        self.assertEqual(resumed, [30, 5])
        self.assertEqual([s["timed_words"][0]["start"] for s in segments], [.25, 30.25, 60.25])
        self.assertEqual(info["completed_blocks_reused"], 1)
        self.assertEqual(info["block_count"], 3)

    def test_checkpoint_tampering_is_rejected(self):
        source, service = self.wav(), self.service()
        with patch.object(BaseService, "transcribe", self.fake_transcribe):
            service.transcribe(source, 2, lambda *args: None)
        checkpoint = self.root/"asr_blocks/block_0000.json"
        data = json.loads(checkpoint.read_text("utf-8"))
        data["payload"]["segments"][0]["text"] = "changed"
        checkpoint.write_text(json.dumps(data), "utf-8")
        with patch.object(BaseService, "transcribe", side_effect=AssertionError("must not run ASR")):
            with self.assertRaisesRegex(UploadValidationError, "verification"):
                service.transcribe(source, 2, lambda *args: None)


if __name__ == "__main__":
    unittest.main()
