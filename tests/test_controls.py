"""Stop requests preserve complete blocks and remain scoped to their session."""
import json
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
from audioseek.base import BaseService
from audioseek.service import RecordingService
from audioseek.audio import UploadValidationError
from audioseek.controls import InteractiveService, UploadStopped, lexical_notice


class UploadControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = InteractiveService.__new__(InteractiveService)
        self.service.root = self.root
        self.service.data_root = self.root/"uploads"
        self.service.config = {**json.loads((ROOT/"audioseek/config/upload.json").read_text("utf-8")), "asr_block_seconds": 30}
        self.service.upload_identity = lambda path: {"asr_packages": {"test": "fixed"}}
        self.service._operations = {}
        self.service._operation_lock = threading.Lock()
        self.service._current_operation = threading.local()
        self.state = self.service.new_session()
        self.source = self.root/"source.wav"
        with wave.open(str(self.source), "wb") as audio:
            audio.setparams((1,2,8000,0,"NONE","not compressed"))
            audio.writeframes(b"\0\0"*(65*8000))

    def tearDown(self):
        self.temp.cleanup()

    def transcribe(self, obj, path, duration, progress):
        return [{"segment_start": .1, "segment_end": .5, "text": "test", "timed_words":
                 [{"start": .1, "end": .5, "char_start": 0, "char_end": 4}]}], {
                 "duration_after_vad": duration, "language_probability": 1.0}

    @staticmethod
    def base_process(obj, source, state, progress):
        return obj.transcribe(source, 65, progress)

    def test_stop_saves_current_block_and_retry_reuses_it(self):
        service = self.service
        calls = []
        def stop_during_first(obj, path, duration, progress):
            calls.append(duration)
            self.assertTrue(service.request_stop(self.state))
            return self.transcribe(obj,path,duration,progress)
        with patch.object(RecordingService,"process_upload",self.base_process), patch.object(BaseService,"transcribe",stop_during_first):
            with self.assertRaises(UploadStopped):
                service.process_upload(self.source,self.state)
        self.assertEqual(calls,[30])
        self.assertEqual(len(list((self.root/"asr_blocks").glob("*.json"))),1)
        self.assertFalse(list((self.root/"asr_blocks").glob("*.wav")))
        self.assertFalse(service.request_stop(self.state))
        calls.clear()
        def finish(obj,path,duration,progress):
            calls.append(duration)
            return self.transcribe(obj,path,duration,progress)
        with patch.object(RecordingService,"process_upload",self.base_process), patch.object(BaseService,"transcribe",finish):
            segments,info=service.process_upload(self.source,self.state)
        self.assertEqual(calls,[30,5])
        self.assertEqual(info["completed_blocks_reused"],1)
        self.assertEqual([s["segment_start"] for s in segments],[.1,30.1,60.1])

    def test_stop_is_session_scoped_and_blocks_clearing_active_work(self):
        other = self.service.new_session()
        event = threading.Event()
        self.service._operations[self.state["session_id"]] = event
        self.assertFalse(self.service.request_stop(other))
        self.assertFalse(event.is_set())
        self.assertTrue(self.service.request_stop(self.state))
        self.assertTrue(event.is_set())
        with self.assertRaises(UploadValidationError):
            self.service.clear_upload(self.state)
        with self.assertRaises(ValueError):
            self.service.request_stop({"session_id":"../outside"})

    def test_same_session_cannot_launch_concurrent_uploads(self):
        self.service._operations[self.state["session_id"]]=threading.Event()
        with patch.object(RecordingService,"process_upload",side_effect=AssertionError("must not start another job")):
            with self.assertRaisesRegex(UploadValidationError,"already processing"):
                self.service.process_upload(self.source,self.state)

    def test_stop_at_decode_boundary_and_operation_cleanup(self):
        def base(obj,source,state,progress):
            obj.request_stop(state)
            progress(.12,"Preparing speech recognition...")
            raise AssertionError("must stop before speech recognition")
        with patch.object(RecordingService,"process_upload",base):
            with self.assertRaises(UploadStopped):
                self.service.process_upload(self.source,self.state)
        self.assertEqual(self.service._operations,{})

    def test_lexical_notice_reports_matches_without_answer_or_confidence_claim(self):
        results=[{"text":"The remote uses a rechargeable battery."}]
        self.assertTrue(lexical_notice("photosynthesis",results))
        self.assertEqual(lexical_notice("battery",results),"")
        self.assertEqual(lexical_notice("",results),"")
        self.assertEqual(lexical_notice("battery",[]),"")


if __name__ == "__main__":
    unittest.main()
