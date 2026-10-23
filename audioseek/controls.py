"""Session-scoped stopping, block recovery and candidate notices."""
from __future__ import annotations
import threading
import wave
from pathlib import Path

from .base import BaseService as OriginalService, file_hash
from .audio import extract_clip
from .service import RecordingService, canonical_hash, offset_segments
from .audio import UploadValidationError
from .io import read_json, write_json
from .retrieval import lexical_tokens


class UploadStopped(UploadValidationError):
    def __init__(self):
        super().__init__("Processing stopped. Any completed transcription blocks are saved. Click Retry processing to continue with this file.")


def lexical_notice(question, results):
    """Report absence of literal token matches, without predicting answer correctness."""
    tokens = set(lexical_tokens(question))
    candidates = {word for result in results for word in lexical_tokens(result["text"])}
    if tokens and results and not tokens.intersection(candidates):
        return "No exact query-word matches in these candidates. They may use different wording or be unrelated; verify the evidence by reading and listening."
    return ""


class InteractiveService(RecordingService):
    def __init__(self, root):
        super().__init__(root)
        self._operations = {}
        self._operation_lock = threading.Lock()
        self._current_operation = threading.local()

    def request_stop(self, state):
        self.session_directory(state)  # Validate the session without acquiring the busy ASR lock.
        with self._operation_lock:
            operation = self._operations.get(state["session_id"])
            if operation is None:
                return False
            operation.set()
            return True

    def _check_stop(self):
        operation = getattr(self._current_operation, "event", None)
        if operation is not None and operation.is_set():
            raise UploadStopped()

    def process_upload(self, source, state, progress=lambda value, message: None):
        self.session_directory(state)
        identifier = state["session_id"]
        event = threading.Event()
        with self._operation_lock:
            if identifier in self._operations:
                raise UploadValidationError("This session is already processing a recording. Wait for it to stop or finish.")
            self._operations[identifier] = event
        self._current_operation.event = event
        def controlled_progress(value, message):
            if value in [.03, .12, .85]:
                self._check_stop()
            progress(value, message)
        try:
            self._check_stop()
            return super().process_upload(source, state, controlled_progress)
        finally:
            del self._current_operation.event
            with self._operation_lock:
                self._operations.pop(identifier, None)

    def clear_upload(self, state):
        with self._operation_lock:
            if state["session_id"] in self._operations:
                raise UploadValidationError("Stop processing first, then clear the upload after it has stopped.")
            return super().clear_upload(state)

    def transcribe(self, path, duration, progress):
        """Finish and save the current block before honoring a stop request."""
        path = Path(path).resolve()
        checkpoint_dir = path.parent / "asr_blocks"
        checkpoint_dir.mkdir(exist_ok=True)
        with wave.open(str(path)) as audio:
            rate, frames = audio.getframerate(), audio.getnframes()
        block_frames = int(self.config["asr_block_seconds"] * rate)
        count = (frames + block_frames - 1)//block_frames
        identity = {"audio_sha256": file_hash(path), "config": self.config,
                    "packages": self.upload_identity(path)["asr_packages"], "frames": frames, "sample_rate": rate}
        segments, reused, speech_seconds, probabilities = [], 0, 0.0, []
        for index in range(count):
            self._check_stop()
            start_frame, end_frame = index*block_frames, min(frames, (index+1)*block_frames)
            start, end = start_frame/rate, end_frame/rate
            expected = {"identity": identity, "block_index": index, "start_frame": start_frame, "end_frame": end_frame}
            checkpoint = checkpoint_dir/f"block_{index:04d}.json"
            if checkpoint.exists():
                saved = read_json(checkpoint)
                payload = saved["payload"]
                if canonical_hash(payload) != saved["sha256"] or payload["header"] != expected:
                    raise UploadValidationError("A saved transcription block failed verification. Clear the upload and try again.")
                block_segments, info = payload["segments"], payload["info"]
                reused += 1
                progress(.15+.65*end/duration, f"Restoring completed block {index+1}/{count}...")
            else:
                block_audio = checkpoint_dir/f"block_{index:04d}.wav"
                try:
                    extract_clip(path, block_audio, start, end)
                    def block_progress(value, message):
                        fraction = max(0.0, min(1.0, (value-.15)/.65))
                        progress(.15+.65*(start+fraction*(end-start))/duration,
                                 f"Transcribing block {index+1}/{count} ({start/60:g}-{end/60:g} minutes)...")
                    local_segments, info = OriginalService.transcribe(self, block_audio, end-start, block_progress)
                    block_segments = offset_segments(local_segments, start)
                    payload = {"header": expected, "segments": block_segments, "info": info}
                    temporary = checkpoint.with_suffix(".json.part")
                    write_json(temporary, {"payload": payload, "sha256": canonical_hash(payload)})
                    temporary.replace(checkpoint)
                finally:
                    block_audio.unlink(missing_ok=True)
            segments.extend(block_segments)
            speech_seconds += info["duration_after_vad"]
            probabilities.append(info["language_probability"])
            self._check_stop()
        return segments, {"language": "en", "language_probability": min(probabilities),
            "duration_after_vad": speech_seconds, "segment_count": len(segments), "block_count": count,
            "completed_blocks_reused": reused, "block_seconds": self.config["asr_block_seconds"],
            "time_origin": "ASR_block_estimates_plus_exact_file_offsets", "segmentation": "nonoverlapping_blocks"}
