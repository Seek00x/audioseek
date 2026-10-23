"""Recording processing with verified resumable transcription blocks."""
from __future__ import annotations
import copy
import hashlib
import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from .base import BaseService, file_hash
from .audio import UploadValidationError, decode_audio, extract_clip, chunk_time_ranges, word_time_coordinates
from .io import read_json, write_json
from .retrieval import build_chunks, canonical_words
from .semantic import cached_encode, validate_model_tokens

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def offset_segments(segments, offset):
    adjusted = copy.deepcopy(segments)
    for segment in adjusted:
        segment["segment_start"] += offset
        segment["segment_end"] += offset
        for word in segment["timed_words"]:
            word["start"] += offset
            word["end"] += offset
    return adjusted


class RecordingService(BaseService):
    def transcribe(self, path, duration, progress):
        path = Path(path).resolve()
        checkpoint_dir = path.parent / "asr_blocks"
        checkpoint_dir.mkdir(exist_ok=True)
        with wave.open(str(path)) as audio:
            rate, frames = audio.getframerate(), audio.getnframes()
        block_frames = int(self.config["asr_block_seconds"] * rate)
        count = (frames + block_frames - 1) // block_frames
        identity = {"audio_sha256": file_hash(path), "config": self.config,
                    "packages": self.upload_identity(path)["asr_packages"], "frames": frames, "sample_rate": rate}
        segments, reused, speech_seconds = [], 0, 0.0
        probabilities = []
        for index in range(count):
            start_frame, end_frame = index * block_frames, min(frames, (index + 1) * block_frames)
            start, end = start_frame / rate, end_frame / rate
            expected = {"identity": identity, "block_index": index, "start_frame": start_frame, "end_frame": end_frame}
            checkpoint = checkpoint_dir / f"block_{index:04d}.json"
            if checkpoint.exists():
                saved = read_json(checkpoint)
                payload = saved["payload"]
                if canonical_hash(payload) != saved["sha256"] or payload["header"] != expected:
                    raise UploadValidationError("A saved transcription block failed verification. Clear the upload and try again.")
                block_segments, info = payload["segments"], payload["info"]
                reused += 1
                progress(.15 + .65 * end / duration, f"Restoring completed block {index+1}/{count}...")
            else:
                block_audio = checkpoint_dir / f"block_{index:04d}.wav"
                try:
                    extract_clip(path, block_audio, start, end)

                    def block_progress(value, message):
                        fraction = max(0.0, min(1.0, (value - .15) / .65))
                        progress(.15 + .65 * (start + fraction * (end - start)) / duration,
                                 f"Transcribing block {index+1}/{count} ({start/60:g}-{end/60:g} minutes)...")

                    local_segments, info = super().transcribe(block_audio, end - start, block_progress)
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
        return segments, {"language": "en", "language_probability": min(probabilities),
            "duration_after_vad": speech_seconds, "segment_count": len(segments), "block_count": count,
            "completed_blocks_reused": reused, "block_seconds": self.config["asr_block_seconds"],
            "time_origin": "ASR_block_estimates_plus_exact_file_offsets", "segmentation": "nonoverlapping_blocks"}

    def process_upload(self, source, state, progress=lambda value, description: None):
        source = Path(source)
        if source.suffix.lower() not in self.config["allowed_extensions"] or not source.is_file():
            raise UploadValidationError("Upload an English WAV, MP3, M4A, FLAC, or OGG recording.")
        if source.stat().st_size > self.config["max_file_bytes"]:
            raise UploadValidationError("The file exceeds 512 MiB. Split it before uploading.")
        identity = self.upload_identity(source)
        key = canonical_hash(identity)
        directory = self.session_directory(state) / key
        manifest_path = directory / "manifest.json"
        with self.lock:
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                if manifest["identity"] != identity or manifest["status"] != "ready":
                    raise UploadValidationError("The recording cache does not match. Clear the upload and try again.")
                for item in manifest["files"]:
                    cached_path = (directory / item["path"]).resolve()
                    cached_path.relative_to(directory.resolve())
                    if file_hash(cached_path) != item["sha256"]:
                        raise UploadValidationError("The recording cache failed verification. Clear the upload and try again.")
                return {**manifest, "cache_reused": True, "directory": str(directory), "key": key}
            directory.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            progress(.03, "Checking and decoding the recording...")
            audio = decode_audio(source, directory / "audio.wav", self.config)
            decoded = time.perf_counter()
            progress(.12, "Preparing speech recognition...")
            segments, info = self.transcribe(directory / "audio.wav", audio["duration_seconds"], progress)
            transcribed = time.perf_counter()
            meeting = {"meeting_id": "upload:" + key[:16], "corpus": "USER_UPLOAD", "split": "application", "segments": segments}
            words, _ = canonical_words(segments)
            if not words:
                raise UploadValidationError("No usable English speech was recognized. Check the recording.")
            progress(.85, "Building the search index...")
            chunks, _, _ = build_chunks(meeting, self.tokenizer, self.shared)
            times = word_time_coordinates(segments, audio["duration_seconds"])
            for chunk in chunks:
                chunk["time_ranges"] = chunk_time_ranges(chunk, times, self.config["merge_audio_gap_seconds"])
                chunk["time_mapping_status"] = "asr_estimated" if chunk["time_ranges"] else "unavailable"
            records = [{"id": chunk["chunk_id"], "text": chunk["text"]} for chunk in chunks]
            tokens = validate_model_tokens(self.encoder, self.tokenizer, records, self.semantic["max_seq_length"])
            # Keep hashed temporary filenames outside the deeply nested upload-key
            # directory. Short final names avoid Windows MAX_PATH failures while
            # the original full cache identity and SHA checks remain in metadata.
            _, vectors = cached_encode(self.encoder, records, "document", self.encoder_identity,
                                       self.session_directory(state) / "_vectors")
            vector_directory = directory / "vectors"
            vector_directory.mkdir(exist_ok=True)
            for name, short in [("array_path", "document.npy"), ("metadata_path", "document.json")]:
                final = vector_directory / short
                Path(vectors[name]).replace(final)
                vectors[name] = str(final)
            write_json(directory / "transcript.json", {"source": "automatic_asr", "human_verified": False,
                "segments": segments, "word_times": times, "asr_info": info, "audio": audio})
            write_json(directory / "chunks.json", chunks)
            (directory / "transcript.txt").write_text("\n".join(s["text"].strip() for s in segments) + "\n", "utf-8")
            vector_paths = [Path(vectors[k]).relative_to(directory).as_posix() for k in ["array_path", "metadata_path"]]
            files = ["audio.wav", "transcript.json", "transcript.txt", "chunks.json", *vector_paths]
            files.extend(p.relative_to(directory).as_posix() for p in sorted((directory / "asr_blocks").glob("block_*.json")))
            manifest = {"status": "ready", "created_at_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
                "original_filename": source.name, "meeting_id": meeting["meeting_id"], "audio": audio, "asr_info": info,
                "chunk_count": len(chunks), "word_count": len(words), "token_check": tokens,
                "time_source": "asr_estimate_not_human_verified",
                "vectors": {k: v for k, v in vectors.items() if k not in ["array_path", "metadata_path"]},
                "vector_array": vector_paths[0], "vector_metadata": vector_paths[1],
                "timings": {"decode_seconds": decoded - started, "asr_and_first_model_load_seconds": transcribed - decoded,
                    "index_and_save_seconds": time.perf_counter() - transcribed, "total_seconds": time.perf_counter() - started},
                "files": [{"path": p, "sha256": file_hash(directory / p), "bytes": (directory / p).stat().st_size} for p in files]}
            write_json(manifest_path, manifest)
            progress(1, "Ready to search.")
            return {**manifest, "cache_reused": False, "directory": str(directory), "key": key}
