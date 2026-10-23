"""Shared model inference, session storage, search and audio playback."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import re
import shutil
import threading
import uuid
from pathlib import Path
import numpy as np
from .io import CONFIG_ROOT, DEFAULT_ROOT, read_json, write_json
from .audio import extract_clip, playback_ranges
from .retrieval import BM25Search, WORD_RE, deduplicate_ranking, load_tokenizer, token_count
from .semantic import cached_encode, load_model, model_cache_identity, rrf_rankings, semantic_ranking

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


class BaseService:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.shared = read_json(CONFIG_ROOT / "retrieval.json")
        self.semantic = read_json(CONFIG_ROOT / "semantic.json")
        self.config = read_json(CONFIG_ROOT / "upload.json")
        from .fetch_semantic_model import fetch
        report = lambda message: print(message, flush=True)
        report("[1/4] Checking and downloading the search model...")
        fetch(self.root, report=report)
        self.tokenizer = load_tokenizer(self.root / self.semantic["model_directory"] / "tokenizer.json")
        report("[2/4] Loading the search encoder on CPU...")
        self.encoder, self.model_report = load_model(self.root, self.semantic)
        self.encoder_identity = model_cache_identity(self.semantic, self.model_report)
        self.data_root = (self.root / self.config["user_data_directory"]).resolve()
        self.data_root.relative_to(self.root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.asr = None
        self.lock = threading.RLock()

    def choices(self, state):
        if not state.get("upload_key"):
            return []
        _, manifest = self.uploaded(state)
        return [(manifest["original_filename"], "uploaded")]

    def new_session(self):
        return {"session_id": uuid.uuid4().hex, "upload_key": None, "results": [], "selected_source": None}

    def session_directory(self, state):
        identifier = state["session_id"]
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Invalid session identity")
        directory = (self.data_root / identifier).resolve()
        directory.relative_to(self.data_root)
        return directory

    def upload_identity(self, source):
        return {"audio_sha256": file_hash(source), "upload_config": self.config,
                "asr_packages": {p: importlib.metadata.version(p) for p in ["faster-whisper", "ctranslate2", "av", "onnxruntime"]},
                "chunk_config": self.shared, "encoder": self.encoder_identity}

    def load_asr(self):
        if self.asr is None:
            from .fetch_asr_model import fetch
            report = lambda message: print(message, flush=True)
            report("[3/4] Checking and downloading the speech model...")
            fetch(self.root, report=report)
            from faster_whisper import WhisperModel
            report("[4/4] Loading speech recognition on CPU...")
            self.asr = WhisperModel(str(self.root / self.config["asr_model_directory"]), device="cpu",
                compute_type="int8", cpu_threads=self.config["asr_cpu_threads"], local_files_only=True)
        return self.asr

    def transcribe(self, path, duration, progress):
        model = self.load_asr()
        iterator, info = model.transcribe(str(path), language="en", task="transcribe",
            beam_size=self.config["asr_beam_size"], word_timestamps=True, vad_filter=True,
            condition_on_previous_text=False, temperature=0)
        segments = []
        for segment in iterator:
            text, timed = "", []
            for word in segment.words or []:
                first = len(text)
                text += word.word
                timed.append({"char_start": first, "char_end": len(text), "start": float(word.start),
                              "end": float(word.end), "text": word.word, "probability": float(word.probability)})
            if not text:
                text = segment.text
            segments.append({"text": text, "speakerName": "Unknown", "timed_words": timed,
                "segment_start": float(segment.start), "segment_end": float(segment.end),
                "avg_logprob": float(segment.avg_logprob), "no_speech_prob": float(segment.no_speech_prob)})
            progress(.15+.65*min(segment.end/duration, 1), "Transcribing English speech...")
        return segments, {"language": info.language, "language_probability": info.language_probability,
                          "duration_after_vad": info.duration_after_vad, "segment_count": len(segments)}

    def uploaded(self, state):
        key = state.get("upload_key")
        if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("Upload a recording and wait until it is ready.")
        directory = self.session_directory(state) / key
        manifest = read_json(directory / "manifest.json")
        return directory, manifest

    def search(self, source, query, method, state):
        if method not in ["BM25", "SBERT", "Hybrid"]:
            raise ValueError("Choose BM25, SBERT, or Hybrid.")
        if not WORD_RE.search(query) or token_count(self.tokenizer, query) > self.semantic["max_seq_length"]:
            raise ValueError("Enter an English question. If it is too long, shorten it and try again.")
        with self.lock:
            if source != "uploaded":
                raise ValueError("Upload and process a recording before searching.")
            directory, manifest = self.uploaded(state)
            chunks = read_json(directory / "chunks.json")
            vector_path = directory / manifest["vector_array"]
            metadata = read_json(directory / manifest["vector_metadata"])
            if file_hash(vector_path) != metadata["array_sha256"]:
                raise ValueError("The search index failed verification. Process the recording again.")
            vectors = np.load(vector_path, allow_pickle=False)
            bm25, kept = BM25Search(chunks, self.shared).search(query, chunks[0]["meeting_id"])
            if method != "BM25":
                query_vector = self.encoder.encode_query([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
                full = semantic_ranking(chunks, vectors, query_vector)
                if method == "Hybrid":
                    full = rrf_rankings(bm25, full, self.semantic["rrf_K"], self.semantic["rrf_weights"])
                kept = deduplicate_ranking(full, self.shared["ranking_depth"], self.shared["dedup_overlap_threshold"])
            return kept[:5]

    def ranges_for_result(self, result, source, state):
        if source != "uploaded" or result["time_mapping_status"] != "asr_estimated":
            return []
        _, manifest = self.uploaded(state)
        return playback_ranges(result["time_ranges"], manifest["audio"]["duration_seconds"],
                               self.config["playback_buffer_seconds"], self.config["max_playback_clip_seconds"])

    def result_clip(self, result, interval_index, source, state):
        ranges = self.ranges_for_result(result, source, state)
        if not 0 <= interval_index < len(ranges):
            raise ValueError("This result has no playable interval.")
        directory, _ = self.uploaded(state)
        start, end = ranges[interval_index]
        clip_key = hashlib.sha256(f"{state['session_id']}:{state['upload_key']}:{result['chunk_id']}:{start}:{end}".encode()).hexdigest()
        target = self.root / "output/demo-clips" / state["session_id"] / (clip_key+".wav")
        return extract_clip(directory / "audio.wav", target, start, end), [start, end]

    def clear_upload(self, state):
        # Deletion is limited to this validated session directory and its derived clips.
        directory = self.session_directory(state)
        clips = (self.root / "output/demo-clips" / state["session_id"]).resolve()
        clips.relative_to((self.root / "output/demo-clips").resolve())
        for target, boundary in [(directory, self.data_root), (clips, (self.root / "output/demo-clips").resolve())]:
            resolved = target.resolve()
            resolved.relative_to(boundary)
            if resolved == boundary or target.is_symlink():
                raise ValueError("Invalid cleanup target")
            if target.exists():
                shutil.rmtree(target)
        return {**state, "upload_key": None, "results": [], "selected_source": None}
