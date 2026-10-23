"""Streaming decoding, word timestamps and bounded WAV playback."""
from __future__ import annotations
import math
import wave
from pathlib import Path
from .retrieval import canonical_words

class UploadValidationError(ValueError):
    def __init__(self, message):
        self.display_message = message
        super().__init__(message)


def decode_audio(source, target, config):
    """Stream decode; enforce source bytes and actual decoded frames independently."""
    import av
    source, target = Path(source), Path(target)
    minutes = config["max_duration_seconds"] / 60
    mib = config["max_file_bytes"] / 1024**2
    if source.suffix.lower() not in config["allowed_extensions"]:
        raise UploadValidationError("Upload an English WAV, MP3, M4A, FLAC, or OGG recording.")
    if not source.is_file() or source.stat().st_size == 0:
        raise UploadValidationError("The recording is empty or unavailable. Choose another file.")
    if source.stat().st_size > config["max_file_bytes"]:
        raise UploadValidationError(f"The file exceeds {mib:g} MiB. Split it before uploading.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".wav.part")
    sample_rate, samples = config["sample_rate"], 0
    try:
        with av.open(str(source)) as container, wave.open(str(temporary), "wb") as output:
            output.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
            if not container.streams.audio:
                raise UploadValidationError("The file contains no decodable audio track.")
            if container.duration is not None and container.duration / av.time_base > config["max_duration_seconds"] + .5:
                raise UploadValidationError(f"The recording exceeds {minutes:g} minutes. Split it first; audio will not be truncated.")
            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)

            def write(frames):
                nonlocal samples
                for frame in frames:
                    samples += frame.samples
                    if samples > sample_rate * config["max_duration_seconds"]:
                        raise UploadValidationError(f"The decoded recording exceeds {minutes:g} minutes. Split it first; audio will not be truncated.")
                    output.writeframesraw(frame.to_ndarray().astype("<i2", copy=False).tobytes())

            for frame in container.decode(audio=0):
                write(resampler.resample(frame))
            write(resampler.resample(None))
        if not samples:
            raise UploadValidationError("No usable audio was found. Choose another recording.")
        temporary.replace(target)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        if isinstance(error, UploadValidationError):
            raise
        raise UploadValidationError("The recording could not be decoded. Check the file or try another supported format.") from error
    return {"duration_seconds": samples / sample_rate, "sample_rate": sample_rate, "channels": 1,
            "sample_width_bytes": 2, "frames": samples, "source_bytes": source.stat().st_size,
            "normalized_bytes": target.stat().st_size}


def word_time_coordinates(segments, duration):
    words, _ = canonical_words(segments)
    times = []
    for word in words:
        pieces = [t for t in segments[word["turn_id"]]["timed_words"] if
                  t["char_start"] < word["char_end"] and t["char_end"] > word["char_start"]]
        covered = sum(min(t["char_end"], word["char_end"])-max(t["char_start"], word["char_start"]) for t in pieces)
        valid = pieces and covered == word["char_end"]-word["char_start"] and all(
            math.isfinite(t["start"]) and math.isfinite(t["end"]) and 0 <= t["start"] <= t["end"] <= duration for t in pieces)
        times.append({"start": min(t["start"] for t in pieces), "end": max(t["end"] for t in pieces)} if valid else None)
    return times


def chunk_time_ranges(chunk, times, max_gap=2):
    selected = times[chunk["word_start"]:chunk["word_end"]]
    if not selected or any(t is None for t in selected):
        return None
    ranges = []
    for t in selected:
        if ranges and t["start"] <= ranges[-1][1]+max_gap:
            ranges[-1][1] = max(ranges[-1][1], t["end"])
        else:
            ranges.append([t["start"], t["end"]])
    return [r for r in ranges if r[1] > r[0]] or None


def playback_ranges(ranges, duration, buffer=1, maximum=90):
    result = []
    for start, end in ranges or []:
        start, end = max(0, start-buffer), min(duration, end+buffer)
        while start < end:
            stop = min(end, start+maximum)
            result.append([start, stop])
            start = stop
    return result


def extract_clip(source, target, start, end):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as audio:
        rate, frames = audio.getframerate(), audio.getnframes()
        if not 0 <= start < end <= frames/rate:
            raise ValueError("The playback interval is outside the recording.")
        first, last = int(start*rate), min(frames, math.ceil(end*rate))
        audio.setpos(first)
        with wave.open(str(target), "wb") as output:
            output.setparams(audio.getparams())
            output.writeframes(audio.readframes(last-first))
    return target
