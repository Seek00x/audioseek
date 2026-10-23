"""Word coordinates, token-limited chunks and lexical passage retrieval."""
from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections import defaultdict

from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
VERSION = "retrieval-1"


def lexical_tokens(text: str) -> list[str]:
    return [m.group().lower().replace("’", "'") for m in WORD_RE.finditer(text)]


def canonical_words(segments: list[dict]) -> tuple[list[dict], list[int]]:
    words, turn_offsets = [], [0]
    for turn_id, segment in enumerate(segments):
        for word_index, match in enumerate(WORD_RE.finditer(segment["text"])):
            words.append({"turn_id": turn_id, "word_index": word_index,
                          "char_start": match.start(), "char_end": match.end(),
                          "text": match.group(), "token": match.group().lower().replace("’", "'"),
                          "speaker": segment["speakerName"]})
        turn_offsets.append(len(words))
    return words, turn_offsets


def load_tokenizer(path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    if len(tokenizer.encode("", add_special_tokens=True).ids) != 2:
        raise ValueError("Expected the checkpoint's CLS/SEP special-token pair")
    return tokenizer


def token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True).ids)


def surface_span(segments: list[dict], words: list[dict], offsets: list[int], start: int, end: int):
    if not 0 <= start < end <= len(words):
        raise ValueError("Invalid word span")
    first_turn, last_turn = words[start]["turn_id"], words[end - 1]["turn_id"]
    parts, spans = [], []
    for turn_id in range(first_turn, last_turn + 1):
        segment = segments[turn_id]
        left, right = max(start, offsets[turn_id]), min(end, offsets[turn_id + 1])
        if left < right:
            char_start = 0 if left == offsets[turn_id] else words[left]["char_start"]
            # Retain punctuation after the last included word, but not the next word.
            char_end = len(segment["text"]) if right == offsets[turn_id + 1] else words[right]["char_start"]
        else:
            char_start, char_end = 0, len(segment["text"])
        text = segment["text"][char_start:char_end].strip()
        if text:
            parts.append(text)
        spans.append({"turn_id": turn_id, "word_start": max(0, left - offsets[turn_id]),
                      "word_end": max(0, right - offsets[turn_id]), "char_start": char_start,
                      "char_end": char_end, "speaker": segment["speakerName"]})
    return "\n".join(parts), spans


def build_chunks(meeting: dict, tokenizer: Tokenizer, config: dict):
    segments = meeting["segments"]
    words, offsets = canonical_words(segments)
    if not words:
        raise ValueError(f"Empty meeting: {meeting['meeting_id']}")
    target, cap = config["target_words"], config["max_tokens_including_special"]
    fraction = config["overlap_fraction"]
    if target < 1 or cap < 3 or not 0 <= fraction < 0.8:
        raise ValueError("Invalid chunk parameters")
    minimum = max(1, math.ceil(target * config["turn_boundary_min_fraction"]))
    boundaries = sorted(set(offsets))
    chunks, start = [], 0
    while start < len(words):
        end = min(start + target, len(words))
        boundary = boundaries[bisect_right(boundaries, end) - 1]
        if end < len(words) and boundary - start >= minimum:
            end = boundary
        text, spans = surface_span(segments, words, offsets, start, end)
        length = token_count(tokenizer, text)
        shortened_for_tokens = False
        while length > cap and end > start + 1:
            end -= 1
            shortened_for_tokens = True
            text, spans = surface_span(segments, words, offsets, start, end)
            length = token_count(tokenizer, text)
        if length > cap:
            raise ValueError("One canonical word exceeds the token cap; no silent truncation")
        if shortened_for_tokens:
            boundary = boundaries[bisect_right(boundaries, end) - 1]
            if boundary - start >= minimum:
                end = boundary
                text, spans = surface_span(segments, words, offsets, start, end)
                length = token_count(tokenizer, text)
        if lexical_tokens(text) != [w["token"] for w in words[start:end]] or length > cap:
            raise ValueError("Chunk text and canonical coordinates disagree")
        chunks.append({"chunk_id": f"{meeting['meeting_id']}:w{start:06d}-{end:06d}",
                       "meeting_id": meeting["meeting_id"], "corpus": meeting["corpus"],
                       "word_start": start, "word_end": end, "word_count": end - start,
                       "token_count": length, "text": text, "source_spans": spans,
                       "turn_ids": [s["turn_id"] for s in spans],
                       "speaker_ids": sorted({s["speaker"] for s in spans}),
                       "start_inside_turn": start != offsets[words[start]["turn_id"]],
                       "end_inside_turn": end != offsets[words[end - 1]["turn_id"] + 1],
                       "shortened_for_token_cap": shortened_for_tokens,
                       "time_mapping_status": "not_audited", "time_ranges": None})
        if end == len(words):
            break
        desired = max(start + 1, end - int((end - start) * fraction + 0.5))
        nearby = [b for b in boundaries if start < b < end and abs(b - desired) <= max(3, (end - start) * .08)]
        start = min(nearby, key=lambda b: (abs(b - desired), b)) if nearby else desired
    return chunks, words, offsets










def stable_ranking(chunks: list[dict], scores) -> list[dict]:
    if len(chunks) != len(scores) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("Invalid retrieval scores")
    return sorted(({**chunk, "score": float(score)} for chunk, score in zip(chunks, scores)),
                  key=lambda c: (-c["score"], c["word_start"], c["word_end"], c["chunk_id"]))


def deduplicate_ranking(ranking: list[dict], k: int, threshold: float = .8) -> list[dict]:
    if k < 1 or not 0 < threshold <= 1:
        raise ValueError("Invalid deduplication parameters")
    kept = []
    for candidate in ranking:
        duplicate = False
        for prior in kept:
            if candidate["meeting_id"] != prior["meeting_id"]:
                continue
            overlap = max(0, min(candidate["word_end"], prior["word_end"]) - max(candidate["word_start"], prior["word_start"]))
            denominator = min(candidate["word_end"] - candidate["word_start"], prior["word_end"] - prior["word_start"])
            if denominator <= 0:
                raise ValueError("Empty candidate range")
            if overlap / denominator >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
        if len(kept) == k:
            break
    return kept


class BM25Search:
    """A complete lexical index for each recording."""

    def __init__(self, chunks: list[dict], config: dict):
        self.chunks = defaultdict(list)
        for chunk in chunks:
            self.chunks[chunk["meeting_id"]].append(chunk)
        self.indices = {key: BM25Okapi([lexical_tokens(c["text"]) for c in rows], **config["bm25"])
                        for key, rows in self.chunks.items()}
        self.config = config

    def search(self, query: str, meeting_id: str, k: int | None = None):
        if meeting_id not in self.indices:
            raise ValueError(f"Unknown meeting: {meeting_id}")
        tokens = lexical_tokens(query)
        if not tokens:
            raise ValueError("Query has no lexical tokens")
        scores = self.indices[meeting_id].get_scores(tokens)
        ranking = stable_ranking(self.chunks[meeting_id], scores)
        kept = deduplicate_ranking(ranking, k or self.config["ranking_depth"], self.config["dedup_overlap_threshold"])
        return ranking, kept
