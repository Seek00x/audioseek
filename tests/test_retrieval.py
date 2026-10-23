"""Chunk boundaries and duplicate filtering with a generated tokenizer."""
import json
import unittest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from audioseek.io import CONFIG_ROOT
from audioseek.retrieval import build_chunks, deduplicate_ranking, lexical_tokens


class ChunkTests(unittest.TestCase):
    def test_token_cap_and_word_coverage(self):
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
        settings = json.loads((CONFIG_ROOT / "retrieval.json").read_text("utf-8"))
        segments = [{"text": " ".join(f"word{i}" for i in range(300)), "speakerName": "Unknown"}]
        chunks, words, _ = build_chunks({"meeting_id": "recording", "corpus": "UPLOAD", "segments": segments}, tokenizer, settings)
        self.assertEqual(chunks[0]["word_start"], 0)
        self.assertEqual(chunks[-1]["word_end"], len(words))
        self.assertTrue(all(c["token_count"] <= 256 for c in chunks))
        for current, following in zip(chunks, chunks[1:]):
            self.assertLess(current["word_start"], following["word_start"])
            self.assertLess(following["word_start"], current["word_end"])

    def test_dedup_refills_after_high_overlap_candidates(self):
        rows = [{"chunk_id": str(i), "meeting_id": "R", "word_start": a, "word_end": b}
                for i, (a, b) in enumerate([(0, 100), (1, 101), (20, 120), (100, 200), (200, 300)])]
        selected = deduplicate_ranking(rows, 3, threshold=.8)
        self.assertEqual([r["chunk_id"] for r in selected], ["0", "3", "4"])

    def test_lexical_normalization_preserves_apostrophes(self):
        self.assertEqual(lexical_tokens("It’s a NEW screen!"), ["it's", "a", "new", "screen"])


if __name__ == "__main__":
    unittest.main()
