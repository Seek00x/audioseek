"""Independent RRF examples, embedding invariants and cache failure cases."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from audioseek.semantic import cached_encode, cache_identity, rrf_rankings, semantic_ranking, validate_vectors
from audioseek.retrieval import deduplicate_ranking


def chunk(name, start):
    return {"chunk_id": name, "meeting_id": "M", "word_start": start, "word_end": start+10, "score": 0.0}


class SemanticTests(unittest.TestCase):
    def test_rrf_one_based_ranks_and_balanced_evidence(self):
        a, b, c, d, e = [chunk(n, i*20) for i, n in enumerate("abcde")]
        fused = rrf_rankings([a, b, c, d, e], [e, b, c, d, a], 60)
        self.assertEqual(fused[0]["chunk_id"], "b")
        self.assertAlmostEqual(fused[0]["score"], 2/62)
        values = {r["chunk_id"]: r["score"] for r in fused}
        self.assertAlmostEqual(values["a"], 1/61 + 1/65)
        self.assertEqual(rrf_rankings([b, a], [a, b])[0]["chunk_id"], "a")

    def test_rrf_rejects_missing_duplicates_or_mixed_meetings(self):
        a, b = chunk("a", 0), chunk("b", 20)
        for left, right in [([a, b], [a]), ([a, a], [a, b]), ([a, b], [a, {**b, "meeting_id": "N"}])]:
            with self.assertRaises(ValueError):
                rrf_rankings(left, right)
        with self.assertRaises(ValueError):
            rrf_rankings([a], [a], weights=(0, 1))

    def test_fusion_precedes_deduplication_and_refills(self):
        a, b, c = chunk("a", 0), chunk("b", 1), chunk("c", 20)
        fused = rrf_rankings([a, b, c], [b, a, c])
        kept = deduplicate_ranking(fused, 2, .8)
        self.assertEqual([r["chunk_id"] for r in kept], ["a", "c"])
        # Truncated, independently deduplicated inputs cannot replace full rankings.
        with self.assertRaises(ValueError):
            rrf_rankings([a, c], [b, c])

    def test_normalized_dot_product_and_invalid_vectors(self):
        a, b = chunk("a", 0), chunk("b", 20)
        vectors = np.array([[1, 0], [.6, .8]], dtype=np.float32)
        ranked = semantic_ranking([a, b], vectors, np.array([0, 1], dtype=np.float32))
        self.assertEqual(ranked[0]["chunk_id"], "b")
        self.assertAlmostEqual(ranked[0]["score"], .8)
        for array in [vectors.astype(np.float64), vectors*2, np.zeros((2, 2), np.float32),
                      np.array([[np.nan, 0], [1, 0]], dtype=np.float32)]:
            with self.assertRaises(ValueError):
                validate_vectors(array, 2, 2)

    def test_cache_key_changes_with_text_order_role_and_model(self):
        rows = [{"id": "a", "text": "battery"}, {"id": "b", "text": "power"}]
        identity = {"revision": "v1", "dimension": 2, "batch_size": 2}
        original = cache_identity(rows, "document", identity)[0]
        variants = [cache_identity(rows[::-1], "document", identity)[0],
                    cache_identity([{**rows[0], "text": "changed"}, rows[1]], "document", identity)[0],
                    cache_identity(rows, "query", identity)[0],
                    cache_identity(rows, "document", {**identity, "revision": "v2"})[0]]
        self.assertEqual(len({original, *variants}), 5)

    def test_cache_reuse_and_corruption_detection(self):
        class EncoderFixture:
            calls = 0
            def encode_document(self, texts, **kwargs):
                self.calls += 1
                return np.array([[1, 0], [0, 1]], dtype=np.float32)
        encoder = EncoderFixture()
        rows = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]
        identity = {"dimension": 2, "batch_size": 2}
        with tempfile.TemporaryDirectory() as directory:
            first, info = cached_encode(encoder, rows, "document", identity, directory)
            second, reused = cached_encode(encoder, rows, "document", identity, directory)
            np.testing.assert_array_equal(first, second)
            self.assertFalse(info["reused"])
            self.assertTrue(reused["reused"])
            self.assertEqual(encoder.calls, 1)
            path = Path(info["array_path"])
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaises(ValueError):
                cached_encode(encoder, rows, "document", identity, directory)

    def test_query_cache_uses_query_api(self):
        class QueryFixture:
            def encode_query(self, texts, **kwargs):
                return np.array([[.6, .8]], dtype=np.float32)
            def encode_document(self, *args, **kwargs):
                raise AssertionError("Wrong encoding API")
        with tempfile.TemporaryDirectory() as directory:
            result, _ = cached_encode(QueryFixture(), [{"id": "q", "text": "query"}], "query",
                                      {"dimension": 2, "batch_size": 1}, directory)
            np.testing.assert_allclose(result, [[.6, .8]], atol=1e-7)
            with self.assertRaises(ValueError):
                cached_encode(QueryFixture(), [{"id": "q", "text": "query"}], "wrong",
                              {"dimension": 2, "batch_size": 1}, directory)


if __name__ == "__main__":
    unittest.main()
