"""Pinned SBERT inference, validated embedding caches and full-ranking RRF."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np

from .retrieval import deduplicate_ranking, stable_ranking


def rrf_rankings(bm25, semantic, rrf_K=60, weights=(1.0, 1.0)):
    if rrf_K < 0 or len(weights) != 2 or any(w <= 0 for w in weights):
        raise ValueError("Invalid RRF parameters")
    left = {r["chunk_id"]: i for i, r in enumerate(bm25, 1)}
    right = {r["chunk_id"]: i for i, r in enumerate(semantic, 1)}
    if len(left) != len(bm25) or len(right) != len(semantic) or set(left) != set(right):
        raise ValueError("RRF needs two complete unique rankings of the same chunks")
    if len({r["meeting_id"] for r in bm25 + semantic}) != 1:
        raise ValueError("RRF cannot mix meetings")
    scores = [weights[0] / (rrf_K + left[c["chunk_id"]]) + weights[1] / (rrf_K + right[c["chunk_id"]]) for c in bm25]
    return stable_ranking(bm25, scores)


def validate_vectors(array, rows, dimension=384):
    if array.shape != (rows, dimension) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Embedding shape, dtype or finite-value check failed")
    if not np.allclose(np.linalg.norm(array, axis=1), 1, rtol=2e-4, atol=1e-4):
        raise ValueError("Expected L2-normalized nonzero embeddings")


def cache_identity(records, role, model_identity):
    payload = {"format": "audioseek-embedding-cache-v1", "role": role, "model": model_identity,
               "inputs": [{"id": r["id"], "text": r["text"]} for r in records]}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), payload


def cached_encode(model, records, role, identity, cache_dir):
    if role not in {"query", "document"} or not records:
        raise ValueError("A nonempty query/document input is required")
    cache_dir = Path(cache_dir)
    key, payload = cache_identity(records, role, identity)
    array_path, metadata_path = cache_dir / f"{role}-{key}.npy", cache_dir / f"{role}-{key}.json"
    started = time.perf_counter()
    reused = array_path.exists() and metadata_path.exists()
    if reused:
        metadata = json.loads(metadata_path.read_text("utf-8"))
        if metadata["cache_key"] != key or metadata["identity"] != payload or hashlib.sha256(array_path.read_bytes()).hexdigest() != metadata["array_sha256"]:
            raise ValueError("Embedding cache checksum/identity mismatch")
        array = np.load(array_path, allow_pickle=False)
        validate_vectors(array, len(records), identity["dimension"])
    else:
        import torch
        method = model.encode_document if role == "document" else model.encode_query
        with torch.inference_mode():
            array = method([r["text"] for r in records], batch_size=identity["batch_size"],
                           convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        array = np.asarray(array, dtype=np.float32)
        validate_vectors(array, len(records), identity["dimension"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_suffix(".npy.part")
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(array_path)
        metadata = {"cache_key": key, "identity": payload, "array_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                    "shape": list(array.shape), "dtype": str(array.dtype)}
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return array, {"role": role, "reused": reused, "seconds": time.perf_counter() - started,
                   "cache_key": key, "array_path": str(array_path), "metadata_path": str(metadata_path),
                   "rows": len(records), "dimension": identity["dimension"]}


def load_model(root, config):
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    import torch
    from sentence_transformers import SentenceTransformer

    manifest = json.loads((root / "data/semantic_model_manifest.json").read_text("utf-8"))
    if manifest["revision"] != config["revision"] or manifest["checkpoint"] != config["checkpoint"]:
        raise ValueError("Loaded model manifest does not identify the configured checkpoint")
    for item in manifest["files"]:
        if hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Model file changed: {item['path']}")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(config["torch_threads"])
    started = time.perf_counter()
    model = SentenceTransformer(str(root / config["model_directory"]), device=config["device"],
                                local_files_only=True, trust_remote_code=False,
                                model_kwargs={"attn_implementation": config["attention_implementation"], "use_safetensors": True})
    model.float().eval()
    model.max_seq_length = config["max_seq_length"]
    if model.get_sentence_embedding_dimension() != config["embedding_dimension"]:
        raise ValueError("Unexpected embedding dimension")
    # ST 5.1.2 inserts empty query/document defaults; only nonempty prefixes change inputs.
    if any(value != "" for value in model.prompts.values()) or model.default_prompt_name is not None:
        raise ValueError("Unexpected prompts; inputs must use the shared token limit")
    if not callable(getattr(model, "encode_query", None)) or not callable(getattr(model, "encode_document", None)):
        raise ValueError("Required query/document encoding APIs are unavailable")
    transformer = model[0].auto_model.config
    pooling = model[1]
    if not pooling.pooling_mode_mean_tokens or transformer.hidden_size != 384:
        raise ValueError("Unexpected encoder/pooling")
    probe_text = ["Find the discussion of the project timeline."]
    with torch.inference_mode():
        query_probe = model.encode_query(probe_text, normalize_embeddings=True, convert_to_numpy=True)
        document_probe = model.encode_document(probe_text, normalize_embeddings=True, convert_to_numpy=True)
    if not np.allclose(query_probe, document_probe, atol=1e-6):
        raise ValueError("Unexpected query/document difference in this prompt-free checkpoint")
    report = {"checkpoint": config["checkpoint"], "revision": config["revision"], "device": str(model.device),
              "dtype": str(next(model.parameters()).dtype), "model_max_seq_length": model.max_seq_length,
              "dimension": model.get_sentence_embedding_dimension(), "transformer_model_type": transformer.model_type,
              "transformer_layers": transformer.num_hidden_layers, "attention_heads": transformer.num_attention_heads,
              "pooling": "attention_mask_mean", "normalization": "checkpoint_Normalize_and_normalize_embeddings",
              "query_encoding_api": "encode_query", "document_encoding_api": "encode_document", "prompts": model.prompts,
              "same_text_query_document_max_abs_difference": float(np.max(np.abs(query_probe - document_probe))),
              "model_load_and_probe_seconds": time.perf_counter() - started, "trust_remote_code": False,
              "weights_sha256": next(r["sha256"] for r in manifest["files"] if r["path"].endswith("model.safetensors"))}
    return model, report


def validate_model_tokens(model, tokenizer, records, max_tokens):
    lengths = []
    for record in records:
        text = record["text"]
        ids = model.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        previous_ids = tokenizer.encode(text, add_special_tokens=True).ids
        if ids != previous_ids:
            raise ValueError("Model tokenizer IDs differ from the shared chunk tokenizer")
        if len(ids) > max_tokens:
            raise ValueError(f"Input too long; no silent truncation: {record['id']}")
        lengths.append(len(ids))
    return {"inputs_checked": len(records), "max_tokens_including_special": max(lengths),
            "all_token_ids_match_shared_tokenizer": True, "silent_truncation_used": False}


def model_cache_identity(config, model_report):
    return {"checkpoint": config["checkpoint"], "revision": config["revision"], "weights_sha256": model_report["weights_sha256"],
            "max_seq_length": config["max_seq_length"], "batch_size": config["batch_size"], "dimension": config["embedding_dimension"],
            "normalize_embeddings": True, "dtype": "float32", "device": config["device"], "prompts": model_report["prompts"],
            "attention_implementation": config["attention_implementation"], "torch_threads": config["torch_threads"],
            "packages": {name: importlib.metadata.version(name) for name in ["torch", "sentence-transformers", "transformers", "tokenizers", "numpy"]}}


def semantic_ranking(chunks, document_vectors, query_vector):
    validate_vectors(document_vectors, len(chunks), len(query_vector))
    validate_vectors(query_vector[None, :], 1, document_vectors.shape[1])
    return stable_ranking(chunks, document_vectors @ query_vector)
