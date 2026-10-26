"""Shared embedding encoder.

Imported verbatim by the offline embedding job, the training scripts and the API. There
is deliberately only one place that turns text into a vector -- if training and serving
embed text differently, ranking breaks in ways that are very hard to see.

Model: BAAI/bge-small-en-v1.5 rather than all-MiniLM-L6-v2. Same 384 dimensions (so the
pgvector schema is unchanged) and the same CPU cost class, but substantially stronger on
MTEB retrieval (~51.7 vs ~42.0). MIT licensed.

The thing that actually matters here: max_seq_length is 256 tokens and the model
truncates past it *silently*. Job descriptions in our corpus average ~3,775 characters,
roughly 700-900 tokens, so naive encoding throws away 60-70% of every posting -- and the
discarded tail is usually the required-qualifications section, which is exactly the part
that determines fit. Jobs are therefore chunked into windows and mean-pooled. Resumes are
front-loaded enough that truncation is acceptable, which keeps per-request latency low.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

# bge is an asymmetric model: the query side gets a prefix, the passage side does not.
# A resume searching for jobs is the query.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384

# Token budget per chunk, leaving room for [CLS]/[SEP].
CHUNK_TOKENS = 200
# Cap on chunks per document. At 200 tokens each this covers ~5,000 tokens, far beyond
# any posting we store after the 4,000-character truncation in the loader.
MAX_CHUNKS = 8


class Encoder:
    """Wraps SentenceTransformer with our chunking and prefix conventions."""

    def __init__(self, model_name: str = DEFAULT_MODEL, use_onnx: bool = True):
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if use_onnx:
            # sentence-transformers >=3.2 exposes an ONNX backend that is a documented
            # 1.4-3x CPU speedup for a one-line change. Fall back silently if the
            # optional dependency isn't present.
            kwargs["backend"] = "onnx"
        try:
            self.model = SentenceTransformer(model_name, **kwargs)
        except Exception:
            self.model = SentenceTransformer(model_name)

        self.model.max_seq_length = 256
        self.tokenizer = self.model.tokenizer
        self._tune_threads()

    @staticmethod
    def _tune_threads() -> None:
        """Avoid thread oversubscription, which hurts more than it helps on CPU."""
        try:
            import torch

            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    # -- chunking ---------------------------------------------------------------

    def _chunk(self, text: str) -> list[str]:
        """Split text into <=CHUNK_TOKENS-token windows using the model's own tokenizer.

        Word-count heuristics drift from real token counts badly on job descriptions,
        which are full of punctuation and acronyms, so we tokenize properly.
        """
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= CHUNK_TOKENS:
            return [text]
        windows = [ids[i:i + CHUNK_TOKENS] for i in range(0, len(ids), CHUNK_TOKENS)]
        windows = windows[:MAX_CHUNKS]
        return [self.tokenizer.decode(w, skip_special_tokens=True) for w in windows]

    def _encode_plain(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    # -- public API -------------------------------------------------------------

    def encode_resumes(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Query side: prefix applied, plain truncation at 256 tokens."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return self._encode_plain([QUERY_PREFIX + (t or "") for t in texts], batch_size)

    def encode_jobs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Passage side: chunked into windows and mean-pooled, then re-normalised.

        Flattens every document's chunks into one batch so the GPU/CPU sees full-sized
        batches regardless of how the chunks are distributed across documents.
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        flat: list[str] = []
        spans: list[tuple[int, int]] = []
        for text in texts:
            chunks = self._chunk(text or "")
            spans.append((len(flat), len(flat) + len(chunks)))
            flat.extend(chunks)

        vectors = self._encode_plain(flat, batch_size)

        pooled = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
        for i, (start, end) in enumerate(spans):
            if end > start:
                pooled[i] = vectors[start:end].mean(axis=0)

        # Mean-pooling unit vectors does not yield a unit vector; renormalise so cosine
        # similarity is a plain dot product everywhere downstream.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        np.divide(pooled, norms, out=pooled, where=norms > 0)
        return pooled

    def encode_titles(self, titles: list[str], batch_size: int = 64) -> np.ndarray:
        """Titles are short; no chunking needed, no prefix (passage side)."""
        if not titles:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return self._encode_plain([t or "" for t in titles], batch_size)


@lru_cache(maxsize=1)
def get_encoder() -> Encoder:
    """Process-wide singleton. The API loads this once at startup via lifespan."""
    return Encoder()


def job_text(title: str | None, company: str | None, description: str | None) -> str:
    """Canonical job text. Used by the loader, trainer and API alike."""
    return " ".join(p for p in (title, company, description) if p).strip()
