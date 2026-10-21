from __future__ import annotations

from functools import lru_cache
import os
import sys
from pathlib import Path

import numpy as np

from .config import Settings


def release_unused_heap():
    """Return unused glibc arenas after large model/index allocations."""
    if sys.platform.startswith("linux"):
        import ctypes
        import gc
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


class LocalEmbeddings:
    """Multilingual semantic embeddings. No API key and no remote inference."""

    def __init__(self, settings: Settings):
        from fastembed import TextEmbedding
        self.model_name = settings.embedding_model
        if self.model_name == "scene-source/multilingual-minilm-int8" and not any(m["model"] == self.model_name for m in TextEmbedding.list_supported_models()):
            from fastembed.common.model_description import ModelSource, PoolingType
            TextEmbedding.add_custom_model(model=self.model_name, pooling=PoolingType.MEAN, normalization=True,
                sources=ModelSource(hf="Xenova/paraphrase-multilingual-MiniLM-L12-v2"), dim=384,
                model_file="onnx/model_quantized.onnx", license="apache-2.0", size_in_gb=.118)
        self.batch_size = settings.batch_size
        cache = Path(os.getenv("EMBEDDING_CACHE_DIR", str(settings.data_dir / "model_cache")))
        cache.mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(model_name=self.model_name, cache_dir=str(cache),
                                   threads=settings.threads, enable_cpu_mem_arena=not settings.public)
        if settings.public:
            release_unused_heap()
        # MiniLM's shipped tokenizer truncates at 128 tokens. Pool 120-token windows
        # so the end of a 1,000-character Korean chunk participates in retrieval.
        self.shared_tokenizer = settings.public
        if self.shared_tokenizer:
            # The multilingual vocabulary is large. Cloning it can exceed a
            # 512MB worker's memory budget. Public jobs run in one worker, so
            # temporarily change and restore the existing tokenizer instead.
            self.tokenizer = self.model.model.tokenizer
        else:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()

    def windows(self, text: str) -> list[str]:
        if self.shared_tokenizer:
            truncation, padding = self.tokenizer.truncation, self.tokenizer.padding
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            try:
                ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            finally:
                if truncation:
                    self.tokenizer.enable_truncation(**truncation)
                if padding:
                    self.tokenizer.enable_padding(**padding)
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if not ids:
            return [""]
        return [self.tokenizer.decode(ids[i:i + 120], skip_special_tokens=True)
                for i in range(0, len(ids), 120)]

    def _encode(self, texts: list[str], query=False) -> list[list[float]]:
        windows, lengths = [], []
        for text in texts:
            parts = self.windows(text)
            windows.extend(parts)
            lengths.append(len(parts))
        if not windows:
            return []
        stream = self.model.query_embed(windows, batch_size=self.batch_size) if query else self.model.passage_embed(windows, batch_size=self.batch_size)
        vectors = list(stream)
        result, start = [], 0
        for length in lengths:
            pooled = np.mean(vectors[start:start + length], axis=0)
            pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
            result.append(pooled.tolist())
            start += length
        return result

    def documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    @lru_cache(maxsize=128)
    def query(self, text: str) -> list[float]:
        return self._encode([text], query=True)[0]


def mmr_indices(query: list[float], vectors: list, k: int, weight: float, relevance_scores=None) -> list[int]:
    if not len(vectors):
        return []
    matrix = np.asarray(vectors, dtype=float)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    q = np.asarray(query, dtype=float)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    relevance = matrix @ q if relevance_scores is None else np.asarray(relevance_scores, dtype=float)
    selected = [int(np.argmax(relevance))]
    remaining = set(range(len(matrix))) - set(selected)
    while remaining and len(selected) < k:
        def score(i):
            redundancy = max(float(matrix[i] @ matrix[j]) for j in selected)
            return weight * float(relevance[i]) - (1 - weight) * redundancy
        chosen = max(sorted(remaining), key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected
