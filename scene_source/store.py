from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .data import movie_text
from .embeddings import mmr_indices
from .models import Evidence
from .retrieval import bm25_scores
from .storage_lock import claim_storage


def split_document(text: str, size: int, overlap: int) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("Invalid chunk settings")
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "。", " ", ""]).split_text(text)


def has_body(text: str) -> bool:
    """MediaWiki may return a section heading with no accompanying paragraph."""
    body = re.sub(r"(?m)^\s*=+[^\n]*?=+\s*$", "", text).strip()
    if re.match(r"^(?:각주|내용주|참조주|참고 자료|참고 문헌|외부 링크|References|External links)(?:\s|$)", body, re.I):
        return False
    # Splitters can strip the equals signs from a standalone heading as well.
    return len(body) >= 15


class VectorStore:
    def __init__(self, settings: Settings, embedding):
        self.settings = settings
        self.embedding = embedding
        claim_storage(settings.data_dir / "vector_store")
        self.client = chromadb.PersistentClient(path=str(settings.data_dir / "vector_store"),
            settings=ChromaSettings(anonymized_telemetry=False))
        self.meta = self.client.get_or_create_collection(settings.meta_collection,
            metadata={"hnsw:space": "cosine"}, embedding_function=None)
        self.docs = self.client.get_or_create_collection(settings.docs_collection,
            metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def index_movies(self, movies: list[dict], progress: Callable = lambda *_: None) -> int:
        # Fixed IDs make repeat indexing idempotent. Content hashes permit changed records.
        existing = self.meta.get(include=["metadatas"])
        known = {i: m.get("content_hash") for i, m in zip(existing["ids"], existing["metadatas"])}
        pending = []
        for m in movies:
            text = movie_text(m)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if known.get(str(m["movie_id"])) != digest:
                pending.append((m, text, digest))
        for start in range(0, len(pending), self.settings.batch_size):
            batch = pending[start:start + self.settings.batch_size]
            texts = [x[1] for x in batch]
            self.meta.upsert(ids=[str(x[0]["movie_id"]) for x in batch], documents=texts,
                embeddings=self.embedding.documents(texts), metadatas=[{
                    "movie_id": str(m["movie_id"]), "title": m["title"], "url": m["tmdb_url"],
                    "source_type": "영화 메타데이터", "content_hash": digest,
                    "year": int(m["year"]), "runtime": int(m.get("runtime") or 0),
                    "language": m.get("original_language", ""),
                } for m, _, digest in batch])
            progress(min(start + len(batch), len(pending)), len(pending))
        return self.meta.count()

    def index_documents(self, docs: list[dict], progress: Callable = lambda *_: None) -> int:
        for pos, doc in enumerate(docs, 1):
            movie_id, page_id = str(doc["movie_id"]), str(doc["page_id"])
            where = {"$and": [{"movie_id": movie_id}, {"page_id": page_id}]}
            content_hash = hashlib.sha256(json.dumps({k: doc.get(k, "") for k in
                ["text", "title", "url", "publisher", "filename", "source", "page_number", "evidence_scope"]},
                ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            existing = self.docs.get(where=where, include=["metadatas"])
            if existing["ids"] and all(m.get("content_hash") == content_hash for m in existing["metadatas"]):
                progress(pos, len(docs))
                continue
            chunks = split_document(doc["text"], self.settings.chunk_size, self.settings.chunk_overlap)
            texts = [f"문서: {doc['title']}\n{chunk}" for chunk in chunks]
            vectors = self.embedding.documents(texts)
            self.docs.delete(where=where)
            for start in range(0, len(chunks), 128):
                indices = list(range(start, min(start + 128, len(chunks))))
                self.docs.upsert(ids=[f"{movie_id}:{page_id}:{i}" for i in indices],
                    documents=[chunks[i] for i in indices], embeddings=[vectors[i] for i in indices],
                    metadatas=[{"movie_id": movie_id, "page_id": page_id, "title": doc["title"],
                        "url": doc["url"], "source_type": doc.get("source", "외부 문서"),
                        "related_movie": doc.get("related_movie", ""),
                        "search_query": doc.get("search_query", ""),
                        "collected_at": doc.get("collected_at", ""), "chunk_index": i,
                        "page_number": doc.get("page_number", 0),
                        "publisher": doc.get("publisher", ""), "filename": doc.get("filename", ""),
                        "evidence_scope": doc.get("evidence_scope", "background"),
                        "content_hash": content_hash}
                        for i in indices])
            progress(pos, len(docs))
        return self.docs.count()

    def document_count(self, movie_id: str) -> int:
        return len(self.docs.get(where={"movie_id": str(movie_id)}, include=[])["ids"])

    @staticmethod
    def _evidence(row_id, text, meta, score=0.0):
        return Evidence(id=row_id, text=text, score=score, title=meta["title"], url=meta["url"],
            source_type=meta["source_type"], movie_id=meta["movie_id"],
            page_id=meta.get("page_id", ""), collected_at=meta.get("collected_at", ""),
            page_number=meta.get("page_number", 0), publisher=meta.get("publisher", ""),
            filename=meta.get("filename", ""), evidence_scope=meta.get("evidence_scope", "film" if meta["source_type"] == "영화 메타데이터" else "background"))

    def retrieve(self, movie_id: str, question: str, method: str | None = None) -> list[Evidence]:
        evidence = []
        record = self.meta.get(ids=[str(movie_id)], include=["documents", "metadatas"])
        if record["ids"]:
            evidence.append(self._evidence(record["ids"][0], record["documents"][0], record["metadatas"][0]))
        count = self.document_count(movie_id)
        if not count:
            return evidence
        vector = self.embedding.query(question)
        result = self.docs.query(query_embeddings=[vector], where={"movie_id": str(movie_id)},
            n_results=min(self.settings.fetch_k, count),
            include=["documents", "metadatas", "distances", "embeddings"])
        rows = {k: result[k][0] for k in ["ids", "documents", "metadatas", "embeddings"]}
        semantic = [1 - d for d in result["distances"][0]]
        ranking = semantic[:]
        if self.settings.hybrid:
            semantic_ids = set(rows["ids"])
            rows = self.docs.get(where={"movie_id": str(movie_id)}, include=["documents", "metadatas", "embeddings"])
            matrix = np.asarray(rows["embeddings"], dtype=float)
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            q = np.asarray(vector, dtype=float)
            q /= max(float(np.linalg.norm(q)), 1e-12)
            semantic = (matrix @ q).tolist()
            lexical = bm25_scores(question, [m["title"] + "\n" + t for m,t in zip(rows["metadatas"], rows["documents"])])
            lexical_top = set(sorted(range(len(lexical)), key=lambda i: lexical[i], reverse=True)[:self.settings.fetch_k])
            pool = {i for i, row_id in enumerate(rows["ids"]) if row_id in semantic_ids} | lexical_top
            ranking = [.75 * semantic[i] + .25 * lexical[i] for i in range(len(semantic))]
        else:
            pool = set(range(len(rows["ids"])))
        eligible = [i for i in pool if has_body(rows["documents"][i])
                    and not (rows["metadatas"][i].get("page_number") and len(rows["documents"][i].strip()) < 100)]
        eligible = sorted(eligible, key=lambda i: (-ranking[i], rows["ids"][i]))[:self.settings.fetch_k]
        indices = eligible[:self.settings.top_k]
        if (method or self.settings.method) == "mmr":
            chosen = mmr_indices(vector, [rows["embeddings"][i] for i in eligible],
                                 self.settings.top_k, self.settings.mmr_lambda,
                                 relevance_scores=[ranking[i] for i in eligible])
            indices = [eligible[i] for i in chosen]
        seen = set()
        for i in indices:
            text = rows["documents"][i]
            digest = hashlib.sha256(text.strip().encode()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                source = self._evidence(rows["ids"][i], text, rows["metadatas"][i], semantic[i])
                source.retrieval_score = ranking[i]
                film_title = record["metadatas"][0]["title"] if record["ids"] else ""
                plain_title = re.sub(r"\s*\((?:\d{4}년\s*)?(?:영화|애니메이션|드라마)\)\s*", "", source.title).strip()
                if film_title and (plain_title == film_title or re.search(r"\((?:\d{4}년\s*)?(?:영화|애니메이션|드라마)\)", source.title)):
                    source.evidence_scope = "film"
                evidence.append(source)
        return evidence

    def search_movies(self, query: str, limit=24, where=None) -> list[tuple[str, float]]:
        if not self.meta.count():
            return []
        kwargs = {"where": where} if where else {}
        result = self.meta.query(query_embeddings=[self.embedding.query(query)],
            n_results=min(limit, self.meta.count()), include=["distances"], **kwargs)
        return list(zip(result["ids"][0], [1 - x for x in result["distances"][0]]))
