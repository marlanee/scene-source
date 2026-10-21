from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    fetch_k: int = 20
    mmr_lambda: float = 0.65
    method: str = "similarity"
    hybrid: bool = True
    public: bool = False
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 16
    threads: int = 4
    source_mode: str = "bundle"
    max_keywords: int = 4
    max_documents: int = 5
    request_interval: float = 0.6
    max_retries: int = 4

    def __post_init__(self):
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("청크 중복 길이는 0 이상, 청크 크기 미만이어야 합니다.")
        if not 1 <= self.top_k <= self.fetch_k <= 100:
            raise ValueError("검색 수는 1 <= Top-k <= 후보 수 <= 100이어야 합니다.")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("MMR lambda는 0~1 사이여야 합니다.")
        if self.method not in {"similarity", "mmr"} or self.source_mode not in {"bundle", "live"}:
            raise ValueError("검색 방식 또는 수집 방식이 올바르지 않습니다.")

    @property
    def data_dir(self):
        return self.root / "data"

    @property
    def index_fingerprint(self):
        value = [self.embedding_model, self.chunk_size, self.chunk_overlap, self.source_mode, "schema-v5-scopes-pool120"]
        if self.public:
            value.append("public-v1")
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()[:12]

    @property
    def meta_collection(self):
        model_id = hashlib.sha256((self.embedding_model + ":pool120-v1").encode()).hexdigest()[:10]
        return f"movie_meta_{model_id}"

    @property
    def docs_collection(self):
        return f"movie_evidence_{self.index_fingerprint}"


def load_settings(root: Path = ROOT) -> Settings:
    load_dotenv(root / ".env", override=False)
    with (root / "config.toml").open("rb") as f:
        raw = tomllib.load(f)
    args = {"root": root, **raw["retrieval"], **raw["collection"]}
    args.update({"embedding_model": raw["embedding"]["model"],
                 "batch_size": raw["embedding"]["batch_size"], "threads": raw["embedding"]["threads"]})
    if os.getenv("SCENE_PUBLIC") == "1":
        args["source_mode"] = "live"
        args["public"] = True
        args["embedding_model"] = os.getenv("PUBLIC_EMBEDDING_MODEL", "scene-source/multilingual-minilm-int8")
        args["batch_size"] = 2
        args["threads"] = 1
    if (root / "public_data/movies.json").exists() and not (root / "data/source_pack.zip").exists() and not (root.parent / "영화도메인_데이터_배포용.zip").exists():
        args["source_mode"] = "live"
    return Settings(**args)


def gms_available() -> bool:
    return bool(os.getenv("GMS_KEY", "").strip())
