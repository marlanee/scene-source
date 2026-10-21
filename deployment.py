"""Prepare a public demo image and start one API process without bundled secrets."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def prepare():
    from scene_source.config import load_settings
    from scene_source.data import load_movies
    from scene_source.remote_embeddings import create_embeddings
    from scene_source.store import VectorStore
    from scene_source.pdf_sources import ensure_catalog, index_pdfs
    from scene_source.experiments import ExperimentStore
    settings = load_settings()
    movies = load_movies(ROOT)
    store = VectorStore(settings, create_embeddings(settings))
    store.index_movies(movies)
    movie = next(m for m in movies if str(m["movie_id"]) == "157336")
    ensure_catalog(settings, movie)
    index_pdfs(settings, store, "157336")
    seed = ROOT / "public_data" / "evaluation.json"
    if seed.exists():
        import json
        report = json.loads(seed.read_text(encoding="utf-8"))
        db = ExperimentStore(ROOT)
        if not db.list():
            db.save(report)
    print("Public demo prepared:", len(movies), "movies; official PDF; no model API calls", flush=True)


def hydrate():
    target = ROOT / "data"
    target.mkdir(parents=True, exist_ok=True)
    seed = Path("/opt/seed-data")
    if seed.exists() and not any(target.iterdir()):
        shutil.copytree(seed, target, dirs_exist_ok=True)
    elif not (target / "vector_store" / "chroma.sqlite3").exists():
        prepare()


def check():
    hydrate()
    from scene_source.config import load_settings
    from scene_source.remote_embeddings import create_embeddings,SMOKE_QUERIES
    from scene_source.store import VectorStore
    settings=load_settings()
    store=VectorStore(settings,create_embeddings(settings))
    assert store.meta.count()==964
    evidence=store.retrieve("157336",SMOKE_QUERIES[0],"mmr")
    assert any(e.source_type=="공식 PDF" for e in evidence)
    print("Prepared NASA + metadata semantic retrieval verified without model API calls",flush=True)


def serve():
    hydrate()
    command = [sys.executable, "-m", "uvicorn", "scene_source.api:app", "--host", "0.0.0.0",
               "--port", os.getenv("PORT", "10000"), "--workers", "1", "--no-access-log"]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    os.environ.setdefault("SCENE_PUBLIC", "1")
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "serve", "check"}:
        raise SystemExit("Usage: python deployment.py prepare|serve|check")
    {"prepare":prepare,"serve":serve,"check":check}[sys.argv[1]]()
