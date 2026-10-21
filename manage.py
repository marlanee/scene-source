"""Key-free project commands. Run with the project's virtual environment."""
from __future__ import annotations

import argparse
import json
import sys

from scene_source.config import load_settings
from scene_source.data import audit_data, load_movies, prepare_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "audit", "model", "index", "demo", "graph", "evaluate"])
    args = parser.parse_args()
    settings = load_settings()
    if args.command == "prepare":
        print(prepare_data(settings.root))
        return
    if args.command == "audit":
        report = audit_data(settings.root)
        folder = settings.root / "docs" / "verification"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "graph":
        from scene_source.pipeline import build_graph
        print(build_graph(None, None).get_graph().draw_mermaid())
        return
    if args.command == "evaluate":
        from scene_source.evaluation import run_local_evaluation
        run_local_evaluation(settings)
        return
    from scene_source.embeddings import LocalEmbeddings
    embedding = LocalEmbeddings(settings)
    if args.command == "model":
        print("Local semantic model ready; dimensions:", len(embedding.query("영화 속 역사와 과학")))
        return
    from scene_source.store import VectorStore
    movies = load_movies(settings.root)
    store = VectorStore(settings, embedding)
    count = store.index_movies(movies, progress=lambda n, total: print(f"Metadata {n}/{total}", flush=True))
    print("Indexed movies:", count, flush=True)
    if args.command == "demo":
        from scene_source.sources import DocumentCollector
        for title, question in [("인터스텔라", "웜홀은 실제로 관측되었나요?"),
                                ("명량", "이순신은 실존 인물인가요?"),
                                ("오펜하이머", "트리니티 실험과 맨해튼 계획은 어떤 관계인가요?")]:
            movie = next(m for m in movies if m["title"] == title)
            collector = DocumentCollector(settings, store)
            result = collector.collect(movie, question, progress=lambda msg: print(msg, flush=True))
            print(title, "documents:", len(result["selected"]), "chunks:", store.document_count(movie["movie_id"]), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
