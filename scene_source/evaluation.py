from __future__ import annotations

import json
import time
from dataclasses import replace

from .data import load_movies
from .embeddings import LocalEmbeddings
from .pipeline import build_graph
from .sources import DocumentCollector
from .store import VectorStore

CASES = [
    {"id": "Q1", "title": "인터스텔라", "question": "웜홀은 실제로 관측되었나요?", "relevant_title": "웜홀"},
    {"id": "Q2", "title": "인터스텔라", "question": "블랙홀의 사건의 지평선은 무엇인가요?", "relevant_title": "블랙홀"},
    {"id": "Q3", "title": "명량", "question": "이순신은 실존 인물인가요?", "relevant_title": "이순신"},
    {"id": "Q4", "title": "명량", "question": "명량 해전은 어떤 전투인가요?", "relevant_title": "명량 해전"},
    {"id": "Q5", "title": "오펜하이머", "question": "트리니티 실험과 맨해튼 계획은 어떤 관계인가요?", "relevant_title": "트리니티"},
    {"id": "Q6", "title": "인터스텔라", "question": "감독이 이 장면의 숨은 의도를 2026년 10월에 직접 설명했나요?", "relevant_title": None},
]


def run_local_evaluation(settings):
    from .discovery import discover
    from .llm import Filters
    from .discovery import matches
    folder = settings.root / "docs" / "verification"
    folder.mkdir(parents=True, exist_ok=True)
    movies = load_movies(settings.root)
    embedding = LocalEmbeddings(settings)
    store = VectorStore(settings, embedding)
    meta_before = store.meta.count()
    store.index_movies(movies)
    meta_after = store.meta.count()
    report = {"scope": "실제 로컬 다국어 모델·ChromaDB·LangGraph 실행. GMS 호출 및 사실 판정 품질은 미검증.",
              "metadata_idempotent": meta_before == meta_after == len(movies), "cases": []}
    for case in CASES:
        movie = next(m for m in movies if m["title"] == case["title"])
        collector = DocumentCollector(settings, store)
        first = collector.collect(movie, case["question"])
        count = store.docs.count()
        cached = collector.collect(movie, case["question"])
        start = time.perf_counter()
        result = build_graph(store).invoke({"movie": movie, "question": case["question"]})
        sim = store.retrieve(movie["movie_id"], case["question"], "similarity")
        mmr = store.retrieve(movie["movie_id"], case["question"], "mmr")
        report["cases"].append({**case, "cache_hit_on_repeat": cached["cache_hit"],
            "repeat_does_not_grow_index": count == store.docs.count(),
            "metadata_always_first": result["evidence"][0].source_type == "영화 메타데이터",
            "all_evidence_same_movie": all(e.movie_id == str(movie["movie_id"]) for e in result["evidence"]),
            "expected_title_retrieved": any(case["relevant_title"] in e.title for e in sim) if case["relevant_title"] else None,
            "similarity_sources": [e.title for e in sim], "mmr_sources": [e.title for e in mmr],
            "similarity_unique_pages": len({e.page_id for e in sim[1:]}),
            "mmr_unique_pages": len({e.page_id for e in mmr[1:]}),
            "similarity_chunk_ids": [e.id for e in sim], "mmr_chunk_ids": [e.id for e in mmr],
            "query_seconds": round(time.perf_counter() - start, 3),
            "answer_mode": result["generation_mode"]})
        print(case["id"], "cache", cached["cache_hit"], "sources", [e.title for e in sim], flush=True)
    query = "2010년 이후 한국 영화 중 130분 이하인 스릴러"
    discovery = discover(query, movies, store)
    report["discovery"] = {"query": query, "count": len(discovery["items"]),
        "all_hard_conditions_pass": all(matches(i["movie"], Filters(min_year=2010, max_runtime=130,
            genres=["스릴러"], language="ko")) for i in discovery["items"]),
        "titles": [i["movie"]["title"] for i in discovery["items"]],
        "semantic_llm_validation": "GMS 키 수령 후 검증"}
    changed = VectorStore(replace(settings, chunk_size=600, chunk_overlap=100), embedding)
    from .data import bundle_documents
    docs = bundle_documents(settings.root, "157336")
    primary = next(d for d in docs if d["title"] == "인터스텔라")
    base = VectorStore(replace(settings, chunk_size=1000, chunk_overlap=200), embedding)
    base.index_documents([primary])
    changed.index_documents([primary])
    page_filter = {"$and": [{"movie_id": "157336"}, {"page_id": str(primary["page_id"])}]}
    base_count = len(base.docs.get(where=page_filter, include=[])["ids"])
    changed_count = len(changed.docs.get(where=page_filter, include=[])["ids"])
    report["chunk_experiment"] = {"document": primary["title"], "1000_200": base_count,
                                    "600_100": changed_count, "counts_differ": base_count != changed_count}
    top2 = VectorStore(replace(settings, top_k=2), embedding).retrieve("157336", CASES[0]["question"])
    top5 = base.retrieve("157336", CASES[0]["question"])
    report["top_k_experiment"] = {"top_2_total": len(top2), "top_5_total": len(top5),
                                 "note": "항목 메타데이터 1개는 외부 Top-k와 별도로 항상 포함"}
    (folder / "local_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": report["metadata_idempotent"], "cases": len(report["cases"]),
                      "chunk_experiment": report["chunk_experiment"], "discovery": report["discovery"]}, ensure_ascii=False, indent=2))
