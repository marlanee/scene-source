from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from scene_source.cache import CollectionCache
from scene_source.config import Settings
from scene_source.data import load_movies, movie_text
from scene_source.discovery import local_filters, matches
from scene_source.embeddings import mmr_indices
from scene_source.llm import Keywords, SelectedDocuments, parse_json
from scene_source.models import Evidence, UserFacingError
from scene_source.pipeline import build_graph, cited_numbers, resolve_question, used_sources, validate_citations
from scene_source.sources import DocumentCollector
from scene_source.store import VectorStore, split_document


class FakeEmbedding:
    """Only for isolated persistence tests; production always uses a real semantic model."""
    def documents(self, texts):
        return [self.query(x) for x in texts]

    def query(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [(x - 128) / 128 for x in digest[:16]]


@pytest.fixture
def movie():
    return {"movie_id": "1", "title": "테스트영화", "original_title": "Test Movie", "year": 2014,
            "release_date": "2014-01-01", "genres": ["SF", "드라마"], "director": [],
            "cast": ["배우"], "overview": "우주를 탐사하는 가족 이야기.", "runtime": 120,
            "original_language": "ko", "production_countries": ["South Korea"],
            "vote_average": 8.0, "tmdb_url": "https://www.themoviedb.org/movie/1"}


@pytest.fixture
def store(tmp_path):
    return VectorStore(Settings(root=tmp_path), FakeEmbedding())


@pytest.fixture
def docs():
    return [{"movie_id": "1", "page_id": "10", "title": "테스트영화 (영화)", "url": "https://ko.wikipedia.org/wiki/Test",
             "text": "영화는 가상의 우주를 묘사한다. " * 100, "source": "한국어 위키백과"},
            {"movie_id": "2", "page_id": "10", "title": "같은 위키 문서", "url": "https://ko.wikipedia.org/wiki/Test",
             "text": "다른 영화에 연결된 문서이다. " * 100, "source": "한국어 위키백과"}]


def test_movie_index_idempotent_and_missing_director(store, movie):
    assert "감독: 정보 없음" in movie_text(movie)
    assert store.index_movies([movie]) == 1
    assert store.index_movies([movie]) == 1
    movie["overview"] = "줄거리 변경"
    store.index_movies([movie])
    assert "줄거리 변경" in store.meta.get(ids=["1"])["documents"][0]


def test_shared_page_isolated_between_movies(store, movie, docs):
    store.index_movies([movie])
    store.index_documents(docs)
    first = store.document_count("1")
    second = store.document_count("2")
    store.index_documents([docs[0]])
    assert store.document_count("1") == first
    assert store.document_count("2") == second
    assert store.docs.count() == first + second
    result = store.retrieve("1", "우주")
    assert result[0].source_type == "영화 메타데이터"
    assert all(e.movie_id == "1" for e in result)


def test_document_update_removes_stale_tail(store, docs):
    store.index_documents([docs[0]])
    assert store.document_count("1") > 1
    store.index_documents([{**docs[0], "text": "짧은 새 원문"}])
    assert store.document_count("1") == 1


def test_mmr_returns_distinct_vectors():
    assert mmr_indices([1, 0], [[1, 0], [0.99, 0.01], [0, 1]], 2, 0.3) == [0, 2]
    assert mmr_indices([1, 0], [], 2, 0.3) == []


def test_empty_wiki_section_is_not_retrieved_as_evidence(store, movie):
    store.index_movies([movie])
    texts = ["=== 특이점 ===", "특이점", "사건의 지평선은 블랙홀의 경계이다."]
    store.docs.upsert(ids=[f"1:10:{i}" for i in range(len(texts))], documents=texts,
        embeddings=store.embedding.documents(texts), metadatas=[{
            "movie_id": "1", "page_id": "10", "title": "블랙홀", "url": "https://ko.wikipedia.org/wiki/Test",
            "source_type": "한국어 위키백과"} for _ in texts])
    for method in ("similarity", "mmr"):
        result = store.retrieve("1", "특이점", method)
        assert all(e.text not in {"=== 특이점 ===", "특이점"} for e in result)
        assert len(result) == 2


def test_chunk_config_changes_counts():
    text = "서로 연결된 문장입니다. " * 100
    assert len(split_document(text, 300, 50)) > len(split_document(text, 1000, 200))
    with pytest.raises(ValueError):
        split_document(text, 100, 100)


def test_collection_fingerprints_prevent_mixed_chunk_configs():
    a, b = Settings(), Settings(chunk_size=600, chunk_overlap=100)
    assert a.meta_collection == b.meta_collection
    assert a.docs_collection != b.docs_collection
    live = Settings(source_mode="live")
    assert a.meta_collection == live.meta_collection
    assert a.docs_collection != live.docs_collection


@pytest.mark.parametrize("values", [{"top_k": 0}, {"top_k": 30}, {"chunk_overlap": 1000}, {"method": "unknown"}, {"mmr_lambda": 1.1}])
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        Settings(**values)


def test_citations_only_actually_used():
    evidence = [Evidence(str(i), f"문서{i}", "https://example.org", "본문", "외부", "1") for i in range(4)]
    assert cited_numbers("주장 [근거 1, 3] 다른 주장 [근거 3]") == {1, 3}
    assert [s["number"] for s in used_sources("주장 [근거 1, 3]", evidence)] == [1, 3]
    assert used_sources("확인 불가", evidence) == []
    with pytest.raises(UserFacingError):
        validate_citations("잘못된 인용 [근거 7]", evidence)


def test_structured_output_validation():
    assert parse_json('```json\n{"keywords":["웜홀"]}\n```', Keywords).keywords == ["웜홀"]
    with pytest.raises(ValueError):
        parse_json('{"keywords": "문자열"}', Keywords)


def test_local_condition_filter_boundaries(movie):
    f = local_filters("2010년 이후 한국 영화 중 120분 이하인 SF")
    assert matches(movie, f)
    assert not matches({**movie, "runtime": 121}, f)
    assert not matches({**movie, "year": 2009}, f)
    assert not matches({**movie, "original_language": "en"}, f)
    assert not matches(movie, local_filters("120분 미만"))
    assert matches(movie, local_filters("2시간 이하"))
    assert not matches({**movie, "runtime": 0}, local_filters("2시간 이하"))


def test_cache_survives_new_instances(tmp_path):
    path = tmp_path / "state.sqlite3"
    CollectionCache(path).put("collection1", {"page_ids": ["10"]})
    assert CollectionCache(path).get("collection1")["page_ids"] == ["10"]


def test_supplemental_documents_preserve_source_and_override_same_page(tmp_path):
    from scene_source.data import bundle_documents
    raw = tmp_path / "data/raw"
    original = raw / "movielab_wiki/wikipedia"
    extra = tmp_path / "data/supplemental_wikipedia"
    original.mkdir(parents=True)
    extra.mkdir(parents=True)
    (raw / "movies.json").write_text("[]", encoding="utf-8")
    (original / "1_old.json").write_text(json.dumps({"page_id": 10, "text": "old"}), encoding="utf-8")
    updated = {"page_id": 10, "text": "new", "url": "https://ko.wikipedia.org/wiki/Test", "collected_at": "2026-10-21"}
    (extra / "1_10.json").write_text(json.dumps(updated), encoding="utf-8")
    (extra / "1_20.json").write_text(json.dumps({"page_id": 20, "text": "extra"}), encoding="utf-8")
    result = bundle_documents(tmp_path, "1")
    assert len(result) == 2
    assert result[0] == updated


def test_primary_document_overrides_llm_exclusion(store, movie, docs, monkeypatch):
    import scene_source.sources as source_module
    monkeypatch.setattr(source_module, "bundle_documents", lambda *_: [docs[0]])
    store.index_movies([movie])

    class RejectAllLLM:
        calls = []
        def structured(self, system, payload, schema):
            self.calls.append(schema)
            return Keywords(keywords=["우주"]) if schema is Keywords else SelectedDocuments(selected_ids=[])

    llm = RejectAllLLM()
    collector = DocumentCollector(store.settings, store, llm)
    first = collector.collect(movie, "우주 질문")
    count = store.document_count("1")
    second = collector.collect(movie, "우주 질문")
    assert first["page_ids"] == ["10"]
    assert second["cache_hit"]
    assert store.document_count("1") == count
    assert llm.calls.count(SelectedDocuments) == 1


def test_no_cache_commit_on_failed_index(store, movie, docs, monkeypatch):
    import scene_source.sources as source_module
    monkeypatch.setattr(source_module, "bundle_documents", lambda *_: [docs[0]])
    original = store.index_documents
    monkeypatch.setattr(store, "index_documents", lambda *_: (_ for _ in ()).throw(RuntimeError("disk failure")))
    collector = DocumentCollector(store.settings, store)
    with pytest.raises(RuntimeError):
        collector.collect(movie, "질문")
    monkeypatch.setattr(store, "index_documents", original)
    assert not collector.collect(movie, "질문")["cache_hit"]


def test_live_keyword_and_page_cache(store):
    class FakeWiki:
        searches = 0
        fetches = 0
        def search(self, keyword):
            self.searches += 1
            return [{"page_id": "10", "title": "테스트"}]
        def fetch(self, title):
            self.fetches += 1
            return {"page_id": "10", "title": title, "text": "본문"}
    wiki = FakeWiki()
    c = DocumentCollector(store.settings, store)
    c.cached_search(wiki, "웜홀")
    c.cached_search(wiki, "블랙홀")
    c.cached_search(wiki, "웜홀")
    c.cached_fetch(wiki, "웜홀")
    c.cached_fetch(wiki, "웜홀")
    assert wiki.searches == 2 and wiki.fetches == 1


def test_reject_hallucinated_document_id(store, movie, docs, monkeypatch):
    import scene_source.sources as source_module
    monkeypatch.setattr(source_module, "bundle_documents", lambda *_: [docs[0]])
    class InvalidLLM:
        def structured(self, system, payload, schema):
            return Keywords(keywords=[]) if schema is Keywords else SelectedDocuments(selected_ids=["fake"])
    with pytest.raises(UserFacingError):
        DocumentCollector(store.settings, store, InvalidLLM()).collect(movie, "질문")
    assert store.document_count("1") == 0


def test_graph_has_two_nodes_and_local_generation_is_explicit(store, movie, docs):
    store.index_movies([movie])
    store.index_documents([docs[0]])
    graph = build_graph(store)
    assert set(graph.get_graph().nodes) == {"__start__", "retrieve", "generate", "__end__"}
    result = graph.invoke({"movie": movie, "question": "우주", "task": "factcheck"})
    assert result["generation_mode"] == "local_extract"
    assert "원문 발췌" in result["answer"]
    assert "판정:" not in result["answer"]
    assert result["used_sources"]


def test_followup_question_uses_recent_user_context():
    assert "웜홀" in resolve_question("그것은 관측됐나요?", [{"role": "user", "content": "웜홀이 뭐야?"}])


def test_gms_missing_key_and_exception_redaction(monkeypatch):
    from scene_source.llm import GMSClient
    monkeypatch.delenv("GMS_KEY", raising=False)
    with pytest.raises(UserFacingError, match="키가 없습니다"):
        GMSClient()
    class Broken:
        def invoke(self, *_):
            raise RuntimeError("SECRET + full prompt + traceback")
    client = object.__new__(GMSClient)
    client.client = Broken()
    with pytest.raises(UserFacingError) as exc:
        client.text("prompt", "payload")
    assert "SECRET" not in str(exc.value) and "prompt" not in str(exc.value)
