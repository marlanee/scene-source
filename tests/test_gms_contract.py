from __future__ import annotations

import json

import pytest

from scene_source.discovery import discover
from scene_source.llm import Filters, MoviePicks
from scene_source.models import Evidence, UserFacingError
from scene_source.pipeline import build_graph
from scene_source.sources import Wikipedia, is_primary


def test_gms_graph_passes_sources_and_task_and_only_used_citations():
    class Store:
        def retrieve(self, *_):
            return [Evidence("1", "영화", "https://example.org/1", "작품 설정", "메타데이터", "1"),
                    Evidence("2", "이론", "https://example.org/2", "관측 여부 정보 없음", "위키백과", "1")]
    class LLM:
        payload = None
        def text(self, system, payload):
            self.payload = json.loads(payload)
            return "판정: 확인 불가. 관측 여부는 제공 문서에서 확인되지 않습니다. [근거 2]"
    llm = LLM()
    result = build_graph(Store(), llm).invoke({"movie": {"movie_id": "1", "title": "작품"},
                                            "question": "이 이론이 관측됐나요?", "task": "factcheck"})
    assert llm.payload["task"] == "factcheck"
    assert len(llm.payload["evidence"]) == 2
    assert [s["number"] for s in result["used_sources"]] == [2]
    assert result["generation_mode"] == "gms"


def test_discovery_filters_before_search_and_verifies_candidate_ids():
    movies = [{"movie_id": str(i), "title": f"영화{i}", "year": 2015, "runtime": 110,
               "genres": ["스릴러"], "original_language": "ko"} for i in range(3)]
    class Store:
        where = None
        def search_movies(self, query, limit, where):
            self.where = where
            return [("2", 0.9), ("1", 0.8)]
    class LLM:
        invalid = False
        def structured(self, system, payload, schema):
            if schema is Filters:
                return Filters(min_year=2010, max_runtime=120, genres=["스릴러"], language="ko")
            return MoviePicks(items=[{"movie_id": "99" if self.invalid else "2", "reason": "조건 확인"}])
    store, llm = Store(), LLM()
    result = discover("조건", movies, store, llm)
    assert store.where["movie_id"]["$in"] == ["0", "1", "2"]
    assert result["items"][0]["movie"]["movie_id"] == "2"
    llm.invalid = True
    with pytest.raises(UserFacingError):
        discover("조건", movies, store, llm)


def test_same_name_old_film_is_not_primary():
    assert not is_primary({"movie_id": "460465", "title": "모탈 컴뱃", "year": 2021},
                          {"title": "모탈 컴뱃 (1995년 영화)", "text": "1995년 영화이다."})
    assert is_primary({"movie_id": "105", "title": "빽 투 더 퓨쳐", "original_title": "Back to the Future", "year": 1985},
                      {"title": "백 투 더 퓨처", "text": "《백 투 더 퓨처》(영어: Back to the Future)는 1985년 영화다."})
    assert not is_primary({"movie_id": "581387", "title": "백두산", "year": 2019},
                          {"title": "백두산", "text": "백두산은 북한과 중국 국경의 화산이다."})
    assert not is_primary({"movie_id": "1194963", "title": "노이즈", "original_title": "노이즈", "year": 2025},
                          {"title": "이선빈", "snippet": "《공기살인》 ... 2025년 영화 《노이즈》 ... 주영 역"})


def test_wiki_retry_after_and_ascii_contact(monkeypatch):
    from scene_source.config import Settings
    import scene_source.sources as source_module
    monkeypatch.setenv("WIKI_CONTACT", "test@example.com")
    wiki = Wikipedia(Settings(request_interval=0))
    class Response:
        status_code = 429
        headers = {"Retry-After": "1"}
        def raise_for_status(self): pass
        def json(self): return {"query": {"search": []}}
    first, second = Response(), Response()
    second.status_code = 200
    responses = iter([first, second])
    monkeypatch.setattr(wiki.session, "get", lambda *_a, **_k: next(responses))
    delays = []
    monkeypatch.setattr(source_module.time, "sleep", lambda t: delays.append(t))
    assert wiki.search("명량") == []
    assert 1 in delays
    monkeypatch.setenv("WIKI_CONTACT", "한글연락처")
    with pytest.raises(UserFacingError):
        Wikipedia(Settings())
