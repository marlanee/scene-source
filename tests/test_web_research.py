from __future__ import annotations

import json
import time
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from scene_source.config import Settings
from scene_source.llm import Keywords, SelectedDocuments
from scene_source.models import UserFacingError
from scene_source.security import PublicQuota
from scene_source.web_research import research_web, web_url
from test_core import FakeEmbedding, movie, docs


class LLM:
    def __init__(self, selected=None):
        self.calls=0
        self.selected=selected

    def structured(self, system, payload, schema):
        self.calls+=1
        if schema is Keywords:
            return Keywords(keywords=["트리니티 맨해튼 계획", "트리니티 실험 뉴스"])
        candidates=json.loads(payload)["candidates"]
        return SelectedDocuments(selected_ids=self.selected if self.selected is not None else [candidates[0]["page_id"]],
                                 reasons={candidates[0]["page_id"]:"실험과 사업의 관계를 다룬 자료"})


def test_web_sources_are_real_candidates_deduplicated_and_cached(tmp_path,movie):
    calls=[]
    def search(query):
        calls.append(query)
        return [{"title":"<b>실험 이야기</b>","href":"https://news.example/story?utm_source=track#part","body":"요약 &amp; 기사"},
                {"title":"중복","href":"https://news.example/story","body":"다른 설명"},
                {"title":"유해 링크","href":"javascript:alert(1)","body":"x"},
                {"title":"내부 주소","href":"http://127.0.0.1/private","body":"x"}]
    settings=Settings(root=tmp_path)
    llm=LLM()
    report=research_web(settings,movie,"관계는?",llm,search=search)
    assert len(report["items"])==1
    item=report["items"][0]
    assert item["url"]=="https://news.example/story"
    assert item["title"]=="실험 이야기" and item["snippet"]=="요약 & 기사"
    assert item["snippet_kind"]=="검색엔진 요약"
    again=research_web(settings,movie,"관계는?",llm,search=lambda *_:pytest.fail("cache must avoid search"))
    assert again["cache_hit"] and llm.calls==2 and len(calls)==2
    assert "인용 근거가 아니" in report["notice"]


@pytest.mark.parametrize("url",["http://localhost/x","http://10.1.1.1/x","http://[::1]/x", "file:///tmp/x", "data:text/html,x", "https://user:secret@example.com/a", "https://x.local/a", "https://example.com:8000/a"])
def test_unsafe_result_urls_are_not_published(url):
    assert web_url(url) is None


def test_ai_cannot_invent_result_ids_and_empty_search_is_not_fabricated(tmp_path,movie):
    settings=Settings(root=tmp_path)
    with pytest.raises(UserFacingError,match="원문 링크"):
        research_web(settings,movie,"질문",LLM(["invented"]),search=lambda *_:[{"title":"Real","href":"https://example.com/a","body":"real"}])
    empty=research_web(settings,movie,"없음",LLM(),search=lambda *_:[])
    assert empty["items"]==[] and not empty["cache_hit"]
    def unavailable(*_):raise RuntimeError("secret provider traceback")
    with pytest.raises(UserFacingError,match="기존 답변은 유지") as exc:
        research_web(settings,movie,"오류",LLM(),search=unavailable)
    assert "secret" not in str(exc.value)


def test_hourly_100_is_enforced_across_burst_buckets_and_new_instance(tmp_path,monkeypatch):
    monkeypatch.delenv("GMS_HOURLY_REQUEST_LIMIT",raising=False)
    tick=[0]
    monkeypatch.setattr("scene_source.security.time.time",lambda:tick[0])
    for i in range(100):
        tick[0]=i*35
        PublicQuota(tmp_path).request("same-client",True)
    with pytest.raises(UserFacingError,match="시간당 100회"):
        PublicQuota(tmp_path).request("same-client",True)
    PublicQuota(tmp_path).request("other-client",True)
    tick[0]=3600
    PublicQuota(tmp_path).request("same-client",True)


def test_research_job_preserves_existing_answer_sources_and_index(tmp_path,movie,docs,monkeypatch):
    import scene_source.api as module
    import scene_source.sources as sources
    import scene_source.llm as llms
    import scene_source.web_research as research
    monkeypatch.delenv("SCENE_PUBLIC",raising=False)
    monkeypatch.setenv("GMS_KEY","test-only")
    monkeypatch.setattr(module,"load_movies",lambda *_:[movie])
    monkeypatch.setattr(sources,"bundle_documents",lambda *_:[docs[0]])
    app=module.create_app(Settings(root=tmp_path),FakeEmbedding())
    client=TestClient(app)
    def completed(payload):
        response=client.post("/api/jobs",json=payload)
        assert response.status_code==202,response.text
        jid=response.json()["id"]
        for _ in range(150):
            job=client.get("/api/jobs/"+jid).json()
            if job["status"] in {"completed","failed"}:break
            time.sleep(.01)
        assert job["status"]=="completed",job
        return jid,job["result"]
    first_id,answer=completed({"action":"chat","movie_id":"1","question":"우주 배경","mode":"local"})
    snapshot=deepcopy(answer)
    count=app.state.runtime.store(app.state.runtime.settings).docs.count()
    monkeypatch.setattr(app.state.runtime,"store",lambda *_:pytest.fail("web search must not open or change vector store"))
    monkeypatch.setattr(llms,"GMSClient",lambda **_:LLM())
    monkeypatch.setattr(research,"search_engine",lambda *_:[{"title":"외부 기사","href":"https://example.com/a","body":"검색 요약"}])
    _,extra=completed({"action":"research","movie_id":"1","question":"우주 배경","mode":"gms"})
    assert extra["items"] and "answer" not in extra and "used_sources" not in extra
    assert client.get("/api/jobs/"+first_id).json()["result"]==snapshot
    assert count>0
    assert client.post("/api/jobs",json={"action":"research","movie_id":"1","question":"x","mode":"local"}).status_code==422
    monkeypatch.delenv("GMS_KEY",raising=False)
    assert client.post("/api/jobs",json={"action":"research","movie_id":"1","question":"x","mode":"gms"}).status_code==400
    app.state.runtime.executor.shutdown(wait=True)
