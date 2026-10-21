import json
import time
from copy import deepcopy
import pytest
from fastapi.testclient import TestClient
from scene_source.config import Settings
from scene_source.models import UserFacingError
from scene_source.security import PublicQuota
from scene_source.web_research import research_web,web_results,web_url
from test_core import FakeEmbedding,movie,docs

def response(rows=None,urls=None,text=None):
    rows=rows if rows is not None else [{"title":"AI 제목","url":"https://news.example/story?utm_source=track#part","summary":"요약 &amp; 기사","reason":"질문 관련 자료"}]
    return {"status":"completed","output":[
        {"type":"web_search_call","status":"completed","action":{"type":"search","queries":["트리니티 맨해튼 계획"],"sources":[{"url":u} for u in (urls if urls is not None else ["https://news.example/story"])]}},
        {"type":"message","content":[{"type":"output_text","text":text if text is not None else json.dumps({"items":rows}),"annotations":[{"type":"url_citation","url":"https://news.example/story","title":"<b>실제 검색 제목</b>"}]}]}]}

class LLM:
    def __init__(self,data=None):self.calls=0;self.data=data
    def web_search(self,instructions,payload):
        self.calls+=1
        return self.data if self.data is not None else response()

def test_tool_sources_are_grounded_and_cached(tmp_path,movie):
    settings=Settings(root=tmp_path);llm=LLM()
    report=research_web(settings,movie,"관계는?",llm)
    assert len(report["items"])==1
    item=report["items"][0]
    assert item["url"]=="https://news.example/story"
    assert item["title"]=="실제 검색 제목" and item["snippet"]=="요약 & 기사"
    assert item["snippet_kind"].startswith("AI 자료 요약")
    again=research_web(settings,movie,"관계는?",llm)
    assert again["cache_hit"] and llm.calls==1
    assert "인용 근거가 아니" in report["notice"]

@pytest.mark.parametrize("url",["http://localhost/x","http://10.1.1.1/x","http://[::1]/x","file:///tmp/x","data:text/html,x","https://user:secret@example.com/a","https://x.local/a","https://example.com:8000/a"])
def test_unsafe_result_urls_are_not_published(url):assert web_url(url) is None

def test_no_invented_links_or_fake_search(tmp_path,movie):
    with pytest.raises(UserFacingError,match="원문 링크"):
        web_results(response(rows=[{"title":"Invented","url":"https://invented.example/a"}]))
    empty=research_web(Settings(root=tmp_path),movie,"없음",LLM(response(rows=[],urls=[])))
    assert empty["items"]==[] and not empty["cache_hit"]
    with pytest.raises(UserFacingError,match="실제 인터넷 검색 기록"):
        web_results({"output":[{"type":"message","content":[]}]})

def test_unparseable_output_falls_back_to_cited_metadata_only():
    items,queries,calls=web_results(response(text="unstructured text claiming https://invented.example"))
    assert len(items)==1 and items[0]["url"]=="https://news.example/story"
    assert "요약을 제공하지 못했" in items[0]["snippet"] and queries and calls==1

def test_native_web_quota_blocks_before_sdk(monkeypatch):
    from scene_source.llm import GMSClient
    client=GMSClient.__new__(GMSClient)
    def no_budget():raise UserFacingError("오늘의 사용량")
    client.before_request=no_budget
    monkeypatch.setattr("openai.OpenAI",lambda **_:pytest.fail("must stop before SDK"))
    with pytest.raises(UserFacingError,match="사용량"):client.web_search("instruction","question")

def test_hourly_100_across_burst_buckets_and_restart(tmp_path,monkeypatch):
    monkeypatch.delenv("GMS_HOURLY_REQUEST_LIMIT",raising=False)
    tick=[0];monkeypatch.setattr("scene_source.security.time.time",lambda:tick[0])
    for i in range(100):
        tick[0]=i*35;PublicQuota(tmp_path).request("same-client",True)
    with pytest.raises(UserFacingError,match="시간당 100회"):PublicQuota(tmp_path).request("same-client",True)
    PublicQuota(tmp_path).request("other-client",True)
    tick[0]=3600;PublicQuota(tmp_path).request("same-client",True)

def test_research_job_preserves_answer_sources_and_index(tmp_path,movie,docs,monkeypatch):
    import scene_source.api as module
    import scene_source.sources as sources
    import scene_source.llm as llms
    monkeypatch.delenv("SCENE_PUBLIC",raising=False);monkeypatch.setenv("GMS_KEY","test-only")
    monkeypatch.setattr(module,"load_movies",lambda *_:[movie])
    monkeypatch.setattr(sources,"bundle_documents",lambda *_:[docs[0]])
    app=module.create_app(Settings(root=tmp_path),FakeEmbedding());client=TestClient(app)
    def completed(payload):
        result=client.post("/api/jobs",json=payload);assert result.status_code==202,result.text
        jid=result.json()["id"]
        for _ in range(150):
            job=client.get("/api/jobs/"+jid).json()
            if job["status"] in {"completed","failed"}:break
            time.sleep(.01)
        assert job["status"]=="completed",job
        return jid,job["result"]
    jid,answer=completed({"action":"chat","movie_id":"1","question":"우주 배경","mode":"local"})
    snapshot=deepcopy(answer)
    assert app.state.runtime.store(app.state.runtime.settings).docs.count()>0
    monkeypatch.setattr(app.state.runtime,"store",lambda *_:pytest.fail("must not touch index"))
    monkeypatch.setattr(llms,"GMSClient",lambda **_:LLM())
    _,extra=completed({"action":"research","movie_id":"1","question":"우주 배경","mode":"gms"})
    assert extra["items"] and "answer" not in extra and "used_sources" not in extra
    assert client.get("/api/jobs/"+jid).json()["result"]==snapshot
    assert client.post("/api/jobs",json={"action":"research","movie_id":"1","question":"x","mode":"local"}).status_code==422
    monkeypatch.delenv("GMS_KEY",raising=False)
    assert client.post("/api/jobs",json={"action":"research","movie_id":"1","question":"x","mode":"gms"}).status_code==400
    app.state.runtime.executor.shutdown(wait=True)
