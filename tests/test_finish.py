from __future__ import annotations
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from scene_source.config import Settings
from scene_source.models import UserFacingError
from scene_source.security import PublicQuota, VisitorSessions
from scene_source.store import VectorStore
from test_core import FakeEmbedding, movie, docs


def test_hybrid_recovers_relationship_semantic_only_misses(tmp_path, movie):
    class Flat:
        def query(self, text): return [1.,0.]
        def documents(self, texts): return [[1.,0.] for _ in texts]
    store=VectorStore(Settings(root=tmp_path,top_k=1,fetch_k=2),Flat())
    store.index_movies([movie])
    store.docs.upsert(ids=["1:x:0","1:y:0"],
        documents=["이 문단은 무기 공장 건설 우선순위와 예산에 관해 설명한다.",
                   "트리니티 실험은 맨해튼 계획에서 만든 핵무기를 시험하기 위한 실험이다."],
        embeddings=[[1.,0.],[.9,.1]],
        metadatas=[{"movie_id":"1","page_id":p,"title":"역사","url":"https://example.org","source_type":"위키백과"} for p in ["x","y"]])
    query="트리니티 실험과 맨해튼 계획은 어떤 관계인가요?"
    plain=VectorStore(replace(store.settings,hybrid=False),Flat()).retrieve("1",query,"similarity")
    mixed=store.retrieve("1",query,"similarity")
    assert plain[1].page_id=="x"
    assert mixed[1].page_id=="y"


def test_visitor_cookie_cannot_be_forged():
    sessions=VisitorSessions()
    first,cookie,new=sessions.issue(None)
    assert new
    assert sessions.issue(cookie)==(first,cookie,False)
    assert sessions.issue("0"*32+"."+cookie.split(".")[1])[0]!=first


def test_public_global_budget_is_persistent_and_blocks_before_sdk(tmp_path,monkeypatch):
    monkeypatch.setenv("GMS_DAILY_CALL_LIMIT","2")
    quota=PublicQuota(tmp_path)
    quota.before_model_call()
    PublicQuota(tmp_path).before_model_call()
    with pytest.raises(UserFacingError,match="사용량"):
        PublicQuota(tmp_path).before_model_call()


def test_public_api_owner_isolation_and_admin_denial(tmp_path,movie,docs,monkeypatch):
    import scene_source.api as module
    monkeypatch.setenv("SCENE_PUBLIC","1")
    monkeypatch.setenv("APP_PUBLIC_URL","https://demo.example")
    monkeypatch.setattr(module,"load_movies",lambda *_:[movie])
    def collect(self,*_):
        self.store.index_documents([docs[0]])
        return {"cache_hit":True}
    monkeypatch.setattr(module.DocumentCollector,"collect",collect)
    app=module.create_app(Settings(root=tmp_path),FakeEmbedding())
    a=TestClient(app,base_url="https://demo.example")
    b=TestClient(app,base_url="https://demo.example")
    assert a.get("/api/status").json()["public_demo"]
    assert a.get("/healthz").status_code==200
    assert a.post("/api/jobs",json={"action":"evaluate","movie_id":"1"}).status_code in {400,403}
    assert a.post("/api/pdf/1?title=t&publisher=p",content=b"private").status_code==403
    assert a.delete("/api/pdf/"+"a"*24).status_code==403
    assert a.put("/api/evaluation/test/Q1/review",json={
        "baseline_accuracy":1,"rag_accuracy":1,"support":1,"baseline_abstention":1,"rag_abstention":1}).status_code==403
    result=a.post("/api/jobs",json={"action":"chat","movie_id":"1","question":"우주 영화의 배경","mode":"local"})
    assert result.status_code==202
    job_id=result.json()["id"]
    for _ in range(100):
        result=a.get("/api/jobs/"+job_id).json()
        if result["status"] in {"failed","completed"}:break
        time.sleep(.02)
    assert result["status"]=="completed",result
    assert "owner" not in result
    assert b.get("/api/jobs/"+job_id).status_code==404
    assert not a.get("/api/evaluation").json()["can_run"]
    assert a.get("/api/status",headers={"Origin":"https://evil.example"}).status_code==403
    assert a.get("/api/status",headers={"Host":"evil.example"}).status_code==403
    assert "Secure" in a.get("/api/status").request.headers.get("cookie","") or a.cookies.get("scene_visitor")
    app.state.runtime.executor.shutdown(wait=True)


def test_film_wiki_is_marked_as_film_not_historical_evidence(tmp_path,movie,docs):
    store=VectorStore(Settings(root=tmp_path),FakeEmbedding())
    store.index_movies([movie]);store.index_documents([docs[0]])
    sources=store.retrieve("1","영화 설명")
    assert all(e.evidence_scope=="film" for e in sources)


def test_public_index_is_separate_and_private_uploads_are_ignored(tmp_path):
    from scene_source.pdf_sources import import_pdf,index_pdfs
    from test_advanced import pdf_bytes
    settings=Settings(root=tmp_path)
    assert settings.docs_collection != replace(settings,public=True).docs_collection
    import_pdf(settings,{"movie_id":"1","title":"test"},pdf_bytes(),title="Private",publisher="Owner",filename="p.pdf")
    store=VectorStore(replace(settings,public=True),FakeEmbedding())
    assert index_pdfs(store.settings,store,"1")==0


def test_gms_keyword_and_selection_are_cached(tmp_path,movie,docs,monkeypatch):
    from scene_source.sources import DocumentCollector
    import scene_source.sources as module
    from scene_source.llm import Keywords,SelectedDocuments
    monkeypatch.setattr(module,"bundle_documents",lambda *_:[docs[0]])
    class LLM:
        calls=[]
        def structured(self,system,payload,schema):
            self.calls.append(schema)
            return Keywords(keywords=["우주"]) if schema is Keywords else SelectedDocuments(selected_ids=[])
    llm=LLM();store=VectorStore(Settings(root=tmp_path),FakeEmbedding())
    c=DocumentCollector(store.settings,store,llm)
    c.collect(movie,"우주에 대해 설명해줘")
    c.collect(movie,"우주에 대해 설명해줘")
    assert llm.calls.count(Keywords)==1
    assert llm.calls.count(SelectedDocuments)==1


def test_public_shared_tokenizer_preserves_long_input_and_restores_model_settings():
    from tokenizers import Tokenizer,models,pre_tokenizers
    from scene_source.embeddings import LocalEmbeddings
    tokenizer=Tokenizer(models.WordLevel({"[UNK]":0,"body":1,"ending":2},unk_token="[UNK]"))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer.enable_truncation(max_length=8)
    tokenizer.enable_padding(length=8)
    original=(tokenizer.truncation,tokenizer.padding)
    embedding=LocalEmbeddings.__new__(LocalEmbeddings)
    embedding.shared_tokenizer=True;embedding.tokenizer=tokenizer
    windows=embedding.windows("body "*245+"ending")
    assert len(windows)==3 and "ending" in windows[-1]
    assert (tokenizer.truncation,tokenizer.padding)==original
    encoded=tokenizer.encode_batch(["body","body ending"])
    assert all(len(e.ids)==8 for e in encoded)
