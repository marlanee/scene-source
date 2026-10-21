from __future__ import annotations

import io
import json
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from scene_source.config import Settings
from scene_source.experiments import ExperimentStore, compare_retrievers, export_csv, export_markdown, run_experiment
from scene_source.models import UserFacingError
from scene_source.pdf_sources import import_pdf, index_pdfs, read_pdf, records, remove_pdf
from test_core import FakeEmbedding, movie, docs, store


def pdf_bytes(text="The event horizon is a boundary around a black hole. Nothing can escape. This test source contains enough body text to be retrieved as evidence."):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject("/Type"):NameObject("/Font"), NameObject("/Subtype"):NameObject("/Type1"), NameObject("/BaseFont"):NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):DictionaryObject({NameObject("/F1"):writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 30 700 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    target = io.BytesIO()
    writer.write(target)
    return target.getvalue()


def test_pdf_page_citation_idempotence_isolation_and_delete(store, movie, docs):
    store.index_movies([movie])
    store.index_documents([docs[0]])
    payload = pdf_bytes()
    item = import_pdf(store.settings, movie, payload, title="Test PDF", publisher="Test Publisher", filename="../../source.pdf")
    assert item["filename"] == "source.pdf"
    assert read_pdf(payload)[0][0] == 1
    assert import_pdf(store.settings, movie, payload, title="Test PDF", publisher="Test Publisher", filename="a.pdf")["id"] == item["id"]
    assert len(records(store.settings.root)) == 1
    assert index_pdfs(store.settings, store, "2") == 0
    index_pdfs(store.settings, store, "1")
    count = store.docs.count()
    index_pdfs(store.settings, store, "1")
    assert store.docs.count() == count
    store.index_documents([{"movie_id":"1", "page_id":f"pdf:{item['id']}:p0", "title":"Cover only",
        "text":"National Aeronautics and Space Administration - A Field Guide to Black Holes",
        "url":item["url"], "source":"공식 PDF", "page_number":1}])
    evidence = store.retrieve("1", "black hole", "mmr")
    assert all(e.title != "Cover only" for e in evidence)
    pdf = next(e for e in evidence if e.source_type == "사용자 PDF")
    assert pdf.page_number == 1 and pdf.url.endswith("#page=1")
    assert pdf.publisher == "Test Publisher"
    remove_pdf(store.settings, item["id"], store)
    assert not records(store.settings.root)
    assert store.docs.count() < count
    assert store.docs.get(where={"page_id":"10"})["ids"]  # wiki survives deletion


@pytest.mark.parametrize("payload",[b"plain text",b"%PDF-broken",b"x"*(10*1024*1024+1)], ids=["not-pdf", "corrupt", "oversize"])
def test_invalid_pdf_is_safe(payload):
    with pytest.raises(UserFacingError):
        read_pdf(payload)


def test_compare_metrics_are_actual_and_movie_filtered(store, movie, docs):
    store.index_movies([movie])
    store.index_documents(docs)
    report = compare_retrievers(store,"1","space")
    for method in report["methods"].values():
        assert all(e["movie_id"]=="1" for e in method["evidence"])
        assert method["metrics"]["external_chunks"] == len(method["evidence"])-1
        assert method["metrics"]["mean_pairwise_similarity"] is not None
    assert "보장하지" in report["note"]


def test_experiment_retry_preserves_arm_and_evidence_then_review(store, movie, docs):
    movie = {**movie, "title":"인터스텔라"}
    store.index_movies([movie]); store.index_documents([docs[0]])
    class Unstable:
        calls = []
        def text(self, system, payload):
            data=json.loads(payload); self.calls.append((system,data))
            if len(self.calls)==2:
                raise UserFacingError("일시적 실패")
            return "설명 [근거 1]" if data["evidence"] else "설명"
    llm=Unstable()
    first=run_experiment(store.settings,[movie],store,lambda *_: {"cache_hit":True},["Q1"],llm=llm)
    assert first["cases"][0]["status"]=="failed"
    assert "without_rag" in first["cases"][0]
    original_evidence=first["cases"][0]["evidence"]
    second=run_experiment(store.settings,[movie],store,lambda *_: pytest.fail("cached evidence must survive"),["Q1"],llm=llm,resume_id=first["id"])
    assert len(llm.calls)==3
    assert second["cases"][0]["status"]=="completed"
    assert second["cases"][0]["evidence"]==original_evidence
    assert llm.calls[0][0]==llm.calls[-1][0]
    assert {k:v for k,v in llm.calls[0][1].items() if k!="evidence"} == {k:v for k,v in llm.calls[-1][1].items() if k!="evidence"}
    db=ExperimentStore(store.settings.root)
    reviewed=db.review(second["id"],"Q1",{"baseline_accuracy":1,"rag_accuracy":2,"support":2,"baseline_abstention":1,"rag_abstention":2,"notes":"=unsafe"})
    assert db.get(second["id"])["cases"][0]["review"]["rag_accuracy"]==2
    assert "RAG 미사용" in export_markdown(reviewed)
    assert "'=unsafe" in export_csv(reviewed)
    with pytest.raises(UserFacingError,match="같은 검색 설정"):
        run_experiment(replace(store.settings,top_k=1),[movie],store,lambda *_:{},["Q1"],llm=llm,resume_id=second["id"])


def test_api_validation_job_key_gate_origin_pdf_and_errors(tmp_path, movie, docs, monkeypatch):
    import scene_source.api as module
    monkeypatch.setattr(module,"load_movies",lambda *_:[movie])
    import scene_source.sources as sources
    monkeypatch.setattr(sources,"bundle_documents",lambda *_:[docs[0]])
    monkeypatch.delenv("GMS_KEY",raising=False)
    app=module.create_app(Settings(root=tmp_path),FakeEmbedding())
    client=TestClient(app)
    assert client.get("/api/status").json()["movie_count"]==1
    assert not client.get("/api/status").json()["gms_available"]
    assert client.post("/api/jobs",json={"action":"evaluate","movie_id":"1"}).status_code==400
    assert client.post("/api/jobs",json={"action":"compare","movie_id":"1","question":"x","top_k":10,"fetch_k":2}).status_code==422
    assert client.post("/api/jobs",json={"action":"chat","movie_id":"999","question":"x"}).status_code==404
    assert client.get("/api/status",headers={"Origin":"https://untrusted.example"}).status_code==403
    assert client.get("/api/status",headers={"Host":"untrusted.example"}).status_code==403
    assert client.get("/api/jobs/not-existing").status_code==404
    assert client.get("/api/pdf/../../env/file").status_code==404
    params={"title":"Test","publisher":"Test","filename":"test.pdf"}
    assert client.post("/api/pdf/1",params=params,content=b"not pdf").status_code==400
    uploaded=client.post("/api/pdf/1",params=params,content=pdf_bytes())
    assert uploaded.status_code==201
    source_id=uploaded.json()["id"]
    response=client.get("/api/pdf/"+source_id+"/file")
    assert response.status_code==200 and response.headers["content-type"]=="application/pdf"
    job=client.post("/api/jobs",json={"action":"compare","movie_id":"1","question":"black hole"}).json()["id"]
    for _ in range(100):
        result=client.get("/api/jobs/"+job).json()
        if result["status"] in {"completed","failed"}: break
        time.sleep(.02)
    assert result["status"]=="completed",result
    assert result["result"]["methods"]["mmr"]["evidence"][0]["source_type"]=="영화 메타데이터"
    assert client.delete("/api/pdf/"+source_id).status_code==200
    app.state.runtime.executor.shutdown(wait=True)
