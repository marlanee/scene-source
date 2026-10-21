import hashlib,json
from dataclasses import replace
import pytest
from scene_source.config import Settings,load_settings
from scene_source.models import UserFacingError
from scene_source.remote_embeddings import GMSRemoteEmbeddings,PUBLIC_MODEL,embedding_key


class Response:
    status_code=200
    def __init__(self,texts):self.texts=texts
    def json(self):
        return {"data":[{"index":i,"embedding":[float(i+1),1.0]+[0.0]*382} for i in reversed(range(len(self.texts)))],"usage":{"total_tokens":12}}


class Session:
    def __init__(self):self.calls=[]
    def post(self,url,**kwargs):
        self.calls.append((url,kwargs))
        return Response(kwargs["json"]["input"])


def test_remote_embedding_batch_order_cache_and_quota(tmp_path,monkeypatch):
    monkeypatch.setenv("GMS_KEY","test-key")
    session=Session();quota=[]
    embedding=GMSRemoteEmbeddings(Settings(root=tmp_path),lambda:quota.append(1),session)
    vectors=embedding.documents(["first","second","first"])
    assert vectors[0]==vectors[2] and vectors[1][0]>vectors[0][0]
    assert len(vectors[0])==384
    assert session.calls[0][1]["json"]["dimensions"]==384
    assert embedding.documents(["second","first"])==[vectors[1],vectors[0]]
    assert len(session.calls)==len(quota)==1
    # A new object reuses the disk cache, not just the process LRU.
    again=GMSRemoteEmbeddings(Settings(root=tmp_path),session=session)
    assert again.query("first")==vectors[0] and len(session.calls)==1


def test_remote_seed_works_without_key_and_unknown_input_is_gated(tmp_path,monkeypatch):
    monkeypatch.delenv("GMS_KEY",raising=False)
    folder=tmp_path/"public_data";folder.mkdir()
    seed=folder/"embeddings.jsonl"
    seed.write_text(json.dumps({"key":embedding_key("prepared"),"vector":[1.0]+[0.0]*383})+"\n",encoding="utf-8")
    (folder/"embedding_manifest.json").write_text(json.dumps({"model":PUBLIC_MODEL,"sha256":hashlib.sha256(seed.read_bytes()).hexdigest()}),encoding="utf-8")
    session=Session();embedding=GMSRemoteEmbeddings(Settings(root=tmp_path),session=session)
    assert embedding.query("prepared")[0]==1 and not session.calls
    with pytest.raises(UserFacingError,match="AI 임베딩 연결"):embedding.query("new question")


def test_remote_rejects_invalid_dimensions_without_exposing_response(tmp_path,monkeypatch):
    monkeypatch.setenv("GMS_KEY","test-key")
    class Bad(Session):
        def post(self,*args,**kwargs):
            result=Response([]);result.json=lambda:{"data":[{"index":0,"embedding":[1,2]}],"private":"must-not-be-shown"}
            return result
    with pytest.raises(UserFacingError) as exc:
        GMSRemoteEmbeddings(Settings(root=tmp_path),session=Bad()).query("question")
    assert "must-not-be-shown" not in str(exc.value)


def test_public_uses_gms_embedding_but_local_keeps_offline_model(monkeypatch):
    monkeypatch.setenv("SCENE_PUBLIC","1")
    monkeypatch.delenv("PUBLIC_EMBEDDING_MODEL",raising=False)
    assert load_settings().embedding_model==PUBLIC_MODEL
    monkeypatch.setenv("SCENE_PUBLIC","0")
    assert not load_settings().embedding_model.startswith("gms/")
