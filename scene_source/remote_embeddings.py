"""GMS embeddings for public servers; local installs keep the offline model."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from functools import lru_cache

import numpy as np
import requests

from .models import UserFacingError

MODEL = "text-embedding-3-small"
DIMENSIONS = 384
PUBLIC_MODEL = "gms/text-embedding-3-small@384"
SMOKE_QUERIES = ["블랙홀의 사건의 지평선은 무엇인가요?", "2010년 이후 한국 영화 중 130분 이하인 스릴러"]


def embedding_key(text):
    return hashlib.sha256((PUBLIC_MODEL+"\0"+text).encode()).hexdigest()


class GMSRemoteEmbeddings:
    def __init__(self, settings, before_request=None, session=None):
        self.settings=settings
        self.before_request=before_request
        self.session=session or requests.Session()
        self.path=settings.data_dir/"remote_embeddings.sqlite3"
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.calls=0;self.usage=[]
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors(k TEXT PRIMARY KEY, body TEXT NOT NULL)")
        self._load_seed()

    def connect(self):
        return sqlite3.connect(self.path,timeout=30)

    def _load_seed(self):
        seed=self.settings.root/"public_data/embeddings.jsonl"
        manifest=self.settings.root/"public_data/embedding_manifest.json"
        if not seed.exists():return
        if not manifest.exists():raise UserFacingError("공개 임베딩 자료의 검증 정보가 없습니다.")
        info=json.loads(manifest.read_text(encoding="utf-8"))
        digest=hashlib.sha256(seed.read_bytes()).hexdigest()
        if info.get("model")!=PUBLIC_MODEL or info.get("sha256")!=digest:
            raise UserFacingError("공개 임베딩 자료가 현재 모델과 맞지 않거나 손상됐습니다.")
        with self.connect() as db,seed.open(encoding="utf-8") as stream:
            for line in stream:
                row=json.loads(line)
                vector=self._valid_vector(row["vector"])
                db.execute("INSERT OR IGNORE INTO vectors VALUES(?,?)",(row["key"],json.dumps(vector)))

    @staticmethod
    def _valid_vector(vector):
        array=np.asarray(vector,dtype=float)
        if array.shape!=(DIMENSIONS,) or not np.isfinite(array).all() or np.linalg.norm(array)<1e-12:
            raise UserFacingError("GMS 임베딩 벡터의 형식이 올바르지 않습니다.")
        return (array/np.linalg.norm(array)).tolist()

    def _request(self,texts):
        key=os.getenv("GMS_KEY","").strip()
        if not key:raise UserFacingError("새로운 검색에는 GMS 임베딩 연결이 필요합니다. 서버의 키 설정을 확인해 주세요.")
        if self.before_request:self.before_request()
        base=os.getenv("GMS_BASE_URL","https://gms.ssafy.io/gmsapi/api.openai.com/v1/").rstrip("/")
        for attempt in range(3):
            try:
                response=self.session.post(base+"/embeddings",headers={"Authorization":"Bearer "+key},
                    json={"model":MODEL,"dimensions":DIMENSIONS,"input":texts},timeout=60)
            except requests.RequestException:
                if attempt<2:time.sleep(2**attempt);continue
                raise UserFacingError("GMS 임베딩에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.") from None
            if response.status_code in {429,500,502,503,504} and attempt<2:
                time.sleep(2**attempt);continue
            if response.status_code in {401,403,404}:
                raise UserFacingError("GMS 임베딩 모델의 키 또는 접근 권한을 확인해 주세요.")
            if response.status_code==429:
                raise UserFacingError("GMS 임베딩 사용 한도에 도달했습니다. 잠시 후 다시 시도해 주세요.")
            if response.status_code!=200:
                raise UserFacingError("GMS 임베딩 응답을 받지 못했습니다. 다시 시도해 주세요.")
            try:
                payload=response.json();rows=payload["data"]
                if len(rows)!=len(texts) or {r["index"] for r in rows}!=set(range(len(texts))):raise ValueError()
                vectors=[self._valid_vector(r["embedding"]) for r in sorted(rows,key=lambda r:r["index"])]
            except (ValueError,KeyError,TypeError):
                raise UserFacingError("GMS 임베딩 응답 형식을 확인하지 못했습니다.") from None
            self.calls+=1;self.usage.append(payload.get("usage"))
            return vectors

    def documents(self,texts):
        if not texts:return []
        unique=list(dict.fromkeys(embedding_key(t) for t in texts))
        found={}
        with self.connect() as db:
            for key in unique:
                row=db.execute("SELECT body FROM vectors WHERE k=?",(key,)).fetchone()
                if row:found[key]=json.loads(row[0])
        missing={embedding_key(t):t for t in texts if embedding_key(t) not in found}
        items=list(missing.items())
        for start in range(0,len(items),128):
            batch=items[start:start+128]
            vectors=self._request([t for _,t in batch])
            with self.connect() as db:
                for (key,_),vector in zip(batch,vectors):
                    found[key]=vector
                    db.execute("INSERT OR REPLACE INTO vectors VALUES(?,?)",(key,json.dumps(vector)))
        return [found[embedding_key(t)] for t in texts]

    @lru_cache(maxsize=128)
    def query(self,text):
        return self.documents([text])[0]


def create_embeddings(settings,before_request=None):
    if settings.embedding_model==PUBLIC_MODEL:
        return GMSRemoteEmbeddings(settings,before_request)
    from .embeddings import LocalEmbeddings
    return LocalEmbeddings(settings)
