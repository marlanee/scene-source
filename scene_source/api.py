"""Local REST API. Run: python -m uvicorn scene_source.api:app --host 127.0.0.1 --port 8503"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .config import gms_available, load_settings
from .data import load_movies
from .discovery import discover
from .evaluation import CASES
from .experiments import ExperimentStore, RUBRIC, compare_retrievers, export_csv, export_markdown, run_experiment
from .models import UserFacingError
from .pdf_sources import CATALOG, MAX_BYTES, ensure_catalog, import_pdf, index_pdfs, records, remove_pdf
from .pipeline import build_graph, resolve_question
from .sources import DocumentCollector
from .store import VectorStore
from .security import PublicQuota, VisitorSessions, allowed_hosts, public_mode, public_url

# Some Windows registries label .js as text/plain; nosniff correctly blocks it.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class WorkRequest(BaseModel):
    action: Literal["chat", "compare", "discover", "evaluate", "official", "research"]
    movie_id: str = Field(default="157336", max_length=30)
    question: str = Field(default="", max_length=1000)
    source_mode: Literal["bundle", "live"] = "bundle"
    mode: Literal["local", "gms"] = "local"
    task: Literal["qa", "factcheck"] = "qa"
    method: Literal["similarity", "mmr"] = "mmr"
    top_k: int = Field(default=5, ge=1, le=20)
    fetch_k: int = Field(default=20, ge=1, le=100)
    mmr_lambda: float = Field(default=.65, ge=0, le=1)
    hybrid: bool = True
    history: list[Message] = Field(default_factory=list, max_length=12)
    case_ids: list[Literal["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]] = Field(default_factory=lambda: [c["id"] for c in CASES], max_length=6)
    resume_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")

    @model_validator(mode="after")
    def valid_options(self):
        if self.fetch_k < self.top_k:
            raise ValueError("후보 수는 Top-k 이상이어야 합니다.")
        if self.action in {"chat", "compare", "discover", "research"} and not self.question.strip():
            raise ValueError("질문 또는 탐색 조건을 입력해 주세요.")
        if self.action == "research" and self.mode != "gms":
            raise ValueError("추가 확인에는 AI 연결이 필요합니다.")
        if self.action == "evaluate" and not self.case_ids:
            raise ValueError("평가 질문을 선택해 주세요.")
        return self


class Review(BaseModel):
    baseline_accuracy: int = Field(ge=0, le=2)
    rag_accuracy: int = Field(ge=0, le=2)
    support: int = Field(ge=0, le=2)
    baseline_abstention: int = Field(ge=0, le=2)
    rag_abstention: int = Field(ge=0, le=2)
    notes: str = Field(default="", max_length=2000)


class Runtime:
    def __init__(self, settings, embedding=None):
        self.settings = settings
        self.movies = load_movies(settings.root)
        self.by_id = {str(m["movie_id"]): m for m in self.movies}
        self.embedding = embedding
        self.lock = threading.RLock()
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene-work")
        self.experiments = ExperimentStore(settings.root)
        self.sessions = VisitorSessions()
        self.quota = PublicQuota(settings.root) if public_mode() else None

    def store(self, settings):
        if self.embedding is None:
            from .remote_embeddings import create_embeddings
            self.embedding = create_embeddings(settings,self.quota.before_model_call if self.quota else None)
        store = VectorStore(settings, self.embedding)
        store.index_movies(self.movies)
        return store

    def execute(self, req, job_id):
        def progress(message):
            with self.lock:
                self.jobs[job_id]["message"] = message
                self.jobs[job_id]["status"] = "running"
        started = time.perf_counter()
        try:
            from .runtime_memory import memory_snapshot
            if self.settings.public: memory_snapshot("job_start")
            if req.action == "research":
                # Independent from Chroma, collection and the original answer graph.
                from .llm import GMSClient
                from .web_research import research_web
                if not gms_available():
                    raise UserFacingError("추가 확인은 AI 연결 후 사용할 수 있습니다.")
                llm = GMSClient(before_request=self.quota.before_model_call if self.quota else None)
                question = resolve_question(req.question, [{"role":m.role,"content":m.content[:1000]} for m in req.history])
                result = research_web(self.settings, self.by_id[req.movie_id], question, llm, progress)
                result["total_seconds"] = round(time.perf_counter() - started, 3)
                if self.settings.public: memory_snapshot("web_research_ready")
                with self.lock:
                    self.jobs[job_id].update(status="completed", message="추가 자료 확인 완료", result=result)
                return
            progress("검색 모델과 영화 색인 확인 중")
            settings = replace(self.settings, source_mode=req.source_mode, method=req.method,
                top_k=req.top_k, fetch_k=req.fetch_k, mmr_lambda=req.mmr_lambda, hybrid=req.hybrid)
            store = self.store(settings)
            if settings.public:
                from .embeddings import release_unused_heap
                release_unused_heap()
                memory_snapshot("store_ready")
            movie = self.by_id[req.movie_id]
            if req.mode == "gms" and not gms_available():
                raise UserFacingError("AI 키가 없습니다. 로컬 검색 모드를 사용해 주세요.")
            from .llm import GMSClient
            llm = GMSClient(before_request=self.quota.before_model_call if self.quota else None) if req.mode == "gms" else None
            if settings.public: memory_snapshot("gms_client_ready")

            def prepare(m, question, use_llm=llm):
                progress(f"{m['title']} · 위키백과와 PDF 자료 준비 중")
                collection = DocumentCollector(settings, store, use_llm).collect(m, question)
                collection["pdf_pages"] = index_pdfs(settings, store, str(m["movie_id"]))
                return collection

            if req.action == "discover":
                progress("자연어 조건으로 영화 후보 검색 중")
                result = discover(req.question, self.movies, store, llm)
            elif req.action == "official":
                progress("NASA 공식 PDF 내려받기와 쪽별 색인 중")
                imported = ensure_catalog(settings, movie)
                pages = index_pdfs(settings, store, req.movie_id)
                result = {"imported": imported, "indexed_pages": pages}
            elif req.action == "evaluate":
                result = run_experiment(settings, self.movies, store,
                    lambda m, q: prepare(m, q, None), req.case_ids, progress, resume_id=req.resume_id)
            else:
                history = [m.model_dump() for m in req.history]
                collection_start = time.perf_counter()
                collection = prepare(movie, resolve_question(req.question, history))
                collection_seconds = time.perf_counter() - collection_start
                if settings.public:
                    release_unused_heap()
                    memory_snapshot("collection_ready")
                progress("같은 영화의 근거를 검색하고 결과 구성 중")
                if req.action == "compare":
                    result = compare_retrievers(store, req.movie_id, req.question)
                else:
                    state = build_graph(store, llm).invoke({"movie": movie, "question": req.question,
                        "history": history, "task": req.task, "method": req.method})
                    result = {k: state[k] for k in ["answer", "used_sources", "generation_mode", "timings"]}
                    result["evidence"] = [e.to_dict() for e in state["evidence"]]
                    result["timings"]["collection_seconds"] = round(collection_seconds, 3)
                result["collection"] = collection
            result["total_seconds"] = round(time.perf_counter() - started, 3)
            if settings.public: memory_snapshot("answer_ready")
            with self.lock:
                self.jobs[job_id].update(status="completed", message="완료", result=result)
        except UserFacingError as exc:
            with self.lock:
                self.jobs[job_id].update(status="failed", message=str(exc))
        except Exception:
            with self.lock:
                self.jobs[job_id].update(status="failed",
                    message="작업을 완료하지 못했습니다. 데이터·네트워크·검색 설정을 확인해 주세요.")


def create_app(settings=None, embedding=None):
    settings = settings or load_settings()
    settings = replace(settings, public=public_mode())
    runtime = Runtime(settings, embedding)
    api = FastAPI(title="Scene & Source", version="2.1.1")
    api.state.runtime = runtime

    @api.middleware("http")
    async def local_origin(request: Request, call_next):
        # Bind only to loopback. Reject hostile browser origins and DNS rebinding.
        host = request.url.hostname
        if host not in allowed_hosts():
            return JSONResponse({"detail": "로컬 주소로 접속해 주세요."}, status_code=403)
        origin = request.headers.get("origin")
        valid_origins = {f"{request.url.scheme}://{request.headers.get('host')}"}
        if public_mode() and public_url():
            valid_origins.add(public_url())
        if origin and origin not in valid_origins:
            return JSONResponse({"detail": "다른 사이트의 요청은 허용하지 않습니다."}, status_code=403)
        if request.url.path.startswith("/api/") and request.headers.get("content-length", "0").isdigit():
            if int(request.headers.get("content-length", "0")) > MAX_BYTES:
                return JSONResponse({"detail": "요청 파일이 너무 큽니다."}, status_code=413)
        visitor_id, cookie, is_new = runtime.sessions.issue(request.cookies.get("scene_visitor"))
        request.state.visitor_id = visitor_id
        response = await call_next(request)
        if is_new:
            response.set_cookie("scene_visitor", cookie, httponly=True, secure=public_mode(), samesite="lax", max_age=86400)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path not in {"/docs", "/redoc"}:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        return response

    @api.exception_handler(UserFacingError)
    async def user_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @api.exception_handler(Exception)
    async def unexpected_error(request, exc):
        return JSONResponse({"detail": "서버 처리에 실패했습니다. 잠시 후 다시 시도해 주세요."}, status_code=500)

    @api.get("/api/status")
    def status():
        return {"gms_available": gms_available(), "movie_count": len(runtime.movies),
                "embedding": settings.embedding_model, "version": "2.1.1",
                "pdf_count": len(records(settings.root)), "local_only": not public_mode(),
                "public_demo": public_mode(), "source_mode": "live" if public_mode() else settings.source_mode,
                "admin_features": not public_mode(), "default_mode": "gms" if gms_available() else "local",
                "hourly_request_limit": max(1, int(os.getenv("GMS_HOURLY_REQUEST_LIMIT", "100"))),
                "daily_model_call_limit": max(1, int(os.getenv("GMS_DAILY_CALL_LIMIT", "100"))),
                "web_research": True}

    @api.get("/healthz", include_in_schema=False)
    def health():
        return {"status": "ok"}

    @api.get("/api/movies")
    def movies():
        return runtime.movies

    @api.post("/api/jobs", status_code=202)
    def start(req: WorkRequest, request: Request):
        if req.movie_id not in runtime.by_id:
            raise HTTPException(404, "영화를 찾지 못했습니다.")
        if req.action == "evaluate" and not gms_available():
            raise UserFacingError("AI 키 연결 후 실제 RAG 전후 비교를 실행할 수 있습니다.")
        if req.action == "research" and not gms_available():
            raise UserFacingError("추가 확인은 AI 연결 후 사용할 수 있습니다.")
        if public_mode():
            if req.action in {"evaluate", "official"}:
                raise HTTPException(403, "공개 체험에서는 자료 변경과 평가 실행을 제한합니다. 결과 열람과 영화 질문을 이용해 주세요.")
            req.source_mode = "live"
            try:
                runtime.quota.request(request.client.host if request.client else "unknown",
                    req.mode == "gms" or settings.embedding_model.startswith("gms/"))
            except UserFacingError as exc:
                raise HTTPException(429, str(exc)) from None
        with runtime.lock:
            if sum(j["status"] in {"queued", "running"} for j in runtime.jobs.values()) >= 4:
                raise HTTPException(429, "진행 중인 작업이 많습니다. 완료 후 다시 시도해 주세요.")
            for key in list(runtime.jobs):
                if len(runtime.jobs) <= 60:
                    break
                if runtime.jobs[key]["status"] in {"completed", "failed"}:
                    del runtime.jobs[key]
            job_id = uuid.uuid4().hex
            runtime.jobs[job_id] = {"id": job_id, "status": "queued", "message": "작업 대기 중", "owner": request.state.visitor_id}
        runtime.executor.submit(runtime.execute, req, job_id)
        return {"id": job_id}

    @api.get("/api/jobs/{job_id}")
    def job(job_id: str, request: Request):
        with runtime.lock:
            if job_id not in runtime.jobs or runtime.jobs[job_id]["owner"] != request.state.visitor_id:
                raise HTTPException(404, "작업을 찾지 못했습니다. 서버 재시작 후에는 다시 실행해 주세요.")
            return {k:v for k,v in runtime.jobs[job_id].items() if k != "owner"}

    @api.get("/api/library")
    def library(movie_id: str = ""):
        return {"catalog": CATALOG, "sources": [s for s in records(settings.root)
                if (not movie_id or s["movie_id"] == movie_id) and (not public_mode() or s.get("catalog_id"))]}

    @api.post("/api/pdf/{movie_id}", status_code=201)
    async def upload(movie_id: str, request: Request, title: str = "", publisher: str = "", filename: str = "document.pdf"):
        if public_mode():
            raise HTTPException(403, "공개 체험에서는 개인 PDF 업로드를 지원하지 않습니다.")
        if movie_id not in runtime.by_id:
            raise HTTPException(404, "영화를 찾지 못했습니다.")
        if not title.strip() or not publisher.strip() or max(len(title), len(publisher), len(filename)) > 200:
            raise HTTPException(422, "문서 제목과 발행기관을 200자 이내로 입력해 주세요.")
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > MAX_BYTES:
                raise HTTPException(413, "PDF는 10MB 이하로 올려 주세요.")
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(import_pdf, settings, runtime.by_id[movie_id], bytes(payload),
                                      title=title.strip(), publisher=publisher.strip(), filename=filename)

    @api.get("/api/pdf/{source_id}/file")
    def pdf_file(source_id: str):
        if not re.fullmatch(r"[a-f0-9]{24}", source_id):
            raise HTTPException(404, "자료를 찾지 못했습니다.")
        path = settings.data_dir / "pdf_sources" / f"{source_id}.pdf"
        if not path.exists():
            raise HTTPException(404, "자료를 찾지 못했습니다.")
        if public_mode() and not any(r["id"] == source_id and r.get("catalog_id") for r in records(settings.root)):
            raise HTTPException(404, "자료를 찾지 못했습니다.")
        return FileResponse(path, media_type="application/pdf", content_disposition_type="inline")

    @api.delete("/api/pdf/{source_id}")
    def delete_pdf(source_id: str):
        if public_mode():
            raise HTTPException(403, "공개 체험에서는 자료를 삭제할 수 없습니다.")
        with runtime.lock:
            if any(j["status"] in {"queued", "running"} for j in runtime.jobs.values()):
                raise HTTPException(409, "진행 중인 검색이 끝난 뒤 자료를 삭제해 주세요.")
            remove_pdf(settings, source_id, runtime.store(settings))
        return {"deleted": True}

    @api.get("/api/evaluation")
    def evaluations():
        return {"cases": CASES, "rubric": RUBRIC, "runs": runtime.experiments.list(),
                "gms_available": gms_available(), "can_run": gms_available() and not public_mode(),
                "can_review": not public_mode()}

    @api.put("/api/evaluation/{run_id}/{case_id}/review")
    def review(run_id: str, case_id: str, scores: Review):
        if public_mode():
            raise HTTPException(403, "공개 체험의 평가 결과는 읽기 전용입니다.")
        return runtime.experiments.review(run_id, case_id, scores.model_dump())

    @api.get("/api/evaluation/{run_id}/export")
    def export(run_id: str, format: Literal["json", "md", "csv"] = "md"):
        report = runtime.experiments.get(run_id)
        payload = json.dumps(report, ensure_ascii=False, indent=2) if format == "json" else (
            export_markdown(report) if format == "md" else export_csv(report))
        return Response(payload, media_type={"json": "application/json", "md": "text/markdown", "csv": "text/csv"}[format],
                        headers={"Content-Disposition": f'attachment; filename="rag_comparison.{format}"'})

    web = settings.root / "web"
    if web.exists():
        api.mount("/assets", StaticFiles(directory=web), name="assets")
        @api.get("/", include_in_schema=False)
        def home():
            return FileResponse(web / "index.html")
    return api


app = create_app()
