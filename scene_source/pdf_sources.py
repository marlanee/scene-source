"""Page-aware external PDF evidence. Local uploads never leave this machine."""
from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from pypdf import PdfReader

from .models import UserFacingError

CATALOG = [{
    "id": "nasa-black-holes", "movie_ids": ["157336"],
    "title": "블랙홀 필드 가이드 (A Field Guide to Black Holes)",
    "publisher": "NASA / Goddard Space Flight Center",
    "url": "https://science.nasa.gov/wp-content/uploads/2023/09/black_hole_field_guide_web.pdf",
    "landing_url": "https://science.nasa.gov/resource/a-field-guide-to-black-holes-booklet/",
    "description": "블랙홀의 구조·사건의 지평선·관측 방법을 설명하는 NASA 공식 영문 자료",
    "skip_pages": [1, 8],  # cover and publisher's letter are retained in the file, not used as evidence
}]
MAX_BYTES = 10 * 1024 * 1024


def read_pdf(payload: bytes) -> list[tuple[int, str]]:
    if len(payload) > MAX_BYTES:
        raise UserFacingError("PDF는 10MB 이하로 올려 주세요.")
    if not payload.startswith(b"%PDF-"):
        raise UserFacingError("유효한 PDF 파일이 아닙니다.")
    try:
        reader = PdfReader(io.BytesIO(payload), strict=False)
        if reader.is_encrypted:
            raise UserFacingError("암호화된 PDF는 지원하지 않습니다.")
        if len(reader.pages) > 100:
            raise UserFacingError("PDF는 100쪽 이하만 지원합니다.")
        pages = []
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
            text = re.sub(r"[ \t]+", " ", text).strip()
            if len(text) >= 50:
                pages.append((number, text))
        if not pages:
            raise UserFacingError("추출 가능한 본문이 없습니다. 스캔 PDF는 텍스트 PDF로 변환해 주세요.")
        return pages
    except UserFacingError:
        raise
    except Exception:
        raise UserFacingError("PDF 본문을 읽지 못했습니다. 다른 파일을 사용해 주세요.") from None


def records(root: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((root / "data/pdf_sources").glob("*.json"))]


def import_pdf(settings, movie: dict, payload: bytes, *, title: str, publisher: str,
               filename: str, url: str = "", catalog_id: str = "") -> dict:
    pages = read_pdf(payload)
    digest = hashlib.sha256(payload).hexdigest()
    source_id = hashlib.sha256((str(movie["movie_id"]) + digest).encode()).hexdigest()[:24]
    folder = settings.data_dir / "pdf_sources"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / f"{source_id}.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("parser_version") == 2:
            return existing
    total_pages = len(PdfReader(io.BytesIO(payload)).pages)
    catalog = next((c for c in CATALOG if c["id"] == catalog_id), {})
    pages = [(n,t) for n,t in pages if n not in catalog.get("skip_pages", [])]
    safe_filename = Path(filename.replace("\\", "/")).name[:120]
    source = {"id": source_id, "movie_id": str(movie["movie_id"]), "title": title,
              "publisher": publisher, "filename": safe_filename, "sha256": digest,
              "url": url or f"/api/pdf/{source_id}/file",
              "source": "공식 PDF" if catalog_id else "사용자 PDF",
              "catalog_id": catalog_id, "pages": len(pages), "total_pages": total_pages, "parser_version": 2,
              "collected_at": datetime.now(timezone.utc).isoformat()}
    documents = [{**source, "page_id": f"pdf:{source_id}:p{number}", "page_number": number,
                  "title": f"{title} · p.{number}", "related_movie": movie["title"],
                  "url": f"{source['url']}#page={number}", "text": text}
                 for number, text in pages]
    (folder / f"{source_id}.pdf").write_bytes(payload)
    (folder / f"{source_id}.pages.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in documents), encoding="utf-8")
    manifest_path.write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    return source


def ensure_catalog(settings, movie: dict) -> list[dict]:
    imported = []
    for item in CATALOG:
        if str(movie["movie_id"]) not in item["movie_ids"]:
            continue
        known = next((r for r in records(settings.root)
                      if r["catalog_id"] == item["id"] and r["movie_id"] == str(movie["movie_id"])), None)
        if known:
            imported.append(import_pdf(settings, movie, (settings.data_dir / "pdf_sources" / f"{known['id']}.pdf").read_bytes(),
                title=item["title"], publisher=item["publisher"], filename=known["filename"], url=item["url"], catalog_id=item["id"]))
            continue
        try:
            with requests.get(item["url"], timeout=30, stream=True) as response:
                response.raise_for_status()
                payload = bytearray()
                for block in response.iter_content(65536):
                    payload.extend(block)
                    if len(payload) > MAX_BYTES:
                        raise UserFacingError("공식 PDF가 허용 크기를 초과했습니다.")
        except requests.RequestException:
            raise UserFacingError("NASA 공식 PDF를 받지 못했습니다. 네트워크를 확인해 다시 시도해 주세요.") from None
        imported.append(import_pdf(settings, movie, bytes(payload), title=item["title"],
            publisher=item["publisher"], filename=item["url"].rsplit("/", 1)[-1],
            url=item["url"], catalog_id=item["id"]))
    return imported


def index_pdfs(settings, store, movie_id: str):
    docs = []
    for source in records(settings.root):
        if settings.public and not source.get("catalog_id"):
            continue
        if source["movie_id"] == str(movie_id):
            path = settings.data_dir / "pdf_sources" / f"{source['id']}.pages.jsonl"
            pages = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            wanted = {p["page_id"] for p in pages}
            old = store.docs.get(where={"movie_id": str(movie_id)}, include=["metadatas"])
            stale = [i for i,m in zip(old["ids"],old["metadatas"]) if m.get("page_id", "").startswith(f"pdf:{source['id']}:") and m["page_id"] not in wanted]
            if stale:
                store.docs.delete(ids=stale)
            docs.extend(pages)
    if docs:
        store.index_documents(docs)
    return len(docs)


def remove_pdf(settings, source_id: str, store):
    if not re.fullmatch(r"[a-f0-9]{24}", source_id):
        raise UserFacingError("자료 ID가 올바르지 않습니다.")
    folder = settings.data_dir / "pdf_sources"
    path = folder / f"{source_id}.json"
    if not path.exists():
        raise UserFacingError("자료를 찾지 못했습니다.")
    source = json.loads(path.read_text(encoding="utf-8"))
    for name in store.client.list_collections():
        collection_name = name if isinstance(name, str) else name.name
        if collection_name.startswith("movie_evidence_"):
            collection = store.client.get_collection(collection_name, embedding_function=None)
            rows = collection.get(where={"movie_id": source["movie_id"]}, include=["metadatas"])
            ids = [i for i, m in zip(rows["ids"], rows["metadatas"])
                   if str(m.get("page_id", "")).startswith(f"pdf:{source_id}:")]
            if ids:
                collection.delete(ids=ids)
    for suffix in [".json", ".pdf", ".pages.jsonl"]:
        (folder / (source_id + suffix)).unlink(missing_ok=True)
