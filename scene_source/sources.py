from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone

import requests

from .cache import CollectionCache
from .data import PRIMARY_ALIASES, bundle_documents
from .llm import Keywords, SelectedDocuments
from .models import UserFacingError
from .prompts import KEYWORDS_SYSTEM, SELECT_SYSTEM


def normalize(text: str) -> str:
    return re.sub(r"[^\w가-힣]", "", unicodedata.normalize("NFKC", text)).casefold()


def is_primary(movie: dict, doc: dict) -> bool:
    title = re.sub(r"\s*\((?:\d{4}년\s*)?(?:영화|애니메이션|드라마)\)\s*", "", doc["title"]).strip()
    lead = (doc.get("text") or doc.get("snippet", ""))[:500]
    names = {normalize(movie["title"]), normalize(movie.get("original_title", ""))} - {""}
    names.update(normalize(x) for x in PRIMARY_ALIASES.get(str(movie["movie_id"]), set()))
    canonical_match = normalize(title) in names
    # Alternate distribution titles occur in the first work-name clause or original title.
    lead_names = [normalize(t) for t in re.findall(r"《([^》]+)》", lead[:300])]
    original = normalize(movie.get("original_title", ""))
    alias_match = "text" in doc and lead.lstrip().startswith("《") and (bool(names.intersection(lead_names)) or
                    (len(original) >= 5 and original in normalize(lead[:300])))
    if not (canonical_match or alias_match):
        return False
    explicit_year = re.search(r"\((\d{4})년\s*영화\)", doc["title"])
    if explicit_year and int(explicit_year[1]) != movie.get("year"):
        return False
    return any(word in doc["title"] + lead for word in ("영화", "애니메이션", "드라마", "무비", "속편"))


def primary_rank(movie: dict, doc: dict) -> int:
    if not is_primary(movie, doc):
        return -1
    score = 10 if f"{movie.get('year')}년 영화" in doc["title"] else 0
    score += 5 if "(영화)" in doc["title"] else 0
    score += 3 if str(movie.get("year", "")) in (doc.get("text") or doc.get("snippet", ""))[:500] else 0
    return score


def lexical_score(query: str, title: str) -> int:
    q, t = normalize(query), normalize(title)
    grams = {q[i:i + 2] for i in range(len(q) - 1)}
    return sum(g in t for g in grams) + (20 if t and t in q else 0)


class Wikipedia:
    def __init__(self, settings):
        self.settings = settings
        contact = os.getenv("WIKI_CONTACT", "").strip()
        if not contact or not contact.isascii() or not ("@" in contact or contact.startswith("https://")):
            raise UserFacingError("실시간 위키 수집은 .env의 WIKI_CONTACT에 영문 이메일 또는 연락처 URL을 설정해야 합니다.")
        self.session = requests.Session()
        self.session.headers["User-Agent"] = f"SceneSource/1.0 ({contact}) python-requests"
        self.last_call = 0.0

    def request(self, params):
        for attempt in range(self.settings.max_retries):
            time.sleep(max(0, self.settings.request_interval - (time.monotonic() - self.last_call)))
            self.last_call = time.monotonic()
            try:
                response = self.session.get("https://ko.wikipedia.org/w/api.php",
                    params={"format": "json", "formatversion": 2, **params}, timeout=20)
                if response.status_code == 429 or response.status_code >= 500:
                    from email.utils import parsedate_to_datetime
                    raw = response.headers.get("Retry-After", "")
                    try:
                        wait = float(raw)
                    except ValueError:
                        try:
                            wait = max(0, (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds())
                        except (TypeError, ValueError):
                            wait = 2 ** (attempt + 1)
                    if wait > 60:
                        raise UserFacingError("문서 출처가 긴 대기 시간을 요청했습니다. 잠시 뒤 재시도하거나 배포 문서 모드를 사용해 주세요.")
                    time.sleep(max(0, wait))
                    continue
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    raise UserFacingError("위키백과 검색에 실패했습니다. 검색어를 바꾸거나 배포 문서를 사용해 주세요.")
                return data
            except (requests.RequestException, ValueError):
                if attempt + 1 == self.settings.max_retries:
                    break
                time.sleep(2 ** attempt)
        raise UserFacingError("위키백과에 연결하지 못했습니다. 배포 문서 모드로 전환할 수 있습니다.")

    def search(self, query):
        data = self.request({"action": "query", "list": "search", "srsearch": query, "srlimit": 5})
        return [{"page_id": str(x["pageid"]), "title": x["title"],
                 "snippet": html.unescape(re.sub(r"<[^>]+>", "", x.get("snippet", "")))}
                for x in data.get("query", {}).get("search", [])]

    def fetch(self, title):
        data = self.request({"action": "query", "prop": "extracts|info", "titles": title,
                            "explaintext": 1, "inprop": "url", "redirects": 1})
        page = data["query"]["pages"][0]
        if page.get("missing") or not page.get("extract", "").strip():
            return None
        return {"page_id": str(page["pageid"]), "title": page["title"], "text": page["extract"],
                "url": page["fullurl"], "source": "한국어 위키백과",
                "collected_at": datetime.now(timezone.utc).date().isoformat()}


class DocumentCollector:
    def __init__(self, settings, store, llm=None, wiki=None):
        self.settings, self.store, self.llm = settings, store, llm
        self.wiki = wiki
        self.cache = CollectionCache(settings.data_dir / "state.sqlite3")

    def cached_search(self, wiki, keyword):
        key = "wiki-search:" + hashlib.sha256(normalize(keyword).encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached["candidates"]
        candidates = wiki.search(keyword)
        self.cache.put(key, {"candidates": candidates})
        return candidates

    def cached_fetch(self, wiki, title):
        key = "wiki-page:" + hashlib.sha256(normalize(title).encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached["document"]
        document = wiki.fetch(title)
        if document:
            self.cache.put(key, {"document": document})
        return document

    def collect(self, movie, question, progress=lambda *_: None):
        available = bundle_documents(self.settings.root, str(movie["movie_id"])) if self.settings.source_mode == "bundle" else []
        progress("질문에서 검색어를 준비하고 있습니다.")
        if self.llm:
            keyword_identity = [movie, normalize(question), os.getenv("GMS_MODEL", "gpt-5.4-nano"), KEYWORDS_SYSTEM]
            keyword_key = "gms-keywords:" + hashlib.sha256(json.dumps(keyword_identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            prepared = self.cache.get(keyword_key)
            if prepared is None:
                keywords = self.llm.structured(KEYWORDS_SYSTEM, json.dumps({"movie": movie, "question": question}, ensure_ascii=False), Keywords).keywords
                self.cache.put(keyword_key, {"keywords": keywords})
            else:
                keywords = prepared["keywords"]
        else:
            # No fake LLM answer: only deterministic candidate preparation in local mode.
            hints = available or bundle_documents(self.settings.root, str(movie["movie_id"]))
            ranked = sorted(hints, key=lambda d: lexical_score(question, d["title"]), reverse=True)
            keywords = [d["title"] for d in ranked if lexical_score(question, d["title"]) > 0]
            if self.settings.source_mode == "live":
                from .retrieval import terms
                keywords = [f"{movie['title']} {movie['year']} 영화", *keywords, *terms(question)[:2],
                            movie.get("original_title", movie["title"])]
        keywords = list(dict.fromkeys([f"{movie['title']} (영화)"] + [k.strip()[:120] for k in keywords if k.strip()]))[:self.settings.max_keywords]
        identity = [movie["movie_id"], self.settings.index_fingerprint, self.settings.source_mode,
                    os.getenv("GMS_MODEL", "gpt-5.4-nano") + SELECT_SYSTEM if self.llm else "local", normalize(question), sorted(normalize(k) for k in keywords)]
        key = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
        cached = self.cache.get(key)
        if cached:
            indexed_ids = set(self.store.docs.get(where={"movie_id": str(movie["movie_id"])}, include=["metadatas"])["ids"])
            if indexed_ids and all(any(f":{pid}:" in x for x in indexed_ids) for pid in cached["page_ids"]):
                return {**cached, "cache_hit": True, "keywords": keywords}

        progress("후보 문서를 모으고 선별하고 있습니다.")
        if self.settings.source_mode == "live":
            wiki = self.wiki or Wikipedia(self.settings)
            candidates = {}
            for keyword in keywords:
                for candidate in self.cached_search(wiki, keyword):
                    candidates[str(candidate["page_id"])] = {**candidate, "search_query": keyword}
            primary = self.cached_fetch(wiki, f"{movie['title']} (영화)")
            if primary and not is_primary(movie, primary):
                primary = None
            if not primary:
                matching = sorted([d for d in candidates.values() if is_primary(movie, d)],
                                  key=lambda d: primary_rank(movie, d), reverse=True)
                primary = self.cached_fetch(wiki, matching[0]["title"]) if matching else None
                if primary and not is_primary(movie, primary):
                    primary = None
            if not primary:
                raise UserFacingError("선택한 영화 자체를 설명하는 문서를 찾지 못했습니다. 배포 문서 모드를 사용하거나 영화 제목을 확인해 주세요.")
            candidates[str(primary["page_id"])] = {**primary, "snippet": primary["text"][:350]}
            candidates = list(candidates.values())
        else:
            candidates = [{**d, "snippet": d["text"][:350]} for d in available]
            primary = max(candidates, key=lambda d: primary_rank(movie, d), default=None)
            if primary is not None and not is_primary(movie, primary):
                primary = None
            if not primary:
                raise UserFacingError("배포 자료에 해당 영화의 본문 문서가 없습니다. 실시간 수집 또는 다른 영화를 선택해 주세요.")

        reasons = {}
        if self.llm:
            payload = {"movie": movie["title"], "question": question,
                       "candidates": [{k: str(d.get(k, "")) for k in ("page_id", "title", "snippet")} for d in candidates]}
            selection = self.llm.structured(SELECT_SYSTEM, json.dumps(payload, ensure_ascii=False), SelectedDocuments)
            allowed = {str(d["page_id"]) for d in candidates}
            if not set(selection.selected_ids).issubset(allowed):
                raise UserFacingError("후보에 없는 문서가 선택되어 중단했습니다. 다시 시도해 주세요.")
            selected_ids, reasons = set(selection.selected_ids), selection.reasons
        else:
            ranked = sorted(candidates, key=lambda d: lexical_score(question, d["title"]), reverse=True)
            selected_ids = {str(d["page_id"]) for d in ranked[:self.settings.max_documents - 1] if lexical_score(question, d["title"]) > 0}
        primary_id = str(primary["page_id"])
        selected_ids.add(primary_id)  # Primary overrides every exclusion, including length.
        selected = [primary] + [d for d in candidates if str(d["page_id"]) in selected_ids and str(d["page_id"]) != primary_id]
        selected = selected[:self.settings.max_documents]
        documents = []
        for d in selected:
            full = self.cached_fetch(wiki, d["title"]) if self.settings.source_mode == "live" and "text" not in d else d
            if full and full.get("text", "").strip():
                documents.append({**full, "movie_id": str(movie["movie_id"]), "related_movie": movie["title"],
                                  "evidence_scope": "film" if is_primary(movie, full) else "background",
                                  "search_query": d.get("search_query", keywords[0])})
        progress("선별된 문서를 청킹하고 색인하고 있습니다.")
        self.store.index_documents(documents)
        result = {"keywords": keywords, "page_ids": [str(d["page_id"]) for d in documents],
                  "selected": [{"title": d["title"], "url": d["url"],
                                "reason": "영화 자체 문서: 항상 포함" if str(d["page_id"]) == primary_id else reasons.get(str(d["page_id"]), "질문 관련 후보")}
                               for d in documents],
                  "rejected": [d["title"] for d in candidates if str(d["page_id"]) not in {str(x["page_id"]) for x in documents}],
                  "source_mode": self.settings.source_mode, "selector": "gms" if self.llm else "local"}
        self.cache.put(key, result)
        return {**result, "cache_hit": False}
