from __future__ import annotations

import json
import re

from .data import movie_text
from .llm import Filters, MoviePicks
from .models import UserFacingError
from .prompts import DISCOVER_SELECT_SYSTEM, DISCOVER_SYSTEM

GENRES = ["SF", "드라마", "코미디", "액션", "모험", "애니메이션", "공포", "스릴러", "역사", "전쟁", "로맨스", "가족", "판타지", "범죄", "음악", "미스터리", "다큐멘터리", "서부", "TV 영화"]


def local_filters(query: str) -> Filters:
    f = Filters(semantic_query=query)
    decade = re.search(r"((?:19|20)\d0)년대", query)
    if decade:
        f.min_year, f.max_year = int(decade[1]), int(decade[1]) + 9
    for match in re.finditer(r"((?:19|20)\d{2})\s*년?\s*(이후|이상|부터|초과|이전|이하|까지|미만)", query):
        value, op = int(match[1]), match[2]
        if op in {"이후", "이상", "부터", "초과"}:
            f.min_year = value + (op == "초과")
        else:
            f.max_year = value - (op == "미만")
    for match in re.finditer(r"(\d+)\s*(시간|분)\s*(이하|미만|이상|초과|이내)", query):
        value, unit, op = int(match[1]), match[2], match[3]
        value *= 60 if unit == "시간" else 1
        if op in {"이하", "미만", "이내"}:
            f.max_runtime = value - (op == "미만")
        else:
            f.min_runtime = value + (op == "초과")
    if re.search(r"한국\s*영화|한국어", query):
        f.language = "ko"
    if re.search(r"영어\s*영화", query):
        f.language = "en"
    f.genres = [g for g in GENRES if (re.search(r"(?i)\bsf\b", query) if g == "SF" else g in query)]
    return f


def matches(movie: dict, f: Filters) -> bool:
    year, runtime = movie.get("year"), movie.get("runtime")
    for value, lower, upper in [(year, f.min_year, f.max_year), (runtime, f.min_runtime, f.max_runtime)]:
        if (lower is not None or upper is not None) and (value is None or value == 0):
            return False
        if lower is not None and value < lower:
            return False
        if upper is not None and value > upper:
            return False
    if f.language and movie.get("original_language") != f.language:
        return False
    if f.genres and not set(f.genres).issubset(set(movie.get("genres", []))):
        return False
    if f.director and not any(f.director.casefold() in n.casefold() for n in movie.get("director", [])):
        return False
    return True


def discover(query: str, movies: list[dict], store, llm=None) -> dict:
    query = query.strip()
    if not query:
        return {"items": [], "filters": {}, "verified": False}
    if len(query) > 1000:
        raise UserFacingError("탐색 조건은 1,000자 이내로 입력해 주세요.")
    filters = llm.structured(DISCOVER_SYSTEM, query, Filters) if llm else local_filters(query)
    if filters.min_year is not None and filters.max_year is not None and filters.min_year > filters.max_year:
        raise UserFacingError("연도 조건의 시작이 끝보다 큽니다.")
    eligible = {str(m["movie_id"]): m for m in movies if matches(m, filters)}
    if not eligible:
        return {"items": [], "filters": filters.model_dump(), "verified": bool(llm)}
    # Filter BEFORE ranking: relevant eligible films outside the global top-k remain discoverable.
    where = {"movie_id": {"$in": list(eligible)}}
    ranked = store.search_movies(filters.semantic_query or query, limit=24, where=where)
    candidates = [eligible[i] for i, _ in ranked if i in eligible]
    if llm:
        payload = {"conditions": query, "candidates": [{"movie_id": m["movie_id"], "metadata": movie_text(m)} for m in candidates]}
        picks = llm.structured(DISCOVER_SELECT_SYSTEM, json.dumps(payload, ensure_ascii=False), MoviePicks)
        allowed = {str(m["movie_id"]) for m in candidates}
        if any(p.movie_id not in allowed for p in picks.items):
            raise UserFacingError("탐색 후보 밖 영화가 반환되어 중단했습니다. 조건을 바꾸거나 다시 시도해 주세요.")
        items, seen = [], set()
        for p in picks.items:
            if p.movie_id not in seen:
                seen.add(p.movie_id)
                items.append({"movie": eligible[p.movie_id], "reason": p.reason})
    else:
        items = [{"movie": m, "reason": "명시된 연도·장르·상영시간 조건을 적용한 의미 검색 후보"} for m in candidates[:6]]
    return {"items": items, "filters": filters.model_dump(), "verified": bool(llm)}
