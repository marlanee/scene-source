from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from .models import UserFacingError

# Distribution-title aliases verified against the supplied document leads.
PRIMARY_ALIASES = {
    "181812": {"스타워즈: 라이즈 오브 스카이워커"},
    "38356": {"트랜스포머: 달의 어둠"},
    "615173": {"마녀(魔女) Part2. The Other One"},
}


def prepare_data(root: Path) -> Path:
    target = root / "data" / "raw"
    if (target / "movies.json").exists():
        return target
    archive = root / "data" / "source_pack.zip"
    if not archive.exists():
        archive = root.parent / "영화도메인_데이터_배포용.zip"
    if not archive.exists():
        public_seed = root / "public_data" / "movies.json"
        if public_seed.exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(public_seed, target / "movies.json")
            return target
        raise UserFacingError("영화 데이터가 없습니다. 배포 압축파일을 data/source_pack.zip에 넣어 주세요.")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            path = (target / entry.filename).resolve()
            if not path.is_relative_to(target.resolve()):
                raise UserFacingError("데이터 압축파일 경로가 올바르지 않습니다.")
        z.extractall(target)
    return target


def load_movies(root: Path) -> list[dict]:
    path = prepare_data(root) / "movies.json"
    movies = json.loads(path.read_text(encoding="utf-8-sig"))
    ids = [str(m["movie_id"]) for m in movies]
    if len(ids) != len(set(ids)):
        raise UserFacingError("영화 데이터에 중복 ID가 있습니다.")
    return movies


def movie_text(movie: dict) -> str:
    lines = [("제목", movie["title"]), ("원제", movie.get("original_title", "")),
             ("개봉연도", movie.get("year", "")), ("개봉일", movie.get("release_date", "")),
             ("장르", ", ".join(movie.get("genres", []))),
             ("감독", ", ".join(movie.get("director", [])) or "정보 없음"),
             ("출연", ", ".join(movie.get("cast", []))), ("상영시간(분)", movie.get("runtime", "")),
             ("제작국가", ", ".join(movie.get("production_countries", []))),
             ("원어", movie.get("original_language", "")), ("TMDB 평점", movie.get("vote_average", "")),
             ("줄거리", movie.get("overview", ""))]
    return "\n".join(f"{k}: {v}" for k, v in lines)


def bundle_documents(root: Path, movie_id: str) -> list[dict]:
    folder = prepare_data(root) / "movielab_wiki" / "wikipedia"
    paths = sorted(folder.glob(f"{movie_id}_*.json"))
    paths += sorted((root / "data/supplemental_wikipedia").glob(f"{movie_id}_*.json"))
    documents = {}
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        documents[str(doc["page_id"])] = doc
    return list(documents.values())


def audit_data(root: Path) -> dict:
    movies = load_movies(root)
    paths = sorted((prepare_data(root) / "movielab_wiki" / "wikipedia").glob("*.json"))
    docs = [json.loads(p.read_text(encoding="utf-8-sig")) for p in paths]
    ids = {str(m["movie_id"]) for m in movies}
    from .sources import is_primary
    by_movie = {}
    for doc in docs:
        by_movie.setdefault(str(doc["movie_id"]), []).append(doc)
    missing_primary = [{"movie_id": str(m["movie_id"]), "title": m["title"],
                        "available_titles": [d["title"] for d in by_movie.get(str(m["movie_id"]), [])]}
                       for m in movies if not any(is_primary(m, d) for d in by_movie.get(str(m["movie_id"]), []))]
    return {"movies": len(movies), "unique_movie_ids": len(ids), "documents": len(docs),
            "unique_pages": len({d["page_id"] for d in docs}),
            "movies_with_documents": len({str(d["movie_id"]) for d in docs}),
            "empty_text": sum(not d["text"].strip() for d in docs),
            "empty_url": sum(not d["url"] for d in docs),
            "orphan_movie_ids": sum(str(d["movie_id"]) not in ids for d in docs),
            "missing_director": [m["title"] for m in movies if not m.get("director")],
            "primary_document_not_identified": missing_primary,
            "supplemental_documents": len(list((root / "data/supplemental_wikipedia").glob("*.json"))),
            "missing_primary_after_supplement": [row for row in missing_primary
                if not any(is_primary(next(m for m in movies if str(m["movie_id"]) == row["movie_id"]), d)
                           for d in bundle_documents(root, row["movie_id"]))]}
