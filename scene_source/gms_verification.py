"""Run explicitly after the user receives a GMS key. Never run during key-free work."""
from __future__ import annotations

import json
import time

from .config import gms_available
from .data import load_movies
from .embeddings import LocalEmbeddings
from .evaluation import CASES
from .llm import GMSClient
from .models import UserFacingError
from .pipeline import build_graph
from .sources import DocumentCollector
from .store import VectorStore


def verify_gms(settings, compare=False):
    if not gms_available():
        raise UserFacingError("GMS_KEY가 없습니다. 키 수령 후 .env에 설정하고 실행하세요.")
    llm = GMSClient()
    store = VectorStore(settings, LocalEmbeddings(settings))
    movies = load_movies(settings.root)
    store.index_movies(movies)
    if compare:
        from .experiments import run_experiment, export_markdown, export_csv
        from .pdf_sources import index_pdfs
        def prepare(movie, question):
            collection = DocumentCollector(settings, store).collect(movie, question)
            collection["pdf_pages"] = index_pdfs(settings, store, str(movie["movie_id"]))
            return collection
        report = run_experiment(settings, movies, store, prepare, [c["id"] for c in CASES], print, llm=llm)
        folder = settings.root / "docs" / "verification"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "gms_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "gms_comparison.md").write_text(export_markdown(report), encoding="utf-8")
        (folder / "gms_comparison.csv").write_text(export_csv(report), encoding="utf-8")
        return report
    report = {"scope": "GMS 실제 호출 결과. 정확성 평가는 사람이 원문과 대조해야 함.", "cases": []}
    for case in CASES if compare else CASES[:1]:
        movie = next(m for m in movies if m["title"] == case["title"])
        start = time.perf_counter()
        collection = DocumentCollector(settings, store, llm).collect(movie, case["question"])
        result = build_graph(store, llm).invoke({"movie": movie, "question": case["question"],
                                               "task": "factcheck", "method": "mmr"})
        baseline = llm.text("너는 영화 배경 지식 전문가다. 모르면 모른다고 답한다.",
                            f"영화: {case['title']}\n질문: {case['question']}") if compare else None
        report["cases"].append({**case, "without_rag": baseline, "with_rag": result["answer"],
            "used_sources": result["used_sources"], "collection": collection,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "human_accuracy_review": "미평가", "human_abstention_review": "미평가"})
        print(case["id"], "GMS response received; human source review pending", flush=True)
    folder = settings.root / "docs" / "verification"
    folder.mkdir(parents=True, exist_ok=True)
    name = "gms_comparison" if compare else "gms_smoke"
    (folder / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# GMS 실행 결과", "", "실제 출력입니다. 정확성·확인 불가 판정은 원문을 보고 사람이 평가해야 합니다.", ""]
    for row in report["cases"]:
        lines += [f"## {row['id']} · {row['title']}", "", row["question"], "",
                  "### RAG 미사용", "", row["without_rag"] or "단일 연결 검사에서는 비교를 실행하지 않음", "",
                  "### RAG 사용", "", row["with_rag"], "", "### 사용한 출처", ""]
        lines += [f"- [근거 {s['number']}] [{s['title']}]({s['url']})" for s in row["used_sources"]]
        lines += ["", "정확성: 미평가 / 근거의 주장 지지 여부: 미평가 / 확인 불가 적절성: 미평가", ""]
    (folder / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
