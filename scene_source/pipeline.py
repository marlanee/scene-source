from __future__ import annotations

import json
import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .models import Evidence, UserFacingError
from .prompts import ANSWER_SYSTEM

CITE_PATTERN = re.compile(r"\[근거\s*([\d,\s]+)\]")


class RAGState(TypedDict, total=False):
    movie: dict
    question: str
    history: list[dict]
    task: str
    method: str
    evidence: list[Evidence]
    answer: str
    used_sources: list[dict]
    timings: dict
    generation_mode: str


def cited_numbers(answer: str) -> set[int]:
    return {int(n.strip()) for group in CITE_PATTERN.findall(answer)
            for n in group.split(",") if n.strip().isdigit()}


def used_sources(answer: str, evidence: list[Evidence]) -> list[dict]:
    numbers = cited_numbers(answer)
    return [{"number": i, **source.to_dict()} for i, source in enumerate(evidence, 1) if i in numbers]


def validate_citations(answer: str, evidence: list[Evidence]) -> None:
    if cited_numbers(answer) - set(range(1, len(evidence) + 1)):
        raise UserFacingError("답변의 근거 번호를 검증하지 못했습니다. 다시 질문해 주세요.")


def resolve_question(question: str, history: list[dict]) -> str:
    # Retrieval needs conversational context too, not only the generator.
    if re.search(r"그것|거기|그럼|그러면|이것|그 사람|그 사건|그 이론|그 영화", question):
        previous = [m["content"] for m in history if m["role"] == "user"][-2:]
        return "\n".join(previous + [question])
    return question


def build_graph(store, llm=None):
    def retrieve(state: RAGState):
        start = time.perf_counter()
        query = resolve_question(state["question"], state.get("history", []))
        evidence = store.retrieve(state["movie"]["movie_id"], query, state.get("method", "similarity"))
        return {"evidence": evidence, "timings": {"retrieve_seconds": round(time.perf_counter() - start, 3)}}

    def generate(state: RAGState):
        start = time.perf_counter()
        evidence = state["evidence"]
        if llm:
            payload = {"movie": state["movie"]["title"], "question": state["question"],
                       "task": state.get("task", "qa"),
                       "recent_conversation": state.get("history", [])[-6:],
                       "evidence": [{"number": n, "source_type": e.source_type,
                                     "title": e.title, "url": e.url, "text": e.text,
                                     "page_number": e.page_number, "publisher": e.publisher,
                                     "evidence_scope": e.evidence_scope}
                                    for n, e in enumerate(evidence, 1)]}
            answer = llm.text(ANSWER_SYSTEM, json.dumps(payload, ensure_ascii=False))
            validate_citations(answer, evidence)
            mode = "gms"
        else:
            # These are verbatim source excerpts, not a generated answer or a fact verdict.
            parts = ["**로컬 검색 결과 · AI 답변 생성 전**",
                     "아래는 검색된 문서의 원문 발췌입니다. 질문에 대한 사실 판정이나 완성된 답변은 AI 연결 후 제공합니다."]
            for i, e in enumerate(evidence, 1):
                if i > 4:
                    break
                excerpt = e.text[:500].replace("\n", "\n> ")
                parts.append(f"**{e.title}** · {e.source_type} [근거 {i}]\n\n> {excerpt}" + ("…" if len(e.text) > 500 else ""))
            answer = "\n\n".join(parts)
            mode = "local_extract"
        return {"answer": answer, "used_sources": used_sources(answer, evidence), "generation_mode": mode,
                "timings": {**state.get("timings", {}), "generate_seconds": round(time.perf_counter() - start, 3)}}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
