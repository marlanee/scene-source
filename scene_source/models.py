from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Evidence:
    id: str
    title: str
    url: str
    text: str
    source_type: str
    movie_id: str
    page_id: str = ""
    collected_at: str = ""
    score: float = 0.0
    page_number: int = 0
    publisher: str = ""
    filename: str = ""
    retrieval_score: float = 0.0
    evidence_scope: str = "background"

    def to_dict(self):
        return asdict(self)


class UserFacingError(RuntimeError):
    """Only these deliberately sanitized messages may be shown in the UI."""
