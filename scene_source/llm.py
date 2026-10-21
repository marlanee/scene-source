from __future__ import annotations

import json
import os
import re

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from .models import UserFacingError


class Keywords(BaseModel):
    keywords: list[str] = Field(max_length=8)


class SelectedDocuments(BaseModel):
    selected_ids: list[str] = Field(max_length=10)
    reasons: dict[str, str] = Field(default_factory=dict)


class Filters(BaseModel):
    min_year: int | None = None
    max_year: int | None = None
    min_runtime: int | None = None
    max_runtime: int | None = None
    genres: list[str] = Field(default_factory=list)
    language: str | None = None
    director: str | None = None
    semantic_query: str | None = ""


class MoviePick(BaseModel):
    movie_id: str
    reason: str


class MoviePicks(BaseModel):
    items: list[MoviePick] = Field(max_length=6)


def parse_json(content: str, schema: type[BaseModel]):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    return schema.model_validate(json.loads(text))


class GMSClient:
    def __init__(self, before_request=None):
        key = os.getenv("GMS_KEY", "").strip()
        if not key:
            raise UserFacingError("GMS 키가 없습니다. .env에 GMS_KEY를 설정한 뒤 앱을 다시 시작해 주세요.")
        self.client = ChatOpenAI(model=os.getenv("GMS_MODEL", "gpt-5.4-nano"), api_key=key,
            base_url=os.getenv("GMS_BASE_URL", "https://gms.ssafy.io/gmsapi/api.openai.com/v1/"),
            timeout=60, max_retries=2, use_responses_api=False)
        self.before_request = before_request
        self.calls = 0
        self.usage = []

    def text(self, system: str, payload: str) -> str:
        before = getattr(self, "before_request", None)
        if before:
            before()
        try:
            result = self.client.invoke([("system", system), ("human", payload)])
            self.calls = getattr(self, "calls", 0) + 1
            self.usage = getattr(self, "usage", []) + [getattr(result, "usage_metadata", None)]
            if isinstance(result.content, str):
                return result.content
            return "\n".join(x.get("text", "") for x in result.content if isinstance(x, dict))
        except Exception as exc:
            # Do not expose SDK exceptions, prompts, headers or response bodies.
            status = getattr(exc, "status_code", None)
            if status == 401:
                raise UserFacingError("GMS 인증에 실패했습니다. 설정된 키를 확인해 주세요.") from None
            if status in {403, 404}:
                raise UserFacingError("설정된 GMS 모델 또는 API에 접근할 수 없습니다. 모델명과 권한을 확인해 주세요.") from None
            if status == 429:
                raise UserFacingError("GMS 사용 한도 또는 요청 속도 제한에 도달했습니다. 잠시 후 다시 시도하거나 사용 한도를 확인해 주세요.") from None
            raise UserFacingError("GMS 응답을 받지 못했습니다. 키·모델 접근 권한·네트워크·사용 한도를 확인해 주세요.") from None

    def structured(self, system: str, payload: str, schema: type[BaseModel]):
        system += "\n다음 JSON Schema에 맞는 객체 하나만 출력한다. 필드 타입과 null 허용 여부를 지킨다:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        for attempt in range(2):
            content = self.text(system, payload)
            try:
                return parse_json(content, schema)
            except (ValueError, ValidationError, TypeError) as exc:
                if attempt:
                    raise UserFacingError("모델의 구조화 응답을 해석하지 못했습니다. 다시 시도해 주세요.") from None
                system += "\n앞선 출력 형식 검증이 실패했다. 지정된 JSON 구조와 필드 타입을 정확히 지켜라."
                if isinstance(exc, ValidationError):
                    system += "\n오류 필드/유형: " + json.dumps([{"field": e["loc"], "type": e["type"]} for e in exc.errors()], ensure_ascii=False)
