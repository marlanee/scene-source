from __future__ import annotations

import argparse
import sys

from scene_source.config import load_settings
from scene_source.gms_verification import verify_gms
from scene_source.models import UserFacingError

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="키 수령 후에만 실행. 실제 GMS API 사용량이 발생합니다.")
    parser.add_argument("--compare", action="store_true", help="대표 질문 6개의 RAG 전후 비교")
    parser.add_argument("--run-api", action="store_true", required=True, help="실제 API 호출을 명시적으로 활성화")
    args = parser.parse_args()
    try:
        verify_gms(load_settings(), compare=args.compare)
    except UserFacingError as exc:
        print(str(exc))
        sys.exit(1)
