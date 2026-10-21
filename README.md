# Scene & Source

어두운 영화관 스타일의 근거 기반 영화 지식 서비스입니다. 영화 964편을 탐색하고 위키백과·NASA PDF를 검색해 GMS가 답변합니다.

## 기능

- HTML/CSS/JS + FastAPI, retrieve/generate LangGraph
- 다국어 의미 검색 + 선택적 BM25 보완, MMR 비교
- 작품 소개와 현실 배경의 근거 범위 구분, 인용 원문과 PDF 쪽 링크
- 실제 RAG 전후 비교 기록, 로컬 버전의 사람 채점·내보내기
- 공개 체험의 방문자별 작업 격리·GMS 호출 상한·관리 기능 제한

## 로컬 실행

Python 3.12에서 setup.ps1 실행 후 .env.example을 참고해 .env를 설정하고 run.ps1을 실행합니다. 키는 Git에 넣지 않습니다. 공개 카탈로그에는 원문 줄거리가 없고 위키백과는 실시간 수집하므로 인터넷이 필요합니다.

## 웹 배포

Render에서 이 저장소를 Docker Web Service로 연결합니다. /healthz로 상태를 확인합니다. GMS_KEY는 서버의 비밀 설정에만 넣고, GMS_MODEL, WIKI_CONTACT, SCENE_PUBLIC=1을 설정합니다. 앱 주소는 Render의 RENDER_EXTERNAL_URL 또는 APP_PUBLIC_URL을 사용합니다.

로컬 모델은 기본 MiniLM, 공개 서버는 같은 계열 INT8 모델·1스레드·작은 배치로 실행합니다. 빌드에서 공개 데이터 색인을 준비하고 런타임에 복사합니다. 한 프로세스·한 worker로 실행합니다. 실제 Render 무료 512MB 서버는 모델 적재 도중 메모리를 초과해 종료됐습니다. 같은 Render 이미지 빌드에서는 모델 적재 최대 약 674MiB가 기록됐으므로 512MB 요금제로 전체 검색 기능을 운영할 수 있다고 주장하지 않습니다. 서버 메모리와 비용은 운영자가 결정해야 합니다. render.yaml의 free 설정은 유료 결제를 자동 신청하지 않기 위해 유지했습니다.

무료 서버의 임시 파일은 재시작 시 사라질 수 있습니다. 영구 저장이 필요하면 /app/data에 영구 볼륨을 연결하세요. 공개 서버는 PDF 업로드/삭제와 평가 실행을 차단하고 결과 열람만 제공합니다. GMS 호출 상한 기본 100회/UTC 일, 접속 환경별 AI 요청은 시간당 3회입니다. 상한은 SDK 논리 호출 기준이며 공급자 내부 재시도는 별도일 수 있습니다.

## 데이터·출처

공개용 카탈로그는 제목·연도·감독·출연·장르·상영시간 등 사실 필드로 구성했습니다. 교육 팩 원본 줄거리·평점·원본 ZIP·요구사항 PDF·강의 코드는 포함하지 않습니다. 각 영화의 TMDB 링크를 유지하며 TMDB의 보증·인증을 받은 서비스는 아닙니다.

위키백과 파생 본문은 CC BY-SA 4.0을 따르며 원문 링크·수집일을 표시합니다. NASA 자료는 공식 URL에서 가져오고 NASA/Goddard 출처와 쪽 번호를 보존합니다. 임베딩 모델은 공개 배포본의 Apache-2.0 라이선스를 확인할 수 있습니다. 상세 출처와 개인정보 흐름은 웹의 '출처·이용 안내'에서 확인하세요.

## 검증

pytest tests/test_core.py tests/test_gms_contract.py tests/test_advanced.py tests/test_finish.py는 API 키 없이 계약을 검사합니다. verify_complete.py --run-api와 verify_gms.py --run-api --compare는 실제 GMS 사용량이 발생합니다. 사람의 셀프/상호 점검 기록은 가상으로 채우지 않습니다.
