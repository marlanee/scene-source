from __future__ import annotations

import html
import time
from dataclasses import replace

import streamlit as st

from scene_source.config import ROOT, gms_available, load_settings
from scene_source.data import load_movies
from scene_source.discovery import discover
from scene_source.embeddings import LocalEmbeddings
from scene_source.llm import GMSClient
from scene_source.models import UserFacingError
from scene_source.pipeline import build_graph
from scene_source.sources import DocumentCollector
from scene_source.store import VectorStore

st.set_page_config(page_title="Scene & Source · 영화와 근거", page_icon="🎬", layout="wide")


@st.cache_data
def movie_catalog():
    return load_movies(ROOT)


@st.cache_resource
def embedding_resource(model, threads, batch_size):
    return LocalEmbeddings(replace(load_settings(), embedding_model=model, threads=threads, batch_size=batch_size))


@st.cache_resource
def store_resource(settings):
    return VectorStore(settings, embedding_resource(settings.embedding_model, settings.threads, settings.batch_size))


def select_movie(movie_id):
    st.session_state.selected_movie_id = str(movie_id)
    st.session_state.discovery_result = None


def render_sources(sources):
    if not sources:
        st.caption("이 답변에는 인용된 근거가 없습니다.")
        return
    with st.expander(f"사용한 근거 {len(sources)}개 · 원문과 발췌 확인"):
        for source in sources:
            st.markdown(f"**[근거 {source['number']}] {source['title']}**")
            st.caption(source["source_type"] + (f" · 수집 {source['collected_at']}" if source.get("collected_at") else ""))
            st.link_button("원문 열기 ↗", source["url"])
            st.text(source["text"][:1500])
            st.divider()


def run_question(movie, question, task, settings, store, llm, mode):
    if len(question) > 2000:
        st.warning("질문은 2,000자 이내로 입력해 주세요.")
        return
    thread_key = f"{movie['movie_id']}:{mode}"
    threads = st.session_state.setdefault("threads", {})
    messages = threads.setdefault(thread_key, [])
    history = [{"role": m["role"], "content": m["content"]} for m in messages[-6:]]
    start = time.perf_counter()
    try:
        with st.status("질문에 필요한 문서를 준비하고 있습니다.", expanded=True) as status:
            collector = DocumentCollector(settings, store, llm)
            collection = collector.collect(movie, question, progress=st.write)
            st.write("저장된 수집 결과를 재사용했습니다." if collection["cache_hit"] else f"문서 {len(collection['selected'])}개를 준비했습니다.")
            status.update(label="근거 검색과 답변 처리를 진행하고 있습니다.")
            result = build_graph(store, llm).invoke({"movie": movie, "question": question,
                "history": history, "task": task, "method": settings.method})
            status.update(label="처리가 완료되었습니다.", state="complete", expanded=False)
        elapsed = round(time.perf_counter() - start, 2)
        messages.extend([{"role": "user", "content": question},
                         {"role": "assistant", "content": result["answer"],
                          "sources": result["used_sources"], "elapsed": elapsed,
                          "mode": result["generation_mode"]}])
        st.session_state.last_diagnostics = {"collection": collection,
            "timings": {"total_seconds": elapsed, **result["timings"]},
            "retrieved": [{"id": e.id, "title": e.title, "source_type": e.source_type,
                           "score": round(e.score, 4)} for e in result["evidence"]]}
        st.rerun()
    except UserFacingError as exc:
        st.error(str(exc))
    except Exception:
        st.error("처리 중 문제가 발생했습니다. 데이터 준비 상태와 네트워크를 확인한 뒤 다시 시도해 주세요.")


def main():
    settings = load_settings()
    movies = movie_catalog()
    catalog = {str(m["movie_id"]): m for m in movies}
    has_key = gms_available()
    with st.sidebar:
        st.markdown("### 🎬 Scene & Source")
        st.caption("작품의 이야기와 현실의 지식을\n근거로 구분합니다.")
        st.divider()
        if "selected_movie_id" not in st.session_state:
            st.session_state.selected_movie_id = "157336"
        movie_id = st.selectbox("질문할 영화", sorted(catalog, key=lambda i: catalog[i]["title"]),
            format_func=lambda i: f"{catalog[i]['title']} ({catalog[i]['year']})", key="selected_movie_id")
        if has_key:
            mode = st.radio("응답 방식", ["GMS 근거 답변", "로컬 원문 검색"], key="response_mode")
        else:
            mode = "로컬 원문 검색"
            st.info("GMS 키 대기 중 · 로컬 원문 검색 가능")
            st.caption("키를 받으면 .env의 GMS_KEY를 채우고 앱을 다시 시작하세요.")
        source_mode = st.selectbox("외부 문서 출처", ["bundle", "live"],
            format_func=lambda x: "로컬 문서 · 배포 자료와 보완 문서" if x == "bundle" else "실시간 위키백과")
        with st.expander("검색 설정"):
            method = st.selectbox("검색 방식", ["similarity", "mmr"], format_func=lambda x: "유사도" if x == "similarity" else "MMR · 다양성")
            top_k = st.slider("외부 근거 수 (Top-k)", 2, 10, settings.top_k)
            st.caption(f"청크 {settings.chunk_size}자 · 중복 {settings.chunk_overlap}자\nconfig.toml에서 변경할 수 있습니다.")
        settings = replace(settings, source_mode=source_mode, method=method, top_k=top_k,
                           fetch_k=max(settings.fetch_k, top_k))
        store = store_resource(settings)
        st.caption(f"영화 색인 {store.meta.count():,} / {len(movies):,}편\n선택 영화의 외부 문서 청크 {store.document_count(movie_id):,}개")
        if st.button("이 영화의 대화 지우기", width="stretch"):
            st.session_state.setdefault("threads", {})[f"{movie_id}:{mode}"] = []
            st.rerun()
    if store.meta.count() < len(movies):
        st.warning("영화 색인이 없거나 준비가 덜 되었습니다. 먼저 전체 데이터를 준비해 주세요.")
        if st.button("영화 964편 색인하기"):
            bar = st.progress(0, text="로컬 모델로 영화 데이터를 색인하고 있습니다.")
            store.index_movies(movies, progress=lambda n, total: bar.progress(n / max(total, 1)))
            st.rerun()
        return
    llm = GMSClient() if mode == "GMS 근거 답변" else None
    movie = catalog[movie_id]
    st.caption("MOVIE KNOWLEDGE, WITH EVIDENCE")
    st.title("영화 속 이야기, 근거로 한 걸음 더.")
    st.write("궁금한 작품을 고르고 질문하세요. 작품의 설정과 외부 자료를 함께 찾아 출처를 보여 드립니다.")
    tabs = st.tabs(["💬 영화와 질문", "🔎 영화 탐색", "🧭 검증 현황"])
    with tabs[0]:
        left, right = st.columns([3, 1])
        with left:
            st.subheader(f"{movie['title']} · {movie['year']}")
            st.caption(" · ".join(movie["genres"]) + "  /  " + ", ".join(movie.get("director", [])))
            st.write(movie["overview"])
        with right:
            st.metric("상영시간", f"{movie['runtime']}분")
            st.link_button("영화 원본 정보 ↗", movie["tmdb_url"])
        st.divider()
        task = st.radio("질문 유형", ["배경 해설", "사실 확인"], horizontal=True)
        if not has_key:
            st.caption("지금은 원문 검색을 검증하는 모드입니다. 설명 생성과 사실 판정은 GMS 연결 후 활성화됩니다.")
        messages = st.session_state.setdefault("threads", {}).setdefault(f"{movie_id}:{mode}", [])
        if not messages:
            st.info("예: 이 영화의 배경이 된 사건은 무엇인가요? / 이 설정을 현실에서도 확인할 수 있나요?")
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    render_sources(message.get("sources", []))
                    st.caption(f"처리 {message.get('elapsed', 0)}초")
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input("영화에 대해 질문하거나 확인할 주장을 입력하세요", max_chars=2000,
                                     placeholder="예: 웜홀은 실제로 관측된 천체인가요?")
            submitted = st.form_submit_button("근거 찾아보기" if not llm else "근거와 함께 답변받기", type="primary")
        if submitted and question.strip():
            run_question(movie, question.strip(), "factcheck" if task == "사실 확인" else "qa", settings, store, llm, mode)
    with tabs[1]:
        st.subheader("보고 싶은 영화의 조건을 알려 주세요.")
        st.caption("예: 2010년 이후 한국 영화 중 130분 이하인 스릴러 / 우주와 가족을 다루는 SF 영화")
        with st.form("discovery_form"):
            conditions = st.text_input("원하는 영화 조건", max_chars=1000)
            search = st.form_submit_button("영화 찾기", type="primary")
        if search and conditions.strip():
            try:
                with st.spinner("조건에 맞는 영화 후보를 찾고 있습니다."):
                    result = discover(conditions, movies, store, llm)
                    st.session_state.discovery_result = {**result, "query": conditions}
            except UserFacingError as exc:
                st.error(str(exc))
            except Exception:
                st.error("영화 탐색을 완료하지 못했습니다. 색인 상태를 확인해 주세요.")
        result = st.session_state.get("discovery_result")
        if result:
            if not result["verified"]:
                st.info("로컬 탐색 후보입니다. 명시적 연도·장르·시간 조건을 적용했으며, 주제·분위기 일치 여부는 GMS 연결 후 추가 검증합니다.")
            if not result["items"]:
                st.info("확인 가능한 조건에 맞는 영화가 없습니다. 조건을 줄여 보세요.")
            for start in range(0, len(result["items"]), 3):
                columns = st.columns(3)
                for col, item in zip(columns, result["items"][start:start + 3]):
                    m = item["movie"]
                    with col, st.container(border=True):
                        st.markdown(f"**{m['title']} ({m['year']})**")
                        st.caption(f"{m['runtime']}분 · " + ", ".join(m["genres"]))
                        st.write(item["reason"])
                        st.link_button("메타데이터 원문", m["tmdb_url"])
                        st.button("이 영화로 질문하기 →", key=f"pick_{m['movie_id']}",
                            on_click=select_movie, args=(m["movie_id"],))
    with tabs[2]:
        st.subheader("현재 연결과 검증 범위")
        a, b, c = st.columns(3)
        a.metric("영화 메타데이터", f"{store.meta.count():,}편")
        b.metric("외부 문서 청크", f"{store.docs.count():,}개")
        c.metric("GMS 연결", "키 설정됨 · 실호출 시 검증" if has_key else "키 발급 대기")
        st.write("로컬 다국어 임베딩 → ChromaDB 두 컬렉션 → LangGraph 검색·생성 노드")
        st.caption("배포 위키 문서는 2026년 9월 자료이며 보완 문서의 수집일은 각 근거에 표시됩니다. 최신 사실을 보장하지 않으며 문서의 주장과 현실의 정확성은 별도로 검토해야 합니다.")
        detail = st.session_state.get("last_diagnostics")
        if detail:
            collection = detail["collection"]
            st.write("최근 검색어:", " / ".join(collection["keywords"]))
            st.write("수집 캐시:", "재사용" if collection["cache_hit"] else "새 색인")
            st.dataframe(collection["selected"], width="stretch")
            with st.expander("후보 제외 목록과 검색 점수"):
                st.write(collection["rejected"] or "제외된 후보 없음")
                st.dataframe(detail["retrieved"], width="stretch")
                st.json(detail["timings"])
        else:
            st.info("질문을 한 번 실행하면 수집·검색 결과를 확인할 수 있습니다.")
        st.markdown("**남은 실검증:** GMS 검색어 추출 · 후보 선별 · 답변 생성 · 사실 확인 · 탐색 조건 검증 · RAG 전후 비교")
    st.divider()
    st.caption("자료: TMDB 기반 배포 영화 메타데이터 / 한국어 위키백과 배포 문서(CC BY-SA 4.0). 각 답변에서 실제 사용한 원문을 확인할 수 있습니다.")


try:
    main()
except UserFacingError as exc:
    st.error(str(exc))
except Exception:
    st.error("앱 준비를 완료하지 못했습니다. 가상환경 설치와 데이터·모델 준비 상태를 확인해 주세요.")
