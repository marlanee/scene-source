from __future__ import annotations

from streamlit.testing.v1 import AppTest

from scene_source.config import ROOT


def test_local_ui_question_history_discovery_and_movie_switch(monkeypatch):
    monkeypatch.setattr("scene_source.config.gms_available", lambda: False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    assert not app.error
    assert app.title[0].value == "영화 속 이야기, 근거로 한 걸음 더."
    assert app.selectbox(key="selected_movie_id").value == "157336"
    app.text_input[0].set_value("웜홀은 실제로 관측되었나요?")
    next(b for b in app.button if b.label == "근거 찾아보기").click().run()
    assert not app.exception
    assert not app.error
    threads = app.session_state["threads"]
    assert len(threads["157336:로컬 원문 검색"]) == 2
    assert "원문 발췌" in threads["157336:로컬 원문 검색"][1]["content"]
    assert threads["157336:로컬 원문 검색"][1]["sources"]
    app.selectbox(key="selected_movie_id").set_value("872585").run()
    assert not app.exception
    assert app.session_state["threads"]["872585:로컬 원문 검색"] == []
    assert len(app.session_state["threads"]["157336:로컬 원문 검색"]) == 2
    app.text_input[1].set_value("2010년 이후 한국 영화 중 130분 이하인 스릴러")
    # Form submit buttons have deterministic labels even after result cards are added.
    next(b for b in app.button if b.label == "영화 찾기").click().run()
    assert not app.exception
    assert not app.error
    result = app.session_state["discovery_result"]
    assert result["items"]
    picked = str(result["items"][0]["movie"]["movie_id"])
    app.button(key=f"pick_{picked}").click().run()
    assert app.selectbox(key="selected_movie_id").value == picked
    assert not app.exception


def test_missing_primary_document_stops_with_clear_message(monkeypatch):
    monkeypatch.setattr("scene_source.config.gms_available", lambda: False)
    from scene_source.models import UserFacingError
    def missing_primary(*_args, **_kwargs):
        raise UserFacingError("배포 자료에 해당 영화의 본문 문서가 없습니다. 실시간 수집 또는 다른 영화를 선택해 주세요.")
    monkeypatch.setattr("scene_source.sources.DocumentCollector.collect", missing_primary)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    app.selectbox(key="selected_movie_id").set_value("460465").run()
    app.text_input[0].set_value("이 영화의 배경을 설명해 주세요.")
    next(b for b in app.button if b.label == "근거 찾아보기").click().run()
    assert not app.exception
    assert any("영화의 본문 문서가 없습니다" in e.value for e in app.error)
    assert app.session_state["threads"]["460465:로컬 원문 검색"] == []
