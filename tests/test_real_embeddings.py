from __future__ import annotations

import numpy as np

from scene_source.config import load_settings
from scene_source.embeddings import LocalEmbeddings


def test_long_korean_tail_changes_real_semantic_vector():
    embedding = LocalEmbeddings(load_settings())
    prefix = "이 문서는 도시의 역사와 경제 및 농업 활동을 설명하는 자료입니다. " * 15
    first = prefix + " 블랙홀의 사건의 지평선과 중력에 관한 관측 연구."
    second = prefix + " 피아노와 바이올린으로 연주하는 음악과 악기."
    windows = embedding.windows(first)
    assert len(windows) > 1
    assert any("지평선" in w for w in windows[1:])
    a, b = embedding.documents([first, second])
    assert len(a) == 384
    assert not np.allclose(a, b, atol=1e-5)
    assert abs(float(np.linalg.norm(a)) - 1) < 1e-5
