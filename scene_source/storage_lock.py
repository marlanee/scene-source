"""One embedded Chroma owner per data directory, shared by instances in this process."""
from __future__ import annotations
import atexit
import os
import threading
from pathlib import Path
from .models import UserFacingError

_HANDLES = {}
_LOCK = threading.Lock()


def claim_storage(path: Path):
    key=str(path.resolve())
    with _LOCK:
        if key in _HANDLES:return
        path.mkdir(parents=True,exist_ok=True)
        handle=(path/".process.lock").open("a+b")
        try:
            if os.name=="nt":
                import msvcrt
                handle.seek(0,2)
                if handle.tell()==0:handle.write(b"0");handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,PermissionError):
            handle.close()
            raise UserFacingError("같은 검색 데이터가 다른 앱 또는 검증 프로그램에서 사용 중입니다. 먼저 실행한 앱을 종료한 뒤 다시 실행해 주세요.") from None
        _HANDLES[key]=handle


def release_all():
    for handle in _HANDLES.values():
        try:handle.close()
        except OSError:pass
    _HANDLES.clear()


atexit.register(release_all)
