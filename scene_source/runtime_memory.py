"""Numeric server diagnostics only: never log questions, prompts or credentials."""
from pathlib import Path
import json
import sys


def memory_snapshot(stage):
    if not sys.platform.startswith("linux"):
        return
    values={"stage":stage}
    try:
        fields=dict(line.split(":",1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        for name in ("VmRSS","VmHWM"):
            values[name+"_MiB"]=round(int(fields[name].split()[0])/1024,1)
        for name in ("current","peak","max"):
            path=Path("/sys/fs/cgroup/memory."+name)
            if path.exists():
                value=path.read_text().strip()
                if value.isdecimal():values["cgroup_"+name+"_MiB"]=round(int(value)/1024**2,1)
    except (OSError,KeyError,ValueError):
        pass
    print("SCENE_MEMORY "+json.dumps(values),flush=True)
