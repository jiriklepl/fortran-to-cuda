"""The memory a run may take: what the host has available, within the limits of the cgroups it runs in."""
from __future__ import annotations
import re
from pathlib import Path


def memory_headroom(proc=Path("/proc")):
    """Respect container/job cgroup limits as well as host available memory."""
    host = int(next(line.split()[1] for line in (proc / "meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:"))) * 1024
    budgets = {"host_available_bytes": host}
    memberships = [line.split(":", 2) for line in (proc / "self/cgroup").read_text().splitlines()]
    for line in (proc / "self/mountinfo").read_text().splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields, filesystem = before.split(), after.split()
        unified = filesystem[0] == "cgroup2"
        legacy = filesystem[0] == "cgroup" and "memory" in filesystem[2].split(",")
        if not (unified or legacy):
            continue
        unescape = lambda text: re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), text)
        mount = Path(unescape(fields[4])).resolve()
        for _, controllers, group in memberships:
            if not ((unified and not controllers) or (legacy and "memory" in controllers.split(","))):
                continue
            try:
                relative = Path(group).relative_to(unescape(fields[3]))
            except ValueError:
                continue
            directory = (mount / relative).resolve()
            if not directory.is_relative_to(mount):
                continue
            while True:
                limit = directory / ("memory.max" if unified else "memory.limit_in_bytes")
                usage = directory / ("memory.current" if unified else "memory.usage_in_bytes")
                if limit.is_file() and usage.is_file():
                    value = limit.read_text().strip()
                    if value != "max":
                        budgets[str(directory)] = max(0, int(value) - int(usage.read_text()))
                if directory == mount:
                    break
                directory = directory.parent
    return budgets
