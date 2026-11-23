import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory import memory_headroom  # noqa: E402


def fixture(tmp_path, *, legacy=False, mount_root="/", membership="/job/step"):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemAvailable: 20000000 kB\n")
    mount = tmp_path / "cgroup with spaces"
    mount.mkdir()
    escaped = str(mount).replace(" ", r"\040")
    fs = "cgroup none rw,memory" if legacy else "cgroup2 none rw"
    (proc / "self/mountinfo").write_text(f"1 0 0:1 {mount_root} {escaped} rw - {fs}\n")
    (proc / "self/cgroup").write_text(f"{'1:memory' if legacy else '0:'}:{membership}\n")
    return proc, mount


def limit(path, maximum, usage, *, legacy=False):
    path.mkdir(parents=True, exist_ok=True)
    (path / ("memory.limit_in_bytes" if legacy else "memory.max")).write_text(str(maximum))
    (path / ("memory.usage_in_bytes" if legacy else "memory.current")).write_text(str(usage))


def test_nested_job_limit_beats_unlimited_child_and_large_host(tmp_path):
    proc, mount = fixture(tmp_path)
    limit(mount, "max", 0)
    limit(mount / "job", 8 * 2**30, 6 * 2**30)
    limit(mount / "job/step", "max", 2**30)
    assert min(memory_headroom(proc).values()) == 2 * 2**30


def test_v1_namespaced_mount_and_exhausted_job(tmp_path):
    proc, mount = fixture(tmp_path, legacy=True, mount_root="/job", membership="/job/step")
    limit(mount, 8 * 2**30, 2**30, legacy=True)
    limit(mount / "step", 4 * 2**30, 5 * 2**30, legacy=True)
    assert min(memory_headroom(proc).values()) == 0


def test_host_remains_bound_with_unlimited_cgroup(tmp_path):
    proc, mount = fixture(tmp_path, membership="/")
    limit(mount, "max", 0)
    assert memory_headroom(proc) == {"host_available_bytes": 20000000 * 1024}
