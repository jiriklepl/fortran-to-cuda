"""Tiny safe fixtures for guard cleanup; no allocation pressure or GPU work."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guard  # noqa: E402

HERE = Path(guard.__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return int(fields[19]), fields[0]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    grandchild = output / "detached.py"
    grandchild.write_text('''import json,os,signal,sys,time
from pathlib import Path
if sys.argv[2] == "ignore": signal.signal(signal.SIGTERM, signal.SIG_IGN)
fields=Path(f"/proc/{os.getpid()}/stat").read_text().rpartition(")")[2].split()
Path(sys.argv[1]).write_text(json.dumps({"pid":os.getpid(),"sid":os.getsid(0),"start":int(fields[19])}))
time.sleep(60)
''')
    parent = output / "parent.py"
    parent.write_text('''import subprocess,sys,time
from pathlib import Path
subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2],sys.argv[4]],start_new_session=True)
while not Path(sys.argv[2]).exists(): time.sleep(.01)
if sys.argv[3] != "orphan": time.sleep(60)
''')
    mocked = output / "simulated_memory.py"
    mocked.write_text('''import importlib.util,sys
from pathlib import Path
ready=Path(sys.argv.pop(1)); source=Path(sys.argv.pop(1))
spec=importlib.util.spec_from_file_location("isolated_guard_under_test",source)
guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
guard.available=lambda: 2*2**30 if ready.exists() else 4*2**30
guard.main()
''')
    rows = []
    for name in ("normal-parent-exit", "cancellation", "simulated-memory-stop", "term-resistant-descendant"):
        directory = output / name
        directory.mkdir()
        ready = directory / "descendant.json"
        record, log = directory / "guard.json", directory / "guard.log"
        memory = name in {"simulated-memory-stop", "term-resistant-descendant"}
        prefix = [sys.executable, str(mocked), str(ready), str(HERE / "guard.py")] if memory else [sys.executable, str(HERE / "guard.py")]
        argv = [*prefix, "--record", str(record), "--log", str(log), "--", sys.executable, str(parent),
                str(grandchild), str(ready), "orphan" if name == "normal-parent-exit" else "wait",
                "ignore" if name == "term-resistant-descendant" else "normal"]
        with (directory / "stdout.log").open("w") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 10
                while not ready.exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("tiny detached fixture did not start: " + name)
                    time.sleep(.01)
                child = json.loads(ready.read_text())
                if name == "cancellation":
                    process.send_signal(signal.SIGTERM)
                code = process.wait(timeout=20)
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=15)
        result = json.loads(record.read_text())
        current = identity(child["pid"])
        assert current is None or current[0] != child["start"] or current[1] == "Z", "detached descendant still running"
        assert child["sid"] == child["pid"] != result["process_group"], "fixture must leave original session"
        assert result["cleanup_complete"] and not result["remaining_descendants"]
        assert any(row["pid"] == child["pid"] and row["start_time_ticks"] == child["start"] for row in result["observed_descendants"])
        assert result["stopped_for_memory"] == memory
        if memory:
            assert result["headroom_bytes"] == 3 * 2**30 and result["minimum_available_bytes"] == 2 * 2**30
        if name == "cancellation":
            assert result["interrupted_signal"] == signal.SIGTERM
        if name == "term-resistant-descendant":
            assert any(row["pid"] == child["pid"] and row["signal"] == signal.SIGKILL for row in result["descendant_signals"])
        if name == "normal-parent-exit":
            assert code == 0
        else:
            assert code != 0
        rows.append({"name": name, "passed": True, "guard_exit_code": code, "record": str(record),
                     "record_sha256": digest(record), "memory_input_simulated": memory,
                     "detached_descendant_terminated": True})
    summary = {"schema_version": 1, "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
               "guard_sha256": digest(HERE / "guard.py"), "fixture_sha256": digest(Path(__file__)),
               "real_memory_pressure_created": False, "gpu_used": False, "cases": rows}
    write(output / "report.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()


def test_detached_descendants_are_cleaned_without_allocating_pressure(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [__file__, "--output", str(tmp_path / "evidence")])
    main()
