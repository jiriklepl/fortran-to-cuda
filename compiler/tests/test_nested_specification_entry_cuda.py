"""Nested bound publication and the ordinary ABI execute their source once."""
import os
import subprocess

import pytest

from compiler.tests.test_native_ancestor_runtime import DRIVER, build_native_ancestor
from compiler.tests.test_specification_entry import NESTED_ENTRY_SOURCE


@pytest.fixture(scope="module")
def nested_binaries(tmp_path_factory):
    driver = DRIVER.replace("logical::flag", "logical::flag\ncharacter(8)::entry_mode").replace(
        "call advance(a,b,n,flag)", """call get_command_argument(2,entry_mode)
if(entry_mode=='ordinary')then
 call wrapper(a,b,n,flag)
else
 call advance(a,b,n,flag)
endif""")
    return build_native_ancestor(tmp_path_factory.mktemp("nested_specification"),
                                 source=NESTED_ENTRY_SOURCE, driver_source=driver)


@pytest.mark.cuda
@pytest.mark.parametrize("entry", ["checked", "ordinary"])
@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("cut", [False, True])
def test_nested_entry_and_ordinary_abi_preserve_fields_without_replay(nested_binaries, entry, hidden, cut):
    build, candidate, native = nested_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
                   "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    mode = f"{entry}-hidden{int(hidden)}-cut{int(cut)}"
    for name, binary in (("native", native), ("candidate", candidate)):
        result = subprocess.run([str(binary), str(int(cut)), entry], cwd=build, env=environment,
                                capture_output=True, text=True, timeout=30)
        for suffix in ("stdout", "stderr"):
            (build / f"{mode}-{name}.{suffix}").write_text(getattr(result, suffix))
        assert result.returncode == 0, result.stdout + result.stderr
        results.append(result)
    reference, actual = results
    assert actual.stdout == reference.stdout
    assert "NATIVE_ANCESTOR_FULL_FIELDS_OK" in actual.stdout
    lines = actual.stderr.splitlines()
    launches = sum(line.startswith("FORT_SCOPED launch ") for line in lines)
    if hidden or entry == "ordinary":
        assert launches == 0, actual.stderr
        assert not any(line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download ")) for line in lines)
    else:
        assert launches == 2, actual.stderr
