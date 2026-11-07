"""A reached point read preserves a live replica despite an unrelated shape."""
import json
import os
import re
from functools import partial

import pytest

from compiler.tests.test_native_ancestor_runtime import build_native_ancestor
from compiler.tests.test_specification_entry_mapped import MAPPED_SOURCE
from compiler.tests.test_specification_entry_mapped_cuda import DRIVER


def source():
    text = MAPPED_SOURCE.replace("\ndo i=1,1\ntile(i)=n\nenddo", "")
    text = text.replace("integer::tile(1)=[8]", "integer,protected::tile(3)=[-701,8,-703]")
    text = text.replace("tile(1)", "tile(2)")
    text = text.replace("b(i)=2.0_8*a(i)", "b(i)=2.0_8*a(i)+real(tile(2),8)")
    text = text.replace("real(8)::scratch(tile(2))", "real(8)::scratch(tile(2))\ninteger::width")
    prefix, consumer = text.split("subroutine consumer(", 1)
    consumer = consumer.replace("do i=1,n", "width=tile(2)\ndo i=1,width", 1)
    text = prefix + "subroutine consumer(" + consumer
    return text.rsplit("end module", 1)[0] + """subroutine set_tile(n)
integer,intent(in)::n
tile(2)=n
end subroutine
end module
"""


def emit_fixture(original, *args, **kwargs):
    path, builder = original(*args, **kwargs)
    builder.facts["captures"]["ancestor_fixture::tile"] = {
        "storage": "stable", "initialized": "none", "escapes": False, "allocation_changes": False}
    return path, builder


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    from compiler.tests import test_native_ancestor_runtime as support

    driver = DRIVER.replace("tile(1)=n", "call set_tile(n)").replace("tile(1)", "tile(2)")
    driver = driver.replace("if(tile(2)/=n.or.observed/=0)",
                            "if(tile(2)/=n.or.tile(1)/=-701.or.tile(3)/=-703.or.observed/=0)")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(support, "fixture", partial(emit_fixture, support.fixture))
        result = build_native_ancestor(tmp_path_factory.mktemp("reached_native_shape"),
                                      source=source(), driver_source=driver)
    validate_package(result[0])
    return result


def validate_package(build):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    original = (build / "scope-manifest.json").read_bytes()
    manifest = json.loads(original)
    validate_checked_execution_plans(manifest)
    _scope_package_build({"artifact_root": str(build), "scope_manifest": manifest})
    assert manifest == json.loads(original)
    assert (build / "scope-manifest.json").read_bytes() == original


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["gpu", "hidden"])
def test_native_point_between_gpu_children_keeps_partial_metadata_and_fields(binaries, hidden):
    from compiler.tests.test_source_scopes import run

    build, candidate, native = binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    mode = "hidden" if hidden else "gpu"
    results = []
    for name, binary in (("native", native), ("candidate", candidate)):
        result = run([str(binary), "0"], cwd=build, env=environment)
        for suffix in ("stdout", "stderr"):
            (build / f"{mode}-{name}.{suffix}").write_text(getattr(result, suffix))
        results.append(result)
    assert results[0].stdout == results[1].stdout
    cases = re.findall(r"MAPPED_ENTRY_BEGIN (\d+) (\d+)\n(.*?)MAPPED_ENTRY_END \1 \2", results[1].stderr, re.S)
    assert len(cases) == 4
    ledger = []
    for size, repeat, trace in cases:
        lines = trace.splitlines()
        launches = [i for i, line in enumerate(lines) if line.startswith("FORT_SCOPED launch ")]
        copies = [dict(re.findall(r"(\w+)=([^\s]+)", line), event=line.split()[1]) for line in lines
                  if line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download "))]
        n = int(size)
        assert len(launches) == (2 if n and not hidden else 0), trace
        if hidden:
            assert not copies
            continue
        assert sum(int(row['bytes']) for row in copies if row['event']=='upload') == (8*n+4 if n else 0), trace
        assert sum(int(row['bytes']) for row in copies if row['event']=='download') == 16*n, trace
        if n:
            assert not any(line.startswith("FORT_SCOPED download ") for line in lines[:launches[-1]]), trace
        ledger.append({'n':n,'repeat':int(repeat),'launches':len(launches),'copies':copies})
    (build / f"{mode}-ledger.json").write_text(json.dumps(ledger,indent=2)+'\n')
