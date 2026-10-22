"""Whole-procedure extraction remains independent of compiler implementation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from elmm_pipeline.local_adapter import (
    LocalAdapterError,
    apply_scope_sources,
    transform_scopes,
)
from elmm_pipeline.scope_extraction import (
    export_analysis_sources,
    export_numerical_sources,
)
from elmm_pipeline.structural import Project
from test_gpu_calls import SOURCE

ROOT=Path(__file__).resolve().parents[2]
FACT={"storage":"stable","initialized":"whole","escapes":False,"allocation_changes":False}


def test_independent_pipeline_exports_leaves_without_grouping_decisions(tmp_path):
    source=tmp_path/"subject.f90"
    source.write_text(SOURCE)
    project=Project(tmp_path,[])
    report=export_numerical_sources(project,"subject::advance",tmp_path/"numerical",source_files=[source])
    package=json.loads(Path(report["package"]).read_text())
    assert report["entries"]==2
    assert not report["rejected"]
    assert {item["procedure"] for item in package["entries"]}=={"subject::prepare","subject::finish"}
    assert all(item["normalization"]=="whole_storage_rebased_v1" for item in package["entries"])
    compiler=tmp_path/"independent-compiler"
    shutil.copytree(ROOT/"compiler",compiler/"compiler",ignore=shutil.ignore_patterns("__pycache__"))
    metadata=transform_scopes(source,compiler,Path(sys.executable),tmp_path/"generated",entry="subject::advance",
                              capture_facts={"schema_version":1,"participation":"serial",
                                             "captures":{"argument::a":FACT,"argument::b":FACT}},
                              numerical_sources=Path(report["package"]))
    manifest=metadata["scope_manifest"]
    assert manifest["scope_count"]==1,manifest["boundaries"]
    assert manifest["scopes"][0]["gpu_leaves"]==["subject::finish","subject::prepare"]
    assert "--numerical-sources" in metadata["command"]
    assert len(manifest["numerical_sources"])==2
    assert source.read_text()==SOURCE


def test_unsafe_leaf_stays_native_in_extraction_report(tmp_path):
    source=tmp_path/"subject.f90"
    source.write_text(SOURCE.replace("scale = coefficient","coefficient = 3"))
    report=export_numerical_sources(Project(tmp_path,[]),"advance",tmp_path/"numerical",source_files=[source])
    package=json.loads(Path(report["package"]).read_text())
    assert [item["procedure"] for item in package["entries"]]==["subject::finish"]
    assert any(item["procedure"]=="subject::prepare" for item in report["rejected"])


def test_configured_source_package_keeps_original_includes_and_cpp_branch(tmp_path):
    source=tmp_path/"subject.f90"
    source.write_text(SOURCE.replace("implicit none",'implicit none\n#include "precision.inc"').replace("real(8)","real(wp)"))
    include=tmp_path/"precision.inc"
    include.write_text("#ifdef DOUBLE_PRECISION\ninteger,parameter::wp=8\n#else\ninteger,parameter::wp=4\n#endif\n")
    project=Project(tmp_path,["DOUBLE_PRECISION"])
    numerical=export_numerical_sources(project,"advance",tmp_path/"numerical",source_files=[source])
    analysis=export_analysis_sources(project,tmp_path/"analysis",source_files=[source])
    compiler=tmp_path/"independent-compiler"
    shutil.copytree(ROOT/"compiler",compiler/"compiler",ignore=shutil.ignore_patterns("__pycache__"))
    metadata=transform_scopes(source,compiler,Path(sys.executable),tmp_path/"generated",entry="advance",
                              capture_facts={"schema_version":1,"participation":"serial",
                                             "captures":{"argument::a":FACT,"argument::b":FACT}},
                              numerical_sources=Path(numerical["package"]),analysis_sources=analysis)
    manifest=metadata["scope_manifest"]
    assert manifest["scope_count"]==1,manifest["boundaries"]
    assert str(include) in manifest["analysis_sources"]["dependencies"]
    assert manifest["analysis_sources"]["configuration"]["defines"]==["DOUBLE_PRECISION"]
    destination=tmp_path/"application-copy.f90"
    destination.write_bytes(source.read_bytes())
    apply_scope_sources(metadata,{str(source):destination})
    assert '#include "precision.inc"' in destination.read_text()
    assert "call fort_scope_owner_" in destination.read_text()
    destination.write_bytes(source.read_bytes())
    include.write_text(include.read_text().replace("wp=8","wp=4"))
    before=destination.read_bytes()
    with pytest.raises(LocalAdapterError,match="dependency changed"):
        apply_scope_sources(metadata,{str(source):destination})
    assert destination.read_bytes()==before


def test_empty_array_lower_bound_is_evaluated_in_original_dummy_coordinates(tmp_path):
    source=tmp_path/"subject.f90"
    source.write_text("""module subject
contains
subroutine prepare(a,b)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(:)
integer::i
if(lbound(a,1)==1) then
do i=1,size(b)
b(i)=7+real(i,8)
enddo
endif
end subroutine
subroutine finish(b)
real(8),intent(inout)::b(:)
b=2*b
end subroutine
subroutine advance(a,b)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
call prepare(a,b)
call finish(b)
end subroutine
end module
""")
    report=export_numerical_sources(Project(tmp_path,[]),"advance",tmp_path/"numerical",source_files=[source])
    assert report["entries"]==1,report # whole-array finish remains a native operation
    metadata=transform_scopes(source,ROOT,Path(sys.executable),tmp_path/"generated",entry="advance",
                              capture_facts={"schema_version":1,"participation":"serial",
                                             "captures":{"argument::a":FACT,"argument::b":FACT}},
                              numerical_sources=Path(report["package"]))
    manifest=metadata["scope_manifest"]
    assert manifest["scope_count"]==1,manifest["boundaries"]
    generated=Path(metadata["artifact_root"])/manifest["sources"][str(source)]["replacement"]
    assert "lbound(a, 1)" in generated.read_text()
    # Verify extraction itself against the original Fortran for an empty A
    # and a nonempty B; hard-coded declared lower bounds would miss this branch.
    item=next(item for item in json.loads(Path(report["package"]).read_text())["entries"]
              if item["procedure"]=="subject::prepare")
    procedure=item["entry"].split("::")
    arguments=[]
    for parameter in item["parameters"]:
        root=parameter["resource"].split("::")[-1]
        dimension=parameter.get("lower_bound_dimension")
        arguments.append(f"lbound({root},{dimension})" if dimension else root)
    driver=tmp_path/"driver.f90"
    driver.write_text(f"""program test
use subject
use {procedure[0]},only:normalized=>{procedure[1]}
real(8)::a(-2:-3),b(4),expected(4)
b=-99
expected=-99
call prepare(a,b)
call checked(a,expected)
if(any(b/=expected)) error stop 'empty lower bound'
contains
subroutine checked(a,b)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(:)
call normalized({','.join(arguments)})
end subroutine
end program
""")
    command=["gfortran","-fcheck=all",str(source),item["path"],str(driver),"-o",str(tmp_path/"check")]
    response=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,timeout=30,check=False)
    assert response.returncode==0,response.stdout+response.stderr
    response=subprocess.run([str(tmp_path/"check")],cwd=tmp_path,capture_output=True,text=True,timeout=30,check=False,
                            env={**os.environ,"OMP_NUM_THREADS":"4"})
    assert response.returncode==0,response.stdout+response.stderr
