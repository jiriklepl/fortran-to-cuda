"""Configured analysis is tied to original edits and source/include hashes."""

from __future__ import annotations

import os
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_inputs import SourceInputs
from compiler.ir import CompilationError
from compiler.tests.test_source_scopes import PROGRAM


def inputs(tmp_path, *, source=PROGRAM):
    original=tmp_path/"original.f90"
    original.write_text(source)
    # Use explicit fixtures to test provenance independently of an extractor.
    lines=source.splitlines()
    prepared=tmp_path/"configured.f90"
    mapping=[]
    selected=[]
    active=True
    for index,line in enumerate(lines,1):
        if line.startswith("#ifdef"):
            active=True
        elif line.startswith("#else"):
            active=False
        elif line.startswith("#endif"):
            active=True
        elif active:
            mapping.append(index)
            selected.append(line)
    prepared.write_text("\n".join(selected)+"\n")
    document={"schema_version":1,"source_inputs":{str(original):sha256(original.read_bytes()).hexdigest()},
              "preserves_source_order":True,"configuration":{"defines":["DOUBLE_PRECISION"]},"dependencies":{},
              "entries":[{"source":str(original),"path":str(prepared),"sha256":sha256(prepared.read_bytes()).hexdigest(),
                          "line_map":mapping}]}
    return original,prepared,document


def test_included_statement_does_not_use_prepared_line_for_original_edits(tmp_path):
    original,_prepared,document=inputs(tmp_path)
    index=next(index for index,line in enumerate(PROGRAM.splitlines()) if line=="call transform(b)")
    document["entries"][0]["line_map"][index]=None
    tree=SourceInputs([original],document).parse(original)
    from fparser.two.utils import walk

    call=next(n for n in walk(tree) if type(n).__name__=="Call_Stmt" and str(n.items[0])=="transform")
    assert call.item.fort_original_span is None


def test_dependency_change_and_bad_line_order_are_rejected(tmp_path):
    original,_prepared,document=inputs(tmp_path)
    dependency=tmp_path/"constants.inc"
    dependency.write_text("integer,parameter::wp=8\n")
    document["dependencies"]={str(dependency):sha256(dependency.read_bytes()).hexdigest()}
    source=SourceInputs([original],document)
    dependency.write_text("integer,parameter::wp=4\n")
    with pytest.raises(CompilationError,match="source changed"):
        source.verify()
    with pytest.raises(CompilationError,match="dependency hash differs"):
        SourceInputs([original],document)
    document["dependencies"]={}
    document["entries"][0]["line_map"][0:2]=[2,1]
    with pytest.raises(CompilationError,match="source order"):
        SourceInputs([original],document)


def test_transitive_public_precision_import_and_private_boundary(tmp_path):
    source="""module kinds
integer,parameter::wp=8
end module
module parameters
use kinds,only:precision=>wp
end module
module helpers
use parameters
contains
subroutine prepare(a)
real(precision),intent(inout)::a(:)
a=2*a
end subroutine
end module
"""
    original,_prepared,document=inputs(tmp_path,source=source)
    analysis=SourceEffects([original],analysis_sources=document)
    assert analysis.routines["helpers::prepare"].scope.bindings["a"].kind==8
    original,_prepared,document=inputs(tmp_path,source=source.replace("use kinds,only:precision=>wp","use kinds,only:precision=>wp\nprivate::precision"))
    analysis=SourceEffects([original],analysis_sources=document)
    assert analysis.routines["helpers::prepare"].scope.bindings["a"].kind is None


def test_output_initialization_before_payload_read_does_not_report_first_write_diagnostic(tmp_path):
    source=PROGRAM.replace("real(8),intent(inout)::b(:)\nb=3*b", "real(8),intent(out)::b(:)\nb=2\nb=3*b")
    original,_prepared,document=inputs(tmp_path,source=source)
    summary=SourceEffects([original],analysis_sources=document).summarize("original::transform")
    assert summary["complete"]
    assert not summary["definition_diagnostics"]


@pytest.mark.native
def test_configured_native_program_builds_with_original_cpp_and_bounds(tmp_path):
    if not shutil.which("gfortran"):
        pytest.skip("GNU Fortran unavailable")
    source=PROGRAM.replace("implicit none","""implicit none
#ifdef DOUBLE_PRECISION
integer,parameter::wp=8
#else
integer,parameter::wp=4
#endif""").replace("real(8)","real(wp)")
    original,_prepared,document=inputs(tmp_path,source=source)
    effects=SourceEffects([original],analysis_sources=document).report("step")
    assert effects["complete"]
    from compiler.tests.test_source_scopes import DRIVER

    driver=tmp_path/"driver.f90"
    driver.write_text(DRIVER)
    binary=tmp_path/"check"
    response=subprocess.run(["gfortran","-cpp","-DDOUBLE_PRECISION","-fcheck=all",str(original),str(driver),
                             "-o",str(binary)],cwd=tmp_path,capture_output=True,text=True,check=False,timeout=30)
    assert response.returncode==0,response.stdout+response.stderr
    response=subprocess.run([str(binary)],cwd=tmp_path,env={**os.environ,"OMP_NUM_THREADS":"4"},
                            capture_output=True,text=True,check=False,timeout=30)
    assert response.returncode==0,response.stdout+response.stderr
