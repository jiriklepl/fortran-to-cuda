"""Configured analysis is tied to original edits and source/include hashes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_inputs import SourceInputs
from compiler.ir import CompilationError
from compiler.tests.test_source_scopes import FACT, PROGRAM, ROOT, run


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


def generate(tmp_path, *, source=PROGRAM):
    original,prepared,document=inputs(tmp_path,source=source)
    facts={"schema_version":1,"participation":"serial","sources":document["source_inputs"],
           "captures":{root:FACT for root in ("argument::a","argument::b","argument::out")}}
    facts_path=tmp_path/"captures.json"
    facts_path.write_text(json.dumps(facts))
    package=tmp_path/"analysis.json"
    package.write_text(json.dumps(document))
    output=tmp_path/"output"
    response=run([sys.executable,"-m","compiler.tests.support.reference_scopes","--input",str(original),"--kernel","step","--form-scopes",
                  "--scope-facts",str(facts_path),"--analysis-sources",str(package),"--memory-model","scoped",
                  "--gpu-policy","sections","--output-dir",str(output),"--json"],cwd=ROOT)
    return original,prepared,output,json.loads(response.stdout)["scopes"]


def test_included_statement_does_not_use_prepared_line_for_original_edits(tmp_path):
    original,_prepared,document=inputs(tmp_path)
    index=next(index for index,line in enumerate(PROGRAM.splitlines()) if line=="call transform(b)")
    document["entries"][0]["line_map"][index]=None
    tree=SourceInputs([original],document).parse(original)
    from fparser.two.utils import walk

    call=next(n for n in walk(tree) if type(n).__name__=="Call_Stmt" and str(n.items[0])=="transform")
    assert call.item.fort_original_span is None


def configured_native_owner(tmp_path):
    """An included native correction, bounded by original joined directives."""
    from compiler.tests.test_lexical_source_owner import SOURCE

    source = SOURCE.replace('if(escape) then\ncall opaque(b,n)\nendif',
        '!$omp parallel do private(i)\n#include "correction.inc"\n!$omp end parallel do')
    original = tmp_path/'owner.f90'
    original.write_text(source)
    include = tmp_path/'correction.inc'
    include.write_text('do i=-2,n-3\nb(i)=b(i)+sum(a)\nenddo\n')
    prepared = tmp_path/'configured.f90'
    lines, mapping = [], []
    for number, line in enumerate(source.splitlines(), 1):
        if line.startswith('#include'):
            included = include.read_text().splitlines()
            lines.extend(included)
            mapping.extend([None]*len(included))
        else:
            lines.append(line)
            mapping.append(number)
    prepared.write_text('\n'.join(lines)+'\n')
    document = {'schema_version': 1, 'source_inputs': {str(original): sha256(original.read_bytes()).hexdigest()},
                'preserves_source_order': True, 'configuration': {'defines': []},
                'dependencies': {str(include): sha256(include.read_bytes()).hexdigest()},
                'entries': [{'source': str(original), 'path': str(prepared),
                             'sha256': sha256(prepared.read_bytes()).hexdigest(), 'line_map': mapping}]}
    facts = {'schema_version': 1, 'participation': 'serial', 'sources': document['source_inputs'],
             'captures': {'argument::'+name: FACT for name in ('a','b','out')}}
    return original, document, facts


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


@pytest.fixture(scope="module")
def compiled_configured_scope(tmp_path_factory):
    nvcc=shutil.which("nvcc")
    host=shutil.which("g++-14") or shutil.which("g++")
    fortran=shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA/Fortran toolchain unavailable")
    directory=tmp_path_factory.mktemp("configured_source_scope")
    source=PROGRAM.replace("implicit none","""implicit none
#ifdef DOUBLE_PRECISION
integer,parameter::wp=8
#else
integer,parameter::wp=4
#endif""").replace("real(8)","real(wp)")
    _original,_prepared,output,manifest=generate(directory,source=source)
    assert manifest["scope_count"]==1
    objects=[]
    for role in ("common_runtime","shared_entry","original_source"):
        for unit in manifest["build_sources"]:
            if unit["role"]!=role:
                continue
            target=(output/unit["path"]).with_suffix(".o")
            command=([nvcc,"-std=c++17","-ccbin",host,"-arch=sm_86","-Xcompiler=-fopenmp","-I",str(output)]
                     if unit["language"]=="cuda" else
                     [fortran,"-std=f2018","-cpp","-DDOUBLE_PRECISION","-fopenmp","-fcheck=all"])
            run([*command,"-c",str(output/unit["path"]),"-o",str(target)],cwd=output)
            objects.append(str(target))
    from compiler.tests.test_source_scopes import DRIVER

    driver=output/"driver.f90"
    driver.write_text(DRIVER)
    target=output/"caller"
    run([fortran,"-fopenmp","-fcheck=all",str(driver),*objects,"-L/usr/local/cuda/lib64",
         "-Wl,-rpath,/usr/local/cuda/lib64","-lcudart","-lstdc++","-o",str(target)],cwd=output)
    return target,output
