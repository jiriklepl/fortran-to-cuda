"""Original allocatable descriptors survive companion association unchanged."""
import re
import shutil
import subprocess
from hashlib import sha256

import pytest
from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import Base

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = """module forwarding
implicit none
contains
subroutine advance(a,q,n)
real(8),intent(inout)::a(:)
real(8),allocatable,intent(in)::q(:)
integer,intent(in)::n
call work(a,q,n)
end subroutine
subroutine work(a,q,n)
real(8),intent(inout)::a(:)
real(8),allocatable,intent(in)::q(:)
integer,intent(in)::n
integer::i
if(allocated(q))then
do i=1,n
a(i)=a(i)+q(lbound(q,1)+i-1)
enddo
else
do i=1,n
a(i)=a(i)+2.0_8
enddo
endif
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, initialized="none"):
    path = tmp_path / "source.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: {"storage": "stable", "initialized": initialized if name == "q" else "whole",
                 "escapes": False, "allocation_changes": False} for name in ("a", "q")}}
    return ScopeBuilder([path], "forwarding::advance", facts=facts,
        options=CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution="graph"),
        config=OffloadConfig(policy="sections", scope_execution="graph"))


def generate(tmp_path, source=SOURCE, initialized="none"):
    return fixture(tmp_path, source, initialized).run()


def test_allocatable_input_companion_retains_exact_original_descriptor_and_none_fact(tmp_path):
    outputs, report = generate(tmp_path)
    assert report["scope_count"] == 1, report
    scope = report["scopes"][0]
    coordinator = next(row for row in scope["module_coordinators"] if row["procedure"] == "forwarding::work")
    assert len(coordinator["readonly_allocatable_inputs"]) == 1
    proof = coordinator["readonly_allocatable_inputs"][0]
    assert proof["payload_definition_authority"] is False
    assert proof["allocation_presence_authority"] is False
    original = next(text for name, text in outputs.items() if name.startswith("sources/"))
    assert "ALLOCATABLE, INTENT(IN), TARGET :: q(:)" in original
    # The original ALLOCATED branch and both loop bodies are retained. Every
    # generated actual inquiry/registration is below its own presence guard.
    assert original.lower().count("if (allocated(q)) then") >= 2
    assert "shape(q, kind=c_size_t)" in original.lower()
    assert "lbound(q, kind=c_int64_t)" in original.lower()
    slot = next(resource["registration_identity"] for resource in scope["checked_execution_plan"]["resources"]
                if resource["resource"] == "argument::q")
    assert re.search(rf"fort_scope_register\([^\n]*, {slot}_c_int64_t, &\n"
                     rf"1_c_int64_t, fort_actual_layout, 0_c_int, fort_buffer_{slot}\)", original)
    assert not re.search(rf"fort_actual_layout, 1_c_int, fort_buffer_{slot}\)", original)
    parsed = ParserFactory().create(std="f2008")(FortranStringReader(original))
    pending, inquiries = [(parsed, ())], 0
    while pending:
        node, parents = pending.pop()
        kind = type(node).__name__
        if kind in {"Function_Reference", "Intrinsic_Function_Reference", "Part_Ref"}:
            function, arguments = node.items
            values = getattr(arguments, "items", ()) or ()
            if str(function).lower() in {"shape", "lbound", "ubound", "size", "c_loc", "is_contiguous"} and values and str(values[0]).lower() == "q":
                inquiries += 1
                assert any(type(parent).__name__ == "If_Construct"
                    and "".join(str(parent.content[0]).lower().split()) == "if(allocated(q))then"
                    for parent in parents), str(node)
        children = node if isinstance(node, (list, tuple)) else getattr(node, "content", getattr(node, "items", ())) or ()
        pending.extend((child, (*parents, node)) for child in children if isinstance(child, (Base, tuple, list)))
    assert inquiries > 0


@pytest.mark.parametrize("replacement", [
    "real(8),allocatable,target,intent(in)::q(:)",
    "real(8),intent(in)::q(:)",
])
def test_actual_requires_its_independent_original_allocatable_descriptor(tmp_path, replacement):
    # Change only the caller declaration. A matching name is never evidence of
    # a legal independent allocatable actual for the new companion.
    source = SOURCE.replace("real(8),allocatable,intent(in)::q(:)", replacement, 1)
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.module_owner import borrow_module

    builder = fixture(tmp_path, source)
    call, = builder.entry.execution.content
    with pytest.raises(CompilationError, match="allocatable"):
        borrow_module(GraphOwner(builder), call)
    _outputs, report = generate(tmp_path, source)
    assert not any(row.get("procedure") == "forwarding::work" and row.get("readonly_allocatable_inputs")
                   for scope in report.get("scopes", ()) for row in scope.get("module_coordinators", ()))
    assert report["scope_count"] == 0


@pytest.mark.native
def test_companion_and_original_allocatable_abi_compile_with_gfortran(tmp_path):
    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    if compiler is None:
        pytest.skip("Fortran compiler unavailable")
    outputs, report = generate(tmp_path)
    for name, text in outputs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] == role and item["language"] == "fortran":
                command = [compiler, "-std=f2018", "-fopenmp", "-fcheck=all", "-c", item["path"]]
                result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
                assert result.returncode == 0, result.stderr
