"""Original ELSEIF reads stay lazy, with their existing access transactions."""
import json
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.guarded_conditions import GuardedConditionChain, chains
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.source import ScopeBuilder
from compiler.scopes.storage_admission import StorageAdmission, active_admission

SOURCE = """module guarded_fixture
implicit none
integer,allocatable::flags(:)
integer::visits=0
contains
subroutine advance(a,n,selector)
real(8),intent(inout)::a(:)
integer,intent(in)::n,selector
integer::i
do i=1,n
a(i)=a(i)+1.0_8
enddo
if(selector==1)then
visits=visits+1
do i=1,n
a(i)=a(i)+10.0_8
enddo
else if(flags(1)==2)then
visits=visits+1
do i=1,n
a(i)=a(i)+20.0_8
enddo
else if(flags(2)==3)then
visits=visits+1
do i=1,n
a(i)=a(i)+30.0_8
enddo
else
visits=visits+1
do i=1,n
a(i)=a(i)+40.0_8
enddo
endif
do i=1,n
a(i)=2.0_8*a(i)
enddo
end subroutine
end module
"""


def source_case(*, managed=False, nested=False):
    source = SOURCE
    if managed:
        source = source.replace("integer,allocatable::flags(:)\n", "")
        source = source.replace("subroutine advance(a,n,selector)", "subroutine advance(a,flags,n,selector)")
        source = source.replace("integer,intent(in)::n,selector", "integer,intent(inout)::flags(:)\ninteger,intent(in)::n,selector")
        source = source.replace("do i=1,n\na(i)=a(i)+1.0_8", "do i=1,n\nflags(i)=flags(i)+1\na(i)=a(i)+1.0_8", 1)
    if nested:
        source = source.replace("if(selector==1)then\nvisits=visits+1\ndo i=1,n\na(i)=a(i)+10.0_8\nenddo", """if(selector==1)then
visits=visits+1
if(n>=0)then
do i=1,n
a(i)=a(i)+10.0_8
enddo
else if(flags(3)==7)then
do i=1,n
a(i)=a(i)+100.0_8
enddo
endif""")
    return source


def emit(directory, monkeypatch, *, managed=False, nested=False, source=None):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "original.f90"
    path.write_text(source or source_case(managed=managed, nested=nested))
    fact = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    captures = {"argument::a": dict(fact)}
    if managed:
        captures["argument::flags"] = dict(fact)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": captures}
    builder = ScopeBuilder([path], "guarded_fixture::advance", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    original = LexicalOwner._emit_checked_lexical
    owners = []

    def observed(self, *args, **keywords):
        owners.append(self)
        return original(self, *args, **keywords)

    with monkeypatch.context() as patch:
        patch.setattr(LexicalOwner, "_emit_checked_lexical", observed)
        outputs, report = builder.run()
    manifest = json.loads(outputs["scope-manifest.json"])
    validate_checked_execution_plans(manifest, capture_facts=facts)
    package = directory / "package"
    package.mkdir()
    for name, content in outputs.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    _scope_package_build({"artifact_root": str(package), "scope_manifest": manifest})
    assert builder.finish()
    return path, builder, owners, outputs, report, package


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_exact_graph_and_package_keep_interleaved_predicates_and_bodies(tmp_path, monkeypatch, managed, nested):
    path, builder, owners, outputs, report, _package = emit(tmp_path, monkeypatch, managed=managed, nested=nested)
    assert report["scope_count"] == 1
    owner, = owners
    plan = owner.execution_plan
    assert not owner.boundaries
    attached = chains(owner, plan.units(owner))
    assert len(attached) == (2 if nested else 1)
    spans = [unit.scope.first for unit in plan.units(owner)]
    assert spans == sorted(spans)
    for chain in attached:
        assert chain.validate(owner) is chain
        for _header, unit in chain.publications:
            assert not unit.leaves
            assert not unit.written
            assert set(unit.arrays) == ({"argument::flags"} if managed else set())
            assert set(unit.host_metadata) == (set() if managed else {"guarded_fixture::flags"})
    generated = outputs[report["sources"][str(path)]["replacement"]]
    compact = "".join(generated.lower().split())
    assert "elseif(flags(1)==2)" not in compact
    assert compact.count("if(flags(1)==2)then") == 1
    assert compact.count("if(flags(2)==3)then") == 1
    assert "guarded_fixture::flags" not in builder.facts["captures"]


DRIVER = """program exercise
use guarded_fixture
use iso_c_binding
implicit none
interface
integer(c_int) function serial_visits()bind(C,name='guarded_test_serial_visits')
import c_int
end function
end interface
real(8),allocatable::a(:)
integer::n,mode,i,case_number
integer,parameter::sizes(3)=[0,1,7]
do case_number=1,3
n=sizes(case_number)
allocate(a(-2:n-3))
do mode=1,4
if(allocated(flags))deallocate(flags)
if(mode==2)then
allocate(flags(1));flags=2
else if(mode>=3)then
allocate(flags(2));flags=[0,0]
if(mode==3)flags(2)=3
endif
do i=-2,n-3
a(i)=real(i+31*mode+n,8)
enddo
visits=0
call advance(a,n,mode)
if(visits/=1)error stop 'original selected side effect did not execute once'
do i=-2,n-3
if(a(i)/=(real(i+31*mode+n,8)+1.0_8+10.0_8*mode)*2.0_8)error stop 'wrong selected body or repeated work'
enddo
enddo
deallocate(a)
enddo
if(serial_visits()==0)error stop 'instrumented owner did not execute'
print *,'GUARDED_ELSEIF_CPU_FIELDS_AND_LAZY_FLAGS_PASS'
end program
"""


@pytest.mark.parametrize("nested", [False, True])
def test_native_cpu_execution_keeps_unallocated_late_guards_and_default_branch(tmp_path, monkeypatch, nested):
    _path, _builder, _owners, _outputs, report, package = emit(tmp_path, monkeypatch, nested=nested)
    assert report["scope_count"] == 1
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    cc, nm = shutil.which("gcc"), shutil.which("nm")
    assert fortran
    assert cc
    assert nm
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(report["build_sources"]):
            if artifact["language"] != "fortran" or artifact["role"] != role:
                continue
            target = package / f"fortran-{index}.o"
            result = subprocess.run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-ffp-contract=off",
                                     "-J", str(package), "-I", str(package), "-c", str(package / artifact["path"]),
                                     "-o", str(target)], cwd=package, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
            objects.append(target)
    undefined = subprocess.run([nm, "-u", *map(str, objects)], capture_output=True, text=True, timeout=15, check=True)
    defined = subprocess.run([nm, "-g", "--defined-only", *map(str, objects)],
                             capture_output=True, text=True, timeout=15, check=True)
    definitions = {line.split()[-1] for line in defined.stdout.splitlines() if line.split()}
    names = {line.split()[-1] for line in undefined.stdout.splitlines() if line.split()
             and line.split()[-1].startswith(("fort_", "cpp_fort_"))} - definitions
    assert "fort_scope_serial_caller" in names
    # Only control geometry and the actual original native fallback execute.
    # Any unexpected runtime/numerical entry aborts; no CUDA library is linked.
    stub = package / "never-executed-runtime.c"
    stub.write_text("#include <stdlib.h>\n#include <stdio.h>\nstatic int visits;\n"
                    "int fort_scope_serial_caller(void){++visits;return 0;}\n"
                    "int guarded_test_serial_visits(void){return visits;}\n" + "\n".join(
                        f'int {name}(void){{fputs("unexpected entry {name}\\n",stderr);abort();}}'
                        for name in sorted(names - {"fort_scope_serial_caller"})) + "\n")
    stub_object = package / "stubs.o"
    compiled = subprocess.run([cc, "-c", str(stub), "-o", str(stub_object)], capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    driver = package / "driver.f90"
    driver.write_text(DRIVER)
    binary = package / "exercise"
    linked = subprocess.run([fortran, "-fopenmp", "-fcheck=all,array-temps", "-J", str(package), "-I", str(package),
                             str(driver), *map(str, objects), str(stub_object), "-o", str(binary)],
                            cwd=package, capture_output=True, text=True, timeout=60)
    assert linked.returncode == 0, linked.stdout + linked.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "GUARDED_ELSEIF_CPU_FIELDS_AND_LAZY_FLAGS_PASS" in executed.stdout


@pytest.mark.parametrize("fault", ["member", "original", "headers", "closing", "publications", "unit_attachment"])
def test_active_admission_rejects_guard_geometry_changes_before_membership(tmp_path, monkeypatch, fault):
    _path, _builder, owners, _outputs, _report, _package = emit(tmp_path, monkeypatch)
    owner, = owners
    plan = owner.execution_plan
    chain, = chains(owner, plan.units(owner))
    member = next(member for member in plan.members if plan.units(member))
    unit = next(unit for unit in plan.units(member) if unit.arrays)

    def inspect():
        with StorageAdmission(plan) as admission:
            if fault == "member":
                object.__setattr__(chain, "member", object())
            elif fault == "original":
                object.__setattr__(chain, "original", chain.headers[0])
            elif fault == "headers":
                object.__setattr__(chain, "headers", tuple(reversed(chain.headers)))
            elif fault == "closing":
                object.__setattr__(chain, "closing", chain.headers[-1])
            elif fault == "publications":
                object.__setattr__(chain, "publications", ())
            else:
                object.__setattr__(chain.publications[0][1], "guarded_condition", None)
            admission.admits_many(plan.storage_epoch(), plan, member, unit.arrays.values())
            pytest.fail("changed guard geometry must reject before canonical membership")

    with pytest.raises(CompilationError):
        inspect()
    assert active_admission() is None


@pytest.mark.parametrize("fault", ["duplicate_publication", "unexpected_header_insertion", "closing_replacement"])
def test_emission_rejects_ambiguous_guard_insertions(tmp_path, monkeypatch, fault):
    original = GuardedConditionChain.apply

    def corrupt(self, member, edits):
        header, unit = next((header, unit) for header, unit in self.publications if header is not self.headers[0])
        first, last, text = member.unit_emission(unit).source_edits[0]
        if fault == "duplicate_publication":
            edits.append((member.routine.scope.path, first, last, text, False))
        elif fault == "unexpected_header_insertion":
            edits.append((member.routine.scope.path, first, last, "! foreign insertion\n", False))
        else:
            first = self.closing.item.span[0]
            edits.append((member.routine.scope.path, first, first, "endif\n", False))
        return original(self, member, edits)

    monkeypatch.setattr(GuardedConditionChain, "apply", corrupt)
    path, _builder, _owners, outputs, report, _package = emit(tmp_path, monkeypatch)
    assert report["scope_count"] == 0
    assert "guarded ELSEIF" in report["graph_owner_attempt"]["error"]
    assert not report["source_edits"]
    assert str(path) not in report["sources"]
    assert path.read_text() == SOURCE


def test_failed_later_publication_discards_the_existing_whole_candidate(tmp_path, monkeypatch):
    original = LexicalOwner.admit

    def reject(self, nodes, **keywords):
        if keywords.get("condition_only") and "flags(2)" in str(nodes[0]).replace(" ", "").lower():
            raise CompilationError("injected late original publication rejection")
        return original(self, nodes, **keywords)

    monkeypatch.setattr(LexicalOwner, "admit", reject)
    path, _builder, _owners, outputs, report, _package = emit(tmp_path, monkeypatch)
    assert report["scope_count"] == 0
    assert "guarded IF publication admission failed" in report["graph_owner_attempt"]["error"]
    assert not report["source_edits"]
    assert str(path) not in report["sources"]
    assert path.read_text() == SOURCE


@pytest.mark.parametrize("fault", ["named", "semicolon_header", "semicolon_end", "continued_header"])
def test_unproved_original_header_geometry_keeps_native_chain(tmp_path, monkeypatch, fault):
    source = SOURCE
    if fault == "named":
        source = source.replace("if(selector==1)then", "outer: if(selector==1)then")
        source = source.replace("endif\ndo i=1,n\na(i)=2", "endif outer\ndo i=1,n\na(i)=2")
    elif fault == "semicolon_header":
        source = source.replace("else if(flags(1)==2)then\n", "else if(flags(1)==2)then; i=1\n")
    elif fault == "semicolon_end":
        source = source.replace("endif\ndo i=1,n\na(i)=2", "endif; i=1\ndo i=1,n\na(i)=2")
    else:
        source = source.replace("else if(flags(1)==2)then", "else if( &\nflags(1)==2)then")
    _path, _builder, owners, _outputs, report, _package = emit(tmp_path, monkeypatch, source=source)
    assert any("guarded ELSEIF" in row["reason"] or "source line" in row["reason"] or "source edit" in row["reason"]
               for row in report["graph_owner_attempt"]["boundaries"])
    assert all(not chains(owner, owner.execution_plan.units(owner)) for owner in owners)
