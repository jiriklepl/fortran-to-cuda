"""Original companions preserve optional presence without capturing a value."""

import re
import shutil
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT, run

ENTRY = "renamed_client::step"


def optional_sources(*, real_kind=8, position="middle", calls=None, nested=False, declaration=None):
    formals = {"first": "option,x,n,escape", "middle": "x,option,n,escape",
               "last": "x,n,escape,option"}[position]
    declaration = declaration or "integer,optional,intent(in)::option"
    leaf = f"""module shifted_leaf
implicit none
integer::child_visits=0
contains
subroutine adjust({formals})
real({real_kind}),intent(inout)::x(-4:)
integer,intent(in)::n
logical,intent(in)::escape
{declaration}
integer::j,increment
increment=5
if(present(option)) increment=option
child_visits=child_visits+1
do j=-4,n-5
x(j)=x(j)+real(j,{real_kind})+real(increment,{real_kind})
enddo
if(escape) then
call opaque(x,n)
endif
end subroutine
end module
"""
    forward = f"""module forwarding_lib
use shifted_leaf,only:renamed_leaf=>adjust
implicit none
contains
subroutine relay(x,option,n,escape)
real({real_kind}),intent(inout)::x(-4:)
integer,optional,intent(in)::option
integer,intent(in)::n
logical,intent(in)::escape
call renamed_leaf(escape=escape,n=n,option=option,x=x)
end subroutine
end module
"""
    module, target = ("forwarding_lib", "relay") if nested else ("shifted_leaf", "adjust")
    calls = calls or ("call renamed(x=b,n=n,escape=escape)",
                      "call renamed(escape=escape,n=n,option=option,x=b)")
    client = f"""module renamed_client
use {module},only:renamed=>{target}
implicit none
integer::visits=0
contains
subroutine step(a,b,out,n,escape,option)
real({real_kind}),intent(in)::a(-2:)
real({real_kind}),intent(inout)::b(-2:),out(-2:)
integer,intent(in)::n
logical,intent(in)::escape
integer,optional,intent(in)::option
integer::i
visits=visits+1
do i=-2,n-3
b(i)=2*a(i)+real(i,{real_kind})
enddo
i=0
{chr(10).join(calls)}
do i=-2,n-3
out(i)=a(i)+b(i)
enddo
end subroutine
end module
"""
    return {"leaf.f90": leaf, **({"forward.f90": forward} if nested else {}), "client.f90": client}


def emit_optional(directory, **options):
    return emit_sources(directory, optional_sources(**options))


def emit_sources(directory, sources):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in sources.items():
        path = directory / name
        path.write_text(text)
        paths.append(path)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}}
    outputs, report = ScopeBuilder(paths, ENTRY, facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="reached")).run()
    return paths, outputs, report


def source_text(outputs):
    return "\n".join(text for name, text in outputs.items() if name.startswith("sources/")).lower()


def companion_calls(text, name):
    return [" ".join(value.replace("&", "").split()) for value in
            re.findall(r"call\s+" + re.escape(name) + r"\s*\((.*?)\)", text, flags=re.S)]


@pytest.mark.parametrize("position", ["first", "middle", "last"])
@pytest.mark.parametrize("real_kind", [4, 8])
def test_omitted_and_reordered_optional_calls_share_one_original_companion(tmp_path, position, real_kind):
    _, outputs, report = emit_optional(tmp_path, position=position, real_kind=real_kind)
    owner, = report["scopes"]
    child, = owner["module_coordinators"]
    assert child["procedure"] == "shifted_leaf::adjust"
    assert child["optional_scalar_formals"] == ["option"]
    assert child["resource_mappings"] == {"argument::x": "argument::b"}
    assert len(child["calls"]) == 2
    assert not owner["boundaries"]
    assert len(child["boundaries"]) == 1
    assert "opaque" in child["boundaries"][0]["reason"]
    # The following statically omitted call retains the existing numerical
    # liveness boundary. Companion argument admission does not relax it.
    assert owner["gpu_leaves"] == ["renamed_client::step#region1", "shifted_leaf::adjust#region1"]
    assert any("requires all original arguments" in item["reason"] for item in report["boundaries"])
    text = source_text(outputs)
    calls = companion_calls(text, child["entry"])
    assert len(calls) == 2
    assert calls[0].lstrip().startswith("x = b, n = n, escape = escape,")
    assert "option =" not in calls[0]
    assert calls[1].lstrip().startswith("escape = escape, n = n, option = option, x = b,")
    for call in calls:
        controls = call[call.index("fort_child_context_"):]
        assert all("=" in arg for arg in controls.replace("&", "").split(","))
        assert "none" not in call
    assert text.count("if(present(option)) increment=option") == 1
    assert text.count("child_visits=child_visits+1") == 1
    assert "optional" not in " ".join(owner["ownership"]["retained_resources"])
    assert not any("option" in item["resource"] for item in owner["resource_bindings"]["resources"])


@pytest.mark.parametrize("nested", [False, True])
def test_original_positional_and_keyword_syntax_keeps_optional_forwarding(tmp_path, nested):
    calls = ("call renamed(b,option,n,escape)", "call renamed(b,n=n,escape=escape)",
             "call renamed(b,option,n=n,escape=escape)")
    _, outputs, report = emit_optional(tmp_path, calls=calls, nested=nested)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    assert len(owner["module_coordinators"]) == (2 if nested else 1)
    for child in owner["module_coordinators"]:
        assert child["optional_scalar_formals"] == ["option"]
        assert child["native_abi"] == "original entry; compiler-control arguments are never referenced"
    text = source_text(outputs)
    child = next(item for item in owner["module_coordinators"]
                 if item["procedure"] == ("forwarding_lib::relay" if nested else "shifted_leaf::adjust"))
    rendered = companion_calls(text, child["entry"])
    assert len(rendered) == 3
    assert rendered[0].lstrip().startswith("b, option, n, escape,")
    assert rendered[1].lstrip().startswith("b, n = n, escape = escape,")
    assert rendered[2].lstrip().startswith("b, option, n = n, escape = escape,")
    if nested:
        leaf = next(item for item in owner["module_coordinators"] if item["procedure"] == "shifted_leaf::adjust")
        assert companion_calls(text, leaf["entry"])[0].lstrip().startswith(
            "escape = escape, n = n, option = option, x = x,")
        calls = [item for item in report["resolved_calls"] if item["procedure"] == "shifted_leaf::adjust"]
        assert calls[0]["resource_mappings"][1]["presence"] == "forwarded_optional"


def test_runtime_optional_absence_can_retain_producer_child_and_consumer(tmp_path):
    _, _, report = emit_optional(tmp_path, calls=(
        "call renamed(escape=escape,n=n,option=option,x=b)",), nested=True)
    owner, = report["scopes"]
    assert owner["gpu_leaves"] == ["renamed_client::step#region1", "renamed_client::step#region2",
                                   "shifted_leaf::adjust#region1"]
    assert not owner["boundaries"]
    assert len(owner["module_coordinators"]) == 2


@pytest.mark.parametrize(("declaration", "value"), [
    ("real(4),optional,intent(in)::option", "int(option)"),
    ("real(8),optional,intent(in)::option", "int(option)"),
    ("integer(8),optional,intent(in)::option", "int(option)"),
    ("logical,optional,intent(in)::option", "merge(7,5,option)"),
])
def test_optional_scalar_native_types_do_not_become_device_captures(tmp_path, declaration, value):
    sources = optional_sources(calls=("call renamed(x=b,n=n,escape=escape)",), declaration=declaration)
    sources["leaf.f90"] = sources["leaf.f90"].replace("increment=option", "increment=" + value)
    _, _, report = emit_sources(tmp_path, sources)
    child, = report["scopes"][0]["module_coordinators"]
    assert child["optional_scalar_formals"] == ["option"]
    assert len(child["boundaries"]) == 1
    assert "opaque" in child["boundaries"][0]["reason"]


@pytest.mark.parametrize("mode", ["array_payload", "source_call", "metadata"])
def test_optional_scalar_bypass_does_not_hide_other_effects(tmp_path, mode):
    sources = optional_sources()
    if mode == "array_payload":
        sources["leaf.f90"] = sources["leaf.f90"].replace("increment=option", "increment=option+int(x(-4))")
    elif mode == "source_call":
        helper = "integer function bump(k)\ninteger,intent(in)::k\nbump=k+1\nend function\n"
        sources["leaf.f90"] = sources["leaf.f90"].replace("contains\n", "contains\n" + helper, 1).replace(
            "increment=option", "increment=bump(option)")
    else:
        sources["leaf.f90"] = sources["leaf.f90"].replace("integer::child_visits=0", """integer::child_visits=0
type point
integer::coordinate
end type
type(point),allocatable,target::items(:)""").replace("if(present(option)) increment=option",
            "if(allocated(items)) then\nif(present(option)) increment=option+items(1)%coordinate\nendif")
    _, outputs, report = emit_sources(tmp_path, sources)
    child, = report["scopes"][0]["module_coordinators"]
    assert len(child["boundaries"]) >= 2
    assert any("opaque" not in item["reason"] for item in child["boundaries"])
    text = source_text(outputs)
    assert re.search(r"if\s*\(\s*present\(option\)\s*\)", text)
    assert not any("option" in item["resource"] for item in report["scopes"][0]["resource_bindings"]["resources"])


@pytest.mark.parametrize("declaration", [
    "integer,optional,intent(out)::option", "integer,optional,intent(inout)::option",
    "integer,optional,pointer,intent(in)::option", "integer,optional,allocatable,intent(in)::option",
    "integer,optional,intent(in)::option(:)",
])
def test_other_optional_associations_remain_boundaries(tmp_path, declaration):
    # Omission is legal for every signature; none grants a companion.
    _, _, report = emit_optional(tmp_path, calls=("call renamed(x=b,n=n,escape=escape)",),
                                declaration=declaration)
    owner, = report["scopes"]
    assert not owner["module_coordinators"]
    assert owner["boundaries"]


@pytest.mark.parametrize("attribute", ["allocatable", "pointer", "allocatable,optional", "pointer,optional"])
def test_allocation_dependent_optional_presence_remains_a_boundary(tmp_path, attribute):
    sources = optional_sources(calls=("call renamed(x=b,option=option,n=n,escape=escape)",))
    sources["client.f90"] = sources["client.f90"].replace("integer,optional,intent(in)::option",
                                                        f"integer,{attribute},intent(in)::option")
    paths = []
    for name, text in sources.items():
        path = tmp_path / name
        path.write_text(text)
        paths.append(path)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(p): sha256(p.read_bytes()).hexdigest() for p in paths},
             "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}}
    _, report = ScopeBuilder(paths, ENTRY, facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="reached")).run()
    assert not report["scopes"][0]["module_coordinators"]
    assert "presence" in report["scopes"][0]["boundaries"][0]["reason"]


@pytest.mark.native
@pytest.mark.parametrize("real_kind", [4, 8])
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_generated_optional_entry_and_original_native_calls_compile(tmp_path, real_kind, position):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    originals, outputs, report = emit_optional(tmp_path / "sources", real_kind=real_kind, position=position, nested=True)
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for role in ("common_runtime", "shared_entry"):
        for item in report["build_sources"]:
            if item["role"] == role and item["language"] == "fortran":
                run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-fsyntax-only",
                     "-J", str(build), "-I", str(build), str(build / item["path"])], cwd=build)
    # The public inventory is not a module topological sort. Preserve the
    # explicit provider -> forwarding module -> client source order instead.
    for original in originals:
        replacement = report["sources"][str(original)]["replacement"]
        run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-fsyntax-only",
             "-J", str(build), "-I", str(build), str(build / replacement)], cwd=build)
    native = build / "native_callers.f90"
    native.write_text(f"""subroutine native_callers(x,n)
use shifted_leaf,only:adjust
use forwarding_lib,only:relay
real({real_kind})::x(-4:)
integer::n
call adjust(x=x,n=n,escape=.false.)
call adjust(escape=.false.,option=7,x=x,n=n)
call relay(x=x,n=n,escape=.false.)
end subroutine
""")
    run([fortran, "-std=f2018", "-fcheck=all", "-fsyntax-only", "-J", str(build), "-I", str(build),
         str(native)], cwd=build)
