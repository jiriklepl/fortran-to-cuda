"""Original module associations demand payload facts only for retained work."""

import re
from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

FACT = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
ENTRY = "saffron_owner::advance"

GATE_SOURCE = """module saffron_gate
implicit none
integer::gate_visits=0
contains
subroutine branch_inputs(work,unproved,n,enabled)
real(8),intent(inout)::work(-4:)
real(8),contiguous,intent(in)::unproved(-4:)
integer,intent(in)::n
logical,intent(in)::enabled
integer::j
gate_visits=gate_visits+1
if(enabled) then
do j=-3,n
work(j)=work(j)+unproved(j)
enddo
endif
end subroutine
end module
"""

SWEEP_SOURCE = """module saffron_sweep
implicit none
integer::sweep_visits=0,middle_visits=0
contains
subroutine diffuse(work,result,n)
real(8),intent(inout)::work(-4:),result(-4:)
integer,intent(in)::n
integer::j
sweep_visits=sweep_visits+1
do j=-3,n
result(j)=2.0_8*work(j)
enddo
if(n>=-3) then
middle_visits=middle_visits+1
result(-3)=result(-3)+0.5_8
endif
do j=-3,n
work(j)=work(j)+result(j)
enddo
end subroutine
end module
"""

OWNER_SOURCE = """module saffron_owner
use saffron_gate,only:gate=>branch_inputs
use saffron_sweep,only:sweep=>diffuse
implicit none
integer::owner_visits=0
contains
subroutine advance(a,b,c,extra,n,flag)
real(8),intent(inout)::a(-4:),b(-4:),c(-4:)
real(8),allocatable,intent(in)::extra(:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
owner_visits=owner_visits+1
do i=-3,n
b(i)=b(i)+a(i)+1.0_8
enddo
call gate(b,extra,n,flag)
call sweep(b,c,n)
do i=-3,n
a(i)=b(i)+c(i)+1.0_8
enddo
end subroutine
end module
"""

RELAY_SOURCE = """module saffron_relay
use saffron_gate,only:original_gate=>branch_inputs
use saffron_sweep,only:process=>diffuse
implicit none
contains
subroutine relay(values,other,n,flag,result)
real(8),intent(inout)::values(-4:),result(-4:)
real(8),intent(in)::other(-4:)
integer,intent(in)::n
logical,intent(in)::flag
call original_gate(values,other,n,flag)
call process(values,result,n)
end subroutine
end module
"""


def write_module_deferred_sources(directory, *, precision=8, gate_source=GATE_SOURCE,
                                  owner_source=OWNER_SOURCE, extra_sources=()):
    """Write exact source strings without constructing compiler objects."""
    directory.mkdir(parents=True, exist_ok=True)
    sources = [("saffron_gate.f90", gate_source), ("saffron_sweep.f90", SWEEP_SOURCE),
               *extra_sources, ("saffron_owner.f90", owner_source)]
    paths = []
    for name, source in sources:
        path = directory / name
        path.write_text(source.replace("real(8)", f"real({precision})")
                       .replace("_8", f"_{precision}"))
        paths.append(path)
    return tuple(paths)


def module_deferred_facts(paths, *, captures=None):
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {"argument::" + name: deepcopy(FACT) for name in ("a", "b", "c")}}
    for root, value in (captures or {}).items():
        if value is None:
            facts["captures"].pop(root, None)
        else:
            facts["captures"][root] = deepcopy(value)
    return facts


def module_deferred_fixture(directory, *, captures=None, **source_options):
    paths = write_module_deferred_sources(directory, **source_options)
    builder = ScopeBuilder(paths, ENTRY, facts=module_deferred_facts(paths, captures=captures),
                           options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    return paths, builder


def emit_module_deferred_fixture(directory, **options):
    paths, builder = module_deferred_fixture(directory, **options)
    outputs, report = builder.run()
    return paths, outputs, report, builder


def replacement(paths, outputs, report, name):
    path = next(path for path in paths if path.name == name)
    record = report["sources"].get(str(path))
    return (outputs[record["replacement"]] if record else path.read_text()).lower()


def coordinator(owner, procedure):
    return next(child for child in owner["module_coordinators"] if child["procedure"] == procedure)


def test_guarded_unused_input_keeps_original_association_and_later_owner(tmp_path):
    paths, outputs, report, builder = emit_module_deferred_fixture(tmp_path)
    owner, = report["scopes"]
    child = coordinator(owner, "saffron_gate::branch_inputs")
    sweep, = report["borrowed_source_coordinators"]
    assert sweep["procedure"] == "saffron_sweep::diffuse"
    assert not owner["boundaries"]
    assert len(owner["gpu_leaves"]) == 4
    assert sweep["ownership"]["close_count"] == 0
    assert len(sweep["gpu_leaves"]) == 2
    point, = (operation for operation in sweep["native_operations"]
              if operation["resource_effects"] == {"argument::result": ["read", "write"]})
    section, = point["sections"]["resources"]
    assert section["reads"][0]["axes"][0]["kind"] == "point"
    assert section["reads"][0]["axes"][0]["lower"] == {"kind": "literal", "value": -3}
    assert len(child["boundaries"]) == 1
    assert "argument::extra" in child["boundaries"][0]["reason"]
    assert child["resource_mappings"] == {"argument::work": "argument::b", "argument::unproved": "argument::extra"}
    assert "argument::extra" not in builder.facts["captures"]
    assert "argument::extra" not in owner["ownership"]["retained_resources"]
    caller = replacement(paths, outputs, report, "saffron_owner.f90")
    native = replacement(paths, outputs, report, "saffron_gate.f90")
    assert "allocated(extra)" in caller
    assert "is_contiguous(extra)" in caller
    assert caller.index("is_contiguous(extra)") < caller.index("call " + child["entry"])
    assert "c_loc(extra)" not in caller
    assert "lbound(extra" not in caller
    assert "ubound(extra" not in caller
    guard = native.index("if(enabled) then")
    work = native.index("work(j)=work(j)+unproved(j)")
    assert guard < native.index("fort_scope_close(", guard) < work
    assert native.count("work(j)=work(j)+unproved(j)") == 1
    assert native.count("gate_visits=gate_visits+1") == 1
    assert "c_loc(unproved)" not in native
    assert "lbound(unproved" not in native
    assert caller.count("owner_visits=owner_visits+1") == 1
    assert caller.count("b(i)=b(i)+a(i)+1.0_8") == 1
    assert caller.count("a(i)=b(i)+c(i)+1.0_8") == 1


def test_relay_uses_descendant_handles_and_original_renamed_actuals(tmp_path):
    source = OWNER_SOURCE.replace("use saffron_gate,only:gate=>branch_inputs",
                                 "use saffron_relay,only:gate=>relay").replace(
        "call gate(b,extra,n,flag)\ncall sweep(b,c,n)", "call gate(b,extra,n,flag,c)")
    paths, outputs, report, _builder = emit_module_deferred_fixture(
        tmp_path, owner_source=source, extra_sources=(("saffron_relay.f90", RELAY_SOURCE),))
    owner, = report["scopes"]
    relay = coordinator(owner, "saffron_relay::relay")
    child = coordinator(owner, "saffron_gate::branch_inputs")
    assert not owner["boundaries"]
    assert not relay["boundaries"]
    assert len(child["boundaries"]) == 1
    assert len(owner["gpu_leaves"]) == 4
    assert "argument::extra" not in owner["ownership"]["retained_resources"]
    text = replacement(paths, outputs, report, "saffron_relay.f90")
    assert "is_contiguous(other)" in text
    assert "call " + child["entry"] in text
    assert "c_loc(other)" not in text
    sweep, = report["borrowed_source_coordinators"]
    assert sweep["procedure"] == "saffron_sweep::diffuse"
    assert relay["resource_mappings"]["argument::values"] == "argument::b"
    assert relay["resource_mappings"]["argument::result"] == "argument::c"
    handles = ["fort_buffer_" + str(item["registration_identity"])
               for item in owner["resource_bindings"]["resources"] if item["resource"] in {"argument::b", "argument::c"}]
    entry = re.search(r"entry " + relay["entry"] + r"\((.*?)\)", text, re.DOTALL)
    assert entry is not None
    assert all(handle in entry.group(1) for handle in handles)


def out_sources(*, prior_use):
    gate = GATE_SOURCE.replace("unproved,n,enabled)", "unproved,n,enabled,discarded)").replace(
        "integer,intent(in)::n", "real(8),intent(out)::discarded(-4:)\ninteger,intent(in)::n")
    owner = OWNER_SOURCE.replace("extra,n,flag)", "extra,n,flag,lost)").replace(
        "real(8),allocatable,intent(in)::extra(:)",
        "real(8),allocatable,intent(in)::extra(:)\nreal(8),intent(inout)::lost(-4:)").replace(
        "call gate(b,extra,n,flag)", "call gate(b,extra,n,flag,lost)").replace(
        "call sweep(b,c,n)", "do i=-3,n\nlost(i)=b(i)\nenddo\ncall sweep(b,c,n)").replace(
        "a(i)=b(i)+c(i)+1.0_8", "a(i)=b(i)+c(i)+lost(i)+1.0_8")
    if prior_use:
        owner = owner.replace("call gate(b,extra,n,flag,lost)",
                              "do i=-3,n\nlost(i)=lost(i)+a(i)\nenddo\ncall gate(b,extra,n,flag,lost)")
    return gate, owner


@pytest.mark.parametrize("prior_use", [False, True])
def test_unused_out_is_registered_then_killed_before_branch_without_late_reseed(tmp_path, prior_use):
    gate_source, owner_source = out_sources(prior_use=prior_use)
    paths, outputs, report, _builder = emit_module_deferred_fixture(
        tmp_path, gate_source=gate_source, owner_source=owner_source, captures={"argument::lost": FACT})
    owner, = report["scopes"]
    child = coordinator(owner, "saffron_gate::branch_inputs")
    assert not owner["boundaries"]
    assert child["resource_mappings"]["argument::discarded"] == "argument::lost"
    caller = replacement(paths, outputs, report, "saffron_owner.f90")
    native = replacement(paths, outputs, report, "saffron_gate.f90")
    match = re.search(r"fort_actual_lowers_(\d+) = lbound\(lost,", caller)
    assert match is not None
    handle = "fort_buffer_" + match.group(1)
    register = match.start()
    call = caller.index("call " + child["entry"])
    assert register < call < caller.index("lost(i)=b(i)")
    assert handle in caller[call:caller.index("endif", call)]
    forget = "fort_scope_forget_definition("
    assert native.count(forget) == 1
    event = native.index(forget)
    assert handle in native[event:native.index("\n", event)]
    assert event < native.index("gate_visits=gate_visits+1") < native.index("if(enabled) then")
    # Later ordinary registration is conditional on this same handle being
    # zero. The caller's whole fact cannot resurrect OUT-invalidated halos.
    assert f".and. {handle} == 0) then" in caller[call:]
    assert f"{handle} = 0" not in caller[call:]
    if prior_use:
        assert "lost(i)=lost(i)+a(i)" in caller
    else:
        assert "lost(i)=lost(i)+a(i)" not in caller


def test_missing_unused_out_fact_rolls_back_companion_and_closes_at_original_call(tmp_path):
    gate_source, owner_source = out_sources(prior_use=False)
    paths, outputs, report, builder = emit_module_deferred_fixture(
        tmp_path, gate_source=gate_source, owner_source=owner_source)
    owner, = report["scopes"]
    assert all(child["procedure"] != "saffron_gate::branch_inputs" for child in owner["module_coordinators"])
    rejected = builder.region_owners.get("saffron_gate::branch_inputs")
    if rejected is not None:
        assert not rejected.used
        assert not rejected.original_numerical_regions
    assert any("argument::lost" in boundary["reason"] for boundary in owner["boundaries"])
    caller = replacement(paths, outputs, report, "saffron_owner.f90")
    call = caller.index("call gate(b,extra,n,flag,lost)")
    assert "fort_scope_close(" in caller[:call]
    assert caller.count("call gate(b,extra,n,flag,lost)") == 1
    native = replacement(paths, outputs, report, "saffron_gate.f90")
    assert "entry fort_scope_entry_" not in native


def test_excluded_input_import_alias_cannot_be_shadowed_by_preflight_local(tmp_path):
    source = OWNER_SOURCE.replace("implicit none\ninteger::owner_visits",
        "use saffron_extra,only:fort_actuals_contiguous=>spare\nimplicit none\ninteger::owner_visits").replace(
        "advance(a,b,c,extra,n,flag)", "advance(a,b,c,n,flag)").replace(
        "real(8),allocatable,intent(in)::extra(:)\n", "").replace(
        "call gate(b,extra,n,flag)", "call gate(b,fort_actuals_contiguous,n,flag)")
    extra = "module saffron_extra\nreal(8),allocatable::spare(:)\nend module\n"
    paths, outputs, report, builder = emit_module_deferred_fixture(
        tmp_path, owner_source=source, extra_sources=(("saffron_extra.f90", extra),))
    owner, = report["scopes"]
    assert all(child["procedure"] != "saffron_gate::branch_inputs" for child in owner["module_coordinators"])
    rejected = builder.region_owners.get("saffron_gate::branch_inputs")
    if rejected is not None:
        assert not rejected.used
        assert not rejected.original_numerical_regions
    assert any(boundary["reason"].endswith(
        "; original-body continuation: original module companion actual conflicts with descriptor guard name")
               for boundary in owner["boundaries"])
    caller = replacement(paths, outputs, report, "saffron_owner.f90")
    assert caller.count("call gate(b,fort_actuals_contiguous,n,flag)") == 1
    assert "c_loc(fort_actuals_contiguous)" not in caller
    native = replacement(paths, outputs, report, "saffron_gate.f90")
    assert "entry fort_scope_entry_" not in native
