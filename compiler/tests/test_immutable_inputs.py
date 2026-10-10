"""Immutable array forwarding preserves original reached element evaluation."""

import copy
import re
from dataclasses import replace
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.immutable_inputs import immutable_call_inputs
from compiler.scopes.regions import extract_region
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT


def source_fixture(tmp_path, *, precision=8, index="stage", coefficient="weights",
                   declaration=None, body=None, caller_actual="renamed", extra="", bounds="-1:1", module_spec=""):
    tmp_path.mkdir(parents=True, exist_ok=True)
    constants = tmp_path / "constants.f90"
    constants.write_text(f"""module coefficient_values
implicit none
real({precision}),parameter::weights(-2:0)=[acos(-1._{precision}),-0._{precision},2._{precision}]
end module
""")
    child = tmp_path / "child.f90"
    declaration = declaration or f"real({precision}),intent(in)::weights({bounds})"
    body = body or f"""do j=1,n
out(j)=a(j)*{coefficient}({index})+{coefficient}({index})
enddo"""
    child.write_text(f"""module update_lib
implicit none
{module_spec}
contains
subroutine update(a,out,n,stage,weights)
real({precision}),intent(in)::a(-2:)
real({precision}),intent(inout)::out(-2:)
integer,intent(in)::n,stage
{declaration}
integer::j
{extra}
{body}
end subroutine
end module
""")
    owner = tmp_path / "owner.f90"
    owner.write_text(f"""module owner_lib
use coefficient_values,only:renamed=>weights
use update_lib,only:renamed_update=>update
implicit none
contains
subroutine advance(a,out,n,stage)
real({precision}),intent(in)::a(-2:)
real({precision}),intent(inout)::out(-2:)
integer,intent(in)::n,stage
call renamed_update(weights={caller_actual},a=a,out=out,n=n,stage=stage)
end subroutine
end module
""")
    paths = constants, child, owner
    analysis = SourceEffects(paths)
    caller, routine = analysis.routines["owner_lib::advance"], analysis.routines["update_lib::update"]
    call = next(node for node in walk(caller.execution) if _kind(node) == "Call_Stmt")
    resolved = resolve_source_call(analysis, caller.scope, call)
    associations = immutable_call_inputs(analysis, caller, call, resolved)
    return analysis, caller, routine, call, associations, paths


def region_fixture(tmp_path, **options):
    analysis, caller, routine, call, associations, paths = source_fixture(tmp_path, **options)
    loop = next(node for node in walk(routine.execution) if _kind(node) == "Block_Nonlabel_Do_Construct")
    region = extract_region(analysis, routine, loop, immutable_inputs=associations)
    return analysis, routine, associations, region, paths


def emit(paths, *, policy="sections"):
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {"argument::" + name: FACT for name in ("a", "out")}}
    return ScopeBuilder(paths, "owner_lib::advance", facts=facts, options=CompilerOptions(),
                        config=OffloadConfig(policy=policy, scope_execution="reached")).run()


@pytest.mark.parametrize("precision", [4, 8])
def test_original_parameter_array_is_an_association_not_a_binding_capture(tmp_path, precision):
    analysis, routine, inputs, region, _ = region_fixture(tmp_path, precision=precision)
    association, = inputs.values()
    assert association.formal is routine.scope.bindings["weights"]
    assert association.actual is analysis.modules["coefficient_values"].bindings["weights"]
    assert association.actual_bounds == (-2, 0)
    assert association.formal_bounds == (-1, 1)
    assert "parameter" in association.actual.attributes
    assert "target" not in association.actual.attributes
    assert region.immutable_array_reads == ("argument::weights",)
    assert "argument::weights" not in {item.root for item in region.bindings}
    assert "argument::stage" not in {item.root for item in region.bindings}
    element, = region.scalar_element_captures
    assert len(element.original_references) == 2
    assert element.expression == "weights(stage)"
    assert element.kind == precision
    assert element.activation_guards == ("(n) >= (1)",)
    assert element.index_guard == "(stage) >= (-1) .and. (stage) <= (1)"
    assert element.name in region.source
    assert "weights(stage)" not in region.source.lower()
    assert "acos" not in region.source.lower()
    assert "weights(" not in region.source.lower()
    assert any(item.resource == element.resource and not item.rank for item in region.parameters)
    assert region.public()["scalar_element_captures"][0]["query"].startswith("typed zero")


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("coefficient_first", [False, True])
def test_grouped_managed_array_declaration_does_not_add_target_to_immutable_formal(
        tmp_path, precision, coefficient_first):
    _, _, _, _, _, paths = source_fixture(tmp_path, precision=precision)
    child = paths[1]
    entities = ["weights(-1:1)", "a(-2:)"] if coefficient_first else ["a(-2:)", "weights(-1:1)"]
    grouped = f"real({precision}),intent(in)::" + ",".join(entities)
    child.write_text(child.read_text().replace(f"real({precision}),intent(in)::a(-2:)\n", "")
                     .replace(f"real({precision}),intent(in)::weights(-1:1)", grouped))
    original = child.read_bytes()
    outputs, report = emit(paths)
    assert report["scope_count"] == 1
    owner, = report["scopes"]
    companion, = owner["module_coordinators"]
    assert companion["immutable_array_inputs"][0]["formal_resource"] == "argument::weights"
    text = outputs[report["sources"][str(child)]["replacement"]]
    declarations = [line for line in text.splitlines() if "::" in line]
    coefficient_declaration, = [line for line in declarations if re.search(r"\bweights\s*\(", line, re.I)]
    managed_declaration, = [line for line in declarations if re.search(r"\ba\s*\(", line, re.I)]
    assert "target" not in coefficient_declaration.lower()
    assert "target" in managed_declaration.lower()
    assert "intent(in)" in coefficient_declaration.lower().replace(" ", "")
    assert (text.index(coefficient_declaration) < text.index(managed_declaration)) == coefficient_first
    assert child.read_bytes() == original


def test_original_outer_guard_is_preserved_and_empty_inner_domain_not_speculated(tmp_path):
    body = """if(stage>1) then
do j=1,n
out(j)=a(j)*weights(stage)
enddo
endif"""
    _, _, _, region, _ = region_fixture(tmp_path, body=body)
    assert region.scalar_element_captures[0].expression == "weights(stage)"
    # Extraction selects the original reached loop, not its guarding IF.
    assert region.source.count("IF") == 0


def test_ordered_activation_checks_are_not_flat_short_circuit_assumptions(tmp_path):
    body = """do j=1,n
do k=1,inner_n
out(j,k)=a(j,k)*weights(stage)
enddo
enddo"""
    analysis, _, routine, _, inputs, _ = source_fixture(tmp_path, body=body)
    # Build a separate valid two-dimensional original fixture, preserving source
    # authority rather than mutating its parsed declarations for this test.
    child = tmp_path / "child.f90"
    text = child.read_text().replace("a(-2:)", "a(-2:,-2:)").replace("out(-2:)", "out(-2:,-2:)")
    text = text.replace("n,stage,weights)", "n,stage,weights,inner_n)").replace("n,stage\n", "n,stage,inner_n\n")
    text = text.replace("integer::j", "integer::j,k")
    child.write_text(text)
    owner = tmp_path / "owner.f90"
    owner.write_text(owner.read_text().replace("a(-2:)", "a(-2:,-2:)").replace("out(-2:)", "out(-2:,-2:)")
                     .replace("n,stage)", "n,stage,inner_n)").replace("n,stage\n", "n,stage,inner_n\n")
                     .replace("n=n,stage=stage)", "n=n,stage=stage,inner_n=inner_n)"))
    analysis = SourceEffects((tmp_path / "constants.f90", child, owner))
    caller, routine = analysis.routines["owner_lib::advance"], analysis.routines["update_lib::update"]
    call = next(node for node in walk(caller.execution) if _kind(node) == "Call_Stmt")
    inputs = immutable_call_inputs(analysis, caller, call, resolve_source_call(analysis, caller.scope, call))
    loop = next(node for node in walk(routine.execution) if _kind(node) == "Block_Nonlabel_Do_Construct")
    region = extract_region(analysis, routine, loop, immutable_inputs=inputs)
    assert region.scalar_element_captures[0].activation_guards == ("(n) >= (1)", "(inner_n) >= (1)")


@pytest.mark.parametrize(("body", "reason"), [
    ("do j=1,n\nout(j)=a(j)*weights(j)\nenddo", "unchanged required INTENT(IN)"),
    ("do j=1,n\nif(a(j)>0) out(j)=weights(stage)\nenddo", "unconditional reached"),
    ("do j=n,1,-1\nout(j)=a(j)*weights(stage)\nenddo", "positive stride"),
    ("do j=1,n\nout(j)=a(j)*sum(weights)\nenddo", "explicit scalar element"),
    ("call opaque(stage)\ndo j=1,n\nout(j)=a(j)*weights(stage)\nenddo", "change or escape"),
])
def test_unproved_payload_activation_or_index_remains_native(tmp_path, body, reason):
    with pytest.raises(CompilationError, match=re.escape(reason)):
        region_fixture(tmp_path, body=body)


@pytest.mark.parametrize("declaration", [
    "real(8),intent(in)::weights(:)",
    "real(8),intent(in),optional::weights(-1:1)",
    "real(8),intent(in),target::weights(-1:1)",
    "real(8),intent(in)::weights(-1:2)",
    "real(8),intent(inout)::weights(-1:1)",
])
def test_only_required_identical_fixed_extent_immutable_associations(tmp_path, declaration):
    with pytest.raises(CompilationError):
        source_fixture(tmp_path, declaration=declaration)


def test_copied_call_and_foreign_parameter_binding_do_not_grant_authority(tmp_path):
    analysis, caller, routine, call, inputs, _ = source_fixture(tmp_path)
    association, = inputs.values()
    with pytest.raises(CompilationError, match="original call"):
        replace(association, call=copy.copy(call)).validate(analysis, routine)
    fake = copy.copy(association.actual)
    with pytest.raises(CompilationError, match="original PARAMETER actual"):
        replace(association, actual=fake).validate(analysis, routine)
    assert inputs["argument::weights"].actual is association.actual
    assert caller.scope.bindings["stage"].intent == "in"


def test_stale_source_cannot_reuse_association(tmp_path):
    analysis, _, routine, _, inputs, paths = source_fixture(tmp_path)
    paths[0].write_text(paths[0].read_text().replace("2._8", "3._8"))
    with pytest.raises(CompilationError):
        inputs["argument::weights"].validate(analysis, routine)


def test_integer_parameter_index_uses_original_declaring_scope(tmp_path):
    _, _, _, region, _ = region_fixture(tmp_path, index="selected", extra="integer,parameter::selected=0")
    element, = region.scalar_element_captures
    assert element.expression == "weights(0)"
    assert element.index_guard == "(0) >= (-1) .and. (0) <= (1)"
    assert "selected" not in region.source.lower()


def test_query_and_dispatch_do_not_read_parameter_payload(tmp_path):
    _, _, _, _, _, paths = source_fixture(tmp_path)
    outputs, report = emit(paths)
    assert report["scope_count"] == 1
    owner, = report["scopes"]
    child, = owner["module_coordinators"]
    assert child["immutable_array_inputs"][0]["actual_resource"] == "coefficient_values::weights"
    assert child["gpu_leaves"] if "gpu_leaves" in child else child["planning_segments"]
    text = "\n".join(value for name, value in outputs.items() if name.startswith("sources/")).lower()
    assert "c_loc(renamed" not in text
    assert "c_loc(weights" not in text
    assert "target :: weights" not in text
    assert "weights = renamed" in text
    assert "weights(stage)" in text
    child_text = text[text.index("module update_lib"):]
    assert child_text.index("fort_elements_active_") < child_text.index("is_contiguous(")
    query_calls = re.findall(r"fort_status\s*=\s*fort_inline_query\((.*?)\)", text.replace("&", ""), re.S)
    assert query_calls
    assert all("weights" not in call and "stage" not in call for call in query_calls)
    assert all("0.0_c_double" in call for call in query_calls)
    assert "if ((n) >= (1)) then" in text
    assert "if ((stage) >= (-1) .and. (stage) <= (1)) then" in text


def test_preparation_cost_is_unavailable_and_auto_keeps_original_native_body(tmp_path):
    _, _, _, _, _, paths = source_fixture(tmp_path)
    outputs, report = emit(paths, policy="auto")
    owner, = report["scopes"]
    assert owner["automatic_preflight"]["selection"] == "native"
    assert owner["automatic_preflight"]["caller_source_unchanged"]
    assert "immutable element preparation" in owner["planning_reason"]
    assert not any(name.startswith("regions/") for name in outputs)
    text = "\n".join(value for name, value in outputs.items() if name.startswith("sources/")).lower()
    assert "fort_scope_create" not in text
    assert "call renamed_update(weights=renamed,a=a,out=out,n=n,stage=stage)" in text


def test_same_child_different_parameter_roots_does_not_reuse_association(tmp_path):
    _, _, _, _, _, paths = source_fixture(tmp_path)
    constants, _, owner = paths
    constants.write_text(constants.read_text().replace("end module", "real(8),parameter::other(-2:0)=[1._8,2._8,3._8]\nend module"))
    owner.write_text(owner.read_text().replace("only:renamed=>weights", "only:renamed=>weights,second=>other")
                     .replace("end subroutine", "call renamed_update(weights=second,a=a,out=out,n=n,stage=stage)\nend subroutine"))
    _, report = emit(paths)
    owner_record, = report["scopes"]
    assert any("one canonical mapping" in item["reason"] for item in owner_record["boundaries"])
    child, = owner_record["module_coordinators"]
    assert child["immutable_array_inputs"][0]["actual_resource"] == "coefficient_values::weights"


def test_module_extent_is_read_at_exact_reached_region(tmp_path):
    _, _, _, region, _ = region_fixture(tmp_path, module_spec="integer::component_count=4",
        body="do j=1,component_count\nout(j)=a(j)*weights(stage)\nenddo")
    element, = region.scalar_element_captures
    assert element.activation_guards == ("(component_count) >= (1)",)
    assert "update_lib::component_count" in element.dependencies
    assert "update_lib::component_count" in {binding.root for binding in region.bindings}


def test_module_value_cannot_be_used_as_an_unproved_coefficient_index(tmp_path):
    with pytest.raises(CompilationError, match=re.escape("unchanged required INTENT(IN)")):
        region_fixture(tmp_path, module_spec="integer::component_count=0", index="component_count")


@pytest.mark.parametrize("body", [
    "do j=1,component_count\ncomponent_count=component_count-1\nout(j)=a(j)*weights(stage)\nenddo",
    "do j=1,component_count\ncall pure_read(a(j))\nout(j)=a(j)*weights(stage)\nenddo",
])
def test_module_extent_changes_or_calls_are_not_invariant(tmp_path, body):
    with pytest.raises(CompilationError):
        region_fixture(tmp_path, module_spec="integer::component_count=4", body=body)


def test_outer_work_cannot_be_skipped_using_an_inner_capture_domain(tmp_path):
    body = """do j=1,n
out(j)=a(j)
do k=1,2
out(j)=out(j)*weights(stage)
enddo
enddo"""
    with pytest.raises(CompilationError, match="sibling or outer-body"):
        region_fixture(tmp_path, extra="integer::k", body=body)


def test_multiple_capture_domains_cannot_authorize_whole_region_skip(tmp_path):
    body = """do j=1,n
out(j)=a(j)*weights(stage)
do k=1,2
out(j)=out(j)*weights(stage)
enddo
enddo"""
    with pytest.raises(CompilationError, match="sibling or outer-body"):
        region_fixture(tmp_path, extra="integer::k", body=body)


def test_original_native_parallel_region_does_not_acquire_element_preparation(tmp_path):
    body = """!$omp parallel do private(j)
do j=1,n
out(j)=a(j)*weights(stage)
enddo
!$omp end parallel do"""
    analysis, _, routine, _, inputs, _ = source_fixture(tmp_path, body=body)
    with pytest.raises(CompilationError):
        extract_region(analysis, routine, tuple(_children(routine.execution)), immutable_inputs=inputs)


def test_immutable_association_can_forward_through_original_fixed_shape_formal(tmp_path):
    analysis, caller, _, call, inputs, paths = source_fixture(tmp_path)
    owner = paths[-1]
    owner.write_text(owner.read_text().replace("use update_lib,only:renamed_update=>update", "use relay_lib,only:renamed_update=>relay"))
    relay = tmp_path / "relay.f90"
    relay.write_text("""module relay_lib
use update_lib,only:update
implicit none
contains
subroutine relay(a,out,n,stage,weights)
real(8),intent(in)::a(-2:),weights(1:3)
real(8),intent(inout)::out(-2:)
integer,intent(in)::n,stage
call update(a=a,out=out,n=n,stage=stage,weights=weights)
end subroutine
end module
""")
    analysis = SourceEffects((*paths[:2], relay, owner))
    caller = analysis.routines["owner_lib::advance"]
    call = next(node for node in walk(caller.execution) if _kind(node) == "Call_Stmt")
    first = immutable_call_inputs(analysis, caller, call, resolve_source_call(analysis, caller.scope, call))
    routine = analysis.routines["relay_lib::relay"]
    inner_call = next(node for node in walk(routine.execution) if _kind(node) == "Call_Stmt")
    second = immutable_call_inputs(analysis, routine, inner_call, resolve_source_call(analysis, routine.scope, inner_call), first)
    proof, = second.values()
    assert proof.parent is first["argument::weights"]
    assert proof.actual.root == "coefficient_values::weights"
    assert proof.original_actual is routine.scope.bindings["weights"]
    assert proof.formal_bounds == (-1, 1)
    proof.validate(analysis)
