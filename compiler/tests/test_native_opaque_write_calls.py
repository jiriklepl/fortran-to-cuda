"""Original fixed-array reductions keep opaque writes out of GPU storage."""

import gc
import json
import sys
import weakref
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_native_formal_array_reductions import CALLER, source


def call_source(*, guarded=True):
    text = source(guarded=guarded, repeated=False)
    text = text.replace("other(4:6)", "other(4:6), output(-2:3,-2:3,-2:3)")
    text = text.replace("logical,intent(in)::masked,go\n", """logical,intent(in)::masked,go
integer::i,j,k
do k=1,n
do j=1,n
do i=1,n
values(i,j,k)=real(i+j+k,8)
enddo
enddo
enddo
""", 1)
    text = text.replace("\n\nend subroutine", """
do k=1,n
do j=1,n
do i=1,n
output(i,j,k)=2.0_8*values(i,j,k)
enddo
enddo
enddo
end subroutine""", 1)
    return text


def source_builder(tmp_path, *, guarded=True, initialized="whole", text=None):
    path = tmp_path / "source.f90"
    path.write_text(call_source(guarded=guarded) if text is None else text)
    fact = {"storage": "stable", "initialized": initialized, "escapes": False,
            "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"formal_reduction::" + name: dict(fact) for name in ("values", "output")}}
    builder = ScopeBuilder([path], CALLER, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    return path, builder


def emit(tmp_path, **options):
    path, builder = source_builder(tmp_path, **options)
    outputs, report = builder.run()
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2))
    for name, data in outputs.items():
        target = tmp_path / "package" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data)
    return path, builder, outputs, json.loads(outputs["scope-manifest.json"])


@pytest.mark.parametrize("guarded", [False, True])
def test_normal_builder_keeps_original_native_write_without_capture(tmp_path, guarded):
    _path, builder, _outputs, report = emit(tmp_path, guarded=guarded)
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    owner, = report["scopes"]
    assert not owner["boundaries"]
    native = [operation for unit in owner["planning_segments"]
              for operation in unit["operations"]["native_operations"]
              if operation["native_opaque_accesses"]]
    assert len(native) == 1
    operation, = native
    assert operation["managed_resources"] == ["formal_reduction::values"]
    access, = operation["native_opaque_accesses"]["accesses"]
    assert access["resource"] == "formal_reduction::mean"
    assert access["mode"] == "RW"
    assert access["permit_kind"] == "original_fixed_array_write"
    assert "formal_reduction::mean" not in builder.facts["captures"]
    assert not access["permit"]["managed_definition_authorized"]
    assert not access["permit"]["declaration_initialization_credit"]
    assert access["execution_control"]["source_call"] in report["resolved_calls"]
    assert access["execution_control"]["selected_node_ids"] == operation["selected_original_nodes"]
    assert access["execution_control"]["original_guard"] == ("go" if guarded else None)
    assert operation["sections"]["available"] is False


def test_unavailable_sections_do_not_waive_remaining_managed_definitions(tmp_path):
    _path, _builder, _outputs, report = emit(tmp_path, initialized="none")
    assert any("whole managed access requires whole initialized" in boundary["reason"]
               for owner in report["scopes"] for boundary in owner["boundaries"])


def test_original_managed_write_keeps_read_and_write_host_hazards(tmp_path):
    text = call_source().replace("contiguous,intent(in)::values", "contiguous,intent(inout)::values")
    text = text.replace("result=result/max(1,n)", "result=result/max(1,n)\nvalues(-2,-2,-2)=values(-2,-2,-2)+1")
    _path, _builder, _outputs, report = emit(tmp_path, text=text)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    operation, = [operation for unit in owner["planning_segments"] for operation in unit["operations"]["native_operations"]
                  if operation["native_opaque_accesses"]]
    assert operation["resource_effects"]["formal_reduction::values"] == ["read", "write"]
    assert operation["managed_resources"] == ["formal_reduction::values"]


def test_managed_out_undefine_is_not_discharged_by_opaque_output(tmp_path):
    text = call_source().replace("contiguous,intent(in)::values", "contiguous,intent(out)::values")
    _path, _builder, _outputs, report = emit(tmp_path, text=text)
    assert any("cannot discharge a managed OUT definition" in boundary["reason"]
               for owner in report["scopes"] for boundary in owner["boundaries"])


def test_allocatable_three_dimensional_input_retains_exact_source_roles(tmp_path):
    text = call_source().replace(
        "real(8)::values(-2:3,-2:3,-2:3), mean(-1:1), other(4:6), output(-2:3,-2:3,-2:3)",
        "real(8)::mean(-1:1),other(4:6)\nreal(8),allocatable::values(:,:,:),output(:,:,:)")
    _path, _builder, _outputs, report = emit(tmp_path, text=text)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    operation, = [operation for unit in owner["planning_segments"] for operation in unit["operations"]["native_operations"]
                  if operation["native_opaque_accesses"]]
    access, = operation["native_opaque_accesses"]["accesses"]
    assert "allocatable" not in access["permit"]["descriptor"]["attributes"]
    mapping = next(row for row in access["execution_control"]["source_call"]["resource_mappings"] if row["formal"] == "values")
    assert mapping["actual_descriptor"]["attributes"] == ["allocatable"]
    assert mapping["formal_descriptor"]["lower_bounds"] == ["- 2"] * 3


def test_repeated_original_native_writes_keep_the_same_opaque_role(tmp_path):
    text = call_source().replace("if(go) call reduce_mean(mean,values,n,masked)",
        "if(go) call reduce_mean(mean,values,n,masked)\nif(go) call reduce_mean(mean,values,n,masked)")
    _path, _builder, _outputs, report = emit(tmp_path, text=text)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    operations = [operation for unit in owner["planning_segments"] for operation in unit["operations"]["native_operations"]
                  if operation["native_opaque_accesses"]]
    assert len(operations) == 2
    permits = [operation["native_opaque_accesses"]["accesses"][0]["permit"] for operation in operations]
    assert permits[0]["identity"] != permits[1]["identity"]
    assert permits[0]["selected_node_ids"] != permits[1]["selected_node_ids"]


def admitted_unit(tmp_path):
    _path, builder = source_builder(tmp_path)
    original = next(node for node in builder.entry.execution.content if type(node).__name__ == "If_Stmt")
    owner = LexicalOwner(builder)
    owner.scan((original,))
    assert not owner.boundaries
    unit, = owner.units
    operation, = unit.scope.native
    access, = operation.native_opaque_accesses.values()
    return owner, unit, operation, access


@pytest.mark.parametrize("fault", ["unit_guard", "unit_action", "tree", "nodes", "original_guard", "managed_mapping"])
def test_final_source_callback_cannot_rewrite_the_reached_route(tmp_path, monkeypatch, fault):
    _owner, unit, operation, access = admitted_unit(tmp_path)
    analysis = access.permit._analysis
    original = analysis.verify_publication
    reached = []

    def mutate():
        original()
        if sys._getframe(1).f_code.co_name != "_derive" or reached:
            return
        reached.append(True)
        if fault == "unit_guard":
            unit.source_guard = ".false."
        elif fault == "unit_action":
            unit.native_source = "continue"
        elif fault == "tree":
            unit.scope.tree = []
        elif fault == "nodes":
            operation.nodes = ()
        elif fault == "original_guard":
            operation.opaque_guard = None
        else:
            call = json.loads(operation.opaque_source_call)
            call["resource_mappings"][1]["resource"] = "formal_reduction::output"
            operation.opaque_source_call = json.dumps(call)
    monkeypatch.setattr(analysis, "verify_publication", mutate)
    with pytest.raises(CompilationError):
        unit.scope.validate_opaque_control(unit)
    assert reached == [True]


def test_final_attachment_fence_retains_replaced_values_strongly(tmp_path, monkeypatch):
    _owner, unit, operation, access = admitted_unit(tmp_path)

    class WireText(str):
        pass

    operation.opaque_source_call = WireText(operation.opaque_source_call)
    retained = weakref.ref(operation.opaque_source_call)
    analysis = access.permit._analysis
    original = analysis.verify_publication
    reached = []

    def mutate():
        original()
        if sys._getframe(1).f_code.co_name != "_derive" or reached:
            return
        operation.opaque_source_call = str(operation.opaque_source_call)
        gc.collect()
        reached.append(retained() is not None)
    monkeypatch.setattr(analysis, "verify_publication", mutate)
    with pytest.raises(CompilationError):
        access.validate(operation)
    assert reached == [True]


@pytest.mark.parametrize("managed_write", [False, True])
def test_emitted_guard_contains_every_native_preparation_and_full_managed_access(tmp_path, managed_write):
    from fparser.common.readfortran import FortranStringReader
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk

    text = call_source()
    if managed_write:
        text = text.replace("contiguous,intent(in)::values", "contiguous,intent(inout)::values")
        text = text.replace("result=result/max(1,n)", "result=result/max(1,n)\nvalues(-2,-2,-2)=values(-2,-2,-2)+1")
    _path, _builder, outputs, report = emit(tmp_path, text=text)
    scope, = report["scopes"]
    plan = scope["checked_execution_plan"]
    root = "formal_reduction::mean"
    reservation = next(r for r in plan["storage_epoch"]["resources"] if r["resource"] == root)
    assert reservation["storage"] == "opaque host reservation"
    assert reservation["device_replica_authorized"] is reservation["initialization_authorized"] is False
    for procedure in plan["procedures"]:
        for unit in procedure["units"]:
            assert root not in unit["resources"]
            assert all(root not in unit["ordered_definitions"][key] for key in
                       ("required_whole", "whole_on_return", "possible_definition_changes"))
    # Parse the final ordinary Fortran, independently of the compiler's private
    # route objects. Both the managed transaction and native fallback CALL must
    # remain below the single original condition, including opaque registration.
    generated, = [data for name, data in outputs.items() if name.startswith("sources/") and name.endswith(".f90")]
    parsed = ParserFactory().create(std="f2008")(FortranStringReader(generated))
    outer, = [node for node in walk(parsed) if type(node).__name__ == "Subroutine_Subprogram"
              and str(node.content[0].items[1]).lower() == "outer"]
    conditions = [node for node in walk(outer) if type(node).__name__ == "If_Then_Stmt"
                  and str(node.items[0]).lower() == "go"]
    assert len(conditions) == 1
    guard = conditions[0].parent
    reached = {id(node) for node in walk(guard)}
    calls = [node for node in walk(outer) if type(node).__name__ == "Call_Stmt"
             and str(node.items[0]).lower() == "reduce_mean"]
    assert len(calls) == 2  # Mutually exclusive prepared and safe native paths.
    assert all(id(call) in reached for call in calls)
    assignments = [node for node in walk(outer) if type(node).__name__ == "Assignment_Stmt"]
    opaque_pointers = [node for node in assignments if "c_loc(mean)" in str(node).lower()]
    assert opaque_pointers
    assert all(id(node) in reached for node in opaque_pointers)
    bindings = [node for node in assignments if "fort_native_task_bindings" in str(node).lower()]
    assert bindings
    assert all(id(node) in reached for node in bindings)
    accesses = [str(node).lower() for node in bindings if "%access" in str(node).lower().replace(" ", "")]
    assert len(accesses) == 1
    assert "fort_scope_read_all" in accesses[0]
    assert ("fort_scope_write_all" in accesses[0]) is managed_write
    opaque_id = reservation["registration_identity"]
    assert all("fort_buffer_" + str(opaque_id) not in str(node).lower() for node in bindings)
    stack = [scope["graph_execution"]]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if "transfers" in value:
                assert all(item["resource"] != root for item in value["transfers"])
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def test_original_call_presence_cannot_be_rewritten_with_matching_public_metadata(tmp_path):
    _owner, _unit, operation, access = admitted_unit(tmp_path)
    mapping = operation.opaque_call_mapping.mappings[1]
    object.__setattr__(mapping, "presence", "omitted")
    call = json.loads(operation.opaque_source_call)
    call.update(operation.opaque_call_mapping.public())
    operation.opaque_source_call = json.dumps(call)
    with pytest.raises(CompilationError, match="original managed argument mapping"):
        access.validate(operation)
