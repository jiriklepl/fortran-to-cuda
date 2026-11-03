"""Original argumentless internal calls retain their actual storage namespace."""
import json
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.scopes.source_task_bindings import checked_call_inventory
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_same_family_native_call import source_fixture as base_source


def source_fixture(*, nested=True, native_call=True):
    text = base_source().replace(
        "real(8)::observed", "real(8)::weights(2)=[3.0_8,4.0_8]\nreal(8)::observed", 1)
    if not native_call:
        text = text.replace("call native_worker(left,right,n,scale)", "continue")
    if nested:
        text = text.replace("call consumer(left,right,n)\nend subroutine\nsubroutine advance", """call internal_helper
contains
subroutine internal_helper
call leaf(n)
end subroutine
end subroutine
subroutine advance""")
    else:
        text = text.replace("call consumer(left,right,n)", "call leaf(n)")
    leaf = """subroutine leaf(n)
integer,intent(in)::n
integer::j
observed=0
do j=1,2
observed=observed+weights(j)
enddo
call consumer(left,right,n)
end subroutine
"""
    return text.replace("end module", leaf + "end module")


def scalar_element_source():
    return (source_fixture()
        .replace("real(8)::observed", "real(8)::observed\nreal(8)::ratios(0:1)=[1.0_8,2.0_8]\ninteger::selector=0", 1)
        .replace("subroutine consumer(a,b,n)", "subroutine consumer(a,b,n,coefficient)")
        .replace("real(8),intent(in)::b(:)", "real(8),intent(in)::b(:)\nreal(8),intent(in)::coefficient")
        .replace("a(i)=2.0_8*b(i)", "a(i)=coefficient*b(i)")
        .replace("call consumer(left,right,n)", "call consumer(left,right,n,ratios(selector))"))


def test_nested_scalar_element_keeps_original_backing_reservation(tmp_path):
    _path, builder, _outputs, _report, manifest = emit_package(tmp_path, scalar_element_source())
    epoch = manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]
    resources = {row["resource"]: row for row in epoch["resources"]}
    assert resources["hierarchical::ratios"]["device_replica_authorized"] is False
    assert "hierarchical::ratios" not in builder.facts["captures"]


def emit_package(tmp_path, source=None):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    path, builder = fixture(tmp_path, source or source_fixture())
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    assert builder._conditional_return_draft is not None
    package = tmp_path / "package"
    for name, content in outputs.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    manifest = json.loads((package / "scope-manifest.json").read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({"artifact_root": str(package), "scope_manifest": manifest})
    assert builder.finish()
    return path, builder, outputs, report, manifest


@pytest.mark.parametrize(("nested", "native_call"), [(False, True), (True, False), (True, True)])
def test_final_exact_hierarchical_package(tmp_path, nested, native_call):
    _path, builder, _outputs, report, manifest = emit_package(
        tmp_path, source_fixture(nested=nested, native_call=native_call))
    scope, = manifest["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]
    returned = epoch["return_contract"]
    assert returned["root_selection"]["selected_node_ids"]
    assert returned["version"] == (4 if nested else 3)
    resources = {row["resource"]: row for row in epoch["resources"]}
    assert resources["hierarchical::weights"]["device_replica_authorized"] is False
    assert "hierarchical::weights" not in builder.facts["captures"]
    if nested:
        coordinator, = scope["internal_coordinators"]
        inherited = coordinator["lexical_namespace"]
        assert inherited["parent"] == inherited["namespace_owner"] == "hierarchical::middle"
        owner = builder._conditional_return_draft.owner
        helper = owner.members[coordinator["procedure"]]
        nodes, edges, _changed = checked_call_inventory(helper)
        assert nodes[helper.routine.qualified]["parameterized"] is False
        assert nodes[helper.routine.qualified]["inherited_namespace"] == inherited
        assert ("hierarchical::middle", "hierarchical::leaf") not in {
            (row["caller"], row["callee"]) for row in edges.values()}
    assert report["scope_count"] == 1


@pytest.mark.parametrize("change", ["copy_receipt", "record", "source_call", "resources", "canonical", "namespace", "missing", "patch", "foreign", "cycle", "slot"])
def test_original_lexical_receipt_cannot_be_replaced(tmp_path, change):
    _path, builder, _outputs, _report, _manifest = emit_package(tmp_path)
    owner = builder._conditional_return_draft.owner
    member = owner.members["hierarchical::middle"]
    receipt, = member.source_lexical_calls
    child = receipt.callee
    if change == "copy_receipt":
        list.__setitem__(member.source_lexical_calls, 0, replace(receipt))
    elif change == "record":
        list.__setitem__(member.internal_calls, 0, dict(receipt.record))
    elif change == "source_call":
        dict.__setitem__(receipt.record, "target", "other")
    elif change == "resources":
        object.__setattr__(child, "resources", dict(child.resources))
    elif change == "canonical":
        object.__setattr__(child, "canonical_bindings", dict(child.canonical_bindings))
    elif change == "namespace":
        object.__setattr__(child, "resource_slot_namespace", child)
    elif change == "missing":
        dict.clear(member._issued_lexical_calls)
    elif change == "foreign":
        object.__setattr__(child, "parent", owner)
    elif change == "cycle":
        object.__setattr__(child, "parent", child)
    elif change == "slot":
        binding, slot = next(iter(child.resources.values()))
        dict.__setitem__(child.resources, binding.root, (binding, slot + 1))
    else:
        index = next(i for i, row in enumerate(member.patches) if row[2] is receipt.patch)
        list.__setitem__(member.patches, index, (*member.patches[index][:2], receipt.patch + "! changed\n"))
    with pytest.raises(CompilationError):
        owner.return_contract.validate()


@pytest.mark.parametrize("attachment", ["node", "patch", "record", "callee"])
def test_active_emission_admission_fences_lexical_receipt_attachments(tmp_path, attachment):
    from compiler.scopes.storage_admission import StorageAdmission

    _path, builder, _outputs, _report, _manifest = emit_package(tmp_path)
    owner = builder._conditional_return_draft.owner
    receipt, = owner.members["hierarchical::middle"].source_lexical_calls
    plan = owner.execution_plan
    def mutate():
        with StorageAdmission(plan) as admission:
            object.__setattr__(receipt, attachment, None)
            admission.validate(plan.storage_epoch(), plan)

    with pytest.raises(CompilationError):
        mutate()


def test_unparameterized_module_lexical_child_keeps_existing_owner_registrations(tmp_path):
    from compiler.tests.test_lexical_source_owner import MODULE_SOURCE, emit

    source = (MODULE_SOURCE
        .replace("child_visits=child_visits+1", "call inner()", 1)
        .replace("end subroutine\nend module", """contains
subroutine inner()
child_visits=child_visits+1
end subroutine
end subroutine
end module""", 1))
    _path, _outputs, report = emit(tmp_path, source)
    assert report["scope_count"] == 1
    lexical, = report["scopes"][0]["internal_coordinators"]
    assert lexical["lexical_namespace"]["namespace_owner"] is None
    assert lexical["lexical_namespace"]["parent"] == "stage_child::adjust"


def test_varying_formal_associations_cross_original_lexical_namespace(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    from compiler.tests.test_atomic_source_call_bindings import SOURCE, atomic_builder

    source = SOURCE.replace("call helper(a,p,n,flag)\nend subroutine\nsubroutine step", """call internal_helper
contains
subroutine internal_helper
call helper(a,p,n,flag)
end subroutine
end subroutine
subroutine step""")
    builder = atomic_builder(tmp_path, source)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    scope, = report["scopes"]
    task, = scope["source_device_tasks"]
    projection = task["execution_binding"]["call_projection"]
    assert projection["version"] == 2
    assert projection["field_roots"] == {
        "argument::a": ["argument::a", "argument::b"],
        "argument::p": ["argument::p", "argument::q"],
    }
    lexical, = [row for row in projection["members"] if "inherited_namespace" in row]
    assert lexical["procedure"] == "atomic_calls::relay::internal_helper"
    assert lexical["parameterized"] is False
    assert len(projection["calls"]) == 3
    assert {(call["caller"], call["callee"]) for call in projection["calls"]} == {
        ("atomic_calls::step", "atomic_calls::relay"),
        ("atomic_calls::relay::internal_helper", "atomic_calls::helper"),
    }
    package = tmp_path / "package"
    for name, content in outputs.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    manifest = json.loads((package / "scope-manifest.json").read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({"artifact_root": str(package), "scope_manifest": manifest})
    assert builder.finish()


def test_active_admission_rejects_in_place_lexical_record_before_membership(tmp_path):
    from compiler.scopes.storage_admission import StorageAdmission

    _path, builder, _outputs, _report, _manifest = emit_package(tmp_path)
    owner = builder._conditional_return_draft.owner
    member = owner.members["hierarchical::middle"]
    receipt, = member.source_lexical_calls
    plan = owner.execution_plan
    queried = []

    def mutate():
        with StorageAdmission(plan) as admission:
            dict.__setitem__(receipt.record, "target", "forged")
            admission.validate(plan.storage_epoch(), plan)
            queried.append(True)

    with pytest.raises(CompilationError):
        mutate()
    assert queried == []


def test_original_continue_preserves_domain_and_graph_coverage(tmp_path):
    _path, builder, outputs, _report, manifest = emit_package(tmp_path, source_fixture(native_call=False))
    scope, = manifest["scopes"]
    owner = builder._conditional_return_draft.owner
    middle = owner.members["hierarchical::middle"]
    assert not middle.boundaries
    operation, = [scope for nodes, scope in middle.graph_native_tasks if str(nodes[0]).strip().lower() == "continue"]
    assert operation.native[0].effects == {}
    assert operation.native[0].summary["complete"]
    assert operation.native[0].summary["native_completion"]["available"]
    member, = [row for row in scope["checked_execution_plan"]["storage_epoch"]["return_contract"]["members"]
               if row["procedure"] == "hierarchical::middle"]
    assert any(route["kind"] == "checked_native" for route in member["routes"])
    assert "continue" in "\n".join(outputs.values()).lower()


def test_labelled_continue_cannot_establish_structured_owner_entry(tmp_path):
    path, builder = fixture(tmp_path, source_fixture(native_call=False).replace("continue", "100 continue", 1))
    _outputs, report = builder.run()
    assert path.read_text().count("100 continue") == 1
    assert all(row["procedure"] != "hierarchical::middle"
               for scope in report["scopes"] for row in scope["module_coordinators"])
    assert "structured entry" in str(report)
