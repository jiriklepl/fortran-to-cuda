"""Declaration-only record aliases do not authorize values or component writes."""
import inspect
import json
from hashlib import sha256

import pytest
from fparser.two import Fortran2003
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.component_bindings import ComponentStorageUnsupported
from compiler.frontend.descriptor_associations import original_fixed_association
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = """module parameters
implicit none
integer::Unx=2,Uny=2,Vnx=2,Vny=2,Wnx=2,Wny=2
end module

module forcing
use parameters
implicit none
real(8),allocatable::u(:,:,:),v(:,:,:),w(:,:,:),output(:,:,:)
real(8)::mean(3)=0
real(8),allocatable::u2(:,:,:),v2(:,:,:),w2(:,:,:)
  type damping_properties
    real(8) :: bottom, top !global constants for all images
    real(8), dimension(:), allocatable :: coefficient(:)
    real(8), dimension(:), allocatable :: coefficientW(:)
    integer :: kbottom = 1, ktop = 0
    integer :: kWbottom = 1, kWtop = 0
    real(8), dimension(:), allocatable :: u_prof, v_prof, w_prof, temperature_prof, moisture_prof
  end type

  type(damping_properties) :: profile

contains
subroutine advance(n,go,masked)
integer,intent(in)::n
logical,intent(in)::go,masked
call middle(n,go,masked)
end subroutine
subroutine middle(n,go,masked)
integer,intent(in)::n
logical,intent(in)::go,masked
real(8)::tke
tke=1
call producer(u2,v2,w2,n)
if(go)call rayleigh_damping_top_uvw(u2,v2,w2,u,v,w)
call consumer(output,u2,v2,w2,n)
end subroutine

subroutine producer(a,b,c,n)
integer,intent(in)::n
real(8),intent(inout)::a(-2:,-2:,-2:),b(-2:,-2:,-2:),c(-2:,-2:,-2:)
integer::i,j,k
do k=1,n
do j=1,n
do i=1,n
a(i,j,k)=real(i,8)
b(i,j,k)=real(j,8)
c(i,j,k)=real(k,8)
enddo
enddo
enddo
end subroutine

subroutine consumer(a,b,c,d,n)
integer,intent(in)::n
real(8),intent(inout)::a(-2:,-2:,-2:)
real(8),intent(in)::b(-2:,-2:,-2:),c(-2:,-2:,-2:),d(-2:,-2:,-2:)
integer::i,j,k
do k=1,n
do j=1,n
do i=1,n
a(i,j,k)=b(i,j,k)+c(i,j,k)+d(i,j,k)
enddo
enddo
enddo
end subroutine

  subroutine rayleigh_damping_top_uvw(U2, V2, W2, U, V, W)
    !an alternative formulation of the top sponge as a tendency
    real(8), contiguous, intent(inout) :: U2(-2:,-2:,-2:), V2(-2:,-2:,-2:), W2(-2:,-2:,-2:)
    real(8), contiguous, intent(in) :: U(-2:,-2:,-2:), V(-2:,-2:,-2:), W(-2:,-2:,-2:)
    integer :: i, j, k

    associate(prop => profile)

      do k = prop%kbottom, prop%ktop
        do j = 1, Uny
          do i = 1, Unx
            U2(i,j,k) = U2(i,j,k) - prop%coefficient(k) * (U(i,j,k) - prop%u_prof(k))
          end do
        end do
      end do

      do k = prop%kbottom, prop%ktop
        do j = 1, Vny
          do i = 1, Vnx
            V2(i,j,k) = V2(i,j,k) - prop%coefficient(k) * (V(i,j,k) - prop%v_prof(k))
          end do
        end do
      end do

      do k = prop%kWbottom, prop%kWtop
        do j = 1, Wny
          do i = 1, Wnx
            W2(i,j,k) = W2(i,j,k) - prop%coefficientW(k) * (W(i,j,k) - prop%w_prof(k))
          end do
        end do
      end do

    end associate

  end subroutine


end module
"""


def fixture(directory, source=SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "damping.f90"
    path.write_text(source)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"forcing::" + name: dict(fact) for name in ("u", "v", "w", "u2", "v2", "w2", "output")}}
    builder = ScopeBuilder([path], "forcing::advance", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"),
        contracts={"forcing::advance": {"native_ancestor": {"version": 1,
            "device_roots": ["forcing::producer", "forcing::consumer"]}}})
    return path, builder


def storage(directory, source=SOURCE):
    path, builder = fixture(directory, source)
    routine = builder.analysis.routines["forcing::middle"]
    selected, = [node for node in walk(routine.execution) if type(node).__name__ == "If_Stmt"]
    proof = prove_storage_mutation_effects(builder.analysis, routine.qualified, selection=(selected,))
    return path, builder, selected, proof


def association(builder):
    routine = builder.analysis.routines["forcing::rayleigh_damping_top_uvw"]
    node, = [node for node in walk(routine.execution) if type(node).__name__ == "Associate_Construct"]
    proof = original_fixed_association(builder.analysis, routine, node, declaration_only=True)
    return routine, node, proof


def test_owned_allocatable_record_alias_is_declaration_only(tmp_path):
    _path, builder, _selected, proof = storage(tmp_path)
    assert proof._record["complete"]
    assert not proof._record["frontiers"]
    assert not proof._record["escapes"]
    routine, node, alias = association(builder)
    assert alias.validate() is alias
    public = alias.public()
    assert public["kind"] == "original_readonly_scalar_record_lexical_declaration"
    assert public["declaration_only"]
    assert all(public[key] is False for key in ("layout_authorized", "capture_authorized",
        "payload_access_authorized", "entry_definition_authority", "descriptor_lifetime_authorized",
        "execution_authorized", "alias_separation_authorized"))
    with pytest.raises(CompilationError, match="readonly lexical declaration role"):
        original_fixed_association(builder.analysis, routine, node)
    assert "forcing::profile" not in builder.facts["captures"]


@pytest.mark.parametrize("guard", ["go", ".false.", ".true."])
def test_actual_guarded_damping_owner_and_strict_package_reader(tmp_path, guard):
    source = SOURCE.replace("if(go)call rayleigh", "if(" + guard + ")call rayleigh")
    _path, builder = fixture(tmp_path, source)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    boundary, = report["boundaries"]
    assert boundary["conditional_close"].lower() == guard
    assert boundary["lifecycle"] == "proved_lifecycle"
    assert boundary["reopen"]
    scope, = report["scopes"]
    assert all(row["resource"] != "forcing::profile" for row in scope["checked_execution_plan"]["storage_epoch"]["resources"])
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    manifest = json.loads(outputs["scope-manifest.json"])
    validate_checked_execution_plans(manifest)
    package = tmp_path / "package"
    for name, text in outputs.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    _scope_package_build({"artifact_root": str(package), "scope_manifest": manifest})


@pytest.mark.parametrize("change", ["selector", "declaration", "role", "source", "control"])
def test_original_alias_declaration_source_and_control_are_revalidated(tmp_path, change):
    path, builder, selected, proof = storage(tmp_path)
    _routine, node, alias = association(builder)
    root = builder.analysis.modules["forcing"].bindings["profile"]
    if change == "selector":
        item = node.content[0].items[1].items[0]
        object.__setattr__(item, "items", (item.items[0], item.items[1], Fortran2003.Name("other")))
    elif change == "declaration":
        object.__setattr__(root, "attributes", frozenset({"target"}))
    elif change == "role":
        object.__setattr__(alias, "_declaration_only", False)
    elif change == "source":
        path.write_text(path.read_text() + "! changed original input\n")
    else:
        object.__setattr__(selected, "items", (Fortran2003.Name("other_guard"), selected.items[1]))
    def revalidate():
        if change == "role":
            return alias.validate()
        return proof.validate(builder.analysis, "forcing::middle", selection=(selected,))

    with pytest.raises(CompilationError):
        revalidate()


@pytest.mark.parametrize("effect", ["component_write", "allocate", "deallocate", "whole_component", "unknown_call",
                                    "escape", "pointer_root", "target_root", "pointer_field", "finalizer", "expression"])
def test_readonly_lexical_mode_does_not_waive_mutations_or_escapes(tmp_path, effect):
    source = SOURCE
    if effect == "pointer_root":
        source = source.replace("type(damping_properties) :: profile", "type(damping_properties),pointer :: profile")
    elif effect == "target_root":
        source = source.replace("type(damping_properties) :: profile", "type(damping_properties),target :: profile")
    elif effect == "pointer_field":
        source = source.replace("allocatable :: coefficient(:)", "pointer :: coefficient(:)")
    elif effect == "finalizer":
        source = source.replace("  end type", "  contains\n    final :: finalize_profile\n  end type", 1)
        source = source.replace("end module\n", "end module\n", 1)
        source = source.rsplit("end module", 1)[0] + "subroutine finalize_profile(value)\ntype(damping_properties)::value\nend subroutine\nend module\n"
    elif effect == "expression":
        source = source.replace("associate(prop => profile)", "associate(prop => profile_copy())")
        source = source.rsplit("end module", 1)[0] + "function profile_copy() result(value)\ntype(damping_properties)::value\nvalue=profile\nend function\nend module\n"
    else:
        statement = {"component_write": "prop%kbottom=1", "allocate": "allocate(prop%coefficient(1:2))",
            "deallocate": "deallocate(prop%coefficient)", "whole_component": "prop%u_prof=0",
            "unknown_call": "call unknown(prop)", "escape": "call retain(prop)"}[effect]
        source = source.replace("    end associate", statement + "\n    end associate")
        if effect == "escape":
            source = source.rsplit("end module", 1)[0] + "subroutine retain(value)\ntype(damping_properties),target,intent(inout)::value\ntype(damping_properties),pointer,save::kept\nkept=>value\nend subroutine\nend module\n"
    if effect in {"allocate", "deallocate"}:
        with pytest.raises(ComponentStorageUnsupported, match="association or lifetime is uncertain"):
            storage(tmp_path, source)
    else:
        _path, _builder, _selected, proof = storage(tmp_path, source)
        assert not proof._record["complete"], effect


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("change", ["members", "container"])
def test_final_publication_fence_covers_schema_after_last_callback(tmp_path, monkeypatch, empty, change):
    source = SOURCE
    if empty:
        start = source.index("    associate(prop => profile)") + len("    associate(prop => profile)")
        end = source.index("    end associate", start)
        source = source[:start] + "\ncontinue\n" + source[end:]
    _path, builder = fixture(tmp_path, source)
    analysis = builder.analysis
    routine = analysis.routines["forcing::middle"]
    selected, = [node for node in walk(routine.execution) if type(node).__name__ == "If_Stmt"]
    schema = analysis.modules["forcing"].component_types["damping_properties"]
    original = type(analysis.inputs).verify_live
    changed = []

    def late(inputs, *args, **kwargs):
        result = original(inputs, *args, **kwargs)
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name == "_publish_derivation" and not changed:
            if change == "members":
                entries = dict(schema.members)
                entries["foreign"] = entries["coefficient"]
                object.__setattr__(schema, "members", entries)
            else:
                object.__setattr__(schema, "members", dict(schema.members))
            changed.append(True)
        return result

    monkeypatch.setattr(type(analysis.inputs), "verify_live", late)
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(analysis, routine.qualified, selection=(selected,))
    assert changed


@pytest.mark.parametrize("change", ["role", "record"])
def test_final_publication_rechecks_declaration_role_after_source_callback(tmp_path, monkeypatch, change):
    from compiler.frontend.descriptor_associations import FixedScalarAssociation

    _path, builder = fixture(tmp_path)
    analysis = builder.analysis
    routine = analysis.routines["forcing::middle"]
    selected, = [node for node in walk(routine.execution) if type(node).__name__ == "If_Stmt"]
    original = type(analysis.inputs).verify_live
    changed = []

    def late(inputs, *args, **kwargs):
        result = original(inputs, *args, **kwargs)
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name == "_publish_derivation" and not changed:
            scanner = caller.f_locals["scanner"]
            proof, = [item[0] for item in scanner.fixed_associations.values()]
            assert type(proof) is FixedScalarAssociation
            attribute = "_declaration_only" if change == "role" else "_record"
            replacement = False if change == "role" else "{}"
            object.__setattr__(proof, attribute, replacement)
            changed.append(True)
        return result

    monkeypatch.setattr(type(analysis.inputs), "verify_live", late)
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(analysis, routine.qualified, selection=(selected,))
    assert changed
