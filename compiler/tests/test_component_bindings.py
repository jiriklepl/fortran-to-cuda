"""Fixed source fields retain their original object and lexical alias owner."""

import pytest
from copy import copy
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.component_bindings import component_access, references, source_scope_for
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def analysis(tmp_path, source):
    path = tmp_path / "fields.f90"
    path.write_text(source)
    return SourceEffects([path])


def test_native_array_metadata_proofs_do_not_enable_numerical_captures(tmp_path):
    effects = analysis(tmp_path, '''module objects
type item
integer::coordinate
real(8)::weight
end type
type(item),allocatable,target::table(:)
contains
subroutine step(a,n)
integer,intent(in)::n
real(8),intent(inout)::a(n)
integer::i
do i=1,size(table)
a(table(i)%coordinate)=a(table(i)%coordinate)+table(i)%weight
enddo
end subroutine
end module''')
    routine = effects.routines['objects::step']
    node = next(iter(walk(routine.execution, F.Data_Ref)))
    with pytest.raises(CompilationError, match='scalar variable object'):
        component_access(effects, routine.scope, node)
    native = copy(effects)
    native._native_metadata = True
    selected = component_access(native, routine.scope, node)
    assert selected.binding.rank == 0
    assert selected.root_binding is routine.scope.parent.bindings['table']
    assert selected.binding.native_metadata_object.rank == 1
    assert native._summary_authority() != effects._summary_authority()
    with pytest.raises(CompilationError, match='scalar variable object'):
        component_access(effects, routine.scope, node)


def test_fixed_fields_and_unsupported_siblings_are_separate(tmp_path):
    effects = analysis(tmp_path, """module objects
type nested
real(8)::rate
end type
type state
type(nested)::inner
real(8)::window(-2:2)
real(8),allocatable::unrelated(:)
end type
type(state)::storage
contains
subroutine step(a)
real(8),intent(out)::a(5)
a=storage%window
a(1)=storage%inner%rate
end subroutine
end module
""")
    routine = effects.routines["objects::step"]
    accesses = [component_access(effects, routine.scope, item) for item in walk(routine.execution, F.Data_Ref)]
    window, rate = accesses
    assert window.binding.root == "objects::storage%window"
    assert window.binding.signature() == ("real", 8, 1)
    assert window.binding.lower_bounds == ("-2",)
    assert rate.binding.root == "objects::storage%inner%rate"
    assert rate.binding.rank == 0
    with pytest.raises(CompilationError, match="association or lifetime"):
        component_access(effects, routine.scope, F.Data_Ref("storage%unrelated"))
    found = {binding.root for binding, _ in references(effects, routine.scope, routine.execution)}
    assert found == {"argument::a", "objects::storage%window", "objects::storage%inner%rate"}


def test_imported_types_and_original_intent_are_preserved(tmp_path):
    kinds = tmp_path / "model.f90"
    kinds.write_text("""module model
type parameters
real(8)::dt
end type
end module
""")
    caller = tmp_path / "application.f90"
    caller.write_text("""module application
use model,only:options=>parameters
contains
subroutine step(config,a)
type(options),intent(in)::config
real(8),intent(out)::a
a=config%dt
end subroutine
end module
""")
    effects = SourceEffects([caller, kinds])
    routine = effects.routines["application::step"]
    access = component_access(effects, routine.scope, next(iter(walk(routine.execution, F.Data_Ref))))
    assert access.binding.root == "argument::config%dt"
    assert access.binding.intent == "in"
    assert access.root_binding.name == "config"


def test_nested_private_field_cannot_leak_through_public_container(tmp_path):
    module = tmp_path / "hidden.f90"
    module.write_text("""module hidden
type inner
private
real(8)::secret
end type
type outer
type(inner)::child
end type
type(outer)::container
end module
""")
    caller = tmp_path / "consumer.f90"
    caller.write_text("""module consumer
use hidden
contains
subroutine step(a)
real(8)::a
a=container%child%secret
end subroutine
end module
""")
    effects = SourceEffects([module, caller])
    routine = effects.routines["consumer::step"]
    with pytest.raises(CompilationError, match="private"):
        component_access(effects, routine.scope, next(iter(walk(routine.execution, F.Data_Ref))))


@pytest.mark.parametrize('private', [False, True])
def test_renamed_object_keeps_its_unimported_type_authority(tmp_path, private):
    module = tmp_path/'model.f90'
    module.write_text('''module model
private
type original_type
''' + ('private\n' if private else '') + '''real(8)::rate
end type
type(original_type),public::state
end module
module relay
use model,only:forwarded=>state
end module
''')
    caller = tmp_path/'consumer.f90'
    caller.write_text('''module consumer
use relay,only:selected=>forwarded
type original_type
integer::rate
end type
contains
subroutine step(a)
real(8)::a
a=selected%rate
end subroutine
end module
''')
    effects = SourceEffects([caller, module])
    routine = effects.routines['consumer::step']
    selector = next(iter(walk(routine.execution, F.Data_Ref)))
    if private:
        with pytest.raises(CompilationError, match='private'):
            component_access(effects, routine.scope, selector)
    else:
        access = component_access(effects, routine.scope, selector)
        assert access.binding.root == 'model::state%rate'
        assert access.binding.signature() == ('real', 8, 0)


def test_associate_aliases_are_lexical_and_do_not_rebind_outer_names(tmp_path):
    effects = analysis(tmp_path, """module aliases
type settings
real(8)::dt
end type
type(settings)::first,second
real(8)::dt
contains
subroutine step(a)
real(8)::a
associate(config=>first)
a=config%dt
associate(config=>second)
a=a+config%dt
end associate
a=a+config%dt
end associate
a=a+dt
end subroutine
end module
""")
    routine = effects.routines["aliases::step"]
    found = [binding.root for binding, node in references(effects, routine.scope, routine.execution)
             if type(node).__name__ == "Data_Ref"]
    assert found == ["aliases::first%dt", "aliases::second%dt", "aliases::first%dt"]
    assert effects._binding(routine.scope, "config") is None
    final = list(walk(routine.execution, F.Assignment_Stmt))[-1]
    assert source_scope_for(effects, final) is routine.scope
    forged = F.Assignment_Stmt("a=config%dt")
    forged.fort_source_origin = list(walk(routine.execution, F.Assignment_Stmt))[0]
    assert source_scope_for(effects, forged) is None


def test_unsupported_associate_is_a_local_boundary(tmp_path):
    effects = analysis(tmp_path, """module local_boundaries
contains
subroutine step(a,x)
real(8)::a(4),x
associate(value=>a(1))
x=value
end associate
x=x+1.d0
end subroutine
end module
""")
    routine = effects.routines["local_boundaries::step"]
    associate = next(iter(walk(routine.execution, F.Associate_Construct)))
    record = effects._associate_scopes[id(associate)]
    assert not record.available
    assert "original scalar variable selector" in record.reason
    final = list(walk(routine.execution, F.Assignment_Stmt))[-1]
    assert source_scope_for(effects, final) is routine.scope


@pytest.mark.parametrize("alias", ["selected=>config", "selected=>config%offset"])
def test_public_scope_keeps_field_consumers_and_lexical_aliases_between_gpu_calls(tmp_path, alias):
    from compiler.tests.test_source_scopes import FACT, PROGRAM, generate
    source = PROGRAM.replace("implicit none", """implicit none
type controls
real(8)::offset
real(8),allocatable::unused(:)
end type
type(controls)::config
""", 1)
    source = source.replace("call transform(b)", """associate(selected=>config)
if(selected%offset>0.d0) then
b(1)=b(1)+selected%offset
endif
end associate""")
    source = source.replace("real(8),intent(out)::b(:)", "real(8),intent(inout)::b(:)", 1)
    if alias.endswith("%offset"):
        source = source.replace("selected=>config", alias).replace("selected%offset", "selected")
    facts = {"schema_version": 1, "participation": "serial", "captures": {
        "argument::a": FACT, "argument::b": FACT, "argument::out": FACT}}
    original, output, manifest = generate(tmp_path, source, facts=facts)
    assert original.read_text() == source
    scope, = manifest["scopes"]
    text = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    owner = text[text.index("subroutine " + scope["owner"]):]
    normalized = "".join(owner.lower().split())
    assert "selected%offset" not in normalized
    assert "config%offset" in normalized
    # Reached alias bodies use canonical storage. The complete original native
    # fallback can retain its original association construct.
    assert "lexical_association" in str(scope["structured_tree"])
    assert scope["ownership"]["close_count"] == 1
    assert [node["kind"] for node in scope["structured_tree"]["nodes"]] == [
        "planning_segment", "lexical_association", "planning_segment"]
    assert scope["structured_tree"]["nodes"][1]["proof"]["available"]
    assert {item["resource"] for item in scope["original_scalar_bindings"]} == {"original::config%offset"}
