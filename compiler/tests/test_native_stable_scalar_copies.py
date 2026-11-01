"""Physical preparation may follow unchanged original scalar entry copies."""

import pytest

from compiler.frontend.source_effects import SourceEffects


def analyze(tmp_path, body, *, declarations="", module="integer::nx=8,ny=6"):
    path = tmp_path / "source.f90"
    path.write_text(
        "module model\n" + module + "\ncontains\nsubroutine boundary(a,periodic)\n"
        "real(8),intent(inout)::a(-1:,-1:)\nlogical,intent(in)::periodic\n"
        "integer::n,m,i,j\n" + declarations + "\n" + body + "\nend subroutine\nend module\n"
    )
    source = SourceEffects([path])
    return source, source.native_sections("model::boundary")


def test_conditional_faces_preserve_every_untouched_value(tmp_path):
    _, result = analyze(
        tmp_path,
        "n=nx\nm=ny\nif(periodic)then\ndo j=1,m\na(0,j)=a(n,j)\nend do\nelse\ndo j=1,m\na(0,j)=a(1,j)\nend do\nend if",
    )
    assert result.available, result.reason
    (resource,) = result.resources
    assert not resource.overwrites
    assert len(resource.reads) == 2
    assert len(resource.writes) == 1
    assert {x.resource for x in resource.dependencies} == {"model::nx", "model::ny"}
    assert resource.writes[0].axes[1].upper.resource == "model::ny"
    assert resource.reads[0].axes[0].lower.resource == "model::nx"


@pytest.mark.parametrize(
    "body",
    [
        "n=nx\nn=3\na(n,:)=1",
        "n=nx\nnx=3\na(n,:)=1",
        "if(periodic) n=nx\na(n,:)=1",
        "a(0,:)=0\nn=nx\na(n,:)=1",
        "n=nx+1\na(n,:)=1",
        "n=nx\ndo nx=1,3\na(n,:)=1\nenddo",
    ],
)
def test_mutable_or_nonentry_copy_keeps_conservative_access(tmp_path, body):
    _, result = analyze(tmp_path, body)
    assert not result.available
    assert result.resources == ()


def test_copied_name_reused_as_iterator_uses_iteration_coordinates(tmp_path):
    _, result = analyze(tmp_path, "n=nx\ndo n=1,3\na(n,:)=1\nenddo")
    assert result.available, result.reason
    axis = result.resources[0].writes[0].axes[0]
    assert axis.lower.value == 1
    assert axis.upper.value == 3


@pytest.mark.parametrize("module", ["integer,target::nx=8,ny=6", "integer,volatile::nx=8,ny=6"])
def test_aliasable_or_volatile_source_is_not_recomputed(tmp_path, module):
    _, result = analyze(tmp_path, "n=nx\na(n,:)=1", module=module)
    assert not result.available


def test_single_line_condition_is_a_may_write(tmp_path):
    _, result = analyze(tmp_path, "if(periodic) a(0,:)=a(1,:)")
    assert result.available, result.reason
    (resource,) = result.resources
    assert len(resource.writes) == 1
    assert len(resource.reads) == 1
    assert resource.overwrites == ()


def test_unknown_call_in_any_branch_keeps_the_whole_native_boundary(tmp_path):
    _, result = analyze(tmp_path, "if(periodic)then\na(0,:)=a(1,:)\nelse\ncall unknown(a)\nend if")
    assert not result.available
    assert result.resources == ()


def test_physical_copy_substitution_grants_no_logical_definition_authority(tmp_path):
    from compiler.frontend.native_sections import analyze_native_logical_accesses

    source, physical = analyze(tmp_path, "n=nx\na(n,:)=1")
    assert physical.available
    logical = analyze_native_logical_accesses(source, source.routines["model::boundary"])
    assert not logical.available
    assert logical.resources == ()
    assert "changes within" in logical.reason


@pytest.mark.parametrize("body", [
    "n=nx\nif(periodic)then\ndo nx=1,3\na(n,:)=1\nenddo\nend if",
    "n=nx\nblock\ninteger::n\nn=3\na(n,:)=1\nend block",
])
def test_nested_mutation_and_shadowing_do_not_reuse_an_entry_copy(tmp_path, body):
    _, result = analyze(tmp_path, body)
    assert not result.available
    assert result.resources == ()


@pytest.mark.parametrize("declarations", ["equivalence(n,m)", "common /indices/n", "save n"])
def test_unproved_local_identity_prevents_copy_substitution(tmp_path, declarations):
    _, result = analyze(tmp_path, "n=nx\nm=3\na(n,:)=1", declarations=declarations)
    assert not result.available
    assert result.resources == ()
