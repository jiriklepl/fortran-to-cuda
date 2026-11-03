"""A genuine original empty helper has no executable line interval to edit."""
import json
from copy import copy

import pytest

from compiler.ir import CompilationError
from compiler.scopes.closed_selection import ClosedIntervalOwner
from compiler.tests.test_hierarchical_child_regions import SOURCE, fixture


def empty_source(*, arity=1, nested=False):
    assert arity in (1, 6)
    first = SOURCE.index("observed=0.0_8")
    last = SOURCE.index("end subroutine\nsubroutine advance", first)
    names = ["field", *["unused" + str(i) for i in range(1, arity)]]
    actuals = ["right", *names[1:]]
    body = "call empty_worker(" + ",".join([*actuals, "n"]) + ")\ncall consumer(left,right,n)\n"
    if nested:
        body = "call internal_helper\ncontains\nsubroutine internal_helper\n" + body + "end subroutine\n"
    text = SOURCE[:first] + body + SOURCE[last:]
    if arity == 6:
        declarations = "\n".join("real(8)::" + name + "(2)=[3.0_8,4.0_8]" for name in names[1:])
        text = text.replace("real(8)::observed", "real(8)::observed\n" + declarations, 1)
    empty = ("subroutine empty_worker(" + ",".join([*names, "n"]) + ")\n"
             "integer,intent(in)::n\nreal(8),intent(in)::" + ",".join(name + "(:)" for name in names)
             + "\nend subroutine\n")
    return text.replace("end module", empty + "end module")


def emit_package(tmp_path, *, arity=1, nested=False, source=None):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    path, builder = fixture(tmp_path, source or empty_source(arity=arity, nested=nested))
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    assert builder._conditional_return_draft is not None
    owner = builder._conditional_return_draft.owner
    empty = owner.members["hierarchical::empty_worker"]
    assert empty.routine.execution is None
    assert not empty.units
    assert not empty.resources
    assert not empty.boundaries
    package = tmp_path / "package"
    for name, content in outputs.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    manifest = json.loads((package / "scope-manifest.json").read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({"artifact_root": str(package), "scope_manifest": manifest})
    assert builder.finish()
    return path, builder, owner, empty, outputs, manifest


@pytest.mark.parametrize(("arity", "nested"), [(1, False), (6, False), (1, True)])
def test_empty_checked_child_retains_full_hierarchical_package(tmp_path, arity, nested):
    _path, _builder, _owner, empty, _outputs, manifest = emit_package(tmp_path, arity=arity, nested=nested)
    scope, = manifest["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]
    returned = epoch["return_contract"]
    assert returned["root_selection"]["selected_node_ids"]
    member, = [m for m in returned["members"] if m["procedure"] == empty.routine.qualified]
    assert member["routes"] == []
    assert member["entry_exit"]["definition_authority"] is False
    assert all(not row["resource"].split("::")[-1].startswith("unused") for row in epoch["resources"])
    assert any(route["kind"] == "checked_child" and route["procedure"] == empty.routine.qualified
               for row in returned["members"] for route in row["routes"])


@pytest.mark.parametrize("which", ["root", "genuine_empty", "nonempty_child"])
def test_no_empty_selection_is_an_executable_edit_interval(tmp_path, which):
    _path, builder, owner, empty, _outputs, _manifest = emit_package(tmp_path)
    routine = builder.entry if which == "root" else empty.routine if which == "genuine_empty" else owner.members["hierarchical::middle"].routine
    with pytest.raises(CompilationError, match="nonempty original source selection"):
        ClosedIntervalOwner._validate_edit_boundaries(builder, (), routine=routine)


@pytest.mark.parametrize("change", ["copy", "foreign", "dropped_nonempty", "fake_empty_projection", "source_edit"])
def test_empty_member_requires_its_captured_original_body(tmp_path, change):
    path, _builder, owner, empty, _outputs, _manifest = emit_package(tmp_path)
    if change == "copy":
        object.__setattr__(empty, "routine", copy(empty.routine))
    elif change == "foreign":
        other = tmp_path / "foreign"
        other.mkdir()
        foreign = emit_package(other)[3]
        object.__setattr__(empty, "routine", foreign.routine)
    elif change == "dropped_nonempty":
        object.__setattr__(owner.members["hierarchical::middle"].routine, "execution", None)
    elif change == "fake_empty_projection":
        object.__setattr__(empty.routine, "execution", ())
    else:
        path.write_text(path.read_text() + "! changed original input\n")
    with pytest.raises(CompilationError):
        owner.validate_plan(owner.closed_selection)


@pytest.mark.parametrize("contains", [False, True])
def test_empty_entry_anchor_is_before_original_contains_or_end(tmp_path, contains):
    source = empty_source()
    if contains:
        source = source.replace("real(8),intent(in)::field(:)\nend subroutine", """real(8),intent(in)::field(:)
contains
subroutine unused_internal()
end subroutine
end subroutine""", 1)
    _path, _builder, _owner, empty, outputs, manifest = emit_package(tmp_path, source=source)
    generated = outputs[manifest["sources"][str(empty.routine.scope.path)]["replacement"]]
    first = generated.lower().index("subroutine empty_worker")
    last = generated.lower().index("end subroutine", first)
    body = generated[first:last].lower()
    assert "entry " + empty.entry.lower() in body
    if contains:
        assert body.index("entry " + empty.entry.lower()) < body.index("contains")
        assert "subroutine unused_internal()" in body
    assert not empty.units
    assert not empty.resources


@pytest.mark.parametrize("line", ["header", "header_spec", "declaration", "contains"])
def test_empty_entry_anchor_cannot_split_an_original_statement_line(tmp_path, line):
    source = empty_source()
    if line == "header":
        source = source.replace("subroutine empty_worker(field,n)\ninteger,intent(in)::n\nreal(8),intent(in)::field(:)\nend subroutine",
                                "subroutine empty_worker(field,n);integer,intent(in)::n;real(8),intent(in)::field(:);end subroutine")
    elif line == "header_spec":
        source = source.replace("subroutine empty_worker(field,n)\ninteger,intent(in)::n", "subroutine empty_worker(field,n);integer,intent(in)::n")
    elif line == "declaration":
        source = source.replace("real(8),intent(in)::field(:)\nend subroutine", "real(8),intent(in)::field(:);end subroutine")
    else:
        source = source.replace("real(8),intent(in)::field(:)\nend subroutine", """real(8),intent(in)::field(:);contains
subroutine unused_internal()
end subroutine
end subroutine""")
    path, builder = fixture(tmp_path, source)
    outputs, report = builder.run()
    assert source == path.read_text()
    assert "source edit line" in str(report) or "shares a source line" in str(report)
    assert not any(row["procedure"] == "hierarchical::empty_worker"
                   for scope in report["scopes"] for row in scope["module_coordinators"])
    assert not any("entry fort_scope_entry_" in text.lower() and "subroutine empty_worker" in text.lower()
                   for text in outputs.values() if text.strip().lower().startswith("subroutine empty_worker"))


@pytest.mark.parametrize("shape", ["declarations", "contains", "no_declarations"])
def test_gfortran_checks_original_empty_companion_entry_anchor(tmp_path, shape):
    import shutil
    import subprocess

    source = empty_source()
    if shape == "contains":
        source = source.replace("real(8),intent(in)::field(:)\nend subroutine", """real(8),intent(in)::field(:)
contains
subroutine unused_internal()
end subroutine
end subroutine""", 1)
    elif shape == "no_declarations":
        source = (source.replace("call empty_worker(right,n)", "call empty_worker()")
            .replace("subroutine empty_worker(field,n)\ninteger,intent(in)::n\nreal(8),intent(in)::field(:)",
                     "subroutine empty_worker()"))
    _path, _builder, _owner, _empty, _outputs, manifest = emit_package(tmp_path, source=source)
    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert compiler is not None
    package = tmp_path / "package"
    for role in ("common_runtime", "shared_entry", "original_source"):
        for artifact in manifest["build_sources"]:
            if artifact["role"] != role or artifact["language"] != "fortran":
                continue
            command = [compiler, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(package),
                       "-I", str(package), "-fsyntax-only", str(package / artifact["path"])]
            result = subprocess.run(command, cwd=package, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr


def test_empty_anchor_ignores_semicolons_in_original_comments(tmp_path):
    source = empty_source().replace("subroutine empty_worker(field,n)\n", "subroutine empty_worker(field,n) ! '; fake end subroutine'\n")
    source = source.replace("real(8),intent(in)::field(:)\nend subroutine", "real(8),intent(in)::field(:) ! x;y;contains\nend subroutine")
    _path, _builder, _owner, empty, _outputs, _manifest = emit_package(tmp_path, source=source)
    assert empty.routine.execution is None
