"""Import lookup reuse preserves complete, fresh kind resolution."""
import pytest

from compiler.frontend.source_effects import SourceEffects


def build(tmp_path, declarations, imports):
    path = tmp_path / "imports.f90"
    path.write_text(declarations + "\nmodule consumer\n" + imports + "\ncontains\n"
                    "subroutine work(a)\ninteger(wide),intent(in)::a\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    return analysis, analysis.routines["consumer::work"].scope


@pytest.mark.parametrize(("tail", "expected"), [("", 8), ("use absent_kinds", None),
                                               ("use short_kinds", None)])
def test_diamond_reexports_keep_all_conflicts_and_exact_intrinsic_identity(tmp_path, tail, expected):
    source = ("module original\nuse iso_fortran_env,only:wide=>int64\nend module\n"
              "module short_kinds\ninteger,parameter::wide=4\nend module\n")
    previous = ["original"]
    for level in range(4):
        names = [f"branch_{level}_{side}" for side in range(2)]
        for name in names:
            source += f"module {name}\n" + "".join(f"use {parent}\n" for parent in previous) + "end module\n"
        previous = names
    analysis, scope = build(tmp_path, source, "\n".join(f"use {name}" for name in previous) + "\n" + tail)
    assert scope.bindings["a"].kind == expected
    assert analysis.modules["original"].kinds.values["wide"] == 8


@pytest.mark.parametrize("order", [False, True])
def test_cyclic_export_result_does_not_reuse_a_different_active_path(tmp_path, order):
    left = "module left\nuse right\nuse iso_fortran_env,only:wide=>int64\nend module\n"
    right = "module right\nuse left\nend module\n"
    _analysis, scope = build(tmp_path, right + left if order else left + right, "use left\nuse right")
    assert scope.bindings["a"].kind is None


def test_later_import_transaction_observes_changed_namespace(tmp_path):
    declarations = ("module long_kinds\nuse iso_fortran_env,only:wide=>int64\nend module\n"
                    "module short_kinds\nuse iso_fortran_env,only:wide=>int32\nend module\n")
    analysis, scope = build(tmp_path, declarations, "use long_kinds,only:wide")
    assert scope.bindings["a"].kind == 8
    # No lookup result from construction may survive a later invocation.
    scope.imports["wide"] = "short_kinds", "wide"
    analysis._import_kinds(scope)
    assert scope.kinds.values["wide"] == 4
    scope.wildcards.append("absent_kinds")
    analysis._import_kinds(scope)
    assert scope.kinds.values["wide"] is None


def test_block_scope_resolves_its_own_import_after_constructor_import_pass(tmp_path):
    path = tmp_path / "block_kinds.f90"
    path.write_text("module block_kinds\nuse iso_fortran_env,only:wide=>int64\ncontains\n"
                    "subroutine work()\ninteger(wide)::outer\nblock\n"
                    "use iso_fortran_env,only:wide=>int32\ninteger(wide)::inner\n"
                    "inner=1\nend block\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    assert analysis.routines["block_kinds::work"].scope.bindings["outer"].kind == 8
    block, = analysis._block_scopes.values()
    assert block.bindings["inner"].kind == 4


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shadow", [False, True])
def test_parameter_chain_retains_import_evaluation_order_and_local_shadow(tmp_path, reverse, shadow):
    modules = ["module leaf\ninteger,parameter::seed=4\nend module\n",
               "module middle\nuse leaf,only:seed\ninteger,parameter::wide=2*seed\nend module\n",
               "module relay\nuse middle\nend module\n"]
    if reverse:
        modules.reverse()
    path = tmp_path / "parameter_chain.f90"
    path.write_text("".join(modules) + "module consumer\nuse relay\ncontains\nsubroutine work(a)\n"
                    + ("integer,parameter::wide=4\n" if shadow else "")
                    + "integer(wide),intent(in)::a\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    assert analysis.modules["middle"].kinds.values["seed"] == 4
    assert analysis.routines["consumer::work"].scope.bindings["a"].kind == (4 if shadow else 8)
