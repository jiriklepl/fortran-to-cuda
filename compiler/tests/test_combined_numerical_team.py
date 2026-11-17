"""Combined numerical groups use the original participants and whole join."""

from copy import copy
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.scopes.combined_team import (
    CombinedNumericalTeam,
    _check_generated_names,
    _narrow_runtime_imports,
    _source_partition,
    is_combined_group,
)
from compiler.scopes.segments import grouped_nodes
from compiler.tests.support.numerical_regions import (
    CHILD,
    CHILD_SOURCE,
    ENVIRONMENT_TEAM,
    MATH_SOURCE,
    EmittedNumericalText,
    combined_units,
    emitted_child,
    team_text)


def test_combined_runtime_imports_introduce_only_required_exact_exports():
    imports = ["use fort_scoped_memory", "use iso_c_binding"]
    _narrow_runtime_imports(imports, ["integer(c_int) :: fort_status"], ["fort_status = FORT_SCOPE_OK"])
    assert imports == ["use fort_scoped_memory, only: fort_scope_ok", "use iso_c_binding, only: c_int"]
    assert all("fort_scope_execution" not in line for line in imports)


@pytest.mark.parametrize(("literal", "kind"), [
    ("1_c_size_t", "c_size_t"), ("0.0_c_double", "c_double"),
    ("-1.25e-3_c_float", "c_float"), (".false._c_bool", "c_bool"),
])
def test_combined_runtime_imports_retain_literal_only_iso_kinds(literal, kind):
    original = "use original_coefficients, only: scale"
    imports = ["use fort_scoped_memory", "use iso_c_binding", original]
    _narrow_runtime_imports(imports, ["integer(c_int) :: fort_status"],
        ["fort_status = FORT_SCOPE_OK", "call consume(" + literal + ")"], original_imports=(original,))
    assert set(imports[1].split("only: ", 1)[1].split(", ")) == {"c_int", kind}
    assert imports[2] == original
