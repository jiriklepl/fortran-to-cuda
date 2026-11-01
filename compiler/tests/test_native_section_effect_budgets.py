"""Read and write rectangle lists retain separate bounded access contracts."""

from __future__ import annotations

import pytest

from compiler.frontend.native_sections import analyze_native_logical_accesses
from compiler.frontend.source_effects import SourceEffects
from compiler.scopes.access import build_native_access
from compiler.tests.test_native_sections import analyze
from compiler.tests.test_native_sections_completion import fixture, loop, sections


@pytest.mark.parametrize("count", [20, 32])
@pytest.mark.parametrize("conditional", [False, True])
def test_disjoint_read_and_write_lists_fit_the_existing_access_abi(tmp_path, count, conditional):
    body = "\n".join(f"a({i},:)=a({i + 100},:)+7" for i in range(count))
    if conditional:
        body = "if(n>0) then\n" + body + "\nendif"
    _, result = analyze(tmp_path, body)
    assert result.available, result.reason
    resource, = result.resources
    assert len(resource.reads) == len(resource.writes) == count
    assert len(set((*resource.reads, *resource.writes))) == 2 * count
    assert len(resource.overwrites) == (0 if conditional else count)
    assert [box.axes[0].lower.value for box in resource.reads] == list(range(100, 100 + count))
    assert [box.axes[0].lower.value for box in resource.writes] == list(range(count))
    code = build_native_access(resource, "handle", "fort_separate")
    specification = "\n".join(code.specification)
    assert f"fort_separate_r_sections({count})" in specification
    assert f"fort_separate_w_sections({count})" in specification
    assert ("fort_separate_o_sections" in specification) is not conditional


@pytest.mark.parametrize("action", ["read", "write"])
def test_one_effect_list_exceeding_32_still_declines(tmp_path, action):
    body = "\n".join(f"a({i if action == 'write' else 0},:)=a({100 + i if action == 'read' else 100},:)" for i in range(33))
    _, result = analyze(tmp_path, body)
    assert not result.available
    assert "rectangle budget exceeded" in result.reason
    assert result.resources == ()


def test_repeated_conditional_sections_deduplicate_each_list_without_overwrite_credit(tmp_path):
    units = [loop(str(i), statement=f"b({i},j)=b({100 + i},j)+1") for i in range(20)]
    body = "if(flag) then\n" + "".join(sections(first=units[i], second=units[i + 1]) for i in range(0, 20, 2)) * 3 + "endif\n"
    path, analysis, graph, selected, group = fixture(tmp_path, body, operations=64)
    assert group is not None
    assert group.available
    before = path.read_bytes()
    proof = analysis.joined_completion("renamed_sections::step", selected)
    result = analysis.native_sections_for_nodes("renamed_sections::step", selected, completion=proof, capture_locals=True)
    assert result.available, result.reason
    resource, = result.resources
    assert resource.resource == "argument::b"
    assert len(resource.reads) == len(resource.writes) == 20
    assert resource.overwrites == ()
    assert all(len(box.iterations) == 1 for box in (*resource.reads, *resource.writes))
    assert graph.native_group_for_selection(selected) is group
    assert path.read_bytes() == before


def test_separate_effect_budgets_keep_unknown_bounds_and_original_read_obligations(tmp_path):
    body = "\n".join(f"a({i},:)=a({i + 100},:)" for i in range(20))
    _, unknown = analyze(tmp_path, body + "\na(n*n,:)=1")
    assert not unknown.available
    assert "affine scalar arithmetic" in unknown.reason
    _, undefined = analyze(tmp_path, body, declaration="real(8),intent(out)::a(-2:,:)")
    assert not undefined.available
    assert "INTENT(OUT) reads" in undefined.reason


def test_logical_geometry_keeps_the_same_independent_budget_without_physical_authority(tmp_path):
    path = tmp_path / "logical.f90"
    body = "\n".join(f"a({i})=a({i + 100})" for i in range(20))
    path.write_text("module model\ncontains\nsubroutine original(a)\nreal(8),intent(inout)::a(:)\n"
                    + body + "\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    result = analyze_native_logical_accesses(analysis, analysis.routines["model::original"])
    assert result.available, result.reason
    resource, = result.resources
    assert len(resource.reads) == len(resource.writes) == len(resource.overwrites) == 20
    assert len(result.original_reads) == 20
    assert result.public()["physical_mapping_authority"] is False
