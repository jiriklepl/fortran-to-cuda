"""Published preparation identities are reproducible; reuse stays source-owned."""

import json
import subprocess
import sys
from copy import copy
from dataclasses import replace
from pathlib import Path

import pytest

from compiler.ir import CompilationError
from compiler.tests.test_numerical_selection_authority import ENTRY, selection_fixture


def prepare(directory):
    _path, analysis, provider, selection = selection_fixture(directory)
    return analysis, provider, selection, provider.prepare(analysis, ENTRY, selection)


def test_fresh_analyses_share_public_identity_but_not_private_authority(tmp_path):
    first_analysis, _, first_selection, first = prepare(tmp_path)
    second_analysis, _, second_selection, second = prepare(tmp_path)
    assert first_analysis is not second_analysis
    assert first_selection[0] is not second_selection[0]
    assert first._extraction_identity != second._extraction_identity
    assert first.identity == second.identity
    assert first.public() == second.public()
    assert first.validate(first_analysis, ENTRY, first_selection) is first
    assert second.validate(second_analysis, ENTRY, second_selection) is second
    with pytest.raises(CompilationError):
        first.validate(second_analysis, ENTRY, second_selection)


def test_fresh_processes_publish_identical_complete_preparation(tmp_path):
    code = """
import json,sys
from pathlib import Path
from compiler.tests.test_preparation_content_identity import prepare
analysis,provider,selection,token=prepare(Path(sys.argv[1]))
print(json.dumps(token.public(),sort_keys=True))
"""
    outputs = [subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                             check=True, capture_output=True, text=True, timeout=30).stdout
               for _ in range(2)]
    assert json.loads(outputs[0])["preparation_identity"]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("member", ["nodes", "bindings"])
def test_same_content_foreign_objects_still_reject(tmp_path, member):
    analysis, _, selection, token = prepare(tmp_path)
    extraction = token.extraction
    values = getattr(extraction, member)
    substituted = (copy(values[0]), *values[1:])
    assert substituted[0] is not values[0]
    replacement = replace(extraction, **{member: substituted})
    assert replacement.public() == extraction.public()
    assert replacement.source == extraction.source
    object.__setattr__(token, "extraction", replacement)
    with pytest.raises(CompilationError, match="source, context or checked IR authority changed"):
        token.validate(analysis, ENTRY, selection)


def test_repeated_preparation_reuses_only_same_registered_token(tmp_path):
    analysis, provider, selection, token = prepare(tmp_path)
    assert provider.prepare(analysis, ENTRY, selection) is token


def test_changed_original_source_changes_public_identity(tmp_path):
    from fparser.two import Fortran2003 as F
    from fparser.two.utils import walk

    from compiler.frontend.numerical_preparation import NumericalPreparationProvider
    from compiler.frontend.source_effects import SourceEffects
    from compiler.tests.test_numerical_selection_authority import CONFIG, OPTIONS

    _analysis, _, _, first = prepare(tmp_path)
    path = Path(tmp_path) / "selection.f90"
    path.write_text(path.read_text().replace("a(i)+1.0_8", "a(i)+2.0_8"))
    analysis = SourceEffects([path])
    selection = (walk(analysis.routines[ENTRY].execution, F.Block_Nonlabel_Do_Construct)[0],)
    provider = NumericalPreparationProvider(analysis, options=OPTIONS, config=CONFIG, environment_root=ENTRY)
    second = provider.prepare(analysis, ENTRY, selection)
    assert first.identity != second.identity
