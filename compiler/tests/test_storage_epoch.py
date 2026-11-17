"""Lifetime attachments cannot widen the checked source/storage authority."""
from copy import copy
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.tests.test_checked_execution_plan import scanned
from compiler.tests.test_graph_source_owner import EPOCH_SOURCE, emit


def test_unknown_native_cuts_are_retained_in_the_shared_lifetime_contract(tmp_path):
    source, outputs, report = emit(tmp_path, EPOCH_SOURCE)
    owner, = report["scopes"]
    epoch = owner["checked_execution_plan"]["storage_epoch"]
    assert epoch["cross_cut_retention_authorized"] is False
    assert epoch["members"][0]["retirement_cuts"] == owner["boundaries"]
    assert len(epoch["members"][0]["retirement_cuts"]) >= 2
    generated = outputs[report["sources"][str(source)]["replacement"]]
    assert "fort_scope_resident_create_v1(0_c_int, fort_context)" in generated
    assert "fort_scope_resident_close_v1(" in generated


def test_checked_child_registration_opens_a_domain_before_dummy_rebasing(tmp_path):
    from compiler.tests.test_graph_source_owner import reusable_helper_source

    source, outputs, report = emit(tmp_path, reusable_helper_source(True))
    generated = outputs[report["sources"][str(source)]["replacement"]]
    assert "fort_scope_resident_create_v1(" in generated
    assert " = fort_scope_create(" not in generated
