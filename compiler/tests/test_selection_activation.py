"""Exact original-read admission after an unchanged native prefix."""

from collections.abc import Mapping
from copy import copy
from dataclasses import replace
from functools import partial

import pytest

from compiler.offload.config import OffloadConfig
from compiler.frontend.selection_activation import (
    SelectionActivationProof,
    prove_selection_activation,
    selection_activation_demands,
)
from compiler.ir import CompilationError
from compiler.scopes.closed_selection import ClosedIntervalOwner
from compiler.tests.test_interval_draft import SOURCE
from compiler.tests.test_original_interval_owner import builder, call_pair

PREFIX = """integer::i
!$omp parallel do private(i)
do i=0,7
out(i)=0.d0
enddo
!$omp end parallel do
visits=visits+1
"""


def activation_fixture(path, *, config=None, prefix=PREFIX):
    selected = builder(path, prefix=prefix, config=config)
    selected.facts["captures"]["interval_fields::out"]["initialized"] = "none"
    site = call_pair(selected)
    owner = ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)
    return selected, owner, site


def emit_activation_fixture(path, *, config=None):
    selected, owner, site = activation_fixture(path, config=config)
    assert owner.run(site.originals)
    outputs, report = owner.builder.finish()
    return selected, owner, outputs, report
