"""Exact original-read admission after an unchanged native prefix."""



from compiler.scopes.closed_selection import ClosedIntervalOwner
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
