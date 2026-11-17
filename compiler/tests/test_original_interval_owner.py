"""Original-position intervals reuse closed-selection control and proofs."""

from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.closed_selection import ClosedIntervalOwner, interval_candidates
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_interval_draft import SOURCE
from compiler.tests.test_source_scopes import FACT

MIXED_SOURCE = SOURCE.replace(
    "endif\n!$omp parallel do private(i) schedule(runtime)\ndo i=0,n-1\ns(i)",
    "endif\n!$omp parallel private(i) shared(a,v,s,n) default(none)\n"
    "!$omp do schedule(runtime)\ndo i=0,n-1\nv(i)=v(i)+sum(a)\nenddo\n!$omp end do\n"
    "!$omp do schedule(runtime)\ndo i=0,n-1\ns(i)",
).replace(
    "s(i)=v(i)+a(i)\nenddo\n!$omp end parallel do",
    "s(i)=v(i)+a(i)\nenddo\n!$omp end do\n!$omp end parallel",
)


def builder(path, *, repeated=False, prefix="", suffix="", source=None, config=None):
    source = source or SOURCE
    start = "call producer(old,viscosity,n)"
    end = "call momentum(old,viscosity,star,out,n,correcting)"
    if repeated:
        source = source.replace(start, "integer:: repetition\ndo repetition=1,2\n" + prefix + start)
        source = source.replace(end, end + "\n" + suffix + "enddo")
    else:
        if prefix:
            source = source.replace(start, prefix + start)
        if suffix:
            source = source.replace(end, end + "\n" + suffix)
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"interval_fields::" + name: dict(FACT)
                          for name in ("old", "viscosity", "star", "out")}}
    config = config or OffloadConfig(policy="sections", scope_execution="reached")
    return ScopeBuilder([path], "interval_fields::advance", facts=facts,
                        options=CompilerOptions(gpu_policy=config.policy), config=config)


def call_pair(selected):
    return next(site for site in interval_candidates(selected)
                if sum(type(node).__name__ == "Call_Stmt" for node in site.originals) == 2
                and all(type(node).__name__ in {"Call_Stmt", "Comment"} for node in site.originals)
                and str(next(node for node in reversed(site.originals)
                             if type(node).__name__ == "Call_Stmt").items[0]).lower() == "momentum")


def owner_for(selected, site):
    return ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)


def emit_repeated_interval(path, *, mixed=True):
    """Shared host/CUDA fixture through the original selected-site owner."""
    selected = builder(path, repeated=True, source=MIXED_SOURCE if mixed else SOURCE)
    site = call_pair(selected)
    owner = owner_for(selected, site)
    assert owner.run(site.originals)
    outputs, report = selected.finish()
    return selected, owner, outputs, report


def absorbing_boundary_source():
    source = SOURCE.replace("integer::visits=0", "integer::visits=0,bad_visits=0\nreal(8),pointer::scratch(:)")
    source = source.replace("advance(n,correcting)", "advance(n,correcting,early)")
    source = source.replace("logical,intent(in)::correcting", "logical,intent(in)::correcting,early")
    source = source.replace("call producer(old,viscosity,n)", "call producer(old,viscosity,n,correcting,early)")
    source = source.replace("call momentum(old,viscosity,star,out,n,correcting)",
                            "call momentum(old,viscosity,star,out,n,correcting,early)")
    source = source.replace("subroutine producer(a,v,n)",
                            "subroutine producer(a,v,n,correcting,early)\nlogical,intent(in)::correcting,early")
    source = source.replace("integer::i\n!$omp", "integer::i\nif(correcting.and.early) call unsupported(v)\n!$omp", 1)
    source = source.replace("v(i)=2.d0*a(i)", "v(i)=v(i)+2.d0*a(i)")
    source = source.replace("momentum(a,v,s,o,n,correcting)", "momentum(a,v,s,o,n,correcting,early)")
    source = source.replace("visits=visits+1", "if(correcting.and..not.early) call unsupported(v)\nvisits=visits+1")
    return source.replace("end module", """subroutine unsupported(v)
real(8),intent(inout)::v(-1:)
scratch(0)=scratch(0)+1.d0
v(0)=v(0)+7.d0
bad_visits=bad_visits+1
end subroutine
end module""")


def emit_absorbing_interval(path, *, config=None):
    selected = builder(path, source=absorbing_boundary_source(), config=config)
    site = call_pair(selected)
    owner = owner_for(selected, site)
    assert owner.run(site.originals)
    outputs, report = selected.finish()
    return selected, owner, outputs, report
