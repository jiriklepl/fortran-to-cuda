"""Exact parameter projection versus independent concrete original accesses."""
from itertools import product

import pytest

from compiler.analysis import build_execution_plan
from compiler.analysis.conditional_affine import ConditionalAffineUnsupported, analyze_conditional_affine
from compiler.frontend import lower_source
from compiler.ir import CompilationError


def source(body, *, declarations="", arguments="n", dimensions=":"):
    return lower_source(
        "module conditional_fixture\ncontains\nsubroutine work(a," + arguments + ")\n"
        "implicit none\nreal(8),intent(inout)::a(" + dimensions + ")\n"
        "integer,intent(in)::" + arguments + "\ninteger::i,j,k\n" + declarations + "\n"
        + body + "\nend subroutine\nend module", "conditional_fixture::work", source_name="conditional_affine_fixture.f90")


def values(domain, **parameters):
    return {name: parameters[symbol.name] for name, symbol in domain.parameters}


def independent(units):
    """Concrete addresses, without ISL or symbolic guard evaluation."""
    for unit in units:
        for number, (reads, writes) in enumerate(unit):
            for other_reads, other_writes in unit[number + 1:]:
                if writes & (other_reads | other_writes) or other_writes & reads:
                    return False
    for number, unit in enumerate(units):
        reads = set().union(*(read for read, _write in unit))
        writes = set().union(*(write for _read, write in unit))
        for other in units[number + 1:]:
            other_reads = set().union(*(read for read, _write in other))
            other_writes = set().union(*(write for _read, write in other))
            if writes & (other_reads | other_writes) or other_writes & reads:
                return False
    return True


@pytest.mark.parametrize("direction", [-1, 1])
def test_shifted_single_unit_domain_matches_original_ordered_conflicts(direction):
    body = f"do i=1,n\na(i)=a(i{direction:+})\nenddo"
    function = source(body)
    with pytest.raises(CompilationError, match="loop-carried conflict"):
        build_execution_plan(function)
    domain = analyze_conditional_affine(function)
    for n in range(-3, 10):
        explicit = [[({i + direction}, {i}) for i in range(1, n + 1)]]
        assert domain.accepts(values(domain, n=n)) is independent(explicit)
    assert not domain.public()['execution_authorized']
    assert not domain.public()['incoming_read_admission_authorized']
    assert not domain.public()['source_authority']


def test_parameterized_halo_copy_checks_shifted_inplace_small_domains():
    function = source("do i=n+1,n+3\na(i)=a(i-n)\nenddo")
    domain = analyze_conditional_affine(function)
    for n in range(-5, 12):
        explicit = [[({i-n}, {i}) for i in range(n+1, n+4)]]
        assert domain.accepts(values(domain, n=n)) is independent(explicit)
    # The mathematical same-point update n=0 is not a claim about the whole
    # twenty-section team, nor permission to replace a dependent small case.
    assert domain.accepts(values(domain, n=0))
    assert not domain.accepts(values(domain, n=1))
    assert not domain.accepts(values(domain, n=2))


@pytest.mark.parametrize("second", ["a(i)=1.d0", "a(i)=a(i-n)", "a(i)=a(i+n)"])
def test_cross_unit_raw_war_waw_domains_match_explicit_accesses(second):
    first = "do i=n+1,n+3\na(i)=a(i-n)\nenddo\n"
    domain = analyze_conditional_affine(source(first + "do i=-2,0\n" + second + "\nenddo"))
    for n in range(-5, 12):
        a = [({i-n}, {i}) for i in range(n+1, n+4)]
        b = [((set() if second.endswith('1.d0') else {i-n if second.endswith('i-n)') else i+n}), {i})
             for i in range(-2, 1)]
        assert domain.accepts(values(domain, n=n)) is independent([a, b])


def test_self_update_and_distinct_storage_remain_unconditionally_proven():
    function = source("do i=1,n\na(i)=a(i)+1.d0\nenddo")
    unconditional = build_execution_plan(function)
    domain = analyze_conditional_affine(function)
    assert len(unconditional.regions) == 1
    assert all(domain.accepts(values(domain, n=n)) for n in range(-5, 12))
    assert not domain.accepts(values(domain, n=True))
    assert not domain.accepts(values(domain, n=2**31))
    assert not domain.accepts({})


@pytest.mark.parametrize("body", [
    "do i=1,n\na(i*i)=a(i*i)+1.d0\nenddo",
    "do i=1,n*n\na(i)=1.d0\nenddo",
    "do i=1,n\nif(i>2) a(i)=1.d0\nenddo",
    "do i=1,n\na(i)=1.d0\na(i+1)=2.d0\nenddo",
    "do i=1,n\ndo j=1,n\na(i)=a(i)+1.d0\nenddo\na(i)=0.d0\nenddo",
    "do i=1,n\na(i)=1.d0\ndo j=1,n\na(i)=a(i)+1.d0\nenddo\nenddo",
    "do i=1,n\na(i)=abs(a(i))\nenddo",
    "do i=1,n\na(i)=merge(a(i-1),a(i+1),a(i)>0.d0)\nenddo",
])
def test_nonaffine_surrogates_and_broader_source_shapes_decline(body):
    with pytest.raises(ConditionalAffineUnsupported):
        analyze_conditional_affine(source(body))


@pytest.mark.parametrize("budget", [0, True, -1, 1, 4])
def test_effort_exhaustion_never_returns_partial_legality(budget):
    with pytest.raises(ConditionalAffineUnsupported):
        analyze_conditional_affine(source("do i=n+1,n+3\na(i)=a(i-n)\nenddo"), operation_budget=budget)


def test_private_isl_budget_does_not_change_unconditional_planner():
    function = source("do i=n+1,n+3\na(i)=a(i-n)\nenddo")
    with pytest.raises(ConditionalAffineUnsupported, match="exact (model|projection) unavailable"):
        analyze_conditional_affine(function, isl_operation_budget=1)
    with pytest.raises(CompilationError, match="loop-carried conflict"):
        build_execution_plan(function)
    assert analyze_conditional_affine(function).accepts(values(analyze_conditional_affine(function), n=3))


def test_entire_conditional_model_avoids_the_ordinary_isl_context():
    import islpy as isl

    function = source("do i=n+1,n+3\na(i)=a(i-n)\nenddo")
    previous = isl.DEFAULT_CONTEXT.get_max_operations()
    try:
        isl.DEFAULT_CONTEXT.set_max_operations(1)
        domain = analyze_conditional_affine(function)
        assert domain.accepts(values(domain, n=3))
        assert isl.DEFAULT_CONTEXT.get_max_operations() == 1
    finally:
        isl.DEFAULT_CONTEXT.set_max_operations(previous)


def periodic_bodies():
    bodies, signs = [], []
    for sign in product((-1, 0, 1), repeat=3):
        if sum(value != 0 for value in sign) < 2:
            continue
        signs.append(sign)
        controls = []
        source_indices = []
        for axis, extent, side in zip(('i', 'j', 'k'), ('nx', 'ny', 'nz'), sign, strict=True):
            lower, upper = ('-2', '0') if side < 0 else (extent+'+1', extent+'+3') if side else ('1', extent)
            controls.append('do '+axis+'='+lower+','+upper)
            source_indices.append(axis+'+'+extent if side < 0 else axis+'-'+extent if side else axis)
        bodies.append('\n'.join([*reversed(controls), 'a(i,j,k)=a('+','.join(source_indices)+')', 'enddo\n'*3]))
    assert len(signs) == 20
    return bodies, signs


def test_twenty_periodic_copy_units_match_independent_small_grid_enumerator():
    """Generic original-style point copies, not an application entry contract."""
    bodies, signs = periodic_bodies()
    domain = analyze_conditional_affine(source('\n'.join(bodies), arguments='nx,ny,nz', dimensions=':,:,:'), operation_budget=10000)
    for shape in product(range(-1, 5), repeat=3):
        units = []
        for sign in signs:
            ranges = [range(-2, 1) if side < 0 else range(n+1, n+4) if side else range(1, n+1)
                      for n, side in zip(shape, sign, strict=True)]
            unit = []
            for point in product(*ranges):
                incoming = tuple(value-side*n for value, side, n in zip(point, sign, shape, strict=True))
                unit.append(({incoming}, {point}))
            units.append(unit)
        assert domain.accepts(values(domain, nx=shape[0], ny=shape[1], nz=shape[2])) is independent(units), shape
    for shape in ((128,64,64), (128,63,64), (128,64,63)):
        assert domain.accepts(values(domain, nx=shape[0], ny=shape[1], nz=shape[2]))


def test_alternative_projection_reuses_exact_unit_models_and_one_budget(monkeypatch):
    import compiler.analysis.conditional_affine as module

    function = source("do i=n+1,n+3\na(i)=a(i-n)\nenddo\n"
                      "do i=-2,0\na(i)=a(i+n)\nenddo")
    original, calls = module._region_model, []
    def count(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_region_model", count)
    models = module.collect_conditional_affine(function)
    for indices in ((0,), (1,), (0, 1), (0,)):
        assert models.project(indices).accepts(values(models.project(indices), n=4))
    assert calls == [0, 1]
    assert len(models._unit_cache) == 2 and len(models._pair_cache) == 1
    for indices in ((), (1, 0), (0, 0), (True,), (2,), [0]):
        with pytest.raises(ConditionalAffineUnsupported):
            models.project(indices)
    models._budget.remaining = 0
    with pytest.raises(ConditionalAffineUnsupported, match="effort exhausted"):
        models.project((0, 1))


def test_twenty_rebased_units_all_paths_fit_one_unchanged_symbolic_budget():
    from compiler.analysis.conditional_affine import collect_conditional_affine

    bodies, signs = periodic_bodies()
    import re
    def rebase(match):
        coordinates = match.group(1).split(',')
        return 'a(' + ','.join('('+value+')-lb'+str(axis)+'+1'
                              for axis, value in enumerate(coordinates, 1)) + ')'
    # Exact ordinary outline-style physical rebases; no dynamic specification
    # expression is granted to the standalone frontend by this math fixture.
    body = re.sub(r'a\(([^()]*)\)', rebase, '\n'.join(bodies))
    function = source(body, arguments='nx,ny,nz,lb1,lb2,lb3', dimensions=':,:,:')
    models = collect_conditional_affine(function, operation_budget=65536, isl_operation_budget=1000000)
    groups = [[index for index, sign in enumerate(signs) if all(sign)],
              *[[index for index, sign in enumerate(signs) if sign[axis] == 0] for axis in range(3)]]
    paths = [tuple(sorted(index for group, active in zip(groups, mask, strict=True) if active for index in group))
             for mask in product((True, False), repeat=4)]
    domains = {path:models.project(path) for path in paths if path}
    assert len(domains) == 15
    assert len(models._unit_cache) == 20 and len(models._pair_cache) == 190
    assert len(models._path_cache) == 15
    for path, domain in domains.items():
        for dims in ((128,64,64),(3,3,3),(2,3,3),(0,0,0)):
            names=dict(zip(('nx','ny','nz'),dims))
            units=[]
            for index in path:
                sign=signs[index]
                ranges=[range(-2,1) if side<0 else range(n+1,n+4) if side else range(1,n+1)
                        for n,side in zip(dims,sign,strict=True)]
                if dims[0] == 128:
                    # Full-size geometry is checked symbolically; the independent
                    # finite point oracle below covers the small exact domains.
                    continue
                units.append([({tuple(value-side*n for value,side,n in zip(point,sign,dims,strict=True))},{point})
                              for point in product(*ranges)])
            for lowers in ((-2,-2,-2),(-19,7,0)):
                parameters={**names,**dict(zip(('lb1','lb2','lb3'),lowers))}
                actual=domain.accepts({name:parameters[symbol.name] for name,symbol in domain.parameters})
                assert actual is (True if dims[0] == 128 else independent(units))
