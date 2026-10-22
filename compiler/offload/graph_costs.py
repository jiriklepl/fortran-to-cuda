"""Generic graph AUTO ranking from hardware rates and a static IR work proxy.

These weights are explicit heuristic flop-equivalent costs, not measured
original-Fortran family timings, GPU legality, or a speedup guarantee. Exact
canonical access geometry and coherence remain independently authenticated.
"""
from dataclasses import replace

from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    If,
    IntrinsicCall,
    Literal,
    Loop,
    Reference,
    ScalarType,
    Size,
    Unary,
)
from compiler.offload.analysis import _constant

# Conservative qualitative libm complexity classes. They are not latency bounds.
_INTRINSIC_WEIGHTS = {
    'sqrt': 8, 'exp': 32, 'log': 32, 'log10': 32,
    'sin': 64, 'cos': 64, 'tan': 64, 'asin': 64, 'acos': 64, 'atan': 64, 'atan2': 64,
}


def graph_work_proxy(region):
    unknown_loops = []
    real = {ScalarType.REAL, ScalarType.REAL32}

    def expression(value):
        # Match the existing real-arithmetic classifier's leaf boundary: array
        # address trees, lower-bound rebasing and INTEGER/logical preparation
        # are not floating-point operations. Memory is priced separately.
        if isinstance(value, Literal):
            return value.dtype, 0, True
        if isinstance(value, (ArrayAccess, Reference)):
            return value.symbol.dtype, 0, isinstance(value, Reference) and value.symbol.parameter
        if isinstance(value, Size):
            return ScalarType.INTEGER, 0, False
        if isinstance(value, Unary):
            dtype, count, constant = expression(value.operand)
            return dtype, count + (dtype in real and value.operator == '-' and not constant), constant
        if isinstance(value, Binary):
            left, left_count, left_constant = expression(value.left)
            right, right_count, right_constant = expression(value.right)
            constant = left_constant and right_constant
            count = left_count + right_count
            if value.operator in {'==', '/=', '<', '<=', '>', '>=', '.eq.', '.ne.', '.lt.', '.le.',
                                   '.gt.', '.ge.', '.and.', '.or.', '.eqv.', '.neqv.'}:
                return ScalarType.LOGICAL, count, constant
            dtype = (ScalarType.REAL if ScalarType.REAL in {left, right} else
                     ScalarType.REAL32 if ScalarType.REAL32 in {left, right} else ScalarType.INTEGER)
            if dtype in real and not constant:
                count += 4 if value.operator == '/' else 1
            return dtype, count, constant
        if isinstance(value, IntrinsicCall):
            arguments = value.arguments[:1] if value.name.lower() in {'real', 'dble', 'int'} else value.arguments
            values = [expression(argument) for argument in arguments]
            constant = all(item[2] for item in values)
            count = sum(item[1] for item in values)
            name = value.name.lower()
            if value.dtype in real and not constant and name not in {'real', 'dble', 'int'}:
                count += len(arguments)-1 if name in {'min', 'max'} else _INTRINSIC_WEIGHTS.get(name, 1)
            return value.dtype, count, constant
        raise TypeError(value)

    def block(body):
        work = 0
        for statement in body.statements:
            if isinstance(statement, Assignment):
                work += expression(statement.value)[1]
            elif isinstance(statement, If):
                work += expression(statement.condition)[1] + max(block(statement.then_body), block(statement.else_body))
            elif isinstance(statement, Loop):
                lower, upper = _constant(statement.lower), _constant(statement.upper)
                step = statement.step if isinstance(statement.step, int) else _constant(statement.step)
                count = None
                if lower is not None and upper is not None and step:
                    count = max(0, (upper - lower) // step + 1) if (upper-lower)*step >= 0 else 0
                if count is None:
                    # Do not invent a dynamic trip count or infer it from
                    # allocation extents. Retain explicit uncertainty.
                    unknown_loops.append(statement.iterator.name)
                    count = 1
                work += count * block(statement.body)
        return work
    return block(region.body), tuple(unknown_loops)


def apply_graph_hardware_costs(units):
    result, provenance = {}, {}
    for identity, unit in units.items():
        proxy, unknown = graph_work_proxy(unit.region)
        result[identity] = replace(unit, work_per_iteration=proxy, compute_model=None,
            cpu_numerical_seconds_per_iteration=0, gpu_numerical_seconds_per_iteration=0,
            work_estimate_reason='heuristic static IR work proxy; measured hardware roofline ranking')
        provenance[identity] = {'kind': 'hardware_roofline_heuristic', 'static_work_proxy': proxy,
            'work_proxy_semantics': 'real arithmetic only; no assignment, address, INTEGER or logical-operation FLOPs',
            'intrinsic_proxy_weights': dict(_INTRINSIC_WEIGHTS),
            'original_work_conditional_or_upper_bound': unit.work_is_upper_bound,
            'dynamic_loop_trips_unpriced': list(unknown), 'original_fortran_family_attestation': False,
            'speedup_guaranteed': False}
    return result, provenance
