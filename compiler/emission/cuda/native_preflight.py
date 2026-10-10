"""Cheap, metadata-only proofs that a fresh numerical entry must stay native.

This deliberately does not reproduce full planning or price GPU startup. It
only recognizes empty domains and dimensions outside validated cost ranges.
"""

from dataclasses import dataclass

from compiler.emission.common.abi import dimension_name
from compiler.emission.cuda.structured import _query_expression
from compiler.ir import Binary, IntrinsicCall, Literal, ParallelRegion, Reference, ScalarType, Size, Unary


@dataclass(frozen=True)
class NativePreflight:
    cpp: tuple[str, ...] = ()
    fortran: tuple[str, ...] = ()
    parameters: tuple[dict, ...] = ()
    reason: str | None = None
    item_range: tuple[int, int] | None = None

    @property
    def available(self):
        return self.reason is None and self.item_range is not None

    def public(self, name):
        return {"abi_version": 1, "available": self.available, "reason": self.reason,
                "entry": name if self.available else None,
                "fortran_procedure": "native_preflight" if self.available else None,
                "parameters": list(self.parameters), "result": {"0": "not proven", "1": "native"},
                "item_range": list(self.item_range) if self.item_range is not None else None,
                "proofs": ["empty domain", "outside validated compute item range"] if self.available else [],
                "position": "fresh owner only, after original allocation and descriptor guards",
                "payload_reads": False, "contexts_created": 0, "registrations": 0,
                "startup_cost_proof": "deferred; no native memory upper bound is assumed",
                "unknown_or_overflow": "not proven; use ordinary reached planning",
                "continuations": "unsupported; earlier work must never be replayed"}


def generate_native_preflight(function, plan, units, name, *, source_compute, collective,
                              planning_available, profile_available, protected):
    """Emit one conservative original-call preflight, with exact ABI metadata."""
    def unavailable(reason):
        return NativePreflight(reason=reason)

    if not source_compute:
        return unavailable("original Fortran compute calibration is required")
    if collective:
        return unavailable("existing-team preflight requires coordinated original participation")
    if not planning_available or not profile_available:
        return unavailable("compatible complete numerical estimates are unavailable")
    if len(units) != 1 or len(plan.steps) != 1 or not isinstance(plan.steps[0], ParallelRegion):
        return unavailable("preflight requires one unconditional numerical unit without host preparation")
    unit, = units
    if unit.region is not plan.steps[0] or unit.compute_model is None:
        return unavailable("complete original numerical model is unavailable")
    if protected.get(unit.region.id):
        return unavailable("original scalar inputs may be protected by numerical control")
    parameters = set(function.parameters)
    used_scalars, used_extents = set(), set()

    def safe(expression):
        if isinstance(expression, Literal):
            return expression.dtype is ScalarType.INTEGER
        if isinstance(expression, Reference):
            symbol = expression.symbol
            if symbol not in parameters or symbol.rank or symbol.dtype is not ScalarType.INTEGER or symbol.intent != "in":
                return False
            used_scalars.add(symbol)
            return True
        if isinstance(expression, Size):
            symbol, dimension = expression.symbol, expression.dimension
            if symbol not in parameters or not symbol.rank or not 1 <= dimension <= symbol.rank:
                return False
            used_extents.add((symbol, dimension))
            return True
        if isinstance(expression, Unary):
            return expression.operator in {"+", "-"} and safe(expression.operand)
        if isinstance(expression, Binary):
            return expression.operator in {"+", "-", "*", "/"} and safe(expression.left) and safe(expression.right)
        if isinstance(expression, IntrinsicCall):
            return expression.name.lower() in {"min", "max"} and all(safe(arg) for arg in expression.arguments)
        return False

    for loop in unit.region.loops:
        if not all(safe(expression) for expression in (loop.lower, loop.upper)):
            return unavailable("loop bounds require unavailable scalar state or array payload")
        if not isinstance(loop.step, int) and not safe(loop.step):
            return unavailable("loop strides require unavailable scalar state or array payload")
    metadata, signature, fortran = [], [], ["      import :: c_int, c_size_t", "      integer(c_int) :: fort_native"]
    for symbol in function.parameters:
        for dimension in range(1, symbol.rank + 1):
            if (symbol, dimension) in used_extents:
                parameter = dimension_name(symbol, dimension)
                metadata.append({"name": parameter, "kind": "array_extent", "parameter": symbol.name,
                                 "dimension": dimension, "passing": "value", "integer_kind": "c_size_t"})
                signature.append(f"std::size_t {parameter}")
                fortran.append(f"      integer(c_size_t), value :: {parameter}")
        if symbol in used_scalars:
            metadata.append({"name": symbol.cpp_name, "kind": "integer_scalar", "parameter": symbol.name,
                             "passing": "reference", "integer_kind": "c_int"})
            signature.append(f"const int *fort_scalar_{symbol.cpp_name}")
            fortran.append(f"      integer(c_int), intent(in) :: {symbol.cpp_name}")
    cpp = [f'extern "C" int {name}({", ".join(signature)}) {{',
           "    struct { bool valid = true; } d;", "    bool active = true; std::size_t points = 1;"]
    for symbol in function.parameters:
        if symbol in used_scalars:
            cpp += [f"    if (!fort_scalar_{symbol.cpp_name}) return 0;",
                    f"    const volatile int &{symbol.cpp_name} = *fort_scalar_{symbol.cpp_name};"]
    for loop in unit.region.loops:
        stride = str(loop.step) if isinstance(loop.step, int) else _query_expression(loop.step, {})
        cpp += ["    if (active) {", f"        const int lo = {_query_expression(loop.lower, {})};",
                f"        const int hi = {_query_expression(loop.upper, {})};", f"        const int step = {stride};",
                "        if (!step) d.valid = false;", "        std::size_t extent = 0;",
                "        if (d.valid && step > 0 && hi >= lo)",
                "            extent = static_cast<std::size_t>((static_cast<long long>(hi) - lo) / step + 1);",
                "        if (d.valid && step < 0 && lo >= hi)",
                "            extent = static_cast<std::size_t>((static_cast<long long>(lo) - hi) / -static_cast<long long>(step) + 1);",
                "        active = d.valid && extent != 0;",
                "        if (active && !offload::mul(points, extent, points)) d.valid = false;", "    }"]
    lower, upper = unit.compute_model["item_range"]
    cpp += ["    if (!d.valid) return 0;", "    if (!active) return 1;",
            f"    return points < {lower}ULL || points > {upper}ULL ? 1 : 0;", "}"]
    from compiler.emission.fortran.formatting import _fortran_list

    fortran = [*_fortran_list("function native_preflight(", [item["name"] for item in metadata],
                             f") bind(C, name='{name}') result(fort_native)", 4),
               *fortran, "    end function"]
    return NativePreflight(tuple(cpp), tuple(fortran), tuple(metadata), item_range=(lower, upper))
