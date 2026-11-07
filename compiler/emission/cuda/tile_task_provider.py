"""A source-issued tile team on the existing non-reentrant device transaction.

The callback only submits original lowered kernels on its supplied stream. All
runtime calls, checked descriptor snapshots and scratch reservation occur outside
it. This first provider is unpriced and retains the whole original team fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.c_family import cpp_type
from compiler.emission.cuda.tile_private import generate_cooperative_kernel, generate_kernel, kernel_parameters
from compiler.frontend.lowering import lower_tile_private
from compiler.frontend.tile_private import TilePrivateTeamProof, prove_cooperative_tile_private
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    CompilationError,
    If,
    IntrinsicCall,
    Literal,
    Loop,
    Reference,
    ScalarType,
    Size,
    Unary,
)

BATCH_TARGET_BYTES = 4 * 1024 * 1024
EXPERIMENTAL_BATCH_TARGET_BYTES = (4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024)


@dataclass(frozen=True, slots=True)
class TileParameter:
    role: str
    binding: object
    axis: int | None
    name: str
    kind: str

    def public(self):
        return {"role": self.role, "binding": self.binding.public(), "axis": self.axis,
                "name": self.name, "kind": self.kind}


@dataclass(frozen=True, slots=True)
class TileTaskProvider:
    entry: str
    module: str
    cuda_source: str
    fortran_source: str
    fields: tuple
    parameters: tuple
    _record: str

    def public(self):
        return json.loads(self._record)


def _fail(message):
    raise CompilationError("tile source provider: " + message)


def _iso(binding):
    key = binding.dtype, binding.kind
    result = {("integer", 4): "c_int", ("real", 4): "c_float", ("real", 8): "c_double"}.get(key)
    if result is None:
        _fail("unsupported original scalar or element ABI")
    return result


def _launch_abi(region):
    """Exact generated CUDA parameter layout on the supported 64-bit ABI."""
    declarations = kernel_parameters(region) + ["std::size_t first", "std::size_t count"]
    offset = 0
    for declaration in declarations:
        if "*" in declaration or declaration.startswith(("std::size_t ", "double ")):
            size = 8
        elif declaration.startswith(("int ", "float ")):
            size = 4
        else:
            _fail("unknown kernel parameter ABI in source submit envelope")
        offset = (offset + size - 1)//size*size + size
    return {"parameters": len(declarations), "bytes": offset}


def _children(expression):
    if isinstance(expression, Binary):
        return (expression.left, expression.right)
    if isinstance(expression, Unary):
        return (expression.operand,)
    if isinstance(expression, IntrinsicCall):
        return expression.arguments
    if isinstance(expression, ArrayAccess):
        return expression.indices
    return ()


def _scalar_safety(lowered, readable, byte_roots=frozenset()):
    """No new value read solely behind an original branch or inactive loop.

    Both exact phases cover a positive point box when the proved tile domain is
    positive. Their point-loop headers therefore execute. Branch-only scalar
    inputs need independent readability or checked byte transport decoded only
    at the original reference. A readonly dummy alone supplies neither.
    """
    parameters = set(lowered.region.parameters)
    unconditional, conditional = set(), set()

    def expression(value, guarded):
        if isinstance(value, Reference) and value.symbol in parameters and not value.symbol.rank:
            (conditional if guarded else unconditional).add(value.symbol)
        for child in _children(value):
            expression(child, guarded)

    def block(value, guarded=False):
        for item in value.statements:
            if isinstance(item, Assignment):
                expression(item.value, guarded)
                if isinstance(item.target, ArrayAccess):
                    expression(item.target, guarded)
            elif isinstance(item, If):
                expression(item.condition, guarded)
                block(item.then_body, True)
                block(item.else_body, True)
            elif isinstance(item, Loop):
                expression(item.lower, guarded)
                expression(item.upper, guarded)
                if type(item.step) is not int:
                    expression(item.step, guarded)
                block(item.body, guarded)
            else:
                _fail("unsupported original control in scalar capture proof")

    for loop in lowered.region.outer_loops:
        expression(loop.lower, False)
        expression(loop.upper, False)
        if type(loop.step) is not int:
            expression(loop.step, False)
    block(lowered.region.producer)
    block(lowered.region.consumer)
    bindings = {symbol: binding for binding, symbol in lowered.bindings}
    for symbol in conditional - unconditional:
        binding = bindings.get(symbol)
        if binding is not None and "parameter" not in binding.attributes and binding.root not in readable and binding.root not in byte_roots:
            _fail("conditional scalar value requires independent initialized-source authority: " + binding.root)


class _Ranges:
    """Emit metadata-only conservative integer/address checks from exact IR."""

    def __init__(self, lowered, scalar, field_index, lower, prefix, *, descriptors="views", envelopes=False):
        self.lowered, self.scalar, self.field_index, self.lower = lowered, scalar, field_index, lower
        self.prefix = prefix
        self.descriptors, self.envelopes = descriptors, [] if envelopes else None
        self.lines = []
        self.serial = 0

    def new(self, value):
        name = f"{self.prefix}r{self.serial}"
        self.serial += 1
        self.lines += [f"const auto {name} = {value};", f"if (!{name}.ok) return FORT_SCOPE_ARGUMENT;"]
        return name

    def expression(self, value, environment):
        if isinstance(value, Literal) and value.dtype is ScalarType.INTEGER:
            return self.new(f"range_value({int(value.value)})")
        if isinstance(value, Reference) and value.symbol.dtype is ScalarType.INTEGER:
            if value.symbol in environment:
                return environment[value.symbol]
            if value.symbol in self.scalar:
                return self.new(f"range_value({self.scalar[value.symbol]})")
        if isinstance(value, Size) and value.symbol in self.field_index:
            index = self.field_index[value.symbol]
            return self.new(f"range_extent({self.descriptors}[{index}].extents[{value.dimension - 1}])")
        if isinstance(value, Unary) and value.operator in {"+", "-"}:
            argument = self.expression(value.operand, environment)
            return argument if value.operator == "+" else self.new(f"range_sub(range_value(0),{argument})")
        if isinstance(value, Binary) and value.operator in {"+", "-", "*"}:
            left, right = self.expression(value.left, environment), self.expression(value.right, environment)
            operation = {"+": "add", "-": "sub", "*": "mul"}[value.operator]
            return self.new(f"range_{operation}({left},{right})")
        if isinstance(value, IntrinsicCall) and value.name.lower() in {"min", "max"}:
            arguments = [self.expression(item, environment) for item in value.arguments]
            result = arguments[0]
            for argument in arguments[1:]:
                result = self.new(f"range_{value.name.lower()}({result},{argument})")
            return result
        _fail("array index/header requires an unsupported or payload-dependent integer value")

    def addresses(self, value, environment, access="reads"):
        if isinstance(value, ArrayAccess) and value.symbol in self.field_index:
            index = self.field_index[value.symbol]
            coordinates = []
            for axis, expression in enumerate(value.indices, 1):
                lb = self.lower[value.symbol, axis]
                # The ordinary lowerer alone introduces this wide descriptor
                # rebase. Check its original index without imposing INT4 on the
                # artificial subtraction of the original lower bound.
                if not (isinstance(expression, Binary) and expression.operator == "+"
                        and expression.right == Literal("1", ScalarType.INTEGER)
                        and isinstance(expression.left, Binary) and expression.left.operator == "-"
                        and expression.left.right == Reference(lb)):
                    _fail("external array lost exact ordinary descriptor rebase")
                bounds = self.expression(expression.left.left, environment)
                self.lines.append(f"if (!inside({bounds},{self.scalar[lb]},{self.descriptors}[{index}].extents[{axis - 1}])) return FORT_SCOPE_ARGUMENT;")
                coordinates.append((bounds, self.scalar[lb]))
            if self.envelopes is not None:
                self.envelopes.append((index, access, tuple(coordinates)))
        elif isinstance(value, ArrayAccess):
            for expression in value.indices:
                self.expression(expression, environment)
        for child in _children(value):
            # Array indices were checked above; the inserted wide rebase must
            # not be re-evaluated as original narrow integer arithmetic.
            if not isinstance(value, ArrayAccess):
                self.addresses(child, environment)

    def loop_terminal(self, item, lower, upper):
        """Additional exact-source consumers may require terminal-value checks."""

    def block(self, value, environment):
        environment = dict(environment)
        for item in value.statements:
            if isinstance(item, Assignment):
                self.addresses(item.value, environment)
                self.addresses(item.target, environment, "writes")
                if isinstance(item.target, Reference) and item.target.symbol.dtype is ScalarType.INTEGER:
                    environment[item.target.symbol] = self.expression(item.value, environment)
            elif isinstance(item, If):
                self.addresses(item.condition, environment)
                self.block(item.then_body, environment)
                self.block(item.else_body, environment)
            elif isinstance(item, Loop):
                if item.step != 1:
                    _fail("point-domain bounds require the original unit positive step")
                lo, hi = self.expression(item.lower, environment), self.expression(item.upper, environment)
                name = self.new(f"range_span({lo},{hi})")
                self.loop_terminal(item, lo, hi)
                nested = {**environment, item.iterator: name}
                self.block(item.body, nested)
            else:
                _fail("unsupported original address-control statement")


_SUPPORT = r"""
struct tile_range { long long lo,hi; bool ok; };
static tile_range range_checked(long long lo,long long hi) {
 return {lo,hi,lo>=INT_MIN && hi<=INT_MAX && lo<=hi};
}
static tile_range range_value(long long x) { return range_checked(x,x); }
static tile_range range_extent(std::size_t x) { return x<=INT_MAX?range_value(x):tile_range{0,0,false}; }
static tile_range range_add(tile_range a,tile_range b) { return range_checked(a.lo+b.lo,a.hi+b.hi); }
static tile_range range_sub(tile_range a,tile_range b) { return range_checked(a.lo-b.hi,a.hi-b.lo); }
static tile_range range_mul(tile_range a,tile_range b) {
 const long long v[]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
 return range_checked(*std::min_element(v,v+4),*std::max_element(v,v+4));
}
static tile_range range_min(tile_range a,tile_range b) { return range_checked(std::min(a.lo,b.lo),std::min(a.hi,b.hi)); }
static tile_range range_max(tile_range a,tile_range b) { return range_checked(std::max(a.lo,b.lo),std::max(a.hi,b.hi)); }
static tile_range range_span(tile_range a,tile_range b) { return range_checked(a.lo,b.hi); }
static bool inside(tile_range x,int lower,std::size_t extent) {
 return x.ok && x.lo>=lower && static_cast<unsigned long long>(x.hi-static_cast<long long>(lower))<extent;
}
static bool multiply(std::size_t a,std::size_t b,std::size_t& result) {
 if (b && a>SIZE_MAX/b) return false; result=a*b; return true;
}
static bool align8(std::size_t& value) {
 if (value>SIZE_MAX-7) return false; value=(value+7)&~std::size_t(7); return true;
}
"""


def emit_tile_task_provider(team, *, mapping="cooperative", readable_scalars=None, pricing=None, placement=None,
                            access_envelopes=False, batch_target_bytes=None, queued_single=False, queued_team=False):
    """Regenerate exact issued source work; accept no caller-supplied IR/code."""
    from compiler.frontend.line_snapshot_team import OriginalLineSnapshotTeamProof

    line_team = type(team) is OriginalLineSnapshotTeamProof
    if type(team) not in {TilePrivateTeamProof, OriginalLineSnapshotTeamProof}:
        _fail("requires an exact compiler-issued whole-team proof")
    team.validate()
    if line_team and (mapping != "thread" or not access_envelopes or pricing is not None
                      or placement is not None or queued_single or queued_team or batch_target_bytes is not None):
        _fail("line snapshot first slice requires direct unpriced thread mapping with exact access envelopes")
    if type(access_envelopes) is not bool:
        _fail("access envelopes require an explicit checked emission choice")
    if access_envelopes and (pricing is not None or placement is not None):
        _fail("access-envelope management costs are unpriced")
    if batch_target_bytes is not None:
        if type(batch_target_bytes) is not int or batch_target_bytes not in EXPERIMENTAL_BATCH_TARGET_BYTES:
            _fail("experimental batch target must be exactly 4, 16 or 64 MiB")
        if not access_envelopes or pricing is not None or placement is not None:
            _fail("experimental batch target requires unpriced access envelopes")
    if type(queued_team) is not bool or (queued_team and queued_single):
        _fail("queued whole-team transport requires its exact exclusive boolean")
    if queued_team and (not access_envelopes or pricing is not None or placement is not None):
        _fail("queued whole-team transport requires unpriced access envelopes")
    if type(queued_single) is not bool:
        _fail("queued transport requires an exact boolean")
    if queued_single and (len(team.units) != 1 or not access_envelopes or pricing is not None
                          or placement is not None or batch_target_bytes is not None):
        _fail("queued single transport requires one unpriced access-envelope unit and default batch target")
    target_bytes = BATCH_TARGET_BYTES if batch_target_bytes is None else batch_target_bytes
    if mapping not in {"cooperative", "thread"}:
        _fail("unknown checked tile mapping")
    readable, byte_roots = set(), set()
    storage_transport = False
    if readable_scalars is not None:
        from compiler.frontend.source_device_tasks import SourceDeviceTaskProof

        if type(readable_scalars) is not SourceDeviceTaskProof or readable_scalars.team is not team:
            _fail("scalar readability requires this team's issued source-device attachment")
        readable_scalars.validate()
        readable = {binding.root for binding in readable_scalars.readable_scalars}
        byte_roots = {binding.root for binding in readable_scalars.byte_scalars}
        storage_transport = readable_scalars.storage_transport
    if storage_transport and (line_team or not access_envelopes or pricing is not None or placement is not None
                              or queued_single or queued_team or batch_target_bytes is not None):
        _fail("source storage transport requires direct unpriced tile access envelopes")
    if byte_roots and (line_team or not access_envelopes or pricing is not None or placement is not None
                       or queued_single or queued_team or batch_target_bytes is not None):
        _fail("byte scalar capture requires direct unpriced tile access envelopes")
    lowered = tuple(lower_tile_private(proof, region_id=index) for index, proof in enumerate(team.units))
    for unit in lowered:
        _scalar_safety(unit, readable, byte_roots)
    suffix = team.identity[:20] + ("_cta" if mapping == "cooperative" else "_thread") + "_batch"
    if batch_target_bytes is not None:
        suffix += f"_b{target_bytes // (1024 * 1024)}m"
    if access_envelopes:
        suffix += "_boxes"
    if pricing is not None:
        from compiler.offload.source_task_pricing import SourceTaskPricing

        if type(pricing) is not SourceTaskPricing or pricing.team is not team or pricing.work.mapping != mapping:
            _fail("pricing requires this exact team's registered source/work attachment")
        pricing.validate()
        suffix += "_q_" + pricing.identity[:10]
    if placement is not None:
        from compiler.frontend.source_placement import SourcePlacement

        if (type(placement) is not SourcePlacement or placement.proof is not readable_scalars
                or placement.pricing is not pricing or placement.proof.team is not team):
            _fail("placement requires this exact registered graph/source/pricing binding")
        placement.validate()
        # The placement already binds pricing. Keep both C/Fortran symbols
        # within Fortran's 63-character identifier limit.
        suffix = team.identity[:12] + ("_cta" if mapping == "cooperative" else "_thread") + "_g_" + placement.identity[:16]
    if queued_single:
        suffix += "_q1"
    elif queued_team:
        suffix += "_qt"
    if byte_roots:
        # One team can have different capture states in different checked scopes.
        suffix = (team.identity[:10] + ("_cta" if mapping == "cooperative" else "_thread")
                  + "_s" + readable_scalars.identity[:8] + "_batch_boxes_bytes")
    if storage_transport:
        suffix = (team.identity[:10] + ("_cta" if mapping == "cooperative" else "_thread")
                  + "_s" + readable_scalars.identity[:8] + "_batch_storage")
    entry, module = "fort_tile_task_" + suffix, "fort_tile_module_" + suffix
    parameters = []
    for binding in team.scalars:
        parameters.append(TileParameter("scalar", binding, None, f"p{len(parameters)}", _iso(binding)))
    for binding, _ in team.fields:
        for axis in range(1, binding.rank + 1):
            parameters.append(TileParameter("external_lower", binding, axis, f"p{len(parameters)}", "c_int"))
    for binding in team.scratch_bindings:
        for axis in range(1, binding.rank + 1):
            for role in ("scratch_lower", "scratch_upper"):
                parameters.append(TileParameter(role, binding, axis, f"p{len(parameters)}", "c_int"))
    parameters = tuple(parameters)
    param = {(item.role, item.binding.root, item.axis): item for item in parameters}
    field_index = {binding.root: index for index, (binding, _) in enumerate(team.fields)}
    ctype = {"c_int": "int", "c_float": "float", "c_double": "double"}
    arguments = ["fort_scope_t context", *[f"fort_buffer_t field{i}" for i in range(len(team.fields))],
                 *[f"const {ctype[p.kind]}* {p.name}" for p in parameters], "fort_scope_external_report_v1* report"]
    def byte_parameter(parameter):
        return parameter.role == "scalar" and parameter.binding.root in byte_roots

    state_fields = [f"fort_scalar_bytes<{ctype[p.kind]}> {p.name};" if byte_parameter(p)
                    else f"{ctype[p.kind]} {p.name};" for p in parameters]
    state_fields += [f"std::size_t total[{len(lowered)}],stride[{len(lowered)}];", "std::size_t batch;", "unsigned char* scratch;"]
    kernels, checks, submits, plan, envelopes = [], [], [], [], []
    queued_launches = []
    scalar_loads, loaded = [], set()

    def load(parameter):
        if parameter.name in loaded:
            return
        loaded.add(parameter.name)
        capture = (f"std::memcpy(state.{parameter.name}.bytes,{parameter.name},sizeof(state.{parameter.name}.bytes));"
                   if byte_parameter(parameter) else f"state.{parameter.name}=*{parameter.name};")
        scalar_loads.extend([f"if (!{parameter.name}) return fail(FORT_SCOPE_ARGUMENT);", capture])

    for index, unit in enumerate(lowered):
        region = unit.region
        scalar_bytes = frozenset(symbol for binding, symbol in unit.bindings if binding.root in byte_roots)
        if mapping == "cooperative":
            cooperative = prove_cooperative_tile_private(team.units[index], region_id=index)
            kernels += generate_cooperative_kernel(region, cooperative, addressing=unit.addressing, scalar_bytes=scalar_bytes)
            kernel = f"kernel_tile_private_cta_{index}"
        else:
            kernels += generate_kernel(region, addressing=unit.addressing, scalar_bytes=scalar_bytes)
            kernel = f"kernel_tile_private_{index}"
        symbols = {symbol: binding for binding, symbol in unit.bindings}
        scalar = {symbol: f"state.{param['scalar', binding.root, None].name}" for binding, symbol in unit.bindings
                  if symbol in region.parameters and not symbol.rank}
        for binding, axis, symbol in unit.lower_bounds:
            scalar[symbol] = f"state.{param['external_lower', binding.root, axis].name}"
        fields = {symbol: field_index[binding.root] for binding, symbol in unit.bindings if symbol.rank and symbol in region.parameters}
        bounds = {(symbol, axis): lb for binding, axis, lb in unit.lower_bounds
                  for original, symbol in unit.bindings if original is binding}
        if line_team:
            from compiler.emission.cuda.line_snapshot import _LineRanges

            line_scratch = symbols[region.scratch_symbols[0]]
            line_lb = param['scratch_lower', line_scratch.root, 1]
            line_ub = param['scratch_upper', line_scratch.root, 1]
            line_extent = f"static_cast<std::size_t>(static_cast<long long>(state.{line_ub.name})-state.{line_lb.name})+1"
            ranges = _LineRanges(unit, scalar, fields, bounds, f"u{index}_", descriptors="layouts", envelopes=True,
                                 scratch_lower=f"state.{line_lb.name}", scratch_extent=f"({line_extent})")
        else:
            ranges = _Ranges(unit, scalar, fields, bounds, f"u{index}_",
                             descriptors="layouts" if access_envelopes else "views", envelopes=access_envelopes)
        environment = {}
        for axis, loop in enumerate(region.outer_loops):
            # Same original common domain for every unit, but each unit retains
            # its own ordinary symbol/axis ordering in the kernel.
            used = []

            def collect(expr, symbols=symbols, used=used):
                if isinstance(expr, Reference) and expr.symbol in symbols and not expr.symbol.rank:
                    used.append(param['scalar', symbols[expr.symbol].root, None])
                for child in _children(expr):
                    collect(child)

            for expression in (loop.lower, loop.upper, loop.step):
                if type(expression) is not int:
                    collect(expression)
            for parameter in used:
                load(parameter)
            geometry = _Ranges(unit, scalar, fields, bounds, f"g{index}_{axis}_")
            lo = geometry.expression(loop.lower, {})
            hi = geometry.expression(loop.upper, {})
            step = geometry.new(f"range_value({loop.step})") if type(loop.step) is int else geometry.expression(loop.step, {})
            # Geometry is loaded and checked before the next original axis.
            plan += scalar_loads
            scalar_loads.clear()
            plan += [line.replace("return FORT_SCOPE_ARGUMENT;", "return fail(FORT_SCOPE_ARGUMENT);") for line in geometry.lines]
            plan += [f"if ({step}.lo<=0) return fail(FORT_SCOPE_ARGUMENT);",
                     f"if ({lo}.lo>{hi}.hi) return fail(FORT_SCOPE_OK);",
                     f"state.lo{index}_{axis}=static_cast<int>({lo}.lo); state.hi{index}_{axis}=static_cast<int>({hi}.hi); state.step{index}_{axis}=static_cast<int>({step}.lo);",
                     f"state.extent{index}_{axis}=static_cast<std::size_t>(({hi}.hi-{lo}.lo)/{step}.lo)+1;"]
            if line_team:
                plan += [f"if ({hi}.hi==INT_MAX || state.extent{index}_{axis}>INT_MAX) return fail(FORT_SCOPE_ARGUMENT);"]
            state_fields += [f"int lo{index}_{axis},hi{index}_{axis},step{index}_{axis}; std::size_t extent{index}_{axis};"]
            environment[loop.iterator] = f"range_span(range_value(state.lo{index}_{axis}),range_value(state.hi{index}_{axis}))"
        plan += [f"state.total[{index}]=1;"]
        for axis in range(len(region.outer_loops)):
            plan += [f"if (!multiply(state.total[{index}],state.extent{index}_{axis},state.total[{index}])) return fail(FORT_SCOPE_RESOURCE);"]
        # Scratch descriptor reads follow all original nonempty-domain checks.
        plan += [f"state.stride[{index}]=0;"]
        for scratch_index, symbol in enumerate(region.scratch_symbols):
            binding = symbols[symbol]
            state_fields.append(f"std::size_t offset{index}_{scratch_index};")
            plan += [f"if (!align8(state.stride[{index}])) return fail(FORT_SCOPE_RESOURCE);",
                     f"state.offset{index}_{scratch_index}=state.stride[{index}];", "{ std::size_t bytes=" + str(4 if symbol.dtype is ScalarType.REAL32 else 8) + ";"]
            axes = None if line_team else team.units[index].public()["axes"]
            by_name = {} if line_team else {axis["tile"]: number for number, axis in enumerate(axes)}
            for axis in range(1, symbol.rank + 1):
                lb = param['scratch_lower', binding.root, axis]
                ub = param['scratch_upper', binding.root, axis]
                load(lb)
                load(ub)
                plan += scalar_loads
                scalar_loads.clear()
                if line_team:
                    plan += [f"if (state.{lb.name}!={unit.line_snapshot_contract.scratch_lower} || state.{ub.name}<state.{lb.name}) return fail(FORT_SCOPE_ARGUMENT);",
                             f"if (static_cast<long long>(state.{ub.name})-state.{lb.name}+1>INT_MAX) return fail(FORT_SCOPE_ARGUMENT);",
                             f"if (!multiply(bytes,static_cast<std::size_t>(static_cast<long long>(state.{ub.name})-state.{lb.name})+1,bytes)) return fail(FORT_SCOPE_RESOURCE);"]
                else:
                    outer_axis = by_name[team.units[index].public()["scratch_axis_order"][symbol.name.lower()][axis - 1]]
                    coverage = axes[outer_axis]
                    plan += [f"if (state.{lb.name}>{coverage['scratch_index_lower']} || static_cast<long long>(state.{ub.name})<static_cast<long long>(state.step{index}_{outer_axis})+{coverage['scratch_index_upper_width_offset']}) return fail(FORT_SCOPE_RESOURCE);",
                             f"if (state.{ub.name}<state.{lb.name}) return fail(FORT_SCOPE_RESOURCE);",
                             f"if (!multiply(bytes,static_cast<std::size_t>(static_cast<long long>(state.{ub.name})-state.{lb.name})+1,bytes)) return fail(FORT_SCOPE_RESOURCE);"]
            plan += [f"if (bytes>SIZE_MAX-state.stride[{index}]) return fail(FORT_SCOPE_RESOURCE); state.stride[{index}]+=bytes;", "}"]
        plan += [f"if (!align8(state.stride[{index}])) return fail(FORT_SCOPE_RESOURCE);"]
        if line_team:
            source_field = field_index[unit.line_snapshot_contract.proof.source_binding.root]
            line_axis = unit.line_snapshot_contract.line_axis
            checks += [f"if (layouts[{source_field}].extents[{line_axis-1}]!=({line_extent})) return FORT_SCOPE_ARGUMENT;"]
            for symbol, field_number in fields.items():
                for axis in range(1, symbol.rank+1):
                    lb = scalar[bounds[symbol, axis]]
                    extent = f"layouts[{field_number}].extents[{axis-1}]"
                    checks += [f"if (!{extent} || {extent}>INT_MAX || static_cast<long long>({lb})+static_cast<long long>({extent})-1>INT_MAX) return FORT_SCOPE_ARGUMENT;"]
        ranges.block(region.producer, environment)
        ranges.block(region.consumer, environment)
        checks += ranges.lines
        if access_envelopes:
            envelopes += ranges.envelopes
        values = []
        for argument in abi_arguments(region.parameters):
            symbol = argument.symbol
            if symbol.rank:
                field = fields[symbol]
                values.append((f"layouts[{field}].extents[{argument.dimension - 1}]" if argument.dimension else f"static_cast<{cpp_type(symbol)}*>(devices[{field}])") if (queued_single or queued_team) else
                              (f"views[{field}].extents[{argument.dimension - 1}]" if argument.dimension else f"static_cast<{cpp_type(symbol)}*>(views[{field}].device)"))
            else:
                values.append(scalar[symbol])
        values += [value for axis in range(len(region.outer_loops)) for value in
                   (f"state.lo{index}_{axis}", f"state.step{index}_{axis}", f"state.extent{index}_{axis}")]
        values += ["state.scratch", f"state.stride[{index}]"]
        for scratch_index, symbol in enumerate(region.scratch_symbols):
            binding = symbols[symbol]
            values.append(f"state.offset{index}_{scratch_index}")
            for axis in range(1, symbol.rank + 1):
                lb, ub = param['scratch_lower', binding.root, axis], param['scratch_upper', binding.root, axis]
                values += [f"state.{lb.name}", f"static_cast<std::size_t>(static_cast<long long>(state.{ub.name})-state.{lb.name})+1"]
        values += ["first", "count"]
        if queued_team:
            queued_launches.append((kernel, tuple(values), _launch_abi(region)))
        blocks = "count" if mapping == "cooperative" else "count/256+(count%256!=0)"
        submits += [f"for (std::size_t first=0; first<state.total[{index}];) {{",
                    f"const std::size_t count=std::min(state.batch,state.total[{index}]-first);",
                    f"const unsigned blocks=static_cast<unsigned>(std::min<std::size_t>({blocks},65535));",
                    "*started=1;",
                    f"{kernel}<<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(" + ",".join(values) + ");",
                    "if (cudaGetLastError()!=cudaSuccess) return FORT_SCOPE_EXECUTION;", "++*accepted;", "first+=count;", "}"]
    for parameter in parameters:
        load(parameter)
    plan += scalar_loads
    bindings = []
    for index, (binding, written) in enumerate(team.fields):
        dtype = {("real", 4): "FORT_SCOPE_REAL32", ("real", 8): "FORT_SCOPE_REAL64", ("integer", 4): "FORT_SCOPE_INTEGER32"}.get((binding.dtype, binding.kind))
        if dtype is None:
            _fail("unsupported original field element representation")
        flags = "0" if access_envelopes else "FORT_SCOPE_READ_ALL" + ("|FORT_SCOPE_WRITE_ALL" if written else "")
        bindings.append(f"{{field{index},{dtype},{binding.rank},{binding.kind},{{{flags},0,nullptr,0,nullptr,0,nullptr}}}}")
    if queued_team:
        from compiler.emission.cuda.queued_team_task_provider import queued_team_provider

        return queued_team_provider(team=team, entry=entry, module=module, parameters=parameters,
            state_fields=state_fields, kernels=kernels, plan=plan, checks=checks, envelopes=envelopes,
            bindings=bindings, launches=queued_launches, mapping=mapping, support=_SUPPORT,
            readable_scalars=readable_scalars, target_bytes=target_bytes)
    if queued_single:
        from compiler.emission.cuda.queued_tile_task_provider import queued_single_provider

        return queued_single_provider(team=team, entry=entry, module=module, parameters=parameters,
            state_fields=state_fields, kernels=kernels, plan=plan, checks=checks, envelopes=envelopes,
            bindings=bindings, values=values, kernel=kernel, mapping=mapping, support=_SUPPORT,
            readable_scalars=readable_scalars, launch_abi=_launch_abi(lowered[0].region), target_bytes=target_bytes)
    from compiler.emission.cuda.scalar_bytes import support_code as scalar_byte_support

    source = ["#include <cuda_runtime.h>", '#include "scoped_runtime.h"', '#include "common_functions.cuh"',
              '#include "numeric.hpp"', "#include <algorithm>", "#include <climits>", "#include <cstdint>",
              "using generated_kernels::indexing::F_IDX;", *(scalar_byte_support() if byte_roots else []), "namespace {", _SUPPORT,
              "struct tile_state {", *state_fields, "};", *kernels,
              "static int submit(void* opaque,const fort_scope_external_view_v1* views,std::size_t count,void* stream,int* started,std::uint64_t* accepted) {",
              "*started=0; *accepted=0;", "auto& state=*static_cast<tile_state*>(opaque);",
              f"if (count!={len(team.fields)}) return FORT_SCOPE_ARGUMENT;", *checks, *submits,
              "return FORT_SCOPE_OK;", "}", "}", f'extern "C" int {entry}(' + ",".join(arguments) + ") {",
              "if (!report) return FORT_SCOPE_ARGUMENT;", "*report={1,FORT_SCOPE_ARGUMENT,0,0};",
              "auto fail=[&](int status) { report->status=status; return status; };",
              "if (!fort_scope_serial_caller() || !fort_scope_numerical_environment_supported()) return fail(FORT_SCOPE_BOUNDARY);",
              "tile_state state{};", *plan,
              "std::size_t max_stride=0,max_tiles=0,scratch_bytes=0; std::uint64_t expected=0;",
              f"for (std::size_t unit=0;unit<{len(lowered)};++unit) {{",
              "max_stride=std::max(max_stride,state.stride[unit]); max_tiles=std::max(max_tiles,state.total[unit]);", "}",
              "if (!max_stride || !max_tiles) return fail(FORT_SCOPE_RESOURCE);",
              f"state.batch=std::min(max_tiles,std::max<std::size_t>(1,{target_bytes}/max_stride));",
              "if (!multiply(max_stride,state.batch,scratch_bytes)) return fail(FORT_SCOPE_RESOURCE);",
              f"for (std::size_t unit=0;unit<{len(lowered)};++unit) {{",
              "const std::size_t launches=state.total[unit]/state.batch+(state.total[unit]%state.batch!=0);",
              "if (launches>UINT64_MAX-expected) return fail(FORT_SCOPE_RESOURCE); expected+=launches;", "}",
              "int status=fort_scope_wait(context); if (status!=FORT_SCOPE_OK) return fail(status);",
              "fort_scope_scratch_lease_v1 lease{}; status=fort_scope_scratch_acquire_v1(context,scratch_bytes,&lease);",
              "if (status!=FORT_SCOPE_OK) return fail(status);", "state.scratch=static_cast<unsigned char*>(lease.device);",
              "fort_scope_external_binding_v1 bindings[]={" + ",".join(bindings) + "};",
              "fort_scope_external_call_v1 call{1,0,bindings," + str(len(bindings)) + ",0,0};",
              "status=fort_scope_source_execute_v1(context,&call,expected,submit,&state,report);",
              "const int released=fort_scope_scratch_release_v1(context,lease.token);",
              "if (released!=FORT_SCOPE_OK) { report->status=FORT_SCOPE_EXECUTION; report->started=1; return FORT_SCOPE_EXECUTION; }",
              "return status;", "}", ""]
    if access_envelopes:
        from compiler.emission.cuda.tile_access import access_preflight

        # Descriptor and address preflight precede CUDA initialization, scratch
        # allocation and payload acquisition. The callback only submits work.
        source[0:0] = ["#include <vector>", "#include <new>"]
        start = source.index("static int submit(void* opaque,const fort_scope_external_view_v1* views,std::size_t count,void* stream,int* started,std::uint64_t* accepted) {")
        source[start:start] = access_preflight(team.fields, checks, envelopes)
        # Exact source-generated range checks now run against metadata layouts
        # in prepare_tile_access, rather than after device acquisition.
        start = source.index("static int submit(void* opaque,const fort_scope_external_view_v1* views,std::size_t count,void* stream,int* started,std::uint64_t* accepted) {")
        del source[start + 4:start + 4 + len(checks)]
        binding_line = "fort_scope_external_binding_v1 bindings[]={" + ",".join(bindings) + "};"
        source.remove(binding_line)
        insertion = source.index("int status=fort_scope_wait(context); if (status!=FORT_SCOPE_OK) return fail(status);")
        source[insertion:insertion] = [
            binding_line, f"OwnedTileAccess accesses[{len(team.fields)}];",
            "const int access_status=prepare_tile_access(context,state,bindings,accesses);",
            "if (access_status!=FORT_SCOPE_OK) return fail(access_status);",
        ]
    if placement is not None:
        # CPU is the authoritative prework alternative. GPU is reported only
        # when the shared runtime transaction actually attempted device work.
        # A prework rejection never invents a committed GPU placement.
        function = next(index for index, line in enumerate(source) if line.startswith('extern "C" int ' + entry))
        source[function] = source[function].replace("fort_scope_external_report_v1* report",
            "int* selected_backend,fort_scope_external_report_v1* report")
        source[function + 1:function + 1] = ["if (!selected_backend) return FORT_SCOPE_ARGUMENT;", "*selected_backend=0;"]
        submitted = source.index("status=fort_scope_source_execute_v1(context,&call,expected,submit,&state,report);")
        source.insert(submitted + 1, "if (report->started) *selected_backend=1;")
    iso_type = {"c_int": "integer(c_int)", "c_float": "real(c_float)", "c_double": "real(c_double)"}
    names = ["context", *[f"field{i}" for i in range(len(team.fields))], *[p.name for p in parameters], "report"]
    fortran = [f"module {module}", "use iso_c_binding", "use fort_scoped_memory, only: fort_scope_external_report_v1", "implicit none", "interface",
               f"integer(c_int) function {entry}( &", " & " + ", &\n & ".join(names) + f') bind(c,name="{entry}")',
               "import", "integer(c_int64_t),value :: context", *[f"integer(c_int64_t),value :: field{i}" for i in range(len(team.fields))],
               *[f"{iso_type[p.kind]},intent(in) :: {p.name}" for p in parameters],
               "type(fort_scope_external_report_v1),intent(out) :: report", "end function", "end interface", "end module", ""]
    if placement is not None:
        fortran[6] = fortran[6].replace("report)", "selected_backend, &\n & report)")
        fortran.insert(fortran.index("type(fort_scope_external_report_v1),intent(out) :: report"),
                       "integer(c_int),intent(out) :: selected_backend")
    if pricing is not None:
        from compiler.emission.cuda.tile_pricing import source_pricing_code

        support, functions, quote = source_pricing_code(pricing, team=team, entry=entry,
                                                       launch_abi=tuple(_launch_abi(unit.region) for unit in lowered))
        source[source.index("namespace {"):source.index("namespace {")] = support
        insertion = next(index for index, line in enumerate(source) if line.startswith("static int submit("))
        source[insertion:insertion] = functions
        binding_line = "fort_scope_external_binding_v1 bindings[]={" + ",".join(bindings) + "};"
        call_line = "fort_scope_external_call_v1 call{1,0,bindings," + str(len(bindings)) + ",0,0};"
        source.remove(binding_line)
        source.remove(call_line)
        insertion = source.index("int status=fort_scope_wait(context); if (status!=FORT_SCOPE_OK) return fail(status);")
        if pricing.native_live:
            source[insertion] = "status=fort_scope_wait(context); if (status!=FORT_SCOPE_OK) return fail(status);"
            source[insertion:insertion] = ["int status=FORT_SCOPE_OK;", binding_line, call_line, *quote]
        else:
            source[insertion:insertion] = [binding_line, call_line]
            source[insertion + 3:insertion + 3] = quote
    native_prepare = None
    if pricing is not None and pricing.native_live:
        native_prepare = "fort_tile_native_" + sha256(entry.encode()).hexdigest()[:20]
        source += [f'extern "C" int {native_prepare}(fort_scope_t context,' +
                   ",".join(f"fort_buffer_t field{i}" for i in range(len(team.fields))) +
                   ",std::uint64_t* token) {", binding_line, call_line,
                   "return fort_scope_source_host_prepare_v1(context,&call,token);", "}"]
        native_names = ["context", *[f"field{i}" for i in range(len(team.fields))], "token"]
        insertion = fortran.index("end interface")
        fortran[insertion:insertion] = [f"integer(c_int) function {native_prepare}( &",
            " & " + ", &\n & ".join(native_names) + f') bind(c,name="{native_prepare}")',
            "import", "integer(c_int64_t),value :: context",
            *[f"integer(c_int64_t),value :: field{i}" for i in range(len(team.fields))],
            "integer(c_int64_t),intent(out) :: token", "end function"]
    if storage_transport:
        original = "status=fort_scope_source_execute_v1(context,&call,expected,submit,&state,report);"
        if source.count(original) != 1:
            _fail("source storage transport lost its exact transaction")
        source[source.index(original)] = original.replace("source_execute_v1", "source_execute_v2")
    cuda_source, fortran_source = "\n".join(source), "\n".join(fortran)
    record = {"version": 2, "kind": "original_tile_private_team", "team_identity": team.identity,
              "scalar_readability_identity": readable_scalars.identity if readable_scalars is not None else None,
              "unit_proof_identities": [unit.identity for unit in team.units], "mapping": mapping,
              "entry": entry, "module": module, "parameters": [item.public() for item in parameters],
              "cuda_sha256": sha256(cuda_source.encode()).hexdigest(), "fortran_sha256": sha256(fortran_source.encode()).hexdigest(),
              "transaction": "fort_scope_source_execute_v2" if storage_transport else "fort_scope_source_execute_v1", "scratch": "bounded original tile batches; same-stream source-order units",
              "batching": {"version": 1 if batch_target_bytes is None else 2, "target_bytes": target_bytes, "minimum_tiles": 1,
                           "planned_launches": "sum ceil(unit_tiles/batch_tiles)"},
              "definitions_authorized": False, "placement_authorized": False, "pricing": "unpriced; explicit sections only"}
    if pricing is not None:
        record.update(version=3, pricing=pricing.public())
    if native_prepare is not None:
        record.update(version=4, native_prepare_entry=native_prepare,
            native_transaction_api={name: "fort_scope_source_host_" + name + "_v1"
                                    for name in ("prepare", "start", "complete", "cancel")})
    if pricing is not None and pricing.public()["version"] == 4:
        record.update(version=5, launch_abi={"version": 1, "pointer_bytes": 8, "size_t_bytes": 8,
                                            "units": [_launch_abi(unit.region) for unit in lowered]})
    if placement is not None:
        record.update(version=6, placement=placement.public(),
                      selected_backend={"argument": "selected_backend", "cpu": 0, "gpu": 1})
    if access_envelopes:
        record.update(version=7 if batch_target_bytes is None else 8, access_envelopes={
            "version": 1, "source": "exact ordinary lowered original indices",
            "reads": "conservative per-access Cartesian envelopes",
            "writes": "conservative per-access Cartesian envelopes; preserve prior values",
            "overwrites_authorized": False, "initialization_authorized": False,
            "preflight": "metadata only before CUDA, scratch and payload acquisition",
            "pricing": "unpriced; explicit sections only",
        })
    if byte_roots or storage_transport:
        record.update(version=9, scalar_transport={
            "version": 1, "resources": [b.root for b in readable_scalars.byte_scalars],
            "host": "copy object representation after positive domain; no typed real value read",
            "device": "decode at each original scalar reference; retain original control",
            "abi": "trivial aligned unsigned-char aggregate; original scalar size and alignment",
            "definitions_authorized": False})
    if storage_transport:
        record.update(version=10, field_transport={
            "version": 2, "source_proof_identity": readable_scalars.identity,
            "regions": "original physical may-access bytes; preserve partial writes without initialization",
            "conditional_reads": "original source control retained",
            "definitions_authorized": False, "pricing_authorized": False,
        })
    if line_team:
        record.update(kind="original_line_snapshot_team", line_snapshot_prework={
            "version": 1, "source": "exact original line proof and regenerated producer/consumer IR",
            "domain": "original independent transverse loops; no tile width or MIN certificate",
            "scratch": "reached original rank-one descriptor; exact source-line extent; bounded existing lease",
            "integer": "checked original operations, coordinates and terminal values before payload acquisition",
            "initialization_authorized": False, "pricing_authorized": False, "cooperative_mapping_authorized": False})
    return TileTaskProvider(entry, module, cuda_source, fortran_source, tuple(team.fields), parameters,
                            json.dumps(record, sort_keys=True, separators=(",", ":")))
