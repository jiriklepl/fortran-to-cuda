"""Correctness-first indexed scratch kernels: one CUDA thread per logical tile.

Original producer/consumer statements remain serial and ordered within a tile.
This emits no allocation, host mirror or recomputation. The owning host wrapper
must acquire a bounded runtime scratch lease before work, keep it live through
ordered same-stream batches, record accepted work and release/abandon correctly.
"""

from __future__ import annotations

from dataclasses import replace

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments, dimension_name
from compiler.emission.common.c_family import cpp_type, indent, wide_iterator_name
from compiler.emission.common.loops import sequential_block
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    If,
    IntrinsicCall,
    Literal,
    Loop,
    Reference,
    ScalarType,
    Symbol,
    Unary,
)
from compiler.ir.plan import RegionAddressing, SubscriptDecision
from compiler.ir.tile_private import TilePrivateRegion


def _bounds(region):
    occupied = (
        *region.parameters,
        *region.private_symbols,
        *region.scratch_symbols,
        *(loop.iterator for loop in region.outer_loops),
    )
    first = max((symbol.id for symbol in occupied), default=0) + 1
    return {
        (array, axis): Symbol(first + offset, f"tile_lower_{array.name}_{axis}", ScalarType.INTEGER)
        for offset, (array, axis) in enumerate(
            (array, axis) for array in region.scratch_symbols for axis in range(array.rank)
        )
    }


def kernel_parameters(region):
    parameters = [cpp_declaration(value) for value in abi_arguments(region.parameters)]
    for axis in range(len(region.outer_loops)):
        parameters += [f"int fort_tile_lower{axis}", f"int fort_tile_step{axis}", f"std::size_t fort_tile_extent{axis}"]
    parameters += ["unsigned char* fort_tile_scratch", "std::size_t fort_tile_stride"]
    bounds = _bounds(region)
    for index, array in enumerate(region.scratch_symbols):
        parameters.append(f"std::size_t fort_tile_offset{index}")
        for axis in range(array.rank):
            parameters += [f"int {bounds[array, axis].cpp_name}", f"std::size_t {dimension_name(array, axis + 1)}"]
    return parameters


def kernel_argument_names(region):
    return (
        [value.name for value in abi_arguments(region.parameters)]
        + [
            value
            for axis in range(len(region.outer_loops))
            for value in (f"fort_tile_lower{axis}", f"fort_tile_step{axis}", f"fort_tile_extent{axis}")
        ]
        + ["fort_tile_scratch", "fort_tile_stride"]
        + [
            value
            for index, array in enumerate(region.scratch_symbols)
            for value in (
                f"fort_tile_offset{index}",
                *(
                    value
                    for axis in range(array.rank)
                    for value in (_bounds(region)[array, axis].cpp_name, dimension_name(array, axis + 1))
                ),
            )
        ]
    )


def _scratch_blocks(region, addressing=None):
    bounds = _bounds(region)
    decisions = list(addressing.decisions if addressing else ())

    def expr(value):
        if isinstance(value, ArrayAccess):
            indices = tuple(expr(x) for x in value.indices)
            if value.symbol in region.scratch_symbols:
                rebased = []
                for axis, index in enumerate(indices):
                    item = Binary(
                        "+", Binary("-", index, Reference(bounds[value.symbol, axis])), Literal("1", ScalarType.INTEGER)
                    )
                    # Widen only address arithmetic; preserve the original scalar
                    # expression/evaluation tree and do not hoist any loads.
                    decisions.append(
                        SubscriptDecision(item, "wide", None, "checked original scratch descriptor rebase")
                    )
                    rebased.append(item)
                indices = tuple(rebased)
            return replace(value, indices=indices)
        if isinstance(value, Binary):
            return replace(value, left=expr(value.left), right=expr(value.right))
        if isinstance(value, Unary):
            return replace(value, operand=expr(value.operand))
        if isinstance(value, IntrinsicCall):
            return replace(value, arguments=tuple(expr(x) for x in value.arguments))
        return value

    def block(value):
        result = []
        for statement in value.statements:
            if isinstance(statement, Assignment):
                result.append(replace(statement, target=expr(statement.target), value=expr(statement.value)))
            elif isinstance(statement, Loop):
                result.append(
                    replace(
                        statement,
                        lower=expr(statement.lower),
                        upper=expr(statement.upper),
                        body=block(statement.body),
                        step=statement.step if type(statement.step) is int else expr(statement.step),
                    )
                )
            elif isinstance(statement, If):
                result.append(
                    replace(
                        statement,
                        condition=expr(statement.condition),
                        then_body=block(statement.then_body),
                        else_body=block(statement.else_body),
                    )
                )
            else:
                raise TypeError("unsupported tile-private ordered IR statement")
        return Block(tuple(result))

    producer, consumer = block(region.producer), block(region.consumer)
    result = RegionAddressing(
        tuple(decisions),
        addressing.iterator_ranges if addressing else (),
        addressing.wide_iterators if addressing else (),
    )
    return producer, consumer, result


def generate_tile_function(region: TilePrivateRegion, *, addressing=None, device=True):
    """Emit one logical tile; usable by the CUDA kernel or a CPU equivalence probe.

    The input IR is supplied by ordinary trusted lowering of proof.phases; the
    issuer is revalidated here. This function does not invent a lowering or an
    initialized-state proof from caller-supplied descriptions.
    """
    from compiler.frontend.lowering import lower_tile_private

    if type(region) is not TilePrivateRegion:
        raise ValueError("tile-private emission requires its exact source-lowered region")
    proof = region.source_proof.validate()
    lowered = lower_tile_private(proof, region_id=region.id)
    if region != lowered.region:
        raise ValueError("tile-private emission received substitute source-lowered work")
    if addressing is not None and addressing != lowered.addressing:
        raise ValueError("tile-private addressing differs from original source lowering")
    addressing = lowered.addressing
    public = proof.public()
    from compiler.frontend.line_snapshot import OriginalLineSnapshotProof
    if type(proof) is OriginalLineSnapshotProof:
        expected_axes = [x["iterator"] for x in public["transverse_domain"]]
        expected_scratch = {public["snapshot"]["scratch"]["name"]}
    else:
        expected_axes = [x["tile"] for x in public["axes"]]
        expected_scratch = {x["name"] for x in public["scratch"]}
    if [x.iterator.name.lower() for x in region.outer_loops] != expected_axes or {
        x.name.lower() for x in region.scratch_symbols
    } != expected_scratch:
        raise ValueError("tile-private lowered storage/axes differ from original source proof")
    producer, consumer, addressing = _scratch_blocks(region, addressing)
    parameters = kernel_parameters(region) + ["std::size_t fort_tile_slot", "std::size_t fort_tile_ordinal"]
    prefix = "static __device__" if device else "static"
    lines = [
        f"{prefix} void fort_tile_one_{region.id}(" + ", ".join(parameters) + ") {",
        "    std::size_t fort_tile_remaining = fort_tile_ordinal;",
    ]
    outer = {loop.iterator for loop in region.outer_loops}
    for axis in reversed(range(len(region.outer_loops))):
        loop = region.outer_loops[axis]
        lines += [
            f"    const std::size_t fort_tile_axis{axis} = fort_tile_remaining % fort_tile_extent{axis};",
            f"    fort_tile_remaining /= fort_tile_extent{axis};",
            f"    const int {loop.iterator.cpp_name} = static_cast<int>(static_cast<long long>(fort_tile_lower{axis})"
            f" + static_cast<long long>(fort_tile_axis{axis}) * fort_tile_step{axis});",
        ]
        if loop.iterator in addressing.wide_iterators:
            lines.append(f"    const long long {wide_iterator_name(loop.iterator)} = {loop.iterator.cpp_name};")
    for symbol in region.private_symbols:
        if symbol not in outer and symbol not in region.scratch_symbols:
            if symbol.rank:
                raise ValueError("unproved additional ranked private storage in tile IR")
            lines.append(f"    {cpp_type(symbol)} {symbol.cpp_name};")
    for index, symbol in enumerate(region.scratch_symbols):
        lines.append(
            f"    {cpp_type(symbol)}* {symbol.cpp_name} = reinterpret_cast<{cpp_type(symbol)}*>("
            f"fort_tile_scratch + fort_tile_slot * fort_tile_stride + fort_tile_offset{index});"
        )
    lines += ["    // Complete original producer phase; no duplicated or omitted evaluations."]
    lines += sequential_block(producer, 1, [0], device=device, addressing=addressing)
    lines += ["    // Same-thread phase barrier: producer completes before original consumer."]
    lines += sequential_block(consumer, 1, [1000], device=device, addressing=addressing)
    return lines + ["}", ""]


def generate_kernel(region: TilePrivateRegion, *, addressing=None):
    lines = generate_tile_function(region, addressing=addressing)
    parameters = kernel_parameters(region) + ["std::size_t fort_tile_first", "std::size_t fort_tile_count"]
    lines += [
        f"static __global__ void kernel_tile_private_{region.id}(" + ", ".join(parameters) + ") {",
        "    const std::size_t fort_tile_grid_stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;",
        "    std::size_t fort_tile_slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;",
        "    while (fort_tile_slot < fort_tile_count) {",
    ]
    arguments = kernel_argument_names(region) + ["fort_tile_slot", "fort_tile_first + fort_tile_slot"]
    lines += indent(
        [
            f"fort_tile_one_{region.id}(" + ", ".join(arguments) + ");",
            "if (fort_tile_count - fort_tile_slot <= fort_tile_grid_stride) break;",
            "fort_tile_slot += fort_tile_grid_stride;",
        ],
        2,
    )
    return lines + ["    }", "}", ""]


def generate_launch(region, arguments, *, context, stream, first="fort_tile_first", count="fort_tile_count"):
    """A direct same-stream submit body for scratch_batch_helper's callback.

    ``arguments`` are the ordered values of kernel_parameters(), with ordinary
    registered arrays already replaced by their admitted device pointers.
    Acquiring fields, source guards and draining any earlier deferred queue
    remain the owning entry's responsibility. No callback RESOURCE is recoverable
    after this body is invoked.
    """
    if len(arguments) != len(kernel_parameters(region)):
        raise ValueError("tile-private kernel argument inventory differs")
    return [
        f"if ({count} == 0) return FORT_SCOPE_OK;",
        f"const std::size_t fort_tile_needed_blocks = ({count} - 1) / 128 + 1;",
        "const unsigned fort_tile_blocks = static_cast<unsigned>(fort_tile_needed_blocks > 65535 ? 65535 : fort_tile_needed_blocks);",
        f"kernel_tile_private_{region.id}<<<fort_tile_blocks, 128, 0, {stream}>>>("
        + ", ".join([*arguments, first, count]) + ");",
        "if (cudaGetLastError() != cudaSuccess) return FORT_SCOPE_EXECUTION;",
        f"return fort_scope_note_launch({context}) == FORT_SCOPE_OK ? FORT_SCOPE_OK : FORT_SCOPE_EXECUTION;",
    ]


def scratch_batch_helper():
    """Runtime lease adapter; submit must order batches on one context stream.

    A submit callback records accepted work by the ordinary runtime launch path.
    Any failure after invoking it is terminal EXEC, even if the callback returns
    RESOURCE: the caller cannot infer absence of an accepted device prefix. The
    owner retains its existing abandon/poison policy; scratch has no CPU mirror.
    """
    return r"""
template<class OrderedSubmit>
static int fort_tile_submit_batches(fort_scope_t context, std::size_t total,
    std::size_t per_batch, std::size_t stride, OrderedSubmit submit) {
    if (total == 0) return FORT_SCOPE_OK;
    if (per_batch == 0 || stride == 0 || per_batch > total ||
        per_batch > static_cast<std::size_t>(-1) / stride) return FORT_SCOPE_RESOURCE;
    fort_scope_scratch_lease_v1 lease{};
    int status = fort_scope_scratch_acquire_v1(context, per_batch * stride, &lease);
    if (status != FORT_SCOPE_OK) return status;
    std::size_t first = 0;
    while (first < total) {
        const std::size_t count = total - first < per_batch ? total - first : per_batch;
        status = submit(first, count, lease.device);
        if (status != FORT_SCOPE_OK) {
            // Even the first callback may have accepted work. Never replay.
            (void)fort_scope_scratch_release_v1(context, lease.token);
            return FORT_SCOPE_EXECUTION;
        }
        first += count;
    }
    status = fort_scope_scratch_release_v1(context, lease.token);
    return status == FORT_SCOPE_OK ? FORT_SCOPE_OK : FORT_SCOPE_EXECUTION;
}
""".strip().splitlines()


def _rectangular_point_phase(loops, body, phase_index, addressing):
    """Distribute independent points without forming a multidimensional size.

    Each extent is at most 2**32 from the original INTEGER loop bounds. Adding
    a block-width carry therefore fits 64-bit size_t even for huge point boxes.
    Uniform nonempty guards retain the original reachability of inner bounds.
    """
    from compiler.emission.common.loops import _loop_snapshot, iteration_value

    lines, depth, suffixes = [], 2, []
    for axis, loop in enumerate(loops):
        suffix = f"_tile_cta_{phase_index}_{axis}"
        suffixes.append(suffix)
        lines += indent(["{"], depth)
        depth += 1
        lines += indent(_loop_snapshot(loop, suffix, device=True), depth)
        lines += indent([f"if (fort_internal_extent{suffix}) {{"], depth)
        depth += 1
    lines += indent(["static_assert(sizeof(std::size_t)>=8, \"cooperative point coordinates require 64-bit size_t\");",
                     "std::size_t fort_point_carry=threadIdx.x;"], depth)
    for suffix in reversed(suffixes):
        lines += indent([f"std::size_t fort_internal_ordinal{suffix}=fort_point_carry%fort_internal_extent{suffix};",
                         f"fort_point_carry/=fort_internal_extent{suffix};"], depth)
    lines += indent(["while (fort_point_carry==0) {"], depth)
    depth += 1
    for loop, suffix in zip(loops, suffixes, strict=True):
        ordinal = "fort_internal_ordinal" + suffix
        lines += indent([f"{loop.iterator.cpp_name} = {iteration_value(suffix, ordinal)};"], depth)
        if loop.iterator in addressing.wide_iterators:
            lines += indent([f"const long long {wide_iterator_name(loop.iterator)} = {loop.iterator.cpp_name};"], depth)
    lines += sequential_block(body, depth, [phase_index * 1000], device=True, addressing=addressing)
    lines += indent(["fort_point_carry=blockDim.x;"], depth)
    for suffix in reversed(suffixes):
        lines += indent(["{", f"const std::size_t advanced=fort_internal_ordinal{suffix}+fort_point_carry;",
                         f"fort_internal_ordinal{suffix}=advanced%fort_internal_extent{suffix};",
                         f"fort_point_carry=advanced/fort_internal_extent{suffix};", "}"], depth)
    depth -= 1
    lines += indent(["}"], depth)
    for _ in loops:
        depth -= 1
        lines += indent(["}"], depth)
        depth -= 1
        lines += indent(["}"], depth)
    return lines


def generate_cooperative_kernel(region, proof, *, addressing=None):
    """One CTA per tile, preserving point bodies and both phase barriers.

    Multidimensional independent points use mixed-radix coordinates without a
    point-count product. The original one-dimensional mapping stays unchanged.
    """
    from compiler.emission.common.loops import _loop_snapshot, iteration_value
    from compiler.frontend.tile_private import CooperativeTilePrivateProof

    if type(proof) is not CooperativeTilePrivateProof:
        raise ValueError("cooperative kernel requires its separate original point-independence proof")
    proof.validate(region)
    if addressing is not None and addressing != proof.lowered.addressing:
        raise ValueError("cooperative addressing differs from original source lowering")
    producer, consumer, addressing = _scratch_blocks(region, proof.lowered.addressing)
    parameters = kernel_parameters(region) + ["std::size_t fort_tile_first", "std::size_t fort_tile_count"]
    lines = [f"static __global__ void kernel_tile_private_cta_{region.id}(" + ", ".join(parameters) + ") {",
             "    std::size_t fort_tile_slot = blockIdx.x;",
             "    while (fort_tile_slot < fort_tile_count) {",
             "        std::size_t fort_tile_remaining = fort_tile_first + fort_tile_slot;"]
    outer = {loop.iterator for loop in region.outer_loops}
    for axis in reversed(range(len(region.outer_loops))):
        loop = region.outer_loops[axis]
        lines += [f"        const std::size_t fort_tile_axis{axis} = fort_tile_remaining % fort_tile_extent{axis};",
                  f"        fort_tile_remaining /= fort_tile_extent{axis};",
                  f"        const int {loop.iterator.cpp_name} = static_cast<int>(static_cast<long long>(fort_tile_lower{axis})"
                  f" + static_cast<long long>(fort_tile_axis{axis}) * fort_tile_step{axis});"]
        if loop.iterator in addressing.wide_iterators:
            lines.append(f"        const long long {wide_iterator_name(loop.iterator)} = {loop.iterator.cpp_name};")
    for symbol in region.private_symbols:
        if symbol not in outer and symbol not in region.scratch_symbols:
            if symbol.rank:
                raise ValueError("unproved additional ranked private storage in cooperative tile IR")
            lines.append(f"        {cpp_type(symbol)} {symbol.cpp_name};")
    for index, symbol in enumerate(region.scratch_symbols):
        lines.append(f"        {cpp_type(symbol)}* {symbol.cpp_name} = reinterpret_cast<{cpp_type(symbol)}*>("
                     f"fort_tile_scratch + fort_tile_slot * fort_tile_stride + fort_tile_offset{index});")
    for phase_index, phase in enumerate((producer, consumer)):
        body = phase
        loops = []
        for _ in region.outer_loops:
            loop = body.statements[0]
            loops.append(loop)
            body = loop.body
        if len(loops) > 1:
            if proof.public()["version"] != 2:
                raise ValueError("rectangular point distribution requires its current source proof")
            lines += _rectangular_point_phase(loops, body, phase_index, addressing)
            lines += ["        __syncthreads(); // Every original point completes before phase/scratch reuse."]
            continue
        depth = 2
        for axis, loop in enumerate(loops):
            suffix = f"_tile_cta_{phase_index}_{axis}"
            ordinal = "fort_internal_ordinal" + suffix
            lines += indent(["{"], depth)
            depth += 1
            lines += indent(_loop_snapshot(loop, suffix, device=True), depth)
            first = "threadIdx.x" if axis == len(loops) - 1 else "0"
            increment = "blockDim.x" if axis == len(loops) - 1 else "1"
            lines += indent([f"for (std::size_t {ordinal} = {first}; {ordinal} < fort_internal_extent{suffix}; {ordinal}+={increment}) {{",
                             f"    {loop.iterator.cpp_name} = {iteration_value(suffix, ordinal)};"], depth)
            depth += 1
            if loop.iterator in addressing.wide_iterators:
                lines += indent([f"const long long {wide_iterator_name(loop.iterator)} = {loop.iterator.cpp_name};"], depth)
        lines += sequential_block(body, depth, [phase_index * 1000], device=True, addressing=addressing)
        for _ in loops:
            depth -= 1
            lines += indent(["}"], depth)
            depth -= 1
            lines += indent(["}"], depth)
        lines += ["        __syncthreads(); // All original points finish before the next phase or scratch reuse."]
    lines += ["        if (fort_tile_count - fort_tile_slot <= gridDim.x) break;",
              "        fort_tile_slot += gridDim.x;", "    }", "}", ""]
    return lines


def generate_cooperative_launch(region, proof, arguments, *, context, stream,
                                first="fort_tile_first", count="fort_tile_count"):
    proof.validate(region)
    if len(arguments) != len(kernel_parameters(region)):
        raise ValueError("cooperative tile kernel argument inventory differs")
    return [f"if ({count} == 0) return FORT_SCOPE_OK;",
            f"const unsigned fort_tile_blocks = static_cast<unsigned>({count} > 65535 ? 65535 : {count});",
            f"kernel_tile_private_cta_{region.id}<<<fort_tile_blocks, 128, 0, {stream}>>>("
            + ", ".join([*arguments, first, count]) + ");",
            "if (cudaGetLastError() != cudaSuccess) return FORT_SCOPE_EXECUTION;",
            f"return fort_scope_note_launch({context}) == FORT_SCOPE_OK ? FORT_SCOPE_OK : FORT_SCOPE_EXECUTION;"]
