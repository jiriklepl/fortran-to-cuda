"""Ordered scalar preparation and guarded kernels with shared section buffers."""

from dataclasses import replace
from hashlib import sha256

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.c.generator import cpp_plan_lines
from compiler.emission.common.abi import abi_arguments, dimension_name
from compiler.emission.common.c_family import cpp_type, indent, render_assignment, render_expression
from compiler.emission.common.symbols import host_symbols
from compiler.emission.cuda.kernels import generate_launch
from compiler.emission.cuda.offload import OffloadEmission, _cpu_worker, _metadata, _precision
from compiler.ir import (
    ArrayAccess,
    Binary,
    ConditionalRegion,
    ExecutionPlan,
    HostBlock,
    IntrinsicCall,
    Literal,
    Reference,
    ScalarType,
    Size,
    Unary,
    referenced_symbols,
)
from compiler.offload.analysis import OffloadAnalysis, _protected_scalar_inputs
from compiler.offload.codegen import profile_expression
from compiler.offload.preparation import prepare_offload, scalar_reads


def _query_expression(expression, scalar_indices):
    """Render source-width checked operations; invalid preparation stops reads."""

    def q(value):
        return _query_expression(value, scalar_indices)

    if isinstance(expression, Literal):
        result = render_expression(expression)
    elif isinstance(expression, Reference):
        name = expression.symbol.cpp_name
        if expression.symbol in scalar_indices:
            result = f"(coverage[{scalar_indices[expression.symbol]}] = true, {name})"
        else:
            result = name
    elif isinstance(expression, Size):
        result = f"offload::query_size({dimension_name(expression.symbol, expression.dimension)}, d.valid)"
    elif isinstance(expression, ArrayAccess):
        dims = ", ".join(dimension_name(expression.symbol, a + 1) for a in range(expression.symbol.rank))
        indices = ", ".join(q(value) for value in expression.indices)
        result = f"offload::query_load({expression.symbol.cpp_name}, {{{dims}}}, {{{indices}}}, d.valid)"
    elif isinstance(expression, Unary):
        operand = q(expression.operand)
        result = (
            f"!({operand})"
            if expression.operator == ".not."
            else f"offload::query_integer({expression.operator}static_cast<long long>({operand}), d.valid)"
        )
    elif isinstance(expression, Binary):
        # Locals force checked evaluation in source order, including both
        # operands of LOGICAL expressions (no reliance on short circuiting).
        left, right = q(expression.left), q(expression.right)
        operators = {
            ".and.": "&&",
            ".or.": "||",
            ".eqv.": "==",
            ".neqv.": "!=",
            "/=": "!=",
            ".eq.": "==",
            ".ne.": "!=",
            ".lt.": "<",
            ".le.": "<=",
            ".gt.": ">",
            ".ge.": ">=",
        }
        operator = operators.get(expression.operator, expression.operator)
        if operator == "/":
            result = "offload::query_divide(left, right, d.valid)"
        elif operator in {"+", "-", "*"}:
            result = f"offload::query_integer(static_cast<long long>(left) {operator} right, d.valid)"
        else:
            result = f"left {operator} right"
        return (
            "([&]() -> int { if (!d.valid) return 0; const int left = "
            + left
            + "; if (!d.valid) return 0; const int right = "
            + right
            + "; if (!d.valid) return 0; return "
            + result
            + "; }())"
        )
    elif isinstance(expression, IntrinsicCall):
        result = f"std::{expression.name.lower()}({{{', '.join(q(a) for a in expression.arguments)}}})"
    else:
        raise TypeError(expression)
    return f"([&]() -> int {{ if (!d.valid) return 0; return {result}; }}())"


def generate_structured(function, plan, config):
    prep = prepare_offload(function, plan)
    analysis = prep.analysis
    name = "fort_structured_" + sha256(f"{function.module}::{function.name}".encode()).hexdigest()[:12]
    query_name = f"{function.name[:40]}_offload_{name[-8:]}"
    abi = abi_arguments(function.parameters)
    signature = ", ".join(cpp_declaration(a) for a in abi)
    query_signature = ", ".join(
        cpp_declaration(a) if a.symbol.rank else f"const {cpp_type(a.symbol)} &{a.name}" for a in abi
    )
    arguments = ", ".join(a.name for a in abi)
    report = {
        "policy": config.policy,
        "analysis": analysis.to_dict(),
        "native_fallback_query": query_name,
        "collective_entry": config.collective,
        "host_threads": config.host_threads,
        "profile_available": config.profile is not None,
        "profile_reason": config.profile_reason,
        "preparation": {
            "available": analysis.available,
            "reason": analysis.reason,
            "kind": "checked_control_slice",
            "shared_interval_storage": analysis.available,
        },
        "supported_strategies": ["native"] + (["sections", "auto"] if analysis.available else []),
        "launches_max": len(analysis.units),
        "transfer_volume": "runtime union of active physical sections; unknown accesses use full arrays",
    }
    known = analysis.available and all(
        u.work_per_iteration is not None and not u.work_is_upper_bound for u in analysis.units
    )
    report["estimate_available"] = bool(known and config.profile is not None
                                        and not function.requires_numerical_environment)
    report["work_per_iteration"] = [u.work_per_iteration for u in analysis.units]
    if not known:
        report["estimate_reason"] = "work estimate is unknown or conditional"
    if function.requires_numerical_environment:
        report["estimate_reason"] = "numerical_environment_protocol_calibration_unavailable"
    fallback = [f"{cpp_type(s)} {s.cpp_name};" for s in host_symbols(function, plan)] + cpp_plan_lines(plan)
    serial_fallback = [line for line in fallback if not line.lstrip().startswith("#pragma omp")]
    if config.collective:
        fallback = ["#pragma omp single", "{", *indent(serial_fallback), "}"]
    else:
        fallback = [
            line.replace("#pragma omp parallel for", f"#pragma omp parallel for num_threads({config.host_threads})")
            for line in fallback
        ]
        fallback = ["if (offload::in_parallel()) {", *indent(serial_fallback), "} else {", *indent(fallback), "}"]
    if not analysis.available:
        return OffloadEmission(
            "",
            fallback,
            [f'offload::decision_trace("{function.name}", "native", 0, 0);', "return 0;"],
            query_name,
            frozenset(),
            report,
        )

    scalar_indices = {s: i for i, s in enumerate(s for s in function.parameters if not s.rank)}

    def q(expression):
        return _query_expression(expression, scalar_indices)

    locals_ = host_symbols(function, plan)
    lines = [f"static const offload::Profile {name}_profile = {profile_expression(config.profile)};"]
    lines += _metadata(function, OffloadAnalysis(True, None), query_signature, name + "_arrays")
    unit_args = {}
    for unit in analysis.units:
        # Each unit evaluates its footprints at its own scalar preparation point.
        used = set()
        for loop in unit.region.loops:
            for expression in (loop.lower, loop.upper) + (() if isinstance(loop.step, int) else (loop.step,)):
                used.update(referenced_symbols(expression))
        for footprint in unit.footprints:
            for box in (*footprint.uploads, *footprint.downloads):
                for expression in (*box.lower, *box.upper):
                    used.update(referenced_symbols(expression))
        captures = tuple(s for s in locals_ if s in used)
        extra_signature = "".join(f", const {cpp_type(s)} &{s.cpp_name}" for s in captures)
        unit_args[unit.region.id] = arguments + "".join(f", {s.cpp_name}" for s in captures) + ", coverage"
        unit_analysis = OffloadAnalysis(True, None, (replace(unit, index=0),))
        lines += _metadata(
            function,
            unit_analysis,
            query_signature + extra_signature + ", std::vector<bool>& coverage",
            name + f"_unit_{unit.region.id}",
            q,
        )

    def mark(expressions):
        symbols = set().union(*(scalar_reads(e) for e in expressions)) if expressions else set()
        return [f"coverage[{scalar_indices[s]}] = true;" for s in sorted(symbols, key=lambda s: s.id)]

    def project(current):
        result = []
        for step in current.steps:
            if isinstance(step, HostBlock):
                for assignment in step.assignments:
                    result += mark((assignment.value,))
                    if assignment in prep.assignments:
                        result += [f"if (d.valid) {assignment.target.symbol.cpp_name} = {q(assignment.value)};"]
            elif isinstance(step, ConditionalRegion):
                result += [
                    "if (d.valid) {",
                    f"    const bool branch = {q(step.condition)};",
                    "    if (d.valid && branch) {",
                    *indent(project(step.then_plan), 2),
                    "    } else if (d.valid) {",
                    *indent(project(step.else_plan), 2),
                    "    }",
                    "}",
                ]
            else:
                _, reads = _protected_scalar_inputs(step, frozenset(function.parameters))
                result += [
                    "if (d.valid) {",
                    f"    auto unit = {name}_unit_{step.id}({unit_args[step.id]});",
                    "    d.valid = unit.valid;",
                    "    if (d.valid && unit.units[0].iterations) {",
                    *indent([f"coverage[{scalar_indices[s]}] = true;" for s in sorted(reads, key=lambda s: s.id)], 2),
                    f"        unit.units[0].source_region = {step.id};",
                    "        d.units.push_back(std::move(unit.units[0]));",
                    "    }",
                    "}",
                ]
        return result

    lines += [
        f"static offload::Data {name}_data({query_signature}) {{",
        f"    auto d = {name}_arrays({arguments});",
        f"    std::vector<bool> coverage({len(scalar_indices)}, false);",
    ]
    lines += indent(
        [f"{cpp_type(s)} {s.cpp_name};" for s in locals_ if s.dtype in {ScalarType.INTEGER, ScalarType.LOGICAL}]
    )
    lines += indent(project(plan))
    for symbol in prep.live_scalars:
        lines += [f"    if (!coverage[{scalar_indices[symbol]}]) d.valid = false;"]
    lines += ["    d.guarded_inputs_checked = true;", "    return d;", "}"]

    worker_args = {}
    for unit in analysis.units:
        captures = host_symbols(function, ExecutionPlan((unit.region,)))
        extra_signature = "".join(f", {cpp_type(s)} {s.cpp_name}" for s in captures)
        worker_args[unit.region.id] = arguments + "".join(f", {s.cpp_name}" for s in captures)
        lines += _cpu_worker(unit, signature + extra_signature, name + f"_cpu_{unit.region.id}")

    arrays = [s for s in function.parameters if s.rank]
    lines += [
        f"struct {name}_state {{",
        "    offload::Data data;",
        "    std::size_t cursor = 0, choice = 0;",
        "    std::vector<offload::Footprint> footprints;",
        "    std::vector<std::unique_ptr<offload::Allocation>> buffers;",
    ]
    lines += indent([f"{cpp_type(s)} {s.cpp_name};" for s in locals_])
    lines += ["};"]
    lines += [
        f"static void {name}_open({name}_state& state, const offload::Choice& choice) {{",
        f"    state.footprints = offload::interval(state.data, choice.begin, choice.end, {name}_profile);",
        "    state.buffers.resize(state.data.arrays.size());",
        "    for (std::size_t a=0; a<state.buffers.size(); ++a) {",
        "        const auto& f = state.footprints[a];",
        "        if (!f.upload.empty() || !f.download.empty())",
        "            state.buffers[a] = std::make_unique<offload::Allocation>(state.data.arrays[a].bytes);",
        "        for (const auto& b : f.upload)",
        "            offload::copy_box(state.data.arrays[a], b, state.buffers[a]->get(), true);",
        "    }",
        "}",
    ]
    lines += [
        f"static void {name}_close({name}_state& state) {{",
        "    CUCH(cudaDeviceSynchronize());",
        "    for (std::size_t a=0; a<state.buffers.size(); ++a) {",
        "        for (const auto& b : state.footprints[a].download)",
        "            offload::copy_box(state.data.arrays[a], b, state.buffers[a]->get(), false);",
        "        if (state.buffers[a]) state.buffers[a]->release_completed();",
        "    }",
        "    state.buffers.clear();",
        "}",
    ]

    def execute(current):
        result = []
        for step in current.steps:
            if isinstance(step, HostBlock):
                result += [
                    "#pragma omp master",
                    "{",
                    *indent([render_assignment(a) for a in step.assignments]),
                    "}",
                    "#pragma omp barrier",
                ]
            elif isinstance(step, ConditionalRegion):
                result += [
                    f"if ({render_expression(step.condition)}) {{",
                    *indent(execute(step.then_plan)),
                    "} else {",
                    *indent(execute(step.else_plan)),
                    "}",
                ]
            else:
                result += [
                    f"if (shared->cursor < shared->data.units.size() && shared->data.units[shared->cursor].source_region == {step.id}) {{",
                    "    const auto choice = shared->data.plan.choices[shared->choice];",
                    "    if (choice.gpu) {",
                    "        #pragma omp master",
                    "        {",
                    "            offload::DeviceScope device_scope(shared->data.device);",
                    f"            if (shared->cursor == choice.begin) {name}_open(*shared, choice);",
                ]
                for i, symbol in enumerate(arrays):
                    result += [
                        f"            auto *{symbol.cpp_name}_device = shared->buffers[{i}] ? static_cast<{cpp_type(symbol)}*>(shared->buffers[{i}]->get()) : nullptr;"
                    ]
                result += indent(generate_launch(step), 3)
                result += [
                    f"            if (shared->cursor+1 == choice.end) {name}_close(*shared);",
                    "        }",
                    "        #pragma omp barrier",
                    "    } else {",
                    f"        {name}_cpu_{step.id}({worker_args[step.id]}, offload::thread_id(), offload::team_size());",
                    "        #pragma omp barrier",
                    "    }",
                    "    #pragma omp master",
                    "    {",
                    "        ++shared->cursor; if (shared->cursor == choice.end) ++shared->choice;",
                    "    }",
                    "    #pragma omp barrier",
                    "}",
                ]
        return result

    compatible = f"offload::compatible({name}_profile, {config.host_threads}, {_precision(function)})"
    ready = f"offload::context_valid({config.host_threads}, {'true' if config.collective else 'false'})"
    automatic = "true" if config.policy == "auto" else "false"
    if config.policy == "sections":
        compatible = "true"
    else:
        ready += f" && {name}_profile.valid" if known and not function.requires_numerical_environment else " && false"
    decision = [
        f"if (!({ready})) {{",
        f'    offload::decision_trace("{function.name}", "native", 0, 0); return 0;',
        "}",
        f"const auto d = {name}_data({arguments});",
        f"auto selected = offload::select(d, {name}_profile, {automatic});",
        f"if (selected.has_gpu() && !({compatible})) selected = offload::native_plan(d);",
        "if (!selected.has_gpu()) {",
        f'    offload::decision_trace("{function.name}", "native", 0, d.units.size());',
        f'    offload::plan_trace("{function.name}", d, {name}_profile, selected);',
        "}",
        "return selected.has_gpu() ? 1 : 0;",
    ]
    lines += [
        f"static void {name}_team({signature}, offload::Data* prepared = nullptr) {{",
        f"    {name}_state *shared = nullptr;",
        "    #pragma omp single copyprivate(shared)",
        "    {",
        f"        shared = new {name}_state;",
        f"        shared->data = prepared ? *prepared : {name}_data({arguments});",
        f"        shared->data.plan = offload::select(shared->data, {name}_profile, {automatic});",
        f"        if (offload::team_size() != {config.host_threads} || (shared->data.plan.has_gpu() && !({compatible})))",
        "            shared->data.plan = offload::native_plan(shared->data);",
        "        if (shared->data.plan.has_gpu() && shared->data.device < 0) CUCH(cudaGetDevice(&shared->data.device));",
        "    }",
    ]
    lines += indent([f"auto &{s.cpp_name} = shared->{s.cpp_name};" for s in locals_])
    lines += [
        "    if (offload::thread_id() == 0) {",
        "        std::size_t gpu=0,cpu=0;",
        "        for (const auto& c : shared->data.plan.choices) (c.gpu ? gpu : cpu) += c.end-c.begin;",
        f'        offload::decision_trace("{function.name}", gpu ? (cpu ? "mixed" : "gpu") : "native", gpu, cpu);',
        f'        offload::plan_trace("{function.name}", shared->data, {name}_profile, shared->data.plan);',
        "    }",
    ]
    lines += indent(execute(plan))
    lines += ["    #pragma omp single", "    { delete shared; }", "}"]
    body = ["#pragma omp barrier"] if config.collective else []
    body += [f"if (!({ready})) {{", *indent(fallback), "    return;", "}"]
    if config.collective:
        # A direct caller must honor the public query. Repeat preflight once
        # collectively so descriptors and protected values are never read early.
        body += [
            "offload::Data *check = nullptr;",
            "#pragma omp single copyprivate(check)",
            "{",
            f"    check = new offload::Data({name}_data({arguments}));",
            "}",
            "if (!check->valid) {",
            *indent(fallback),
            "} else {",
            f"    {name}_team({arguments}, check);",
            "}",
            "#pragma omp single",
            "{ delete check; }",
        ]
    else:
        body += [
            f"auto check = {name}_data({arguments});",
            "if (!check.valid) {",
            *indent(fallback),
            "    return;",
            "}",
            f"check.plan = offload::select(check, {name}_profile, {automatic});",
            f"if (check.plan.has_gpu() && ({compatible})) CUCH(cudaGetDevice(&check.device));",
            f"#pragma omp parallel num_threads({config.host_threads})",
            "{",
            f"    {name}_team({arguments}, &check);",
            "}",
        ]
    return OffloadEmission(
        "\n".join(lines), body, decision, query_name, prep.query_scalars, report, prep.unused_scalars
    )
