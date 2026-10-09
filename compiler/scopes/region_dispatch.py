"""One mode-bearing dispatcher for borrowed inline regions of an original owner."""

import copy
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
from types import SimpleNamespace

from fparser.two import Fortran2003 as F
from fparser.two.utils import Base, walk

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources
from compiler.emission.fortran.formatting import _fortran_list
from compiler.frontend import lower_source
from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError
from compiler.scopes.regions import extract_region


class InlineRegions:
    def __init__(self, builder):
        self.builder = builder
        self.regions, self.nodes, self.generated, self.ir, self.entries = {}, {}, {}, {}, {}
        self.used = set()
        self.bindings = {}
        # These identities authorize projections made here, never an arbitrary
        # AST carrying a plausible span or fort_inline_region attribute.
        self.source_selections = {}
        self.facade_proofs = {}
        self.name = "fort_regions_" + sha256(builder.entry.qualified.encode()).hexdigest()[:12]
        self.path = "regions/" + self.name + ".f90"

    def _register_projection(self, node, sources):
        from compiler.scopes.segments import statement_span
        self.source_selections[id(node)] = (node, tuple(sources), str(node), statement_span(node))

    def statement_projection(self, action, header):
        """Attach an original single-line IF span without changing its AST."""
        from compiler.scopes.segments import statement_span
        self.original_selection(header)
        if _kind(header) != "If_Stmt" or header.items[1] is not action:
            raise CompilationError("statement projection requires the original IF action")
        projected = copy.copy(action)
        span = statement_span(header)
        projected.item = SimpleNamespace(span=span, fort_original_span=span, label=None, name=None)
        self._register_projection(projected, (action,))
        return projected

    def grouped_nodes(self, nodes):
        """Group approved nodes and register any peeled directive projections."""
        from compiler.scopes.segments import grouped_nodes
        nodes = tuple(nodes)
        for node in nodes:
            self.original_selection(node)
        groups = grouped_nodes(nodes)
        for group in groups:
            for node in group if isinstance(group, tuple) else (group,):
                if id(node) in self.source_selections or any(node is original for original in nodes):
                    continue
                source = next((candidate for candidate in nodes
                               if _kind(candidate) in {"Block_Nonlabel_Do_Construct", "If_Construct", "Associate_Construct"}
                               and _kind(node) == _kind(candidate)
                               and node.content and len(candidate.content) >= len(node.content)
                               and all(left is right for left, right in
                                       zip(candidate.content[-len(node.content):], node.content, strict=True))), None)
                if source is None:
                    # Exposed prefix comments remain original descendants.
                    self.original_selection(node)
                else:
                    self._register_projection(node, self.original_selection(source))
        return groups

    def prepare(self, nodes):
        """Outline bounded numerical work, including reached structured branches."""
        from compiler.scopes.segments import grouped_nodes, statement_span
        original = tuple(nodes)
        authoritative = {id(node): node for node in walk(self.builder.entry.execution)}

        def register_projection(node, sources):
            self._register_projection(node, sources)

        def original_group(group, sequence):
            selected = []
            for node in group:
                if id(node) in authoritative:
                    selected.append(node)
                    continue
                # grouped_nodes peels the original directive prefix off a DO.
                # Recover only that compiler-created projection, by exact
                # child identity, while we hold its authoritative input list.
                source = next((candidate for candidate in sequence
                               if _kind(candidate) in {"Block_Nonlabel_Do_Construct", "If_Construct", "Associate_Construct"}
                               and _kind(node) == _kind(candidate)
                               and node.content and len(candidate.content) >= len(node.content)
                               and all(left is right for left, right in
                                       zip(candidate.content[-len(node.content):], node.content, strict=True))), None)
                if source is None:
                    raise CompilationError("inline grouping lost original source authority")
                register_projection(node, (source,))
                selected.append(source)
            # A peeled directive can also be contained in its selected DO.
            # Materialize the original construct once, including its prefix.
            descendants = {id(child) for node in selected for child in walk(node) if child is not node}
            unique = {id(node): node for node in selected}
            return tuple(node for key, node in unique.items() if key not in descendants)
        computations, operations, attempts = {}, 0, 0
        lifetime = {"Return_Stmt", "Exit_Stmt", "Cycle_Stmt", "Allocate_Stmt", "Deallocate_Stmt",
                    "Pointer_Assignment_Stmt", "Nullify_Stmt", "Stop_Stmt", "Error_Stop_Stmt"}

        def context(first, last):
            # Maximal original nodes outside this region retain allocation-site
            # and scalar-liveness authority even when the loop is in a branch.
            before, after = [], []

            def visit(node):
                try:
                    low, high = statement_span(node)
                except CompilationError:
                    return
                if high < first:
                    before.append(node)
                elif low > last:
                    after.append(node)
                elif _kind(node) == "If_Construct":
                    for child in node.content:
                        visit(child)
            for node in original:
                visit(node)
            return tuple(before), tuple(after)

        def prepare_sequence(sequence, depth=0):
            nonlocal operations, attempts
            result = []
            try:
                groups = grouped_nodes(sequence)
            except CompilationError:
                return tuple(sequence)
            for grouped in groups:
                group = tuple(grouped) if isinstance(grouped, tuple) else (grouped,)
                sources = original_group(group, sequence)
                node = group[0]
                if len(group) == 1 and _kind(node) in {"If_Construct", "Associate_Construct"}:
                    # Lifetimes/exits remain original boundaries. In particular
                    # do not borrow a region through a possibly reallocating
                    # branch and then pretend that its owner is unchanged.
                    association = self.builder.analysis._associate_scopes.get(id(node)) if _kind(node) == "Associate_Construct" else None
                    if (depth < 8 and (_kind(node) != "Associate_Construct" or association is not None and association.available)
                            and not any(_kind(item) in lifetime for item in walk(node))):
                        clone = copy.copy(node)
                        clone.content = list(prepare_sequence(node.content, depth + 1))
                        register_projection(clone, sources)
                        result.append(clone)
                    else:
                        result.append(node)
                    continue
                array_operation = len(group) == 1 and _kind(node) == "Assignment_Stmt"
                if array_operation:
                    from compiler.frontend.component_bindings import source_scope_for
                    original_target = node.items[0]
                    base = original_target.items[0] if _kind(original_target) == "Part_Ref" else original_target
                    target = self.builder.analysis._binding(source_scope_for(self.builder.analysis, original_target,
                                                                            self.builder.entry.scope), base)
                    array_operation = target is not None and bool(target.rank)
                if not array_operation and not any(_kind(item) == "Block_Nonlabel_Do_Construct" for item in group):
                    result.extend(group)
                    continue
                first, last = statement_span(group[0])[0], statement_span(group[-1])[1]
                before, after = context(first, last)
                try:
                    own_operations = sum(_kind(item) in {"Assignment_Stmt", "Block_Nonlabel_Do_Construct", "If_Stmt", "If_Construct"}
                                         for root in group for item in walk(root))
                    if attempts >= 32 or operations + own_operations > 256:
                        raise CompilationError("bounded inline region/operation budget exhausted")
                    attempts += 1
                    operations += own_operations
                    if array_operation:
                        from compiler.scopes.array_operations import extract_array_operation
                        extraction = extract_array_operation(self.builder.analysis, self.builder.entry, node,
                                                             preceding=before, following=after)
                    else:
                        extraction = extract_region(self.builder.analysis, self.builder.entry,
                                                    sources if len(sources) > 1 else sources[0],
                                                    preceding=before, following=after)
                    for binding in extraction.bindings:
                        if binding.rank:
                            self.builder.capture(binding)
                    source_name = str(self.builder.entry.scope.path) + "#inline:" + extraction.source_identity
                    function, plan = prepare_function(lower_source(extraction.source, extraction.entry, source_name=source_name),
                                                      options=self.builder.options)
                    if not plan.regions:
                        raise CompilationError("inline numerical candidate has no proven parallel region")
                    # The original path remains public provenance, while
                    # compiler-owned computational identities are stable across
                    # independent source directories after legality proofs.
                    function, plan = (self.artifact_ir(value, source_name, extraction.source_identity)
                                      for value in (function, plan))
                    generated = generate_sources(function, plan, offload_config=self.builder.config, memory_model="scoped")
                    if generated.scoped is None:
                        raise CompilationError("inline numerical candidate has no shared numerical entry")
                    procedure = self.builder.entry.qualified + "#region" + str(len(self.regions) + 1)
                    # Module identity/span differences do not require copies of an
                    # otherwise identical computation within one original owner.
                    computation = sha256("\n".join(extraction.source.splitlines()[1:-1]).encode()).hexdigest()
                    canonical = computations.setdefault(computation, procedure)
                    if canonical != procedure:
                        generated, function, plan = self.generated[canonical], *self.ir[canonical]
                    self.regions[procedure], self.generated[procedure] = extraction, generated
                    self.ir[procedure], self.entries[procedure] = (function, plan), canonical
                    self.bindings.update((binding.root, binding) for binding in extraction.bindings)
                    facade = F.Call_Stmt("call " + self.name + "()")
                    facade.item = SimpleNamespace(span=extraction.span, fort_original_span=extraction.span, label=None, name=None)
                    facade.fort_inline_region = procedure
                    self.nodes[procedure] = facade
                    register_projection(facade, sources)
                    self.facade_proofs[procedure] = extraction
                    result.append(facade)
                except CompilationError as error:
                    self.builder.boundaries.append({"first_line": first, "last_line": last,
                                                    "reason": "inline numerical boundary: " + str(error)})
                    result.extend(group)
            return tuple(result)

        return prepare_sequence(original)

    def original_selection(self, node):
        """Recover authoritative original nodes from approved preparations.

        Source summaries use these nodes rather than interpreting synthetic
        calls, their arguments or normalized numerical INTENT declarations.
        """
        analysis, routine = self.builder.analysis, self.builder.entry
        analysis.inputs.verify()
        role = analysis._source_roles.get(routine.qualified)
        if (analysis.routines.get(routine.qualified) is not routine or role is None
                or routine.execution is not role[0] or analysis._routine_signature(routine) != role[1]
                or str(routine.execution) != role[2]):
            raise CompilationError("inline selection requires unchanged original source authority")
        if isinstance(node, (tuple, list)):
            return tuple(source for child in node for source in self.original_selection(child))
        registered = self.source_selections.get(id(node))
        if registered is not None:
            from compiler.scopes.segments import statement_span
            original_node, sources, text, span = registered
            if node is not original_node or str(node) != text or statement_span(node) != span:
                raise CompilationError("inline source projection was changed after registration")
            if hasattr(node, "fort_inline_region"):
                procedure = node.fort_inline_region
                if (self.nodes.get(procedure) is not node
                        or self.regions.get(procedure) is not self.facade_proofs.get(procedure)):
                    raise CompilationError("inline numerical facade lacks its registered proof")
            return sources
        if any(node is original for original in walk(routine.execution)):
            return (node,)
        raise CompilationError("source selection is not an original node or approved inline projection")

    @staticmethod
    def artifact_ir(value, source_name, identity):
        """Canonicalize private proof locations, retaining original provenance."""
        if isinstance(value, (tuple, list)):
            return type(value)(InlineRegions.artifact_ir(child, source_name, identity) for child in value)
        if isinstance(value, str):
            return value.replace(source_name, "inline:" + identity)
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{field.name: InlineRegions.artifact_ir(getattr(value, field.name), source_name, identity)
                                     for field in fields(value) if field.name != "source"})
        return value

    def call(self, node):
        from compiler.scopes.source import Call
        self.original_selection(node)
        procedure = node.fort_inline_region
        region = self.regions[procedure]
        bindings = {"argument::" + parameter.name: self.bindings[parameter.resource]
                    for parameter in region.parameters if parameter.lower_bound_dimension is None}
        return Call(node, procedure, tuple(F.Name(binding.name) for binding in bindings.values()), bindings,
                    {"complete": True, "summary_identity": region.source_identity}, region=region)

    def restore(self, node):
        """Expand private CALL facades before emitting any native fallback."""
        def visit(value):
            if isinstance(value, (tuple, list)):
                result = []
                for child in value:
                    if hasattr(child, "fort_inline_region"):
                        self.original_selection(child)
                        result.extend(self.regions[child.fort_inline_region].nodes)
                    else:
                        result.append(visit(child))
                return type(value)(result)
            if not isinstance(value, Base):
                return value
            if hasattr(value, "fort_inline_region"):
                self.original_selection(value)
                return self.regions[value.fort_inline_region].nodes
            # fparser comments do not implement Base's pickle/copy protocol
            # (they have no ``string``). Keep their source and directive
            # metadata intact while restoring neighbouring private facades.
            if _kind(value) == "Comment":
                result = object.__new__(type(value))
                result.__dict__ = dict(value.__dict__)
            else:
                result = copy.copy(value)
            for attribute in ("content", "items"):
                if hasattr(value, attribute):
                    setattr(result, attribute, visit(getattr(value, attribute)))
            return result
        restored = visit(node)
        return restored if isinstance(restored, tuple) else (restored,)

    def original(self, call, parameters):
        from compiler.scopes.segments import renamed
        return [line for node in call.region.nodes for line in renamed(self.builder, node, parameters).splitlines()]

    def effects(self, call, *, mapped=False):
        result = {}
        for parameter in call.region.parameters:
            if not parameter.rank:
                continue
            root = parameter.resource if mapped else "argument::" + parameter.name
            result[root] = {"read", "write"} if parameter.resource in call.region.written_resources else {"read"}
        return result, set(), set()

    def inputs(self, procedure, mapping, *, estimates):
        public = self.generated[procedure].scoped
        planning = public["planning"]
        if estimates and (not planning["available"] or not planning["profile_available"]):
            raise CompilationError(planning["reason"] or planning["profile_reason"])
        if not estimates and not planning["query_available"]:
            raise CompilationError(planning["query_reason"])
        parameters = {parameter.name: parameter for parameter in self.regions[procedure].parameters}
        payload = {parameters[name.lower()].resource for name in planning["payload_arrays"]}
        scalars = {parameters[name.lower()].resource for name in planning["scalar_inputs"]
                   if parameters[name.lower()].lower_bound_dimension is None}
        return {mapping.get(root, root) for root in payload}, {mapping.get(root, root) for root in scalars}

    @property
    def arrays(self):
        return {root: binding for root, binding in sorted(self.bindings.items()) if binding.rank}

    @property
    def scalars(self):
        return {root: binding for root, binding in sorted(self.bindings.items()) if not binding.rank}

    def parameters(self):
        arrays = {root: "fort_buffer_" + str(index) for index, root in enumerate(self.arrays)}
        scalars = {root: "fort_scalar_" + str(index) for index, root in enumerate(self.scalars)}
        lowers = {(root, axis): "fort_lower_" + str(index) + "_" + str(axis)
                  for index, (root, binding) in enumerate(self.arrays.items()) for axis in range(1, binding.rank + 1)}
        return arrays, scalars, lowers

    def emit_call(self, call, handles, values, imports, *, query, mode="fort_mode"):
        from compiler.scopes.source import DTYPES
        procedure, region = call.procedure, call.region
        self.builder.entry_artifacts(procedure)
        self.used.add(procedure)
        self.builder.variants.register(
            self.builder.entry.qualified, interface="source_inline_dispatch_v1", role="region_dispatcher",
            name=self.name + "::run", summary_identity=self.builder.analysis.summarize(self.builder.entry.qualified)["summary_identity"],
            requirements=("bounded original region IDs", "original saved storage and guards retained", "mode-bearing CPU/GPU execution"),
            shared_artifacts=(self.path,))
        alias = "fort_inline_query" if query else "fort_inline_run"
        imports.append(f"use {self.name}, only: {alias} => {'query' if query else 'run'}")
        active = {binding.root for binding in region.bindings}
        arguments = ["fort_context", *([] if query else [mode]), str(list(self.regions).index(procedure) + 1) + "_c_int"]
        arguments += [handles[root] if root in active else "0_c_int64_t" for root in self.arrays]
        zero = {"real(c_double)": "0.0_c_double", "real(c_float)": "0.0_c_float", "integer(c_int)": "0_c_int",
                "logical(c_bool)": ".false._c_bool"}
        arguments += [(values[root] if root in values else self.builder.visible(self.builder.entry, root))
                      if root in active else zero[DTYPES[binding.signature()[:2]][0]]
                      for root, binding in self.scalars.items()]
        arguments += [f"int(lbound({values[root]},{axis},kind=c_int64_t),c_int)" if root in active else "0_c_int"
                      for root, binding in self.arrays.items() for axis in range(1, binding.rank + 1)]
        lines = _fortran_list("fort_status = " + alias + "(", arguments, ")", 0)
        if not query:
            lines += ["if (fort_status /= FORT_SCOPE_OK) error stop 'inline numerical region failed'"]
        return lines

    def finish(self):
        if not self.used:
            return
        from compiler.scopes.segments import fortran_lines
        from compiler.scopes.source import DTYPES
        arrays, scalars, lowers = self.parameters()
        imports, cases = [], {True: [], False: []}
        for procedure in self.regions:
            if procedure not in self.used:
                continue
            public, _ = self.builder.entry_artifacts(procedure)
            region = self.regions[procedure]
            region_id = list(self.regions).index(procedure) + 1
            mapping = {parameter.name: parameter for parameter in region.parameters}
            for query in (True, False):
                alias = "fort_entry_" + ("query_" if query else "run_") + str(region_id)
                target = public["planning"]["fortran_procedure"] if query else public["fortran_procedure"]
                imports.append(f"use {public['fortran_module']}, only: {alias} => {target}")
                arguments = ["context", *([] if query else ["mode"])]
                arguments += [arrays[mapping[item["name"].lower()].resource] for item in public["array_parameters"]]
                for item in public["scalar_parameters"]:
                    if query and item["name"] not in public["planning"]["scalar_inputs"]:
                        continue
                    parameter = mapping[item["name"].lower()]
                    arguments.append(lowers[parameter.resource, parameter.lower_bound_dimension]
                                     if parameter.lower_bound_dimension else scalars[parameter.resource])
                cases[query] += [f"case ({region_id})", *_fortran_list("status = " + alias + "(", arguments, ")", 0)]
            self.builder.outputs["regions/" + region.source_identity[:16] + ".source"] = region.source
        lines = ["module " + self.name, "use iso_c_binding", "use fort_scoped_memory", *dict.fromkeys(imports),
                 "implicit none", "private", "public :: run, query", "contains"]
        for query in (True, False):
            name = "query" if query else "run"
            parameters = ["context", *([] if query else ["mode"]), "region", *arrays.values(), *scalars.values(), *lowers.values()]
            lines += _fortran_list("function " + name + "(", parameters, ") result(status)", 0)
            lines += ["integer(c_int64_t), intent(in) :: context", "integer(c_int), intent(in) :: region",
                      *([] if query else ["integer(c_int), intent(in) :: mode"]), "integer(c_int) :: status"]
            lines += [f"integer(c_int64_t), intent(in) :: {name}" for name in arrays.values()]
            lines += [f"{DTYPES[self.scalars[root].signature()[:2]][0]}, intent(in) :: {name}" for root, name in scalars.items()]
            lines += [f"integer(c_int), intent(in) :: {name}" for name in lowers.values()]
            lines += ["select case (region)", *cases[query], "case default", "status = FORT_SCOPE_BOUNDARY",
                      "end select", "end function " + name]
        lines += ["end module " + self.name, ""]
        self.builder.outputs[self.path] = "\n".join(fortran_lines(lines))

    def public(self):
        return {"dispatcher": self.name + "::run" if self.used else None, "variant_interface": "source_inline_dispatch_v1",
                "limits": {"regions": 32, "operations": 256},
                "regions": [{**region.public(), "region_id": index + 1, "used": procedure in self.used,
                             "shared_computation": self.entries[procedure]} for index, (procedure, region) in enumerate(self.regions.items())],
                "boundaries": ["unknown effects and allocation changes",
                               "saved or live scalar outputs", "unsupported numerical computation"]}
