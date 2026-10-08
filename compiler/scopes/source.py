"""Compiler-owned, bounded source scopes through public build/source artifacts.

The capture document supplies storage lifetime/initialization facts. It does not
choose GPU calls, scopes, transfers, or numerical lowering. Those decisions live
here. Initial source scopes contain straight-line helper paths and serial callers;
control-flow around sibling spans remains in the original source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from fparser.two.utils import walk

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.fortran.formatting import _fortran_line, _fortran_list
from compiler.frontend import lower_file
from compiler.frontend.source_effects import SourceEffects, _children, _kind, _part
from compiler.ir import CompilationError

DTYPES = {
    ("real", 8): ("real(c_double)", "FORT_SCOPE_REAL64", 8),
    ("real", 4): ("real(c_float)", "FORT_SCOPE_REAL32", 4),
    ("integer", 4): ("integer(c_int)", "FORT_SCOPE_INTEGER32", 4),
    ("logical", 1): ("logical(c_bool)", "FORT_SCOPE_LOGICAL", 1),
}


def _name(prefix, identity):
    return prefix + sha256(identity.encode()).hexdigest()[:12]


def _call(name, arguments):
    return _fortran_list("call " + name + "(", list(arguments), ")", 0)


def _checked(expression):
    return ["fort_status = " + expression,
            "if (fort_status /= FORT_SCOPE_OK) error stop 'shared scope operation failed'"]


def _span(node):
    item = getattr(node, "item", None)
    if item is None:
        raise CompilationError("source scope requires exact statement spans")
    return item.span


@dataclass
class Call:
    node: object
    procedure: str
    actuals: tuple
    bindings: dict
    summary: dict


class ScopeBuilder:
    def __init__(self, paths, entry, *, facts, options, config, contracts=None):
        if not isinstance(facts, dict) or facts.get("schema_version") != 1:
            raise CompilationError("scope capture facts require schema_version 1")
        if facts.get("participation") != "serial":
            raise CompilationError("source scopes currently require proved serial caller participation")
        if not isinstance(facts.get("captures"), dict):
            raise CompilationError("scope captures must be keyed by canonical source resource")
        if config.policy not in {"sections", "auto"} or config.collective:
            raise CompilationError("source scopes require sections/auto and a serial coordinator")
        self.analysis = SourceEffects(paths, contracts=contracts)
        if facts.get("sources") != self.analysis.sources:
            raise CompilationError("scope capture facts do not match the supplied source hashes")
        names = [name for name in self.analysis.routines
                 if name == entry.lower() or ("::" not in entry and name.split("::")[-1] == entry.lower())]
        if len(names) != 1:
            raise CompilationError("source scope entry is unavailable or ambiguous")
        self.entry = self.analysis.routines[names[0]]
        self.facts, self.options, self.config = facts, options, config
        self.device_budget = facts.get("device_budget_bytes", 256*1024*1024)
        if type(self.device_budget) is not int or not 0 <= self.device_budget < 2**63:
            raise CompilationError("scope device budget must be a nonnegative signed-64-bit byte count")
        self.outputs, self.edits, self.clones, self.generated = {}, {}, {}, {}
        self.boundaries, self.scopes = [], []
        self.runtime_outputs, self.runtime = read_scoped_runtime()
        self.visiting = set()

    def resolve(self, routine, node):
        target, args = node.items
        if _kind(target) != "Name":
            raise CompilationError("indirect calls are scope boundaries")
        actuals = tuple(_children(args))
        if any(_kind(a) == "Actual_Arg_Spec" for a in actuals):
            raise CompilationError("keyword call mappings are not yet supported in source scopes")
        candidates = self.analysis._candidates(routine.scope, target)
        matches = []
        for candidate in candidates:
            callee = self.analysis.routines.get(candidate)
            if callee is None:
                # Opaque contracts are native operations; positional whole
                # actuals still need a known resource identity.
                if candidate in self.analysis.contracts:
                    matches.append(candidate)
                continue
            formals = [callee.scope.bindings.get(a) for a in callee.arguments]
            supplied = [self.analysis._signature(routine.scope, a) for a in actuals]
            if len(formals) == len(actuals) and all(
                f is not None and s is not None and None not in (f.kind, s[1]) and f.signature() == s
                for f, s in zip(formals, supplied, strict=True)
            ):
                matches.append(candidate)
        if len(matches) != 1:
            raise CompilationError(f"source call is unresolved or ambiguous: {target}")
        procedure = matches[0]
        bindings = {}
        callee = self.analysis.routines.get(procedure)
        if callee is None:
            raise CompilationError("opaque source-scope hooks require a source binding interface")
        for formal, actual in zip(callee.arguments, actuals, strict=True):
            binding = self.analysis._actual_binding(routine.scope, actual)
            if binding:
                bindings["argument::" + formal] = binding
        summary = self.analysis.summarize(procedure)
        if not summary["complete"]:
            raise CompilationError("native effects incomplete: " + "; ".join(summary["reasons"]))
        effects, definitions, _ = self.native_effects(procedure)
        aliases = {}
        for formal, binding in bindings.items():
            if binding.rank:
                aliases.setdefault(binding.root, []).append(formal)
        if any(len(formals) > 1 and any("write" in effects.get(formal, set()) or
                                        formal in definitions for formal in formals)
               for formals in aliases.values()):
            raise CompilationError("writable source call arguments alias")
        return Call(node, procedure, actuals, bindings, summary)

    def numerical(self, procedure):
        if procedure in self.generated:
            return self.generated[procedure]
        routine = self.analysis.routines[procedure]
        summary = self.analysis.summarize(procedure)
        result = None
        # Keep procedure-entry definition events at their original positions.
        # Numerical helper inlining currently discards those source events;
        # wrappers therefore get explicit context/handle clones instead.
        leaf = not any(_kind(node) == "Call_Stmt" for node in walk(routine.execution))
        if summary["cloneable"] and leaf:
            try:
                if any(b.dtype == "logical" and b.kind != 1 for b in routine.scope.bindings.values()
                       if b.name in routine.arguments):
                    raise CompilationError("source LOGICAL conversion requires guarded ABI handling")
                function = lower_file(routine.scope.path, procedure)
                function, plan = prepare_function(function, options=self.options)
                if plan.regions:
                    sources = generate_sources(function, plan, offload_config=self.config, memory_model="scoped")
                    if sources.scoped:
                        result = sources
            except CompilationError:
                pass  # Supported native effects do not imply GPU eligibility.
        self.generated[procedure] = result
        return result

    def closure(self, procedure, active=()):
        """Return GPU leaves and cloneable call-only wrappers, or a boundary."""
        if procedure in active or len(active) >= 8:
            raise CompilationError("recursive or over-depth source scope")
        routine = self.analysis.routines[procedure]
        summary = self.analysis.summarize(procedure)
        if not summary["complete"]:
            raise CompilationError("incomplete source effect closure")
        if self.numerical(procedure):
            return {procedure}, {procedure}
        children = [n for n in _children(routine.execution) if _kind(n) != "Comment"]
        if not children or any(_kind(n) != "Call_Stmt" for n in children):
            return set(), set()
        if not summary["cloneable"]:
            return set(), set()
        leaves, wrappers = set(), {procedure}
        for node in children:
            call = self.resolve(routine, node)
            own_leaves, own_wrappers = self.closure(call.procedure, active + (procedure,))
            leaves.update(own_leaves)
            wrappers.update(own_wrappers)
        return leaves, wrappers if leaves else set()

    def native_effects(self, procedure, mapping=None, active=()):
        """Flatten bounded effects with formal-to-root identities, never IR."""
        if procedure in active:
            raise CompilationError("recursive native effects")
        summary = self.analysis.summarize(procedure)
        mapping = {} if mapping is None else mapping
        effects, definitions, overwrites = {}, set(), set()

        def mapped(root):
            return mapping.get(root, root)

        definitions.update(mapped(root) for root in summary["definition_changes"])
        overwrites.update(mapped(root) for root in summary["guaranteed_whole_overwrites"])
        for operation in summary["operations"]:
            kind = operation["kind"]
            if kind in {"read", "write", "overwrite"} and operation["rank"]:
                root = mapped(operation["resource"])
                effects.setdefault(root, set()).add("read" if kind == "read" else "write")
            elif kind == "call":
                child_mapping = {formal: mapped(actual) for formal, actual in operation["resource_mapping"].items()}
                child_effects, child_definitions, child_overwrites = self.native_effects(
                    operation["procedure"], child_mapping, active + (procedure,))
                for root, actions in child_effects.items():
                    effects.setdefault(root, set()).update(actions)
                definitions.update(child_definitions)
                # Only unconditional calls supply entry-wide must overwrites.
                if not operation["guard"]:
                    overwrites.update(child_overwrites)
            elif kind == "native_contract":
                for effect in operation["effects"]:
                    root = mapped(effect["resource"])
                    effects.setdefault(root, set()).add("read" if effect["kind"] == "read" else "write")
                    if effect["kind"] == "overwrite" and not operation["guard"]:
                        overwrites.add(root)
        return effects, definitions, overwrites

    def roots_for(self, call):
        mapping = {formal: binding.root for formal, binding in call.bindings.items()}
        return self.native_effects(call.procedure, mapping)

    def visible(self, routine, root):
        candidates = set(routine.scope.bindings)
        current = routine.scope
        while current:
            candidates.update(current.bindings)
            candidates.update(current.imports)
            for module in current.wildcards:
                target = self.analysis.modules.get(module)
                if target:
                    candidates.update(target.bindings)
            current = current.parent
        found = [name for name in sorted(candidates)
                 if (binding := self.analysis._binding(routine.scope, name)) and binding.root == root]
        if not found:
            raise CompilationError("hidden resource is unavailable at owning scope: " + root)
        return found[0]

    def capture(self, binding):
        if (binding.dtype, binding.kind) not in DTYPES:
            raise CompilationError("unsupported scoped capture type: " + binding.name)
        if {"pointer", "optional", "volatile", "asynchronous", "value"} & binding.attributes:
            raise CompilationError("capture association or participation is uncertain: " + binding.name)
        fact = self.facts["captures"].get(binding.root)
        if (not isinstance(fact, dict) or fact.get("storage") != "stable"
                or fact.get("escapes") is not False or fact.get("allocation_changes") is not False
                or not isinstance(fact.get("initialized"),str)
                or fact.get("initialized") not in {"whole", "none", "sections"}):
            raise CompilationError("missing stable storage/definition facts: " + binding.root)
        if fact["initialized"] == "sections":
            boxes = fact.get("sections")
            if not isinstance(boxes,list) or len(boxes) > 32:
                raise CompilationError("initialized capture sections exceed the bounded interface")
            for box in boxes:
                if not isinstance(box,dict) or set(box) != {"lower","upper"}:
                    raise CompilationError("initialized capture section requires lower/upper physical coordinates")
                lower,upper=box["lower"],box["upper"]
                if (not isinstance(lower,list) or not isinstance(upper,list)
                        or len(lower) != binding.rank or len(upper) != binding.rank
                        or any(type(v) is not int or not 0 <= v < 2**63 for v in [*lower,*upper])
                        or any(lo > hi for lo,hi in zip(lower,upper,strict=True))):
                    raise CompilationError("invalid initialized physical capture section")
        if "allocatable" in binding.attributes:
            raise CompilationError("allocatable source captures require lifetime proof integration")
        return fact

    def add_edit(self, path, first, last, replacement):
        self.edits.setdefault(path, []).append((first, last, replacement))

    def entry_artifacts(self, procedure):
        sources = self.numerical(procedure)
        if sources is None:
            raise CompilationError("requested source leaf has no shared numerical entry")
        directory = "entries/" + _name("entry_", procedure)
        if directory + "/shared_entry.cu" not in self.outputs:
            for name in ("shared_entry.cu", "shared_interface.f90"):
                self.outputs[directory + "/" + name] = sources.artifacts[name]
            self.outputs[directory + "/common_functions.cuh"] = read_common_header()
            self.outputs[directory + "/scoped_entry.hpp"] = self.runtime_outputs["scoped_entry.hpp"]
            self.outputs[directory + "/scoped_runtime.h"] = self.runtime_outputs["scoped_runtime.h"]
        return sources.scoped, directory

    def clone(self, procedure):
        if procedure in self.clones:
            return self.clones[procedure]
        routine = self.analysis.routines[procedure]
        if any(name.startswith("fort_") for name in routine.scope.bindings):
            raise CompilationError("source names conflict with the initial scope helper namespace")
        summary = self.analysis.summarize(procedure)
        if not summary["cloneable"]:
            raise CompilationError("scope helper cannot be cloned: " + procedure)
        _, wrappers = self.closure(procedure)
        numeric = self.numerical(procedure)
        array_names = [a for a in routine.arguments if routine.scope.bindings[a].rank]
        # Hidden arrays need explicit extra mappings. Initial clones reject them;
        # native calls may still access a visible root behind their outer hooks.
        effects, _, _ = self.native_effects(procedure)
        if any(root not in {"argument::" + a for a in array_names} for root in effects):
            raise CompilationError("cloned helper needs hidden array mappings")
        name = _name("fort_scope_clone_", procedure)
        handles = {a: "fort_handle_" + str(i) for i, a in enumerate(array_names)}
        self.clones[procedure] = (name, array_names)
        header = _fortran_list("subroutine " + name + "(",
                               ["fort_context", "fort_mode", *routine.arguments, *handles.values()], ")", 0)
        uses = ["use iso_c_binding", "use fort_scoped_memory"]
        spec = [str(n) for n in _children(_part(routine.scope.node, "Specification_Part"))]
        body = []
        if numeric:
            public, _ = self.entry_artifacts(procedure)
            uses += [f"use {public['fortran_module']}, only: fort_run => {public['fortran_procedure']}"]
            arrays = [handles[p["name"].lower()] for p in public["array_parameters"]]
            scalars = [p["name"] for p in public["scalar_parameters"]]
            body += _fortran_list("fort_status = fort_run(",
                                   ["fort_context", "fort_mode", *arrays, *scalars], ")", 0)
            body += ["if (fort_status /= FORT_SCOPE_OK) error stop 'shared numerical entry failed'"]
        else:
            for array in array_names:
                if routine.scope.bindings[array].intent == "out":
                    body += _checked(f"fort_scope_forget_definition(fort_context, {handles[array]})")
            for node in _children(routine.execution):
                if _kind(node) == "Comment":
                    body.append(str(node))
                    continue
                call = self.resolve(routine, node)
                leaves, child_wrappers = self.closure(call.procedure)
                root_handles = {formal: handles[b.name] for formal, b in call.bindings.items() if b.rank}
                if leaves:
                    child_name, child_arrays = self.clone(call.procedure)
                    module = self.analysis.routines[call.procedure].scope.module
                    if module != routine.scope.module:
                        uses.append(f"use {module}, only: {child_name}")
                    body += _call(child_name, ["fort_context", "fort_mode",
                                              *map(str, call.actuals),
                                              *[root_handles["argument::" + a] for a in child_arrays]])
                else:
                    actions, definitions, overwrites = self.native_effects(call.procedure)
                    body += self.native_call(call, actions, definitions, overwrites, root_handles)
        # USE statements precede the original specification, including IMPLICIT.
        extra = ["integer(c_int64_t), intent(in) :: fort_context",
                 "integer(c_int), intent(in) :: fort_mode",
                 "integer(c_int64_t), intent(in) :: " + ", ".join(handles.values()),
                 "integer(c_int) :: fort_status", "type(fort_scope_access) :: fort_access"]
        text = "\n".join([*header, *dict.fromkeys(uses), *spec, *extra, *body,
                          "end subroutine " + name, ""])
        self.append_procedure(routine.scope.parent, name, text)
        return name, array_names

    def native_call(self, call, actions, definitions, overwrites, handles):
        lines = []
        # Only outer dummy definition changes can be performed before the call.
        # A nested INTENT(OUT) event must stay at its original source position.
        outer = set(self.analysis.summarize(call.procedure)["definition_changes"])
        if (definitions - outer) & handles.keys():
            raise CompilationError("nested native definition changes require original-position hooks")
        for root in sorted(definitions):
            if root in handles:
                lines += _checked(f"fort_scope_forget_definition(fort_context, {handles[root]})")
        # Multiple read-only formals may identify one canonical root. Prepare
        # that handle once; the runtime forbids overlapping prepared accesses.
        accesses = {}
        for root, kinds in sorted(actions.items()):
            if root not in handles:
                raise CompilationError("native operation needs unavailable buffer mapping: " + root)
            handle = handles[root]
            merged, overwrite = accesses.get(handle, (set(), False))
            accesses[handle] = (merged | kinds, overwrite or root in overwrites)
        for handle, (kinds, overwrite) in accesses.items():
            flags = []
            if "read" in kinds:
                flags.append("FORT_SCOPE_READ_ALL")
            if "write" in kinds:
                flags.append("FORT_SCOPE_WRITE_ALL")
            if overwrite:
                flags.append("FORT_SCOPE_OVERWRITE_ALL")
            lines += ["fort_access = fort_scope_access()",
                      "fort_access%flags = " + " + ".join(flags)]
            lines += _checked(f"fort_scope_host_begin(fort_context, {handle}, fort_access)")
        lines += _call(str(call.node.items[0]), map(str,call.actuals))
        for handle in accesses:
            lines += _checked(f"fort_scope_host_end(fort_context, {handle})")
        return lines

    def append_procedure(self, module, name, code):
        end = _part(module.node, "End_Module_Stmt")
        contains = next(n for n in _children(_part(module.node, "Module_Subprogram_Part"))
                        if _kind(n) == "Contains_Stmt")
        self.add_edit(module.path, _span(end)[0], _span(end)[0]-1, code)
        self.add_edit(module.path, _span(contains)[0], _span(contains)[0]-1, "public :: " + name + "\n")

    def owner(self, calls, leaves):
        routine = self.entry
        if any(name.startswith("fort_") for name in routine.scope.bindings):
            raise CompilationError("source names conflict with the initial scope owner namespace")
        digest = f"{routine.qualified}:{_span(calls[0].node)[0]}:{_span(calls[-1].node)[1]}"
        name = _name("fort_scope_owner_", digest)
        arrays, scalars, written = {}, {}, set()
        for call in calls:
            actions, definitions, overwrites = self.roots_for(call)
            written.update(root for root,kinds in actions.items() if "write" in kinds)
            _, formal_definitions, _ = self.native_effects(call.procedure)
            for formal,binding in call.bindings.items():
                if binding.rank:
                    self.capture(binding)
                    arrays[binding.root] = binding
                    if formal in formal_definitions:
                        written.add(binding.root)
            for root in actions:
                visible = self.visible(routine, root)
                binding = self.analysis._binding(routine.scope, visible)
                self.capture(binding)
                arrays[root] = binding
            for actual in call.actuals:
                binding = self.analysis._actual_binding(routine.scope, actual)
                if binding and not binding.rank:
                    if binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}:
                        raise CompilationError("unsupported scalar source capture")
                    scalars[binding.root] = binding
        names = [self.visible(routine, root) for root in (*arrays, *scalars)]
        if any(n.startswith("fort_") for n in names):
            raise CompilationError("capture names conflict with the initial scope owner namespace")
        header = _fortran_list("subroutine " + name + "(", names, ")", 0)
        imports = [str(n) for n in _children(_part(routine.scope.node, "Specification_Part"))
                   if _kind(n) == "Use_Stmt"]
        spec = []
        handles, layouts, flags = {}, {}, {}
        for i, (root, binding) in enumerate(arrays.items()):
            visible = self.visible(routine, root)
            dtype, enum, width = DTYPES[binding.signature()[:2]]
            handles[root] = "fort_handle_" + str(i)
            layouts[root] = "fort_layout_" + str(i)
            flags[root] = int(self.capture(binding)["initialized"] == "whole")
            if binding.intent == "in" and root in written:
                raise CompilationError("scope writes an INTENT(IN) capture")
            intent = "inout" if root in written else "in"
            spec += [*_fortran_line(f"{dtype}, target, intent({intent}) :: {visible}({','.join(':' for _ in range(binding.rank))})", 0),
                     f"integer(c_size_t), target :: fort_extents_{i}({binding.rank})",
                     f"integer(c_int64_t), target :: fort_lowers_{i}({binding.rank})",
                     f"type(fort_scope_layout) :: {layouts[root]}"]
            fact=self.capture(binding)
            if fact["initialized"] == "sections" and fact["sections"]:
                spec += [f"type(fort_scope_section), target :: fort_defined_{i}({len(fact['sections'])})"]
                for j,_box in enumerate(fact["sections"]):
                    for bound in ["lower","upper"]:
                        spec += [f"integer(c_size_t), target :: fort_defined_{i}_{j}_{bound}({binding.rank})"]
        for root, binding in scalars.items():
            if binding.signature()[:2] not in DTYPES and binding.signature()[:2] != ("logical",4):
                raise CompilationError("unsupported scalar capture width")
            dtype = "logical" if binding.signature()[:2] == ("logical",4) else DTYPES[binding.signature()[:2]][0]
            intent = "in" if binding.intent == "in" else "inout"
            spec += [f"{dtype}, intent({intent}) :: {self.visible(routine, root)}"]
        spec += ["integer(c_int64_t) :: fort_context = 0",
                 "integer(c_int64_t) :: " + ", ".join(handles.values()),
                 "integer(c_int) :: fort_status, fort_cleanup",
                 "type(c_ptr) :: fort_host_pointer",
                 f"integer(c_int), parameter :: fort_mode = {1 if self.config.policy == 'sections' else 2}",
                 "type(fort_scope_access) :: fort_access"]
        # Explicit initialization on a local declaration implies SAVE. Assign at
        # entry instead: no scope-local context or state persists across calls.
        spec = [s.replace("fort_context = 0", "fort_context") for s in spec]
        original = [line for call in calls for line in _call(str(call.node.items[0]), map(str,call.actuals))]
        native = [*original, "return"]
        body = ["fort_context = 0"]
        if self.config.policy == "auto":
            body += ["! Coherent estimates are unavailable: successful whole-span native selection.", *native]
        else:
            condition = " .or. ".join(".not. is_contiguous(" + n + ")" for n in names[:len(arrays)])
            body += ["if (fort_scope_serial_caller() == 0) then", *native, "endif",
                     "if (" + condition + ") then", *native, "endif",
                     "fort_status = fort_scope_create(0_c_int, fort_context)"]
            body += ["if (fort_status == FORT_SCOPE_OK) &",
                     f"  fort_status = fort_scope_set_device_budget(fort_context, {self.device_budget}_c_size_t)"]
            for i, (root, binding) in enumerate(arrays.items()):
                visible = self.visible(routine, root)
                _, enum, width = DTYPES[binding.signature()[:2]]
                body += ["if (fort_status == FORT_SCOPE_OK) then"]
                body += _fortran_list(f"fort_extents_{i} = [",
                                     [f"size({visible},{axis},kind=c_size_t)" for axis in range(1,binding.rank+1)], "]", 0)
                body += [f"fort_lowers_{i} = 1_c_int64_t",
                         "fort_host_pointer = c_null_ptr",
                         f"if (all(fort_extents_{i} > 0)) fort_host_pointer = c_loc({visible})",
                         f"{layouts[root]} = fort_scope_layout({binding.rank}, {enum}, {width}_c_size_t, &",
                         f"    fort_host_pointer, c_loc(fort_extents_{i}), c_loc(fort_lowers_{i}), 1_c_int64_t)"]
                fact=self.capture(binding)
                if fact["initialized"] == "sections":
                    boxes=fact["sections"]
                    for j,box in enumerate(boxes):
                        for bound in ["lower","upper"]:
                            body += _fortran_list(f"fort_defined_{i}_{j}_{bound} = [",
                                                  [f"{v}_c_size_t" for v in box[bound]], "]", 0)
                            body += [f"fort_defined_{i}({j+1})%{bound} = c_loc(fort_defined_{i}_{j}_{bound})"]
                    body += _fortran_list("fort_status = fort_scope_register_sections(",
                                          ["fort_context", f"{i+1}_c_int64_t", "1_c_int64_t", layouts[root],
                                           f"c_loc(fort_defined_{i})" if boxes else "c_null_ptr",
                                           f"{len(boxes)}_c_size_t", handles[root]], ")", 0)
                else:
                    body += _fortran_list("fort_status = fort_scope_register(",
                                         ["fort_context", f"{i+1}_c_int64_t", "1_c_int64_t",
                                          layouts[root], f"{flags[root]}_c_int", handles[root]], ")", 0)
                body += ["endif"]
            body += ["if (fort_status /= FORT_SCOPE_OK) then",
                     "if (fort_context /= 0) then",
                     "fort_cleanup = fort_scope_close(fort_context)",
                     "if (fort_cleanup /= FORT_SCOPE_OK) error stop 'shared scope preflight cleanup failed'",
                     "endif", *native, "endif"]
            for call in calls:
                child_leaves, _ = self.closure(call.procedure)
                if child_leaves:
                    clone, arrays_ = self.clone(call.procedure)
                    module = self.analysis.routines[call.procedure].scope.module
                    if module != routine.scope.module:
                        imports.append(f"use {module}, only: {clone}")
                    body += _call(clone, ["fort_context", "fort_mode", *map(str,call.actuals),
                                         *[handles[call.bindings["argument::" + a].root] for a in arrays_]])
                else:
                    mapping = {formal: binding.root for formal, binding in call.bindings.items()}
                    actions, definitions, overwrites = self.native_effects(call.procedure)
                    native_handles = {formal: handles[root] for formal, root in mapping.items() if root in handles}
                    # Hidden visible roots retain their canonical identity.
                    native_handles.update({root: handle for root,handle in handles.items() if not root.startswith("argument::")})
                    body += self.native_call(call, actions, definitions, overwrites, native_handles)
            body += _checked("fort_scope_close(fort_context)")
        text = "\n".join([*header, "use iso_c_binding", "use fort_scoped_memory",
                          *dict.fromkeys(imports), "implicit none", *spec, *body, "end subroutine " + name, ""])
        self.append_procedure(routine.scope.parent, name, text)
        first, last = _span(calls[0].node)[0], _span(calls[-1].node)[1]
        self.add_edit(routine.scope.path, first, last, "\n".join(_call(name,names)) + "\n")
        return {"owner": name, "path": str(routine.scope.path), "first_line": first, "last_line": last,
                "calls": [c.procedure for c in calls], "gpu_leaves": sorted(leaves),
                "resources": [{"resource": root, "visible": self.visible(routine,root),
                               "initialized": self.capture(binding)["initialized"],
                               "initialized_sections": self.capture(binding).get("sections",[])}
                              for root,binding in arrays.items()],
                "estimate_available": False,
                "placement": "forced scoped GPU with native helpers" if self.config.policy == "sections" else
                             "native; coherent scope estimates unavailable",
                "mode": self.config.policy, "participation": "serial"}

    def build_span(self, nodes):
        if len(nodes) < 2:
            return
        before = (dict(self.outputs), {p:list(edits) for p,edits in self.edits.items()}, dict(self.clones))
        try:
            calls = [self.resolve(self.entry,node) for node in nodes]
            leaves = set()
            for call in calls:
                own, _ = self.closure(call.procedure)
                leaves.update(own)
            if not leaves:
                return
            self.scopes.append(self.owner(calls,leaves))
        except CompilationError as error:
            self.outputs, self.edits, self.clones = before
            self.boundaries.append({"first_line":_span(nodes[0])[0], "last_line":_span(nodes[-1])[1],
                                    "reason":str(error)})

    def scan(self, nodes):
        run = []
        for node in nodes:
            if _kind(node) == "Comment" and not str(node).lower().lstrip().startswith("!$omp"):
                continue
            if _kind(node) == "Call_Stmt":
                first,last = _span(node)
                text = self.entry.scope.path.read_text().splitlines()[first-1:last]
                if len(run) < 32 and text and text[0].lstrip().lower().startswith("call ") and all(";" not in line for line in text):
                    try:
                        self.resolve(self.entry,node)
                    except CompilationError as error:
                        self.build_span(run)
                        run = []
                        self.boundaries.append({"first_line":first,"last_line":last,"reason":str(error)})
                    else:
                        run.append(node)
                    continue
            self.build_span(run)
            run = []
            if _kind(node) == "If_Construct":
                self.scan(node.content)
        self.build_span(run)

    def run(self):
        self.scan(_children(self.entry.execution))
        if any(sha256(Path(path).read_bytes()).hexdigest() != digest for path,digest in self.analysis.sources.items()):
            raise CompilationError("source changed while compiler scope artifacts were being prepared")
        provenance = {}
        patches = []
        for path, edits in self.edits.items():
            source = path.read_text()
            lines = source.splitlines(keepends=True)
            # Group insertions at the same boundary; preserve discovery order.
            grouped = {}
            for first,last,replacement in edits:
                grouped.setdefault((first,last),[]).append(replacement)
            ordered = sorted(grouped.items(), reverse=True)
            previous = len(lines)+1
            for (first,last),values in ordered:
                if last >= previous:
                    raise CompilationError("overlapping compiler source edits")
                lines[first-1:last] = ["".join(values)]
                previous = first if last >= first else previous
                patches.append({"source":str(path), "source_sha256":self.analysis.sources[str(path)],
                                "first_line":first, "last_line":last, "replacement":"".join(values)})
            destination = "sources/" + _name("source_",str(path)) + ".f90"
            self.outputs[destination] = "".join(lines)
            provenance[str(path)] = {"sha256":self.analysis.sources[str(path)], "replacement":destination}
        if self.scopes:
            self.outputs.update(self.runtime_outputs)
            self.outputs["scoped-runtime.json"] = json.dumps(self.runtime,indent=2)+"\n"
        report = {
            "schema_version":1, "entry":self.entry.qualified,
            "automatic_scope_available":bool(self.scopes), "scope_count":len(self.scopes),
            "scopes":self.scopes, "boundaries":self.boundaries, "source_edits":patches,
            "sources":provenance, "runtime":self.runtime if self.scopes else None,
            "native_effects":self.analysis.report(self.entry.qualified),
            "automatic_estimate_available":False,
            "capture_facts_sha256":sha256(json.dumps(self.facts,sort_keys=True).encode()).hexdigest(),
            "source_inputs":self.analysis.sources,
            "artifacts_sha256":{name:sha256(content.encode()).hexdigest() for name,content in self.outputs.items()},
            "device_budget_bytes":self.device_budget,
            "limits":{"calls_per_span":32, "closure_depth":8, "physical_native_sections":"whole resources"},
            "build_sources":[
                {"path":name,"language":"cuda" if name.endswith(".cu") else "fortran",
                 "role":"common_runtime" if name in self.runtime_outputs else
                        "original_source" if name.startswith("sources/") else "shared_entry"}
                for name in self.outputs if name.endswith((".cu",".f90"))
            ],
        }
        self.outputs["scope-manifest.json"] = json.dumps(report,indent=2)+"\n"
        return self.outputs, report


def form_source_scopes(paths, entry, *, facts, options, config, contracts=None):
    return ScopeBuilder(paths,entry,facts=facts,options=options,config=config,contracts=contracts).run()
