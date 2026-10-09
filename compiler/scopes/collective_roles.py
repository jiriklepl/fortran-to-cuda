"""Bounded original-source roles for qualified existing-team companions.

Completion for a serial caller is not authority to enter an existing team. This
proof additionally requires direct orphaned worksharing loops and excludes
effects whose participation or per-thread state needs a more general model.
"""

from __future__ import annotations

from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError

KIND = "existing_team_worksharing"
COMPLETION = "all_participants_before_effect_commit"
LOOPS = {"Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct"}


def _automatic_scalar_iterations(analysis, routine, loop):
    """Prove local temporaries do not carry values between work iterations.

    This is a source participation prerequisite. Numerical lowering still
    independently proves privatization and dependencies before offering a GPU
    worker. Conditional definitions are deliberately outside this small proof.
    """
    temporaries = {}
    forbidden = {"save", "allocatable", "pointer", "optional", "volatile", "asynchronous"}
    for node in walk(loop):
        if _kind(node) != "Assignment_Stmt" or _kind(node.items[0]) != "Name":
            continue
        binding = analysis._binding(routine.scope, node.items[0])
        if binding is not None and binding.rank:
            continue
        if (binding is None or binding.name in routine.arguments
                or not binding.root.startswith(routine.qualified + "::")
                or binding.attributes & forbidden):
            return "existing-team scalar temporary requires nonpersistent procedure-local storage"
        temporaries[binding.name] = binding

    def reads(node, defined):
        return any(str(name).lower() in temporaries and str(name).lower() not in defined
                   for name in walk(node) if _kind(name) == "Name")

    def sequence(nodes, defined):
        defined = set(defined)
        for node in nodes:
            kind = _kind(node)
            if kind in LOOPS:
                children = [child for child in _children(node) if _kind(child) != "Comment"]
                control = next((item for item in _children(children[0]) if _kind(item) == "Loop_Control"), None)
                if control is None or control.items[1] is None:
                    return "existing-team local temporary requires counted loops"
                iterator, bounds = control.items[1]
                if any(reads(bound, defined) for bound in bounds):
                    return "existing-team local temporary is read before definition in an iteration"
                error = sequence(children[1:-1], defined | {str(iterator).lower()})
                if error:
                    return error
                # A nested loop may be empty, so its definitions do not escape.
            elif kind == "Assignment_Stmt":
                if reads(node.items[2], defined):
                    return "existing-team local temporary is read before definition in an iteration"
                lhs = node.items[0]
                if _kind(lhs) == "Name" and str(lhs).lower() in temporaries:
                    defined.add(str(lhs).lower())
                elif reads(lhs, defined):
                    return "existing-team local temporary is read before definition in an iteration"
            elif kind not in {"Comment", "Continue_Stmt"}:
                # Preserve existing scalar-free conditional support. A richer
                # path join is unnecessary for the initial automatic locals.
                if any(_kind(item) == "Assignment_Stmt" and _kind(item.items[0]) == "Name"
                       and str(item.items[0]).lower() in temporaries for item in walk(node)):
                    return "existing-team local temporary has an unsupported conditional definition"
                if reads(node, defined):
                    return "existing-team local temporary is read before definition in an iteration"
        return None

    return sequence([loop], set())


def prove_existing_team_worksharing(analysis: SourceEffects, qualified_procedure: str, *, assertion=None):
    """Describe original clause-free DO worksharing; never authorize callers.

    The source companion separately proves full-team participation and capture
    agreement before any barriers. This helper does not choose CPU/GPU roles,
    infer an assertion, rewrite source, or relax the serial completion contract.
    """
    routine = analysis.routines.get(qualified_procedure)
    proof = {"available": False, "reason": None,
             "source": str(routine.scope.path) if routine else None,
             "source_sha256": analysis.sources[str(routine.scope.path)] if routine else None,
             "kind": KIND, "completion": COMPLETION}

    def boundary(reason):
        proof["reason"] = reason
        return proof

    if routine is None:
        return boundary("existing-team role requires a source-backed qualified procedure")
    try:
        summary = analysis.summarize(qualified_procedure)
    except CompilationError as error:
        return boundary(str(error))
    if not summary["complete"]:
        return boundary("existing-team role requires complete original source effects")
    if any(operation["kind"] in {"call", "native_contract"} for operation in summary["operations"]):
        return boundary("existing-team role initially requires a direct leaf without source or opaque calls")
    if summary["persistent_state"]:
        return boundary("existing-team role excludes persistent local state")
    for operation in summary["operations"]:
        if operation["kind"] in {"write", "overwrite"} and not operation["rank"]:
            return boundary("existing-team role excludes externally visible scalar writes")
        if operation["kind"] == "control" and operation["source"].strip().lower().startswith("return"):
            return boundary("existing-team worksharing cannot contain an early return")
    # Reuse its bounded completion/module-ownership proof only as a prerequisite;
    # the stricter original-source participation checks below establish the role.
    completion = summary["native_completion"]
    if not completion["available"]:
        return boundary("native completion is unproven: " + completion["reason"])

    direct = {}
    for node in _children(routine.execution):
        if _kind(node) in LOOPS:
            body = [child for child in _children(node) if _kind(child) != "Comment"]
            if body and _kind(body[-1]) == "End_Do_Stmt":
                direct[id(body[0])] = (node, body[-1])
        elif _kind(node) != "Comment":
            return boundary("existing-team role requires direct worksharing loops without outside computation")
    nodes = [node for node in walk(routine.scope.node)
             if _kind(node) == "Comment" or _kind(node).endswith("_Stmt")]
    positions = {id(node): index for index, node in enumerate(nodes)}

    def directive(node):
        text = str(node).lstrip().lower()
        return text[5:].split() if _kind(node) == "Comment" and text.startswith("!$omp") else None

    def next_statement(index):
        while index < len(nodes) and _kind(nodes[index]) == "Comment" and directive(nodes[index]) is None:
            index += 1
        return index

    covered, index = set(), 0
    while index < len(nodes):
        tokens = directive(nodes[index])
        if tokens is None:
            index += 1
            continue
        if tokens != ["do"]:
            return boundary("existing-team role requires only matched clause-free orphaned DO directives")
        header = next_statement(index + 1)
        if header == len(nodes) or id(nodes[header]) not in direct:
            return boundary("existing-team DO must be associated with a direct complete original loop")
        loop, end = direct[id(nodes[header])]
        last = positions[id(end)]
        if any(directive(node) is not None for node in nodes[header:last + 1]):
            return boundary("existing-team role excludes nested worksharing or other directives")
        close = next_statement(last + 1)
        if close == len(nodes) or directive(nodes[close]) != ["end", "do"]:
            return boundary("existing-team DO requires a matching clause-free end directive")
        scalar_reason = _automatic_scalar_iterations(analysis, routine, loop)
        if scalar_reason:
            return boundary(scalar_reason)
        covered.add(id(loop))
        index = close + 1
    if not covered:
        return boundary("existing-team role requires original worksharing directives")
    if covered != {id(loop) for loop, _end in direct.values()}:
        return boundary("an original loop lacks direct existing-team worksharing")
    if assertion is not None and (not isinstance(assertion, dict) or any(
            assertion.get(field) != proof[field] for field in ("source", "source_sha256", "kind", "completion"))):
        return boundary("supplied native participation assertion disagrees with original source role")
    try:
        analysis.inputs.verify()
    except CompilationError as error:
        return boundary(str(error))
    proof["available"] = True
    return proof
