"""Restrict outline discovery while reusing the ordinary checked graph path."""
from compiler.ir import CompilationError


def native_only(builder, routine=None):
    selection = getattr(builder, "_native_ancestor_selection", None)
    if selection is None:
        return False
    selection.validate(builder.analysis)
    return not (getattr(builder, "_native_ancestor_device_subtree", False)
                or selection.selects(builder.entry if routine is None else routine))


def prepare(builder, nodes):
    selection = getattr(builder, "_native_ancestor_selection", None)
    if selection is not None:
        selection.validate(builder.analysis)
        if native_only(builder):
            return tuple(nodes)
        builder._native_ancestor_device_subtree = True
    return builder.inline.prepare(nodes)


def validate_member(member):
    selection = getattr(member, "native_ancestor_selection", None)
    if selection is None:
        return
    if selection is not getattr(member.builder, "_native_ancestor_selection", None):
        raise CompilationError("native ancestor member lost its original selection authority")
    selection.validate(member.builder.analysis)
    if member.native_ancestor_only != native_only(member.builder):
        raise CompilationError("native ancestor member discovery mode changed")
