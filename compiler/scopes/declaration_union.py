"""Compose narrowly checked declaration additions, retaining source semantics."""

from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import FortranSyntaxError, NoMatchError


def _specifications(text):
    parser = ParserFactory().create(std="f2008")
    tree = parser(FortranStringReader("module fort_target_union\n" + text + "\nend module\n"))
    if len(tree.content) != 1:
        return None
    module = tree.content[0]
    if type(module).__name__ != "Module" or len(module.content) != 3:
        return None
    specification = module.content[1]
    if type(specification).__name__ != "Specification_Part":
        return None
    return tuple(specification.content)


def _declarations(text):
    result = _specifications(text)
    return result if result and all(type(item).__name__ == "Type_Declaration_Stmt" for item in result) else None


def union_public_entries(insertions):
    """Deduplicate named PUBLIC declarations at one original module boundary.

    Each insertion must consist entirely of explicit PUBLIC statements naming
    compiler-issued ENTRY symbols. Mixed declarations, visibility defaults,
    operators and executable statements retain ordinary composition rules.
    """
    import re

    names = {}
    try:
        for insertion in insertions:
            declarations = _specifications(insertion)
            if not declarations:
                return None
            for declaration in declarations:
                if type(declaration).__name__ != "Access_Stmt":
                    return None
                access, identifiers = declaration.items
                if str(access).upper() != "PUBLIC" or identifiers is None:
                    return None
                for identifier in identifiers.items:
                    name = str(identifier).lower()
                    if (type(identifier).__name__ != "Name"
                            or not re.fullmatch(r"fort_scope_entry_[0-9a-f]{12}", name)):
                        return None
                    names.setdefault(name, str(identifier))
        return "".join("public :: " + name + "\n" for name in names.values())
    except (ValueError, TypeError, AttributeError, IndexError, FortranSyntaxError, NoMatchError):
        return None


def union_targets(original, replacements):
    """Return an exact intrinsic declaration union, or None for any other edit.

    The original entity order, declarations, shapes and initializers remain
    authoritative. Each alternative must contain every original entity exactly
    once and may only add TARGET. This is deliberately separate from executable
    edit composition, whose overlaps still require one shared implementation.
    """
    try:
        declarations = _declarations(original)
        if declarations is None or len(declarations) != 1:
            return None
        declaration, = declarations
        dtype, attributes, entities = declaration.items
        if type(dtype).__name__ != "Intrinsic_Type_Spec":
            return None
        original_attributes = tuple(str(item) for item in attributes.items) if attributes else ()
        if any(item.partition("(")[0] in {"POINTER", "VOLATILE", "ASYNCHRONOUS", "PARAMETER"}
               for item in original_attributes):
            return None
        common = tuple(item for item in original_attributes if item != "TARGET")
        original_entities = tuple(str(item) for item in entities.items)
        by_name = {str(item.items[0]).lower(): str(item) for item in entities.items}
        if len(by_name) != len(original_entities):
            return None
        targeted = set(by_name) if "TARGET" in original_attributes else set()
        for replacement in replacements:
            alternatives = _declarations(replacement)
            if alternatives is None:
                return None
            seen = set()
            for alternative in alternatives:
                other_type, other_attributes, other_entities = alternative.items
                attrs = tuple(str(item) for item in other_attributes.items) if other_attributes else ()
                if (str(other_type) != str(dtype)
                        or tuple(item for item in attrs if item != "TARGET") != common
                        or len(set(attrs)) != len(attrs)
                        or ("TARGET" in original_attributes and "TARGET" not in attrs)):
                    return None
                for entity in other_entities.items:
                    name = str(entity.items[0]).lower()
                    if name in seen or by_name.get(name) != str(entity):
                        return None
                    seen.add(name)
                    if "TARGET" in attrs:
                        targeted.add(name)
            if seen != set(by_name):
                return None
        groups = []
        for entity in entities.items:
            target = str(entity.items[0]).lower() in targeted
            if not groups or groups[-1][0] != target:
                groups.append((target, []))
            groups[-1][1].append(str(entity))
        return "".join(str(dtype) + (", " + ", ".join((*common, "TARGET") if target else common)
                                     if common or target else "")
                       + " :: " + ", ".join(group) + "\n" for target, group in groups)
    except (ValueError, TypeError, AttributeError, IndexError, FortranSyntaxError, NoMatchError):
        return None
