import json

from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory


def ast_to_dict(node):
    """Recursively converts an fparser2 AST node into a D3-compatible dictionary."""
    if node is None:
        return None

    # Base cases for primitive types
    if isinstance(node, (str, int, float, bool)):
        return {"name": str(node), "value": str(node)}

    # Handle tuples/lists which fparser sometimes uses to group children
    if isinstance(node, (tuple, list)):
        children = [ast_to_dict(child) for child in node if child is not None]
        if not children:
            return None
        return {"name": "Group", "children": children}

    # Main fparser object representation
    result = {"name": node.__class__.__name__}

    # If the node has a direct string representation (usually leaf nodes)
    if hasattr(node, "string") and isinstance(node.string, str):
        result["value"] = node.string

    children = []
    # fparser exposes sub-nodes typically through 'children'
    if hasattr(node, "children") and node.children:
        for child in node.children:
            child_dict = ast_to_dict(child)
            if child_dict is not None:
                children.append(child_dict)

    # Fallback to 'items' if 'children' is empty but 'items' exists
    elif hasattr(node, "items") and node.items:
        for item in node.items:
            # Prevent infinite recursion on self-referential items
            if item != node and item is not None:
                child_dict = ast_to_dict(item)
                if child_dict is not None:
                    children.append(child_dict)

    if children:
        result["children"] = children

    return result


# --- Your Parsing Code ---
filepath = r"C:\Users\matya\source\repos\fortran-tool-v2\fortran-stencils\gol_module.f90"
reader = FortranFileReader(filepath, ignore_comments=False)
f2008_parser = ParserFactory().create(std="f2008")
parse_tree = f2008_parser(reader)

# --- Generate and save JSON ---
ast_json_data = ast_to_dict(parse_tree)

with open("ast_output.json", "w", encoding="utf-8") as f:
    json.dump(ast_json_data, f, indent=2)

print("AST successfully exported to ast_output.json")
