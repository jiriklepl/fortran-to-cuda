"""Public resident-workspace procedures and private interoperable bridges."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from compiler.emission.common.abi import AbiArgument, abi_arguments
from compiler.emission.common.sessions import session_names
from compiler.emission.fortran.declarations import FortranKinds, abi_call, abi_declaration, public_declaration
from compiler.emission.fortran.formatting import _fortran_line, _fortran_list
from compiler.ir import FunctionIR


def fortran_sessions(
    function: FunctionIR, kinds: FortranKinds, unique: Callable[[str], str]
) -> tuple[list[str], list[str], list[str]]:
    names = session_names(function)
    arrays = tuple(symbol for symbol in function.parameters if symbol.rank)
    scalars = tuple(symbol for symbol in function.parameters if not symbol.rank)
    work = unique("work")
    token = unique("fort_internal_token")
    result = unique("fort_internal_result")
    create = unique("fort_internal_create")
    run = unique("fort_internal_run")
    destroy = unique("fort_internal_destroy")
    validate = unique("fort_internal_validate")
    declarations = [
        *(f"  public :: {name}" for name in names.__dict__.values()),
        f"  type :: {names.workspace}",
        "    private",
        f"    integer({kinds.token}) :: token = 0",
        f"  end type {names.workspace}",
        "",
    ]
    interfaces = []
    bodies = []

    def interface(name, c_name, arguments, *, returning=False):
        keyword = "function" if returning else "subroutine"
        suffix = f") bind(C, name='{c_name}')"
        if returning:
            suffix += f" result({result})"
        interfaces.extend(_fortran_list(f"{keyword} {name}(", arguments, suffix, 4))
        interfaces.extend(
            f"      import :: {kind}"
            for kind in (kinds.integer, kinds.real, kinds.real32, kinds.extent, kinds.logical, kinds.token)
        )
        if returning:
            interfaces.append(f"      integer({kinds.token}) :: {result}")

    array_abi = abi_arguments(arrays)
    interface(create, "cpp_" + names.create, [a.name for a in array_abi], returning=True)
    for argument in array_abi:
        interfaces.extend(
            _fortran_line(
                abi_declaration(argument, kinds)
                .replace("intent(out)", "intent(in)")
                .replace("intent(inout)", "intent(in)"),
                6,
            )
        )
    interfaces.append(f"    end function {create}")
    interface(run, "cpp_" + names.run, [token, *[s.cpp_name for s in scalars]])
    interfaces.append(f"      integer({kinds.token}), value, intent(in) :: {token}")
    for symbol in scalars:
        interfaces.extend(_fortran_line(abi_declaration(AbiArgument(symbol), kinds), 6))
    interfaces.append(f"    end subroutine {run}")
    for local, c_name in ((destroy, "cpp_" + names.destroy), (validate, "cpp_" + names.workspace + "_validate")):
        interface(local, c_name, [token])
        interfaces.extend(
            [f"      integer({kinds.token}), value, intent(in) :: {token}", f"    end subroutine {local}"]
        )
    update_names = {}
    for direction in ("device", "host"):
        for symbol in arrays:
            local = unique(f"fort_internal_update_{direction}_{symbol.id}")
            update_names[direction, symbol.id] = local
            symbol_abi = abi_arguments((symbol,))
            interface(
                local,
                f"cpp_{getattr(names, 'update_' + direction)}_{symbol.id}",
                [token, *[a.name for a in symbol_abi]],
            )
            interfaces.append(f"      integer({kinds.token}), value, intent(in) :: {token}")
            for argument in symbol_abi:
                declaration = abi_declaration(argument, kinds)
                if argument.dimension is None:
                    declaration = abi_declaration(
                        replace(argument, symbol=replace(symbol, intent="in" if direction == "device" else "out")),
                        kinds,
                    )
                interfaces.extend(_fortran_line(declaration, 6))
            interfaces.append(f"    end subroutine {local}")

    def procedure(name, arguments, workspace_intent):
        bodies.extend(_fortran_list(f"subroutine {name}(", [work, *arguments], ")", 2))
        bodies.append(f"    type({names.workspace}), intent({workspace_intent}) :: {work}")

    def end(name):
        bodies.extend([f"  end subroutine {name}", ""])

    # Public keyword labels stay intact; private bridge names cannot hide intrinsics.
    for action, symbols in (("create", arrays), ("run", scalars), ("update_device", arrays), ("update_host", arrays)):
        public = getattr(names, action)
        bridge = unique("fort_internal_" + action + "_bridge")
        optional = action.startswith("update_")
        workspace_intent = "inout" if action == "create" else "in"
        intent = "out" if action == "update_host" else "in"
        procedure(public, [s.name.lower() for s in symbols], workspace_intent)
        for symbol in symbols:
            declaration = public_declaration(replace(symbol, intent=intent), symbol.name.lower())
            if optional:
                declaration = declaration.replace(" ::", ", optional ::")
            bodies.extend(_fortran_line(declaration, 4))
        bodies.extend(_fortran_list(f"call {bridge}(", [work, *[s.name.lower() for s in symbols]], ")", 4))
        end(public)
        bodies.extend(_fortran_list(f"subroutine {bridge}(", [work, *[s.cpp_name for s in symbols]], ")", 2))
        bodies.append("    intrinsic :: size, int, real, logical, present")
        bodies.append(f"    type({names.workspace}), intent({workspace_intent}) :: {work}")
        for symbol in symbols:
            declaration = public_declaration(replace(symbol, intent=intent), symbol.cpp_name)
            if optional:
                declaration = declaration.replace(" ::", ", optional ::")
            bodies.extend(_fortran_line(declaration, 4))
        if action == "create":
            bodies.append(f"    if ({work}%token /= 0) error stop 'workspace already initialized'")
            bodies.extend(_fortran_list(f"{work}%token = {create}(", [abi_call(a, kinds) for a in array_abi], ")", 4))
        elif action == "run":
            bodies.extend(
                _fortran_list(
                    f"call {run}(", [f"{work}%token", *[abi_call(AbiArgument(s), kinds) for s in scalars]], ")", 4
                )
            )
        else:
            direction = action.removeprefix("update_")
            bodies.append(f"    call {validate}({work}%token)")
            for symbol in arrays:
                bodies.append(f"    if (present({symbol.cpp_name})) then")
                bodies.extend(
                    _fortran_list(
                        f"call {update_names[direction, symbol.id]}(",
                        [f"{work}%token", *[abi_call(a, kinds) for a in abi_arguments((symbol,))]],
                        ")",
                        6,
                    )
                )
                bodies.append("    end if")
        end(bridge)
    procedure(names.destroy, [], "inout")
    bodies.extend([f"    call {destroy}({work}%token)", f"    {work}%token = 0"])
    end(names.destroy)
    return declarations, interfaces, bodies
