"""Owned emission values and their one-shot consumption, after checked preparation.

The checked execution plan remains source/effect authority. These private
capsules carry only finished compiler output; they cannot be imported from a
manifest or used to acquire storage authority. No builder, binding, source node
or provider is reachable from an issued image's publication payload.
"""
import json
from collections import Counter
from hashlib import sha256
from pathlib import PurePosixPath
from types import FunctionType
from weakref import WeakSet

from compiler.ir import CompilationError

_ISSUER = object()
_CONTROLS = WeakSet()


def _fail(message):
    raise CompilationError("owned emission image " + message)


def _primitive(value):
    kind = type(value)
    if kind is str or kind is int or kind is bool or kind is bytes or value is None:
        return
    if kind is tuple:
        for item in value:
            _primitive(item)
        return
    _fail("requires exact primitive values")


def _control(value):
    if type(value) is not EmissionPreparation and type(value) is not OwnedEmissionImage:
        _fail("requires its exact issued capsule")
    if tuple.__len__(value) < 3:
        _fail("lost its issued control")
    control = tuple.__getitem__(value, 2)
    if type(control) is not FunctionType or control not in _CONTROLS:
        _fail("lost its issued control")
    return control


def _json_record(value):
    if type(value) is not bytes:
        _fail("diagnostics require owned JSON bytes")
    try:
        record = json.loads(value.decode("utf-8"), parse_constant=lambda _: _fail("rejects nonfinite JSON"))
    except (UnicodeError, ValueError) as error:
        raise CompilationError("owned emission image diagnostics are invalid JSON") from error
    if type(record) is not dict:
        _fail("diagnostics require a JSON object")
    try:
        json.dumps(record, allow_nan=False)
    except ValueError as error:
        raise CompilationError("owned emission image diagnostics are not finite JSON") from error
    return value


def _edits(values):
    if type(values) is not tuple:
        _fail("edits require an exact tuple")
    for edit in values:
        if (type(edit) is not tuple or len(edit) != 5 or type(edit[0]) is not str
                or not edit[0] or type(edit[1]) is not int or type(edit[2]) is not int
                or edit[1] < 1 or edit[2] < edit[1] - 1
                or type(edit[3]) is not str or type(edit[4]) is not bool):
            _fail("has an invalid finished source edit")
    return values


class EmissionPreparation(tuple):
    """Plan-bound preparation capability; never execution/storage authority."""
    __slots__ = ()

    def __new__(cls, plan, tickets, control, sources, *, issuer=None):
        if issuer is not _ISSUER:
            _fail("preparation requires its issuing plan")
        return tuple.__new__(cls, (plan, tickets, control, sources))

    @property
    def tickets(self):
        _control(self)("identity", None, self)
        return tuple.__getitem__(self, 1)

    @property
    def sealed(self):
        return _control(self)("sealed", None, self)

    @property
    def source_paths(self):
        _control(self)("identity", None, self)
        return tuple.__getitem__(self, 3)

    def check(self, plan):
        return _control(self)("check", plan, self)

    def ticket(self, member, unit):
        if type(member) is not int or type(unit) is not int:
            _fail("ticket coordinates require exact integers")
        return _control(self)("ticket", (member, unit), self)

    def release(self):
        _control(self)("release", None, self)


def _prepare(plan, tickets, sources):
    if type(tickets) is not tuple or type(sources) is not tuple:
        _fail("tickets require an exact tuple")
    _primitive(tickets)
    if any(type(path) is not str or not path for path in sources):
        _fail("source paths require exact strings")
    coordinates = {}
    for ticket in tickets:
        if (type(ticket) is not tuple or len(ticket) != 9 or type(ticket[8]) is not tuple
                or len(ticket[8]) != 3 or ticket[8][0] not in sources
                or type(ticket[2]) is not int or type(ticket[4]) is not int
                or ticket[2] < 0 or ticket[4] < 0
                or type(ticket[8][1]) is not int or type(ticket[8][2]) is not int
                or ticket[8][1] < 1 or ticket[8][2] < ticket[8][1]):
            _fail("ticket lost its original source geometry")
        key = ticket[2], ticket[4]
        if key in coordinates:
            _fail("has duplicate planned unit coordinates")
        coordinates[key] = ticket
    if len(set(tickets)) != len(tickets):
        _fail("has duplicate planned tickets")
    builders = {id(builder): builder for builder, _analysis, _routine, _config, _facts in plan._contexts}
    builders[id(plan._owner.builder)] = plan._owner.builder
    markers = tuple((builder, getattr(builder, "_generation_transaction_marker", None))
                    for builder in builders.values())
    phase = "preparing"

    def control(operation, candidate, receiver):
        nonlocal phase
        if receiver is not preparation:
            _fail("preparation is not its issued capsule")
        if operation == "identity":
            return None
        if operation == "release":
            phase = "released"
            return None
        if operation == "sealed":
            return phase == "sealed"
        if operation == "ticket":
            if phase == "released" or candidate not in coordinates:
                _fail("does not own the requested live unit ticket")
            return coordinates[candidate]
        if candidate is not plan or phase == "released":
            _fail("preparation is foreign or released")
        if any(getattr(builder, "_generation_transaction_marker", None) is not marker
               for builder, marker in markers):
            _fail("preparation changed its generation transaction")
        if operation == "check":
            return None
        if operation == "seal" and phase == "preparing":
            phase = "sealed"
            return None
        _fail("preparation has already issued its image")

    _CONTROLS.add(control)
    preparation = EmissionPreparation(plan, tickets, control, sources, issuer=_ISSUER)
    return preparation


class OwnedEmissionImage(tuple):
    """Immutable payload plus a private one-shot cursor, with no live inputs."""
    __slots__ = ()

    def __new__(cls, identity, payload, consume, *, issuer=None):
        if issuer is not _ISSUER:
            _fail("requires its issuing checked plan")
        return tuple.__new__(cls, (identity, payload, consume))

    @property
    def identity(self):
        _control(self)("identity", None, self)
        return tuple.__getitem__(self, 0)

    @property
    def payload(self):
        _control(self)("identity", None, self)
        return tuple.__getitem__(self, 1)

    def consume(self, ticket):
        return _control(self)("unit", ticket, self)

    def finish(self):
        return _control(self)("finish", None, self)


def _seal(preparation, plan, *, units, edits=(), artifacts=(), scopes=()):
    if type(preparation) is not EmissionPreparation:
        _fail("requires its exact preparation capability")
    preparation.check(plan)
    if type(units) is not tuple or len(units) != len(preparation.tickets):
        _fail("requires every planned unit exactly once")
    owned_units = []
    for expected, unit in zip(preparation.tickets, units, strict=True):
        if type(unit) is not tuple or len(unit) != 3:
            _fail("requires a ticket, finished edits and diagnostic bytes")
        ticket, unit_edits, diagnostics = unit
        _primitive(ticket)
        if type(ticket) is not tuple or ticket != expected:
            _fail("unit ticket or order differs from the checked plan")
        unit_edits = _edits(unit_edits)
        if not unit_edits:
            _fail("unit has no finished source edit")
        path, first, last = expected[8]
        if any(edit[0] != path or edit[1] < first or edit[1] > last + 1
               or edit[2] > last for edit in unit_edits):
            _fail("unit edit differs from its original source position")
        owned_units.append((ticket, unit_edits, _json_record(diagnostics)))
    owned_units = tuple(owned_units)
    edits = _edits(edits)
    if any(edit[0] not in preparation.source_paths for edit in edits):
        _fail("edit has no checked source owner")
    # Validate exact builtin shapes before hashing. Nonunit declarations and
    # controls may add other edits; every consumed unit's actual code must be
    # present in the complete published state with its exact multiplicity.
    published = Counter(edits)
    required = Counter(edit for _ticket, unit_edits, _diagnostics in owned_units for edit in unit_edits)
    if any(published[edit] != count for edit, count in required.items()):
        _fail("published source edits omit, replace or repeat a checked unit")
    if type(artifacts) is not tuple or type(scopes) is not tuple:
        _fail("artifacts and scope records require exact tuples")
    paths = set()
    for item in artifacts:
        if (type(item) is not tuple or len(item) != 2 or type(item[0]) is not str
                or type(item[1]) is not bytes or not item[0]
                or PurePosixPath(item[0]).is_absolute() or ".." in PurePosixPath(item[0]).parts
                or item[0] in paths):
            _fail("has an invalid or repeated generated artifact")
        paths.add(item[0])
    for scope in scopes:
        _json_record(scope)
    # Only exact primitive values cross this boundary. The encoded bytes also
    # bind the complete output independently of legacy graph object aliases.
    payload = json.dumps({
        "version": 1, "plan": plan.identity, "execution": plan.execution_identity,
        "units": [(ticket, unit_edits, diagnostics.decode("utf-8"))
                  for ticket, unit_edits, diagnostics in owned_units],
        "edits": edits, "artifacts": [(path, data.hex()) for path, data in artifacts],
        "scopes": [scope.decode("utf-8") for scope in scopes],
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    identity = sha256(payload).hexdigest()
    position, finished = 0, False

    def consume(operation, ticket, receiver):
        nonlocal position, finished
        if receiver is not image:
            _fail("publication is not its issued capsule")
        if operation == "identity":
            return None
        if finished:
            _fail("publication has already completed")
        if operation == "finish":
            if position != len(owned_units):
                _fail("publication omitted a checked unit")
            finished = True
            return edits, artifacts, scopes
        _primitive(ticket)
        if (type(ticket) is not tuple or position >= len(owned_units)
                or ticket != owned_units[position][0]):
            _fail("publication repeats, replaces or reorders a checked unit")
        _, unit_edits, diagnostics = owned_units[position]
        position += 1
        return unit_edits, diagnostics

    _control(preparation)("seal", plan, preparation)
    _CONTROLS.add(consume)
    image = OwnedEmissionImage(identity, payload, consume, issuer=_ISSUER)
    return image
