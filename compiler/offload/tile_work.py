"""Source-bound work geometry for tile-team ranking, without placement authority.

The operation weights are the public graph's static real-arithmetic proxy.
Array traffic counts source loads/stores, including indexed scratch; it is not
a cache-miss measurement. Conditional bodies use an explicit upper bound.
Original phase visits, tile counts and batch launches are exact. Protocol and
compute seconds must come from independently compatible calibration elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from types import SimpleNamespace

from compiler.frontend.lowering import lower_tile_private
from compiler.frontend.tile_private import TilePrivateTeamProof, _clipped, _linear, _nest
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    CompilationError,
    If,
    IntrinsicCall,
    Literal,
    Loop,
    Reference,
    ScalarType,
    Size,
    Unary,
)
from compiler.offload.graph_costs import graph_work_proxy

UINT64_MAX = (1 << 64) - 1


class TileWorkUnavailable(ValueError):
    """Unsupported/overflowing work remains unpriced, with native fallback."""


def _uint(value):
    if type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise TileWorkUnavailable("tile work count exceeds uint64")
    return value


@dataclass(frozen=True)
class PointWork:
    real_work_proxy: int = 0
    external_read_bytes: int = 0
    external_write_bytes: int = 0
    scratch_read_bytes: int = 0
    scratch_write_bytes: int = 0
    conditional_upper_bound: bool = False

    def plus(self, other):
        return PointWork(*(getattr(self, key) + getattr(other, key) for key in _COUNTS),
                         self.conditional_upper_bound or other.conditional_upper_bound)

    def times(self, count):
        return PointWork(*(_uint(getattr(self, key) * count) for key in _COUNTS), self.conditional_upper_bound)


_COUNTS = tuple(name for name in PointWork.__dataclass_fields__ if name != "conditional_upper_bound")


@dataclass(frozen=True)
class AxisWork:
    lower: str
    upper: str
    width: str
    producer_lower_offset: int
    producer_clip_offset: int
    producer_after_clip_offset: int


@dataclass(frozen=True)
class UnitWork:
    source_identity: str
    axes: tuple[AxisWork, ...]
    producer: PointWork
    consumer: PointWork


@dataclass(frozen=True)
class TileWorkTotals:
    work: PointWork
    tiles: int
    producer_visits: int
    consumer_visits: int
    launches: int
    scratch_lease_bytes: int
    original_teams: int
    original_worksharing_units: int
    cooperative_phase_barriers: int


def axis_visits(lower, upper, width, before, clip, after):
    """Sum every original MIN-clipped phase length in O(1), including odd tails.

    Original producer bounds are tile+before:MIN(tile+width+clip,upper)+after.
    The consumer is the special case before=0, clip=-1, after=0.
    """
    if any(type(value) is not int for value in (lower, upper, width, before, clip, after)):
        raise TileWorkUnavailable("tile work requires exact integer geometry")
    if not -(1 << 31) <= lower <= (1 << 31)-1 or not -(1 << 31) <= upper <= (1 << 31)-1:
        raise TileWorkUnavailable("tile work requires original INTEGER4 domain snapshots")
    if not 0 < width <= (1 << 31)-1:
        raise TileWorkUnavailable("tile work requires positive original width")
    if upper < lower:
        return 0, 0
    tiles = (upper-lower)//width + 1
    ceiling, first = width+clip+after-before+1, upper+after-lower-before+1
    if ceiling <= 0:
        return tiles, 0
    full = min(tiles, max(0, (first-ceiling)//width+1))
    positive = min(tiles, max(0, (first-1)//width+1))
    tail = positive-full
    visits = full*ceiling + tail*first - width*(full+positive-1)*tail//2
    return _uint(tiles), _uint(visits)


def _point_body(body, rank):
    for _ in range(rank):
        if len(body.statements) != 1 or not isinstance(body.statements[0], Loop) or body.statements[0].step != 1:
            raise TileWorkUnavailable("tile work requires the issued rectangular phase nest")
        body = body.statements[0].body
    return body


def _point_work(body, scratch):
    def expression(value):
        if isinstance(value, ArrayAccess):
            size = {ScalarType.REAL: 8, ScalarType.REAL32: 4, ScalarType.INTEGER: 4}.get(value.symbol.dtype)
            if size is None:
                raise TileWorkUnavailable("tile work has an unknown field representation")
            return PointWork(**{"scratch_read_bytes" if value.symbol in scratch else "external_read_bytes": size})
        if isinstance(value, (Literal, Reference, Size)):
            return PointWork()
        children = ((value.left, value.right) if isinstance(value, Binary) else
                    (value.operand,) if isinstance(value, Unary) else value.arguments if isinstance(value, IntrinsicCall) else None)
        if children is None:
            raise TileWorkUnavailable("tile work has an unsupported source expression")
        result = PointWork()
        for child in children:
            result = result.plus(expression(child))
        return result

    def block(value):
        result = PointWork()
        for statement in value.statements:
            if isinstance(statement, Assignment):
                item = expression(statement.value)
                if isinstance(statement.target, ArrayAccess):
                    target = expression(statement.target)
                    item = item.plus(PointWork(external_write_bytes=target.external_read_bytes,
                                               scratch_write_bytes=target.scratch_read_bytes))
            elif isinstance(statement, If):
                left, right = block(statement.then_body), block(statement.else_body)
                item = expression(statement.condition).plus(PointWork(
                    *(max(getattr(left, name), getattr(right, name)) for name in _COUNTS), True))
            else:
                raise TileWorkUnavailable("tile work cannot price an unproved inner loop")
            result = result.plus(item)
        return result

    result = block(body)
    arithmetic, unknown = graph_work_proxy(SimpleNamespace(body=body))
    if unknown:
        raise TileWorkUnavailable("tile work cannot invent a dynamic inner trip count")
    return PointWork(arithmetic, result.external_read_bytes, result.external_write_bytes,
                     result.scratch_read_bytes, result.scratch_write_bytes, result.conditional_upper_bound)


def _derive(team, mapping):
    if type(team) is not TilePrivateTeamProof:
        raise CompilationError("tile work requires an exact compiler-issued whole-team proof")
    team.validate()
    if mapping not in {"cooperative", "thread"}:
        raise TileWorkUnavailable("tile work has an unknown mapping")
    units = []
    for index, proof in enumerate(team.units):
        lowered = lower_tile_private(proof, region_id=index)
        rank = len(lowered.region.outer_loops)
        producer, _ = _nest(proof.phases[0], rank)
        axes = []
        for source, (_node, _point, bounds) in zip(proof.public()["axes"], producer, strict=True):
            clip, after = _clipped(bounds[1], source["tile"], source["width"], source["upper"])
            before = _linear(bounds[0])[0] - _linear(source["tile"])[0]
            if any(not -(1 << 31) <= value <= (1 << 31)-1 for value in (before, clip, after)):
                raise TileWorkUnavailable("tile work clipping constant exceeds INTEGER4")
            axes.append(AxisWork(source["lower"], source["upper"], source["width"], before, clip, after))
        scratch = set(lowered.region.scratch_symbols)
        units.append(UnitWork(proof.identity, tuple(axes),
            _point_work(_point_body(lowered.region.producer, rank), scratch),
            _point_work(_point_body(lowered.region.consumer, rank), scratch)))
    record = {"version": 1, "team_identity": team.identity, "mapping": mapping, "units": [asdict(unit) for unit in units],
              "model": "static real-arithmetic and source-array-traffic proxy; measured rates required separately",
              "original_fortran_family_attestation": False, "speedup_guaranteed": False,
              "placement_authorized": False, "definitions_authorized": False,
              "unpriced": ["source transaction and scratch lifecycle", "native team/worksharing protocol", "mapping/control/barrier efficiency"]}
    return tuple(units), json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class TileTeamWork:
    identity: str
    units: tuple[UnitWork, ...]
    mapping: str
    _team: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)

    def validate(self):
        units, record = _derive(self._team, self.mapping)
        if units != self.units or record != self._record or sha256(record.encode()).hexdigest() != self.identity:
            raise CompilationError("tile work changed its exact issued source geometry or operation counts")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def evaluate(self, domain, *, batch_tiles, scratch_lease_bytes):
        """Use reached geometry suppliers, never private descriptors or payload.

        domain(axis) yields (lower,upper,width). Later suppliers and batch/lease
        suppliers are untouched after an earlier empty axis. This is work-only:
        source/alias/FP/definition/budget checks remain independent obligations.
        """
        self.validate()
        shape = []
        for axis in range(len(self.units[0].axes)):
            lower, upper, width = domain(axis)
            tiles, _ = axis_visits(lower, upper, width, 0, -1, 0)
            if not tiles:
                return TileWorkTotals(PointWork(), 0, 0, 0, 0, 0, 0, 0, 0)
            shape.append((lower, upper, width))
        batch, lease = _uint(batch_tiles()), _uint(scratch_lease_bytes())
        if not batch or not lease:
            raise TileWorkUnavailable("positive tile work requires its preplanned batch and lease")
        work, all_tiles, producer_visits, consumer_visits, launches = PointWork(), 0, 0, 0, 0
        for unit in self.units:
            tiles = producer = consumer = 1
            for original, values in zip(unit.axes, shape, strict=True):
                count, produced = axis_visits(*values, original.producer_lower_offset,
                    original.producer_clip_offset, original.producer_after_clip_offset)
                _, consumed = axis_visits(*values, 0, -1, 0)
                tiles, producer, consumer = _uint(tiles*count), _uint(producer*produced), _uint(consumer*consumed)
            all_tiles, producer_visits = _uint(all_tiles+tiles), _uint(producer_visits+producer)
            consumer_visits = _uint(consumer_visits+consumer)
            launches = _uint(launches+tiles//batch+int(tiles % batch != 0))
            work = work.plus(unit.producer.times(producer)).plus(unit.consumer.times(consumer))
            for name in _COUNTS:
                _uint(getattr(work, name))
        return TileWorkTotals(work, all_tiles, producer_visits, consumer_visits, launches, lease, 1,
                              len(self.units), _uint(2*all_tiles) if self.mapping == "cooperative" else 0)


def tile_team_work(team, *, mapping="cooperative"):
    units, record = _derive(team, mapping)
    return TileTeamWork(sha256(record.encode()).hexdigest(), units, mapping, team, record)
