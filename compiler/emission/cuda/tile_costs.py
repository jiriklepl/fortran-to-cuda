"""Pure checked work-count emission; no CUDA, allocation, quote or authority.

Only already-reached integer geometry and planned batch/lease expressions are
inputs. Counts describe the source work proxy; calibrated runtime costs and live
coherence must be supplied independently before an AUTO decision can be made.
"""

import re

from compiler.ir import CompilationError
from compiler.offload.tile_work import TileTeamWork

_SUPPORT = r"""
#include <algorithm>
#include <cstdint>
#include <climits>
namespace fort_tile_cost {
struct totals {
 std::uint64_t real_work_proxy=0,external_read_bytes=0,external_write_bytes=0,scratch_read_bytes=0,scratch_write_bytes=0;
 std::uint64_t tiles=0,producer_visits=0,consumer_visits=0,launches=0,scratch_lease_bytes=0;
 std::uint64_t original_teams=0,original_worksharing_units=0,cooperative_phase_barriers=0;
 bool conditional_upper_bound=false;
};
static bool add(std::uint64_t a,std::uint64_t b,std::uint64_t& out) {
 if(b>UINT64_MAX-a)return false;out=a+b;return true;
}
static bool multiply(std::uint64_t a,std::uint64_t b,std::uint64_t& out) {
 if(b&&a>UINT64_MAX/b)return false;out=a*b;return true;
}
static bool axis(long long lower,long long upper,long long width,long long before,long long clip,long long after,
                 std::uint64_t& tiles,std::uint64_t& visits) {
 if(lower<INT_MIN||lower>INT_MAX||upper<INT_MIN||upper>INT_MAX||width<=0||width>INT_MAX)return false;
 tiles=visits=0;if(upper<lower)return true;
 tiles=static_cast<std::uint64_t>((upper-lower)/width+1);
 const long long ceiling=width+clip+after-before+1,first=upper+after-lower-before+1;
 if(ceiling<=0||first<=0)return true;
 // Avoid signed floor-division differences when no saturated tile exists.
 const std::uint64_t full=first>=ceiling?std::min(tiles,static_cast<std::uint64_t>((first-ceiling)/width+1)):0;
 const std::uint64_t positive=std::min(tiles,static_cast<std::uint64_t>((first-1)/width+1));
 const std::uint64_t tail=positive-full;
 std::uint64_t fixed=0,residual=0;
 if(!multiply(full,static_cast<std::uint64_t>(ceiling),fixed))return false;
 if(tail) {
  const auto a=static_cast<std::uint64_t>(first-static_cast<long long>(full)*width);
  const auto b=static_cast<std::uint64_t>(first-static_cast<long long>(positive-1)*width);
  const auto pair=a+b;
  if(!multiply(tail%2?tail:tail/2,tail%2?pair/2:pair,residual))return false;
 }
 return add(fixed,residual,visits);
}
static bool accumulate(totals& out,std::uint64_t visits,std::uint64_t real,std::uint64_t er,std::uint64_t ew,
                       std::uint64_t sr,std::uint64_t sw) {
 const std::uint64_t weights[]={real,er,ew,sr,sw};
 std::uint64_t* values[]={&out.real_work_proxy,&out.external_read_bytes,&out.external_write_bytes,&out.scratch_read_bytes,&out.scratch_write_bytes};
 for(unsigned i=0;i<5;++i){std::uint64_t amount=0;if(!multiply(visits,weights[i],amount)||!add(*values[i],amount,*values[i]))return false;}
 return true;
}
}
"""


def tile_work_support():
    return _SUPPORT


def generate_tile_work_function(work, *, name, parameters, domains, batch_tiles, scratch_lease_bytes):
    """Emit a bool function; false means unavailable and leaves output zero.

    Expression strings are internal compiler expressions, not user-provided
    source. Later axes and private batch/lease expressions remain behind every
    earlier positive-domain check. This function creates no runtime state.
    """
    if type(work) is not TileTeamWork:
        raise CompilationError("tile cost emission requires source-bound tile work")
    work.validate()
    if not re.fullmatch(r"[A-Za-z_]\w*", name) or len(domains) != len(work.units[0].axes):
        raise CompilationError("tile cost emission has an invalid name or domain rank")
    lines = [f"static bool {name}("+",".join((*parameters, "fort_tile_cost::totals& output"))+") {",
             "output={}; fort_tile_cost::totals result{};"]
    for index, expressions in enumerate(domains):
        if len(expressions) != 3:
            raise CompilationError("tile cost emission requires lower/upper/width snapshots")
        lines += [f"const long long lo{index}=({expressions[0]}),hi{index}=({expressions[1]}),width{index}=({expressions[2]});",
                  f"std::uint64_t nt{index}=0,nc{index}=0;",
                  f"if(!fort_tile_cost::axis(lo{index},hi{index},width{index},0,-1,0,nt{index},nc{index}))return false;",
                  f"if(!nt{index})return true;"]
    lines += [f"const std::uint64_t batch=({batch_tiles}),lease=({scratch_lease_bytes});", "if(!batch||!lease)return false;"]
    for unit in work.units:
        lines += ["{std::uint64_t tiles=1,producer=1,consumer=1;"]
        for index, axis in enumerate(unit.axes):
            lines += ["{std::uint64_t nt=0,np=0;",
                      f"if(!fort_tile_cost::axis(lo{index},hi{index},width{index},{axis.producer_lower_offset},{axis.producer_clip_offset},{axis.producer_after_clip_offset},nt,np))return false;",
                      f"if(!fort_tile_cost::multiply(tiles,nt,tiles)||!fort_tile_cost::multiply(producer,np,producer)||!fort_tile_cost::multiply(consumer,nc{index},consumer))return false;", "}"]
        lines += ["if(!fort_tile_cost::add(result.tiles,tiles,result.tiles)||!fort_tile_cost::add(result.producer_visits,producer,result.producer_visits)||!fort_tile_cost::add(result.consumer_visits,consumer,result.consumer_visits))return false;",
                  "const std::uint64_t launches=tiles/batch+(tiles%batch!=0);",
                  "if(!fort_tile_cost::add(result.launches,launches,result.launches))return false;"]
        for visits, point in (("producer", unit.producer), ("consumer", unit.consumer)):
            counts = (point.real_work_proxy, point.external_read_bytes, point.external_write_bytes,
                      point.scratch_read_bytes, point.scratch_write_bytes)
            lines.append(f"if(!fort_tile_cost::accumulate(result,{visits},"+",".join(map(str, counts))+"))return false;")
            if point.conditional_upper_bound:
                lines.append("result.conditional_upper_bound=true;")
        lines.append("}")
    lines += [f"result.original_teams=1;result.original_worksharing_units={len(work.units)};result.scratch_lease_bytes=lease;"]
    if work.mapping == "cooperative":
        lines.append("if(!fort_tile_cost::multiply(result.tiles,2,result.cooperative_phase_barriers))return false;")
    lines += ["output=result;return true;", "}"]
    return "\n".join(lines)
