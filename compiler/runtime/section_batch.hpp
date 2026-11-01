// Exact physical-copy batches. No definition, coherence or scheduling authority.
#ifndef FORT_SECTION_BATCH_HPP
#define FORT_SECTION_BATCH_HPP
#include "section_copy.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fort_physical { namespace compact {
constexpr std::size_t descriptor_limit=48;
constexpr std::size_t operation_limit=256;
struct Descriptor {
    std::uint64_t offset,width,height,depth,pitch,physical_height,packed_offset;
};
struct Chunk {
    Descriptor ops[descriptor_limit]{};
    std::uint64_t count=0;
};
static_assert(sizeof(Chunk)+2*sizeof(void*)<4096,"kernel parameter budget");
struct Ineligible : std::runtime_error { using std::runtime_error::runtime_error; };
struct Plan {
    std::vector<Chunk> chunks;
    std::size_t bytes=0;
    unsigned word_bytes=8;
    std::size_t rows=0, work=0, minimum_row_bytes=std::numeric_limits<std::size_t>::max(), maximum_row_bytes=0;
    std::size_t minimum_pitch=std::numeric_limits<std::size_t>::max(), maximum_pitch=0;
    std::size_t first_offset=std::numeric_limits<std::size_t>::max(), last_offset=0;
};
inline std::size_t add(std::size_t a,std::size_t b) {
    if(b>std::numeric_limits<std::size_t>::max()-a)throw Ineligible("compact offset overflow");
    return a+b;
}
inline std::size_t multiply(std::size_t a,std::size_t b) {
    if(b && a>std::numeric_limits<std::size_t>::max()/b)throw Ineligible("compact extent overflow");
    return a*b;
}
inline std::size_t row_start(const Descriptor &d,std::size_t row) {
    return add(d.offset,add(multiply(row/d.height,multiply(d.pitch,d.physical_height)),multiply(row%d.height,d.pitch)));
}
inline std::size_t end(const Descriptor &d) {
    return add(row_start(d,multiply(d.height,d.depth)-1),d.width);
}
inline bool overlap(const Descriptor &a,const Descriptor &b,std::size_t &work) {
    ++work;
    if(end(a)<=b.offset || end(b)<=a.offset)return false;
    const auto an=multiply(a.height,a.depth),bn=multiply(b.height,b.depth);
    std::size_t ai=0,bi=0;
    while(ai<an && bi<bn) {
        ++work;
        const auto x=row_start(a,ai),y=row_start(b,bi),xe=add(x,a.width),ye=add(y,b.width);
        if(x<ye && y<xe)return true;
        if(xe<=ye)++ai;else ++bi;
    }
    return false;
}
inline Plan make_plan(const std::vector<fort_physical::CopyOperation> &operations,
                      std::size_t physical_bytes,std::size_t packed_budget) {
    Plan result;std::vector<Descriptor> descriptors;
    if(operations.size()>operation_limit)throw Ineligible("compact operation budget exceeded");
    descriptors.reserve(operations.size());
    for(const auto &op:operations) {
        if(!op.width || !op.height || !op.depth || op.width>op.pitch ||
           (op.depth>1 && op.height>op.physical_height))throw Ineligible("invalid compact copy shape");
        Descriptor d{op.offset,op.width,op.height,op.depth,op.pitch,op.physical_height,result.bytes};
        if(end(d)>physical_bytes)throw Ineligible("compact copy outside allocation");
        const auto bytes=multiply(multiply(op.width,op.height),op.depth);
        result.bytes=add(result.bytes,bytes);
        if(result.bytes>packed_budget)throw Ineligible("compact one-DMA workspace budget exceeded");
        for(const auto &previous:descriptors)if(overlap(previous,d,result.work))throw Ineligible("overlapping compact destinations");
        while(result.word_bytes>1 && (d.offset%result.word_bytes || d.width%result.word_bytes ||
              d.pitch%result.word_bytes || d.packed_offset%result.word_bytes))result.word_bytes/=2;
        if(result.word_bytes==2)result.word_bytes=1;
        result.rows=add(result.rows,multiply(d.height,d.depth)); ++result.work;
        result.minimum_row_bytes=std::min(result.minimum_row_bytes,std::size_t(d.width));
        result.maximum_row_bytes=std::max(result.maximum_row_bytes,std::size_t(d.width));
        result.minimum_pitch=std::min(result.minimum_pitch,std::size_t(d.pitch));
        result.maximum_pitch=std::max(result.maximum_pitch,std::size_t(d.pitch));
        result.first_offset=std::min(result.first_offset,std::size_t(d.offset));
        result.last_offset=std::max(result.last_offset,end(d));
        descriptors.push_back(d);
    }
    for(std::size_t begin=0;begin<descriptors.size();begin+=descriptor_limit) {
        Chunk chunk{};chunk.count=std::min(descriptor_limit,descriptors.size()-begin);
        for(std::size_t i=0;i<chunk.count;++i)chunk.ops[i]=descriptors[begin+i];
        result.chunks.push_back(chunk);
    }
    return result;
}
#if defined(__CUDACC__) && !defined(FORT_SCOPE_CPU_TEST)
template<class Word,bool scatter>
__global__ void copy_chunk(unsigned char *physical,unsigned char *packed,Chunk chunk) {
    const auto op=chunk.ops[blockIdx.y];
    const std::uint64_t width=op.width/sizeof(Word),items=width*op.height*op.depth;
    for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        i<items;i+=std::uint64_t(gridDim.x)*blockDim.x) {
        const auto row=i/width;
        const auto offset=op.offset+(row/op.height)*op.pitch*op.physical_height+
                          (row%op.height)*op.pitch+(i%width)*sizeof(Word);
        auto *a=reinterpret_cast<Word*>(physical+offset);
        auto *b=reinterpret_cast<Word*>(packed+op.packed_offset+i*sizeof(Word));
        if(scatter)*a=*b;else *b=*a;
    }
}
template<class Word,bool scatter>
inline void launch(unsigned char *physical,unsigned char *packed,const Chunk &chunk,cudaStream_t stream) {
    std::size_t maximum=0;
    for(std::size_t i=0;i<chunk.count;++i) {
        const auto &op=chunk.ops[i];maximum=std::max(maximum,std::size_t(op.width/sizeof(Word)*op.height*op.depth));
    }
    if(!chunk.count || !maximum)throw Ineligible("empty compact launch");
    const auto blocks=std::min<std::size_t>(65535,(maximum+255)/256);
    copy_chunk<Word,scatter><<<dim3(unsigned(blocks),unsigned(chunk.count)),256,0,stream>>>(physical,packed,chunk);
}
#endif
}} // namespace fort_physical::compact
#endif
