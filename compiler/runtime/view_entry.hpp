/* Borrowed numerical views of canonical full-layout allocations. */
#ifndef FORT_SCOPED_VIEW_ENTRY_HPP
#define FORT_SCOPED_VIEW_ENTRY_HPP
#include "scoped_entry.hpp"
#include "scoped_regions.hpp"
#include <vector>
#include <new>
#ifdef __CUDACC__
#define FORT_VIEW_CALLABLE __host__ __device__
#else
#define FORT_VIEW_CALLABLE
#endif
namespace fort_scoped {
template<class T, size_t Rank> struct RootView {
    T *root = nullptr;
    size_t origins[Rank]{}, extents[Rank]{}, strides[Rank]{}, base=0;
    RootView() = default;
    explicit RootView(const fort_scope_view_layout_v1 &layout, T *storage) : root(storage) {
        for (size_t k=0; k<Rank; ++k) {
            origins[k]=layout.origins[k]; extents[k]=layout.extents[k];
            strides[k]=layout.byte_strides[k]/sizeof(T);
        }
    }
    explicit RootView(const fort_scope_view_layout_v2 &layout, T *storage) : root(storage) {
        base=layout.byte_offset/sizeof(T);
        for (size_t k=0; k<Rank; ++k) {
            extents[k]=layout.extents[k];
            strides[k]=layout.root_byte_strides[layout.axes[k]]/sizeof(T);
        }
    }
    FORT_VIEW_CALLABLE T &operator[](size_t ordinal) const {
        size_t offset=base;
        for (size_t k=0; k<Rank; ++k) {
            offset+=(origins[k]+ordinal%extents[k])*strides[k]; ordinal/=extents[k];
        }
        return root[offset];
    }
};

// One begin/end per canonical root. Formal-coordinate rectangles are projected
// before merging; may-write views must be disjoint even when a particular
// runtime branch writes less. Read-only overlapping views remain exact unions.
template<size_t Capacity> class ViewAccessBatch {
    using Box = coherence::Box;
    using Region = coherence::Region;
    struct Root {
        fort_buffer_t handle=0;
        coherence::Effects effects;
        std::vector<std::pair<Box,bool>> views;
        std::vector<fort_scope_section> reads, writes, overwrites;
    };
    fort_scope_t context_;
    AccessBatch<Capacity> batch_;
    std::array<Root,Capacity> roots_{};
    size_t size_=0;
    bool sealed_=false;
    static Box full(const fort_scope_view_v1 &view) {
        Box result;
        for (size_t k=0; k<view.rank; ++k) {
            result.lo.push_back(view.origins[k]);
            result.hi.push_back(view.origins[k]+view.extents[k]);
        }
        return result;
    }
    static Region project(const fort_scope_view_v1 &view, const fort_scope_section *sections,
                          size_t count, bool all) {
        if (count>coherence::rectangle_limit || (count && !sections)) throw int(FORT_SCOPE_BOUNDARY);
        Region result;
        if (all) { coherence::append(result,full(view)); return result; }
        coherence::Budget budget;
        for (size_t i=0; i<count; ++i) {
            if (!sections[i].lower || !sections[i].upper) throw int(FORT_SCOPE_ARGUMENT);
            Box box;
            for (size_t k=0; k<view.rank; ++k) {
                const size_t lo=sections[i].lower[k], hi=sections[i].upper[k];
                if (lo>hi || hi>view.extents[k]) throw int(FORT_SCOPE_ARGUMENT);
                box.lo.push_back(view.origins[k]+lo); box.hi.push_back(view.origins[k]+hi);
            }
            result=coherence::unite(std::move(result),{box},budget);
        }
        return result;
    }
    static Box full(const fort_scope_view_v2 &view) {
        Box result;
        const bool empty=std::find(view.extents,view.extents+view.rank,size_t(0))!=view.extents+view.rank;
        if(empty) {
            result.lo.resize(view.root_rank,0); result.hi.resize(view.root_rank,0);
            return result;
        }
        for(size_t axis=0;axis<view.root_rank;++axis) {
            size_t extent=1;
            for(size_t k=0;k<view.rank;++k) if(view.axes[k]==axis) extent=view.extents[k];
            result.lo.push_back(view.origins[axis]);
            result.hi.push_back(view.origins[axis]+extent);
        }
        return result;
    }
    static Region project(const fort_scope_view_v2 &view,const fort_scope_section *sections,
                          size_t count,bool all) {
        if(count>coherence::rectangle_limit || (count && !sections)) throw int(FORT_SCOPE_BOUNDARY);
        Region result;
        if(all) { coherence::append(result,full(view)); return result; }
        coherence::Budget budget;
        for(size_t i=0;i<count;++i) {
            if(!sections[i].lower || !sections[i].upper) throw int(FORT_SCOPE_ARGUMENT);
            auto box=full(view);
            for(size_t k=0;k<view.rank;++k) {
                const size_t lo=sections[i].lower[k],hi=sections[i].upper[k],axis=view.axes[k];
                if(lo>hi || hi>view.extents[k]) throw int(FORT_SCOPE_ARGUMENT);
                box.lo[axis]=view.origins[axis]+lo; box.hi[axis]=view.origins[axis]+hi;
            }
            result=coherence::unite(std::move(result),{box},budget);
        }
        return result;
    }
    template<class View> int add_validated(const View &view,const fort_scope_access &access) {
        if(access.flags&~uint32_t(7)) return fort_scope_report_error(FORT_SCOPE_ARGUMENT,"invalid view access flags");
        try {
            auto reads=project(view,access.reads,access.read_count,access.flags&FORT_SCOPE_READ_ALL);
            auto writes=project(view,access.writes,access.write_count,access.flags&FORT_SCOPE_WRITE_ALL);
            auto overwrites=project(view,access.overwrites,access.overwrite_count,access.flags&FORT_SCOPE_OVERWRITE_ALL);
            coherence::Budget budget;
            if(!coherence::difference(overwrites,writes,budget).empty()) throw int(FORT_SCOPE_ARGUMENT);
            const bool writable=(access.flags&FORT_SCOPE_WRITE_ALL) || access.write_count;
            const auto box=full(view);
            size_t k=0; while(k<size_ && roots_[k].handle!=view.buffer) ++k;
            if(k==size_ && size_==Capacity) throw int(FORT_SCOPE_ARGUMENT);
            auto candidate=roots_[k]; // Reject without partially changing accumulated effects.
            for(const auto &old:candidate.views)
                if((writable || old.second) && coherence::intersection(old.first,box,budget)) throw int(FORT_SCOPE_ALIAS);
            candidate.handle=view.buffer;
            candidate.effects.reads=coherence::unite(std::move(candidate.effects.reads),reads,budget);
            candidate.effects.writes=coherence::unite(std::move(candidate.effects.writes),writes,budget);
            candidate.effects.overwrites=coherence::unite(std::move(candidate.effects.overwrites),overwrites,budget);
            candidate.views.push_back({box,writable}); roots_[k]=std::move(candidate);
            if(k==size_) ++size_;
            return FORT_SCOPE_OK;
        } catch(int status) { return fort_scope_report_error(status,"invalid or aliased canonical view access"); }
        catch(const coherence::Fragmented &) { return fort_scope_report_error(FORT_SCOPE_BOUNDARY,"view effect rectangle budget exceeded"); }
        catch(const std::bad_alloc &) { return fort_scope_report_error(FORT_SCOPE_RESOURCE,"view metadata allocation failed"); }
    }
    int seal() {
        if (sealed_) return FORT_SCOPE_OK;
        try {
        for (size_t k=0; k<size_; ++k) {
            auto &root=roots_[k];
            auto pack=[](Region &region,std::vector<fort_scope_section> &sections) {
                sections.clear();
                for (auto &box:region) sections.push_back({box.lo.data(),box.hi.data()});
            };
            pack(root.effects.reads,root.reads); pack(root.effects.writes,root.writes);
            pack(root.effects.overwrites,root.overwrites);
        }
        } catch (const std::bad_alloc &) {
            return fort_scope_report_error(FORT_SCOPE_RESOURCE,"view preparation metadata allocation failed");
        }
        for (size_t k=0; k<size_; ++k) {
            auto &root=roots_[k];
            fort_scope_access access{0,root.reads.size(),root.reads.data(),root.writes.size(),root.writes.data(),
                                     root.overwrites.size(),root.overwrites.data()};
            if (const int status=batch_.add(root.handle,access)) return status;
        }
        sealed_=true; return FORT_SCOPE_OK;
    }
public:
    explicit ViewAccessBatch(fort_scope_t context):context_(context),batch_(context) {}
    ViewAccessBatch(const ViewAccessBatch &)=delete;
    int add(const fort_scope_view_v1 *view,const fort_scope_access &access) {
        if (sealed_) return fort_scope_report_error(FORT_SCOPE_STATE,"view access added after preparation");
        fort_scope_view_layout_v1 layout{};
        if (const int status=fort_scope_view_get_v1(context_,view,&layout)) return status;
        return add_validated(*view,access);
    }
    int add(const fort_scope_view_v2 *view,const fort_scope_access &access) {
        if(sealed_) return fort_scope_report_error(FORT_SCOPE_STATE,"view access added after preparation");
        fort_scope_view_layout_v2 layout{};
        if(const int status=fort_scope_view_get_v2(context_,view,&layout)) return status;
        return add_validated(*view,access);
    }
    int record(uint32_t kind,uint64_t unit,double flops,double memory,int gpu,
               double cpu_numerical_seconds=0,double gpu_numerical_seconds=0) {
        if(const int status=seal()) return status;
        return batch_.record(kind,unit,flops,memory,gpu,cpu_numerical_seconds,gpu_numerical_seconds);
    }
    int record_compute(uint32_t kind,uint64_t unit,double flops,double memory,int gpu,
                       const fort_scope_compute_costs_v1 &compute) {
        if(const int status=seal()) return status;
        return batch_.record_compute(kind,unit,flops,memory,gpu,compute);
    }
    int decision(uint64_t unit,bool &gpu) { if(const int status=seal()) return status; return batch_.decision(unit,gpu); }
    int begin(bool device) { if(const int status=seal()) return status; return batch_.begin(device); }
    void cancel() noexcept { batch_.cancel(); }
    void executing() { batch_.executing(); }
    int finish() { return batch_.finish(); }
    void *device(fort_buffer_t handle) const { return batch_.device(handle); }
};
inline int forget_view(fort_scope_t context,const fort_scope_view_v1 *view,bool planning) {
    fort_scope_view_layout_v1 layout{};
    if(const int status=fort_scope_view_get_v1(context,view,&layout)) return status;
    try {
        std::vector<size_t> upper(view->rank);
        for(size_t k=0;k<view->rank;++k) upper[k]=view->origins[k]+view->extents[k];
        const fort_scope_section section{view->origins,upper.data()};
        return planning ? fort_scope_plan_forget_sections_v1(context,view->buffer,&section,1)
                        : fort_scope_forget_sections_v1(context,view->buffer,&section,1);
    } catch(const std::bad_alloc &) { return fort_scope_report_error(FORT_SCOPE_RESOURCE,"view definition metadata allocation failed"); }
}
inline int forget_view(fort_scope_t context,const fort_scope_view_v2 *view,bool planning) {
    fort_scope_view_layout_v2 layout{};
    if(const int status=fort_scope_view_get_v2(context,view,&layout)) return status;
    if(!layout.elements)
        return planning ? fort_scope_plan_forget_sections_v1(context,view->buffer,nullptr,0)
                        : fort_scope_forget_sections_v1(context,view->buffer,nullptr,0);
    try {
        std::vector<size_t> upper(view->root_rank);
        for(size_t axis=0;axis<view->root_rank;++axis) upper[axis]=view->origins[axis]+1;
        for(size_t k=0;k<view->rank;++k) upper[view->axes[k]]=view->origins[view->axes[k]]+view->extents[k];
        const fort_scope_section section{view->origins,upper.data()};
        // Empty logical views discard no definitions, including fixed axes
        // which have no address in an empty allocation.
        const size_t count=layout.elements ? 1 : 0;
        return planning ? fort_scope_plan_forget_sections_v1(context,view->buffer,&section,count)
                        : fort_scope_forget_sections_v1(context,view->buffer,&section,count);
    } catch(const std::bad_alloc &) { return fort_scope_report_error(FORT_SCOPE_RESOURCE,"view definition metadata allocation failed"); }
}
} // namespace fort_scoped
#undef FORT_VIEW_CALLABLE
#endif
