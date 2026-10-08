// One physical copy schedule used by ordinary transfers, retained buffers, and
// cost estimates. Coordinates are half-open; no dependence-analysis coordinates.
#ifndef FORT_SECTION_COPY_HPP
#define FORT_SECTION_COPY_HPP
#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace fort_physical {
struct CopyOperation {
    std::size_t offset, width, height, depth, pitch, physical_height;
};
class CopyPlan {
    const std::vector<std::size_t> &lower_, &upper_;
    std::vector<std::size_t> strides_;
    std::size_t outer_ = 0;
    CopyOperation shape_{};
    bool product(std::size_t a, std::size_t b, std::size_t &out) {
        if (b && a > std::numeric_limits<std::size_t>::max()/b) { valid = false; return false; }
        out = a*b; return true;
    }
public:
    bool valid = true;
    std::size_t bytes = 0, copies = 0;
    CopyPlan(std::size_t element_bytes, const std::vector<std::size_t> &extents,
             const std::vector<std::size_t> &lower, const std::vector<std::size_t> &upper)
        : lower_(lower), upper_(upper) {
        const auto rank = extents.size();
        if (!rank || !element_bytes || lower.size()!=rank || upper.size()!=rank) { valid=false; return; }
        bool empty = false;
        for (std::size_t k=0; k<rank; ++k) {
            if (lower[k]>upper[k] || upper[k]>extents[k]) { valid=false; return; }
            empty |= lower[k]==upper[k];
        }
        if (empty) return; // Empty shapes do not overflow other physical axes.
        std::size_t stride = element_bytes;
        bytes = element_bytes;
        for (std::size_t k=0; k<rank; ++k) {
            strides_.push_back(stride);
            if (!product(stride,extents[k],stride) || !product(bytes,upper[k]-lower[k],bytes)) return;
        }
        if (!product(upper[0]-lower[0],element_bytes,shape_.width)) return;
        std::size_t axis=1;
        while (axis<rank && lower[axis-1]==0 && upper[axis-1]==extents[axis-1]) {
            if (!product(shape_.width,upper[axis]-lower[axis],shape_.width)) return;
            ++axis;
        }
        shape_.height = axis<rank ? upper[axis]-lower[axis] : 1;
        shape_.pitch = axis<rank ? strides_[axis] : shape_.width;
        shape_.depth = axis+1<rank ? upper[axis+1]-lower[axis+1] : 1;
        shape_.physical_height = axis<rank ? extents[axis] : 1;
        outer_ = axis+2;
        // A one-row slice is a 2D rectangle across slices, with the original
        // physical slice pitch. This also preserves the old plane fast path.
        if (shape_.height==1 && shape_.depth>1) {
            shape_.height = shape_.depth;
            shape_.pitch = strides_[axis+1];
            shape_.depth = 1;
        }
        copies=1;
        for (std::size_t k=outer_; k<rank; ++k)
            if (!product(copies,upper[k]-lower[k],copies)) return;
    }
    template<class Visitor> bool visit(Visitor visitor) const {
        if (!valid) return false;
        if (!bytes) return true;
        auto coordinate = lower_;
        for (;;) {
            auto op = shape_;
            for (std::size_t k=0; k<coordinate.size(); ++k) op.offset += coordinate[k]*strides_[k];
            if (!visitor(op)) return false;
            std::size_t k=outer_;
            while (k<coordinate.size() && ++coordinate[k]==upper_[k]) { coordinate[k]=lower_[k]; ++k; }
            if (k>=coordinate.size()) break;
        }
        return true;
    }
};
} // namespace fort_physical
#endif
