#include <cmath>

// Scalar intrinsic folds evaluate each argument once without widening its type.
namespace generated_kernels::numeric {

template <typename T, typename... Rest>
CUDA_CALLABLE T minimum(T value, Rest... remaining) {
    ((value = value < remaining ? value : remaining), ...);
    return value;
}

template <typename T, typename... Rest>
CUDA_CALLABLE T maximum(T value, Rest... remaining) {
    ((value = value > remaining ? value : remaining), ...);
    return value;
}

// A finite leading constant is the unordered/tie result of the supported
// optimized GNU Fortran scalar MIN/MAX form. Keep this separate from the
// ordered runtime-argument fold: replacing every call by fmin/fmax changes
// NaN operand order and signed-zero results for ordinary variable arguments.
template <typename T>
CUDA_CALLABLE T minimum_constant_first(T constant, T value) {
    return value < constant ? value : constant;
}
template <typename T>
CUDA_CALLABLE T maximum_constant_first(T constant, T value) {
    return value > constant ? value : constant;
}

// Addressing analysis proves that negation stays representable before using this.
template <typename T>
CUDA_CALLABLE T absolute(T value) {
    return value < T{0} ? -value : value;
}

CUDA_CALLABLE inline int nint(float value) { return static_cast<int>(::roundf(value)); }
CUDA_CALLABLE inline int nint(double value) { return static_cast<int>(::round(value)); }
CUDA_CALLABLE inline int floor(float value) { return static_cast<int>(::floorf(value)); }
CUDA_CALLABLE inline int floor(double value) { return static_cast<int>(::floor(value)); }
CUDA_CALLABLE inline int ceiling(float value) { return static_cast<int>(::ceilf(value)); }
CUDA_CALLABLE inline int ceiling(double value) { return static_cast<int>(::ceil(value)); }

// The remainder is representable even for INTEGER_MIN and a divisor of -1.
CUDA_CALLABLE inline int mod(int a, int p) { return p == -1 ? 0 : a % p; }
CUDA_CALLABLE inline float mod(float a, float p) { return ::fmodf(a, p); }
CUDA_CALLABLE inline double mod(double a, double p) { return ::fmod(a, p); }

CUDA_CALLABLE inline int sign(int a, int b) {
    const long long magnitude = a < 0 ? -static_cast<long long>(a) : a;
    return static_cast<int>(b < 0 ? -magnitude : magnitude);
}
CUDA_CALLABLE inline float sign(float a, float b) { return ::copysignf(a, b); }
CUDA_CALLABLE inline double sign(double a, double b) { return ::copysign(a, b); }

template <typename T>
CUDA_CALLABLE T modulo(T a, T p) {
    const T remainder = mod(a, p);
    if (remainder == T{0}) return sign(T{0}, p);
    return (remainder < T{0}) != (p < T{0}) ? remainder + p : remainder;
}

template <typename T>
CUDA_CALLABLE T dim(T a, T b) {
    return a > b ? a - b : T{0};
}

template <typename T>
CUDA_CALLABLE T merge(T tsource, T fsource, bool mask) {
    return mask ? tsource : fsource;
}

} // namespace generated_kernels::numeric
