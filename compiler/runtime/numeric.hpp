#include <cmath>

// Scalar intrinsic folds evaluate each argument once without widening its type.
namespace generated_kernels::numeric {

// A normal power of two with a normal reciprocal has an exactly representable
// inverse. Multiplication then has the same exact real result and single
// rounding as division, including subnormal results and signed zero. Keep the
// numerical build contract (no contraction, no flush-to-zero). Exceptional
// numerators and all other divisors retain the original operation and its NaNs.
CUDA_CALLABLE inline double power_of_two_divide(double a, double b) {
#if defined(__CUDA_ARCH__)
    const unsigned long long bits = __double_as_longlong(b);
    const unsigned exponent = static_cast<unsigned>((bits >> 52) & 2047);
    if ((bits & 0x000fffffffffffffULL) == 0 && exponent > 0 && exponent < 2046 &&
            (__double_as_longlong(a) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL) {
        const double reciprocal = __longlong_as_double((bits & 0x8000000000000000ULL) |
            (static_cast<unsigned long long>(2046 - exponent) << 52));
        return a * reciprocal;
    }
#endif
    return a / b;
}

CUDA_CALLABLE inline float power_of_two_divide(float a, float b) {
#if defined(__CUDA_ARCH__)
    const unsigned bits = __float_as_uint(b), exponent = (bits >> 23) & 255;
    if ((bits & 0x007fffffU) == 0 && exponent > 0 && exponent < 254 &&
            (__float_as_uint(a) & 0x7f800000U) != 0x7f800000U) {
        const float reciprocal = __uint_as_float((bits & 0x80000000U) | ((254 - exponent) << 23));
        return a * reciprocal;
    }
#endif
    return a / b;
}

// Apply the ordinary C++ arithmetic conversion before inspecting bits. This
// preserves mixed REAL/INTEGER and REAL32/REAL division and evaluates each
// operand once. Integer division and compile-time constant divisors do not use
// this helper. The host implementation remains ordinary division.
template <typename Left, typename Right>
CUDA_CALLABLE auto divide(Left a, Right b) -> decltype(a / b) {
    using Result = decltype(a / b);
    return power_of_two_divide(static_cast<Result>(a), static_cast<Result>(b));
}

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
