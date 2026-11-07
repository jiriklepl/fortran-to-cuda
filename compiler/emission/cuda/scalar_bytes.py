"""Transport a real object's representation without hoisting its value read.

Only generated source-task entry points with checked plain scalar storage use
this representation. It supplies neither definition nor finite-value evidence.
The kernel parameter remains trivial; no host conversion operator can read it.
"""


def support_code():
    return r'''
#include <cstring>
#include <type_traits>
template<class T> struct alignas(T) fort_scalar_bytes { unsigned char bytes[sizeof(T)]; };
static_assert(sizeof(fort_scalar_bytes<float>)==sizeof(float));
static_assert(sizeof(fort_scalar_bytes<double>)==sizeof(double));
static_assert(alignof(fort_scalar_bytes<float>)==alignof(float));
static_assert(alignof(fort_scalar_bytes<double>)==alignof(double));
static_assert(std::is_trivially_copyable<fort_scalar_bytes<float>>::value);
static_assert(std::is_trivially_copyable<fort_scalar_bytes<double>>::value);
template<class T>
static __host__ __device__ T fort_scalar_value(const fort_scalar_bytes<T>& storage) {
    T value;
    memcpy(&value,storage.bytes,sizeof(T));
    return value;
}
'''.strip().splitlines()
