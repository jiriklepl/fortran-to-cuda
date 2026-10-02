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

} // namespace generated_kernels::numeric
