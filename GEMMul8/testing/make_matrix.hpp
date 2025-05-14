#pragma once
#include <curand_kernel.h>

namespace makemat {

#pragma clang optimize off
template <typename T>
__global__ void randmat_kernel(size_t m,                      // rows of A
                               size_t n,                      // columns of A
                               T *const A,                    // output
                               T phi,                         // difficulty for matrix multiplication
                               const unsigned long long seed) // seed for random numbers
{
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= m * n) return;
    curandState state;
    curand_init(seed, idx, 0, &state);
    const T rand  = static_cast<T>(curand_uniform_double(&state));
    const T randn = static_cast<T>(curand_normal_double(&state));
    A[idx]        = (rand - 0.5) * exp(randn * phi);
}
#pragma clang optimize on

template <typename T>
void randmat(size_t m,                      // rows of A
             size_t n,                      // columns of A
             T *const A,                    // output
             T phi,                         // difficulty for matrix multiplication
             const unsigned long long seed) // seed for random numbers
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    randmat_kernel<T><<<grid_size, block_size>>>(m, n, A, phi, seed);
    cudaDeviceSynchronize();
}

__global__ void ones_kernel(size_t sizeA, int8_t *const __restrict__ A) {
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= sizeA) return;
    A[idx] = 1;
}

void ones(size_t sizeA, int8_t *const A) {
    constexpr size_t block_size = 256;
    const size_t grid_size      = (sizeA + block_size - 1) / block_size;
    ones_kernel<<<grid_size, block_size>>>(sizeA, A);
    cudaDeviceSynchronize();
}

__global__ void f2d_kernel(size_t sizeA, const float *const __restrict__ in, double *const __restrict__ out) {
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= sizeA) return;
    out[idx] = static_cast<double>(in[idx]);
}

void f2d(size_t m,              // rows of A
         size_t n,              // columns of A
         const float *const in, // input
         double *const out)     // output
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    f2d_kernel<<<grid_size, block_size>>>(m * n, in, out);
    cudaDeviceSynchronize();
}

__global__ void d2f_kernel(size_t sizeA, const double *const __restrict__ in, float *const __restrict__ out) {
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= sizeA) return;
    out[idx] = static_cast<double>(in[idx]);
}

void d2f(size_t m,               // rows of A
         size_t n,               // columns of A
         const double *const in, // input
         float *const out)       // output
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    d2f_kernel<<<grid_size, block_size>>>(m * n, in, out);
    cudaDeviceSynchronize();
}

} // namespace makemat
