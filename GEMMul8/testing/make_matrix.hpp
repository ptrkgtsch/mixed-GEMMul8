#pragma once
#include "gpu_arch.hpp"

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
    gpurandState state;
    gpurand_init(seed, idx, 0, &state);
    const T rand  = static_cast<T>(gpurand_uniform_double(&state));
    const T randn = static_cast<T>(gpurand_normal_double(&state));
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
    gpuDeviceSynchronize();
}

#pragma clang optimize off
template <typename T>
__global__ void randmat_C_kernel(size_t m,                      // rows of A
                               size_t n,                        // columns of A
                               T *const A,                      // output
                               typename real_type<T>::type phi, // difficulty for matrix multiplication
                               const unsigned long long seed)   // seed for random numbers
{
    using T_real = typename real_type<T>::type;

    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= m * n) return;
    gpurandState state;
    gpurand_init(seed, idx, 0, &state);
    const T_real rand_real  = static_cast<T_real>(gpurand_uniform_double(&state));
    const T_real randn_real = static_cast<T_real>(gpurand_normal_double(&state));
    const T_real rand_imag  = static_cast<T_real>(gpurand_uniform_double(&state));
    const T_real randn_imag = static_cast<T_real>(gpurand_normal_double(&state));
    const T rand_res = makeComplex<T>((rand_real - 0.5) * exp(randn_real * phi), (rand_imag - 0.5) * exp(randn_imag * phi));
    A[idx] = rand_res;
}
#pragma clang optimize on

template <typename T>
void randmat_C(size_t m,                      // rows of A
             size_t n,                        // columns of A
             T *const A,                      // output
             typename real_type<T>::type phi, // difficulty for matrix multiplication
             const unsigned long long seed)   // seed for random numbers
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    randmat_C_kernel<T><<<grid_size, block_size>>>(m, n, A, phi, seed);
    gpuDeviceSynchronize();
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
    gpuDeviceSynchronize();
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
    gpuDeviceSynchronize();
}

__global__ void f2d_C_kernel(size_t sizeA, const gpuFloatComplex *const __restrict__ in, gpuDoubleComplex *const __restrict__ out) {
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= sizeA) return;
    out[idx] = gpuComplexFloatToDouble(in[idx]);
}

void f2d_C(size_t m,              // rows of A
         size_t n,              // columns of A
         const gpuFloatComplex *const in, // input
         gpuDoubleComplex *const out)     // output
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    f2d_C_kernel<<<grid_size, block_size>>>(m * n, in, out);
    gpuDeviceSynchronize();
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
    gpuDeviceSynchronize();
}

__global__ void d2f_C_kernel(size_t sizeA, const gpuDoubleComplex *const __restrict__ in, gpuFloatComplex *const __restrict__ out) {
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= sizeA) return;
    out[idx] = gpuComplexDoubleToFloat(in[idx]);
}

void d2f_C(size_t m,               // rows of A
         size_t n,               // columns of A
         const gpuDoubleComplex *const in, // input
         gpuFloatComplex *const out)       // output
{
    constexpr size_t block_size = 256;
    const size_t grid_size      = (m * n + block_size - 1) / block_size;
    d2f_C_kernel<<<grid_size, block_size>>>(m * n, in, out);
    gpuDeviceSynchronize();
}

} // namespace makemat
