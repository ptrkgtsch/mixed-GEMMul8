#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <vector>

#include "gpu_arch.hpp"

namespace eval {

//------------------------------
// dd
//------------------------------
namespace dd {

#pragma clang optimize off
inline void two_sum(const double a, const double b, double &c, double &d) {
    c        = a + b;
    double s = c - a;
    double t = b - s;
    double u = c - s;
    d        = (a - u) + t;
}
#pragma clang optimize on

#pragma clang optimize off
inline void two_sub(const double a, const double b, double &c, double &d) {
    c        = a - b;
    double s = c - a;
    double t = b + s;
    double u = c - s;
    d        = (a - u) - t;
}
#pragma clang optimize on

#pragma clang optimize off
inline void fast_two_sum(const double a, const double b, double &c, double &d) {
    c = a + b;
    d = (a - c) + b;
}
#pragma clang optimize on

#pragma clang optimize off
inline void two_prod(const double a, const double b, double &c, double &d) {
    c = a * b;
    d = std::fma(a, b, -c);
}
#pragma clang optimize on

#pragma clang optimize off
inline void add(const double a1,
                const double a2,
                const double b1,
                const double b2,
                double &c1,
                double &c2) {
    dd::two_sum(a1, b1, c1, c2);
    c2 += a2;
    c2 += b2;
    dd::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
inline void sub(const double a1,
                const double a2,
                const double b1,
                const double b2,
                double &c1,
                double &c2) {
    dd::two_sub(a1, b1, c1, c2);
    c2 += a2;
    c2 -= b2;
    dd::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
inline void mul(const double a1,
                const double a2,
                const double b1,
                const double b2,
                double &c1,
                double &c2) {
    dd::two_prod(a1, b1, c1, c2);
    c2 = std::fma(a2, b1, std::fma(a1, b2, c2));
    dd::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
inline void div(const double a1,
                const double a2,
                const double b1,
                const double b2,
                double &c1,
                double &c2) {
    double s, t;
    c1 = a1 / b1;
    dd::two_prod(c1, b1, s, t);
    double u = a1 - s;
    u -= t;
    u += a2;
    u = std::fma(-c1, b2, u);
    u /= b1;
    dd::fast_two_sum(c1, u, c1, c2);
}
#pragma clang optimize on

void simple_gemm(
    size_t m,
    size_t p,
    size_t n,
    double *A,  // m*k
    double *B,  // k*n
    double *C1, // m*n result
    double *C2) // m*n result
{
    constexpr size_t block_size = 64;
    constexpr double dzero      = 0.0;

#pragma omp parallel for
    for (size_t i = 0; i < m * p; i++) {
        C1[i] = 0.0;
        C2[i] = 0.0;
    }

#pragma omp parallel
    {
        double *C1_local = (double *)calloc(m * p, sizeof(double));
        double *C2_local = (double *)calloc(m * p, sizeof(double));

#pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < m; ii += block_size) {
            for (int jj = 0; jj < p; jj += block_size) {
                for (int kk = 0; kk < n; kk += block_size) {
                    for (int i = ii; i < ii + block_size && i < m; i++) {
                        for (int j = jj; j < jj + block_size && j < p; j++) {
                            double sum1 = 0.0;
                            double sum2 = 0.0;
                            for (int k = kk; k < kk + block_size && k < n; k++) {
                                double ab1, ab2;
                                dd::mul(A[i * n + k], dzero, B[k * p + j], dzero, ab1, ab2);
                                dd::add(ab1, ab2, sum1, sum2, sum1, sum2);
                            }
                            dd::add(C1_local[i * p + j], C2_local[i * p + j], sum1, sum2, C1_local[i * p + j], C2_local[i * p + j]);
                        }
                    }
                }
            }
        }

#pragma omp critical
        for (size_t i = 0; i < m * p; i++) {
            dd::add(C1[i], C2[i], C1_local[i], C2_local[i], C1[i], C2[i]);
        }

        free(C1_local);
        free(C2_local);
    }
}

} // namespace dd

//------------------------------
// dd on gpu
//------------------------------
namespace dd_gpu {

#pragma clang optimize off
__device__ void two_sum(const double a, const double b, double &c, double &d) {
    c        = a + b;
    double s = c - a;
    double t = b - s;
    double u = c - s;
    d        = (a - u) + t;
}
#pragma clang optimize on

#pragma clang optimize off
__device__ void two_sub(const double a, const double b, double &c, double &d) {
    c        = a - b;
    double s = c - a;
    double t = b + s;
    double u = c - s;
    d        = (a - u) - t;
}
#pragma clang optimize on

#pragma clang optimize off
__device__ void fast_two_sum(const double a, const double b, double &c, double &d) {
    c = a + b;
    d = (a - c) + b;
}
#pragma clang optimize on

#pragma clang optimize off
__device__ void two_prod(const double a, const double b, double &c, double &d) {
    c = a * b;
    d = fma(a, b, -c);
}
#pragma clang optimize on

#pragma clang optimize off
__device__ void add(const double a1,
                    const double a2,
                    const double b1,
                    const double b2,
                    double &c1,
                    double &c2) {
    dd_gpu::two_sum(a1, b1, c1, c2);
    c2 += a2;
    c2 += b2;
    dd_gpu::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
__device__ void sub(const double a1,
                    const double a2,
                    const double b1,
                    const double b2,
                    double &c1,
                    double &c2) {
    dd_gpu::two_sub(a1, b1, c1, c2);
    c2 += a2;
    c2 -= b2;
    dd_gpu::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
__device__ void mul(const double a1,
                    const double a2,
                    const double b1,
                    const double b2,
                    double &c1,
                    double &c2) {
    dd_gpu::two_prod(a1, b1, c1, c2);
    c2 = fma(a2, b1, fma(a1, b2, c2));
    dd_gpu::fast_two_sum(c1, c2, c1, c2);
}
#pragma clang optimize on

#pragma clang optimize off
__device__ void div(const double a1,
                    const double a2,
                    const double b1,
                    const double b2,
                    double &c1,
                    double &c2) {
    double s, t;
    c1 = a1 / b1;
    dd_gpu::two_prod(c1, b1, s, t);
    double u = a1 - s;
    u -= t;
    u += a2;
    u = fma(-c1, b2, u);
    u /= b1;
    dd_gpu::fast_two_sum(c1, u, c1, c2);
}
#pragma clang optimize on

__global__ void simple_gemm_device(size_t m, size_t n, size_t k, const double *A, const double *B, double *C1, double *C2) {
    __shared__ double Asub[32][33];
    __shared__ double Bsub[32][33];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int t = 0; t < (k + 32 - 1) / 32; ++t) {
        if (row < m && t * 32 + threadIdx.x < k)
            Asub[threadIdx.y][threadIdx.x] = __ldg(A + row + (t * 32 + threadIdx.x) * m);
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0;

        if (col < n && t * 32 + threadIdx.y < k)
            Bsub[threadIdx.y][threadIdx.x] = __ldg(B + col * k + (t * 32 + threadIdx.y));
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < 32; ++i) {
            double ab1, ab2;
            dd_gpu::two_prod(Asub[threadIdx.y][i], Bsub[i][threadIdx.x], ab1, ab2);
            dd_gpu::add(ab1, ab2, sum1, sum2, sum1, sum2);
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C1[col * m + row] = sum1;
        C2[col * m + row] = sum2;
    }
}

void simple_gemm(size_t m, size_t n, size_t k, const double *A, const double *B, double *C1, double *C2) {
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dd_gpu::simple_gemm_device<<<numBlocks, threadsPerBlock>>>(m, n, k, A, B, C1, C2);
}

} // namespace dd_gpu

//------------------------------
// evaluate error
//------------------------------
namespace err {

void gemm_err(const size_t m,
              const size_t n,
              double *const C,        // calculated value
              const double *const C1, // true value
              const double *const C2, // true value
              double &err1,           // max
              double &err2)           // median
{
    size_t sizeC = m * n;

#pragma omp parallel for
    for (size_t i = 0; i < sizeC; i++) {
        double tmp1, tmp2, tmp3 = 0.0, tmp4;
        dd::sub(C[i], tmp3, C1[i], C2[i], tmp1, tmp2);
        dd::div(tmp1, tmp2, C1[i], C2[i], tmp3, tmp4);
        C[i] = std::fabs(tmp3);
    }

    std::sort(C, C + sizeC);
    err1 = C[sizeC - 1];
    err2 = (sizeC & 1) ? C[sizeC / 2] : ((C[sizeC / 2] + C[sizeC / 2 - 1]) * 0.5);
}

void gemm_err(const size_t m,
              const size_t n,
              float *const C,         // calculated value
              const double *const C1, // true value
              double &err1,           // max
              double &err2)           // median
{
    size_t sizeC = m * n;

#pragma omp parallel for
    for (size_t i = 0; i < sizeC; i++) {
        double tmp = (double(C[i]) - C1[i]) / C1[i];
        C[i]       = float(std::fabs(tmp));
    }

    std::sort(C, C + sizeC);
    err1 = double(C[sizeC - 1]);
    err2 = (sizeC & 1) ? double(C[sizeC / 2]) : ((double(C[sizeC / 2]) + double(C[sizeC / 2 - 1])) * 0.5);
}

void gemm_err_C(const size_t m,
              const size_t n,
              gpuFloatComplex *const C,         // calculated value
              const gpuDoubleComplex *const C1, // true value
              double &err1,           // max
              double &err2)           // median
{
    size_t sizeC = m * n;
    std::vector<float> newC(sizeC);

#pragma omp parallel for
    for (size_t i = 0; i < sizeC; i++) {
        double tmp = gpuCabs(gpuCsub(gpuComplexFloatToDouble(C[i]), C1[i])) / gpuCabs(C1[i]);
        newC[i]    = float(std::fabs(tmp));
    }

    std::sort(newC.begin(), newC.end());
    err1 = double(newC[sizeC - 1]);
    err2 = (sizeC & 1) ? double(newC[sizeC / 2]) : ((double(newC[sizeC / 2]) + double(newC[sizeC / 2 - 1])) * 0.5);
}

} // namespace err

} // namespace eval
