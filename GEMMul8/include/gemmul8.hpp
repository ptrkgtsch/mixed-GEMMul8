#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

namespace gemmul8 {

// workSize returns work size required in gemm
// Usage:
//  void *work;
//  cudaMalloc(&work, workSize(m,n,k,num_moduli));
size_t workSize(const size_t m,             // size(A,1) & size(C,1)
                const size_t n,             // size(B,2) & size(C,2)
                const size_t k,             // size(A,2) & size(B,1) <= 2^17
                const unsigned num_moduli); // #moduli, 2 <= num_moduli <= (DGEMM emulation) ? 20 : 19

// gemm returns computation time in second of each part
// Usage:
//  std::vector<double> times = gemmul8::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, fastmode, work);
//  or
//  gemmul8::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, fastmode, work);
template <typename TA, typename TB = TA, typename TC = TA>
std::vector<double> gemm(cublasHandle_t handle,        // handle
                         const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                         const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                         const size_t m,               // size(A,1) & size(C,1)
                         const size_t n,               // size(B,2) & size(C,2)
                         const size_t k,               // size(A,2) & size(B,1) <= 2^17
                         const TC *alpha,              //
                         const TA *const A,            // input
                         const size_t lda,             // leading dimension
                         const TB *const B,            // input
                         const size_t ldb,             // leading dimension
                         const TC *beta,               //
                         TC *const C,                  // output A*B
                         const size_t ldc,             // leading dimension
                         const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 20
                         const bool fastmode,          // false (accurate-mode) or true (fast-mode)
                         void *const work);            // workspace allocated in advance

template <>
std::vector<double> gemm<double>(cublasHandle_t handle,        // handle
                                 const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                                 const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                                 const size_t m,               // size(A,1) & size(C,1)
                                 const size_t n,               // size(B,2) & size(C,2)
                                 const size_t k,               // size(A,2) & size(B,1) <= 2^17
                                 const double *alpha,          //
                                 const double *const A,        // input
                                 const size_t lda,             // leading dimension
                                 const double *const B,        // input
                                 const size_t ldb,             // leading dimension
                                 const double *beta,           //
                                 double *const C,              // output A*B
                                 const size_t ldc,             // leading dimension
                                 const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 20
                                 const bool fastmode,          // false (accurate-mode) or true (fast-mode)
                                 void *const work);            // workspace allocated in advance

template <>
std::vector<double> gemm<float>(cublasHandle_t handle,        // handle
                                const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                                const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                                const size_t m,               // size(A,1) & size(C,1)
                                const size_t n,               // size(B,2) & size(C,2)
                                const size_t k,               // size(A,2) & size(B,1) <= 2^17
                                const float *alpha,           //
                                const float *const A,         // input
                                const size_t lda,             // leading dimension
                                const float *const B,         // input
                                const size_t ldb,             // leading dimension
                                const float *beta,            //
                                float *const C,               // output A*B
                                const size_t ldc,             // leading dimension
                                const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 19
                                const bool fastmode,          // false (accurate-mode) or true (fast-mode)
                                void *const work);            // workspace allocated in advance

template <>
std::vector<double> gemm<double, float, double>(cublasHandle_t handle,        // handle
                         const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                         const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                         const size_t m,               // size(A,1) & size(C,1)
                         const size_t n,               // size(B,2) & size(C,2)
                         const size_t k,               // size(A,2) & size(B,1) <= 2^17
                         const double *alpha,          //
                         const double *const A,        // input
                         const size_t lda,             // leading dimension
                         const float *const B,         // input
                         const size_t ldb,             // leading dimension
                         const double *beta,           //
                         double *const C,              // output A*B
                         const size_t ldc,             // leading dimension
                         const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 20
                         const bool fastmode,          // false (accurate-mode) or true (fast-mode)
                         void *const work);            // workspace allocated in advance

template <>
std::vector<double> gemm<float, double, double>(cublasHandle_t handle,        // handle
                         const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                         const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                         const size_t m,               // size(A,1) & size(C,1)
                         const size_t n,               // size(B,2) & size(C,2)
                         const size_t k,               // size(A,2) & size(B,1) <= 2^17
                         const double *alpha,          //
                         const float *const A,         // input
                         const size_t lda,             // leading dimension
                         const double *const B,        // input
                         const size_t ldb,             // leading dimension
                         const double *beta,           //
                         double *const C,              // output A*B
                         const size_t ldc,             // leading dimension
                         const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 20
                         const bool fastmode,          // false (accurate-mode) or true (fast-mode)
                         void *const work);            // workspace allocated in advance

template <>
std::vector<double> gemm<double, float, float>(cublasHandle_t handle,        // handle
                         const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                         const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                         const size_t m,               // size(A,1) & size(C,1)
                         const size_t n,               // size(B,2) & size(C,2)
                         const size_t k,               // size(A,2) & size(B,1) <= 2^17
                         const float *alpha,           //
                         const double *const A,        // input
                         const size_t lda,             // leading dimension
                         const float *const B,         // input
                         const size_t ldb,             // leading dimension
                         const float *beta,            //
                         float *const C,               // output A*B
                         const size_t ldc,             // leading dimension
                         const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 20
                         const bool fastmode,          // false (accurate-mode) or true (fast-mode)
                         void *const work);            // workspace allocated in

template <>
std::vector<double> gemm<float, double, float>(cublasHandle_t handle,        // handle
                         const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                         const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                         const size_t m,               // size(A,1) & size(C,1)
                         const size_t n,               // size(B,2) & size(C,2)
                         const size_t k,               // size(A,2) & size(B,1) <= 2^17
                         const float *alpha,           //
                         const float *const A,         // input
                         const size_t lda,             // leading dimension
                         const double *const B,        // input
                         const size_t ldb,             // leading dimension
                         const float *beta,            //
                         float *const C,               // output A*B
                         const size_t ldc,             // leading dimension
                         const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 20
                         const bool fastmode,          // false (accurate-mode) or true (fast-mode)
                         void *const work);            // workspace allocated in advance

} // namespace gemmul8
