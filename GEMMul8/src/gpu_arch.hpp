#ifndef __HEADER_GPU_ARCH_HPP__
#define __HEADER_GPU_ARCH_HPP__

#if defined(__NVCC__)
#define WARP_GPU_SIZE 32

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>

#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpublasHandle_t cublasHandle_t
#define gpublasOperation_t cublasOperation_t
#define gpuMemcpyToSymbol cudaMemcpyToSymbol
#define gpublasGemmEx cublasGemmEx
#define GPUBLAS_OP_N CUBLAS_OP_N
#define GPUBLAS_OP_T CUBLAS_OP_T
#define GPUBLAS_OP_C CUBLAS_OP_C
#define GPU_R_8I CUDA_R_8I
#define GPU_R_32I CUDA_R_32I
#define GPUBLAS_COMPUTE_32I CUBLAS_COMPUTE_32I
#define GPUBLAS_GEMM_DEFAULT CUBLAS_GEMM_DEFAULT

#define gpuDoubleComplex cuDoubleComplex
#define gpuFloatComplex cuFloatComplex
#define gpuCreal cuCreal
#define gpuCrealf cuCrealf
#define gpuCimag cuCimag
#define gpuCimagf cuCimagf
#define make_gpuFloatComplex make_cuFloatComplex
#define make_gpuDoubleComplex make_cuDoubleComplex
#define gpuCmul cuCmul
#define gpuCmulf cuCmulf
#define gpuCadd cuCadd
#define gpuCaddf cuCaddf
#define gpuCfma cuCfma
#define gpuCfmaf cuCfmaf

#endif // defined(__NVCC__)

#if defined(__HIPCC__)

#undef WARP_GPU_SIZE
#if defined(__GFX7__) || defined(__GFX8__) || defined(__GFX9__)
#define WARP_GPU_SIZE  64
#else
#define WARP_GPU_SIZE  32
#endif

#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <hip/hip_complex.h>

#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpublasHandle_t hipblasHandle_t
#define gpublasOperation_t hipblasOperation_t
#define gpuMemcpyToSymbol(symbol, src, count) hipMemcpyToSymbol(HIP_SYMBOL(symbol), src, count)
#define gpublasGemmEx hipblasGemmEx_v2
#define GPUBLAS_OP_N HIPBLAS_OP_N
#define GPUBLAS_OP_T HIPBLAS_OP_T
#define GPUBLAS_OP_C HIPBLAS_OP_C
#define GPU_R_8I HIP_R_8I
#define GPU_R_32I HIP_R_32I
#define GPUBLAS_COMPUTE_32I HIPBLAS_COMPUTE_32I
#define GPUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT

#define gpuDoubleComplex hipDoubleComplex
#define gpuFloatComplex hipFloatComplex
#define gpuCreal hipCreal
#define gpuCrealf hipCrealf
#define gpuCimag hipCimag
#define gpuCimagf hipCimagf
#define make_gpuFloatComplex make_hipFloatComplex
#define make_gpuDoubleComplex make_hipDoubleComplex
#define gpuCmul hipCmul
#define gpuCmulf hipCmulf
#define gpuCadd hipCadd
#define gpuCaddf hipCaddf
#define gpuCfma hipCfma
#define gpuCfmaf hipCfmaf

#endif // defined(__HIPCC__)

#endif // ndef __HEADER_GPU_ARCH_HPP__
