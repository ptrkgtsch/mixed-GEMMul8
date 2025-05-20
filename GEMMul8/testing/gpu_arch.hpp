#ifndef __TESTING_HEADER_GPU_ARCH_HPP__
#define __TESTING_HEADER_GPU_ARCH_HPP__
#include "../src/gpu_arch.hpp"

#if defined(__NVCC__)

#define gpurandState curandState
#define gpurand_init curand_init
#define gpurand_uniform_double curand_uniform_double
#define gpurand_normal_double curand_normal_double

#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpublasCreate cublasCreate
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuFree cudaFree
#define gpublasDestroy cublasDestroy

#define GPU_R_32F CUDA_R_32F
#define GPU_R_64F CUDA_R_64F
#define GPUBLAS_COMPUTE_32F CUBLAS_COMPUTE_32F
#define GPUBLAS_COMPUTE_64F CUBLAS_COMPUTE_64F
#define GPUBLAS_COMPUTE_32F_FAST_TF32 CUBLAS_COMPUTE_32F_FAST_TF32

#endif // defined(__NVCC__)

#if defined(__HIPCC__)

#define gpurandState hiprandState
#define gpurand_init hiprand_init
#define gpurand_uniform_double hiprand_uniform_double
#define gpurand_normal_double hiprand_normal_double

#define gpuDeviceProp hipDeviceProp_t
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpublasCreate hipblasCreate
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuFree hipFree
#define gpublasDestroy hipblasDestroy

#define GPU_R_32F HIP_R_32F
#define GPU_R_64F HIP_R_64F
#define GPUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define GPUBLAS_COMPUTE_64F HIPBLAS_COMPUTE_64F
#define GPUBLAS_COMPUTE_32F_FAST_TF32 HIPBLAS_COMPUTE_32F_FAST_TF32

#endif // defined(__HIPCC__)

#endif // ndef __TESTING_HEADER_GPU_ARCH_HPP__