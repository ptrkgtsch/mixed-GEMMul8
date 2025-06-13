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
#define GPU_C_32F CUDA_C_32F
#define GPU_C_64F CUDA_C_64F
#define GPUBLAS_COMPUTE_32F CUBLAS_COMPUTE_32F
#define GPUBLAS_COMPUTE_64F CUBLAS_COMPUTE_64F
#define GPUBLAS_COMPUTE_32F_FAST_TF32 CUBLAS_COMPUTE_32F_FAST_TF32

#define gpuCabs cuCabs
#define gpuCsub cuCsub
#define gpuComplexFloatToDouble cuComplexFloatToDouble
#define gpuComplexDoubleToFloat cuComplexDoubleToFloat

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
#define GPU_C_32F HIP_C_32F
#define GPU_C_64F HIP_C_64F
#define GPUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define GPUBLAS_COMPUTE_64F HIPBLAS_COMPUTE_64F
#define GPUBLAS_COMPUTE_32F_FAST_TF32 HIPBLAS_COMPUTE_32F_FAST_TF32

#define gpuCabs hipCabs
#define gpuCsub hipCsub
#define gpuComplexFloatToDouble hipComplexFloatToDouble
#define gpuComplexDoubleToFloat hipComplexDoubleToFloat

#endif // defined(__HIPCC__)

template <typename T> struct real_type {
    using type = T;
};

template <> struct real_type<gpuDoubleComplex> {
    using type = double;
};

template <> struct real_type<gpuFloatComplex> {
    using type = float;
};


template <typename T> __host__ __device__ static __inline__ typename real_type<T>::type Creal(T z) {
    return reinterpret_cast<typename real_type<T>::type>(z);
}
template <> __host__ __device__ __inline__ float Creal(gpuFloatComplex z) {
    return gpuCrealf(z);
};
template <> __host__ __device__ __inline__ double Creal(gpuDoubleComplex z) {
    return gpuCreal(z);
}

template <typename T> __host__ __device__ static __inline__ typename real_type<T>::type Cimag(T z) {
    return 0;
}
template <> __host__ __device__ __inline__ float Cimag(gpuFloatComplex z) {
    return gpuCimagf(z);
};
template <> __host__ __device__ __inline__ double Cimag(gpuDoubleComplex z) {
    return gpuCimag(z);
}

template <typename T> __host__ __device__ static __inline__ T makeComplex(typename real_type<T>::type a, typename real_type<T>::type b);
template <> __host__ __device__ __inline__ gpuFloatComplex makeComplex(float a, float b) {
    return make_gpuFloatComplex(a, b);
}
template <> __host__ __device__ __inline__ gpuDoubleComplex makeComplex(double a, double b) {
    return make_gpuDoubleComplex(a, b);
}


#endif // ndef __TESTING_HEADER_GPU_ARCH_HPP__