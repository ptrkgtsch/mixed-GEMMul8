#pragma once
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "gpu_arch.hpp"

namespace oz2_const {

size_t grids_invscaling;
size_t grids_conv32i8u;

size_t threads_scaling;
size_t threads_conv32i8u;
size_t threads_invscaling;

} // namespace oz2_const

namespace oz2_type_utils {

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

template <typename T> __host__ __device__ static __inline__ T CMul(T a, T b);
template <> __host__ __device__ __inline__ gpuFloatComplex CMul(gpuFloatComplex a, gpuFloatComplex b) {
    return gpuCmulf(a, b);
}
template <> __host__ __device__ __inline__ gpuDoubleComplex CMul(gpuDoubleComplex a, gpuDoubleComplex b) {
    return gpuCmul(a, b);
}

template <typename T> __host__ __device__ static __inline__ T CAdd(T a, T b);
template <> __host__ __device__ __inline__ gpuFloatComplex CAdd(gpuFloatComplex a, gpuFloatComplex b) {
    return gpuCaddf(a, b);
}
template <> __host__ __device__ __inline__ gpuDoubleComplex CAdd(gpuDoubleComplex a, gpuDoubleComplex b) {
    return gpuCadd(a, b);
}

} // namespace oz2_type_utils