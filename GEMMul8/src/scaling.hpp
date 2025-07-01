#pragma once
#include "common.hpp"
#include "table.hpp"
#include "mat_utils.hpp"

#define TILE_DIM 32
#define NY (TILE_DIM / 4)

namespace oz2_util {

namespace {

template <typename T>
struct Vec4 {
    T x, y, z, w;
};

template <typename T> __forceinline__ __device__ T Tabs(T in);
template <> __forceinline__ __device__ double Tabs<double>(double in) { return fabs(in); };
template <> __forceinline__ __device__ float Tabs<float>(float in) { return fabsf(in); };
template <> __forceinline__ __device__ int32_t Tabs<int32_t>(int32_t in) { return abs(in); };

template <typename T> __forceinline__ __device__ int __T2int_ru(T in);
template <> __forceinline__ __device__ int __T2int_ru<double>(double in) { return __double2int_ru(in); };
template <> __forceinline__ __device__ int __T2int_ru<float>(float in) { return __float2int_ru(in); };

template <typename T> __forceinline__ __device__ T Tscalbn(T in, const int sft);
template <> __forceinline__ __device__ double Tscalbn<double>(double in, const int sft) { return scalbn(in, sft); };
template <> __forceinline__ __device__ float Tscalbn<float>(float in, const int sft) { return scalbnf(in, sft); };

template <typename T> __forceinline__ __device__ T Ttrunc(T in);
template <> __forceinline__ __device__ double Ttrunc<double>(double in) { return trunc(in); };
template <> __forceinline__ __device__ float Ttrunc<float>(float in) { return truncf(in); };

template <typename T> __forceinline__ __device__ int Tilogb(T in);
template <> __forceinline__ __device__ int Tilogb<double>(double in) { return ilogb(in); };
template <> __forceinline__ __device__ int Tilogb<float>(float in) { return ilogbf(in); };

template <typename T> __forceinline__ __device__ T Tzero() { return 0; };
template <> __forceinline__ __device__ double Tzero<double>() { return 0.0; };
template <> __forceinline__ __device__ float Tzero<float>() { return 0.0F; };
template <> __forceinline__ __device__ int32_t Tzero<int32_t>() { return 0; };

template <typename T> __forceinline__ __device__ T __Tfma_ru(T in1, T in2, T in3);
template <> __forceinline__ __device__ double __Tfma_ru<double>(double in1, double in2, double in3) { return __fma_ru(in1, in2, in3); };
template <> __forceinline__ __device__ float __Tfma_ru<float>(float in1, float in2, float in3) { return __fmaf_ru(in1, in2, in3); };

template <typename T> __forceinline__ __device__ void inner_warp_max(T &amax) {
#if defined(__NVCC__)
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 16)); // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 8));  // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 4));  // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 2));  // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 1));  // warp-level reduction
#endif
#if defined(__HIPCC__)
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFUL, amax, 16)); // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFUL, amax, 8));  // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFUL, amax, 4));  // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFUL, amax, 2));  // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFUL, amax, 1));  // warp-level reduction
#endif
}

template <typename T> __forceinline__ __device__ void inner_warp_sum(T &sum);
template <> __forceinline__ __device__ void inner_warp_sum<double>(double &sum) {
#if defined(__NVCC__)
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 16)); // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 8));  // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 4));  // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 2));  // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 1));  // warp-level reduction
#endif
#if defined(__HIPCC__)
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFUL, sum, 16)); // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFUL, sum, 8));  // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFUL, sum, 4));  // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFUL, sum, 2));  // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFUL, sum, 1));  // warp-level reduction
#endif
}
template <> __forceinline__ __device__ void inner_warp_sum<float>(float &sum) {
#if defined(__NVCC__)
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 16)); // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 8));  // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 4));  // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 2));  // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 1));  // warp-level reduction
#endif
#if defined(__HIPCC__)
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFUL, sum, 16)); // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFUL, sum, 8));  // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFUL, sum, 4));  // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFUL, sum, 2));  // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFUL, sum, 1));  // warp-level reduction
#endif
}
template <> __forceinline__ __device__ void inner_warp_sum<int32_t>(int32_t &sum) {
#if defined(__NVCC__)
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, 16); // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, 8);  // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, 4);  // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, 2);  // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, 1);  // warp-level reduction
#endif
#if defined(__HIPCC__)
    sum += __shfl_down_sync(0xFFFFFFFFUL, sum, 16); // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFUL, sum, 8);  // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFUL, sum, 4);  // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFUL, sum, 2);  // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFUL, sum, 1);  // warp-level reduction
#endif
}

template <typename T>
__device__ typename oz2_type_utils::real_type<T>::type find_amax(const T *const ptr,    //
                       const unsigned length, //
                       const unsigned inc,    // leading dimension
                       typename oz2_type_utils::real_type<T>::type *shm)                //
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    // max in thread
    T_real amax = Tzero<T_real>();
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        if constexpr(std::is_same_v<T, T_real>) {
            T_real tmp = Tabs<T_real>(ptr[i * inc]);
            amax  = max(amax, tmp);
        }
        else {
            T_real tmp_real = Tabs<T_real>(oz2_type_utils::Creal(ptr[i * inc]));
            T_real tmp_imag = Tabs<T_real>(oz2_type_utils::Cimag(ptr[i * inc]));
            amax = max(amax, max(tmp_real, tmp_imag));
        }
    }

    // inner-warp reduction
    inner_warp_max<T_real>(amax);

    // inner-threadblock reduction
    if ((threadIdx.x & 0x1f) == 0) shm[threadIdx.x >> 5] = amax; // shm[warp-id] = max in warp

    __syncthreads();
    amax = Tzero<T_real>();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) amax = shm[threadIdx.x];
        inner_warp_max<T_real>(amax);
        if (threadIdx.x == 0) shm[0] = amax;
    }

    __syncthreads();
    return shm[0];
}

template <typename T>
__device__ typename oz2_type_utils::real_type<T>::type find_amax_and_nrm(const T *const ptr,    //
                               const unsigned length, //
                               const unsigned inc,    // leading dimension
                               typename oz2_type_utils::real_type<T>::type *shm,                //
                               typename oz2_type_utils::real_type<T>::type &vecnrm)             //
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    T_real *shm1 = shm;
    T_real *shm2 = shm + 32;

    // max in thread
    T_real amax = Tzero<T_real>();
    T_real sum  = Tzero<T_real>();
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        if constexpr(std::is_same_v<T, T_real>) {
            T_real tmp = Tabs<T_real>(ptr[i * inc]);
            amax  = max(amax, tmp);
            sum   = __Tfma_ru<T_real>(tmp, tmp, sum); // round-up mode
        }
        else {
            T_real tmp_real = Tabs<T_real>(oz2_type_utils::Creal(ptr[i * inc]));
            T_real tmp_imag = Tabs<T_real>(oz2_type_utils::Cimag(ptr[i * inc]));
            amax = max(amax, max(tmp_real, tmp_imag));
            sum  = __Tfma_ru<T_real>(tmp_real, tmp_real, sum);
            sum  = __Tfma_ru<T_real>(tmp_imag, tmp_imag, sum);
        }
    }

    // inner-warp reduction
    inner_warp_max<T_real>(amax);
    inner_warp_sum<T_real>(sum);

    // inner-threadblock reduction
    const auto id = (threadIdx.x & 0x1f);
    if (id == 0) {
        shm1[threadIdx.x >> 5] = amax; // shm[warp-id] = max in warp
    } else if (id == 1) {
        shm2[(threadIdx.x - 1) >> 5] = sum; // shm[warp-id] = sum in warp
    }

    __syncthreads();
    amax = Tzero<T_real>();
    sum  = Tzero<T_real>();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) amax = shm1[threadIdx.x];
        inner_warp_max<T_real>(amax);
        if (threadIdx.x == 0) shm1[0] = amax;
    } else if (threadIdx.x < 64) {
        if ((threadIdx.x - 32) < (blockDim.x >> 5)) sum = shm[threadIdx.x];
        inner_warp_sum<T_real>(sum);
        if (threadIdx.x == 32) shm2[0] = sum;
    }

    __syncthreads();
    vecnrm = shm2[0];
    return shm[0];
}

template <typename T> __device__ int8_t mod_8i(T a, unsigned j);
template <> __device__ int8_t mod_8i<double>(double a, unsigned j) {
    const auto val = oz2_table::moduli_dev[j];
    float tmp      = __double2float_rn(fma(rint(a * val.y), val.x, a));
    tmp            = __fmaf_rn(rintf(tmp * val.w), val.z, tmp);
    tmp            = __fmaf_rn(rintf(tmp * val.w), val.z, tmp);
    return static_cast<int8_t>(tmp);
}
template <> __device__ int8_t mod_8i<float>(float a, unsigned j) {
    const auto val = oz2_table::modulif_dev[j];
    float tmp      = __fmaf_rn(rintf(a * val.y), val.x, a);
    tmp            = __fmaf_rn(rintf(tmp * val.y), val.x, tmp);
    tmp            = __fmaf_rn(rintf(tmp * val.y), val.x, tmp);
    tmp            = __fmaf_rn(rintf(tmp * val.y), val.x, tmp);
    return static_cast<int8_t>(tmp);
}

template <typename T>
__global__ void stair_kernel(const size_t m, const T *const A, const size_t lda, T *const A_cpy, const size_t new_lda) {
    const auto col_idx = blockIdx.x;
    unsigned i    = threadIdx.x;
    for (; i < m; i += blockDim.x) {
        A_cpy[col_idx * new_lda + i] = A[col_idx * lda + i];
    }
}

template <typename T>
__global__ void scalingA_kernel_separated(const size_t m,               // size(A,1)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                int8_t *const __restrict__ A8i,         // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                const int16_t *const __restrict__ sftA) // exponent of shift values
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    if (k == m) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    if (x < m) {
#pragma unroll
        for (int t = 0; t < TILE_DIM; t += NY)
            if (y + t < k)
                tile[threadIdx.y + t][threadIdx.x] = A[(y + t) * lda + x];
    }

    __syncthreads();

    unsigned int tx = (threadIdx.x % 8) * 4;
    unsigned int ty = (threadIdx.x / 8) + threadIdx.y * 4;
    int nx = blockIdx_y * TILE_DIM + tx;
    int ny = blockIdx_x * TILE_DIM + ty;
    if (nx < lda8i && ny < m) {
        const int16_t sft = - sftA[ny];

        Vec4<T> in4;
        in4.x = nx     < k ? Ttrunc<T>(Tscalbn<T>(tile[tx][ty], sft))     : 0;
        in4.y = nx + 1 < k ? Ttrunc<T>(Tscalbn<T>(tile[tx + 1][ty], sft)) : 0;
        in4.z = nx + 2 < k ? Ttrunc<T>(Tscalbn<T>(tile[tx + 2][ty], sft)) : 0;
        in4.w = nx + 3 < k ? Ttrunc<T>(Tscalbn<T>(tile[tx + 3][ty], sft)) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {
            out4.x = mod_8i<T>(in4.x, j);
            out4.y = mod_8i<T>(in4.y, j);
            out4.z = mod_8i<T>(in4.z, j);
            out4.w = mod_8i<T>(in4.w, j);
            *reinterpret_cast<char4 *>(A8i + j * incA8i + ny * lda8i + nx) = out4;
        }
    }
}

template <typename T, bool addCol>
__global__ void scalingA_kernel_separated_bigmatrix(const size_t m,               // size(A,1)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                int8_t *const __restrict__ A8i,         // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                const int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    if (k == m) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    if (x < m) {
#pragma unroll
        for (int t = 0; t < TILE_DIM; t += NY)
            if (y + t < k)
                tile[threadIdx.y + t][threadIdx.x] = A[(y + t) * lda + x];
    }

    __syncthreads();

    unsigned int tx = (threadIdx.x % 8) * 4;
    unsigned int ty = (threadIdx.x / 8) + threadIdx.y * 4;
    int nx = blockIdx_y * TILE_DIM + tx;
    int ny = blockIdx_x * TILE_DIM + ty;

    const int16_t sft = - sftA[ny];

    const int kmax = (k >> 2) << 2;
    if (nx < kmax && ny < m) {
        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx][ty]), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 1][ty]), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 2][ty]), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 3][ty]), sft));

        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx][ty]), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 1][ty]), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 2][ty]), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 3][ty]), sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {
            out4.x = mod_8i<T_real>(in4_real.x, j);
            out4.y = mod_8i<T_real>(in4_real.y, j);
            out4.z = mod_8i<T_real>(in4_real.z, j);
            out4.w = mod_8i<T_real>(in4_real.w, j);
            *reinterpret_cast<char4 *>(A8i + j * incA8i + ny       * lda8i + nx    ) = out4;
            if constexpr (addCol) {
                *reinterpret_cast<char4 *>(A8i + j * incA8i + (ny + m) * lda8i + nx + k) = out4;

                out4.x = mod_8i<T_real>(in4_imag.x, j);
                out4.y = mod_8i<T_real>(in4_imag.y, j);
                out4.z = mod_8i<T_real>(in4_imag.z, j);
                out4.w = mod_8i<T_real>(in4_imag.w, j);
                *reinterpret_cast<char4 *>(A8i + j * incA8i + (ny + m) * lda8i + nx    ) = out4;
                out4.x = - out4.x;
                out4.y = - out4.y;
                out4.z = - out4.z;
                out4.w = - out4.w;
                *reinterpret_cast<char4 *>(A8i + j * incA8i + ny       * lda8i + nx + k) = out4;
            }
            else {
                out4.x = - mod_8i<T_real>(in4_imag.x, j);
                out4.y = - mod_8i<T_real>(in4_imag.y, j);
                out4.z = - mod_8i<T_real>(in4_imag.z, j);
                out4.w = - mod_8i<T_real>(in4_imag.w, j);
                *reinterpret_cast<char4 *>(A8i + j * incA8i + ny       * lda8i + nx + k) = out4;
            }
        }
    }
    else if (nx < k && ny < m) {
        T_real val_real = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx][ty]), sft));
        T_real val_imag = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx][ty]), sft));

        for (unsigned j = 0; j < num_moduli; ++j) {
            *(A8i + j * incA8i + ny       * lda8i + nx    ) = mod_8i<T_real>(val_real, j);
            if constexpr (addCol) {
                *(A8i + j * incA8i + (ny + m) * lda8i + nx + k) = mod_8i<T_real>(val_real, j);
                *(A8i + j * incA8i + (ny + m) * lda8i + nx    ) = mod_8i<T_real>(val_imag, j);
            }
            *(A8i + j * incA8i + ny       * lda8i + nx + k) = - mod_8i<T_real>(val_imag, j);
        }
    }
    else if (nx < lda8i && ny < m) {
        for (unsigned j = 0; j < num_moduli; ++j) {
            if constexpr (addCol)
                *(A8i + j * incA8i + (ny + m) * lda8i + nx + k) = 0;
            *(A8i + j * incA8i + ny       * lda8i + nx + k) = 0;
        }
    }
}

template <typename T>
__global__ void scalingA_kernel_separated_kara(const size_t m,               // size(A,1)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                int8_t *const __restrict__ A8i_real,    // output (lda8i * m)
                                int8_t *const __restrict__ A8i_imag,    // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                const int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    if (k == m) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    if (x < m) {
#pragma unroll
        for (int t = 0; t < TILE_DIM; t += NY)
            if (y + t < k)
                tile[threadIdx.y + t][threadIdx.x] = A[(y + t) * lda + x];
    }

    __syncthreads();

    unsigned int tx = (threadIdx.x % 8) * 4;
    unsigned int ty = (threadIdx.x / 8) + threadIdx.y * 4;
    int nx = blockIdx_y * TILE_DIM + tx;
    int ny = blockIdx_x * TILE_DIM + ty;

    const int16_t sft = - sftA[ny];

    const int kmax = (k >> 2) << 2;
    if (nx < kmax && ny < m) {
        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx][ty]), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 1][ty]), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 2][ty]), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 3][ty]), sft));

        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx][ty]), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 1][ty]), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 2][ty]), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 3][ty]), sft));

        char4 out4_real;
        char4 out4_imag;
        for (unsigned j = 0; j < num_moduli; ++j) {
            out4_real.x = mod_8i<T_real>(in4_real.x, j);
            out4_real.y = mod_8i<T_real>(in4_real.y, j);
            out4_real.z = mod_8i<T_real>(in4_real.z, j);
            out4_real.w = mod_8i<T_real>(in4_real.w, j);
            *reinterpret_cast<char4 *>(A8i_real + j * incA8i + ny * lda8i + nx) = out4_real;

            out4_imag.x = mod_8i<T_real>(in4_imag.x, j);
            out4_imag.y = mod_8i<T_real>(in4_imag.y, j);
            out4_imag.z = mod_8i<T_real>(in4_imag.z, j);
            out4_imag.w = mod_8i<T_real>(in4_imag.w, j);
            *reinterpret_cast<char4 *>(A8i_imag + j * incA8i + ny * lda8i + nx) = out4_imag;
        }
    }
    else if (nx < k && ny < m) {
        T_real val_real = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx][ty]), sft));
        T_real val_imag = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx][ty]), sft));

        for (unsigned j = 0; j < num_moduli; ++j) {
            *(A8i_real + j * incA8i + ny * lda8i + nx) = mod_8i<T_real>(val_real, j);
            *(A8i_imag + j * incA8i + ny * lda8i + nx) = mod_8i<T_real>(val_imag, j);
        }
    }
    else if (nx < lda8i && ny < m) {
        for (unsigned j = 0; j < num_moduli; ++j) {
            *(A8i_real + j * incA8i + ny * lda8i + nx) = 0;
            *(A8i_real + j * incA8i + ny * lda8i + nx) = 0;
        }
    }
}

template <typename T, bool addCol>
__global__ void scalingA_kernel_separated_bigmatrix_minusTR(const size_t m,               // size(A,1)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                int8_t *const __restrict__ A8i,         // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                const int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    if (k == m) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    if (x < m) {
#pragma unroll
        for (int t = 0; t < TILE_DIM; t += NY)
            if (y + t < k)
                tile[threadIdx.y + t][threadIdx.x] = A[(y + t) * lda + x];
    }

    __syncthreads();

    unsigned int tx = (threadIdx.x % 8) * 4;
    unsigned int ty = (threadIdx.x / 8) + threadIdx.y * 4;
    int nx = blockIdx_y * TILE_DIM + tx;
    int ny = blockIdx_x * TILE_DIM + ty;

    const int16_t sft = - sftA[ny];

    const int kmax = (k >> 2) << 2;
    if (nx < kmax && ny < m) {
        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx][ty]), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 1][ty]), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 2][ty]), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 3][ty]), sft));

        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx][ty]), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 1][ty]), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 2][ty]), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 3][ty]), sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {
            out4.x = mod_8i<T_real>(in4_real.x, j);
            out4.y = mod_8i<T_real>(in4_real.y, j);
            out4.z = mod_8i<T_real>(in4_real.z, j);
            out4.w = mod_8i<T_real>(in4_real.w, j);
            *reinterpret_cast<char4 *>(A8i + j * incA8i + ny       * lda8i + nx    ) = out4;
            if constexpr (addCol)
                *reinterpret_cast<char4 *>(A8i + j * incA8i + (ny + m) * lda8i + nx + k) = out4;

            out4.x = mod_8i<T_real>(in4_imag.x, j);
            out4.y = mod_8i<T_real>(in4_imag.y, j);
            out4.z = mod_8i<T_real>(in4_imag.z, j);
            out4.w = mod_8i<T_real>(in4_imag.w, j);
            *reinterpret_cast<char4 *>(A8i + j * incA8i + ny       * lda8i + nx + k) = out4;
            if constexpr (addCol) {
                out4.x = - out4.x;
                out4.y = - out4.y;
                out4.z = - out4.z;
                out4.w = - out4.w;
                *reinterpret_cast<char4 *>(A8i + j * incA8i + (ny + m) * lda8i + nx    ) = out4;
            }
        }
    }
    else if (nx < k && ny < m) {
        T_real val_real = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx][ty]), sft));
        T_real val_imag = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx][ty]), sft));

        for (unsigned j = 0; j < num_moduli; ++j) {
            *(A8i + j * incA8i + ny       * lda8i + nx    ) = mod_8i<T_real>(val_real, j);
            if constexpr (addCol) {
                *(A8i + j * incA8i + (ny + m) * lda8i + nx + k) = mod_8i<T_real>(val_real, j);
                *(A8i + j * incA8i + (ny + m) * lda8i + nx    ) = - mod_8i<T_real>(val_imag, j);
            }
            *(A8i + j * incA8i + ny       * lda8i + nx + k) = mod_8i<T_real>(val_imag, j);
        }
    }
    else if (nx < lda8i && ny < m) {
        for (unsigned j = 0; j < num_moduli; ++j) {
            if constexpr (addCol)
                *(A8i + j * incA8i + (ny + m) * lda8i + nx + k) = 0;
            *(A8i + j * incA8i + ny       * lda8i + nx + k) = 0;
        }
    }
}

template <typename T>
__global__ void scalingA_kernel_separated_kara_conj(const size_t m,               // size(A,1)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                int8_t *const __restrict__ A8i_real,    // output (lda8i * m)
                                int8_t *const __restrict__ A8i_imag,    // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                const int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    if (k == m) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    if (x < m) {
#pragma unroll
        for (int t = 0; t < TILE_DIM; t += NY)
            if (y + t < k)
                tile[threadIdx.y + t][threadIdx.x] = A[(y + t) * lda + x];
    }

    __syncthreads();

    unsigned int tx = (threadIdx.x % 8) * 4;
    unsigned int ty = (threadIdx.x / 8) + threadIdx.y * 4;
    int nx = blockIdx_y * TILE_DIM + tx;
    int ny = blockIdx_x * TILE_DIM + ty;

    const int16_t sft = - sftA[ny];

    const int kmax = (k >> 2) << 2;
    if (nx < kmax && ny < m) {
        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx][ty]), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 1][ty]), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 2][ty]), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx + 3][ty]), sft));

        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx][ty]), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 1][ty]), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 2][ty]), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx + 3][ty]), sft));

        char4 out4_real;
        char4 out4_imag;
        for (unsigned j = 0; j < num_moduli; ++j) {
            out4_real.x = mod_8i<T_real>(in4_real.x, j);
            out4_real.y = mod_8i<T_real>(in4_real.y, j);
            out4_real.z = mod_8i<T_real>(in4_real.z, j);
            out4_real.w = mod_8i<T_real>(in4_real.w, j);
            *reinterpret_cast<char4 *>(A8i_real + j * incA8i + ny * lda8i + nx) = out4_real;

            out4_imag.x = - mod_8i<T_real>(in4_imag.x, j);
            out4_imag.y = - mod_8i<T_real>(in4_imag.y, j);
            out4_imag.z = - mod_8i<T_real>(in4_imag.z, j);
            out4_imag.w = - mod_8i<T_real>(in4_imag.w, j);
            *reinterpret_cast<char4 *>(A8i_imag + j * incA8i + ny * lda8i + nx) = out4_imag;
        }
    }
    else if (nx < k && ny < m) {
        T_real val_real = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(tile[tx][ty]), sft));
        T_real val_imag = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(tile[tx][ty]), sft));

        for (unsigned j = 0; j < num_moduli; ++j) {
            *(A8i_real + j * incA8i + ny * lda8i + nx) = mod_8i<T_real>(val_real, j);
            *(A8i_imag + j * incA8i + ny * lda8i + nx) = - mod_8i<T_real>(val_imag, j);
        }
    }
    else if (nx < lda8i && ny < m) {
        for (unsigned j = 0; j < num_moduli; ++j) {
            *(A8i_real + j * incA8i + ny * lda8i + nx) = 0;
            *(A8i_real + j * incA8i + ny * lda8i + nx) = 0;
        }
    }
}

template <typename T>
__forceinline__ __device__ void scalingA(const size_t row_idx,
                                    const size_t k,                 // size(A,2)
                                    const size_t incA8i,            // lda8i * m
                                    const unsigned num_moduli,      // #moduli
                                    const T *const __restrict__ A,  // input (lda * m)
                                    const size_t lda,               // leading dimension
                                    int8_t *const __restrict__ A8i, // output (lda8i * m)
                                    const size_t lda8i,             // leading dimension
                                    const int16_t sft)              // exponent of shift value
{

    const T *const __restrict__ in = A + row_idx;
    int8_t *const __restrict__ out = A8i + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = Ttrunc<T>(Tscalbn<T>(in[idx * lda], sft));
        in4.y = Ttrunc<T>(Tscalbn<T>(in[(idx + 1) * lda], sft));
        in4.z = Ttrunc<T>(Tscalbn<T>(in[(idx + 2) * lda], sft));
        in4.w = Ttrunc<T>(Tscalbn<T>(in[(idx + 3) * lda], sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T>(in4.x, j);
            out4.y = mod_8i<T>(in4.y, j);
            out4.z = mod_8i<T>(in4.z, j);
            out4.w = mod_8i<T>(in4.w, j);

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? Ttrunc<T>(Tscalbn<T>(in[idx * lda], sft)) : 0;
        in4.y = (idx + 1 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 1) * lda], sft)) : 0;
        in4.z = (idx + 2 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 2) * lda], sft)) : 0;
        in4.w = (idx + 3 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 3) * lda], sft)) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = (idx < k) ? mod_8i<T>(in4.x, j) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T>(in4.y, j) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T>(in4.z, j) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T>(in4.w, j) : 0;

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }
}

template <typename T, bool addCol>
__forceinline__ __device__ void scalingA_bigmatrix(const size_t row_idx,
                                    const size_t m,                 // size(A,1)
                                    const size_t k,                 // size(A,2)
                                    const size_t incA8i,            // lda8i * m
                                    const unsigned num_moduli,      // #moduli
                                    const T *const __restrict__ A,  // input (lda * m)
                                    const size_t lda,               // leading dimension
                                    int8_t *const __restrict__ A8i, // output (lda8i * m)
                                    const size_t lda8i,             // leading dimension
                                    const int16_t sft)              // exponent of shift value
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    const T *const __restrict__ in = A + row_idx;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[idx * lda]), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda]), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda]), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda]), sft));
        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[idx * lda]), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda]), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda]), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda]), sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T_real>(in4_real.x, j);
            out4.y = mod_8i<T_real>(in4_real.y, j);
            out4.z = mod_8i<T_real>(in4_real.z, j);
            out4.w = mod_8i<T_real>(in4_real.w, j);

            *reinterpret_cast<char4 *>(A8i + row_idx       * lda8i + j * incA8i + idx    ) = out4;
            if constexpr (addCol) {
                *reinterpret_cast<char4 *>(A8i + (row_idx + m) * lda8i + j * incA8i + idx + k) = out4;

                out4.x = mod_8i<T_real>(in4_imag.x, j);
                out4.y = mod_8i<T_real>(in4_imag.y, j);
                out4.z = mod_8i<T_real>(in4_imag.z, j);
                out4.w = mod_8i<T_real>(in4_imag.w, j);

                *reinterpret_cast<char4 *>(A8i + (row_idx + m) * lda8i + j * incA8i + idx    ) = out4;
                out4.x = - out4.x;
                out4.y = - out4.y;
                out4.z = - out4.z;
                out4.w = - out4.w;
                *reinterpret_cast<char4 *>(A8i + row_idx       * lda8i + j * incA8i + idx + k) = out4;
            }
            else {
                out4.x = - mod_8i<T_real>(in4_imag.x, j);
                out4.y = - mod_8i<T_real>(in4_imag.y, j);
                out4.z = - mod_8i<T_real>(in4_imag.z, j);
                out4.w = - mod_8i<T_real>(in4_imag.w, j);
                *reinterpret_cast<char4 *>(A8i + row_idx       * lda8i + j * incA8i + idx + k) = out4;
            }
        }
    }
    i = i << 2;  // fill rest of the matrices
    for (; i < k; i += blockDim.x) {
        T_real val_real = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[i * lda]), sft));
        T_real val_imag = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[i * lda]), sft));
        for (unsigned j = 0; j < num_moduli; ++j) {
            *(A8i + row_idx       * lda8i + j * incA8i + i    ) = mod_8i<T_real>(val_real, j);
            if constexpr (addCol) {
                *(A8i + (row_idx + m) * lda8i + j * incA8i + i + k) = mod_8i<T_real>(val_real, j);
                *(A8i + (row_idx + m) * lda8i + j * incA8i + i    ) = mod_8i<T_real>(val_imag, j);
            }
            *(A8i + row_idx       * lda8i + j * incA8i + i + k) = - mod_8i<T_real>(val_imag, j);
        }
    }
    for (; i + k < lda8i; i += blockDim.x) {
        for (unsigned j = 0; j < num_moduli; ++j) {
            if constexpr (addCol)
                *(A8i + (row_idx + m) * lda8i + j * incA8i + i + k) = 0;
            *(A8i + row_idx       * lda8i + j * incA8i + i + k) = 0;
        }
    }
}

template <typename T>
__forceinline__ __device__ void scalingA_kara(const size_t row_idx,
                                    const size_t k,                 // size(A,2)
                                    const size_t incA8i,            // lda8i * m
                                    const unsigned num_moduli,      // #moduli
                                    const T *const __restrict__ A,  // input (lda * m)
                                    const size_t lda,               // leading dimension
                                    int8_t *const __restrict__ A8i_real, // output (lda8i * m)
                                    int8_t *const __restrict__ A8i_imag, // output (lda8i * m)
                                    const size_t lda8i,             // leading dimension
                                    const int16_t sft)              // exponent of shift value
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    const T *const __restrict__ in = A + row_idx;
    int8_t *const __restrict__ out_real = A8i_real + row_idx * lda8i;
    int8_t *const __restrict__ out_imag = A8i_imag + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[idx * lda]), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda]), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda]), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda]), sft));
        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[idx * lda]), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda]), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda]), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda]), sft));

        char4 out4_real;
        char4 out4_imag;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4_real.x = mod_8i<T_real>(in4_real.x, j);
            out4_real.y = mod_8i<T_real>(in4_real.y, j);
            out4_real.z = mod_8i<T_real>(in4_real.z, j);
            out4_real.w = mod_8i<T_real>(in4_real.w, j);
            *reinterpret_cast<char4 *>(out_real + j * incA8i + idx) = out4_real;

            out4_imag.x = mod_8i<T_real>(in4_imag.x, j);
            out4_imag.y = mod_8i<T_real>(in4_imag.y, j);
            out4_imag.z = mod_8i<T_real>(in4_imag.z, j);
            out4_imag.w = mod_8i<T_real>(in4_imag.w, j);
            *reinterpret_cast<char4 *>(out_imag + j * incA8i + idx) = out4_imag;
        }
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = (idx < k)     ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[idx * lda]), sft))       : 0;
        in4_real.y = (idx + 1 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda]), sft)) : 0;
        in4_real.z = (idx + 2 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda]), sft)) : 0;
        in4_real.w = (idx + 3 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda]), sft)) : 0;
        Vec4<T_real> in4_imag;
        in4_imag.x = (idx < k)     ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[idx * lda]), sft))       : 0;
        in4_imag.y = (idx + 1 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda]), sft)) : 0;
        in4_imag.z = (idx + 2 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda]), sft)) : 0;
        in4_imag.w = (idx + 3 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda]), sft)) : 0;

        char4 out4_real;
        char4 out4_imag;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4_real.x = (idx < k)     ? mod_8i<T_real>(in4_real.x, j) : 0;
            out4_real.y = (idx + 1 < k) ? mod_8i<T_real>(in4_real.y, j) : 0;
            out4_real.z = (idx + 2 < k) ? mod_8i<T_real>(in4_real.z, j) : 0;
            out4_real.w = (idx + 3 < k) ? mod_8i<T_real>(in4_real.w, j) : 0;
            *reinterpret_cast<char4 *>(out_real + j * incA8i + idx) = out4_real;

            out4_imag.x = (idx < k)     ? mod_8i<T_real>(in4_imag.x, j) : 0;
            out4_imag.y = (idx + 1 < k) ? mod_8i<T_real>(in4_imag.y, j) : 0;
            out4_imag.z = (idx + 2 < k) ? mod_8i<T_real>(in4_imag.z, j) : 0;
            out4_imag.w = (idx + 3 < k) ? mod_8i<T_real>(in4_imag.w, j) : 0;
            *reinterpret_cast<char4 *>(out_imag + j * incA8i + idx) = out4_imag;
        }
    }
}

template <typename T, bool addCol>
__forceinline__ __device__ void scalingA_bigmatrix_minusTR(const size_t row_idx,
                                const size_t m,                 // size(A,1)
                                const size_t k,                 // size(A,2)
                                const size_t incA8i,            // lda8i * m
                                const unsigned num_moduli,      // #moduli
                                const T *const __restrict__ A,  // input (lda * n)
                                const size_t lda,               // leading dimension
                                int8_t *const __restrict__ A8i, // output (lda8i * m)
                                const size_t lda8i,             // leading dimension
                                const int16_t sft)              // exponent of shift value
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    const T *const __restrict__ in = A + row_idx;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[idx * lda]), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda]), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda]), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda]), sft));
        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[idx * lda]), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda]), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda]), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda]), sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T_real>(in4_real.x, j);
            out4.y = mod_8i<T_real>(in4_real.y, j);
            out4.z = mod_8i<T_real>(in4_real.z, j);
            out4.w = mod_8i<T_real>(in4_real.w, j);

            *reinterpret_cast<char4 *>(A8i + row_idx       * lda8i + j * incA8i + idx    ) = out4;
            if constexpr (addCol)
                *reinterpret_cast<char4 *>(A8i + (row_idx + m) * lda8i + j * incA8i + idx + k) = out4;

            out4.x = mod_8i<T_real>(in4_imag.x, j);
            out4.y = mod_8i<T_real>(in4_imag.y, j);
            out4.z = mod_8i<T_real>(in4_imag.z, j);
            out4.w = mod_8i<T_real>(in4_imag.w, j);

            *reinterpret_cast<char4 *>(A8i + row_idx       * lda8i + j * incA8i + idx + k) = out4;
            if constexpr (addCol) {
                out4.x = - out4.x;
                out4.y = - out4.y;
                out4.z = - out4.z;
                out4.w = - out4.w;
                *reinterpret_cast<char4 *>(A8i + (row_idx + m) * lda8i + j * incA8i + idx    ) = out4;
            }
        }
    }
    i = i << 2;  // fill rest of the matrices
    for (; i < k; i += blockDim.x) {
        T_real val_real = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[i * lda]), sft));
        T_real val_imag = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[i * lda]), sft));
        for (unsigned j = 0; j < num_moduli; ++j) {
            *(A8i + row_idx       * lda8i + j * incA8i + i    ) = mod_8i<T_real>(val_real, j);
            if constexpr (addCol) {
                *(A8i + (row_idx + m) * lda8i + j * incA8i + i + k) = mod_8i<T_real>(val_real, j);
                *(A8i + (row_idx + m) * lda8i + j * incA8i + i    ) = - mod_8i<T_real>(val_imag, j);
            }
            *(A8i + row_idx       * lda8i + j * incA8i + i + k) = mod_8i<T_real>(val_imag, j);
        }
    }
    for (; i + k < lda8i; i += blockDim.x) {
        for (unsigned j = 0; j < num_moduli; ++j) {
            if constexpr (addCol)
                *(A8i + (row_idx + m) * lda8i + j * incA8i + i + k) = 0;
            *(A8i + row_idx       * lda8i + j * incA8i + i + k) = 0;
        }
    }
}

template <typename T>
__forceinline__ __device__ void scalingA_kara_conj(const size_t row_idx,
                                    const size_t k,                 // size(A,2)
                                    const size_t incA8i,            // lda8i * m
                                    const unsigned num_moduli,      // #moduli
                                    const T *const __restrict__ A,  // input (lda * m)
                                    const size_t lda,               // leading dimension
                                    int8_t *const __restrict__ A8i_real, // output (lda8i * m)
                                    int8_t *const __restrict__ A8i_imag, // output (lda8i * m)
                                    const size_t lda8i,             // leading dimension
                                    const int16_t sft)              // exponent of shift value
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    const T *const __restrict__ in = A + row_idx;
    int8_t *const __restrict__ out_real = A8i_real + row_idx * lda8i;
    int8_t *const __restrict__ out_imag = A8i_imag + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[idx * lda]), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda]), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda]), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda]), sft));
        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[idx * lda]), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda]), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda]), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda]), sft));

        char4 out4_real;
        char4 out4_imag;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4_real.x = mod_8i<T_real>(in4_real.x, j);
            out4_real.y = mod_8i<T_real>(in4_real.y, j);
            out4_real.z = mod_8i<T_real>(in4_real.z, j);
            out4_real.w = mod_8i<T_real>(in4_real.w, j);
            *reinterpret_cast<char4 *>(out_real + j * incA8i + idx) = out4_real;

            out4_imag.x = - mod_8i<T_real>(in4_imag.x, j);
            out4_imag.y = - mod_8i<T_real>(in4_imag.y, j);
            out4_imag.z = - mod_8i<T_real>(in4_imag.z, j);
            out4_imag.w = - mod_8i<T_real>(in4_imag.w, j);
            *reinterpret_cast<char4 *>(out_imag + j * incA8i + idx) = out4_imag;
        }
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = (idx < k)     ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[idx * lda]), sft))       : 0;
        in4_real.y = (idx + 1 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda]), sft)) : 0;
        in4_real.z = (idx + 2 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda]), sft)) : 0;
        in4_real.w = (idx + 3 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda]), sft)) : 0;
        Vec4<T_real> in4_imag;
        in4_imag.x = (idx < k)     ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[idx * lda]), sft))       : 0;
        in4_imag.y = (idx + 1 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda]), sft)) : 0;
        in4_imag.z = (idx + 2 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda]), sft)) : 0;
        in4_imag.w = (idx + 3 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda]), sft)) : 0;

        char4 out4_real;
        char4 out4_imag;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4_real.x = (idx < k)     ? mod_8i<T_real>(in4_real.x, j) : 0;
            out4_real.y = (idx + 1 < k) ? mod_8i<T_real>(in4_real.y, j) : 0;
            out4_real.z = (idx + 2 < k) ? mod_8i<T_real>(in4_real.z, j) : 0;
            out4_real.w = (idx + 3 < k) ? mod_8i<T_real>(in4_real.w, j) : 0;
            *reinterpret_cast<char4 *>(out_real + j * incA8i + idx) = out4_real;

            out4_imag.x = (idx < k)     ? - mod_8i<T_real>(in4_imag.x, j) : 0;
            out4_imag.y = (idx + 1 < k) ? - mod_8i<T_real>(in4_imag.y, j) : 0;
            out4_imag.z = (idx + 2 < k) ? - mod_8i<T_real>(in4_imag.z, j) : 0;
            out4_imag.w = (idx + 3 < k) ? - mod_8i<T_real>(in4_imag.w, j) : 0;
            *reinterpret_cast<char4 *>(out_imag + j * incA8i + idx) = out4_imag;
        }
    }
}

template <typename T>
__forceinline__ __device__ void scalingB(const size_t col_idx,
                                const size_t k,                 // size(B,1)
                                const size_t incB8i,            // ldb8i * n
                                const unsigned num_moduli,      // #moduli
                                const T *const __restrict__ B,  // input (ldb * n)
                                const size_t ldb,               // leading dimension
                                int8_t *const __restrict__ B8i, // output (ldb8i * n)
                                const size_t ldb8i,             // leading dimension
                                const int16_t sft)              // exponent of shift values
{
    const T *const __restrict__ in = B + col_idx * ldb;
    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4 = *reinterpret_cast<const Vec4<T> *>(in + idx);
        in4.x       = Ttrunc<T>(Tscalbn<T>(in4.x, sft));
        in4.y       = Ttrunc<T>(Tscalbn<T>(in4.y, sft));
        in4.z       = Ttrunc<T>(Tscalbn<T>(in4.z, sft));
        in4.w       = Ttrunc<T>(Tscalbn<T>(in4.w, sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T>(in4.x, j);
            out4.y = mod_8i<T>(in4.y, j);
            out4.z = mod_8i<T>(in4.z, j);
            out4.w = mod_8i<T>(in4.w, j);

            *reinterpret_cast<char4 *>(out + j * incB8i + idx) = out4;
        }
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? Ttrunc<T>(Tscalbn<T>(in[idx], sft)) : 0;
        in4.y = (idx + 1 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 1], sft)) : 0;
        in4.z = (idx + 2 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 2], sft)) : 0;
        in4.w = (idx + 3 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 3], sft)) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = (idx < k) ? mod_8i<T>(in4.x, j) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T>(in4.y, j) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T>(in4.z, j) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T>(in4.w, j) : 0;

            *reinterpret_cast<char4 *>(out + j * incB8i + idx) = out4;
        }
    }
}

template <typename T, bool addCol>
__forceinline__ __device__ void scalingB_bigmatrix(const size_t col_idx,
                                const size_t k,                 // size(B,1)
                                const size_t n,                 // size(B,2)
                                const size_t incB8i,            // ldb8i * n
                                const unsigned num_moduli,      // #moduli
                                const T *const __restrict__ B,  // input (ldb * n)
                                const size_t ldb,               // leading dimension
                                int8_t *const __restrict__ B8i, // output (ldb8i * n)
                                const size_t ldb8i,             // leading dimension
                                const int16_t sft)              // exponent of shift value
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    const T *const __restrict__ in = B + col_idx * ldb;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx)), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 1)), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 2)), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 3)), sft));

        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx)), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 1)), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 2)), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 3)), sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T_real>(in4_real.x, j);
            out4.y = mod_8i<T_real>(in4_real.y, j);
            out4.z = mod_8i<T_real>(in4_real.z, j);
            out4.w = mod_8i<T_real>(in4_real.w, j);

            *reinterpret_cast<char4 *>(B8i + col_idx       * ldb8i + j * incB8i + idx    ) = out4;
            if constexpr (addCol)
                *reinterpret_cast<char4 *>(B8i + (col_idx + n) * ldb8i + j * incB8i + idx + k) = out4;

            out4.x = mod_8i<T_real>(in4_imag.x, j);
            out4.y = mod_8i<T_real>(in4_imag.y, j);
            out4.z = mod_8i<T_real>(in4_imag.z, j);
            out4.w = mod_8i<T_real>(in4_imag.w, j);
            *reinterpret_cast<char4 *>(B8i + col_idx       * ldb8i + j * incB8i + idx + k) = out4;
            if constexpr (addCol) {
                out4.x = - out4.x;
                out4.y = - out4.y;
                out4.z = - out4.z;
                out4.w = - out4.w;
                *reinterpret_cast<char4 *>(B8i + (col_idx + n) * ldb8i + j * incB8i + idx    ) = out4;
            }
        }
    }
    i = i << 2;
    for (; i < k; i += blockDim.x) {
        T_real val_real = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[i]), sft));
        T_real val_imag = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[i]), sft));

        for (unsigned j = 0; j < num_moduli; ++j) {
            *(B8i + col_idx       * ldb8i + j * incB8i + i    ) = mod_8i<T_real>(val_real, j);
            if constexpr (addCol)
                *(B8i + (col_idx + n) * ldb8i + j * incB8i + i + k) = mod_8i<T_real>(val_real, j);
            *(B8i + col_idx       * ldb8i + j * incB8i + i + k) = mod_8i<T_real>(val_imag, j);
            if constexpr (addCol)
                *(B8i + (col_idx + n) * ldb8i + j * incB8i + i    ) = - mod_8i<T_real>(val_imag, j);
        }
    }
    for (; i + k < ldb8i; i += blockDim.x) {
        for (unsigned j = 0; j < num_moduli; ++j) {
            if constexpr (addCol)
                *(B8i + (col_idx + n) * ldb8i + j * incB8i + i + k) = 0;
            *(B8i + col_idx       * ldb8i + j * incB8i + i + k) = 0;
        }
    }
}

template <typename T>
__forceinline__ __device__ void scalingB_kara(const size_t col_idx,
                                const size_t k,                 // size(B,1)
                                const size_t incB8i,            // ldb8i * n
                                const unsigned num_moduli,      // #moduli
                                const T *const __restrict__ B,  // input (ldb * n)
                                const size_t ldb,               // leading dimension
                                int8_t *const __restrict__ B8i_real, // output (ldb8i * n)
                                int8_t *const __restrict__ B8i_imag, // output (ldb8i * n)
                                const size_t ldb8i,             // leading dimension
                                const int16_t sft)              // exponent of shift value
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    const T *const __restrict__ in = B + col_idx * ldb;
    int8_t *const __restrict__ out_real = B8i_real + col_idx * ldb8i;
    int8_t *const __restrict__ out_imag = B8i_imag + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx)), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 1)), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 2)), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 3)), sft));

        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx)), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 1)), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 2)), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 3)), sft));

        char4 out4_real;
        char4 out4_imag;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4_real.x = mod_8i<T_real>(in4_real.x, j);
            out4_real.y = mod_8i<T_real>(in4_real.y, j);
            out4_real.z = mod_8i<T_real>(in4_real.z, j);
            out4_real.w = mod_8i<T_real>(in4_real.w, j);

            *reinterpret_cast<char4 *>(out_real + j * incB8i + idx) = out4_real;

            out4_imag.x = mod_8i<T_real>(in4_imag.x, j);
            out4_imag.y = mod_8i<T_real>(in4_imag.y, j);
            out4_imag.z = mod_8i<T_real>(in4_imag.z, j);
            out4_imag.w = mod_8i<T_real>(in4_imag.w, j);
            *reinterpret_cast<char4 *>(out_imag + j * incB8i + idx) = out4_imag;
        }
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = (idx < k)     ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx)), sft))     : 0;
        in4_real.y = (idx + 1 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 1)), sft)) : 0;
        in4_real.z = (idx + 2 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 2)), sft)) : 0;
        in4_real.w = (idx + 3 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 3)), sft)) : 0;

        Vec4<T_real> in4_imag;
        in4_imag.x = (idx < k)     ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx)), sft))     : 0;
        in4_imag.y = (idx + 1 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 1)), sft)) : 0;
        in4_imag.z = (idx + 2 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 2)), sft)) : 0;
        in4_imag.w = (idx + 3 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 3)), sft)) : 0;

        char4 out4_real;
        char4 out4_imag;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4_real.x = (idx < k)     ? mod_8i<T_real>(in4_real.x, j) : 0;
            out4_real.y = (idx + 1 < k) ? mod_8i<T_real>(in4_real.y, j) : 0;
            out4_real.z = (idx + 2 < k) ? mod_8i<T_real>(in4_real.z, j) : 0;
            out4_real.w = (idx + 3 < k) ? mod_8i<T_real>(in4_real.w, j) : 0;

            *reinterpret_cast<char4 *>(out_real + j * incB8i + idx) = out4_real;

            out4_imag.x = (idx < k)     ? mod_8i<T_real>(in4_imag.x, j) : 0;
            out4_imag.y = (idx + 1 < k) ? mod_8i<T_real>(in4_imag.y, j) : 0;
            out4_imag.z = (idx + 2 < k) ? mod_8i<T_real>(in4_imag.z, j) : 0;
            out4_imag.w = (idx + 3 < k) ? mod_8i<T_real>(in4_imag.w, j) : 0;
            *reinterpret_cast<char4 *>(out_imag + j * incB8i + idx) = out4_imag;
        }
    }
}

template <typename T, bool addCol>
__forceinline__ __device__ void scalingB_bigmatrix_minusBL(const size_t col_idx,
                                const size_t k,                 // size(B,1)
                                const size_t n,                 // size(B,2)
                                const size_t incB8i,            // ldb8i * n
                                const unsigned num_moduli,      // #moduli
                                const T *const __restrict__ B,  // input (ldb * n)
                                const size_t ldb,               // leading dimension
                                int8_t *const __restrict__ B8i, // output (ldb8i * n)
                                const size_t ldb8i,             // leading dimension
                                const int16_t sft)              // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    const T *const __restrict__ in = B + col_idx * ldb;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx)), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 1)), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 2)), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 3)), sft));

        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx)), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 1)), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 2)), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 3)), sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T_real>(in4_real.x, j);
            out4.y = mod_8i<T_real>(in4_real.y, j);
            out4.z = mod_8i<T_real>(in4_real.z, j);
            out4.w = mod_8i<T_real>(in4_real.w, j);

            *reinterpret_cast<char4 *>(B8i + col_idx       * ldb8i + j * incB8i + idx    ) = out4;
            if constexpr (addCol) {
                *reinterpret_cast<char4 *>(B8i + (col_idx + n) * ldb8i + j * incB8i + idx + k) = out4;

                out4.x = mod_8i<T_real>(in4_imag.x, j);
                out4.y = mod_8i<T_real>(in4_imag.y, j);
                out4.z = mod_8i<T_real>(in4_imag.z, j);
                out4.w = mod_8i<T_real>(in4_imag.w, j);

                *reinterpret_cast<char4 *>(B8i + (col_idx + n) * ldb8i + j * incB8i + idx    ) = out4;
                out4.x = - out4.x;
                out4.y = - out4.y;
                out4.z = - out4.z;
                out4.w = - out4.w;
                *reinterpret_cast<char4 *>(B8i + col_idx       * ldb8i + j * incB8i + idx + k) = out4;
            }
            else {
                out4.x = - mod_8i<T_real>(in4_imag.x, j);
                out4.y = - mod_8i<T_real>(in4_imag.y, j);
                out4.z = - mod_8i<T_real>(in4_imag.z, j);
                out4.w = - mod_8i<T_real>(in4_imag.w, j);
                *reinterpret_cast<char4 *>(B8i + col_idx       * ldb8i + j * incB8i + idx + k) = out4;
            }
        }
    }
    i = i << 2;
    for (; i < k; i += blockDim.x) {
        T_real val_real = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(in[i]), sft));
        T_real val_imag = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(in[i]), sft));

        for (unsigned j = 0; j < num_moduli; ++j) {
            *(B8i + col_idx       * ldb8i + j * incB8i + i    ) = mod_8i<T_real>(val_real, j);
            if constexpr (addCol)
                *(B8i + (col_idx + n) * ldb8i + j * incB8i + i + k) = mod_8i<T_real>(val_real, j);

            *(B8i + col_idx       * ldb8i + j * incB8i + i + k) = - mod_8i<T_real>(val_imag, j);
            if constexpr (addCol)
                *(B8i + (col_idx + n) * ldb8i + j * incB8i + i    ) = mod_8i<T_real>(val_imag, j);
        }
    }
    for (; i + k < ldb8i; i += blockDim.x) {
        for (unsigned j = 0; j < num_moduli; ++j) {
            if constexpr (addCol)
                *(B8i + (col_idx + n) * ldb8i + j * incB8i + i + k) = 0;
            *(B8i + col_idx       * ldb8i + j * incB8i + i + k) = 0;
        }
    }
}

template <typename T>
__forceinline__ __device__ void scalingB_kara_conj(const size_t col_idx,
                                const size_t k,                 // size(B,1)
                                const size_t incB8i,            // ldb8i * n
                                const unsigned num_moduli,      // #moduli
                                const T *const __restrict__ B,  // input (ldb * n)
                                const size_t ldb,               // leading dimension
                                int8_t *const __restrict__ B8i_real, // output (ldb8i * n)
                                int8_t *const __restrict__ B8i_imag, // output (ldb8i * n)
                                const size_t ldb8i,             // leading dimension
                                const int16_t sft)              // exponent of shift value
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    const T *const __restrict__ in = B + col_idx * ldb;
    int8_t *const __restrict__ out_real = B8i_real + col_idx * ldb8i;
    int8_t *const __restrict__ out_imag = B8i_imag + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx)), sft));
        in4_real.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 1)), sft));
        in4_real.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 2)), sft));
        in4_real.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 3)), sft));

        Vec4<T_real> in4_imag;
        in4_imag.x = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx)), sft));
        in4_imag.y = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 1)), sft));
        in4_imag.z = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 2)), sft));
        in4_imag.w = Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 3)), sft));

        char4 out4_real;
        char4 out4_imag;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4_real.x = mod_8i<T_real>(in4_real.x, j);
            out4_real.y = mod_8i<T_real>(in4_real.y, j);
            out4_real.z = mod_8i<T_real>(in4_real.z, j);
            out4_real.w = mod_8i<T_real>(in4_real.w, j);

            *reinterpret_cast<char4 *>(out_real + j * incB8i + idx) = out4_real;

            out4_imag.x = - mod_8i<T_real>(in4_imag.x, j);
            out4_imag.y = - mod_8i<T_real>(in4_imag.y, j);
            out4_imag.z = - mod_8i<T_real>(in4_imag.z, j);
            out4_imag.w = - mod_8i<T_real>(in4_imag.w, j);
            *reinterpret_cast<char4 *>(out_imag + j * incB8i + idx) = out4_imag;
        }
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T_real> in4_real;
        in4_real.x = (idx < k)     ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx)), sft))     : 0;
        in4_real.y = (idx + 1 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 1)), sft)) : 0;
        in4_real.z = (idx + 2 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 2)), sft)) : 0;
        in4_real.w = (idx + 3 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Creal(*(in + idx + 3)), sft)) : 0;

        Vec4<T_real> in4_imag;
        in4_imag.x = (idx < k)     ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx)), sft))     : 0;
        in4_imag.y = (idx + 1 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 1)), sft)) : 0;
        in4_imag.z = (idx + 2 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 2)), sft)) : 0;
        in4_imag.w = (idx + 3 < k) ? Ttrunc<T_real>(Tscalbn<T_real>(oz2_type_utils::Cimag(*(in + idx + 3)), sft)) : 0;

        char4 out4_real;
        char4 out4_imag;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4_real.x = (idx < k)     ? mod_8i<T_real>(in4_real.x, j) : 0;
            out4_real.y = (idx + 1 < k) ? mod_8i<T_real>(in4_real.y, j) : 0;
            out4_real.z = (idx + 2 < k) ? mod_8i<T_real>(in4_real.z, j) : 0;
            out4_real.w = (idx + 3 < k) ? mod_8i<T_real>(in4_real.w, j) : 0;

            *reinterpret_cast<char4 *>(out_real + j * incB8i + idx) = out4_real;

            out4_imag.x = (idx < k)     ? - mod_8i<T_real>(in4_imag.x, j) : 0;
            out4_imag.y = (idx + 1 < k) ? - mod_8i<T_real>(in4_imag.y, j) : 0;
            out4_imag.z = (idx + 2 < k) ? - mod_8i<T_real>(in4_imag.z, j) : 0;
            out4_imag.w = (idx + 3 < k) ? - mod_8i<T_real>(in4_imag.w, j) : 0;
            *reinterpret_cast<char4 *>(out_imag + j * incB8i + idx) = out4_imag;
        }
    }
}

} // namespace

namespace int8tc {

__forceinline__ __device__ int compute_sft(int amax, int16_t sftA, const float log2M) {
    return sftA + __float2int_rd(__fmaf_rd(-0.51F, __log2f(__int2float_rn(amax)), log2M));
}

template <typename T>
__global__ void extract_A8i_cmpt_sftA_kernel(const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[32];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    const T_real amax                   = find_amax<T>(in, k, lda, smem);
    const int sft                  = 5 - Tilogb<T_real>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftA[row_idx] = sft;
    }
}


template <typename T>
__global__ void extract_A8i_kernel_separated(const size_t m,         // size(A,1)
                                   const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    if (k == m) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    if (x < m) {
#pragma unroll
        for (int t = 0; t < TILE_DIM; t += NY)
            if (y + t < k)
                tile[threadIdx.y + t][threadIdx.x] = A[(y + t) * lda + x];
    }

    __syncthreads();

    unsigned int tx = (threadIdx.x % 8) * 4;
    unsigned int ty = (threadIdx.x / 8) + threadIdx.y * 4;
    int nx = blockIdx_y * TILE_DIM + tx;
    int ny = blockIdx_x * TILE_DIM + ty;
    if (nx < lda8i && ny < m) {
        const int16_t sft = sftA[ny];

        char4 out4;
        out4.x = nx     < k ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(tile[tx][ty]), sft))     : 0;
        out4.y = nx + 1 < k ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(tile[tx + 1][ty]), sft)) : 0;
        out4.z = nx + 1 < k ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(tile[tx + 2][ty]), sft)) : 0;
        out4.w = nx + 1 < k ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(tile[tx + 3][ty]), sft)) : 0;
        *reinterpret_cast<char4 *>(A8i + ny * lda8i + nx) = out4;
    }
}

template <typename T>
__global__ void extract_A8i_kernel_separated_bigmatrix(const size_t m,         // size(A,1)
                                   const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    if (k == m) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    if (x < m) {
#pragma unroll
        for (int t = 0; t < TILE_DIM; t += NY)
            if (y + t < k)
                tile[threadIdx.y + t][threadIdx.x] = A[(y + t) * lda + x];
    }

    __syncthreads();

    unsigned int tx = (threadIdx.x % 8) * 4;
    unsigned int ty = (threadIdx.x / 8) + threadIdx.y * 4;
    int nx = blockIdx_y * TILE_DIM + tx;
    int ny = blockIdx_x * TILE_DIM + ty;

    const int16_t sft = sftA[ny];

    const int kmax = (k >> 2) << 2;
    if (nx < kmax && ny < m) {
        char4 out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx][ty])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 1][ty])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 2][ty])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 3][ty])), sft));

        *reinterpret_cast<char4 *>(A8i + ny       * lda8i + nx    ) = out4;
        *reinterpret_cast<char4 *>(A8i + (ny + m) * lda8i + nx + k) = out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx][ty])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 1][ty])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 2][ty])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 3][ty])), sft));

        *reinterpret_cast<char4 *>(A8i + (ny + m) * lda8i + nx    ) = out4;
        out4.x = - out4.x;
        out4.y = - out4.y;
        out4.z = - out4.z;
        out4.w = - out4.w;
        *reinterpret_cast<char4 *>(A8i + ny       * lda8i + nx + k) = out4;
    }
    else if (nx < k && ny < m) {
        int8_t mod_real = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx][ty])), sft));
        int8_t mod_imag = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx][ty])), sft));

        *(A8i + ny       * lda8i + nx    ) = mod_real;
        *(A8i + (ny + m) * lda8i + nx + k) = mod_real;

        *(A8i + (ny + m) * lda8i + nx    ) = mod_imag;
        *(A8i + ny       * lda8i + nx + k) = - mod_imag;
    }
    else if (nx < lda8i && ny < m) {
        *(A8i + (ny + m) * lda8i + nx + k) = 0;
        *(A8i + ny       * lda8i + nx + k) = 0;
    }
}

template <typename T>
__global__ void extract_A8i_kernel_separated_kara(const size_t m,    // size(A,1)
                                   const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i_real,   // output (lda8i * m)
                                   int8_t *const __restrict__ A8i_imag,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    if (k == m) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    if (x < m) {
#pragma unroll
        for (int t = 0; t < TILE_DIM; t += NY)
            if (y + t < k)
                tile[threadIdx.y + t][threadIdx.x] = A[(y + t) * lda + x];
    }

    __syncthreads();

    unsigned int tx = (threadIdx.x % 8) * 4;
    unsigned int ty = (threadIdx.x / 8) + threadIdx.y * 4;
    int nx = blockIdx_y * TILE_DIM + tx;
    int ny = blockIdx_x * TILE_DIM + ty;

    const int16_t sft = sftA[ny];

    const int kmax = (k >> 2) << 2;
    if (nx < kmax && ny < m) {
        char4 out4_real;
        char4 out4_imag;

        out4_real.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx][ty])), sft));
        out4_real.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 1][ty])), sft));
        out4_real.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 2][ty])), sft));
        out4_real.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 3][ty])), sft));
        *reinterpret_cast<char4 *>(A8i_real + ny * lda8i + nx) = out4_real;

        out4_imag.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx][ty])), sft));
        out4_imag.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 1][ty])), sft));
        out4_imag.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 2][ty])), sft));
        out4_imag.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 3][ty])), sft));
        *reinterpret_cast<char4 *>(A8i_imag + ny * lda8i + nx) = out4_imag;
    }
    else if (nx < k && ny < m) {
        int8_t mod_real = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx][ty])), sft));
        int8_t mod_imag = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx][ty])), sft));

        *(A8i_real + ny * lda8i + nx) = mod_real;
        *(A8i_imag + ny * lda8i + nx) = mod_imag;
    }
    else if (nx < lda8i && ny < m) {
        *(A8i_real + ny * lda8i + nx) = 0;
        *(A8i_imag + ny * lda8i + nx) = 0;
    }
}

template <typename T>
__global__ void extract_A8i_kernel_separated_bigmatrix_minusTR(const size_t m,         // size(A,1)
                                   const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    if (k == m) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    if (x < m) {
#pragma unroll
        for (int t = 0; t < TILE_DIM; t += NY)
            if (y + t < k)
                tile[threadIdx.y + t][threadIdx.x] = A[(y + t) * lda + x];
    }

    __syncthreads();

    unsigned int tx = (threadIdx.x % 8) * 4;
    unsigned int ty = (threadIdx.x / 8) + threadIdx.y * 4;
    int nx = blockIdx_y * TILE_DIM + tx;
    int ny = blockIdx_x * TILE_DIM + ty;

    const int16_t sft = sftA[ny];

    const int kmax = (k >> 2) << 2;
    if (nx < kmax && ny < m) {
        char4 out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx][ty])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 1][ty])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 2][ty])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 3][ty])), sft));

        *reinterpret_cast<char4 *>(A8i + ny       * lda8i + nx    ) = out4;
        *reinterpret_cast<char4 *>(A8i + (ny + m) * lda8i + nx + k) = out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx][ty])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 1][ty])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 2][ty])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 3][ty])), sft));

        *reinterpret_cast<char4 *>(A8i + ny       * lda8i + nx + k) = out4;
        out4.x = - out4.x;
        out4.y = - out4.y;
        out4.z = - out4.z;
        out4.w = - out4.w;
        *reinterpret_cast<char4 *>(A8i + (ny + m) * lda8i + nx    ) = out4;
    }
    else if (nx < k && ny < m) {
        int8_t mod_real = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx][ty])), sft));
        int8_t mod_imag = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx][ty])), sft));

        *(A8i + ny       * lda8i + nx    ) = mod_real;
        *(A8i + (ny + m) * lda8i + nx + k) = mod_real;

        *(A8i + (ny + m) * lda8i + nx    ) = - mod_imag;
        *(A8i + ny       * lda8i + nx + k) = mod_imag;
    }
    else if (nx < lda8i && ny < m) {
        *(A8i + (ny + m) * lda8i + nx + k) = 0;
        *(A8i + ny       * lda8i + nx + k) = 0;
    }
}

template <typename T>
__global__ void extract_A8i_kernel_separated_kara_conj(const size_t m,    // size(A,1)
                                   const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i_real,   // output (lda8i * m)
                                   int8_t *const __restrict__ A8i_imag,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    if (k == m) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    if (x < m) {
#pragma unroll
        for (int t = 0; t < TILE_DIM; t += NY)
            if (y + t < k)
                tile[threadIdx.y + t][threadIdx.x] = A[(y + t) * lda + x];
    }

    __syncthreads();

    unsigned int tx = (threadIdx.x % 8) * 4;
    unsigned int ty = (threadIdx.x / 8) + threadIdx.y * 4;
    int nx = blockIdx_y * TILE_DIM + tx;
    int ny = blockIdx_x * TILE_DIM + ty;

    const int16_t sft = sftA[ny];

    const int kmax = (k >> 2) << 2;
    if (nx < kmax && ny < m) {
        char4 out4_real;
        char4 out4_imag;

        out4_real.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx][ty])), sft));
        out4_real.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 1][ty])), sft));
        out4_real.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 2][ty])), sft));
        out4_real.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx + 3][ty])), sft));
        *reinterpret_cast<char4 *>(A8i_real + ny * lda8i + nx) = out4_real;

        out4_imag.x = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx][ty])), sft));
        out4_imag.y = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 1][ty])), sft));
        out4_imag.z = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 2][ty])), sft));
        out4_imag.w = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx + 3][ty])), sft));
        *reinterpret_cast<char4 *>(A8i_imag + ny * lda8i + nx) = out4_imag;
    }
    else if (nx < k && ny < m) {
        int8_t mod_real = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(tile[tx][ty])), sft));
        int8_t mod_imag = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(tile[tx][ty])), sft));

        *(A8i_real + ny * lda8i + nx) = mod_real;
        *(A8i_imag + ny * lda8i + nx) = - mod_imag;
    }
    else if (nx < lda8i && ny < m) {
        *(A8i_real + ny * lda8i + nx) = 0;
        *(A8i_imag + ny * lda8i + nx) = 0;
    }
}

template <typename T>
__global__ void extract_A8i_kernel(const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    __shared__ T smem[32];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    const T amax                   = find_amax<T>(in, k, lda, smem);
    const int sft                  = 5 - Tilogb<T>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftA[row_idx] = sft;
    }

    int8_t *const __restrict__ out = A8i + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        out4.x = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx * lda]), sft));
        out4.y = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 1) * lda]), sft));
        out4.z = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 2) * lda]), sft));
        out4.w = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 3) * lda]), sft));

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        out4.x = (idx < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx * lda]), sft)) : 0;
        out4.y = (idx + 1 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 1) * lda]), sft)) : 0;
        out4.z = (idx + 2 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 2) * lda]), sft)) : 0;
        out4.w = (idx + 3 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 3) * lda]), sft)) : 0;

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
}

template <typename T>
__global__ void extract_A8i_kernel_bigmatrix(const size_t m,                 // size(A,1)
                                   const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[32];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    const T_real amax              = find_amax<T>(in, k, lda, smem);
    const int sft                  = 5 - Tilogb<T_real>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftA[row_idx] = sft;
    }

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx * lda])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda])), sft));

        *reinterpret_cast<char4 *>(A8i + row_idx       * lda8i + idx    ) = out4;
        *reinterpret_cast<char4 *>(A8i + (row_idx + m) * lda8i + idx + k) = out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx * lda])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda])), sft));

        *reinterpret_cast<char4 *>(A8i + (row_idx + m) * lda8i + idx    ) = out4;
        out4.x = - out4.x;
        out4.y = - out4.y;
        out4.z = - out4.z;
        out4.w = - out4.w;
        *reinterpret_cast<char4 *>(A8i + row_idx       * lda8i + idx + k) = out4;
    }
    i = i << 2;  // fill rest of the matrices
    for (; i < k; i += blockDim.x) {
        T_real val_real = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[i * lda])), sft));
        T_real val_imag = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[i * lda])), sft));

        *(A8i + row_idx       * lda8i + i    ) = val_real;
        *(A8i + (row_idx + m) * lda8i + i + k) = val_real;

        *(A8i + (row_idx + m) * lda8i + i    ) = val_imag;
        *(A8i + row_idx       * lda8i + i + k) = - val_imag;
    }
    for (; i + k < lda8i; i += blockDim.x) {
        *(A8i + (row_idx + m) * lda8i + i + k) = 0;
        *(A8i + row_idx       * lda8i + i + k) = 0;
    }
}

template <typename T>
__global__ void extract_A8i_kernel_kara(const size_t k,              // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i_real,   // output (lda8i * m)
                                   int8_t *const __restrict__ A8i_imag,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[32];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    const T_real amax              = find_amax<T>(in, k, lda, smem);
    const int sft                  = 5 - Tilogb<T_real>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftA[row_idx] = sft;
    }

    int8_t *const __restrict__ out_real = A8i_real + row_idx * lda8i;
    int8_t *const __restrict__ out_imag = A8i_imag + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4_real;
        char4 out4_imag;

        out4_real.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx * lda])), sft));
        out4_real.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda])), sft));
        out4_real.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda])), sft));
        out4_real.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda])), sft));
        *reinterpret_cast<char4 *>(out_real + idx) = out4_real;

        out4_imag.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx * lda])), sft));
        out4_imag.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda])), sft));
        out4_imag.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda])), sft));
        out4_imag.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda])), sft));
        *reinterpret_cast<char4 *>(out_imag + idx) = out4_imag;
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4_real;
        char4 out4_imag;

        out4_real.x = (idx < k)     ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx * lda])), sft))       : 0;
        out4_real.y = (idx + 1 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda])), sft)) : 0;
        out4_real.z = (idx + 2 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda])), sft)) : 0;
        out4_real.w = (idx + 3 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda])), sft)) : 0;
        *reinterpret_cast<char4 *>(out_real + idx) = out4_real;

        out4_imag.x = (idx < k)     ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx * lda])), sft))       : 0;
        out4_imag.y = (idx + 1 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda])), sft)) : 0;
        out4_imag.z = (idx + 2 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda])), sft)) : 0;
        out4_imag.w = (idx + 3 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda])), sft)) : 0;
        *reinterpret_cast<char4 *>(out_imag + idx) = out4_imag;
    }
}

template <typename T>
__global__ void extract_A8i_kernel_bigmatrix_minusTR(const size_t m,         // size(A,1)
                                   const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[32];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    const T_real amax              = find_amax<T>(in, k, lda, smem);
    const int sft                  = 5 - Tilogb<T_real>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftA[row_idx] = sft;
    }

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx * lda])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda])), sft));

        *reinterpret_cast<char4 *>(A8i + row_idx       * lda8i + idx    ) = out4;
        *reinterpret_cast<char4 *>(A8i + (row_idx + m) * lda8i + idx + k) = out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx * lda])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda])), sft));

        *reinterpret_cast<char4 *>(A8i + row_idx       * lda8i + idx + k) = out4;
        out4.x = - out4.x;
        out4.y = - out4.y;
        out4.z = - out4.z;
        out4.w = - out4.w;
        *reinterpret_cast<char4 *>(A8i + (row_idx + m) * lda8i + idx    ) = out4;
    }
    i = i << 2;  // fill rest of the matrices
    for (; i < k; i += blockDim.x) {
        T_real val_real = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[i * lda])), sft));
        T_real val_imag = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[i * lda])), sft));

        *(A8i + row_idx       * lda8i + i    ) = val_real;
        *(A8i + (row_idx + m) * lda8i + i + k) = val_real;

        *(A8i + (row_idx + m) * lda8i + i    ) = - val_imag;
        *(A8i + row_idx       * lda8i + i + k) = val_imag;
    }
    for (; i + k < lda8i; i += blockDim.x) {
        *(A8i + (row_idx + m) * lda8i + i + k) = 0;
        *(A8i + row_idx       * lda8i + i + k) = 0;
    }
}

template <typename T>
__global__ void extract_A8i_kernel_kara_conj(const size_t k,              // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i_real,   // output (lda8i * m)
                                   int8_t *const __restrict__ A8i_imag,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[32];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    const T_real amax              = find_amax<T>(in, k, lda, smem);
    const int sft                  = 5 - Tilogb<T_real>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftA[row_idx] = sft;
    }

    int8_t *const __restrict__ out_real = A8i_real + row_idx * lda8i;
    int8_t *const __restrict__ out_imag = A8i_imag + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4_real;
        char4 out4_imag;

        out4_real.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx * lda])), sft));
        out4_real.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda])), sft));
        out4_real.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda])), sft));
        out4_real.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda])), sft));
        *reinterpret_cast<char4 *>(out_real + idx) = out4_real;

        out4_imag.x = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx * lda])), sft));
        out4_imag.y = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda])), sft));
        out4_imag.z = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda])), sft));
        out4_imag.w = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda])), sft));
        *reinterpret_cast<char4 *>(out_imag + idx) = out4_imag;
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4_real;
        char4 out4_imag;

        out4_real.x = (idx < k)     ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx * lda])), sft))       : 0;
        out4_real.y = (idx + 1 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 1) * lda])), sft)) : 0;
        out4_real.z = (idx + 2 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 2) * lda])), sft)) : 0;
        out4_real.w = (idx + 3 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[(idx + 3) * lda])), sft)) : 0;
        *reinterpret_cast<char4 *>(out_real + idx) = out4_real;

        out4_imag.x = (idx < k)     ? - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx * lda])), sft))       : 0;
        out4_imag.y = (idx + 1 < k) ? - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 1) * lda])), sft)) : 0;
        out4_imag.z = (idx + 2 < k) ? - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 2) * lda])), sft)) : 0;
        out4_imag.w = (idx + 3 < k) ? - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[(idx + 3) * lda])), sft)) : 0;
        *reinterpret_cast<char4 *>(out_imag + idx) = out4_imag;
    }
}

template <typename T>
__global__ void extract_B8i_kernel(const size_t k,                   // size(B,1)
                                   const T *const __restrict__ B,    // input (ldb * n)
                                   const size_t ldb,                 // leading dimension
                                   int8_t *const __restrict__ B8i,   // output (ldb8i * n)
                                   const size_t ldb8i,               // leading dimension
                                   int16_t *const __restrict__ sftB) // exponent of shift values
{
    __shared__ T smem[32];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    const T amax                   = find_amax<T>(in, k, 1u, smem);
    const int sft                  = 5 - Tilogb<T>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftB[col_idx] = sft;
    }

    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        Vec4<T> in4 = *reinterpret_cast<const Vec4<T> *>(in + idx);
        out4.x      = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in4.x), sft));
        out4.y      = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in4.y), sft));
        out4.z      = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in4.z), sft));
        out4.w      = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in4.w), sft));

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        out4.x = (idx < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx]), sft)) : 0;
        out4.y = (idx + 1 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx + 1]), sft)) : 0;
        out4.z = (idx + 2 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx + 2]), sft)) : 0;
        out4.w = (idx + 3 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx + 3]), sft)) : 0;

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
}

template <typename T>
__global__ void extract_B8i_kernel_bigmatrix(const size_t k,                 // size(B,1)
                                   const size_t n,                   // size(B,2)
                                   const T *const __restrict__ B,    // input (ldb * n)
                                   const size_t ldb,                 // leading dimension
                                   int8_t *const __restrict__ B8i,   // output (ldb8i * n)
                                   const size_t ldb8i,               // leading dimension
                                   int16_t *const __restrict__ sftB) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[32];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    const T_real amax              = find_amax<T>(in, k, 1u, smem);
    const int sft                  = 5 - Tilogb<T_real>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftB[col_idx] = sft;
    }

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 1])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 2])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 3])), sft));

        *reinterpret_cast<char4 *>(B8i + col_idx       * ldb8i + + idx    ) = out4;
        *reinterpret_cast<char4 *>(B8i + (col_idx + n) * ldb8i + + idx + k) = out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 1])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 2])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 3])), sft));

        *reinterpret_cast<char4 *>(B8i + col_idx       * ldb8i + idx + k) = out4;
        out4.x = - out4.x;
        out4.y = - out4.y;
        out4.z = - out4.z;
        out4.w = - out4.w;
        *reinterpret_cast<char4 *>(B8i + (col_idx + n) * ldb8i + idx    ) = out4;
    }
    i = i << 2;
    for (; i < k; i += blockDim.x) {
        T_real val_real = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[i])), sft));
        T_real val_imag = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[i])), sft));

        *(B8i + col_idx       * ldb8i + i    ) = val_real;
        *(B8i + (col_idx + n) * ldb8i + i + k) = val_real;

        *(B8i + col_idx       * ldb8i + i + k) = val_imag;
        *(B8i + (col_idx + n) * ldb8i + i    ) = - val_imag;
    }
    for (; i + k < ldb8i; i += blockDim.x) {
        *(B8i + (col_idx + n) * ldb8i + i + k) = 0;
        *(B8i + col_idx       * ldb8i + i + k) = 0;
    }
}

template <typename T>
__global__ void extract_B8i_kernel_kara(const size_t k,                 // size(B,1)
                                   const T *const __restrict__ B,    // input (ldb * n)
                                   const size_t ldb,                 // leading dimension
                                   int8_t *const __restrict__ B8i_real,   // output (ldb8i * n)
                                   int8_t *const __restrict__ B8i_imag,   // output (ldb8i * n)
                                   const size_t ldb8i,               // leading dimension
                                   int16_t *const __restrict__ sftB) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[32];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    const T_real amax              = find_amax<T>(in, k, 1u, smem);
    const int sft                  = 5 - Tilogb<T_real>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftB[col_idx] = sft;
    }

    int8_t *const __restrict__ out_real = B8i_real + col_idx * ldb8i;
    int8_t *const __restrict__ out_imag = B8i_imag + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4_real;
        char4 out4_imag;

        out4_real.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx])), sft));
        out4_real.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 1])), sft));
        out4_real.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 2])), sft));
        out4_real.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 3])), sft));
        *reinterpret_cast<char4 *>(out_real + idx) = out4_real;

        out4_imag.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx])), sft));
        out4_imag.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 1])), sft));
        out4_imag.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 2])), sft));
        out4_imag.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 3])), sft));
        *reinterpret_cast<char4 *>(out_imag + idx) = out4_imag;
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4_real;
        char4 out4_imag;

        out4_real.x = (idx < k)     ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx])), sft))     : 0;
        out4_real.y = (idx + 1 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 1])), sft)) : 0;
        out4_real.z = (idx + 2 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 2])), sft)) : 0;
        out4_real.w = (idx + 3 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 3])), sft)) : 0;
        *reinterpret_cast<char4 *>(out_real + idx) = out4_real;

        out4_imag.x = (idx < k)     ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx])), sft))     : 0;
        out4_imag.y = (idx + 1 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 1])), sft)) : 0;
        out4_imag.z = (idx + 2 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 2])), sft)) : 0;
        out4_imag.w = (idx + 3 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 3])), sft)) : 0;
        *reinterpret_cast<char4 *>(out_imag + idx) = out4_imag;
    }
}

template <typename T>
__global__ void extract_B8i_kernel_bigmatrix_minusBL(const size_t k,         // size(B,1)
                                   const size_t n,                   // size(B,2)
                                   const T *const __restrict__ B,    // input (ldb * n)
                                   const size_t ldb,                 // leading dimension
                                   int8_t *const __restrict__ B8i,   // output (ldb8i * n)
                                   const size_t ldb8i,               // leading dimension
                                   int16_t *const __restrict__ sftB) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[32];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    const T_real amax              = find_amax<T>(in, k, 1u, smem);
    const int sft                  = 5 - Tilogb<T_real>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftB[col_idx] = sft;
    }

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 1])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 2])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 3])), sft));

        *reinterpret_cast<char4 *>(B8i + col_idx       * ldb8i + + idx    ) = out4;
        *reinterpret_cast<char4 *>(B8i + (col_idx + n) * ldb8i + + idx + k) = out4;

        out4.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx])), sft));
        out4.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 1])), sft));
        out4.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 2])), sft));
        out4.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 3])), sft));

        *reinterpret_cast<char4 *>(B8i + (col_idx + n) * ldb8i + idx    ) = out4;
        out4.x = - out4.x;
        out4.y = - out4.y;
        out4.z = - out4.z;
        out4.w = - out4.w;
        *reinterpret_cast<char4 *>(B8i + col_idx       * ldb8i + idx + k) = out4;
    }
    i = i << 2;
    for (; i < k; i += blockDim.x) {
        T_real val_real = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[i])), sft));
        T_real val_imag = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[i])), sft));

        *(B8i + col_idx       * ldb8i + i    ) = val_real;
        *(B8i + (col_idx + n) * ldb8i + i + k) = val_real;

        *(B8i + col_idx       * ldb8i + i + k) = - val_imag;
        *(B8i + (col_idx + n) * ldb8i + i    ) = val_imag;
    }
    for (; i + k < ldb8i; i += blockDim.x) {
        *(B8i + (col_idx + n) * ldb8i + i + k) = 0;
        *(B8i + col_idx       * ldb8i + i + k) = 0;
    }
}

template <typename T>
__global__ void extract_B8i_kernel_kara_conj(const size_t k,                 // size(B,1)
                                   const T *const __restrict__ B,    // input (ldb * n)
                                   const size_t ldb,                 // leading dimension
                                   int8_t *const __restrict__ B8i_real,   // output (ldb8i * n)
                                   int8_t *const __restrict__ B8i_imag,   // output (ldb8i * n)
                                   const size_t ldb8i,               // leading dimension
                                   int16_t *const __restrict__ sftB) // exponent of shift values
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[32];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    const T_real amax              = find_amax<T>(in, k, 1u, smem);
    const int sft                  = 5 - Tilogb<T_real>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftB[col_idx] = sft;
    }

    int8_t *const __restrict__ out_real = B8i_real + col_idx * ldb8i;
    int8_t *const __restrict__ out_imag = B8i_imag + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4_real;
        char4 out4_imag;

        out4_real.x = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx])), sft));
        out4_real.y = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 1])), sft));
        out4_real.z = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 2])), sft));
        out4_real.w = __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 3])), sft));
        *reinterpret_cast<char4 *>(out_real + idx) = out4_real;

        out4_imag.x = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx])), sft));
        out4_imag.y = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 1])), sft));
        out4_imag.z = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 2])), sft));
        out4_imag.w = - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 3])), sft));
        *reinterpret_cast<char4 *>(out_imag + idx) = out4_imag;
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        char4 out4_real;
        char4 out4_imag;

        out4_real.x = (idx < k)     ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx])), sft))     : 0;
        out4_real.y = (idx + 1 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 1])), sft)) : 0;
        out4_real.z = (idx + 2 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 2])), sft)) : 0;
        out4_real.w = (idx + 3 < k) ? __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Creal(in[idx + 3])), sft)) : 0;
        *reinterpret_cast<char4 *>(out_real + idx) = out4_real;

        out4_imag.x = (idx < k)     ? - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx])), sft))     : 0;
        out4_imag.y = (idx + 1 < k) ? - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 1])), sft)) : 0;
        out4_imag.z = (idx + 2 < k) ? - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 2])), sft)) : 0;
        out4_imag.w = (idx + 3 < k) ? - __T2int_ru<T_real>(Tscalbn<T_real>(Tabs<T_real>(oz2_type_utils::Cimag(in[idx + 3])), sft)) : 0;
        *reinterpret_cast<char4 *>(out_imag + idx) = out4_imag;
    }
}

template <typename T>
__global__ void scalingA_kernel(const size_t n,                         // size(C,2)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ A8i,         // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                int16_t *const __restrict__ sftA,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto row_idx = blockIdx.x;
    const int sftAi    = sftA[row_idx];
    const int32_t amax = find_amax<int32_t>(C32i + row_idx, n, ldc32i, smem);
    const int sft      = compute_sft(amax, sftAi, log2M);

    scalingA<T>(row_idx, k, incA8i, num_moduli, A, lda, A8i, lda8i, sft);

    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }
}

template <typename T, bool addCol>
__global__ void scalingA_kernel_bigmatrix(const size_t n,                       // size(C,2)
                                const size_t m,                         // size(A,1)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ A8i,         // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                int16_t *const __restrict__ sftA,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto row_idx = blockIdx.x;
    const int sftAi    = sftA[row_idx];
    const int32_t amax = find_amax<int32_t>(C32i + row_idx, 2 * n, ldc32i, smem);
    const int sft      = compute_sft(amax, sftAi, log2M);

    scalingA_bigmatrix<T, addCol>(row_idx, m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sft);

    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingA_kernel_kara(const size_t n,                       // size(C,2)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                const int32_t *const __restrict__ C32i_real, // input (ldc32i * n)
                                const int32_t *const __restrict__ C32i_imag, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ A8i_real,    // output (lda8i * m)
                                int8_t *const __restrict__ A8i_imag,    // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                int16_t *const __restrict__ sftA,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[64];
    const auto row_idx = blockIdx.x;
    const int sftAi    = sftA[row_idx];
    const int32_t amax_real = find_amax<int32_t>(C32i_real + row_idx, n, ldc32i, smem);
    const int32_t amax_imag = find_amax<int32_t>(C32i_imag + row_idx, n, ldc32i, smem + 32);
    const int sft      = compute_sft(max(amax_real, amax_imag), sftAi, log2M);

    scalingA_kara<T>(row_idx, k, incA8i, num_moduli, A, lda, A8i_real, A8i_imag, lda8i, sft);

    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }
}

template <typename T, bool addCol>
__global__ void scalingA_kernel_bigmatrix_minusTR(const size_t n,               // size(C,2)
                                const size_t m,                         // size(A,1)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ A8i,         // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                int16_t *const __restrict__ sftA,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto row_idx = blockIdx.x;
    const int sftAi    = sftA[row_idx];
    const int32_t amax = find_amax<int32_t>(C32i + row_idx, 2 * n, ldc32i, smem);
    const int sft      = compute_sft(amax, sftAi, log2M);

    scalingA_bigmatrix_minusTR<T, addCol>(row_idx, m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sft);

    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingA_kernel_kara_conj(const size_t n,               // size(C,2)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                const int32_t *const __restrict__ C32i_real, // input (ldc32i * n)
                                const int32_t *const __restrict__ C32i_imag, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ A8i_real,    // output (lda8i * m)
                                int8_t *const __restrict__ A8i_imag,    // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                int16_t *const __restrict__ sftA,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[64];
    const auto row_idx = blockIdx.x;
    const int sftAi    = sftA[row_idx];
    const int32_t amax_real = find_amax<int32_t>(C32i_real + row_idx,  n, ldc32i, smem);
    const int32_t amax_imag = find_amax<int32_t>(C32i_imag + row_idx,  n, ldc32i, smem + 32);
    const int sft      = compute_sft(max(amax_real, amax_imag), sftAi, log2M);

    scalingA_kara_conj<T>(row_idx, k, incA8i, num_moduli, A, lda, A8i_real, A8i_imag, lda8i, sft);

    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingB_kernel(const size_t m,                         // size(C,1)
                                const size_t k,                         // size(B,1)
                                const size_t incB8i,                    // ldb8i * n
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ B,          // input (ldb * n)
                                const size_t ldb,                       // leading dimension
                                const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ B8i,         // output (ldb8i * n)
                                const size_t ldb8i,                     // leading dimension
                                int16_t *const __restrict__ sftB,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto col_idx = blockIdx.x;
    const int sftBi    = sftB[col_idx];
    const int32_t amax = find_amax<int32_t>(C32i + col_idx * ldc32i, m, 1u, smem);
    const int sft      = compute_sft(amax, sftBi, log2M);

    scalingB<T>(col_idx, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sft);

    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }
}

template <typename T, bool addCol>
__global__ void scalingB_kernel_bigmatrix(const size_t m,                         // size(C,1)
                                const size_t k,                         // size(B,1)
                                const size_t n,                         // size(B,2)
                                const size_t incB8i,                    // ldb8i * n
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ B,          // input (ldb * n)
                                const size_t ldb,                       // leading dimension
                                const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ B8i,         // output (ldb8i * n)
                                const size_t ldb8i,                     // leading dimension
                                int16_t *const __restrict__ sftB,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto col_idx = blockIdx.x;
    const int sftBi    = sftB[col_idx];
    const int32_t amax = find_amax<int32_t>(C32i + col_idx * ldc32i, 2 * m, 1u, smem);
    const int sft      = compute_sft(amax, sftBi, log2M);

    scalingB_bigmatrix<T, addCol>(col_idx, k, n, incB8i, num_moduli, B, ldb, B8i, ldb8i, sft);

    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingB_kernel_kara(const size_t m,                         // size(C,1)
                                const size_t k,                         // size(B,1)
                                const size_t incB8i,                    // ldb8i * n
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ B,          // input (ldb * n)
                                const size_t ldb,                       // leading dimension
                                const int32_t *const __restrict__ C32i_real, // input (ldc32i * n)
                                const int32_t *const __restrict__ C32i_imag, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ B8i_real,    // output (ldb8i * n)
                                int8_t *const __restrict__ B8i_imag,    // output (ldb8i * n)
                                const size_t ldb8i,                     // leading dimension
                                int16_t *const __restrict__ sftB,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[64];
    const auto col_idx = blockIdx.x;
    const int sftBi    = sftB[col_idx];
    const int32_t amax_real = find_amax<int32_t>(C32i_real + col_idx * ldc32i, m, 1u, smem);
    const int32_t amax_imag = find_amax<int32_t>(C32i_imag + col_idx * ldc32i, m, 1u, smem + 32);
    const int sft      = compute_sft(max(amax_real, amax_imag), sftBi, log2M);

    scalingB_kara<T>(col_idx, k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sft);

    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }
}

template <typename T, bool addCol>
__global__ void scalingB_kernel_bigmatrix_minusBL(const size_t m,               // size(C,1)
                                const size_t k,                         // size(B,1)
                                const size_t n,                         // size(B,2)
                                const size_t incB8i,                    // ldb8i * n
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ B,          // input (ldb * n)
                                const size_t ldb,                       // leading dimension
                                const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ B8i,         // output (ldb8i * n)
                                const size_t ldb8i,                     // leading dimension
                                int16_t *const __restrict__ sftB,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto col_idx = blockIdx.x;
    const int sftBi    = sftB[col_idx];
    const int32_t amax = find_amax<int32_t>(C32i + col_idx * ldc32i, 2 * m, 1u, smem);
    const int sft      = compute_sft(amax, sftBi, log2M);

    scalingB_bigmatrix_minusBL<T, addCol>(col_idx, k, n, incB8i, num_moduli, B, ldb, B8i, ldb8i, sft);

    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingB_kernel_kara_conj(const size_t m,               // size(C,1)
                                const size_t k,                         // size(B,1)
                                const size_t incB8i,                    // ldb8i * n
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ B,          // input (ldb * n)
                                const size_t ldb,                       // leading dimension
                                const int32_t *const __restrict__ C32i_real, // input (ldc32i * n)
                                const int32_t *const __restrict__ C32i_imag, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ B8i_real,    // output (ldb8i * n)
                                int8_t *const __restrict__ B8i_imag,    // output (ldb8i * n)
                                const size_t ldb8i,                     // leading dimension
                                int16_t *const __restrict__ sftB,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[64];
    const auto col_idx = blockIdx.x;
    const int sftBi    = sftB[col_idx];
    const int32_t amax_real = find_amax<int32_t>(C32i_real + col_idx * ldc32i, m, 1u, smem);
    const int32_t amax_imag = find_amax<int32_t>(C32i_imag + col_idx * ldc32i, m, 1u, smem + 32);
    const int sft      = compute_sft(max(amax_real, amax_imag), sftBi, log2M);

    scalingB_kara_conj<T>(col_idx, k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sft);

    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingAT_kernel(const size_t n,                         // size(C,2)
                                 const size_t k,                         // size(AT,1)
                                 const size_t incA8i,                    // lda8i * n
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ A,          // input (lda * n)
                                 const size_t lda,                       // leading dimension
                                 const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ A8i,         // output (lda8i * n)
                                 const size_t lda8i,                     // leading dimension
                                 int16_t *const __restrict__ sftA,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto col_idx = blockIdx.x;
    const int sftAi    = sftA[col_idx];
    const int32_t amax = find_amax<int32_t>(C32i + col_idx, n, ldc32i, smem);
    const int sft      = compute_sft(amax, sftAi, log2M);

    scalingB<T>(col_idx, k, incA8i, num_moduli, A, lda, A8i, lda8i, sft);

    if (threadIdx.x == 0) {
        sftA[col_idx] = -sft;
    }
}

template <typename T, bool addCol>
__global__ void scalingAT_kernel_bigmatrix(const size_t n,                       // size(C,2)
                                 const size_t k,                         // size(AT,1)
                                 const size_t m,                         // size(AT,2)
                                 const size_t incA8i,                    // lda8i * n
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ A,          // input (lda * n)
                                 const size_t lda,                       // leading dimension
                                 const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ A8i,         // output (lda8i * n)
                                 const size_t lda8i,                     // leading dimension
                                 int16_t *const __restrict__ sftA,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto col_idx = blockIdx.x;
    const int sftAi    = sftA[col_idx];
    const int32_t amax = find_amax<int32_t>(C32i + col_idx, 2 * n, ldc32i, smem);
    const int sft      = compute_sft(amax, sftAi, log2M);

    scalingB_bigmatrix<T, addCol>(col_idx, k, m, incA8i, num_moduli, A, lda, A8i, lda8i, sft);

    if (threadIdx.x == 0) {
        sftA[col_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingAT_kernel_kara(const size_t n,                       // size(C,2)
                                 const size_t k,                         // size(AT,1)
                                 const size_t incA8i,                    // lda8i * n
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ A,          // input (lda * n)
                                 const size_t lda,                       // leading dimension
                                 const int32_t *const __restrict__ C32i_real, // input (ldc32i * n)
                                 const int32_t *const __restrict__ C32i_imag, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ A8i_real,    // output (lda8i * n)
                                 int8_t *const __restrict__ A8i_imag,    // output (lda8i * n)
                                 const size_t lda8i,                     // leading dimension
                                 int16_t *const __restrict__ sftA,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[64];
    const auto col_idx = blockIdx.x;
    const int sftAi    = sftA[col_idx];
    const int32_t amax_real = find_amax<int32_t>(C32i_real + col_idx, n, ldc32i, smem);
    const int32_t amax_imag = find_amax<int32_t>(C32i_imag + col_idx, n, ldc32i, smem + 32);
    const int sft      = compute_sft(max(amax_real, amax_imag), sftAi, log2M);

    scalingB_kara<T>(col_idx, k, incA8i, num_moduli, A, lda, A8i_real, A8i_imag, lda8i, sft);

    if (threadIdx.x == 0) {
        sftA[col_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingAT_kernel_kara_conj(const size_t n,                       // size(C,2)
                                 const size_t k,                         // size(AT,1)
                                 const size_t incA8i,                    // lda8i * n
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ A,          // input (lda * n)
                                 const size_t lda,                       // leading dimension
                                 const int32_t *const __restrict__ C32i_real, // input (ldc32i * n)
                                 const int32_t *const __restrict__ C32i_imag, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ A8i_real,    // output (lda8i * n)
                                 int8_t *const __restrict__ A8i_imag,    // output (lda8i * n)
                                 const size_t lda8i,                     // leading dimension
                                 int16_t *const __restrict__ sftA,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[64];
    const auto col_idx = blockIdx.x;
    const int sftAi    = sftA[col_idx];
    const int32_t amax_real = find_amax<int32_t>(C32i_real + col_idx, n, ldc32i, smem);
    const int32_t amax_imag = find_amax<int32_t>(C32i_imag + col_idx, n, ldc32i, smem + 32);
    const int sft      = compute_sft(max(amax_real, amax_imag), sftAi, log2M);

    scalingB_kara_conj<T>(col_idx, k, incA8i, num_moduli, A, lda, A8i_real, A8i_imag, lda8i, sft);

    if (threadIdx.x == 0) {
        sftA[col_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingBT_kernel(const size_t m,                         // size(C,1)
                                 const size_t k,                         // size(B,2)
                                 const size_t incB8i,                    // ldb8i * m
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ B,          // input (ldb * m)
                                 const size_t ldb,                       // leading dimension
                                 const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ B8i,         // output (ldb8i * m)
                                 const size_t ldb8i,                     // leading dimension
                                 int16_t *const __restrict__ sftB,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto row_idx = blockIdx.x;
    const int sftBi    = sftB[row_idx];
    const int32_t amax = find_amax<int32_t>(C32i + row_idx * ldc32i, m, 1u, smem);
    const int sft      = compute_sft(amax, sftBi, log2M);

    scalingA<T>(row_idx, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sft);

    if (threadIdx.x == 0) {
        sftB[row_idx] = -sft;
    }
}

template <typename T, bool addCol>
__global__ void scalingBT_kernel_bigmatrix(const size_t m,                       // size(C,1)
                                 const size_t k,                         // size(B,2)
                                 const size_t n,                         // size(B,1)
                                 const size_t incB8i,                    // ldb8i * m
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ B,          // input (ldb * m)
                                 const size_t ldb,                       // leading dimension
                                 const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ B8i,         // output (ldb8i * m)
                                 const size_t ldb8i,                     // leading dimension
                                 int16_t *const __restrict__ sftB,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto row_idx = blockIdx.x;
    const int sftBi    = sftB[row_idx];
    const int32_t amax = find_amax<int32_t>(C32i + row_idx * ldc32i, 2 * m, 1u, smem);
    const int sft      = compute_sft(amax, sftBi, log2M);

    scalingA_bigmatrix<T, addCol>(row_idx, n, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sft);

    if (threadIdx.x == 0) {
        sftB[row_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingBT_kernel_kara(const size_t m,                       // size(C,1)
                                 const size_t k,                         // size(B,2)
                                 const size_t incB8i,                    // ldb8i * m
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ B,          // input (ldb * m)
                                 const size_t ldb,                       // leading dimension
                                 const int32_t *const __restrict__ C32i_real, // input (ldc32i * n)
                                 const int32_t *const __restrict__ C32i_imag, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ B8i_real,    // output (ldb8i * m)
                                 int8_t *const __restrict__ B8i_imag,    // output (ldb8i * m)
                                 const size_t ldb8i,                     // leading dimension
                                 int16_t *const __restrict__ sftB,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[64];
    const auto row_idx = blockIdx.x;
    const int sftBi    = sftB[row_idx];
    const int32_t amax_real = find_amax<int32_t>(C32i_real + row_idx * ldc32i, m, 1u, smem);
    const int32_t amax_imag = find_amax<int32_t>(C32i_imag + row_idx * ldc32i, m, 1u, smem + 32);
    const int sft      = compute_sft(max(amax_real, amax_imag), sftBi, log2M);

    scalingA_kara<T>(row_idx, k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sft);

    if (threadIdx.x == 0) {
        sftB[row_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingBT_kernel_kara_conj(const size_t m,                       // size(C,1)
                                 const size_t k,                         // size(B,2)
                                 const size_t incB8i,                    // ldb8i * m
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ B,          // input (ldb * m)
                                 const size_t ldb,                       // leading dimension
                                 const int32_t *const __restrict__ C32i_real, // input (ldc32i * n)
                                 const int32_t *const __restrict__ C32i_imag, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ B8i_real,    // output (ldb8i * m)
                                 int8_t *const __restrict__ B8i_imag,    // output (ldb8i * m)
                                 const size_t ldb8i,                     // leading dimension
                                 int16_t *const __restrict__ sftB,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[64];
    const auto row_idx = blockIdx.x;
    const int sftBi    = sftB[row_idx];
    const int32_t amax_real = find_amax<int32_t>(C32i_real + row_idx * ldc32i, m, 1u, smem);
    const int32_t amax_imag = find_amax<int32_t>(C32i_imag + row_idx * ldc32i, m, 1u, smem + 32);
    const int sft      = compute_sft(max(amax_real, amax_imag), sftBi, log2M);

    scalingA_kara_conj<T>(row_idx, k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sft);

    if (threadIdx.x == 0) {
        sftB[row_idx] = -sft;
    }
}

template <typename TA, typename TB>
__inline__ void scaling(gpublasHandle_t handle,        // handle
                        const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const size_t m,                // size(A,1) & size(C,1)
                        const size_t n,                // size(B,2) & size(C,2)
                        const size_t k,                // size(A,2) & size(B,1)
                        const unsigned num_moduli,     // #moduli
                        const TA *const A,             // input
                        const size_t lda,              // leading dimension
                        const TB *const B,             // input
                        const size_t ldb,              // leading dimension
                        int8_t *const A8i,             // output (k * m)
                        const size_t lda8i,            // leading dimension
                        const size_t incA8i,           // increment between the A8i
                        int16_t *const sftA,           // exponent of shift values for rows of A
                        int8_t *const B8i,             // output (k * n)
                        const size_t ldb8i,            // leading dimension
                        const size_t incB8i,           // increment between the B8i
                        int16_t *const sftB,           // exponent of shift values for cols of B
                        int32_t *const C32i,           // tmp (m * n)
                        const unsigned table_idx)      //
{
    // extract first 7-bit from A and B
    if (op_A == GPUBLAS_OP_N) {
#if defined(__HIPCC__)
        if (lda % 1024 == 0) {
            stair_kernel<TA><<<k, oz2_const::threads_scaling>>>(m, A, lda, reinterpret_cast<TA *>(A8i), lda + 1);
            extract_A8i_cmpt_sftA_kernel<TA><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TA *>(A8i), lda + 1, sftA);
            dim3 grid((m + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            extract_A8i_kernel_separated<TA><<<grid, threads_scalingA>>>(m, k, A, lda, A8i, lda8i, sftA);
        }
        else
#endif
            extract_A8i_kernel<TA><<<m, oz2_const::threads_scaling>>>(k, A, lda, A8i, lda8i, sftA);
    }
    if (op_B != GPUBLAS_OP_N) {
#if defined(__HIPCC__)
        if (ldb % 1024 == 0) {
            stair_kernel<TB><<<k, oz2_const::threads_scaling>>>(n, B, ldb, reinterpret_cast<TB *>(B8i), ldb + 1);
            extract_A8i_cmpt_sftA_kernel<TB><<<n, oz2_const::threads_scaling>>>(k, reinterpret_cast<TB *>(B8i), ldb + 1, sftB);
            dim3 grid((n + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            extract_A8i_kernel_separated<TB><<<grid, threads_scalingA>>>(n, k, B, ldb, B8i, ldb8i, sftB);
        }
        else
#endif
            extract_A8i_kernel<TB><<<n, oz2_const::threads_scaling>>>(k, B, ldb, B8i, ldb8i, sftB);
    }
    if (op_A != GPUBLAS_OP_N) {
        extract_B8i_kernel<TA><<<m, oz2_const::threads_scaling>>>(k, A, lda, A8i, lda8i, sftA);
    }
    if (op_B == GPUBLAS_OP_N) {
        extract_B8i_kernel<TB><<<n, oz2_const::threads_scaling>>>(k, B, ldb, B8i, ldb8i, sftB);
    }

    // C32i := A8i^T*B8i
    constexpr int32_t alpha = 1;
    constexpr int32_t beta  = 0;
    gpuDeviceSynchronize();
#if defined(__HIPCC__)
    const size_t m_pad = m;
    const size_t ldc32i = m_pad % 1024 == 0 ? m_pad + 4 : m_pad;
#else
    const size_t m_pad = ((m + 3) >> 2) << 2;
    const size_t ldc32i = m_pad;
#endif
    gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &alpha, A8i, GPU_R_8I, lda8i, B8i, GPU_R_8I, ldb8i, &beta, C32i, GPU_R_32I, ldc32i, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);

    // extract high order bits from A and B
    gpuDeviceSynchronize();
    const float log2M = oz2_table::int8tc::log2M[table_idx]; // fld(log2(M-1)/2 - 0.5)
    if (op_A == GPUBLAS_OP_N) {
        scalingA_kernel<TA><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, ldc32i, A8i, lda8i, sftA, log2M);
    } else {
        scalingAT_kernel<TA><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i, ldc32i, A8i, lda8i, sftA, log2M);
    }
    if (op_B == GPUBLAS_OP_N) {
        scalingB_kernel<TB><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, ldc32i, B8i, ldb8i, sftB, log2M);
    } else {
        scalingBT_kernel<TB><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i, ldc32i, B8i, ldb8i, sftB, log2M);
    }
}

template <typename TA, typename TB>
__inline__ void scaling_bigmatrix(gpublasHandle_t handle,        // handle
                        const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const size_t m,                // size(A,1) & size(C,1)
                        const size_t n,                // size(B,2) & size(C,2)
                        const size_t k,                // size(A,2) & size(B,1)
                        const unsigned num_moduli,     // #moduli
                        const TA *const A,             // input
                        const size_t lda,              // leading dimension
                        const TB *const B,             // input
                        const size_t ldb,              // leading dimension
                        int8_t *const A8i,             // output (k * m)
                        const size_t lda8i,            // leading dimension
                        const size_t incA8i,           // increment between the A8i
                        int16_t *const sftA,           // exponent of shift values for rows of A
                        int8_t *const B8i,             // output (k * n)
                        const size_t ldb8i,            // leading dimension
                        const size_t incB8i,           // increment between the B8i
                        int16_t *const sftB,           // exponent of shift values for cols of B
                        int32_t *const C32i,           // tmp (m * n)
                        const unsigned table_idx)      //
{
    // extract first 7-bit from A and B
    if (op_A == GPUBLAS_OP_N) {
#if defined(__HIPCC__)
        if (lda % 1024 == 0) {
            stair_kernel<TA><<<k, oz2_const::threads_scaling>>>(m, A, lda, reinterpret_cast<TA *>(A8i), lda + 1);
            extract_A8i_cmpt_sftA_kernel<TA><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TA *>(A8i), lda + 1, sftA);
            dim3 grid((m + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            extract_A8i_kernel_separated_bigmatrix<TA><<<grid, threads_scalingA>>>(m, k, A, lda, A8i, lda8i, sftA);
        }
        else
#endif
            extract_A8i_kernel_bigmatrix<TA><<<m, oz2_const::threads_scaling>>>(m, k, A, lda, A8i, lda8i, sftA);
    }
    if (op_B == GPUBLAS_OP_T) {
#if defined(__HIPCC__)
        if (ldb % 1024 == 0) {
            stair_kernel<TB><<<k, oz2_const::threads_scaling>>>(n, B, ldb, reinterpret_cast<TB *>(B8i), ldb + 1);
            extract_A8i_cmpt_sftA_kernel<TB><<<n, oz2_const::threads_scaling>>>(k, reinterpret_cast<TB *>(B8i), ldb + 1, sftB);
            dim3 grid((n + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            extract_A8i_kernel_separated_bigmatrix_minusTR<TB><<<grid, threads_scalingA>>>(n, k, B, ldb, B8i, ldb8i, sftB);
        }
        else
#endif
            extract_A8i_kernel_bigmatrix_minusTR<TB><<<n, oz2_const::threads_scaling>>>(m, k, B, ldb, B8i, ldb8i, sftB);
    }
    if (op_B == GPUBLAS_OP_C) {
#if defined(__HIPCC__)
        if (ldb % 1024 == 0) {
            stair_kernel<TB><<<k, oz2_const::threads_scaling>>>(n, B, ldb, reinterpret_cast<TB *>(B8i), ldb + 1);
            extract_A8i_cmpt_sftA_kernel<TB><<<n, oz2_const::threads_scaling>>>(k, reinterpret_cast<TB *>(B8i), ldb + 1, sftB);
            dim3 grid((n + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            extract_A8i_kernel_separated_bigmatrix<TB><<<grid, threads_scalingA>>>(n, k, B, ldb, B8i, ldb8i, sftB);
        }
        else
#endif
            extract_A8i_kernel_bigmatrix<TB><<<n, oz2_const::threads_scaling>>>(m, k, B, ldb, B8i, ldb8i, sftB);
    }
    if (op_B == GPUBLAS_OP_N) {
        extract_B8i_kernel_bigmatrix<TB><<<n, oz2_const::threads_scaling>>>(k, n, B, ldb, B8i, ldb8i, sftB);
    }
    if (op_A == GPUBLAS_OP_T) {
        extract_B8i_kernel_bigmatrix_minusBL<TA><<<m, oz2_const::threads_scaling>>>(k, n, A, lda, A8i, lda8i, sftA);
    }
    if (op_A == GPUBLAS_OP_C) {
        extract_B8i_kernel_bigmatrix<TA><<<m, oz2_const::threads_scaling>>>(k, n, A, lda, A8i, lda8i, sftA);
    }

    // C32i := A8i^T*B8i
    constexpr int32_t alpha = 1;
    constexpr int32_t beta  = 0;
    gpuDeviceSynchronize();
#if defined(__HIPCC__)
    const size_t m2_pad = 2 * m;
    const size_t ldc32i = m2_pad % 512 == 0 ? m2_pad + 1 : m2_pad;
#else
    const size_t m2_pad = ((2 * m + 3) >> 2) << 2;
    const size_t ldc32i = m2_pad;
#endif
    gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m2_pad, 2 * n, lda8i, &alpha, A8i, GPU_R_8I, lda8i, B8i, GPU_R_8I, ldb8i, &beta, C32i, GPU_R_32I, ldc32i, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);

    // extract high order bits from A and B
    gpuDeviceSynchronize();
    const float log2M = oz2_table::int8tc::log2M[table_idx]; // fld(log2(M-1)/2 - 0.5)
    if (op_A == GPUBLAS_OP_N) {
        scalingA_kernel_bigmatrix<TA, true><<<m, oz2_const::threads_scaling>>>(n, m, k, incA8i, num_moduli, A, lda, C32i, ldc32i, A8i, lda8i, sftA, log2M);
    }
    if (op_B == GPUBLAS_OP_N) {
        scalingB_kernel_bigmatrix<TB, false><<<n, oz2_const::threads_scaling>>>(m, k, n, incB8i, num_moduli, B, ldb, C32i, ldc32i, B8i, ldb8i, sftB, log2M);
    }
    if (op_A == GPUBLAS_OP_T) {
        scalingB_kernel_bigmatrix_minusBL<TA, true><<<m, oz2_const::threads_scaling>>>(n, k, m, incA8i, num_moduli, A, lda, C32i, ldc32i, A8i, lda8i, sftA, log2M);
    }
    if (op_B == GPUBLAS_OP_T) {
        scalingA_kernel_bigmatrix_minusTR<TB, false><<<n, oz2_const::threads_scaling>>>(m, n, k, incB8i, num_moduli, B, ldb, C32i, ldc32i, B8i, ldb8i, sftB, log2M);
    }
    if (op_A == GPUBLAS_OP_C) {
        scalingAT_kernel_bigmatrix<TA, true><<<m, oz2_const::threads_scaling>>>(n, k, m, incA8i, num_moduli, A, lda, C32i, ldc32i, A8i, lda8i, sftA, log2M);
    }
    if (op_B == GPUBLAS_OP_C) {
        scalingBT_kernel_bigmatrix<TB, false><<<n, oz2_const::threads_scaling>>>(m, k, n, incB8i, num_moduli, B, ldb, C32i, ldc32i, B8i, ldb8i, sftB, log2M);
    }
}

template <typename TA, typename TB>
__inline__ void scaling_kara(gpublasHandle_t handle,   // handle
                        const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const size_t m,                // size(A,1) & size(C,1)
                        const size_t n,                // size(B,2) & size(C,2)
                        const size_t k,                // size(A,2) & size(B,1)
                        const unsigned num_moduli,     // #moduli
                        const TA *const A,             // input
                        const size_t lda,              // leading dimension
                        const TB *const B,             // input
                        const size_t ldb,              // leading dimension
                        int8_t *const A8i_real,        // output (k * m)
                        int8_t *const A8i_imag,        // output (k * m)
                        const size_t lda8i,            // leading dimension
                        const size_t incA8i,           // increment between the A8i
                        int16_t *const sftA,           // exponent of shift values for rows of A
                        int8_t *const B8i_real,        // output (k * n)
                        int8_t *const B8i_imag,        // output (k * n)
                        const size_t ldb8i,            // leading dimension
                        const size_t incB8i,           // increment between the B8i
                        int16_t *const sftB,           // exponent of shift values for cols of B
                        int32_t *const C32i_real,      // tmp (m * n)
                        int32_t *const C32i_imag,      // tmp (m * n)
                        const unsigned table_idx)      //
{
    // extract first 7-bit from A and B
    if (op_A == GPUBLAS_OP_N) {
#if defined(__HIPCC__)
        if (lda % 1024 == 0) {
            stair_kernel<TA><<<k, oz2_const::threads_scaling>>>(m, A, lda, reinterpret_cast<TA *>(A8i_real), lda + 1);
            extract_A8i_cmpt_sftA_kernel<TA><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TA *>(A8i_real), lda + 1, sftA);
            dim3 grid((m + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            extract_A8i_kernel_separated_kara<TA><<<grid, threads_scalingA>>>(m, k, A, lda, A8i_real, A8i_imag, lda8i, sftA);
        }
        else
#endif
            extract_A8i_kernel_kara<TA><<<m, oz2_const::threads_scaling>>>(k, A, lda, A8i_real, A8i_imag, lda8i, sftA);
    }
    if (op_B == GPUBLAS_OP_T) {
#if defined(__HIPCC__)
        if (ldb % 1024 == 0) {
            stair_kernel<TB><<<k, oz2_const::threads_scaling>>>(n, B, ldb, reinterpret_cast<TB *>(B8i_real), ldb + 1);
            extract_A8i_cmpt_sftA_kernel<TB><<<n, oz2_const::threads_scaling>>>(k, reinterpret_cast<TB *>(B8i_real), ldb + 1, sftB);
            dim3 grid((n + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            extract_A8i_kernel_separated_kara<TB><<<grid, threads_scalingA>>>(n, k, B, ldb, B8i_real, B8i_imag, ldb8i, sftB);
        }
        else
#endif
            extract_A8i_kernel_kara<TB><<<n, oz2_const::threads_scaling>>>(k, B, ldb, B8i_real, B8i_imag, ldb8i, sftB);
    }
    if (op_B == GPUBLAS_OP_C) {
#if defined(__HIPCC__)
        if (ldb % 1024 == 0) {
            stair_kernel<TB><<<k, oz2_const::threads_scaling>>>(n, B, ldb, reinterpret_cast<TB *>(B8i_real), ldb + 1);
            extract_A8i_cmpt_sftA_kernel<TB><<<n, oz2_const::threads_scaling>>>(k, reinterpret_cast<TB *>(B8i_real), ldb + 1, sftB);
            dim3 grid((n + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            extract_A8i_kernel_separated_kara_conj<TB><<<grid, threads_scalingA>>>(n, k, B, ldb, B8i_real, B8i_imag, ldb8i, sftB);
        }
        else
#endif
            extract_A8i_kernel_kara_conj<TB><<<n, oz2_const::threads_scaling>>>(k, B, ldb, B8i_real, B8i_imag, ldb8i, sftB);
    }
    if (op_B == GPUBLAS_OP_N) {
        extract_B8i_kernel_kara<TB><<<n, oz2_const::threads_scaling>>>(k, B, ldb, B8i_real, B8i_imag, ldb8i, sftB);
    }
    if (op_A == GPUBLAS_OP_T) {
        extract_B8i_kernel_kara<TA><<<m, oz2_const::threads_scaling>>>(k, A, lda, A8i_real, A8i_imag, lda8i, sftA);
    }
    if (op_A == GPUBLAS_OP_C) {
        extract_B8i_kernel_kara_conj<TA><<<m, oz2_const::threads_scaling>>>(k, A, lda, A8i_real, B8i_imag, lda8i, sftA);
    }

    // C32i := A8i^T*B8i
    constexpr int32_t one = 1;
    constexpr int32_t zero  = 0;
    constexpr int32_t m_one  = -1;
    gpuDeviceSynchronize();
#if defined(__HIPCC__)
    const size_t m_pad = m;
    const size_t ldc32i = m_pad % 1024 == 0 ? m_pad + 1 : m_pad;
#else
    const size_t m_pad = ((m + 3) >> 2) << 2;
    const size_t ldc32i = m_pad;
#endif

    // F = Im(A)Im(B)
    gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_imag, GPU_R_8I, lda8i, B8i_imag, GPU_R_8I, ldb8i, &zero, C32i_real, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
    // C32i_real = Re(A)Re(B) - F
    gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_real, GPU_R_8I, lda8i, B8i_real, GPU_R_8I, ldb8i, &m_one, C32i_real, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);

    // H = Im(A)Re(B)
    gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_imag, GPU_R_8I, lda8i, B8i_real, GPU_R_8I, ldb8i, &zero, C32i_imag, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
    // C32i_imag = Re(A)Im(B) + H
    gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_real, GPU_R_8I, lda8i, B8i_imag, GPU_R_8I, ldb8i, &one, C32i_imag, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);

    // extract high order bits from A and B
    gpuDeviceSynchronize();
    const float log2M = oz2_table::int8tc::log2M[table_idx]; // fld(log2(M-1)/2 - 0.5)
    if (op_A == GPUBLAS_OP_N) {
        scalingA_kernel_kara<TA><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i_real, C32i_imag, ldc32i, A8i_real, A8i_imag, lda8i, sftA, log2M);
    }
    if (op_B == GPUBLAS_OP_N) {
        scalingB_kernel_kara<TB><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i_real, C32i_imag, ldc32i, B8i_real, B8i_imag, ldb8i, sftB, log2M);
    }
    if (op_A == GPUBLAS_OP_T) {
        scalingAT_kernel_kara<TA><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i_real, C32i_imag, ldc32i, A8i_real, A8i_imag, lda8i, sftA, log2M);
    }
    if (op_B == GPUBLAS_OP_T) {
        scalingBT_kernel_kara<TB><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i_real, C32i_imag, ldc32i, B8i_real, B8i_imag, ldb8i, sftB, log2M);
    }
    if (op_A == GPUBLAS_OP_C) {
        scalingAT_kernel_kara_conj<TA><<<m, oz2_const::threads_scaling>>>(n, k, incA8i, num_moduli, A, lda, C32i_real, C32i_imag, ldc32i, A8i_real, A8i_imag, lda8i, sftA, log2M);
    }
    if (op_B == GPUBLAS_OP_C) {
        scalingBT_kernel_kara_conj<TB><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, C32i_real, C32i_imag, ldc32i, B8i_real, B8i_imag, ldb8i, sftB, log2M);
    }
}

} // namespace int8tc

namespace vecnorm {

template <typename T> __forceinline__ __device__ int compute_sft(T amax, T vecnrm, const float log2M);
template <> __forceinline__ __device__ int compute_sft<double>(double amax, double vecnrm, const float log2M) {
    const int exponent  = ilogb(vecnrm);
    const float vecnrmf = __double2float_ru(scalbn(vecnrm, -exponent));
    const int k         = __float2int_rd(__fmaf_rd(-0.51F, __fadd_ru(__log2f(vecnrmf), exponent), log2M));
    return min(__float2int_rd(log2M - 1.0f), k) - ilogb(amax);
}
template <> __forceinline__ __device__ int compute_sft<float>(float amax, float vecnrm, const float log2M) {
    const int k = __float2int_rd(__fmaf_rd(-0.51F, __log2f(vecnrm), log2M));
    return min(__float2int_rd(log2M - 1.0f), k) - ilogbf(amax);
}

template <typename T>
__global__ void compute_sftA_kernel(const size_t k,               // size(A,2)
                                const T *const __restrict__ A,    // input (lda * n)
                                const size_t lda,                 // leading dimension
                                int16_t *const __restrict__ sftA, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[64];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    T_real vecnrm;
    const T_real amax  = find_amax_and_nrm<T>(in, k, lda, smem, vecnrm);
    const int sft = compute_sft<T_real>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingA_kernel(const size_t k,                   // size(A,2)
                                const size_t incA8i,              // lda8i * m
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ A,    // input (lda * n)
                                const size_t lda,                 // leading dimension
                                int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                const size_t lda8i,               // leading dimension
                                int16_t *const __restrict__ sftA, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    __shared__ T smem[64];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    T vecnrm;
    const T amax  = find_amax_and_nrm<T>(in, k, lda, smem, vecnrm);
    const int sft = compute_sft<T>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }

    scalingA<T>(row_idx, k, incA8i, num_moduli, A, lda, A8i, lda8i, sft);
}

/* Writes into real matrices that encode the complex matrix W as such (+ transposed for scalingA):
 *     W_r  -W_i
 * W ~ W_i   W_r
 *
 * The second column of matrices of the result is only added if addCol = true.
 */
template <typename T, bool addCol>
__global__ void scalingA_kernel_bigmatrix(const size_t m,           // size(A,1)
                                const size_t k,                   // size(A,2)
                                const size_t incA8i,              // lda8i * m
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ A,    // input (lda * n)
                                const size_t lda,                 // leading dimension
                                int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                const size_t lda8i,               // leading dimension
                                int16_t *const __restrict__ sftA, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[64];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    T_real vecnrm;
    const T_real amax  = find_amax_and_nrm<T>(in, k, lda, smem, vecnrm);
    const int sft = compute_sft<T_real>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }

    scalingA_bigmatrix<T, addCol>(row_idx, m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sft);
}

template <typename T>
__global__ void scalingA_kernel_kara(const size_t k,              // size(A,2)
                                const size_t incA8i,              // lda8i * m
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ A,    // input (lda * n)
                                const size_t lda,                 // leading dimension
                                int8_t *const __restrict__ A8i_real,   // output (lda8i * m)
                                int8_t *const __restrict__ A8i_imag,   // output (lda8i * m)
                                const size_t lda8i,               // leading dimension
                                int16_t *const __restrict__ sftA, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[64];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    T_real vecnrm;
    const T_real amax  = find_amax_and_nrm<T>(in, k, lda, smem, vecnrm);
    const int sft = compute_sft<T_real>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }

    scalingA_kara<T>(row_idx, k, incA8i, num_moduli, A, lda, A8i_real, A8i_imag, lda8i, sft);
}

// Same but for applying GPUBLAS_OP_T, so do not exchange imaginary part blocks, just transpose.
template <typename T, bool addCol>
__global__ void scalingA_kernel_bigmatrix_minusTR(const size_t m,           // size(A,1)
                                const size_t k,                   // size(A,2)
                                const size_t incA8i,              // lda8i * m
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ A,    // input (lda * n)
                                const size_t lda,                 // leading dimension
                                int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                const size_t lda8i,               // leading dimension
                                int16_t *const __restrict__ sftA, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[64];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    T_real vecnrm;
    const T_real amax  = find_amax_and_nrm<T>(in, k, lda, smem, vecnrm);
    const int sft = compute_sft<T_real>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }

    scalingA_bigmatrix_minusTR<T, addCol>(row_idx, m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sft);
}

template <typename T>
__global__ void scalingA_kernel_kara_conj(const size_t k,         // size(A,2)
                                const size_t incA8i,              // lda8i * m
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ A,    // input (lda * n)
                                const size_t lda,                 // leading dimension
                                int8_t *const __restrict__ A8i_real,   // output (lda8i * m)
                                int8_t *const __restrict__ A8i_imag,   // output (lda8i * m)
                                const size_t lda8i,               // leading dimension
                                int16_t *const __restrict__ sftA, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[64];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    T_real vecnrm;
    const T_real amax  = find_amax_and_nrm<T>(in, k, lda, smem, vecnrm);
    const int sft = compute_sft<T_real>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }

    scalingA_kara_conj<T>(row_idx, k, incA8i, num_moduli, A, lda, A8i_real, A8i_imag, lda8i, sft);
}

template <typename T>
__global__ void scalingB_kernel(const size_t k,                   // size(B,1)
                                const size_t incB8i,              // ldb8i * n
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ B,    // input (ldb * n)
                                const size_t ldb,                 // leading dimension
                                int8_t *const __restrict__ B8i,   // output (ldb8i * n)
                                const size_t ldb8i,               // leading dimension
                                int16_t *const __restrict__ sftB, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    __shared__ T smem[64];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    T vecnrm;
    const T amax  = find_amax_and_nrm<T>(in, k, 1u, smem, vecnrm);
    const int sft = compute_sft<T>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }

    scalingB<T>(col_idx, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sft);
}

/* Writes into real matrices that encode the complex matrix W as such:
 *     W_r  -W_i
 * W ~ W_i   W_r
 */
template <typename T, bool addCol>
__global__ void scalingB_kernel_bigmatrix(const size_t k,           // size(B,1)
                                const size_t n,                   // size(B,2)
                                const size_t incB8i,              // ldb8i * n
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ B,    // input (ldb * n)
                                const size_t ldb,                 // leading dimension
                                int8_t *const __restrict__ B8i,   // output (ldb8i * n)
                                const size_t ldb8i,               // leading dimension
                                int16_t *const __restrict__ sftB, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[64];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    T_real vecnrm;
    const T_real amax  = find_amax_and_nrm<T>(in, k, 1u, smem, vecnrm);
    const int sft = compute_sft<T_real>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }

    scalingB_bigmatrix<T, addCol>(col_idx, k, n, incB8i, num_moduli, B, ldb, B8i, ldb8i, sft);
}

template <typename T>
__global__ void scalingB_kernel_kara(const size_t k,           // size(B,1)
                                const size_t incB8i,              // ldb8i * n
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ B,    // input (ldb * n)
                                const size_t ldb,                 // leading dimension
                                int8_t *const __restrict__ B8i_real,   // output (ldb8i * n)
                                int8_t *const __restrict__ B8i_imag,   // output (ldb8i * n)
                                const size_t ldb8i,               // leading dimension
                                int16_t *const __restrict__ sftB, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[64];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    T_real vecnrm;
    const T_real amax  = find_amax_and_nrm<T>(in, k, 1u, smem, vecnrm);
    const int sft = compute_sft<T_real>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }

    scalingB_kara<T>(col_idx, k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sft);
}

template <typename T, bool addCol>
__global__ void scalingB_kernel_bigmatrix_minusBL(const size_t k,           // size(B,1)
                                const size_t n,                   // size(B,2)
                                const size_t incB8i,              // ldb8i * n
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ B,    // input (ldb * n)
                                const size_t ldb,                 // leading dimension
                                int8_t *const __restrict__ B8i,   // output (ldb8i * n)
                                const size_t ldb8i,               // leading dimension
                                int16_t *const __restrict__ sftB, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[64];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    T_real vecnrm;
    const T_real amax  = find_amax_and_nrm<T>(in, k, 1u, smem, vecnrm);
    const int sft = compute_sft<T_real>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }

    scalingB_bigmatrix_minusBL<T, addCol>(col_idx, k, n, incB8i, num_moduli, B, ldb, B8i, ldb8i, sft);
}

template <typename T>
__global__ void scalingB_kernel_kara_conj(const size_t k,           // size(B,1)
                                const size_t incB8i,              // ldb8i * n
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ B,    // input (ldb * n)
                                const size_t ldb,                 // leading dimension
                                int8_t *const __restrict__ B8i_real,   // output (ldb8i * n)
                                int8_t *const __restrict__ B8i_imag,   // output (ldb8i * n)
                                const size_t ldb8i,               // leading dimension
                                int16_t *const __restrict__ sftB, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    using T_real = typename oz2_type_utils::real_type<T>::type;

    __shared__ T_real smem[64];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    T_real vecnrm;
    const T_real amax  = find_amax_and_nrm<T>(in, k, 1u, smem, vecnrm);
    const int sft = compute_sft<T_real>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }

    scalingB_kara_conj<T>(col_idx, k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sft);
}

template <typename TA, typename TB>
__inline__ void scaling(const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const size_t m,                // size(A,1) & size(C,1)
                        const size_t n,                // size(B,2) & size(C,2)
                        const size_t k,                // size(A,2) & size(B,1)
                        const unsigned num_moduli,     // #moduli
                        const TA *const A,             // input
                        const size_t lda,              // leading dimension
                        const TB *const B,             // input
                        const size_t ldb,              // leading dimension
                        int8_t *const A8i,             // output (k * m)
                        const size_t lda8i,            // leading dimension
                        const size_t incA8i,           // increment between the A8i
                        int16_t *const sftA,           // exponent of shift values for rows of A
                        int8_t *const B8i,             // output (k * n)
                        const size_t ldb8i,            // leading dimension
                        const size_t incB8i,           // increment between the B8i
                        int16_t *const sftB,           // exponent of shift values for cols of B
                        const unsigned table_idx)      //
{
    const float log2M = oz2_table::vecnorm::log2M[table_idx]; // fld(log2(M-1)/2 - 1.5)
    if (op_A == GPUBLAS_OP_N) {
#if defined(__HIPCC__)
        if (lda % 1024 == 0) {  // If lda multiple of 1024 on AMD, copy to mitigate channel conflicts.
            stair_kernel<TA><<<k, oz2_const::threads_scaling>>>(m, A, lda, reinterpret_cast<TA *>(A8i), lda + 1);
            compute_sftA_kernel<TA><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TA *>(A8i), lda + 1, sftA, log2M);
            dim3 grid((m + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            scalingA_kernel_separated<TA><<<grid, threads_scalingA>>>(m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA);
        }
        else
#endif
            scalingA_kernel<TA><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
    }
    if (op_B != GPUBLAS_OP_N) {
#if defined(__HIPCC__)
        if (ldb % 1024 == 0) {  // If ldb multiple of 1024 on AMD, copy to mitigate channel conflicts.
            stair_kernel<TB><<<k, oz2_const::threads_scaling>>>(n, B, ldb, reinterpret_cast<TB *>(B8i), ldb + 1);
            compute_sftA_kernel<TB><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TB *>(B8i), ldb + 1, sftB, log2M);
            dim3 grid((n + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            scalingA_kernel_separated<TB><<<grid, threads_scalingA>>>(n, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB);
        }
        else
#endif
            scalingA_kernel<TB><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
    }
    if (op_A != GPUBLAS_OP_N) {
        scalingB_kernel<TA><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
    }
    if (op_B == GPUBLAS_OP_N) {
        scalingB_kernel<TB><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
    }
}

template <typename TA, typename TB>
__inline__ void scaling_bigmatrix(const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const size_t m,                // size(A,1) & size(C,1)
                        const size_t n,                // size(B,2) & size(C,2)
                        const size_t k,                // size(A,2) & size(B,1)
                        const unsigned num_moduli,     // #moduli
                        const TA *const A,             // input
                        const size_t lda,              // leading dimension
                        const TB *const B,             // input
                        const size_t ldb,              // leading dimension
                        int8_t *const A8i,             // output (k * m)
                        const size_t lda8i,            // leading dimension
                        const size_t incA8i,           // increment between the A8i
                        int16_t *const sftA,           // exponent of shift values for rows of A
                        int8_t *const B8i,             // output (k * n)
                        const size_t ldb8i,            // leading dimension
                        const size_t incB8i,           // increment between the B8i
                        int16_t *const sftB,           // exponent of shift values for cols of B
                        const unsigned table_idx)      //
{
    const float log2M = oz2_table::vecnorm::log2M[table_idx]; // fld(log2(M-1)/2 - 1.5)
    if (op_A == GPUBLAS_OP_N) {
#if defined(__HIPCC__)
        if (lda % 1024 == 0) {  // If lda multiple of 1024 on AMD, copy to mitigate channel conflicts.
            stair_kernel<TA><<<k, oz2_const::threads_scaling>>>(m, A, lda, reinterpret_cast<TA *>(A8i), lda + 1);
            compute_sftA_kernel<TA><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TA *>(A8i), lda + 1, sftA, log2M);
            dim3 grid((m + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            scalingA_kernel_separated_bigmatrix<TA, true><<<grid, threads_scalingA>>>(m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA);
        }
        else
#endif
            scalingA_kernel_bigmatrix<TA, true><<<m, oz2_const::threads_scaling>>>(m, k, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
    }
    if (op_B == GPUBLAS_OP_T) {
#if defined(__HIPCC__)
        if (ldb % 1024 == 0) {  // If ldb multiple of 1024 on AMD, copy to mitigate channel conflicts.
            stair_kernel<TB><<<k, oz2_const::threads_scaling>>>(n, B, ldb, reinterpret_cast<TB *>(B8i), ldb + 1);
            compute_sftA_kernel<TB><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TB *>(B8i), ldb + 1, sftB, log2M);
            dim3 grid((n + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            scalingA_kernel_separated_bigmatrix_minusTR<TB, false><<<grid, threads_scalingA>>>(n, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB);
        }
        else
#endif
            scalingA_kernel_bigmatrix_minusTR<TB, false><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
    }
    if (op_B == GPUBLAS_OP_C) {
#if defined(__HIPCC__)
        if (ldb % 1024 == 0) {  // If ldb multiple of 1024 on AMD, copy to mitigate channel conflicts.
            stair_kernel<TB><<<k, oz2_const::threads_scaling>>>(n, B, ldb, reinterpret_cast<TB *>(B8i), ldb + 1);
            compute_sftA_kernel<TB><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TB *>(B8i), ldb + 1, sftB, log2M);
            dim3 grid((n + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            scalingA_kernel_separated_bigmatrix<TB, false><<<grid, threads_scalingA>>>(n, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB);
        }
        else
#endif
            scalingA_kernel_bigmatrix<TB, false><<<n, oz2_const::threads_scaling>>>(m, k, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
    }
    if (op_B == GPUBLAS_OP_N) {
        scalingB_kernel_bigmatrix<TB, false><<<n, oz2_const::threads_scaling>>>(k, n, incB8i, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
    }
    if (op_A == GPUBLAS_OP_T) {
        scalingB_kernel_bigmatrix_minusBL<TA, true><<<m, oz2_const::threads_scaling>>>(k, m, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
    }
    if (op_A == GPUBLAS_OP_C) {
        scalingB_kernel_bigmatrix<TA, true><<<m, oz2_const::threads_scaling>>>(k, m, incA8i, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
    }
}

template <typename TA, typename TB>
__inline__ void scaling_kara(const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                        const size_t m,                // size(A,1) & size(C,1)
                        const size_t n,                // size(B,2) & size(C,2)
                        const size_t k,                // size(A,2) & size(B,1)
                        const unsigned num_moduli,     // #moduli
                        const TA *const A,             // input
                        const size_t lda,              // leading dimension
                        const TB *const B,             // input
                        const size_t ldb,              // leading dimension
                        int8_t *const A8i_real,             // output (k * m)
                        int8_t *const A8i_imag,             // output (k * m)
                        const size_t lda8i,            // leading dimension
                        const size_t incA8i,           // increment between the A8i
                        int16_t *const sftA,           // exponent of shift values for rows of A
                        int8_t *const B8i_real,             // output (k * n)
                        int8_t *const B8i_imag,             // output (k * n)
                        const size_t ldb8i,            // leading dimension
                        const size_t incB8i,           // increment between the B8i
                        int16_t *const sftB,           // exponent of shift values for cols of B
                        const unsigned table_idx)      //
{
    const float log2M = oz2_table::vecnorm::log2M[table_idx]; // fld(log2(M-1)/2 - 1.5)
    if (op_A == GPUBLAS_OP_N) {
#if defined(__HIPCC__)
        if (lda % 1024 == 0) {  // If lda multiple of 1024 on AMD, copy to mitigate channel conflicts.
            stair_kernel<TA><<<k, oz2_const::threads_scaling>>>(m, A, lda, reinterpret_cast<TA *>(A8i_real), lda + 1);
            compute_sftA_kernel<TA><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TA *>(A8i_real), lda + 1, sftA, log2M);
            dim3 grid((m + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            scalingA_kernel_separated_kara<TA><<<grid, threads_scalingA>>>(m, k, incA8i, num_moduli, A, lda, A8i_real, A8i_imag, lda8i, sftA);
        }
        else
#endif
            scalingA_kernel_kara<TA><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i_real, A8i_imag, lda8i, sftA, log2M);
    }
    if (op_B == GPUBLAS_OP_T) {
#if defined(__HIPCC__)
        if (ldb % 1024 == 0) {  // If ldb multiple of 1024 on AMD, copy to mitigate channel conflicts.
            stair_kernel<TB><<<k, oz2_const::threads_scaling>>>(n, B, ldb, reinterpret_cast<TB *>(B8i_real), ldb + 1);
            compute_sftA_kernel<TB><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TB *>(B8i_real), ldb + 1, sftB, log2M);
            dim3 grid((n + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            scalingA_kernel_separated_kara<TB><<<grid, threads_scalingA>>>(n, k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sftB);
        }
        else
#endif
            scalingA_kernel_kara<TB><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sftB, log2M);
    }
    if (op_B == GPUBLAS_OP_C) {
#if defined(__HIPCC__)
        if (ldb % 1024 == 0) {  // If ldb multiple of 1024 on AMD, copy to mitigate channel conflicts.
            stair_kernel<TB><<<k, oz2_const::threads_scaling>>>(n, B, ldb, reinterpret_cast<TB *>(B8i_real), ldb + 1);
            compute_sftA_kernel<TB><<<m, oz2_const::threads_scaling>>>(k, reinterpret_cast<TB *>(B8i_real), ldb + 1, sftB, log2M);
            dim3 grid((n + TILE_DIM-1) / TILE_DIM, (k + TILE_DIM-1) / TILE_DIM);
            dim3 threads_scalingA(TILE_DIM, NY);
            scalingA_kernel_separated_kara_conj<TB><<<grid, threads_scalingA>>>(n, k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sftB);
        }
        else
#endif
            scalingA_kernel_kara_conj<TB><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sftB, log2M);
    }
    if (op_B == GPUBLAS_OP_N) {
        scalingB_kernel_kara<TB><<<n, oz2_const::threads_scaling>>>(k, incB8i, num_moduli, B, ldb, B8i_real, B8i_imag, ldb8i, sftB, log2M);
    }
    if (op_A == GPUBLAS_OP_T) {
        scalingB_kernel_kara<TA><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i_real, A8i_imag, lda8i, sftA, log2M);
    }
    if (op_A == GPUBLAS_OP_C) {
        scalingB_kernel_kara_conj<TA><<<m, oz2_const::threads_scaling>>>(k, incA8i, num_moduli, A, lda, A8i_real, A8i_imag, lda8i, sftA, log2M);
    }
}

} // namespace vecnorm

} // namespace oz2_util
