#pragma once
#include "common.hpp"
#include "table.hpp"

namespace oz2_util {

namespace {

template <typename T> __forceinline__ __device__ T Tcast(double in);
template <> __forceinline__ __device__ double Tcast<double>(double in) { return in; };
template <> __forceinline__ __device__ float Tcast<float>(double in) { return __double2float_rn(in); };

template <typename T> __forceinline__ __device__ T Tcast(gpuDoubleComplex in);
template <> __forceinline__ __device__ gpuDoubleComplex Tcast<gpuDoubleComplex>(gpuDoubleComplex in) { return in; };
template <> __forceinline__ __device__ gpuFloatComplex Tcast<gpuFloatComplex>(gpuDoubleComplex in) {
    return make_gpuFloatComplex(__double2float_rn(gpuCreal(in)), __double2float_rn(gpuCimag(in)));
};

template <typename T> __forceinline__ __device__ T Tfma(const T in1, T in2, T in3);
template <> __forceinline__ __device__ double Tfma<double>(const double in1, double in2, double in3) {
    return fma(in1, in2, in3);
};
template <> __forceinline__ __device__ float Tfma<float>(const float in1, float in2, float in3) {
    return __fmaf_rn(in1, in2, in3);
};
template <> __forceinline__ __device__ gpuFloatComplex Tfma<gpuFloatComplex>(const gpuFloatComplex in1, gpuFloatComplex in2, gpuFloatComplex in3) {
    /*return make_gpuFloatComplex(__fadd_rn(__fmaf_rn(in1.x, in2.y, in3.x), - __fmul_rn(in1.y, in2.x)),
                                __fadd_rn(__fmaf_rn(in1.y, in2.x, in3.y), - __fmul_rn(in1.x, in2.y)));*/
    return gpuCfmaf(in1, in2, in3);
};
template <> __forceinline__ __device__ gpuDoubleComplex Tfma<gpuDoubleComplex>(const gpuDoubleComplex in1, gpuDoubleComplex in2, gpuDoubleComplex in3) {
    return gpuCfma(in1, in2, in3);
};

template <typename T>
__forceinline__ __device__ T inverse_scaling_1_base(
                                     const size_t col,
                                     const size_t row,
                                     const unsigned num_moduli,
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto mem_idx = col * ldc8u + row;

    double C64f = 0.0;
    for (unsigned i = 0; i < num_moduli; ++i) {
        const double C8u_tmp = __uint2double_rn(0u + C8u[i * incC8u + mem_idx]); // __uint2double_rn(static_cast<uint32_t>(C8u[i * incC8u + idx]));
        const double NMi     = oz2_table::NMi_dev[i];                        // constant memory
        C64f                 = fma(NMi, C8u_tmp, C64f);                      // accumulation
    }

    const double quot = -rint(C64f * invM);
    double tmpC       = fma(quot, M, C64f);
    tmpC              = scalbn(tmpC, sftA[row] + sftB[col]);

    return Tcast<T>(tmpC);
}

template <typename T>
__forceinline__ __device__ T inverse_scaling_1_base_C(
                                     const size_t col,
                                     const size_t row,
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto mem_idx = col * ldc8u + row;

    double C64f_real = 0.0;
    double C64f_imag = 0.0;
    for (unsigned i = 0; i < num_moduli; ++i) {
        const double C8u_tmp_real = __uint2double_rn(0u + C8u[i * incC8u + mem_idx]); // __uint2double_rn(static_cast<uint32_t>(C8u[i * incC8u + idx]));
        const double C8u_tmp_imag = __uint2double_rn(0u + C8u[i * incC8u + mem_idx + m]); // __uint2double_rn(static_cast<uint32_t>(C8u[i * incC8u + idx]));
        const double NMi     = oz2_table::NMi_dev[i];                        // constant memory
        C64f_real                 = fma(NMi, C8u_tmp_real, C64f_real);                      // error-free
        C64f_imag                 = fma(NMi, C8u_tmp_imag, C64f_imag);
    }

    const double quot_real = -rint(C64f_real * invM);
    double tmpC_real       = fma(quot_real, M, C64f_real);
    tmpC_real              = scalbn(tmpC_real, sftA[row] + sftB[col]);

    const double quot_imag = -rint(C64f_imag * invM);
    double tmpC_imag       = fma(quot_imag, M, C64f_imag);
    tmpC_imag              = scalbn(tmpC_imag, sftA[row] + sftB[col]);

    return Tcast<T>(make_gpuDoubleComplex(tmpC_real, tmpC_imag));
}

// C := C64f - round(C64f1/M1 + C64f2/M1)*(M1 + M2)
// C := diag(2^sftA) * C * diag(2^sftB)
template <typename T>
__forceinline__ __device__ T inverse_scaling_2_base(
                                     const size_t col,
                                     const size_t row,
                                     const unsigned num_moduli,
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto mem_idx = col * ldc8u + row;

    double C64f1 = 0.0;
    double C64f2 = 0.0;
    for (unsigned i = 0; i < num_moduli; ++i) {
        const double C8u_tmp = __uint2double_rn(0u + C8u[i * incC8u + mem_idx]); // __uint2double_rn(static_cast<uint32_t>(C8u[i * incC8u + idx]));
        const double NMi1    = oz2_table::NMi_dev[i * 2];                    // constant memory
        const double NMi2    = oz2_table::NMi_dev[i * 2 + 1];                // constant memory
        C64f1                = fma(NMi1, C8u_tmp, C64f1);                    // error-free
        C64f2                = fma(NMi2, C8u_tmp, C64f2);                    // not error-free
    }

    const double quot  = -rint(fma(C64f1, invM, C64f2 * invM));
    const double tmpC1 = fma(quot, M1, C64f1) + C64f2;
    double tmpC2       = fma(quot, M2, tmpC1);
    tmpC2              = scalbn(tmpC2, sftA[row] + sftB[col]);

    return Tcast<T>(tmpC2);
}

template <typename T>
__forceinline__ __device__ T inverse_scaling_2_base_C(
                                     const size_t col,
                                     const size_t row,
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto mem_idx = col * ldc8u + row;

    double C64f1_real = 0.0;
    double C64f2_real = 0.0;
    double C64f1_imag = 0.0;
    double C64f2_imag = 0.0;
    for (unsigned i = 0; i < num_moduli; ++i) {
        const double C8u_tmp_real = __uint2double_rn(0u + C8u[i * incC8u + mem_idx]); // __uint2double_rn(static_cast<uint32_t>(C8u[i * incC8u + idx]));
        const double C8u_tmp_imag = __uint2double_rn(0u + C8u[i * incC8u + mem_idx + m]);
        const double NMi1    = oz2_table::NMi_dev[i * 2];                    // constant memory
        const double NMi2    = oz2_table::NMi_dev[i * 2 + 1];                // constant memory
        C64f1_real                = fma(NMi1, C8u_tmp_real, C64f1_real);                    // error-free
        C64f2_real                = fma(NMi2, C8u_tmp_real, C64f2_real);                    // not error-free
        C64f1_imag                = fma(NMi1, C8u_tmp_imag, C64f1_imag);                    // error-free
        C64f2_imag                = fma(NMi2, C8u_tmp_imag, C64f2_imag);                    // not error-free
    }

    const double quot_real  = -rint(fma(C64f1_real, invM, C64f2_real * invM));
    const double tmpC1_real = fma(quot_real, M1, C64f1_real) + C64f2_real;
    double tmpC2_real       = fma(quot_real, M2, tmpC1_real);
    tmpC2_real              = scalbn(tmpC2_real, sftA[row] + sftB[col]);

    const double quot_imag  = -rint(fma(C64f1_imag, invM, C64f2_imag * invM));
    const double tmpC1_imag = fma(quot_imag, M1, C64f1_imag) + C64f2_imag;
    double tmpC2_imag       = fma(quot_imag, M2, tmpC1_imag);
    tmpC2_imag              = scalbn(tmpC2_imag, sftA[row] + sftB[col]);

    return Tcast<T>(make_gpuDoubleComplex(tmpC2_real, tmpC2_imag));
}

} // namespace

// C := C64f - round(C64f/M)*M
// C := diag(2^sftA) * C * diag(2^sftB)
template <typename T>
__global__ void inverse_scaling_1_10(const unsigned num_moduli,
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC = inverse_scaling_1_base<T>(col, row, num_moduli, incC8u, C8u, ldc8u, invM, M, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = tmpC;
}

template <typename T>
__global__ void inverse_scaling_1_10_C(const unsigned num_moduli,
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC = inverse_scaling_1_base_C<T>(col, row, num_moduli, m, incC8u, C8u, ldc8u, invM, M, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = tmpC;
}

template <typename T>
__global__ void inverse_scaling_1_11(const unsigned num_moduli,
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC = inverse_scaling_1_base<T>(col, row, num_moduli, incC8u, C8u, ldc8u, invM, M, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC] += tmpC;
}

template <typename T>
__global__ void inverse_scaling_1_11_C(const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC = inverse_scaling_1_base_C<T>(col, row, num_moduli, m, incC8u, C8u, ldc8u, invM, M, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = oz2_type_utils::CAdd(C[idxC], tmpC);
}

template <typename T>
__global__ void inverse_scaling_1_1b(const T beta,                           //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC = inverse_scaling_1_base<T>(col, row, num_moduli, incC8u, C8u, ldc8u, invM, M, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(beta, tmpC, C[idxC]);
}

template <typename T>
__global__ void inverse_scaling_1_1b_C(const T beta,                           //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC = inverse_scaling_1_base_C<T>(col, row, num_moduli, m, incC8u, C8u, ldc8u, invM, M, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(beta, tmpC, C[idxC]);
}

template <typename T>
__global__ void inverse_scaling_1_a1(const T alpha,                          //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC = inverse_scaling_1_base<T>(col, row, num_moduli, incC8u, C8u, ldc8u, invM, M, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(alpha, tmpC, C[idxC]);
}

template <typename T>
__global__ void inverse_scaling_1_a1_C(const T alpha,                          //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC = inverse_scaling_1_base_C<T>(col, row, num_moduli, m, incC8u, C8u, ldc8u, invM, M, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(alpha, tmpC, C[idxC]);
}

template <typename T>
__global__ void inverse_scaling_1_ab(const T alpha,                          //
                                     const T beta,                           //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC = inverse_scaling_1_base<T>(col, row, num_moduli, incC8u, C8u, ldc8u, invM, M, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(beta, C[idxC], alpha * tmpC);
}

template <typename T>
__global__ void inverse_scaling_1_ab_C(const T alpha,                          //
                                     const T beta,                           //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M,                         //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC = inverse_scaling_1_base_C<T>(col, row, num_moduli, m, incC8u, C8u, ldc8u, invM, M, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(beta, C[idxC], oz2_type_utils::CMul(alpha, tmpC));
}

// C := C64f - round(C64f1/M1 + C64f2/M1)*(M1 + M2)
// C := diag(2^sftA) * C * diag(2^sftB)
template <typename T>
__global__ void inverse_scaling_2_10(const unsigned num_moduli,
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC2 = inverse_scaling_2_base<T>(col, row, num_moduli, incC8u, C8u, ldc8u, invM, M1, M2, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = tmpC2;
}

template <typename T>
__global__ void inverse_scaling_2_10_C(const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC2 = inverse_scaling_2_base_C<T>(col, row, num_moduli, m, incC8u, C8u, ldc8u, invM, M1, M2, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = tmpC2;
}

template <typename T>
__global__ void inverse_scaling_2_11(const unsigned num_moduli,
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC2 = inverse_scaling_2_base<T>(col, row, num_moduli, incC8u, C8u, ldc8u, invM, M1, M2, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC] += tmpC2;
}

template <typename T>
__global__ void inverse_scaling_2_11_C(const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC2 = inverse_scaling_2_base_C<T>(col, row, num_moduli, m, incC8u, C8u, ldc8u, invM, M1, M2, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC] = oz2_type_utils::CAdd<T>(tmpC2, C[idxC]);
}

template <typename T>
__global__ void inverse_scaling_2_1b(const T beta,                           //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC2 = inverse_scaling_2_base<T>(col, row, num_moduli, incC8u, C8u, ldc8u, invM, M1, M2, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(beta, tmpC2, C[idxC]);
}

template <typename T>
__global__ void inverse_scaling_2_1b_C(const T beta,                           //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC2 = inverse_scaling_2_base_C<T>(col, row, num_moduli, m, incC8u, C8u, ldc8u, invM, M1, M2, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(beta, tmpC2, C[idxC]);
}

template <typename T>
__global__ void inverse_scaling_2_a1(const T alpha,                          //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC2 = inverse_scaling_2_base<T>(col, row, num_moduli, incC8u, C8u, ldc8u, invM, M1, M2, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(alpha, C[idxC], tmpC2);
}

template <typename T>
__global__ void inverse_scaling_2_a1_C(const T alpha,                          //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC2 = inverse_scaling_2_base_C<T>(col, row, num_moduli, m, incC8u, C8u, ldc8u, invM, M1, M2, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(alpha, C[idxC], tmpC2);
}

template <typename T>
__global__ void inverse_scaling_2_ab(const T alpha,                          //
                                     const T beta,                           //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC2 = inverse_scaling_2_base<T>(col, row, num_moduli, incC8u, C8u, ldc8u, invM, M1, M2, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(beta, C[idxC], alpha * tmpC2);
}

template <typename T>
__global__ void inverse_scaling_2_ab_C(const T alpha,                          //
                                     const T beta,                           //
                                     const unsigned num_moduli,              //
                                     const size_t m,                         // size(C64f,1)
                                     const size_t sizeC,                     //
                                     const size_t incC8u,                    //
                                     const uint8_t *const __restrict__ C8u,  // input
                                     const size_t ldc8u,                     // leading dim of C8u
                                     T *const __restrict__ C,                // output
                                     const size_t ldc,                       // leading dimension
                                     const double invM,                      //
                                     const double M1,                        //
                                     const double M2,                        //
                                     const int16_t *const __restrict__ sftA, // exponent of shift values for rows of A
                                     const int16_t *const __restrict__ sftB) // exponent of shift values for cols of B
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    const auto col = idx / m;
    const auto row = idx - col * m;

    const T tmpC2 = inverse_scaling_2_base_C<T>(col, row, num_moduli, m, incC8u, C8u, ldc8u, invM, M1, M2, sftA, sftB);

    const auto idxC = col * ldc + row;
    C[idxC]         = Tfma<T>(beta, C[idxC], oz2_type_utils::CMul(alpha, tmpC2));
}

// interface!!
template <typename T>
__inline__ void inverse_scaling(const unsigned num_moduli,
                                const size_t m,            // size(C,1)
                                const size_t n,            // size(C,2)
                                const uint8_t *const C8u,  // input
                                const size_t ldc8u,        // leading dim of C8u
                                const size_t incC8u,       //
                                T *const C,                // output
                                const size_t ldc,          // leading dimension
                                const int16_t *const sftA, // exponent of shift values for rows of A
                                const int16_t *const sftB, // exponent of shift values for cols of B
                                const T alpha,             //
                                const T beta)              //
{
    const unsigned table_idx = num_moduli - 2;
    const size_t sizeC       = m * n;
    const double invM        = oz2_table::invM[table_idx];
    const double M           = oz2_table::M[table_idx][0];
    if (alpha == 1.0F) {
        if (beta == 0.0F) {
            inverse_scaling_1_10<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
        } else if (beta == 1.0F) {
            inverse_scaling_1_11<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
        } else {
            inverse_scaling_1_1b<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
        }
    } else {
        if (beta == 1.0F) {
            inverse_scaling_1_a1<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
        } else {
            inverse_scaling_1_ab<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
        }
    }
}

template <typename T>
__inline__ void inverse_scaling_C(const unsigned num_moduli,
                                const size_t m,            // size(C,1)
                                const size_t n,            // size(C,2)
                                const uint8_t *const C8u,  // input
                                const size_t ldc8u,        // leading dim of C8u
                                const size_t incC8u,       //
                                T *const C,                // output
                                const size_t ldc,          // leading dimension
                                const int16_t *const sftA, // exponent of shift values for rows of A
                                const int16_t *const sftB, // exponent of shift values for cols of B
                                const T alpha,             //
                                const T beta)              //
{
    const unsigned table_idx = num_moduli - 2;
    const size_t sizeC       = m * n;
    const double invM        = oz2_table::invM[table_idx];
    const double M           = oz2_table::M[table_idx][0];
    if (oz2_type_utils::Creal(alpha) == 1.0F && oz2_type_utils::Cimag(alpha) == 0.0F) {
        if (oz2_type_utils::Creal(beta) == 0.0F && oz2_type_utils::Cimag(beta) == 0.0F) {
            inverse_scaling_1_10_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
        } else if (oz2_type_utils::Creal(beta) == 1.0F && oz2_type_utils::Cimag(beta) == 0.0F) {
            inverse_scaling_1_11_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
        } else {
            inverse_scaling_1_1b_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
        }
    } else {
        if (oz2_type_utils::Creal(beta) == 1.0F && oz2_type_utils::Cimag(beta) == 0.0F) {
            inverse_scaling_1_a1_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
        } else {
            inverse_scaling_1_ab_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
        }
    }
}

template <typename T>
__inline__ void inverse_scaling(const bool is_numM_1,
                                const unsigned num_moduli,
                                const size_t m,            // size(C,1)
                                const size_t n,            // size(C,2)
                                const uint8_t *const C8u,  // input
                                const size_t ldc8u,        // leading dim of C8u
                                const size_t incC8u,       //
                                T *const C,                // output
                                const size_t ldc,          // leading dimension
                                const int16_t *const sftA, // exponent of shift values for rows of A
                                const int16_t *const sftB, // exponent of shift values for cols of B
                                const T alpha,             //
                                const T beta)              //
{
    const unsigned table_idx = num_moduli - 2;
    const size_t sizeC       = m * n;
    if (is_numM_1) {
        const double invM = oz2_table::invM[table_idx];
        const double M    = oz2_table::M[table_idx][0];
        if (alpha == 1.0) {
            if (beta == 0.0) {
                inverse_scaling_1_10<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
            } else if (beta == 1.0) {
                inverse_scaling_1_11<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
            } else {
                inverse_scaling_1_1b<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
            }
        } else {
            if (beta == 1.0) {
                inverse_scaling_1_a1<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
            } else {
                inverse_scaling_1_ab<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
            }
        }
    } else {
        const double invM = oz2_table::invM[table_idx];
        const double M1   = oz2_table::M[table_idx][0];
        const double M2   = oz2_table::M[table_idx][1];
        if (alpha == 1.0) {
            if (beta == 0.0) {
                inverse_scaling_2_10<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M1, M2, sftA, sftB);
            } else if (beta == 1) {
                inverse_scaling_2_11<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M1, M2, sftA, sftB);
            } else {
                inverse_scaling_2_1b<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M1, M2, sftA, sftB);
            }
        } else {
            if (beta == 1.0) {
                inverse_scaling_2_a1<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M1, M2, sftA, sftB);
            } else {
                inverse_scaling_2_ab<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M1, M2, sftA, sftB);
            }
        }
    }
}

template <typename T>
__inline__ void inverse_scaling_C(const bool is_numM_1,
                                const unsigned num_moduli,
                                const size_t m,            // size(C,1)
                                const size_t n,            // size(C,2)
                                const uint8_t *const C8u,  // input
                                const size_t ldc8u,        // leading dim of C8u
                                const size_t incC8u,       //
                                T *const C,                // output
                                const size_t ldc,          // leading dimension
                                const int16_t *const sftA, // exponent of shift values for rows of A
                                const int16_t *const sftB, // exponent of shift values for cols of B
                                const T alpha,             //
                                const T beta)              //
{
    const unsigned table_idx = num_moduli - 2;
    const size_t sizeC       = m * n;
    if (is_numM_1) {
        const double invM = oz2_table::invM[table_idx];
        const double M    = oz2_table::M[table_idx][0];
        if (oz2_type_utils::Creal(alpha) == 1.0 && oz2_type_utils::Cimag(alpha) == 0.0) {
            if (oz2_type_utils::Creal(beta) == 0.0 && oz2_type_utils::Cimag(beta) == 0.0) {
                inverse_scaling_1_10_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
            } else if (oz2_type_utils::Creal(beta) == 1.0 && oz2_type_utils::Cimag(beta) == 0.0) {
                inverse_scaling_1_11_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
            } else {
                inverse_scaling_1_1b_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
            }
        } else {
            if (oz2_type_utils::Creal(beta) == 1.0 && oz2_type_utils::Cimag(beta) == 0.0) {
                inverse_scaling_1_a1_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
            } else {
                inverse_scaling_1_ab_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M, sftA, sftB);
            }
        }
    } else {
        const double invM = oz2_table::invM[table_idx];
        const double M1   = oz2_table::M[table_idx][0];
        const double M2   = oz2_table::M[table_idx][1];
        if (oz2_type_utils::Creal(alpha) == 1.0 && oz2_type_utils::Cimag(alpha) == 0.0) {
            if (oz2_type_utils::Creal(beta) == 0.0 && oz2_type_utils::Cimag(beta) == 0.0) {
                inverse_scaling_2_10_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M1, M2, sftA, sftB);
            } else if (oz2_type_utils::Creal(beta) == 1.0 && oz2_type_utils::Cimag(beta) == 0.0) {
                inverse_scaling_2_11_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M1, M2, sftA, sftB);
            } else {
                inverse_scaling_2_1b_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M1, M2, sftA, sftB);
            }
        } else {
            if (oz2_type_utils::Creal(beta) == 1.0 && oz2_type_utils::Cimag(beta) == 0.0) {
                inverse_scaling_2_a1_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M1, M2, sftA, sftB);
            } else {
                inverse_scaling_2_ab_C<T><<<oz2_const::grids_invscaling, oz2_const::threads_invscaling>>>(alpha, beta, num_moduli, m, sizeC, incC8u, C8u, ldc8u, C, ldc, invM, M1, M2, sftA, sftB);
            }
        }
    }
}

} // namespace oz2_util
