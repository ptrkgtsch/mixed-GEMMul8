#include "../include/gemmul8.hpp"
#include "common.hpp"
#include "conv_32i_2_8u.hpp"
#include "inverse_scaling.hpp"
#include "scaling.hpp"
#include "table.hpp"
#include "mat_utils.hpp"

namespace {
void timing_start(std::chrono::system_clock::time_point &timetmp) {
    gpuDeviceSynchronize();
    timetmp = std::chrono::system_clock::now();
}

void timing_stop(std::chrono::system_clock::time_point &timetmp, double &timer) {
    gpuDeviceSynchronize();
    timer += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - timetmp).count();
}

} // namespace

namespace gemmul8 {

//------------------------------
// Calculating required work size
//------------------------------
size_t workSize_real(const size_t m,            // size(A,1) & size(C,1) <= 2^17
                const size_t n,            // size(B,2) & size(C,2) <= 2^17
                const size_t k,            // size(A,2) & size(B,1) <= 2^17
                const unsigned num_moduli) // 2 <= num_moduli <= 20
{
#if defined(__HIPCC__)
    const size_t k16       = ((k + 15) >> 4) << 4;
    const size_t lda8i     =  k16 % 1024 == 0 ? k16 + 64 : k16;
    const size_t m_pad     = m;
#else
    const size_t lda8i     = ((k + 15) >> 4) << 4;
    const size_t m_pad     = ((m + 3) >> 2) << 2;
#endif
    const size_t ldb8i     = lda8i;
    const size_t sizeA     = lda8i * m_pad;
    const size_t sizeB     = ldb8i * n;
    const size_t sizeC     = ((m_pad * n + 15) >> 4) << 4;
#if defined(__HIPCC__)
    const size_t sizeC32i  = m % 1024 == 0 ? (((m_pad+1) * n + 15) >> 4) << 4 : sizeC;
#else
    const size_t sizeC32i  = sizeC;
#endif
    const size_t size_vecA = (((m + 15) >> 4) << 4); // multiple of 16
    const size_t size_vecB = (((n + 15) >> 4) << 4); // multiple of 16

    size_t total_size = 0;
    total_size += sizeof(int8_t) * (sizeA + sizeB) * num_moduli;
    total_size += sizeof(uint8_t) * sizeC * num_moduli;
    total_size += sizeof(int32_t) * sizeC32i;
    total_size += sizeof(int16_t) * (size_vecA + size_vecB);

    return total_size;
}

size_t workSize_bigmatrix(const size_t m,            // size(A,1) & size(C,1) <= 2^17
                  const size_t n,            // size(B,2) & size(C,2) <= 2^17
                  const size_t k,            // size(A,2) & size(B,1) <= 2^17
                  const unsigned num_moduli) // 2 <= num_moduli <= 20
{
#if defined(__HIPCC__)
    const size_t k2_16     = ((2 * k + 15) >> 4) << 4;
    const size_t lda8i     =  k2_16 % 1024 == 0 ? k2_16 + 64 : k2_16;
    const size_t m2_pad    = 2 * m;
#else
    const size_t lda8i     = ((2 * k + 15) >> 4) << 4;
    const size_t m2_pad    = ((2 * m + 3) >> 2) << 2;
#endif
    const size_t ldb8i     = lda8i;
    const size_t sizeA     = lda8i * m2_pad;
    const size_t sizeB     = ldb8i * 2 * n;
    const size_t sizeC     = ((m2_pad * n + 15) >> 4) << 4;
#if defined(__HIPCC__)
    const size_t sizeC32i  = m % 512 == 0 ? ((2 * (m+1) * 2 * n + 15) >> 4) << 4 : ((m2_pad * 2 * n + 15) >> 4) << 4;
#else
    const size_t sizeC32i  = ((m2_pad * 2 * n + 15) >> 4) << 4;
#endif
    const size_t size_vecA = (((m + 15) >> 4) << 4); // multiple of 16
    const size_t size_vecB = (((n + 15) >> 4) << 4); // multiple of 16

    size_t total_size = 0;
    total_size += sizeof(int8_t) * (sizeA + sizeB) * num_moduli;
    total_size += sizeof(uint8_t) * sizeC * num_moduli;
    total_size += sizeof(int32_t) * sizeC32i;
    total_size += sizeof(int16_t) * (size_vecA + size_vecB);

    return total_size;
}

size_t workSize_kara(const size_t m,            // size(A,1) & size(C,1) <= 2^17
                  const size_t n,            // size(B,2) & size(C,2) <= 2^17
                  const size_t k,            // size(A,2) & size(B,1) <= 2^17
                  const unsigned num_moduli) // 2 <= num_moduli <= 20
{
#if defined(__HIPCC__)
    const size_t k16       = ((k + 15) >> 4) << 4;
    const size_t lda8i     =  k16 % 1024 == 0 ? k16 + 64 : k16;
    const size_t m_pad     = m;
#else
    const size_t lda8i     = ((k + 15) >> 4) << 4;
    const size_t m_pad     = ((m + 3) >> 2) << 2;
#endif
    const size_t ldb8i     = lda8i;
    const size_t sizeA     = lda8i * m_pad;
    const size_t sizeB     = ldb8i * n;
    const size_t sizeC     = ((m_pad * n + 15) >> 4) << 4;
#if defined(__HIPCC__)
    const size_t sizeC32i  = m % 1024 == 0 ? (((m_pad+1) * n + 15) >> 4) << 4 : sizeC;
#else
    const size_t sizeC32i  = sizeC;
#endif
    const size_t size_vecA = (((m + 15) >> 4) << 4); // multiple of 16
    const size_t size_vecB = (((n + 15) >> 4) << 4); // multiple of 16

    size_t total_size = 0;
    total_size += sizeof(int8_t) * (sizeA + sizeB) * num_moduli * 2;
    total_size += sizeof(uint8_t) * sizeC * num_moduli * 2;
    total_size += sizeof(int32_t) * sizeC32i * 2;
    total_size += sizeof(int16_t) * (size_vecA + size_vecB);

    return total_size;
}

size_t workSize(const size_t m,            // size(A,1) & size(C,1) <= 2^17
                const size_t n,            // size(B,2) & size(C,2) <= 2^17
                const size_t k,            // size(A,2) & size(B,1) <= 2^17
                const unsigned num_moduli, // 2 <= num_moduli <= 20
                const computeType_t computeType)
{
    switch (computeType) {
        case REAL_DEFAULT:
            return workSize_real(m, n, k, num_moduli);
        case COMPLEX_BIG_MATRIX_ENCODE:
            return workSize_bigmatrix(m, n, k, num_moduli);
        case COMPLEX_KARATSUBA_MULT:
            return workSize_kara(m, n, k, num_moduli);
        default: {
            fprintf(stderr, "Unknown compute type\n");
            return 0;
        }
    }
}

template <>
std::vector<double> gemm<double>(gpublasHandle_t handle,        // handle
                                 const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const size_t m,               // size(A,1) & size(C,1) <= 2^17
                                 const size_t n,               // size(B,2) & size(C,2) <= 2^17
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
                                 const bool fastmode,          // false (int8-tc) or true (vecnorm)
                                 void *const work,             // workspace
                                 const computeType_t computeType)
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);
    if (computeType != REAL_DEFAULT) {
        fprintf(stderr, "Unsupported compute type for the argument types.\n");
        return timer;
    }

    //------------------------------
    // set constants
    //------------------------------
#if defined(__HIPCC__)
    const size_t k16         = ((k + 15) >> 4) << 4;
    const size_t lda8i       =  k16 % 1024 == 0 ? k16 + 64 : k16;
    const size_t m_pad       = m;
#else
    const size_t lda8i       = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t m_pad       = ((m + 3) >> 2) << 2;
#endif
    const size_t ldb8i       = lda8i;
    const size_t sizeA       = lda8i * m_pad;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ((m_pad * n + 15) >> 4) << 4; // multiple of 16
#if defined(__HIPCC__)
    const size_t sizeC32i    = m % 1024 == 0 ? (((m_pad+1) * n + 15) >> 4) << 4 : sizeC;
#else
    const size_t sizeC32i    = sizeC;
#endif
    const size_t size_vecA   = (((m + 15) >> 4) << 4);   // multiple of 16
    const unsigned table_idx = num_moduli - 2;
    const unsigned numM      = oz2_table::numM[table_idx]; // numM <= 2
    const bool is_numM_1     = numM == 1;
    constexpr int32_t one    = 1;
    constexpr int32_t zero   = 0;

#if GEMMul8_ARCH < 89
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 1024;
    oz2_const::threads_invscaling = 256;
#elif GEMMul8_ARCH < 90
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#elif GEMMul8_ARCH < 100
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#else
    oz2_const::threads_scaling    = 128;
    oz2_const::threads_conv32i8u  = 256;
    oz2_const::threads_invscaling = 512;
#endif
    oz2_const::grids_invscaling = (m * n + oz2_const::threads_invscaling - 1) / oz2_const::threads_invscaling;
    oz2_const::grids_conv32i8u  = ((sizeC >> 2) + oz2_const::threads_conv32i8u - 1) / oz2_const::threads_conv32i8u;

    //------------------------------
    // set workspace (16byte align)
    //------------------------------
    int8_t *A8i   = reinterpret_cast<int8_t *>(work);                      // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *B8i   = A8i + sizeA * num_moduli;                              // ldb8i*n*sizeod(int8_t)*num_moduli
    uint8_t *C8u  = reinterpret_cast<uint8_t *>(B8i + sizeB * num_moduli); // (m*n+15)/16*16*sizeof(uint8_t)*num_moduli
    int32_t *C32i = reinterpret_cast<int32_t *>(C8u + sizeC * num_moduli); // (m*n+15)/16*16*sizeof(int32_t)
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i + sizeC32i);          // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)

    if (is_numM_1) {
        gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_1[table_idx][0], num_moduli * sizeof(double));
    } else {
        gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_2[num_moduli - 8][0][0], 2 * num_moduli * sizeof(double));
    }
    gpuMemcpyToSymbol(oz2_table::moduli_dev, oz2_table::moduli, num_moduli * sizeof(oz2_table::tab_t<double>));

    //------------------------------
    // Scaling
    // A =: diag(2^sftA) * A64f, A64f is integer
    // B =: B64f * diag(2^sftB), B64f is integer
    // Then, calculating mod for all moduli
    // A8i := A64f - round(A64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    // B8i := B64f - round(B64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    //------------------------------
    timing_start(timetmp);
    if (fastmode) {
        oz2_util::vecnorm::scaling<double>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<double>(handle, op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, C32i, table_idx);
    }
    timing_stop(timetmp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i + i * sizeA, GPU_R_8I, lda8i, B8i + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
        timing_stop(timetmp, timer[1]);

        //------------------------------
        // Calculating mod
        // C8u[i] := mod(C32i, modulus[i]) >= 0
        //------------------------------
        timing_start(timetmp);
        oz2_util::conv_32i_2_8u(i, sizeC, C32i, C8u + i * sizeC);
        timing_stop(timetmp, timer[2]);
    }

    //------------------------------
    // Accumulation and Inverse scaling
    // C64f = sum(Ni*Mi*C8u[i]),
    //  where
    //      Mi := M/modulus[i],
    //      M := prod(modulus[all]),
    //      mod(Ni*Mi, modulus[i]) == 1.
    // C := C64f - round(C64f/M)*M
    // C := diag(2^sftA) * C * diag(2^sftB)
    //------------------------------
    timing_start(timetmp);
    oz2_util::inverse_scaling(is_numM_1, num_moduli, m, n, C8u, m_pad, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
    timing_stop(timetmp, timer[3]);

    return timer;
}

template <>
std::vector<double> gemm<float>(gpublasHandle_t handle,        // handle
                                const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                const size_t m,               // size(A,1) & size(C,1) <= 2^17
                                const size_t n,               // size(B,2) & size(C,2) <= 2^17
                                const size_t k,               // size(A,2) & size(B,1) <= 2^17
                                const float *alpha,           //
                                const float *const A,         // input
                                const size_t lda,             // leading dimension
                                const float *const B,         // input
                                const size_t ldb,             // leading dimension
                                const float *beta,            //
                                float *const C,               // output A*B
                                const size_t ldc,             // leading dimension
                                const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 20
                                const bool fastmode,          // false (int8-tc) or true (vecnorm)
                                void *const work,             // workspace
                                const computeType_t computeType)
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);
    if (computeType != REAL_DEFAULT) {
        fprintf(stderr, "Unsupported compute type for the argument types.\n");
        return timer;
    }

    //------------------------------
    // set constants
    //------------------------------
#if defined(__HIPCC__)
    const size_t k16         = ((k + 15) >> 4) << 4;
    const size_t lda8i       =  k16 % 1024 == 0 ? k16 + 64 : k16;
    const size_t m_pad       = m;
#else
    const size_t lda8i       = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t m_pad       = ((m + 3) >> 2) << 2;
#endif
    const size_t ldb8i       = lda8i;
    const size_t sizeA       = lda8i * m_pad;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ((m_pad * n + 15) >> 4) << 4; // multiple of 16
#if defined(__HIPCC__)
    const size_t sizeC32i    = m % 1024 == 0 ? (((m_pad+1) * n + 15) >> 4) << 4 : sizeC;
#else
    const size_t sizeC32i    = sizeC;
#endif
    const size_t size_vecA   = (((m + 15) >> 4) << 4);
    const unsigned table_idx = num_moduli - 2;
    constexpr int32_t one    = 1;
    constexpr int32_t zero   = 0;

#if GEMMul8_ARCH < 89
    oz2_const::threads_scaling    = 512;
    oz2_const::threads_conv32i8u  = 1024;
    oz2_const::threads_invscaling = 128;
#elif GEMMul8_ARCH < 90
    oz2_const::threads_scaling    = 512;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#elif GEMMul8_ARCH < 100
    oz2_const::threads_scaling    = 64;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 128;
#else
    oz2_const::threads_scaling    = 512;
    oz2_const::threads_conv32i8u  = 256;
    oz2_const::threads_invscaling = 512;
#endif
    oz2_const::grids_invscaling = (m * n + oz2_const::threads_invscaling - 1) / oz2_const::threads_invscaling;
    oz2_const::grids_conv32i8u  = ((sizeC >> 2) + oz2_const::threads_conv32i8u - 1) / oz2_const::threads_conv32i8u;

    //------------------------------
    // set workspace (16byte align)
    //------------------------------
    int8_t *A8i   = reinterpret_cast<int8_t *>(work);                      // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *B8i   = A8i + sizeA * num_moduli;                              // ldb8i*n*sizeod(int8_t)*num_moduli
    uint8_t *C8u  = reinterpret_cast<uint8_t *>(B8i + sizeB * num_moduli); // (m*n+15)/16*16*sizeof(uint8_t)*num_moduli
    int32_t *C32i = reinterpret_cast<int32_t *>(C8u + sizeC * num_moduli); // (m*n+15)/16*16*sizeof(int32_t)
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i + sizeC32i);          // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)

    gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_1[table_idx][0], num_moduli * sizeof(double));
    gpuMemcpyToSymbol(oz2_table::modulif_dev, oz2_table::modulif, num_moduli * sizeof(oz2_table::tab_t<float>));

    //------------------------------
    // Scaling
    // A =: diag(2^sftA) * A64f, A64f is integer
    // B =: B64f * diag(2^sftB), B64f is integer
    // Then, calculating mod for all moduli
    // A8i := mod(A64f, modulus[i]) - 128 (-128 <= A8i <= 127)
    // B8i := mod(B64f, modulus[i]) - 128 (-128 <= A8i <= 127)
    //------------------------------
    timing_start(timetmp);
    if (fastmode) {
        oz2_util::vecnorm::scaling<float>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<float>(handle, op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, C32i, table_idx);
    }
    timing_stop(timetmp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i + i * sizeA, GPU_R_8I, lda8i, B8i + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
        timing_stop(timetmp, timer[1]);

        //------------------------------
        // Calculating mod
        // C8u[i] := mod(C32i, modulus[i]) >= 0
        //------------------------------
        timing_start(timetmp);
        oz2_util::conv_32i_2_8u(i, sizeC, C32i, C8u + i * sizeC);
        timing_stop(timetmp, timer[2]);
    }

    //------------------------------
    // Accumulation and Inverse scaling
    // C64f = sum(Ni*Mi*C8u[i]),
    //  where
    //      Mi := M/modulus[i],
    //      M := prod(modulus[all]),
    //      mod(Ni*Mi, modulus[i]) == 1.
    // C := C64f - round(C64f/M)*M
    // C := diag(2^sftA) * C * diag(2^sftB)
    //------------------------------
    timing_start(timetmp);
    oz2_util::inverse_scaling(num_moduli, m, n, C8u, m_pad, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
    timing_stop(timetmp, timer[3]);

    return timer;
}

template <typename TA, typename TB, typename TC>
std::vector<double> gemm_mixed(gpublasHandle_t handle,          // handle
                                 const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const size_t m,               // size(A,1) & size(C,1) <= 2^17
                                 const size_t n,               // size(B,2) & size(C,2) <= 2^17
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
                                 const bool fastmode,          // false (int8-tc) or true (vecnorm)
                                 void *const work,             // workspace
                                 const computeType_t computeType)
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);
    if (computeType != REAL_DEFAULT) {
        fprintf(stderr, "Unsupported compute type for the argument types.\n");
        return timer;
    }

    //------------------------------
    // set constants
    //------------------------------
#if defined(__HIPCC__)
    const size_t k16         = ((k + 15) >> 4) << 4;
    const size_t lda8i       =  k16 % 1024 == 0 ? k16 + 64 : k16;
    const size_t m_pad       = m;
#else
    const size_t lda8i       = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t m_pad       = ((m + 3) >> 2) << 2;
#endif
    const size_t ldb8i       = lda8i;
    const size_t sizeA       = lda8i * m_pad;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ((m_pad * n + 15) >> 4) << 4; // multiple of 16
#if defined(__HIPCC__)
    const size_t sizeC32i    = m % 1024 == 0 ? (((m_pad+1) * n + 15) >> 4) << 4 : sizeC;
#else
    const size_t sizeC32i    = sizeC;
#endif
    const size_t size_vecA   = (((m + 15) >> 4) << 4);   // multiple of 16
    const unsigned table_idx = num_moduli - 2;
    const unsigned numM      = oz2_table::numM[table_idx]; // numM <= 2
    const bool is_numM_1     = numM == 1 || std::is_same_v<TC, float>;
    constexpr int32_t one    = 1;
    constexpr int32_t zero   = 0;

#if GEMMul8_ARCH < 89
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 1024;
    oz2_const::threads_invscaling = 256;
#elif GEMMul8_ARCH < 90
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#elif GEMMul8_ARCH < 100
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#else
    oz2_const::threads_scaling    = 128;
    oz2_const::threads_conv32i8u  = 256;
    oz2_const::threads_invscaling = 512;
#endif
    oz2_const::grids_invscaling = (m * n + oz2_const::threads_invscaling - 1) / oz2_const::threads_invscaling;
    oz2_const::grids_conv32i8u  = ((sizeC >> 2) + oz2_const::threads_conv32i8u - 1) / oz2_const::threads_conv32i8u;

    //------------------------------
    // set workspace (16byte align)
    //------------------------------
    int8_t *A8i   = reinterpret_cast<int8_t *>(work);                      // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *B8i   = A8i + sizeA * num_moduli;                              // ldb8i*n*sizeod(int8_t)*num_moduli
    uint8_t *C8u  = reinterpret_cast<uint8_t *>(B8i + sizeB * num_moduli); // (m*n+15)/16*16*sizeof(uint8_t)*num_moduli
    int32_t *C32i = reinterpret_cast<int32_t *>(C8u + sizeC * num_moduli); // (m*n+15)/16*16*sizeof(int32_t)
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i + sizeC32i);          // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)

    if (is_numM_1) {
        gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_1[table_idx][0], num_moduli * sizeof(double));
    } else {
        gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_2[num_moduli - 8][0][0], 2 * num_moduli * sizeof(double));
    }
    gpuMemcpyToSymbol(oz2_table::moduli_dev, oz2_table::moduli, num_moduli * sizeof(oz2_table::tab_t<double>));
    gpuMemcpyToSymbol(oz2_table::modulif_dev, oz2_table::modulif, num_moduli * sizeof(oz2_table::tab_t<float>));

    //------------------------------
    // Scaling
    // A =: diag(2^sftA) * A64f, A64f is integer
    // B =: B64f * diag(2^sftB), B64f is integer
    // Then, calculating mod for all moduli
    // A8i := A64f - round(A64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    // B8i := B64f - round(B64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    //------------------------------
    timing_start(timetmp);
    if (fastmode) {
        oz2_util::vecnorm::scaling<TA, TB>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<TA, TB>(handle, op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, C32i, table_idx);
    }
    timing_stop(timetmp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i + i * sizeA, GPU_R_8I, lda8i, B8i + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
        timing_stop(timetmp, timer[1]);

        //------------------------------
        // Calculating mod
        // C8u[i] := mod(C32i, modulus[i]) >= 0
        //------------------------------
        timing_start(timetmp);
        oz2_util::conv_32i_2_8u(i, sizeC, C32i, C8u + i * sizeC);
        timing_stop(timetmp, timer[2]);
    }

    //------------------------------
    // Accumulation and Inverse scaling
    // C64f = sum(Ni*Mi*C8u[i]),
    //  where
    //      Mi := M/modulus[i],
    //      M := prod(modulus[all]),
    //      mod(Ni*Mi, modulus[i]) == 1.
    // C := C64f - round(C64f/M)*M
    // C := diag(2^sftA) * C * diag(2^sftB)
    //------------------------------
    timing_start(timetmp);
    oz2_util::inverse_scaling<TC>(is_numM_1, num_moduli, m, n, C8u, m_pad, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
    timing_stop(timetmp, timer[3]);

    return timer;
}

template <typename TA, typename TB, typename TC>
std::vector<double> gemm_mixed_bigmatrix(gpublasHandle_t handle,        // handle
                                 const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const size_t m,                // size(A,1) & size(C,1) <= 2^17
                                 const size_t n,                // size(B,2) & size(C,2) <= 2^17
                                 const size_t k,                // size(A,2) & size(B,1) <= 2^17
                                 const TC *alpha,               //
                                 const TA *const A,             // input
                                 const size_t lda,              // leading dimension
                                 const TB *const B,             // input
                                 const size_t ldb,              // leading dimension
                                 const TC *beta,                //
                                 TC *const C,                   // output A*B
                                 const size_t ldc,              // leading dimension
                                 const unsigned num_moduli,     // #moduli, 2 <= num_moduli <= 20
                                 const bool fastmode,           // false (int8-tc) or true (vecnorm)
                                 void *const work,              // workspace
                                 const computeType_t computeType)
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);
    if (computeType != COMPLEX_BIG_MATRIX_ENCODE) {
        fprintf(stderr, "Unsupported compute type for the argument types.\n");
        return timer;
    }

    //------------------------------
    // set constants
    //------------------------------
#if defined(__HIPCC__)
    const size_t k2_16       = ((2 * k + 15) >> 4) << 4;
    const size_t lda8i       =  k2_16 % 1024 == 0 ? k2_16 + 64 : k2_16;
    const size_t m2_pad      = 2 * m;
#else
    const size_t lda8i       = ((2 * k + 15) >> 4) << 4; // multiple of 16
    const size_t m2_pad      = ((2 * m + 3) >> 2) << 2;
#endif
    const size_t ldb8i       = lda8i;
    const size_t sizeA       = lda8i * m2_pad;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ((m2_pad * n + 15) >> 4) << 4; // multiple of 16
#if defined(__HIPCC__)
    const size_t sizeC32i    = m % 512 == 0 ? ((2 * (m+1) * 2 * n + 15) >> 4) << 4 :((m2_pad * 2 * n + 15) >> 4) << 4 ;
#else
    const size_t sizeC32i    = ((m2_pad * 2 * n + 15) >> 4) << 4; // multiple of 16
#endif
    const size_t size_vecA   = (((m + 15) >> 4) << 4);   // multiple of 16
    const unsigned table_idx = num_moduli - 2;
    const unsigned numM      = oz2_table::numM[table_idx]; // numM <= 2
    const bool is_numM_1     = numM == 1 || std::is_same_v<TC, gpuFloatComplex>;
    constexpr int32_t one    = 1;
    constexpr int32_t zero   = 0;

#if GEMMul8_ARCH < 89
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 1024;
    oz2_const::threads_invscaling = 256;
#elif GEMMul8_ARCH < 90
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#elif GEMMul8_ARCH < 100
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#else
    oz2_const::threads_scaling    = 128;
    oz2_const::threads_conv32i8u  = 256;
    oz2_const::threads_invscaling = 512;
#endif
    oz2_const::grids_invscaling = (m * n + oz2_const::threads_invscaling - 1) / oz2_const::threads_invscaling;
    oz2_const::grids_conv32i8u  = ((sizeC >> 2) + oz2_const::threads_conv32i8u - 1) / oz2_const::threads_conv32i8u;

    //------------------------------
    // set workspace (16byte align)
    //------------------------------
    int8_t *A8i   = reinterpret_cast<int8_t *>(work);                      // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *B8i   = A8i + sizeA * num_moduli;                              // ldb8i*n*sizeod(int8_t)*num_moduli
    uint8_t *C8u  = reinterpret_cast<uint8_t *>(B8i + sizeB * num_moduli); // (m*n+15)/16*16*sizeof(uint8_t)*num_moduli
    int32_t *C32i = reinterpret_cast<int32_t *>(C8u + sizeC * num_moduli); // (m*n+15)/16*16*sizeof(int32_t)
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i + sizeC32i);          // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)

    if (is_numM_1) {
        gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_1[table_idx][0], num_moduli * sizeof(double));
    } else {
        gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_2[num_moduli - 8][0][0], 2 * num_moduli * sizeof(double));
    }
    gpuMemcpyToSymbol(oz2_table::moduli_dev, oz2_table::moduli, num_moduli * sizeof(oz2_table::tab_t<double>));
    gpuMemcpyToSymbol(oz2_table::modulif_dev, oz2_table::modulif, num_moduli * sizeof(oz2_table::tab_t<float>));

    //------------------------------
    // Scaling
    // A =: diag(2^sftA) * A64f, A64f is integer
    // B =: B64f * diag(2^sftB), B64f is integer
    // Then, calculating mod for all moduli
    // A8i := A64f - round(A64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    // B8i := B64f - round(B64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    //------------------------------
    timing_start(timetmp);
    if (fastmode) {
        oz2_util::vecnorm::scaling_bigmatrix<TA, TB>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m2_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling_bigmatrix<TA, TB>(handle, op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m2_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, C32i, table_idx);
    }
    timing_stop(timetmp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m2_pad, n, lda8i, &one, A8i + i * sizeA, GPU_R_8I, lda8i, B8i + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i, GPU_R_32I, m2_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
        timing_stop(timetmp, timer[1]);

        //------------------------------
        // Calculating mod
        // C8u[i] := mod(C32i, modulus[i]) >= 0
        //------------------------------
        timing_start(timetmp);
        oz2_util::conv_32i_2_8u(i, sizeC, C32i, C8u + i * sizeC);
        timing_stop(timetmp, timer[2]);
    }

    //------------------------------
    // Accumulation and Inverse scaling
    // C64f = sum(Ni*Mi*C8u[i]),
    //  where
    //      Mi := M/modulus[i],
    //      M := prod(modulus[all]),
    //      mod(Ni*Mi, modulus[i]) == 1.
    // C := C64f - round(C64f/M)*M
    // C := diag(2^sftA) * C * diag(2^sftB)
    //------------------------------
    timing_start(timetmp);
    oz2_util::inverse_scaling_bigmatrix<TC>(is_numM_1, num_moduli, m, n, C8u, m2_pad, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
    timing_stop(timetmp, timer[3]);

    return timer;
}

template <typename TA, typename TB, typename TC>
std::vector<double> gemm_mixed_kara(gpublasHandle_t handle,        // handle
                                 const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const size_t m,                // size(A,1) & size(C,1) <= 2^17
                                 const size_t n,                // size(B,2) & size(C,2) <= 2^17
                                 const size_t k,                // size(A,2) & size(B,1) <= 2^17
                                 const TC *alpha,               //
                                 const TA *const A,             // input
                                 const size_t lda,              // leading dimension
                                 const TB *const B,             // input
                                 const size_t ldb,              // leading dimension
                                 const TC *beta,                //
                                 TC *const C,                   // output A*B
                                 const size_t ldc,              // leading dimension
                                 const unsigned num_moduli,     // #moduli, 2 <= num_moduli <= 20
                                 const bool fastmode,           // false (int8-tc) or true (vecnorm)
                                 void *const work,              // workspace
                                 const computeType_t computeType)
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);
    if (computeType != COMPLEX_KARATSUBA_MULT) {
        fprintf(stderr, "Unsupported compute type for the argument types.\n");
        return timer;
    }

    //------------------------------
    // set constants
    //------------------------------
#if defined(__HIPCC__)
    const size_t k16         = ((k + 15) >> 4) << 4;
    const size_t lda8i       =  k16 % 1024 == 0 ? k16 + 64 : k16;
    const size_t m_pad       = m;
#else
    const size_t lda8i       = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t m_pad       = ((m + 3) >> 2) << 2;
#endif
    const size_t ldb8i       = lda8i;
    const size_t sizeA       = lda8i * m_pad;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ((m_pad * n + 15) >> 4) << 4; // multiple of 16
#if defined(__HIPCC__)
    const size_t sizeC32i    = m % 1024 == 0 ? (((m_pad+1) * n + 15) >> 4) << 4 : sizeC;
#else
    const size_t sizeC32i    = sizeC;
#endif
    const size_t size_vecA   = (((m + 15) >> 4) << 4);
    const unsigned table_idx = num_moduli - 2;
    const unsigned numM      = oz2_table::numM[table_idx]; // numM <= 2
    const bool is_numM_1     = numM == 1 || std::is_same_v<TC, gpuFloatComplex>;
    constexpr int32_t one    = 1;
    constexpr int32_t zero   = 0;
    constexpr int32_t m_one  = -1;
    constexpr int32_t m_two  = -2;

#if GEMMul8_ARCH < 89
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 1024;
    oz2_const::threads_invscaling = 256;
#elif GEMMul8_ARCH < 90
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#elif GEMMul8_ARCH < 100
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#else
    oz2_const::threads_scaling    = 128;
    oz2_const::threads_conv32i8u  = 256;
    oz2_const::threads_invscaling = 256;
#endif
    oz2_const::grids_invscaling = (m * n + oz2_const::threads_invscaling - 1) / oz2_const::threads_invscaling;
    oz2_const::grids_conv32i8u  = ((sizeC >> 2) + oz2_const::threads_conv32i8u - 1) / oz2_const::threads_conv32i8u;

    //------------------------------
    // set workspace (16byte align)
    //------------------------------
    int8_t *A8i_real   = reinterpret_cast<int8_t *>(work);                      // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *A8i_imag   = A8i_real + sizeA * num_moduli;                              // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *B8i_real   = A8i_imag + sizeA * num_moduli;                              // ldb8i*n*sizeod(int8_t)*num_moduli
    int8_t *B8i_imag   = B8i_real + sizeB * num_moduli;                              // ldb8i*n*sizeod(int8_t)*num_moduli
    uint8_t *C8u_real  = reinterpret_cast<uint8_t *>(B8i_imag + sizeB * num_moduli); // (m*n+15)/16*16*sizeof(uint8_t)*num_moduli
    uint8_t *C8u_imag  = C8u_real + sizeC * num_moduli;
    int32_t *C32i_real = reinterpret_cast<int32_t *>(C8u_imag + sizeC * num_moduli); // (m*n+15)/16*16*sizeof(int32_t)
    int32_t *C32i_imag = C32i_real + sizeC32i;
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i_imag + sizeC32i);          // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)

    if (is_numM_1) {
        gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_1[table_idx][0], num_moduli * sizeof(double));
    } else {
        gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_2[num_moduli - 8][0][0], 2 * num_moduli * sizeof(double));
    }
    gpuMemcpyToSymbol(oz2_table::moduli_dev, oz2_table::moduli, num_moduli * sizeof(oz2_table::tab_t<double>));
    gpuMemcpyToSymbol(oz2_table::modulif_dev, oz2_table::modulif, num_moduli * sizeof(oz2_table::tab_t<float>));

    //------------------------------
    // Scaling
    // A =: diag(2^sftA) * A64f, A64f is integer
    // B =: B64f * diag(2^sftB), B64f is integer
    // Then, calculating mod for all moduli
    // A8i := A64f - round(A64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    // B8i := B64f - round(B64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    //------------------------------
    timing_start(timetmp);
    if (fastmode) {
        oz2_util::vecnorm::scaling_kara<TA, TB>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i_real, A8i_imag, lda8i, lda8i * m_pad, sftA, B8i_real, B8i_imag, ldb8i, ldb8i * n, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling_kara<TA, TB>(handle, op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i_real, A8i_imag, lda8i, lda8i * m_pad, sftA, B8i_real, B8i_imag, ldb8i, ldb8i * n, sftB, C32i_real, C32i_imag, table_idx);
    }
    timing_stop(timetmp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);

        // E = Re(A)Re(B)
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_real + i * sizeA, GPU_R_8I, lda8i, B8i_real + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i_real, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
        // F = Im(A)Im(B)
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_imag + i * sizeA, GPU_R_8I, lda8i, B8i_imag + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i_imag, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
        // G1 = Re(A) + Im(A)
        oz2_mat_utils::add_int8_mat(i, sizeA, A8i_real + i * sizeA, A8i_imag + i * sizeA);
        // G2 = Re(B) + Im(B)
        oz2_mat_utils::add_int8_mat(i, sizeB, B8i_real + i * sizeB, B8i_imag + i * sizeB);
        // C32i_real = E - F
        oz2_mat_utils::sub_int32_mat(sizeC32i, C32i_real, C32i_imag);
        // C32i_imag = G1 * G2 - 2F
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_real + i * sizeA, GPU_R_8I, lda8i, B8i_real + i * sizeB, GPU_R_8I, ldb8i, &m_two, C32i_imag, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
        // C32i_imag = G1 * G2 - E - F
        oz2_mat_utils::sub_int32_mat(sizeC32i, C32i_imag, C32i_real);

        timing_stop(timetmp, timer[1]);

        //------------------------------
        // Calculating mod
        // C8u[i] := mod(C32i, modulus[i]) >= 0
        //------------------------------
        timing_start(timetmp);
        oz2_util::conv_32i_2_8u(i, sizeC, C32i_real, C8u_real + i * sizeC);
        oz2_util::conv_32i_2_8u(i, sizeC, C32i_imag, C8u_imag + i * sizeC);
        timing_stop(timetmp, timer[2]);
    }

    //------------------------------
    // Accumulation and Inverse scaling
    // C64f = sum(Ni*Mi*C8u[i]),
    //  where
    //      Mi := M/modulus[i],
    //      M := prod(modulus[all]),
    //      mod(Ni*Mi, modulus[i]) == 1.
    // C := C64f - round(C64f/M)*M
    // C := diag(2^sftA) * C * diag(2^sftB)
    //------------------------------
    timing_start(timetmp);
    oz2_util::inverse_scaling_kara<TC>(is_numM_1, num_moduli, m, n, C8u_real, C8u_imag, m_pad, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
    timing_stop(timetmp, timer[3]);

    return timer;
}

template <typename TA, typename TB, typename TC>
std::vector<double> gemm_mixed_classic(gpublasHandle_t handle,        // handle
                                 const gpublasOperation_t op_A, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const gpublasOperation_t op_B, // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const size_t m,                // size(A,1) & size(C,1) <= 2^17
                                 const size_t n,                // size(B,2) & size(C,2) <= 2^17
                                 const size_t k,                // size(A,2) & size(B,1) <= 2^17
                                 const TC *alpha,               //
                                 const TA *const A,             // input
                                 const size_t lda,              // leading dimension
                                 const TB *const B,             // input
                                 const size_t ldb,              // leading dimension
                                 const TC *beta,                //
                                 TC *const C,                   // output A*B
                                 const size_t ldc,              // leading dimension
                                 const unsigned num_moduli,     // #moduli, 2 <= num_moduli <= 20
                                 const bool fastmode,           // false (int8-tc) or true (vecnorm)
                                 void *const work,              // workspace
                                 const computeType_t computeType)
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);
    if (computeType != COMPLEX_CLASSIC_MULT) {
        fprintf(stderr, "Unsupported compute type for the argument types.\n");
        return timer;
    }

    //------------------------------
    // set constants
    //------------------------------
#if defined(__HIPCC__)
    const size_t k16         = ((k + 15) >> 4) << 4;
    const size_t lda8i       =  k16 % 1024 == 0 ? k16 + 64 : k16;
    const size_t m_pad       = m;
#else
    const size_t lda8i       = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t m_pad       = ((m + 3) >> 2) << 2;
#endif
    const size_t ldb8i       = lda8i;
    const size_t sizeA       = lda8i * m_pad;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ((m_pad * n + 15) >> 4) << 4; // multiple of 16
#if defined(__HIPCC__)
    const size_t sizeC32i    = m % 1024 == 0 ? (((m_pad+1) * n + 15) >> 4) << 4 : sizeC;
#else
    const size_t sizeC32i    = sizeC;
#endif
    const size_t size_vecA   = (((m + 15) >> 4) << 4);
    const unsigned table_idx = num_moduli - 2;
    const unsigned numM      = oz2_table::numM[table_idx]; // numM <= 2
    const bool is_numM_1     = numM == 1 || std::is_same_v<TC, gpuFloatComplex>;
    constexpr int32_t one    = 1;
    constexpr int32_t zero   = 0;
    constexpr int32_t m_one  = -1;
    constexpr int32_t m_two  = -2;

#if GEMMul8_ARCH < 89
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 1024;
    oz2_const::threads_invscaling = 256;
#elif GEMMul8_ARCH < 90
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#elif GEMMul8_ARCH < 100
    oz2_const::threads_scaling    = 256;
    oz2_const::threads_conv32i8u  = 128;
    oz2_const::threads_invscaling = 64;
#else
    oz2_const::threads_scaling    = 128;
    oz2_const::threads_conv32i8u  = 256;
    oz2_const::threads_invscaling = 256;
#endif
    oz2_const::grids_invscaling = (m * n + oz2_const::threads_invscaling - 1) / oz2_const::threads_invscaling;
    oz2_const::grids_conv32i8u  = ((sizeC >> 2) + oz2_const::threads_conv32i8u - 1) / oz2_const::threads_conv32i8u;

    //------------------------------
    // set workspace (16byte align)
    //------------------------------
    int8_t *A8i_real   = reinterpret_cast<int8_t *>(work);                      // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *A8i_imag   = A8i_real + sizeA * num_moduli;                              // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *B8i_real   = A8i_imag + sizeA * num_moduli;                              // ldb8i*n*sizeod(int8_t)*num_moduli
    int8_t *B8i_imag   = B8i_real + sizeB * num_moduli;                              // ldb8i*n*sizeod(int8_t)*num_moduli
    uint8_t *C8u_real  = reinterpret_cast<uint8_t *>(B8i_imag + sizeB * num_moduli); // (m*n+15)/16*16*sizeof(uint8_t)*num_moduli
    uint8_t *C8u_imag  = C8u_real + sizeC * num_moduli;
    int32_t *C32i_real = reinterpret_cast<int32_t *>(C8u_imag + sizeC * num_moduli); // (m*n+15)/16*16*sizeof(int32_t)
    int32_t *C32i_imag = C32i_real + sizeC32i;
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i_imag + sizeC32i);          // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)

    if (is_numM_1) {
        gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_1[table_idx][0], num_moduli * sizeof(double));
    } else {
        gpuMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_2[num_moduli - 8][0][0], 2 * num_moduli * sizeof(double));
    }
    gpuMemcpyToSymbol(oz2_table::moduli_dev, oz2_table::moduli, num_moduli * sizeof(oz2_table::tab_t<double>));
    gpuMemcpyToSymbol(oz2_table::modulif_dev, oz2_table::modulif, num_moduli * sizeof(oz2_table::tab_t<float>));

    //------------------------------
    // Scaling
    // A =: diag(2^sftA) * A64f, A64f is integer
    // B =: B64f * diag(2^sftB), B64f is integer
    // Then, calculating mod for all moduli
    // A8i := A64f - round(A64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    // B8i := B64f - round(B64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    //------------------------------
    timing_start(timetmp);
    if (fastmode) {
        oz2_util::vecnorm::scaling_kara<TA, TB>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i_real, A8i_imag, lda8i, lda8i * m_pad, sftA, B8i_real, B8i_imag, ldb8i, ldb8i * n, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling_kara<TA, TB>(handle, op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i_real, A8i_imag, lda8i, lda8i * m_pad, sftA, B8i_real, B8i_imag, ldb8i, ldb8i * n, sftB, C32i_real, C32i_imag, table_idx);
    }
    timing_stop(timetmp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);

        // F = Im(A)Im(B)
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_imag + i * sizeA, GPU_R_8I, lda8i, B8i_imag + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i_real, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
        // C32i_real = Re(A)Re(B) - F
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_real + i * sizeA, GPU_R_8I, lda8i, B8i_real + i * sizeB, GPU_R_8I, ldb8i, &m_one, C32i_real, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);

        // H = Im(A)Re(B)
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_imag + i * sizeA, GPU_R_8I, lda8i, B8i_real + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i_imag, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
        // C32i_imag = Re(A)Im(B) + H
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m_pad, n, lda8i, &one, A8i_real + i * sizeA, GPU_R_8I, lda8i, B8i_imag + i * sizeB, GPU_R_8I, ldb8i, &one, C32i_imag, GPU_R_32I, m_pad, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);

        timing_stop(timetmp, timer[1]);

        //------------------------------
        // Calculating mod
        // C8u[i] := mod(C32i, modulus[i]) >= 0
        //------------------------------
        timing_start(timetmp);
        oz2_util::conv_32i_2_8u(i, sizeC, C32i_real, C8u_real + i * sizeC);
        oz2_util::conv_32i_2_8u(i, sizeC, C32i_imag, C8u_imag + i * sizeC);
        timing_stop(timetmp, timer[2]);
    }

    //------------------------------
    // Accumulation and Inverse scaling
    // C64f = sum(Ni*Mi*C8u[i]),
    //  where
    //      Mi := M/modulus[i],
    //      M := prod(modulus[all]),
    //      mod(Ni*Mi, modulus[i]) == 1.
    // C := C64f - round(C64f/M)*M
    // C := diag(2^sftA) * C * diag(2^sftB)
    //------------------------------
    timing_start(timetmp);
    oz2_util::inverse_scaling_kara<TC>(is_numM_1, num_moduli, m, n, C8u_real, C8u_imag, m_pad, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
    timing_stop(timetmp, timer[3]);

    return timer;
}

template <>
std::vector<double> gemm<double, float, double>(gpublasHandle_t handle, // handle
                                 const gpublasOperation_t op_A,         // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const gpublasOperation_t op_B,         // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const size_t m,                       // size(A,1) & size(C,1) <= 2^17
                                 const size_t n,                       // size(B,2) & size(C,2) <= 2^17
                                 const size_t k,                       // size(A,2) & size(B,1) <= 2^17
                                 const double *alpha,                  //
                                 const double *const A,                // input
                                 const size_t lda,                     // leading dimension
                                 const float *const B,                 // input
                                 const size_t ldb,                     // leading dimension
                                 const double *beta,                   //
                                 double *const C,                      // output A*B
                                 const size_t ldc,                     // leading dimension
                                 const unsigned num_moduli,            // #moduli, 2 <= num_moduli <= 20
                                 const bool fastmode,                  // false (int8-tc) or true (vecnorm)
                                 void *const work,                     // workspace
                                 const computeType_t computeType)
{
    return gemm_mixed<double, float, double>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
}

template <>
std::vector<double> gemm<float, double, double>(gpublasHandle_t handle, // handle
                                 const gpublasOperation_t op_A,         // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const gpublasOperation_t op_B,         // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const size_t m,                       // size(A,1) & size(C,1) <= 2^17
                                 const size_t n,                       // size(B,2) & size(C,2) <= 2^17
                                 const size_t k,                       // size(A,2) & size(B,1) <= 2^17
                                 const double *alpha,                  //
                                 const float *const A,                 // input
                                 const size_t lda,                     // leading dimension
                                 const double *const B,                // input
                                 const size_t ldb,                     // leading dimension
                                 const double *beta,                   //
                                 double *const C,                      // output A*B
                                 const size_t ldc,                     // leading dimension
                                 const unsigned num_moduli,            // #moduli, 2 <= num_moduli <= 20
                                 const bool fastmode,                  // false (int8-tc) or true (vecnorm)
                                 void *const work,                     // workspace
                                 const computeType_t computeType)
{
    return gemm_mixed<float, double, double>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
}

template <>
std::vector<double> gemm<double, float, float>(gpublasHandle_t handle, // handle
                                 const gpublasOperation_t op_A,        // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const gpublasOperation_t op_B,        // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const size_t m,                      // size(A,1) & size(C,1) <= 2^17
                                 const size_t n,                      // size(B,2) & size(C,2) <= 2^17
                                 const size_t k,                      // size(A,2) & size(B,1) <= 2^17
                                 const float *alpha,                  //
                                 const double *const A,               // input
                                 const size_t lda,                    // leading dimension
                                 const float *const B,                // input
                                 const size_t ldb,                    // leading dimension
                                 const float *beta,                   //
                                 float *const C,                      // output A*B
                                 const size_t ldc,                    // leading dimension
                                 const unsigned num_moduli,           // #moduli, 2 <= num_moduli <= 20
                                 const bool fastmode,                 // false (int8-tc) or true (vecnorm)
                                 void *const work,                    // workspace
                                 const computeType_t computeType)
{
    return gemm_mixed<double, float, float>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
}

template <>
std::vector<double> gemm<float, double, float>(gpublasHandle_t handle, // handle
                                 const gpublasOperation_t op_A,        // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const gpublasOperation_t op_B,        // GPUBLAS_OP_N or GPUBLAS_OP_T
                                 const size_t m,                      // size(A,1) & size(C,1) <= 2^17
                                 const size_t n,                      // size(B,2) & size(C,2) <= 2^17
                                 const size_t k,                      // size(A,2) & size(B,1) <= 2^17
                                 const float *alpha,                  //
                                 const float *const A,                // input
                                 const size_t lda,                    // leading dimension
                                 const double *const B,               // input
                                 const size_t ldb,                    // leading dimension
                                 const float *beta,                   //
                                 float *const C,                      // output A*B
                                 const size_t ldc,                    // leading dimension
                                 const unsigned num_moduli,           // #moduli, 2 <= num_moduli <= 20
                                 const bool fastmode,                 // false (int8-tc) or true (vecnorm)
                                 void *const work,                    // workspace
                                 const computeType_t computeType)
{
    return gemm_mixed<float, double, float>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
}

template <typename TA, typename TB, typename TC>
std::vector<double> gemm_complex_switch(gpublasHandle_t handle,         // handle
                                        const gpublasOperation_t op_A,  // GPUBLAS_OP_N or GPUBLAS_OP_T
                                        const gpublasOperation_t op_B,  // GPUBLAS_OP_N or GPUBLAS_OP_T
                                        const size_t m,                 // size(A,1) & size(C,1)
                                        const size_t n,                 // size(B,2) & size(C,2)
                                        const size_t k,                 // size(A,2) & size(B,1) <= 2^17
                                        const TC *alpha,                //
                                        const TA *const A,              // input
                                        const size_t lda,               // leading dimension
                                        const TB *const B,              // input
                                        const size_t ldb,               // leading dimension
                                        const TC *beta,                 //
                                        TC *const C,                    // output A*B
                                        const size_t ldc,               // leading dimension
                                        const unsigned num_moduli,      // #moduli, 2 <= num_moduli <= 20
                                        const bool fastmode,            // false (accurate-mode) or true (fast-mode)
                                        void *const work,               // workspace allocated in advance
                                        const computeType_t computeType)
{
    switch (computeType) {
        case COMPLEX_BIG_MATRIX_ENCODE:
            return gemm_mixed_bigmatrix<TA, TB, TC>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
        case COMPLEX_CLASSIC_MULT:
            return gemm_mixed_classic<TA, TB, TC>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
        case COMPLEX_KARATSUBA_MULT:
            return gemm_mixed_kara<TA, TB, TC>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
        default: {
            fprintf(stderr, "Unsupported compute type for the argument types.\n");
            return std::vector<double>(4, 0.0);;
        }
    }
}

template <>
std::vector<double> gemm<gpuFloatComplex, gpuFloatComplex, gpuFloatComplex>(gpublasHandle_t handle,        // handle
                             const gpublasOperation_t op_A,  // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const gpublasOperation_t op_B,  // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const size_t m,                 // size(A,1) & size(C,1)
                             const size_t n,                 // size(B,2) & size(C,2)
                             const size_t k,                 // size(A,2) & size(B,1) <= 2^17
                             const gpuFloatComplex *alpha,   //
                             const gpuFloatComplex *const A, // input
                             const size_t lda,               // leading dimension
                             const gpuFloatComplex *const B, // input
                             const size_t ldb,               // leading dimension
                             const gpuFloatComplex *beta,    //
                             gpuFloatComplex *const C,       // output A*B
                             const size_t ldc,               // leading dimension
                             const unsigned num_moduli,      // #moduli, 2 <= num_moduli <= 20
                             const bool fastmode,            // false (accurate-mode) or true (fast-mode)
                             void *const work,               // workspace allocated in advance
                             const computeType_t computeType)
{
    return gemm_complex_switch<gpuFloatComplex, gpuFloatComplex, gpuFloatComplex>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
}

template <>
std::vector<double> gemm<gpuDoubleComplex, gpuDoubleComplex, gpuDoubleComplex>(gpublasHandle_t handle,        // handle
                             const gpublasOperation_t op_A,   // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const gpublasOperation_t op_B,   // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const size_t m,                  // size(A,1) & size(C,1)
                             const size_t n,                  // size(B,2) & size(C,2)
                             const size_t k,                  // size(A,2) & size(B,1) <= 2^17
                             const gpuDoubleComplex *alpha,   //
                             const gpuDoubleComplex *const A, // input
                             const size_t lda,                // leading dimension
                             const gpuDoubleComplex *const B, // input
                             const size_t ldb,                // leading dimension
                             const gpuDoubleComplex *beta,    //
                             gpuDoubleComplex *const C,       // output A*B
                             const size_t ldc,                // leading dimension
                             const unsigned num_moduli,       // #moduli, 2 <= num_moduli <= 20
                             const bool fastmode,             // false (accurate-mode) or true (fast-mode)
                             void *const work,                // workspace allocated in advance
                             const computeType_t computeType)
{
    return gemm_complex_switch<gpuDoubleComplex, gpuDoubleComplex, gpuDoubleComplex>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
}

template <>
std::vector<double> gemm<gpuFloatComplex, gpuDoubleComplex, gpuDoubleComplex>(gpublasHandle_t handle,        // handle
                             const gpublasOperation_t op_A,   // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const gpublasOperation_t op_B,   // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const size_t m,                  // size(A,1) & size(C,1)
                             const size_t n,                  // size(B,2) & size(C,2)
                             const size_t k,                  // size(A,2) & size(B,1) <= 2^17
                             const gpuDoubleComplex *alpha,   //
                             const gpuFloatComplex *const A,  // input
                             const size_t lda,                // leading dimension
                             const gpuDoubleComplex *const B, // input
                             const size_t ldb,                // leading dimension
                             const gpuDoubleComplex *beta,    //
                             gpuDoubleComplex *const C,       // output A*B
                             const size_t ldc,                // leading dimension
                             const unsigned num_moduli,       // #moduli, 2 <= num_moduli <= 20
                             const bool fastmode,             // false (accurate-mode) or true (fast-mode)
                             void *const work,                // workspace allocated in advance
                             const computeType_t computeType)
{
    return gemm_complex_switch<gpuFloatComplex, gpuDoubleComplex, gpuDoubleComplex>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
}

template <>
std::vector<double> gemm<gpuDoubleComplex, gpuFloatComplex, gpuDoubleComplex>(gpublasHandle_t handle,        // handle
                             const gpublasOperation_t op_A,   // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const gpublasOperation_t op_B,   // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const size_t m,                  // size(A,1) & size(C,1)
                             const size_t n,                  // size(B,2) & size(C,2)
                             const size_t k,                  // size(A,2) & size(B,1) <= 2^17
                             const gpuDoubleComplex *alpha,   //
                             const gpuDoubleComplex *const A, // input
                             const size_t lda,                // leading dimension
                             const gpuFloatComplex *const B,  // input
                             const size_t ldb,                // leading dimension
                             const gpuDoubleComplex *beta,    //
                             gpuDoubleComplex *const C,       // output A*B
                             const size_t ldc,                // leading dimension
                             const unsigned num_moduli,       // #moduli, 2 <= num_moduli <= 20
                             const bool fastmode,             // false (accurate-mode) or true (fast-mode)
                             void *const work,                // workspace allocated in advance
                             const computeType_t computeType)
{
    return gemm_complex_switch<gpuDoubleComplex, gpuFloatComplex, gpuDoubleComplex>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
}

template <>
std::vector<double> gemm<gpuDoubleComplex, gpuFloatComplex, gpuFloatComplex>(gpublasHandle_t handle,        // handle
                             const gpublasOperation_t op_A,   // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const gpublasOperation_t op_B,   // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const size_t m,                  // size(A,1) & size(C,1)
                             const size_t n,                  // size(B,2) & size(C,2)
                             const size_t k,                  // size(A,2) & size(B,1) <= 2^17
                             const gpuFloatComplex *alpha,    //
                             const gpuDoubleComplex *const A, // input
                             const size_t lda,                // leading dimension
                             const gpuFloatComplex *const B,  // input
                             const size_t ldb,                // leading dimension
                             const gpuFloatComplex *beta,     //
                             gpuFloatComplex *const C,        // output A*B
                             const size_t ldc,                // leading dimension
                             const unsigned num_moduli,       // #moduli, 2 <= num_moduli <= 20
                             const bool fastmode,             // false (accurate-mode) or true (fast-mode)
                             void *const work,                // workspace allocated in advance
                             const computeType_t computeType)
{
    return gemm_complex_switch<gpuDoubleComplex, gpuFloatComplex, gpuFloatComplex>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
}

template <>
std::vector<double> gemm<gpuFloatComplex, gpuDoubleComplex, gpuFloatComplex>(gpublasHandle_t handle,        // handle
                             const gpublasOperation_t op_A,   // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const gpublasOperation_t op_B,   // GPUBLAS_OP_N or GPUBLAS_OP_T
                             const size_t m,                  // size(A,1) & size(C,1)
                             const size_t n,                  // size(B,2) & size(C,2)
                             const size_t k,                  // size(A,2) & size(B,1) <= 2^17
                             const gpuFloatComplex *alpha,    //
                             const gpuFloatComplex *const A,  // input
                             const size_t lda,                // leading dimension
                             const gpuDoubleComplex *const B, // input
                             const size_t ldb,                // leading dimension
                             const gpuFloatComplex *beta,     //
                             gpuFloatComplex *const C,        // output A*B
                             const size_t ldc,                // leading dimension
                             const unsigned num_moduli,       // #moduli, 2 <= num_moduli <= 20
                             const bool fastmode,             // false (accurate-mode) or true (fast-mode)
                             void *const work,                // workspace allocated in advance
                             const computeType_t computeType)
{
    return gemm_complex_switch<gpuFloatComplex, gpuDoubleComplex, gpuFloatComplex>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work, computeType);
}

} // namespace gemmul8
