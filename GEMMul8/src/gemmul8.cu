#include "../include/gemmul8.hpp"
#include "common.hpp"
#include "conv_32i_2_8u.hpp"
#include "inverse_scaling.hpp"
#include "scaling.hpp"
#include "table.hpp"
#if defined(__HIPCC__)
#define DO_TRANSLATE 1
#endif

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
size_t workSize(const size_t m,            // size(A,1) & size(C,1) <= 2^17
                const size_t n,            // size(B,2) & size(C,2) <= 2^17
                const size_t k,            // size(A,2) & size(B,1) <= 2^17
                const unsigned num_moduli, // 2 <= num_moduli <= 20
                const gpublasOperation_t op_A, // operation used in GEMM, not needed for NVIDIA
                const gpublasOperation_t op_B, // operation used in GEMM, not needed for NVIDIA
                const size_t sizeof_A,     // size in bytes of elements of B, not needed for NVIDIA
                const size_t sizeof_B)     // size in bytes of elements of A, not needed for NVIDIA
{
    const size_t lda8i     = ((k + 15) >> 4) << 4;
    const size_t ldb8i     = lda8i;
    const size_t sizeA     = lda8i * m;
    const size_t sizeB     = ldb8i * n;
    const size_t sizeC     = ((m * n + 15) >> 4) << 4;
    const size_t size_vecA = (((m + 15) >> 4) << 4); // multiple of 16
    const size_t size_vecB = (((n + 15) >> 4) << 4); // multiple of 16

    size_t total_size = 0;
    total_size += sizeof(int8_t) * (sizeA + sizeB) * num_moduli;
    total_size += sizeof(uint8_t) * sizeC * num_moduli;
    total_size += sizeof(int32_t) * sizeC;
    total_size += sizeof(int16_t) * (size_vecA + size_vecB);

#ifdef DO_TRANSLATE
    if (op_A == GPUBLAS_OP_N) {
        total_size += sizeof_A * m * k;
    }
    if (op_B == GPUBLAS_OP_T) {
        total_size += sizeof_B * k * n;
    }
#endif

    return total_size;
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
                                 void *const work)             // workspace
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);

    //------------------------------
    // set constants
    //------------------------------
    const size_t lda8i       = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t ldb8i       = lda8i;
    const size_t sizeA       = lda8i * m;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ((m * n + 15) >> 4) << 4; // multiple of 16
    const size_t size_vecA   = (((m + 15) >> 4) << 4);   // multiple of 16
#ifdef DO_TRANSLATE
    const size_t size_vecB   = (((n + 15) >> 4) << 4);
#endif
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
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i + sizeC);             // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)
#ifdef DO_TRANSLATE
    double *A_trans = reinterpret_cast<double *>(sftB + size_vecB);
    double *B_trans = A_trans;
    if (op_A == GPUBLAS_OP_N && op_B == GPUBLAS_OP_T)
        B_trans = A_trans + m * k;
#endif

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
#ifdef DO_TRANSLATE
    const double* A_use = A;
    const double* B_use = B;
    size_t lda_use = lda;
    size_t ldb_use = ldb;
    if (op_A == GPUBLAS_OP_N) {
        oz2_util::transpose<double>(m, k, A, lda, A_trans);
        A_use = A_trans;
        lda_use = k;
    }
    if (op_B == GPUBLAS_OP_T) {
        oz2_util::transpose<double>(m, k, B, ldb, B_trans);
        B_use = B_trans;
        ldb_use = n;
    }
    if (fastmode) {
        oz2_util::vecnorm::scaling<double>(GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, k, num_moduli, A_use, lda_use, B_use, ldb_use, A8i, lda8i, sftA, B8i, ldb8i, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<double>(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, k, num_moduli, A_use, lda_use, B_use, ldb_use, A8i, lda8i, sftA, B8i, ldb8i, sftB, C32i, table_idx);
    }
#else
    if (fastmode) {
        oz2_util::vecnorm::scaling<double>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, sftA, B8i, ldb8i, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<double>(handle, op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, sftA, B8i, ldb8i, sftB, C32i, table_idx);
    }
#endif
    timing_stop(timetmp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, lda8i, &one, A8i + i * sizeA, GPU_R_8I, lda8i, B8i + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i, GPU_R_32I, m, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
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
    oz2_util::inverse_scaling(is_numM_1, num_moduli, m, n, C8u, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
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
                                void *const work)             // workspace
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);

    //------------------------------
    // set constants
    //------------------------------
    const size_t lda8i       = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t ldb8i       = lda8i;
    const size_t sizeA       = lda8i * m;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ((m * n + 15) >> 4) << 4; // multiple of 16
    const size_t size_vecA   = (((m + 15) >> 4) << 4);
#ifdef DO_TRANSLATE
    const size_t size_vecB   = (((n + 15) >> 4) << 4);
#endif
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
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i + sizeC);             // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)
#ifdef DO_TRANSLATE
    float *A_trans = reinterpret_cast<float *>(sftB + size_vecB);
    float *B_trans = A_trans;
    if (op_A == GPUBLAS_OP_N && op_B == GPUBLAS_OP_T)
        B_trans = A_trans + m * k;
#endif

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
#ifdef DO_TRANSLATE
    const float* A_use = A;
    const float* B_use = B;
    size_t lda_use = lda;
    size_t ldb_use = ldb;
    if (op_A == GPUBLAS_OP_N) {
        oz2_util::transpose<float>(m, k, A, lda, A_trans);
        A_use = A_trans;
        lda_use = k;
    }
    if (op_B == GPUBLAS_OP_T) {
        oz2_util::transpose<float>(m, k, B, ldb, B_trans);
        B_use = B_trans;
        ldb_use = n;
    }
    if (fastmode) {
        oz2_util::vecnorm::scaling<float>(GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, k, num_moduli, A_use, lda_use, B_use, ldb_use, A8i, lda8i, sftA, B8i, ldb8i, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<float>(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, k, num_moduli, A_use, lda_use, B_use, ldb_use, A8i, lda8i, sftA, B8i, ldb8i, sftB, C32i, table_idx);
    }
#else
    if (fastmode) {
        oz2_util::vecnorm::scaling<float>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, sftA, B8i, ldb8i, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<float>(handle, op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, sftA, B8i, ldb8i, sftB, C32i, table_idx);
    }
#endif
    timing_stop(timetmp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, lda8i, &one, A8i + i * sizeA, GPU_R_8I, lda8i, B8i + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i, GPU_R_32I, m, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
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
    oz2_util::inverse_scaling(num_moduli, m, n, C8u, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
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
                                 void *const work)             // workspace
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);

    //------------------------------
    // set constants
    //------------------------------
    const size_t lda8i       = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t ldb8i       = lda8i;
    const size_t sizeA       = lda8i * m;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = ((m * n + 15) >> 4) << 4; // multiple of 16
    const size_t size_vecA   = (((m + 15) >> 4) << 4);   // multiple of 16
#ifdef DO_TRANSLATE
    const size_t size_vecB   = (((n + 15) >> 4) << 4);
#endif
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
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i + sizeC);             // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)
#ifdef DO_TRANSLATE
    TA *A_trans = reinterpret_cast<TA *>(sftB + size_vecB);
    TB *B_trans = reinterpret_cast<TB *>(A_trans);
    if (op_A == GPUBLAS_OP_N && op_B == GPUBLAS_OP_T)
        B_trans = reinterpret_cast<TB *>(A_trans + m * k);
#endif

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
#ifdef DO_TRANSLATE
    const TA* A_use = A;
    const TB* B_use = B;
    size_t lda_use = lda;
    size_t ldb_use = ldb;
    if (op_A == GPUBLAS_OP_N) {
        oz2_util::transpose<TA>(m, k, A, lda, A_trans);
        A_use = A_trans;
        lda_use = k;
    }
    if (op_B == GPUBLAS_OP_T) {
        oz2_util::transpose<TB>(m, k, B, ldb, B_trans);
        B_use = B_trans;
        ldb_use = n;
    }
    if (fastmode) {
        oz2_util::vecnorm::scaling<TA, TB>(GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, k, num_moduli, A_use, lda_use, B_use, ldb_use, A8i, lda8i, sftA, B8i, ldb8i, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<TA, TB>(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, k, num_moduli, A_use, lda_use, B_use, ldb_use, A8i, lda8i, sftA, B8i, ldb8i, sftB, C32i, table_idx);
    }
#else
    if (fastmode) {
        oz2_util::vecnorm::scaling<TA, TB>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, sftA, B8i, ldb8i, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<TA, TB>(handle, op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, sftA, B8i, ldb8i, sftB, C32i, table_idx);
    }
#endif
    timing_stop(timetmp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, lda8i, &one, A8i + i * sizeA, GPU_R_8I, lda8i, B8i + i * sizeB, GPU_R_8I, ldb8i, &zero, C32i, GPU_R_32I, m, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
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
    oz2_util::inverse_scaling<TC>(is_numM_1, num_moduli, m, n, C8u, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
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
                                 void *const work)                     // workspace
{
    return gemm_mixed<double, float, double>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work);
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
                                 void *const work)                     // workspace
{
    return gemm_mixed<float, double, double>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work);
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
                                 void *const work)                    // workspace
{
    return gemm_mixed<double, float, float>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work);
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
                                 void *const work)                    // workspace
{
    return gemm_mixed<float, double, float>(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, num_moduli, fastmode, work);
}

} // namespace gemmul8
