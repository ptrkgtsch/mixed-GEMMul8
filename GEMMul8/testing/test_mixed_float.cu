#include "../include/gemmul8.hpp"
#include "eval.hpp"
#include "getWatt.hpp"
#include "make_matrix.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <nvml.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define AVERAGE    100
#define SEED       123456
#define PHI        0.0, 0.5, 1, 1.5
#define SIZE       1024, 2048, 4096, 8192//, 16384
#define NUM_MODULI 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

#if defined(cuMpSGEMM_FLAG)
    #include "cumpsgemm/cumpsgemm.hpp"
void gemm_FP16TCEC_SCALING(cumpsgemm::handle_t const cuMpSGEMM_handle,
                           const cublasOperation_t op_A,
                           const cublasOperation_t op_B,
                           const unsigned m,
                           const unsigned n,
                           const unsigned k,
                           const float alpha,
                           float *const a_ptr,
                           const unsigned lda,
                           float *const b_ptr,
                           const unsigned ldb,
                           const float beta,
                           float *const c_ptr,
                           const unsigned ldc) {
    const cuMpSGEMM_compute_mode_t compute_mode = CUMPSGEMM_FP16TCEC;
    unsigned module_stage                       = 0;

    cumpsgemm::exp_stats_ext<float>(cuMpSGEMM_handle, (op_A == CUBLAS_OP_N ? m : k), (op_A == CUBLAS_OP_N ? k : m), a_ptr, lda);
    unsigned exp_stats_id_A = cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
    cumpsgemm::scale_A<float>(cuMpSGEMM_handle, exp_stats_id_A, 1, (op_A == CUBLAS_OP_N ? m : k), (op_A == CUBLAS_OP_N ? k : m), a_ptr, lda);

    cumpsgemm::exp_stats_ext<float>(cuMpSGEMM_handle, (op_B == CUBLAS_OP_N ? k : n), (op_B == CUBLAS_OP_N ? n : k), b_ptr, ldb);
    unsigned exp_stats_id_B = cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
    cumpsgemm::scale_B<float>(cuMpSGEMM_handle, exp_stats_id_B, 1, (op_B == CUBLAS_OP_N ? k : n), (op_B == CUBLAS_OP_N ? n : k), b_ptr, ldb);

    cumpsgemm::gemm<float>(cuMpSGEMM_handle, op_A, op_B, m, n, k, &alpha, a_ptr, lda, b_ptr, ldb, &beta, c_ptr, ldc, compute_mode, &module_stage);

    cumpsgemm::scale_C<float>(cuMpSGEMM_handle, exp_stats_id_A, exp_stats_id_B, 1, m, n, c_ptr, ldc);

    cumpsgemm::reset_scale_A<float>(cuMpSGEMM_handle, exp_stats_id_A, 1, (op_A == CUBLAS_OP_N ? m : k), (op_A == CUBLAS_OP_N ? k : m), a_ptr, lda);
    cumpsgemm::reset_scale_B<float>(cuMpSGEMM_handle, exp_stats_id_B, 1, (op_B == CUBLAS_OP_N ? k : n), (op_B == CUBLAS_OP_N ? n : k), b_ptr, ldb);
}
#endif

std::string getDeviceName() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::string deviceName = deviceProp.name;

    for (char &c : deviceName) {
        if (c == ' ' || c == '/' || c == '\\') {
            c = '_';
        }
    }
    return deviceName;
}

std::string getCurrentDateTime() {
    auto now             = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

void accuracy_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_dff_accuracy_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    std::vector<float> phi_list{PHI};
    std::vector<size_t> k_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const size_t m = 1024;
    const size_t n = 1024;

    //--------------------
    // workspace
    //--------------------
    const size_t k_max          = *max_element(begin(k_list), end(k_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *workd_cpu           = new double[m * n];
    float *workf_cpu            = new float[m * n];
    size_t worksize             = gemmul8::workSize(m, n, k_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, (m * k_max + k_max * n + m * n) * (sizeof(double) + sizeof(float)));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    outFile << "phi,function,";
    std::cout << "phi,function,";
    for (auto &moduli : num_moduli_list) {
        outFile << moduli << ",";
        std::cout << moduli << ",";
    }
    outFile << std::endl;
    std::cout << std::endl;

    for (auto &phi : phi_list) {
        for (auto &k : k_list) {
            double *cpuCd = workd_cpu;
            float *cpuCf  = workf_cpu;
            double *devAd = reinterpret_cast<double *>(work_gpu);
            double *devBd = devAd + m * k;
            double *devCd = devBd + k * n;
            float *devAf  = reinterpret_cast<float *>(devCd + m * n);
            float *devBf  = devAf + m * k;
            float *devCf  = devBf + k * n;
            double errmax, errmed;

            //--------------------
            // generate matrices
            //--------------------
            makemat::randmat<double>(m, k, devAd, phi, seed);
            makemat::d2f(m, k, devAd, devAf);
            makemat::randmat<float>(k, n, devBf, phi, seed);

            //--------------------
            // C1+C2 := A*B by FP64
            //--------------------
            double alpha = 1.0;
            double beta  = 0.0;
            makemat::f2d(k, n, devBf, devBd);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devAd, CUDA_R_64F, m, devBd, CUDA_R_64F, k, &beta, devCd, CUDA_R_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCd, devCd, m * n * sizeof(double), cudaMemcpyDeviceToHost);

            //--------------------
            // C := A*B by FP32
            //--------------------
            float alphaf = 1.0f;
            float betaf  = 0.0f;
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_R_32F, m, devBf, CUDA_R_32F, k, &betaf, devCf, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);

            outFile << phi << ",SGEMM (k=" + std::to_string(k) + "),";
            std::cout << phi << ",SGEMM (k=" + std::to_string(k) + "),";
            for (int i = 0; i < num_moduli_list.size(); ++i) {
                outFile << std::scientific << errmax << ",";
                std::cout << std::scientific << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

            //--------------------
            // C := A*B by FP32 with CUBLAS_COMPUTE_32F_FAST_TF32
            //--------------------
            cudaDeviceSynchronize();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_R_32F, m, devBf, CUDA_R_32F, k, &betaf, devCf, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);

            outFile << phi << ",SGEMM-TF32 (k=" + std::to_string(k) + "),";
            std::cout << phi << ",SGEMM-TF32 (k=" + std::to_string(k) + "),";
            for (int i = 0; i < num_moduli_list.size(); ++i) {
                outFile << std::scientific << errmax << ",";
                std::cout << std::scientific << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            outFile << phi << ",OS2-fast (k=" + std::to_string(k) + "),";
            std::cout << phi << ",OS2-fast (k=" + std::to_string(k) + "),";
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> timestmp(4, 0);
                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<double, float, float>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAd, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);
                outFile << std::scientific << errmax << ",";
                std::cout << std::scientific << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;

            //--------------------
            // C := A*B by ozaki-scheme2
            //--------------------
            outFile << phi << ",OS2-accu (k=" + std::to_string(k) + "),";
            std::cout << phi << ",OS2-accu (k=" + std::to_string(k) + "),";
            for (auto &num_moduli : num_moduli_list) {
                std::vector<double> timestmp(4);
                cudaDeviceSynchronize();
                timestmp = gemmul8::gemm<double, float, float>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAd, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
                cudaDeviceSynchronize();
                cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);
                outFile << std::scientific << errmax << ",";
                std::cout << std::scientific << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;
        }
    }

    delete[] workd_cpu;
    delete[] workf_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void time_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_dff_time_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "TFLOPS,"
            << "total_time [sec],"
            << "conv_32f_2_8i,"
            << "cublasGemmEx,"
            << "conv_32i_2_8u,"
            << "inverse_scaling,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "TFLOPS,"
              << "total_time [sec],"
              << "conv_32f_2_8i,"
              << "cublasGemmEx,"
              << "conv_32i_2_8u,"
              << "inverse_scaling,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const float phi         = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const int itermax = AVERAGE;

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *workd_cpu           = new double[n_max * n_max];
    float *workf_cpu            = new float[n_max * n_max];
    size_t worksize             = gemmul8::workSize(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * 3 * sizeof(float) + n_max * n_max * sizeof(double));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m           = n;
        size_t k           = n;
        double *cpuCd      = workd_cpu;
        float *cpuCf       = workf_cpu;
        double *devAd       = reinterpret_cast<double *>(work_gpu);
        float *devAf       = reinterpret_cast<float *>(devAd);
        float *devBf       = devAf + m * k;
        float *devCf       = devBf + k * n;
        const size_t lda8i = ((k + 15) >> 4) << 4;
        const size_t ldb8i = lda8i;
        int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
        int8_t *B8i        = A8i + lda8i * m;
        int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
        double maxerr = 0.0, mederr = 0.0;
        double time = 0.0;
        std::chrono::system_clock::time_point start, stop;

        //--------------------
        // generate matrices
        //--------------------
        makemat::randmat<double>(m, k, devAd, phi, seed);
        makemat::d2f(m, k, devAd, devAf);
        makemat::randmat<float>(k, n, devBf, phi, seed);

        //--------------------
        // C1+C2 := A*B by FP64
        //--------------------
        void *workd_gpu;
        cudaMalloc(&workd_gpu, (m * k + k * n + m * n) * sizeof(double));
        double *devBd = reinterpret_cast<double *>(workd_gpu);
        double *devCd = devBd + k * n;
        makemat::f2d(k, n, devBf, devBd);

        double alpha = 1.0;
        double beta  = 0.0;
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devAd, CUDA_R_64F, m, devBd, CUDA_R_64F, k, &beta, devCd, CUDA_R_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCd, devCd, m * n * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(workd_gpu);

        //--------------------
        // C := A*B (int8-TC)
        //--------------------
        makemat::ones(lda8i * m + ldb8i * n, A8i);
        int32_t ialpha = 1;
        int32_t ibeta  = 0;
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &ibeta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &ibeta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        outFile << std::scientific << "," << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        std::cout << std::scientific << "," << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

        //--------------------
        // C := A*B by FP32
        //--------------------
        float alphaf = 1.0f;
        float betaf  = 0.0f;
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_R_32F, m, devBf, CUDA_R_32F, k, &betaf, devCf, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_R_32F, m, devBf, CUDA_R_32F, k, &betaf, devCf, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "SGEMM" << ",";
        outFile << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "SGEMM" << ",";
        std::cout << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

        //--------------------
        // C := A*B by FP32 with CUBLAS_COMPUTE_32F_FAST_TF32
        //--------------------
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_R_32F, m, devBf, CUDA_R_32F, k, &betaf, devCf, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAf, CUDA_R_32F, m, devBf, CUDA_R_32F, k, &betaf, devCf, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "SGEMM-TF32" << ",";
        outFile << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "SGEMM-TF32" << ",";
        std::cout << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

#if defined(cuMpSGEMM_FLAG)
        cumpsgemm::handle_t cuMpSGEMM_handle;
        cumpsgemm::create(cuMpSGEMM_handle);
        cudaDeviceSynchronize();

        gemm_FP16TCEC_SCALING(cuMpSGEMM_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphaf, devAf, m, devBf, k, betaf, devCf, m);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            cudaDeviceSynchronize();
            start = std::chrono::system_clock::now();
            gemm_FP16TCEC_SCALING(cuMpSGEMM_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphaf, devAf, m, devBf, k, betaf, devCf, m);
            cudaDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "FP16TCEC_SCALING" << ",";
        outFile << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "FP16TCEC_SCALING" << ",";
        std::cout << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

        cumpsgemm::destroy(cuMpSGEMM_handle);
#endif

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            std::vector<double> times(4, 0);
            std::vector<double> timestmp(4, 0);

            cudaDeviceSynchronize();
            timestmp = gemmul8::gemm<double, float, float>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAd, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start    = std::chrono::system_clock::now();
                timestmp = gemmul8::gemm<double, float, float>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAd, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                for (int j = 0; j < 4; ++j) times[j] += timestmp[j];
            }
            time = time / itermax * 1.e-9;
            for (int j = 0; j < 4; ++j) times[j] = times[j] / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            outFile << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            std::cout << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
        }

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            std::vector<double> times(4, 0);
            std::vector<double> timestmp(4);

            cudaDeviceSynchronize();
            timestmp = gemmul8::gemm<double, float, float>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAd, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                cudaDeviceSynchronize();
                start    = std::chrono::system_clock::now();
                timestmp = gemmul8::gemm<double, float, float>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAd, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
                cudaDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                for (int j = 0; j < 4; ++j) times[j] += timestmp[j];
            }
            time = time / itermax * 1.e-9;
            for (int j = 0; j < 4; ++j) times[j] = times[j] / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            outFile << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            std::cout << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << times[0] << "," << times[1] << "," << times[2] << "," << times[3] << "," << std::endl;
        }
    }

    delete[] workd_cpu;
    delete[] workf_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

void watt_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_dff_watt_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "watt,"
            << "GFLOPS/watt,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "watt,"
              << "GFLOPS/watt,"
              << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const float phi         = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *workd_cpu           = new double[n_max * n_max];
    float *workf_cpu            = new float[n_max * n_max];
    size_t worksize             = gemmul8::workSize(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    cudaMalloc(&work_gpu, n_max * n_max * 3 * (sizeof(double) + sizeof(float)));
    cudaDeviceSynchronize();
    void *work_gemm;
    cudaMalloc(&work_gemm, worksize);
    cudaDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m           = n;
        size_t k           = n;
        double *cpuCd      = workd_cpu;
        float *cpuCf       = workf_cpu;
        double *devAd       = reinterpret_cast<double *>(work_gpu);
        float *devAf       = reinterpret_cast<float *>(devAd);
        float *devBf       = devAf + m * k;
        float *devCf       = devBf + k * n;
        const size_t lda8i = ((k + 15) >> 4) << 4;
        const size_t ldb8i = lda8i;
        int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
        int8_t *B8i        = A8i + lda8i * m;
        int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
        double maxerr = 0.0, mederr = 0.0;

        //--------------------
        // generate matrices
        //--------------------
        makemat::randmat<double>(m, k, devAd, phi, seed);
        makemat::d2f(m, k, devAd, devAf);
        makemat::randmat<float>(k, n, devBf, phi, seed);

        //--------------------
        // C1+C2 := A*B by FP64
        //--------------------
        void *workd_gpu;
        cudaMalloc(&workd_gpu, (m * k + k * n + m * n) * sizeof(double));
        double *devBd = reinterpret_cast<double *>(workd_gpu);;
        double *devCd = devBd + k * n;
        makemat::f2d(m, k, devAf, devAd);
        makemat::f2d(k, n, devBf, devBd);

        double alpha = 1.0;
        double beta  = 0.0;
        cudaDeviceSynchronize();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devAd, CUDA_R_64F, m, devBd, CUDA_R_64F, k, &beta, devCd, CUDA_R_64F, m, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCd, devCd, m * n * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(workd_gpu);

        //--------------------
        // C := A*B (int8-TC)
        //--------------------
        makemat::ones(lda8i * m + ldb8i * n, A8i);
        int32_t ialpha          = 1;
        int32_t ibeta           = 0;
        std::vector<double> res = getWatt::getWatt(
            [&]() {
                cublasGemmEx(handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             m,
                             n,
                             lda8i,
                             &ialpha,
                             A8i,
                             CUDA_R_8I,
                             lda8i,
                             B8i,
                             CUDA_R_8I,
                             ldb8i,
                             &ibeta,
                             C32i,
                             CUDA_R_32I,
                             m,
                             CUBLAS_COMPUTE_32I,
                             CUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);

        outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        outFile << std::scientific << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        std::cout << std::scientific << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

        //--------------------
        // C := A*B by FP32
        //--------------------
        float alphaf = 1.0f;
        float betaf  = 0.0f;
        res          = getWatt::getWatt(
            [&]() {
                cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m,
                             n,
                             k,
                             &alphaf,
                             devAf,
                             CUDA_R_32F,
                             m,
                             devBf,
                             CUDA_R_32F,
                             k,
                             &betaf,
                             devCf,
                             CUDA_R_32F,
                             m,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "SGEMM" << ",";
        outFile << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "SGEMM" << ",";
        std::cout << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

        //--------------------
        // C := A*B by FP32 with CUBLAS_COMPUTE_32F_FAST_TF32
        //--------------------
        res = getWatt::getWatt(
            [&]() {
                cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m,
                             n,
                             k,
                             &alphaf,
                             devAf,
                             CUDA_R_32F,
                             m,
                             devBf,
                             CUDA_R_32F,
                             k,
                             &betaf,
                             devCf,
                             CUDA_R_32F,
                             m,
                             CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "SGEMM-TF32" << ",";
        outFile << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "SGEMM-TF32" << ",";
        std::cout << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

#if defined(cuMpSGEMM_FLAG)
        cumpsgemm::handle_t cuMpSGEMM_handle;
        cumpsgemm::create(cuMpSGEMM_handle);
        cudaDeviceSynchronize();

        res = getWatt::getWatt(
            [&]() {
                gemm_FP16TCEC_SCALING(cuMpSGEMM_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphaf, devAf, m, devBf, k, betaf, devCf, m);
            },
            m,
            n,
            k);

        cudaDeviceSynchronize();
        cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "FP16TCEC_SCALING" << ",";
        outFile << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "FP16TCEC_SCALING" << ",";
        std::cout << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

        cumpsgemm::destroy(cuMpSGEMM_handle);
#endif

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {

            cudaDeviceSynchronize();
            res = getWatt::getWatt(
                [&]() {
                    gemmul8::gemm<double, float, float>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAd, m, devBf, k, &betaf, devCf, m, num_moduli, true, work_gemm);
                },
                m,
                n,
                k);

            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            outFile << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            std::cout << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        }

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            cudaDeviceSynchronize();

            res = getWatt::getWatt(
                [&]() {
                    gemmul8::gemm<double, float, float>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alphaf, devAd, m, devBf, k, &betaf, devCf, m, num_moduli, false, work_gemm);
                },
                m,
                n,
                k);

            cudaDeviceSynchronize();
            cudaMemcpy(cpuCf, devCf, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuCf, cpuCd, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            outFile << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            std::cout << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        }
    }

    delete[] workd_cpu;
    delete[] workf_cpu;
    cudaFree(work_gpu);
    cudaFree(work_gemm);
    cublasDestroy(handle);
    outFile.close();
}

int main(int argc, char **argv) {
    std::string deviceName = getDeviceName();
    std::string dateTime   = getCurrentDateTime();

    bool run_accuracy = false;
    bool run_flops    = false;
    bool run_watt     = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "accuracy_check") {
            run_accuracy = true;
        } else if (arg == "flops_check") {
            run_flops = true;
        } else if (arg == "watt_check") {
            run_watt = true;
        } else if (arg == "all") {
            run_accuracy = true;
            run_flops    = true;
            run_watt     = true;
        }
    }

    if (run_accuracy)
        accuracy_check(deviceName, dateTime);
    if (run_flops)
        time_check(deviceName, dateTime);
    if (run_watt)
        watt_check(deviceName, dateTime);

    return 0;
}
