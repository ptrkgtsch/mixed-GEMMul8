#include "../include/gemmul8.hpp"
#include "eval.hpp"
#include "getWatt.hpp"
#include "make_matrix.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "gpu_arch.hpp"

#define AVERAGE    100
#define SEED       123456
#define PHI        0.5, 1, 2, 3, 4
#define SIZE       1024, 2048, 4096, 8192//, 16384
#define NUM_MODULI 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

#if defined(ozIMMU_EF_FLAG) && defined(__NVCC__)
    #include "ozimmu/ozimmu.hpp"
std::vector<mtk::ozimmu::compute_mode_t> mode_list{
    mtk::ozimmu::fp64_int8_3,
    mtk::ozimmu::fp64_int8_4,
    mtk::ozimmu::fp64_int8_5,
    mtk::ozimmu::fp64_int8_6,
    mtk::ozimmu::fp64_int8_7,
    mtk::ozimmu::fp64_int8_8,
    mtk::ozimmu::fp64_int8_9,
    mtk::ozimmu::fp64_int8_10,
    mtk::ozimmu::fp64_int8_11,
    mtk::ozimmu::fp64_int8_12,
    mtk::ozimmu::fp64_int8_13,
    mtk::ozimmu::fp64_int8_14,
    mtk::ozimmu::fp64_int8_15,
    mtk::ozimmu::fp64_int8_16};
#endif

std::string getDeviceName() {
    gpuDeviceProp deviceProp;
    gpuGetDeviceProperties(&deviceProp, 0);
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
    std::string fileName = "oz2_results_dfd_accuracy_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    gpublasHandle_t handle;
    gpublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    std::vector<double> phi_list{PHI};
    std::vector<size_t> k_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const size_t m = 1024;
    const size_t n = 1024;

    //--------------------
    // workspace
    //--------------------
    const size_t k_max          = *max_element(begin(k_list), end(k_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *work_cpu            = new double[m * n * 3];
    size_t worksize             = gemmul8::workSize(m, n, k_max, num_moduli_max);
    void *work_gpu;
    gpuMalloc(&work_gpu, (m * k_max + k_max * n + m * n * 2) * sizeof(double) + (k_max * n) * sizeof(float));
    gpuDeviceSynchronize();
    void *work_gemm;
    gpuMalloc(&work_gemm, worksize);
    gpuDeviceSynchronize();

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
            double *cpuC  = work_cpu;
            double *cpuC1 = cpuC + m * n;
            double *cpuC2 = cpuC1 + m * n;
            double *devA  = reinterpret_cast<double *>(work_gpu);
            float *devB  = reinterpret_cast<float *>(devA + m * k);
            double *devBd = reinterpret_cast<double *>(devB + k * n);
            double *devC  = devBd + k * n;
            double *devC1 = devC;
            double *devC2 = devC1 + m * n;
            double errmax, errmed;

            //--------------------
            // generate matrices
            //--------------------
            makemat::randmat<double>(m, k, devA, phi, seed);
            makemat::randmat<float>(k, n, devB, phi, seed);

            //--------------------
            // C1+C2 := A*B (double-double arithmetic)
            //--------------------
            makemat::f2d(k, n, devB, devBd);
            eval::dd_gpu::simple_gemm(m, n, k, devA, devBd, devC1, devC2);
            gpuDeviceSynchronize();
            gpuMemcpy(cpuC1, devC1, m * n * sizeof(double), gpuMemcpyDeviceToHost);
            gpuMemcpy(cpuC2, devC2, m * n * sizeof(double), gpuMemcpyDeviceToHost);

            //--------------------
            // C := A*B by FP64
            //--------------------
            double alpha = 1.0;
            double beta  = 0.0;
            gpuDeviceSynchronize();
            gpublasGemmEx(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, GPU_R_64F, m, devBd, GPU_R_64F, k, &beta, devC, GPU_R_64F, m, GPUBLAS_COMPUTE_64F, GPUBLAS_GEMM_DEFAULT);
            gpuDeviceSynchronize();
            gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);

            outFile << phi << ",DGEMM (k=" + std::to_string(k) + "),";
            std::cout << phi << ",DGEMM (k=" + std::to_string(k) + "),";
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
                gpuDeviceSynchronize();
                timestmp = gemmul8::gemm<double, float, double>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                gpuDeviceSynchronize();
                gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
                gpuDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);
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
                gpuDeviceSynchronize();
                timestmp = gemmul8::gemm<double, float, double>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                gpuDeviceSynchronize();
                gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
                gpuDeviceSynchronize();
                eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, errmax, errmed);
                outFile << std::scientific << errmax << ",";
                std::cout << std::scientific << errmax << ",";
            }
            outFile << std::endl;
            std::cout << std::endl;
        }
    }

    delete[] work_cpu;
    gpuFree(work_gpu);
    gpuFree(work_gemm);
    gpublasDestroy(handle);
    outFile.close();
}

void time_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_dfd_time_" + deviceName + "_" + dateTime + ".csv";
    std::ofstream outFile(fileName);
    outFile << "phi,m,n,k,"
            << "function,"
            << "relerr_max,relerr_med,"
            << "TFLOPS,"
            << "total_time [sec],"
            << "conv_64&32f_2_8i,"
            << "gpublasGemmEx,"
            << "conv_32i_2_8u,"
            << "inverse_scaling,"
            << std::endl;
    std::cout << "phi,m,n,k,"
              << "function,"
              << "relerr_max,relerr_med,"
              << "TFLOPS,"
              << "total_time [sec],"
              << "conv_64&32f_2_8i,"
              << "gpublasGemmEx,"
              << "conv_32i_2_8u,"
              << "inverse_scaling,"
              << std::endl;
    gpublasHandle_t handle;
    gpublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};
    const int itermax = AVERAGE;

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *work_cpu            = new double[n_max * n_max * 3];
    size_t worksize             = gemmul8::workSize(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    gpuMalloc(&work_gpu, n_max * n_max * sizeof(double) * ((num_moduli_max >= 5) ? 3 : 4) + n_max * n_max * sizeof(float));
    gpuDeviceSynchronize();
    void *work_gemm;
    gpuMalloc(&work_gemm, worksize);
    gpuDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m           = n;
        size_t k           = n;
        double *cpuC       = work_cpu;
        double *cpuC1      = cpuC + m * n;
        double *cpuC2      = cpuC1 + m * n;
        double *devA       = reinterpret_cast<double *>(work_gpu);
        float *devB        = reinterpret_cast<float *>(devA + m * k);
        double *devBd      = reinterpret_cast<double *>(devB + k * n);
        double *devC       = devBd + k * n;
        double *devC1      = devC;
        double *devC2      = (num_moduli_max >= 5) ? (reinterpret_cast<double *>(work_gemm)) : (devC1 + m * n);
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
        makemat::randmat<double>(m, k, devA, phi, seed);
        makemat::randmat<float>(k, n, devB, phi, seed);

        //--------------------
        // C1+C2 := A*B (double-double arithmetic)
        //--------------------
        makemat::f2d(k, n, devB, devBd);
        eval::dd_gpu::simple_gemm(m, n, k, devA, devBd, devC1, devC2);
        gpuDeviceSynchronize();
        gpuMemcpy(cpuC1, devC1, m * n * sizeof(double), gpuMemcpyDeviceToHost);
        gpuMemcpy(cpuC2, devC2, m * n * sizeof(double), gpuMemcpyDeviceToHost);

        //--------------------
        // C := A*B (int8-TC)
        //--------------------
        makemat::ones(lda8i * m + ldb8i * n, A8i);
        int32_t ialpha = 1;
        int32_t ibeta  = 0;
        gpuDeviceSynchronize();
        gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, GPU_R_8I, lda8i, B8i, GPU_R_8I, ldb8i, &ibeta, C32i, GPU_R_32I, m, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
        gpuDeviceSynchronize();
        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            gpuDeviceSynchronize();
            start = std::chrono::system_clock::now();
            gpublasGemmEx(handle, GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, lda8i, &ialpha, A8i, GPU_R_8I, lda8i, B8i, GPU_R_8I, ldb8i, &ibeta, C32i, GPU_R_32I, m, GPUBLAS_COMPUTE_32I, GPUBLAS_GEMM_DEFAULT);
            gpuDeviceSynchronize();
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
        // C := A*B by FP64
        //--------------------
        double alpha = 1.0;
        double beta  = 0.0;
        gpuDeviceSynchronize();
        gpublasGemmEx(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, GPU_R_64F, m, devBd, GPU_R_64F, k, &beta, devC, GPU_R_64F, m, GPUBLAS_COMPUTE_64F, GPUBLAS_GEMM_DEFAULT);
        gpuDeviceSynchronize();
        gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
        gpuDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

        time = 0.0;
        for (int iter = 0; iter < itermax; ++iter) {
            gpuDeviceSynchronize();
            start = std::chrono::system_clock::now();
            gpublasGemmEx(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, GPU_R_64F, m, devBd, GPU_R_64F, k, &beta, devC, GPU_R_64F, m, GPUBLAS_COMPUTE_64F, GPUBLAS_GEMM_DEFAULT);
            gpuDeviceSynchronize();
            stop = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }
        time = time / itermax * 1.e-9;

        outFile << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
        outFile << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                << "," << "," << "," << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
        std::cout << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                  << "," << "," << "," << "," << std::endl;

        //--------------------
        // C := A*B by ozIMMU_EF
        //--------------------
#if defined(ozIMMU_EF_FLAG) && defined(__NVCC__)
        mtk::ozimmu::gemm_list_t fp64in_gemm;
        fp64in_gemm.push_back({mtk::ozimmu::op_n, mtk::ozimmu::op_n, m, n, k, mtk::ozimmu::real, mtk::ozimmu::fp64_int8_16});
        mtk::ozimmu::handle_t ozimmu_handle;
        mtk::ozimmu::create(&ozimmu_handle);
        mtk::ozimmu::reallocate_working_memory(ozimmu_handle, fp64in_gemm);
        int slice = 3;
        for (auto &mode : mode_list) {
            gpuDeviceSynchronize();
            mtk::ozimmu::gemm(ozimmu_handle,
                              mtk::ozimmu::op_n,
                              mtk::ozimmu::op_n,
                              m,
                              n,
                              k,
                              &alpha,
                              devA,
                              m,
                              devB,
                              k,
                              &beta,
                              devC,
                              m,
                              mode,
                              mtk::ozimmu::real);
            gpuDeviceSynchronize();
            gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                gpuDeviceSynchronize();
                start = std::chrono::system_clock::now();
                mtk::ozimmu::gemm(ozimmu_handle,
                                  mtk::ozimmu::op_n,
                                  mtk::ozimmu::op_n,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  devA,
                                  m,
                                  devB,
                                  k,
                                  &beta,
                                  devC,
                                  m,
                                  mode,
                                  mtk::ozimmu::real);
                gpuDeviceSynchronize();
                stop = std::chrono::system_clock::now();
                time += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            }
            time = time / itermax * 1.e-9;

            outFile << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
            outFile << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                    << "," << "," << "," << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
            std::cout << std::scientific << maxerr << "," << mederr << "," << 2.0 * m * n * k / time * 1.e-12 << "," << time << ","
                      << "," << "," << "," << "," << std::endl;

            slice++;
        }
        mtk::ozimmu::destroy(ozimmu_handle);
#endif

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            std::vector<double> times(4, 0);
            std::vector<double> timestmp(4, 0);

            gpuDeviceSynchronize();
            timestmp = gemmul8::gemm<double, float, double>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
            gpuDeviceSynchronize();
            gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                gpuDeviceSynchronize();
                start    = std::chrono::system_clock::now();
                timestmp = gemmul8::gemm<double, float, double>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                gpuDeviceSynchronize();
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

            gpuDeviceSynchronize();
            timestmp = gemmul8::gemm<double, float, double>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
            gpuDeviceSynchronize();
            gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            time = 0.0;
            for (int iter = 0; iter < itermax; ++iter) {
                gpuDeviceSynchronize();
                start    = std::chrono::system_clock::now();
                timestmp = gemmul8::gemm<double, float, double>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                gpuDeviceSynchronize();
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

    delete[] work_cpu;
    gpuFree(work_gpu);
    gpuFree(work_gemm);
    gpublasDestroy(handle);
    outFile.close();
}

void watt_check(std::string &deviceName, std::string &dateTime) {
    std::string fileName = "oz2_results_dfd_watt_" + deviceName + "_" + dateTime + ".csv";
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
    gpublasHandle_t handle;
    gpublasCreate(&handle);

    //--------------------
    // settings
    //--------------------
    unsigned long long seed = SEED;
    const double phi        = 0.5;
    std::vector<size_t> n_list{SIZE};
    std::vector<unsigned> num_moduli_list{NUM_MODULI};

    //--------------------
    // workspace
    //--------------------
    const size_t n_max          = *max_element(begin(n_list), end(n_list));
    const size_t num_moduli_max = *max_element(begin(num_moduli_list), end(num_moduli_list));
    double *work_cpu            = new double[n_max * n_max * 3];
    size_t worksize             = gemmul8::workSize(n_max, n_max, n_max, num_moduli_max);
    void *work_gpu;
    gpuMalloc(&work_gpu, n_max * n_max * 5 * sizeof(double) + n_max * n_max * sizeof(float));
    gpuDeviceSynchronize();
    void *work_gemm;
    gpuMalloc(&work_gemm, worksize);
    gpuDeviceSynchronize();

    for (auto &n : n_list) {
        size_t m           = n;
        size_t k           = n;
        double *cpuC       = work_cpu;
        double *cpuC1      = cpuC + m * n;
        double *cpuC2      = cpuC1 + m * n;
        double *devA       = reinterpret_cast<double *>(work_gpu);
        float *devB       = reinterpret_cast<float *>(devA + m * k);
        double *devBd      = reinterpret_cast<double *>(devB + k * n);
        double *devC       = devBd + k * n;
        double *devC1      = devC + m * n;
        double *devC2      = devC1 + m * n;
        const size_t lda8i = ((k + 15) >> 4) << 4;
        const size_t ldb8i = lda8i;
        int8_t *A8i        = reinterpret_cast<int8_t *>(work_gemm);
        int8_t *B8i        = A8i + lda8i * m;
        int32_t *C32i      = reinterpret_cast<int32_t *>(B8i + ldb8i * n);
        double maxerr = 0.0, mederr = 0.0;

        //--------------------
        // generate matrices
        //--------------------
        makemat::randmat<double>(m, k, devA, phi, seed);
        makemat::randmat<float>(k, n, devB, phi, seed);

        //--------------------
        // C1+C2 := A*B (double-double arithmetic)
        //--------------------
        makemat::f2d(k, n, devB, devBd);
        eval::dd_gpu::simple_gemm(m, n, k, devA, devBd, devC1, devC2);
        gpuDeviceSynchronize();
        gpuMemcpy(cpuC1, devC1, m * n * sizeof(double), gpuMemcpyDeviceToHost);
        gpuMemcpy(cpuC2, devC2, m * n * sizeof(double), gpuMemcpyDeviceToHost);

        //--------------------
        // C := A*B (int8-TC)
        //--------------------
        makemat::ones(lda8i * m + ldb8i * n, A8i);
        int32_t ialpha          = 1;
        int32_t ibeta           = 0;
        std::vector<double> res = getWatt::getWatt(
            [&]() {
                gpublasGemmEx(handle,
                             GPUBLAS_OP_T,
                             GPUBLAS_OP_N,
                             m,
                             n,
                             lda8i,
                             &ialpha,
                             A8i,
                             GPU_R_8I,
                             lda8i,
                             B8i,
                             GPU_R_8I,
                             ldb8i,
                             &ibeta,
                             C32i,
                             GPU_R_32I,
                             m,
                             GPUBLAS_COMPUTE_32I,
                             GPUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);

        outFile << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        outFile << std::scientific << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "INT8-GEMM" << ",";
        std::cout << std::scientific << "," << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

        //--------------------
        // C := A*B by FP64
        //--------------------
        double alpha = 1.0;
        double beta  = 0.0;
        gpuDeviceSynchronize();
        res = getWatt::getWatt(
            [&]() {
                gpublasGemmEx(handle,
                             GPUBLAS_OP_N,
                             GPUBLAS_OP_N,
                             m,
                             n,
                             k,
                             &alpha,
                             devA,
                             GPU_R_64F,
                             m,
                             devBd,
                             GPU_R_64F,
                             k,
                             &beta,
                             devC,
                             GPU_R_64F,
                             m,
                             GPUBLAS_COMPUTE_64F,
                             GPUBLAS_GEMM_DEFAULT);
            },
            m,
            n,
            k);
        gpuDeviceSynchronize();
        gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
        gpuDeviceSynchronize();
        eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

        outFile << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
        outFile << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        std::cout << phi << "," << m << "," << n << "," << k << "," << "DGEMM" << ",";
        std::cout << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

        //--------------------
        // C := A*B by ozIMMU_EF
        //--------------------
#if defined(ozIMMU_EF_FLAG) && defined(__NVCC__)
        mtk::ozimmu::gemm_list_t fp64in_gemm;
        fp64in_gemm.push_back({mtk::ozimmu::op_n, mtk::ozimmu::op_n, m, n, k, mtk::ozimmu::real, mtk::ozimmu::fp64_int8_16});
        mtk::ozimmu::handle_t ozimmu_handle;
        mtk::ozimmu::create(&ozimmu_handle);
        mtk::ozimmu::reallocate_working_memory(ozimmu_handle, fp64in_gemm);
        int slice = 3;
        for (auto &mode : mode_list) {
            gpuDeviceSynchronize();
            res = getWatt::getWatt(
                [&]() {
                    mtk::ozimmu::gemm(ozimmu_handle,
                                      mtk::ozimmu::op_n,
                                      mtk::ozimmu::op_n,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      devA,
                                      m,
                                      devB,
                                      k,
                                      &beta,
                                      devC,
                                      m,
                                      mode,
                                      mtk::ozimmu::real);
                },
                m,
                n,
                k);
            gpuDeviceSynchronize();
            gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
            outFile << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "ozIMMU_EF-" << slice << ",";
            std::cout << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;

            slice++;
        }
        mtk::ozimmu::destroy(ozimmu_handle);
#endif

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            gpuDeviceSynchronize();

            res = getWatt::getWatt(
                [&]() {
                    gemmul8::gemm<double, float, double>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, true, work_gemm);
                },
                m,
                n,
                k);

            gpuDeviceSynchronize();
            gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            outFile << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-fast-" << num_moduli << ",";
            std::cout << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        }

        //--------------------
        // C := A*B by ozaki-scheme2
        //--------------------
        for (auto &num_moduli : num_moduli_list) {
            gpuDeviceSynchronize();

            res = getWatt::getWatt(
                [&]() {
                    gemmul8::gemm<double, float, double>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, false, work_gemm);
                },
                m,
                n,
                k);

            gpuDeviceSynchronize();
            gpuMemcpy(cpuC, devC, m * n * sizeof(double), gpuMemcpyDeviceToHost);
            gpuDeviceSynchronize();
            eval::err::gemm_err(m, n, cpuC, cpuC1, cpuC2, maxerr, mederr);

            outFile << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            outFile << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
            std::cout << phi << "," << m << "," << n << "," << k << "," << "OS2-accu-" << num_moduli << ",";
            std::cout << std::scientific << maxerr << "," << mederr << "," << res[0] << "," << res[1] * 1.e-9 << "," << std::endl;
        }
    }

    delete[] work_cpu;
    gpuFree(work_gpu);
    gpuFree(work_gemm);
    gpublasDestroy(handle);
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
