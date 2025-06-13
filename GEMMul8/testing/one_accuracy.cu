#include <iostream>

#include "../include/gemmul8.hpp"
#include "eval.hpp"
#include "make_matrix.hpp"
#include "gpu_arch.hpp"
#define NUM_MOD 15
#define PHI 0.5

int main() {
    gpublasHandle_t handle;
    gpublasCreate(&handle);

    unsigned long long seed = 123456;
    int m = 1024;
    int k = m;
    int n = m;
    double *workd_cpu           = new double[m * n];
    float *workf_cpu            = new float[m * n];
    size_t worksize = gemmul8::workSize(m, n, k, NUM_MOD);
    void *work_gpu;
    gpuMalloc(&work_gpu, (m * k + k * n + m * n) * (sizeof(float) + sizeof(double)));
    gpuDeviceSynchronize();
    void *work_gemm;
    gpuMalloc(&work_gemm, worksize);
    gpuDeviceSynchronize();

    float *devAf = reinterpret_cast<float *>(work_gpu);
    float *devBf = devAf + m * k;
    float *devCf = devBf + k * n;

    double *cpuCd = workd_cpu;
    float *cpuCf  = workf_cpu;
    double *devAd = reinterpret_cast<double *>(devCf + m * n);
    double *devBd = devAd + m * k;
    double *devCd = devBd + k * n;

    makemat::randmat<float>(m, k, devAf, PHI, seed);
    makemat::randmat<float>(k, n, devBf, PHI, seed);

    double alpha = 1.0;
    double beta  = 0.0;
    makemat::f2d(m, k, devAf, devAd);
    makemat::f2d(k, n, devBf, devBd);
    gpublasGemmEx(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devAd, GPU_R_64F, m, devBd, GPU_R_64F, k, &beta, devCd, GPU_R_64F, m, GPUBLAS_COMPUTE_64F, GPUBLAS_GEMM_DEFAULT);
    gpuDeviceSynchronize();
    gpuMemcpy(cpuCd, devCd, m * n * sizeof(double), gpuMemcpyDeviceToHost);
    gpuDeviceSynchronize();

    double errmax, errmed;
    float alphaf = 1.0;
    float betaf = 0.0;

    gpuDeviceSynchronize();
    gpublasGemmEx(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alphaf, devAf, GPU_R_32F, m, devBf, GPU_R_32F, k, &betaf, devCf, GPU_R_32F, m, GPUBLAS_COMPUTE_32F, GPUBLAS_GEMM_DEFAULT);
    gpuDeviceSynchronize();
    gpuMemcpy(cpuCf, devCf, m * n * sizeof(float), gpuMemcpyDeviceToHost);
    gpuDeviceSynchronize();
    eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);
    std::cout << "SGEMM: " << std::scientific << errmax << " " << errmed << std::endl;

    gpuDeviceSynchronize();
    gemmul8::gemm<float>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m,
                         NUM_MOD, true, work_gemm);
    gpuDeviceSynchronize();
    gpuMemcpy(cpuCf, devCf, m * n * sizeof(float), gpuMemcpyDeviceToHost);
    gpuDeviceSynchronize();

    eval::err::gemm_err(m, n, cpuCf, cpuCd, errmax, errmed);
    std::cout << "OZ-II: " << std::scientific << errmax << " " << errmed << std::endl;

    delete[] workd_cpu;
    delete[] workf_cpu;
    gpuFree(work_gpu);
    gpuFree(work_gemm);
    gpublasDestroy(handle);

    return 0;
}
