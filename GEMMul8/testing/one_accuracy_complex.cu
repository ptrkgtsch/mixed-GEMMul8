#include <iostream>

#include "../include/gemmul8.hpp"
#include "eval.hpp"
#include "make_matrix.hpp"
#include "gpu_arch.hpp"
#define NUM_MOD 15
#define PHI 0.5
#define COMPUTE_TYPE gemmul8::COMPLEX_KARATSUBA_MULT

int main() {
    gpublasHandle_t handle;
    gpublasCreate(&handle);

    unsigned long long seed = 123456;
    int m = 1024;
    int k = m;
    int n = m;
    gpuDoubleComplex *workd_cpu           = new gpuDoubleComplex[m * n];
    gpuFloatComplex *workf_cpu            = new gpuFloatComplex[m * n];
    size_t worksize = gemmul8::workSize(m, n, k, NUM_MOD, COMPUTE_TYPE);
    void *work_gpu;
    gpuMalloc(&work_gpu, (m * k + k * n + m * n) * (sizeof(gpuFloatComplex) + sizeof(gpuDoubleComplex)));
    gpuDeviceSynchronize();
    void *work_gemm;
    gpuMalloc(&work_gemm, worksize);
    gpuDeviceSynchronize();

    gpuDoubleComplex *devAd = reinterpret_cast<gpuDoubleComplex *>(work_gpu);
    gpuDoubleComplex *devBd = devAd + m * k;
    gpuDoubleComplex *devCd = devBd + k * n;
    gpuFloatComplex *devAf = reinterpret_cast<gpuFloatComplex *>(devCd + m * n);
    gpuFloatComplex *devBf = devAf + m * k;
    gpuFloatComplex *devCf = devBf + k * n;

    gpuDoubleComplex *cpuCd = workd_cpu;
    gpuFloatComplex *cpuCf  = workf_cpu;

    makemat::randmat_C<gpuFloatComplex>(m, k, devAf, PHI, seed);
    makemat::randmat_C<gpuFloatComplex>(k, n, devBf, PHI, seed);

    gpuDeviceSynchronize();
    gpuDoubleComplex alpha = make_gpuDoubleComplex(1.0, 0.0);
    gpuDoubleComplex beta  = make_gpuDoubleComplex(0.0, 0.0);
    makemat::f2d_C(m, k, devAf, devAd);
    makemat::f2d_C(k, n, devBf, devBd);
    gpuDeviceSynchronize();
    gpublasGemmEx(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alpha, devAd, GPU_C_64F, m, devBd, GPU_C_64F, k, &beta, devCd, GPU_C_64F, m, GPUBLAS_COMPUTE_64F, GPUBLAS_GEMM_DEFAULT);
    gpuDeviceSynchronize();
    gpuMemcpy(cpuCd, devCd, m * n * sizeof(gpuDoubleComplex), gpuMemcpyDeviceToHost);
    gpuDeviceSynchronize();

    double errmax, errmed;
    gpuFloatComplex alphaf = make_gpuFloatComplex(1.0, 0.0);
    gpuFloatComplex betaf = make_gpuFloatComplex(0.0, 0.0);

    gpuDeviceSynchronize();
    gpublasGemmEx(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alphaf, devAf, GPU_C_32F, m, devBf, GPU_C_32F, k, &betaf, devCf, GPU_C_32F, m, GPUBLAS_COMPUTE_32F, GPUBLAS_GEMM_DEFAULT);
    gpuDeviceSynchronize();
    gpuMemcpy(cpuCf, devCf, m * n * sizeof(gpuFloatComplex), gpuMemcpyDeviceToHost);
    gpuDeviceSynchronize();
    eval::err::gemm_err_C(m, n, cpuCf, cpuCd, errmax, errmed);
    std::cout << "SGEMM: " << std::scientific << errmax << " " << errmed << std::endl;

    gpuDeviceSynchronize();
    gemmul8::gemm<gpuFloatComplex>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m,
                         NUM_MOD, false, work_gemm, COMPUTE_TYPE);
    gpuDeviceSynchronize();
    gpuMemcpy(cpuCf, devCf, m * n * sizeof(gpuFloatComplex), gpuMemcpyDeviceToHost);
    gpuDeviceSynchronize();

    eval::err::gemm_err_C(m, n, cpuCf, cpuCd, errmax, errmed);
    std::cout << "OS-II: " << std::scientific << errmax << " " << errmed << std::endl;

    delete[] workd_cpu;
    delete[] workf_cpu;
    gpuFree(work_gpu);
    gpuFree(work_gemm);
    gpublasDestroy(handle);

    return 0;
}
