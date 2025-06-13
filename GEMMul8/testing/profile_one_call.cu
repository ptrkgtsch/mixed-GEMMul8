#include "../include/gemmul8.hpp"
#include "eval.hpp"
#include "make_matrix.hpp"
#include "gpu_arch.hpp"
#if defined(__HIPCC__)
#define NB_ITER 20
#else
#define NB_ITER 1
#endif

int main() {
    gpublasHandle_t handle;
    gpublasCreate(&handle);

    unsigned long long seed = 123456;
    int m = 8192;
    int k = 8192;
    int n = 8192;
    size_t worksize = gemmul8::workSize(m, n, k, 14);
    void *work_gpu;
    gpuMalloc(&work_gpu, (m * k + k * n + m * n) * sizeof(float));
    gpuDeviceSynchronize();
    void *work_gemm;
    gpuMalloc(&work_gemm, worksize);
    gpuDeviceSynchronize();

    float *devAf = reinterpret_cast<float *>(work_gpu);
    float *devBf = devAf + m * k;
    float *devCf = devBf + k * n;
    makemat::randmat<float>(m, k, devAf, 1, seed);
    makemat::randmat<float>(k, n, devBf, 1, seed);

    float alphaf = 1.0;
    float betaf = 0.0;
    for (int i = 0; i < NB_ITER; i++) {
        gpuDeviceSynchronize();
        gemmul8::gemm<float>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N, m, n, k, &alphaf, devAf, m, devBf, k, &betaf, devCf, m,
                             14, true, work_gemm);
        gpuDeviceSynchronize();
    }

    gpuFree(work_gpu);
    gpuFree(work_gemm);
    gpublasDestroy(handle);
}
