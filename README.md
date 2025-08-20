# mixed-GEMMul8

This is an extension to [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) (GEMMulate): GEMM emulation using int8 matrix engines based on the Ozaki Scheme II developed by Yuki Uchino (yuki.uchino.fe (at) riken.jp). It adds mixed-precision multiplications, complex matrices support, HIP platforms support, and a low-memory mode in the `memory-lt` branch. It also includes a few bug fixes/optimizations.

## Build

CMake definitions were to the original project for a smoother experience. It is made to be compatible with the previous compilation method of [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8).

1. (Option) Build `cuMpSGEMM` and `ozIMMU_EF` according to [cuMpSGEMM](https://github.com/enp1s0/cuMpSGEMM) and [ozIMMU](https://github.com/enp1s0/ozIMMU) (see also [Accelerator for ozIMMU](https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU)).

2. Navigate to the `GEMMul8` directory, create a `build` (or anything else) and navigate into it.

3. Use CMake to generate the makefiles: For example, `cmake ..`, or specify all options with this AMD example: `cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUMP=OFF -DUSE_OZIMMU_EF=OFF -DCMAKE_GPU_RUNTIME:STRING=HIP -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.0/bin/hipcc -DCMAKE_PREFIX_PATH=/opt/rocm-6.4.0 ..`.
   - `CMAKE_GPU_RUNTIME` can be `CUDA` (default) or `HIP`, which tells the project if it should be compiled for NVIDIA or AMD platforms.
   - `USE_CUMP`can be ON or OFF, if cuMpSGEMM should be used in the test suite. It looks at this project's `cuMpSGEMM/build` folder to find the library, `CUMP_DIR` can me modified in `GEMMul8/testing/CMakeLists.txt` can be modified if it is somewhere else.
   - `USE_OZIMMU_EF` can be ON or OFF, if ozIMMU_EF should be used in the test suite. Similar to the previous point.

4. Compile the library and all tests with `make`, or a specific target with `cmake --build . --target test_float -- -j 6`

5. Navigate to the `testing` directory inside your `build` folder and then run following commands to run sample codes.
   - `./execurable_name args...`. The args in the tests can be any combination of `accuracy_check`, `flops_check`, `watt_check`, or use `all` to run everything.
   - test list, some parameters can be changed as macroes inside the source files:
      - `test_float` tests SGEMM emulation
      - `test_double` tests DGEMM emulation
      - `test_mixed_double` tests FP64\*FP32->FP64 emulation
      - `test_mixed_float` tests FP64\*FP32->FP32 emulation
      - `test_ffd` tests FP32\*FP32->FP64
      - `test_ddf` tests FP64\*FP64->FP32
      - `test_float_complex` tests complex SGEMM emulation
      - `one_accuracy` and `one_accuracy_complex` prints the accuracy for one call
      - `profile_one_call` is made to run profilers on a certain call

## Usage

```
//----------
// settings
//----------
const unsigned num_moduli = 14;   // 2 <= num_moduli <= 20 for DGEMM emulation
// const unsigned num_moduli = 6;   // 2 <= num_moduli <= 19 for SGEMM emulation
const bool fastmode = true;       // true (fast-mode) or false (accurate-mode)

//----------
// (if needed) allocate workspace
//----------
const size_t worksize = gemmul8::workSize(m,n,k,num_moduli, REAL_DEFAULT); // last argument optional, it is REAL_DEFAULT by default.
void *work;
cudaMalloc(&work, worksize);

//----------
// (if needed) output variable
//----------
std::vector<double> time_breakdown(4,0);

//----------
// run emulation
// gemmul8::gemm returns execution time (sec.) of each part
//----------
time_breakdown = gemmul8::gemm(cublas_handle,   // Handle to the cuBLAS library context
                               CUBLAS_OP_N,     // non- or transpose devA
                               CUBLAS_OP_N,     // non- or transpose devB
                               m,               // Number of rows of devC
                               n,               // Number of columns of devC
                               k,               // Inner dimension
                               &alpha,          // Scaling factor for devA*devB
                               devA,            // 1-D device array of dimensions lda*k (CUBLAS_OP_N) or lda*m (CUBLAS_OP_T)
                               lda,             // Leading dimension of devA
                               devB,            // 1-D device array of dimensions ldb*n (CUBLAS_OP_N) or ldb*k (CUBLAS_OP_T)
                               ldb,             // Leading dimension of devB
                               &beta,           // Scaling factor for devC
                               devC,            // 1-D device array of dimensions ldc*n
                               ldc,             // Leading dimension of devC
                               num_moduli,      // #moduli (controlling accuracy)
                               fastmode,        // Computing mode
                               work,            // workspace
                               REAL_DEFAULT);   // optional, is REAL_DEFAULT by default.
```

If you are using complex numbers, use or cast the matrices into `cuFloatComplex`/`hipFloatComplex` and `cuDoubleComplex`/`hipDoubleComplex`, and specify the last argument of type `computeType_t` with `COMPLEX_BIG_MATRIX_ENCODE`, `COMPLEX_CLASSIC_MULT`, or `COMPLEX_KARATSUBA_MULT`. For tested matrices, `COMPLEX_BIG_MATRIX_ENCODE` was always the fastest solution but uses more memory.

## Original [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) Numerical results (DGEMM emulation on GH200)

The constant $\phi$ controls the difficulty of matrix multiplication (exponent distribution of input matrices).

The difficulty of $\phi = 0.5$ is comparable to that of matrix multiplication in HPL.

### Accuracy

![accuracy_dgemm](./GEMMul8/testing/results_in_paper/fig/oz2_results_d_accuracy_NVIDIA_GH200_480GB.png)

### Throughput performance

![throughput_dgemm](./GEMMul8/testing/results_in_paper/fig/oz2_results_d_time_NVIDIA_GH200_480GB.png)

### Power efficiency

![power_dgemm](./GEMMul8/testing/results_in_paper/fig/oz2_results_d_watt_NVIDIA_GH200_480GB.png)

## Original [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) Numerical results (SGEMM emulation on GH200)

### Accuracy

![accuracy_sgemm](./GEMMul8/testing/results_in_paper/fig/oz2_results_f_accuracy_NVIDIA_GH200_480GB.png)

### Throughput performance

![throughput_sgemm](./GEMMul8/testing/results_in_paper/fig/oz2_results_f_time_NVIDIA_GH200_480GB.png)

### Power efficiency

![power_sgemm](./GEMMul8/testing/results_in_paper/fig/oz2_results_f_watt_NVIDIA_GH200_480GB.png)

## Attention

ozIMMU_EF is from [ozIMMU](https://github.com/enp1s0/ozIMMU) by Ootomo and [Accelerator for ozIMMU](https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU) by RIKEN R-CCS.

cuMpSGEMM is from [cuMpSGEMM](https://github.com/enp1s0/cuMpSGEMM) by Ootomo.

If you use these libraries, you must agree to the licenses terms of ozIMMU, Accelerator for ozIMMU, and cuMpSGEMM in addition to the license for GEMMul8.

## References

- Hiroyuki Ootomo and Rio Yokota. 2022. Recovering single precision accuracy from Tensor Cores while surpassing the FP32 theoretical peak performance. The International Journal of High Performance Computing Applications 36, 4 (2022), 475--491.
- Hiroyuki Ootomo, Hidetaka Manabe, Kenji Harada, and Rio Yokota. 2023. Quantum Circuit Simulation by SGEMM Emulation on Tensor Cores and Automatic Precision Selection. In High Performance Computing. Springer Nature Switzerland, Cham, 259--276.
- Hiroyuki Ootomo, Katsuhisa Ozaki, and Rio Yokota. 2024. DGEMM on integer matrix multiplication unit. The International Journal of High Performance Computing Applications 38, 4 (2024), 297--313.
- Yuki Uchino, Katsuhisa Ozaki, and Toshiyuki Imamura. 2025. Performance enhancement of the Ozaki Scheme on integer matrix multiplication unit. The International Journal of High Performance Computing Applications 39, 3 (2025), 462--476.

## Citation

Please use the citations from the original repository at [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8), and link to this repository.

## License

MIT
