#ifndef MAT_UTILS_H
#define MAT_UTILS_H

namespace oz2_mat_utils {

__global__ void add_int8_mat_256_kernel(const size_t sizeMat,                 // cols
                                        int8_t *const __restrict__ A8i,       // m*k matrix
                                        const int8_t *const __restrict__ B8i) // m*k matrix
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeMat) return;

    char4 in4_a = reinterpret_cast<char4 *>(A8i)[idx];
    char4 in4_b = reinterpret_cast<const char4 *>(B8i)[idx];

    int4 tmp4;
    tmp4.x = in4_a.x + in4_b.x;
    tmp4.y = in4_a.y + in4_b.y;
    tmp4.z = in4_a.z + in4_b.z;
    tmp4.w = in4_a.w + in4_b.w;

    char4 out4;
    out4.x = static_cast<int8_t>(tmp4.x);
    out4.y = static_cast<int8_t>(tmp4.y);
    out4.z = static_cast<int8_t>(tmp4.z);
    out4.w = static_cast<int8_t>(tmp4.w);
    reinterpret_cast<char4 *>(A8i)[idx] = out4;
}

__global__ void add_int8_mat_not256_kernel(const size_t sizeMat,
                                           int8_t *const __restrict__ A8i,       // m*k matrix
                                           const int8_t *const __restrict__ B8i, // m*k matrix
                                           const uint8_t modulus,                //
                                           const int32_t invm)                   // 2^32 / modulus
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeMat) return;

    char4 in4_a = reinterpret_cast<char4 *>(A8i)[idx];
    char4 in4_b = reinterpret_cast<const char4 *>(B8i)[idx];

    int4 tmp4;
    tmp4.x = in4_a.x + in4_b.x;
    tmp4.y = in4_a.y + in4_b.y;
    tmp4.z = in4_a.z + in4_b.z;
    tmp4.w = in4_a.w + in4_b.w;

    tmp4.x -= __mulhi(tmp4.x, invm) * modulus;
    tmp4.x -= (tmp4.x >= modulus / 2) * modulus;
    tmp4.x += (tmp4.x < - modulus / 2) * modulus;
    tmp4.y -= __mulhi(tmp4.y, invm) * modulus;
    tmp4.y -= (tmp4.y >= modulus / 2) * modulus;
    tmp4.y += (tmp4.y < - modulus / 2) * modulus;
    tmp4.z -= __mulhi(tmp4.z, invm) * modulus;
    tmp4.z -= (tmp4.z >= modulus / 2) * modulus;
    tmp4.z += (tmp4.z < - modulus / 2) * modulus;
    tmp4.w -= __mulhi(tmp4.w, invm) * modulus;
    tmp4.w -= (tmp4.w >= modulus / 2) * modulus;
    tmp4.w += (tmp4.w < - modulus / 2) * modulus;

    char4 out4;
    out4.x = static_cast<int8_t>(tmp4.x);
    out4.y = static_cast<int8_t>(tmp4.y);
    out4.z = static_cast<int8_t>(tmp4.z);
    out4.w = static_cast<int8_t>(tmp4.w);
    reinterpret_cast<char4 *>(A8i)[idx] = out4;
}

void add_int8_mat(const unsigned i,
                  const size_t sizeMat,
                  int8_t *const __restrict__ A8i,       // m*k stored transposed
                  const int8_t *const __restrict__ B8i) // m*k stored normal
{
    const size_t blockSize = oz2_const::threads_conv32i8u;
    const size_t numBlocks = ((sizeMat >> 2) + blockSize - 1) / blockSize;
    if (i == 0) {
        add_int8_mat_256_kernel<<<numBlocks, blockSize>>>(sizeMat >> 2, A8i, B8i);
    } else {
        const uint8_t modulus = static_cast<uint8_t>(-oz2_table::moduli[i].z);
        const int32_t invm    = oz2_table::invm_32i[i - 1];
        add_int8_mat_not256_kernel<<<numBlocks, blockSize>>>(sizeMat >> 2, A8i, B8i, modulus, invm);
    }
}

__global__ void sub_int32_mat_kernel(const size_t m,                         //
                                     const size_t k,                         //
                                     int32_t *const __restrict__ A32i,       // m*k matrix
                                     const int32_t *const __restrict__ B32i, // m*k matrix
                                     const size_t ld32i)                     // leading dimension
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * k) return;
    const auto col = idx / m;
    const auto row = idx - col * m;
    const auto mem_idx = col * ld32i + row;

    A32i[mem_idx] -= B32i[mem_idx];
}

void sub_int32_mat(const size_t m,                         //
                   const size_t k,                         //
                   int32_t *const __restrict__ A32i,       // m*k matrix
                   const int32_t *const __restrict__ B32i, // m*k matrix
                   const size_t ld32i)                     // leading dimension
{
    const size_t blockSize = oz2_const::threads_invscaling;
    const size_t numBlocks = (m * k + blockSize - 1) / blockSize;
    sub_int32_mat_kernel<<<numBlocks, blockSize>>>(m, k, A32i, B32i, ld32i);
}

}

#endif