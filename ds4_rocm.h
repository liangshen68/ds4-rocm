#pragma once

// HIP/ROCm shim for ds4_cuda.cu — lets us compile the CUDA source with hipcc
// for AMD GPUs (validated on Strix Halo / gfx1151, RDNA 3.5).
//
// Pattern: the CUDA source includes <cuda_runtime.h>, <cuda_fp16.h>, <mma.h>,
// <cublas_v2.h>, and <cub/block/block_radix_sort.cuh>. When __HIP_PLATFORM_AMD__
// is defined we substitute HIP/rocWMMA/hipCUB/hipBLAS instead and remap the
// API names via macros so the rest of the source compiles unchanged.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <rocwmma/rocwmma.hpp>
#include <hipcub/block/block_radix_sort.hpp>

// nvcuda::wmma → rocwmma
namespace nvcuda {
    namespace wmma = ::rocwmma;
}

// cub:: → hipcub:: (just an alias namespace; both expose BlockRadixSort etc)
namespace cub = hipcub;

// Runtime types & status
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaMemLocation hipMemLocation

#define cudaSuccess hipSuccess
#define cudaErrorNotSupported hipErrorNotSupported
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString

// Device query & control
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDevAttrPageableMemoryAccess hipDeviceAttributePageableMemoryAccess
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice

// Kernel function attributes (used to opt large dynamic LDS into bigger LDS slots)
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// Memory
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaFree hipFree
#define cudaFreeHost hipFreeHost
#define cudaMemset hipMemset
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemsetAsync hipMemsetAsync

#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly

#define cudaMemAdvise(p1, p2, p3, p4) hipMemAdvise(p1, p2, p3, p4.id)
#define cudaMemPrefetchAsync(devPtr, count, location, flags, stream) hipMemPrefetchAsync(devPtr, count, location.id, stream)
#define cudaMemAdviseSetReadMostly hipMemAdviseSetReadMostly
#define cudaMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation

// Streams
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking

// Events
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDisableTiming hipEventDisableTiming

// cuBLAS → hipBLAS
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasMath_t hipblasMath_t

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_TF32_TENSOR_OP_MATH HIPBLAS_TF32_TENSOR_OP_MATH
#define CUDA_R_16F HIPBLAS_R_16F
#define CUDA_R_32F HIPBLAS_R_32F

#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetMathMode hipblasSetMathMode
#define cublasSgemm hipblasSgemm
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasGemmEx hipblasGemmEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx

// Per-byte SIMD primitives that NVIDIA exposes as PTX intrinsics. AMDGCN has
// no single-instruction equivalents for these byte-mask patterns, so we
// reconstruct them with bit-twiddling.
static __device__ __forceinline__ int32_t __vcmpne4(uint32_t a, uint32_t b) {
    // For each byte: 0xFF if a != b, 0x00 if a == b.
    uint32_t diff = a ^ b;
    diff |= (diff >> 1); diff |= (diff >> 2); diff |= (diff >> 4);
    diff &= 0x01010101u;
    diff *= 0xFFu;
    return (int32_t)diff;
}

static __device__ __forceinline__ int32_t __vsub4(int32_t a, int32_t b) {
    // Per-byte wrapping subtract — avoid cross-byte borrows.
    uint32_t ua = (uint32_t)a, ub = (uint32_t)b;
    uint32_t diff = ((ua | 0x80808080u) - (ub & 0x7F7F7F7Fu)) ^ ((ua ^ ~ub) & 0x80808080u);
    return (int32_t)diff;
}

// __dp4a: 4× int8 dot product accumulated into int32.
// gfx1151 has dot8-insts (v_dot4_i32_iu8) reachable via __builtin_amdgcn_sudot4
// with signed/signed flags. This compiles to a single VALU op vs ~7 scalar ops
// in the byte-by-byte fallback — meaningful on the q8_0/IQ2_XXS hot loops.
static __device__ __forceinline__ int32_t __dp4a(int32_t a, int32_t b, int32_t c) {
    return __builtin_amdgcn_sudot4(/*a_sign=*/1, a, /*b_sign=*/1, b, c, /*clamp=*/0);
}
