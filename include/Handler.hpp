#pragma once

#include "cuda_runtime_api.h"

namespace KernelFuctions {

    // y += alpha*x
    template <typename Type>
    __global__ auto axpy(int size, Type alpha, Type* x, Type* y) -> void {
        int myIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (myIndex < size) {
            y[myIndex] += alpha * x[myIndex];
        }
    }

    // x *= alpha
    template <typename Type>
    __global__ auto scal(int size, Type alpha, Type* x) -> void {
        int myIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (myIndex < size) {
            x[myIndex] *= alpha;
        }
    }

    // x[i] *= alpha[i]
    template <typename Type>
    __global__ auto vec_scal(int size, Type* x, Type* alpha) -> void {
        int myIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (myIndex < size) {
            x[myIndex] *= alpha[myIndex];
        }
    }

    constexpr int MAX_BLOCK_SIZE = 1024;

    template <typename Type>
    __global__ auto dot(int size, Type* x, Type* y, Type* results) -> void {
        __shared__ Type xsh[MAX_BLOCK_SIZE];
        __shared__ Type ysh[MAX_BLOCK_SIZE];

        if (blockIdx.x * blockDim.x + threadIdx.x < size) {
            xsh[threadIdx.x] = x[blockIdx.x * blockDim.x + threadIdx.x];
            ysh[threadIdx.x] = y[blockIdx.x * blockDim.x + threadIdx.x];
        } else {
            xsh[threadIdx.x] = 0;
            ysh[threadIdx.x] = 0;
        }

        __syncthreads();
        Type sum = 0.0;
        if (threadIdx.x == 0) {
            sum = 0.0;
            for (int i = 0; i < blockDim.x; ++i) {
                sum += xsh[i] * ysh[i];
            }
            results[blockIdx.x] = sum;
        }
    }

}  // namespace KernelFuctions

template <typename Type>
class Handler {
 public:
    Handler() {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        m_maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    }

    auto scal(int size, Type alpha, Type* x) -> void {
        int numOfBlocks = static_cast<int>(std::ceil(static_cast<Type>(size) / m_maxThreadsPerBlock));
        KernelFuctions::scal<<<numOfBlocks, m_maxThreadsPerBlock>>>(size, alpha, x);
        cudaDeviceSynchronize();
    }
    auto vec_scal(int size, Type alpha, Type* x) -> void {
        int numOfBlocks = static_cast<int>(std::ceil(static_cast<Type>(size) / m_maxThreadsPerBlock));
        KernelFuctions::vec_scal<<<numOfBlocks, m_maxThreadsPerBlock>>>(size, x, alpha);
        cudaDeviceSynchronize();
    }

    auto axpy(int size, Type alpha, Type* x, Type* y) -> void {
        int numOfBlocks = static_cast<int>(std::ceil(static_cast<Type>(size) / m_maxThreadsPerBlock));
        KernelFuctions::axpy<<<numOfBlocks, m_maxThreadsPerBlock>>>(size, alpha, x, y);
        cudaDeviceSynchronize();
    }

    auto dot(int size, Type* x, Type* y, Type& result) -> void {
        int numOfBlocks = static_cast<int>(std::ceil(static_cast<Type>(size) / m_maxThreadsPerBlock));
        Type* dev_results;
        cudaMalloc(&dev_results, numOfBlocks * sizeof(Type));
        KernelFuctions::dot<<<numOfBlocks, m_maxThreadsPerBlock>>>(size, x, y, dev_results);
        cudaDeviceSynchronize();
        Type* results = new Type[numOfBlocks];
        cudaMemcpy(results, dev_results, numOfBlocks * sizeof(Type), cudaMemcpyDeviceToHost);
        result = 0;
        for (int i = 0; i < numOfBlocks; ++i) {
            result += results[i];
        }
        cudaFree(dev_results);
        delete[] results;
    }

    auto norm2(int size, Type* x, Type& result) -> void {
        dot(size, x, x, result);
        result = std::sqrt(result);
    }

 private:
    int m_maxThreadsPerBlock{};
};
