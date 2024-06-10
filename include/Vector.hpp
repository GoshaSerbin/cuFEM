#pragma once

#include "cuda_runtime_api.h"

template <typename Type>
class Vector {
 public:
    Vector(int size, Type value = 0) : m_size(size) {
        m_host_vector = new Type[m_size];
        std::fill_n(m_host_vector, size, value);
        cudaMalloc(&m_dev_vector, m_size * sizeof(Type));
    }
    auto toHost() -> void { cudaMemcpy(m_host_vector, m_dev_vector, sizeof(Type) * m_size, cudaMemcpyDeviceToHost); }
    auto toDevice() -> void { cudaMemcpy(m_dev_vector, m_host_vector, sizeof(Type) * m_size, cudaMemcpyHostToDevice); }
    auto getDev() -> Type* { return m_dev_vector; }
    auto getHost() -> Type* { return m_host_vector; }

    ~Vector() {
        delete[] m_host_vector;
        cudaFree(m_dev_vector);
    }

 private:
    Type* m_dev_vector{};
    Type* m_host_vector{};
    int m_size;
};
