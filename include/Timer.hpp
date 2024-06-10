#pragma once

#include "cuda_runtime_api.h"

class Timer {
 public:
    Timer() {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }
    auto start() -> void { cudaEventRecord(m_start); }
    auto stop() -> void {
        cudaEventRecord(m_stop);
        cudaEventSynchronize(m_stop);
        cudaEventElapsedTime(&m_elapsedTime, m_start, m_stop);
    }
    auto getElapsedTime() -> float { return m_elapsedTime; }
    ~Timer() {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }

 private:
    cudaEvent_t m_start, m_stop;
    float m_elapsedTime{};
};
