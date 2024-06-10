#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "CG.hpp"
#include "Handler.hpp"
#include "Timer.hpp"
#include "Vector.hpp"
#include "cublas_v2.h"
#include "cuda_runtime_api.h"
#include "cusparse.h"

template <typename Type>
struct CSR {
    std::vector<Type> values;
    std::vector<int> columnIndices;
    std::vector<int> rowOffsets;
    int size;
};

template <typename Type>
CSR<Type> toCSR(Type* matrix, size_t cols, size_t rows) {
    CSR<Type> csr;

    csr.rowOffsets.reserve(rows + 1);
    csr.columnIndices.reserve(rows);  // predict
    csr.values.reserve(rows);         // predict

    csr.rowOffsets.push_back(0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (matrix[i * cols + j] != 0.f) {
                csr.values.push_back(matrix[i * cols + j]);
                csr.columnIndices.push_back(j);
            }
        }
        csr.rowOffsets.push_back(csr.columnIndices.size());
    }
    return csr;
}

template <typename Type>
class IPreconditioner {
 public:
    virtual auto inverseMatrixVectorMultiplication(Type* dev_vector, Type* dev_result) const -> void = 0;
    virtual ~IPreconditioner() = default;
};

template <typename Type>
class IdentityPreconditioner : public IPreconditioner<Type> {
 public:
    IdentityPreconditioner(int size) : m_size(size) {}

    auto inverseMatrixVectorMultiplication(Type* dev_vector, Type* dev_result) const -> void override {
        cudaMemcpy(dev_result, dev_vector, m_size * sizeof(Type), cudaMemcpyDeviceToDevice);
    };

 private:
    int m_size{};
};

template <typename Type>
class JacobiPreconditioner : public IPreconditioner<Type> {
 public:
    JacobiPreconditioner(cusparseSpMatDescr_t matrix, int size) : m_size(size) {
        // cusparseSpMatGetSize()
        // get inverseDiagonalElements
    }
    JacobiPreconditioner(Type* dev_inverseDiagonalElements, int size)
        : m_size(size), m_dev_inverseDiagonalElements(dev_inverseDiagonalElements) {}

    auto inverseMatrixVectorMultiplication(Type* dev_vector, Type* dev_result) -> void override {
        cudaMemcpy(dev_result, dev_vector, m_size * sizeof(Type), cudaMemcpyDeviceToDevice);
        Handler<Type> handler;
        // handler.vec_scal
    };

 private:
    int m_size{};
    Type* m_dev_inverseDiagonalElements;
};

template <typename Type>
void PreconditionedCG(cusparseSpMatDescr_t A, Type* dev_x, Type* dev_b, int N, const IPreconditioner<Type>& preconditioner) {
    cudaDataType dataType;
    if (std::is_same<Type, float>::value) {
        std::cout << "float" << std::endl;
        dataType = CUDA_R_32F;
    } else {
        std::cout << "double" << std::endl;
        dataType = CUDA_R_64F;
    }
    cusparseStatus_t cusparseStatus;
    cusparseHandle_t cusparseHandle;

    cusparseStatus = cusparseCreate(&cusparseHandle);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE initialization failed\n");
        return;
    }

    cusparseDnVecDescr_t x;
    cusparseStatus = cusparseCreateDnVec(&x, N, dev_x, dataType);
    // todo: init x0

    int memorySize = N * sizeof(Type);

    Type* dev_r;
    cudaMalloc(&dev_r, memorySize);
    cusparseDnVecDescr_t r;
    cusparseStatus = cusparseCreateDnVec(&r, N, dev_r, dataType);

    // r = b
    cudaMemcpy(dev_r, dev_b, memorySize, cudaMemcpyDeviceToDevice);

    Type alpha = -1;
    Type beta = 1;

    void* dBuffer = NULL;
    size_t bufferSize = 0;

    // r = -1*A . x + 1* r;
    cusparseStatus = cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, x, &beta, r, dataType,
                                             CUSPARSE_SPMV_CSR_ALG2, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    // todo: if bufferSize > prev => realloc
    cusparseStatus =
        cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, x, &beta, r, dataType, CUSPARSE_SPMV_CSR_ALG2, dBuffer);
    cudaDeviceSynchronize();
    cudaFree(dBuffer);

    Type* dev_z;
    cudaMalloc(&dev_z, memorySize);
    // zNew = invM . rNew;
    preconditioner.inverseMatrixVectorMultiplication(dev_r, dev_z);
    // фактически cudaMemcpy(dev_z, dev_r, memorySize, cudaMemcpyDeviceToDevice);

    Type* dev_p;
    cudaMalloc(&dev_p, memorySize);
    cusparseDnVecDescr_t p;
    cusparseStatus = cusparseCreateDnVec(&p, N, dev_p, dataType);
    // p = z
    cudaMemcpy(dev_p, dev_z, memorySize, cudaMemcpyDeviceToDevice);

    Handler<Type> handler;

    Type rz;  // r.z
    handler.dot(N, dev_r, dev_z, rz);

    Type r_norm, b_norm;
    handler.norm2(N, dev_r, r_norm);
    handler.norm2(N, dev_b, b_norm);

    if (r_norm / b_norm < 0.001) {
        std::cout << "Done" << std::endl;
        return;
    }

    Type* dev_Ap;
    cudaMalloc(&dev_Ap, memorySize);
    cusparseDnVecDescr_t Ap;
    cusparseStatus = cusparseCreateDnVec(&Ap, N, dev_Ap, dataType);

    int iter = 0;

    while (iter < 1000) {
        std::cout << "r_norm = " << r_norm << std::endl;
        iter += 1;

        // Ap = 1* A . p + 0*Ap;
        beta = 0.0;
        alpha = 1.0;
        cusparseStatus = cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, p, &beta, Ap, dataType,
                                                 CUSPARSE_SPMV_CSR_ALG2, &bufferSize);
        cudaMalloc(&dBuffer, bufferSize);

        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
            printf("CUSPARSE failed\n");
            return;
        }
        // todo: if bufferSize > prev => realloc
        cusparseStatus = cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, p, &beta, Ap, dataType,
                                      CUSPARSE_SPMV_CSR_ALG2, dBuffer);
        cudaDeviceSynchronize();
        cudaFree(dBuffer);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
            printf("CUSPARSE failed\n");
            return;
        }

        Type p_norm;
        handler.norm2(N, dev_p, p_norm);
        std::cout << "p_norm =" << p_norm << std::endl;

        Type App;

        Type Ap_norm;
        handler.norm2(N, dev_Ap, Ap_norm);
        std::cout << "Ap_norm = " << Ap_norm << std::endl;

        handler.dot(N, dev_Ap, dev_p, App);
        Type a = rz / App;
        std::cout << "rz = " << rz << std::endl;
        std::cout << "App = " << App << std::endl;
        // std::cout << "a = " << a << std::endl;
        //  x = x + a * p;

        handler.axpy(N, a, dev_p, dev_x);

        // r = r - a *  Ap;
        Type a_minus = -a;
        handler.axpy(N, -a, dev_Ap, dev_r);

        handler.norm2(N, dev_r, r_norm);
        if (r_norm / b_norm < 0.001) {
            std::cout << "Done on iter " << iter << std::endl;
            return;
        }

        // zNew = invM . rNew;
        preconditioner.inverseMatrixVectorMultiplication(dev_r, dev_z);

        // to do: handler operations return result value

        Type rNewzNew;  // rNew.zNew
        handler.dot(N, dev_r, dev_z, rNewzNew);

        Type b = rNewzNew / rz;

        // if (iter % 10 == 0) {
        //   b = 0.0;
        // }

        // p= b*p
        handler.scal(N, b, dev_p);
        a = 1;
        // p = z + a * p;
        handler.axpy(N, a, dev_z, dev_p);
        rz = rNewzNew;
    }
}

template <typename Type>
CSR<Type> readMTX(const std::string& fileName) {
    std::unique_ptr<std::ifstream> fileUPtr(new std::ifstream(fileName));
    if (!fileUPtr->is_open()) {
        std::cout << "Can not open file!" << std::endl;
        return {};
    }
    std::string line;
    // skip header
    do {
        std::getline(*fileUPtr, line);
    } while (line.rfind("%", 0) == 0);
    std::stringstream ss(line);
    int rows, cols, nnz;
    ss >> rows >> cols >> nnz;
    CSR<Type> csr;
    csr.size = rows;

    csr.values.reserve(nnz);
    csr.columnIndices.reserve(nnz);
    csr.rowOffsets.reserve(rows + 1);

    csr.rowOffsets.push_back(0);
    int prev_j = 1;
    for (int iter = 0; iter < nnz; ++iter) {
        int i, j;
        Type val;
        *fileUPtr >> i >> j >> val;
        if (j != prev_j) {
            prev_j = j;
            csr.rowOffsets.push_back(csr.values.size());
        }
        csr.values.push_back(val);
        csr.columnIndices.push_back(i - 1);
    }

    csr.rowOffsets.push_back(csr.values.size());
    return csr;
}

void test() {
    auto csr = readMTX<T>("bcsstk01.mtx");

    Vector<T> x(csr.size, 0.);
    x.toDevice();

    Vector<T> b(csr.size, 1.);
    b.toDevice();

    T* dev_values;
    int* dev_rowOffsets;
    int* dev_columnIndices;

    cudaMalloc(&dev_values, csr.values.size() * sizeof(T));
    cudaMalloc(&dev_rowOffsets, csr.rowOffsets.size() * sizeof(int));
    cudaMalloc(&dev_columnIndices, csr.columnIndices.size() * sizeof(int));

    cudaMemcpy(dev_values, csr.values.data(), csr.values.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_rowOffsets, csr.rowOffsets.data(), csr.rowOffsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_columnIndices, csr.columnIndices.data(), csr.columnIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

    cusparseSpMatDescr_t cusparseCSR;
    cudaDataType dataType;
    if (std::is_same<T, float>::value) {
        std::cout << "float" << std::endl;
        dataType = CUDA_R_32F;
    } else {
        std::cout << "double" << std::endl;
        dataType = CUDA_R_64F;
    }

    cusparseCreateCsr(&cusparseCSR, csr.size, csr.size, csr.values.size(), dev_rowOffsets, dev_columnIndices, dev_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dataType);
    cudaDeviceSynchronize();

    Timer timer;

    timer.start();
    IdentityPreconditioner<T> prec(csr.size);

    PreconditionedCG<T>(cusparseCSR, x.getDev(), b.getDev(), csr.size, prec);
    timer.stop();
    std::cout << "Time: " << timer.getElapsedTime() << " milliseconds " << std::endl;
    x.toHost();
    // for (int i = 0; i < csr.size; ++i) {
    //   std::cout << x.getHost()[i] << " ";
    // }
}
