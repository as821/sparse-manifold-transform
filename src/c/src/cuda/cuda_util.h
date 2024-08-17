#pragma once

#include <sys/mman.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cusparse.h>

#include "util.h"


#define CHECK_CUDA(func)                                                        \
{                                                                               \
    cudaError_t status = (func);                                                \
    if (status != cudaSuccess) {                                                \
        printf("CUDA API failed at line %d with error: %s (%d) (%s)\n",         \
               __LINE__, cudaGetErrorString(status), status, __FILE__);         \
        return EXIT_FAILURE;                                                    \
    }                                                                           \
}

#define CHECK_CUDA_NORET(func)                                                  \
{                                                                               \
    cudaError_t status = (func);                                                \
    if (status != cudaSuccess) {                                                \
        printf("CUDA API failed at line %d with error: %s (%d) (%s)\n",         \
               __LINE__, cudaGetErrorString(status), status, __FILE__);         \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}


#define CHECK_CUSPARSE(func)                                                    \
{                                                                               \
    cusparseStatus_t status = (func);                                           \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                    \
        printf("CUSPARSE API failed at line %d with error: %s (%d) (%s)\n",     \
               __LINE__, cusparseGetErrorString(status), status, __FILE__);     \
        return EXIT_FAILURE;                                                    \
    }                                                                           \
}


#define CHECK_CUSPARSE_NORET(func)                                              \
{                                                                               \
    cusparseStatus_t status = (func);                                           \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                    \
        printf("CUSPARSE API failed at line %d with error: %s (%d) (%s)\n",     \
               __LINE__, cusparseGetErrorString(status), status, __FILE__);     \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}



struct CSR {
    ssize_t nnz;
    ssize_t nrow;
    ssize_t ncol;

    ssize_t item_sz;
    ssize_t idx_sz;

    char* buf;
    float* data_ptr;
    int* indices_ptr;       // NOTE: these are 32-bit integers as required by most cuSPARSE functions
    int* indptr_ptr;
};



struct Dense {
    ssize_t nrow;
    ssize_t ncol;
    ssize_t item_sz;
    char* buf;
};

struct DenseSerialize {
    char* fname;
    ssize_t nrow;
    ssize_t ncol;
    ssize_t start; 
    ssize_t end;
};

struct CSRSerialize {
    char* fname;
    ssize_t nnz;
    ssize_t nrow;
    ssize_t ncol;
    ssize_t start; 
    ssize_t end;
};

struct CUDA_spgemm_context {
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
};

void init_spgemm_ctx(struct CUDA_spgemm_context* ctx, struct CSR* a);
void deinit_spgemm_ctx(struct CUDA_spgemm_context* ctx);

void mmap_cuda_transfer(int fd, ssize_t sz, char** dst_buf);
void cuda_mmap_write(int fd, ssize_t sz, char* cuda_buf);
void read_csr_onto_gpu(char* fname, struct CSR* csr);




