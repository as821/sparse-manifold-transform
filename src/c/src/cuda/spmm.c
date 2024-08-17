#include "cuda_util.h"

struct CUDA_spmm_context {
    cusparseHandle_t handle;
    cusparseDnMatDescr_t matA;
};

void init_spmm_ctx(struct CUDA_spmm_context* ctx, struct Dense* a) {
    CHECK_CUSPARSE_NORET(cusparseCreate(&ctx->handle))
    CHECK_CUSPARSE_NORET(cusparseCreateDnMat(&ctx->matA, a->nrow, a->ncol, a->ncol, a->buf, CUDA_R_32F, CUSPARSE_ORDER_ROW))
}

void deinit_spmm_ctx(struct CUDA_spmm_context* ctx) {
    CHECK_CUSPARSE_NORET(cusparseDestroyDnMat(ctx->matA))
    CHECK_CUSPARSE_NORET(cusparseDestroy(ctx->handle))
}

void read_dense_onto_gpu(char* fname, struct Dense* dense) {
    ssize_t data_bytes = dense->item_sz * dense->nrow * dense->ncol;
    int fd = open(fname, O_RDONLY);
    CHECK(fd != -1)
    mmap_cuda_transfer(fd, data_bytes, &dense->buf);
    CHECK(close(fd) != -1)
}


int cuda_spmm_batched(struct CUDA_spmm_context* ctx, struct Dense* a_T_dense, struct CSR* b_csr, char* c_fname) {
    // Compute (b_csr @ a_T_dense).T

    float alpha = 1.0f;
    float beta = 0.0f;
    void* dBuffer = NULL;
    float *dC = NULL;
    size_t bufferSize = 0;

    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matC;
    ssize_t matC_sz = sizeof(float) * b_csr->nrow * a_T_dense->ncol;

    // printf("B: %ld %ld\n", b_csr->nrow, b_csr->ncol);
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, b_csr->nrow, b_csr->ncol, b_csr->nnz, b_csr->indptr_ptr, b_csr->indices_ptr, b_csr->data_ptr, 
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    
    // printf("C: %ld %ld\n", b_csr->nrow, a_T_dense->ncol);
    CHECK_CUDA(cudaMalloc((void**) &dC, matC_sz))
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, b_csr->nrow, a_T_dense->ncol, a_T_dense->ncol, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW))

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(ctx->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matB, ctx->matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

    CHECK_CUSPARSE(cusparseSpMM(ctx->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matB, ctx->matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, dBuffer))

    CHECK_CUSPARSE(cusparseDestroySpMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))

    // write out results to disk
    int fd = open(c_fname, O_RDWR | O_CREAT | O_TRUNC, 0777);
    CHECK(fd != -1)
    cuda_mmap_write(fd, matC_sz, (char*) dC);
    CHECK(close(fd) != -1)

    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(b_csr->buf))
    CHECK_CUDA(cudaFree(dC))
    return EXIT_SUCCESS;
}


void spmm_batched(struct DenseSerialize* a_serial, struct CSRSerialize** b_serial_arr, char** c_fname_arr, int b_arr_len) {
    int item_sz = sizeof(float);
    int idx_sz = sizeof(int);

    struct Dense a_T_dense = {.nrow=a_serial->nrow, .ncol=a_serial->ncol, .item_sz=item_sz};
    read_dense_onto_gpu(a_serial->fname, &a_T_dense);
    struct CUDA_spmm_context ctx;
    init_spmm_ctx(&ctx, &a_T_dense);

    for(int idx = 0; idx < b_arr_len; idx++) {
        struct CSRSerialize* b_serial = b_serial_arr[idx];
        char* c_fname = c_fname_arr[idx];

        // Read inputs from A and B files
        struct CSR b_csr = {.nnz=b_serial->nnz, .nrow=b_serial->nrow, .ncol=b_serial->ncol, .item_sz=item_sz, .idx_sz=idx_sz};
        read_csr_onto_gpu(b_serial->fname, &b_csr);

        // Setup + execute matmul on CUDA device + return required deserialization info to caller
        CHECK(cuda_spmm_batched(&ctx, &a_T_dense, &b_csr, c_fname) == EXIT_SUCCESS);
    }
    deinit_spmm_ctx(&ctx);
    CHECK_CUDA_NORET(cudaFree(a_T_dense.buf))
}


