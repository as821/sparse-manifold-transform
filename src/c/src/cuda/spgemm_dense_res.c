#include "cuda_util.h"




int cuda_spgemm_sparse_result_alg3(struct CUDA_spgemm_context* ctx, struct CSR* a, struct CSR* b, cusparseSpMatDescr_t* matC, struct CSR* c, cudaStream_t* stream) {
    CHECK(matC != NULL);
    
    // Host problem definition
    const ssize_t A_num_rows = a->nrow;
    const ssize_t A_num_cols = a->ncol;
    const ssize_t A_nnz = a->nnz;
    const ssize_t B_num_rows = b->nrow;
    const ssize_t B_num_cols = b->ncol;
    const ssize_t B_nnz = b->nnz;

    float alpha = 1.0f;
    float beta = 0.0f;

    //
    //  Tutorial starts below this line
    //
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_ALG3;

    // allocate C offsets
    if(stream)
        CHECK_CUDA(cudaMallocAsync((void**) &c->indptr_ptr, (A_num_rows + 1) * sizeof(int), *stream))
    else
        CHECK_CUDA(cudaMalloc((void**) &c->indptr_ptr, (A_num_rows + 1) * sizeof(int)))

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseSpMatDescr_t matB;
    void*  dBuffer1 = NULL, *dBuffer2 = NULL, *dBuffer3 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0, bufferSize3 = 0;
    float chunk_fraction = 0.3;

    // Create sparse matrix A in CSR format
    // printf("%ld %ld %ld", b->nrow, b->ncol, b->nnz);
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz, b->indptr_ptr, b->indices_ptr, b->data_ptr,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateCsr(matC, A_num_rows, B_num_cols, 0, c->indptr_ptr, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))


    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC,
                                    computeType, alg, spgemmDesc, &bufferSize1, NULL))
    if(stream) 
        CHECK_CUDA(cudaMallocAsync((void**) &dBuffer1, bufferSize1, *stream))
    else 
        CHECK_CUDA(cudaMalloc((void**) &dBuffer1, bufferSize1))
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC,
                                    computeType, alg, spgemmDesc, &bufferSize1, dBuffer1))

    /*
        NOTE: same above this line
    */

    
    // inspect the matrices A and B to understand the memory requirement for the next step
    // int64_t num_prods;
    // CHECK_CUSPARSE(cusparseSpGEMM_getNumProducts(spgemmDesc, &num_prods))



    // ask bufferSize3 bytes for external memory
    CHECK_CUSPARSE(cusparseSpGEMM_estimateMemory(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC, computeType, 
                                        alg, spgemmDesc, chunk_fraction, &bufferSize3, NULL, NULL))
    if(stream)
        CHECK_CUDA(cudaMallocAsync((void**) &dBuffer3, bufferSize3, *stream))
    else
        CHECK_CUDA(cudaMalloc((void**) &dBuffer3, bufferSize3))


    // inspect the matrices A and B to understand the memory requirement for the next step
    CHECK_CUSPARSE(cusparseSpGEMM_estimateMemory(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC, computeType,
                                                alg, spgemmDesc, chunk_fraction, &bufferSize3, dBuffer3, &bufferSize2))
    if(stream) {
        CHECK_CUDA(cudaFreeAsync(dBuffer3, *stream));
        CHECK_CUDA(cudaMallocAsync((void**) &dBuffer2, bufferSize2, *stream))
    }
    else {
        CHECK_CUDA(cudaFree(dBuffer3))
        CHECK_CUDA(cudaMalloc((void**) &dBuffer2, bufferSize2))
    }    
    
    /*
        NOTE: same below this line...
    */    
    
    
    
    // compute the intermediate product of A * B
    CHECK_CUSPARSE(cusparseSpGEMM_compute(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC,
                                        computeType, alg, spgemmDesc, &bufferSize2, dBuffer2))
    if(stream)
        CHECK_CUDA(cudaFreeAsync(dBuffer1, *stream))
    else
        CHECK_CUDA(cudaFree(dBuffer1))

    // get matrix C non-zero entries C_nnz1
    CHECK_CUSPARSE(cusparseSpMatGetSize(*matC, &c->nrow, &c->ncol, &c->nnz))

    // allocate matrix C
    if(stream) {
        CHECK_CUDA(cudaMallocAsync((void**) &c->indices_ptr, c->nnz * sizeof(int), *stream))
        CHECK_CUDA(cudaMallocAsync((void**) &c->data_ptr, c->nnz * sizeof(float), *stream))
    }
    else {
        CHECK_CUDA(cudaMalloc((void**) &c->indices_ptr, c->nnz * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void**) &c->data_ptr, c->nnz * sizeof(float)))
    }

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(*matC, c->indptr_ptr, c->indices_ptr, c->data_ptr))

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(cusparseSpGEMM_copy(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC,
                                    computeType, alg, spgemmDesc))

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
    CHECK_CUSPARSE(cusparseDestroySpMat(matB))
    if(b->buf && !stream) {    // handles if GPU memory backing b was allocated as one chunk or as multiple allocations
        
        // TODO(as) actually we cant do this here if we want graph capture to work....
            
        CHECK_CUDA(cudaFree(b->buf))
        b->buf = NULL;
    }
    else if(!b->buf) {
        if(stream) {
            CHECK_CUDA(cudaFreeAsync(b->data_ptr, *stream))
            CHECK_CUDA(cudaFreeAsync(b->indices_ptr, *stream))
            CHECK_CUDA(cudaFreeAsync(b->indptr_ptr, *stream))
        }
        else {
            CHECK_CUDA(cudaFree(b->data_ptr))
            CHECK_CUDA(cudaFree(b->indices_ptr))
            CHECK_CUDA(cudaFree(b->indptr_ptr))
        }
        b->data_ptr = NULL;
        b->indices_ptr = NULL;
        b->indptr_ptr = NULL;
    }
    if(stream)
        CHECK_CUDA(cudaFreeAsync(dBuffer2, *stream))
    else
        CHECK_CUDA(cudaFree(dBuffer2))
    return EXIT_SUCCESS;
}



int cuda_spgemm_sparse_result(struct CUDA_spgemm_context* ctx, struct CSR* a, struct CSR* b, cusparseSpMatDescr_t* matC, struct CSR* c, cudaStream_t* stream) {
    CHECK(matC != NULL);
    
    // Host problem definition
    const ssize_t A_num_rows = a->nrow;
    const ssize_t A_num_cols = a->ncol;
    const ssize_t A_nnz = a->nnz;
    const ssize_t B_num_rows = b->nrow;
    const ssize_t B_num_cols = b->ncol;
    const ssize_t B_nnz = b->nnz;

    float alpha = 1.0f;
    float beta = 0.0f;

    //
    //  Tutorial starts below this line
    //
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;

    // allocate C offsets
    if(stream)
        CHECK_CUDA(cudaMallocAsync((void**) &c->indptr_ptr, (A_num_rows + 1) * sizeof(int), *stream))
    else
        CHECK_CUDA(cudaMalloc((void**) &c->indptr_ptr, (A_num_rows + 1) * sizeof(int)))

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseSpMatDescr_t matB;
    void*  dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    // Create sparse matrix A in CSR format
    // printf("%ld %ld %ld", b->nrow, b->ncol, b->nnz);
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz, b->indptr_ptr, b->indices_ptr, b->data_ptr,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateCsr(matC, A_num_rows, B_num_cols, 0, c->indptr_ptr, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))


    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL))
    if(stream) 
        CHECK_CUDA(cudaMallocAsync((void**) &dBuffer1, bufferSize1, *stream))
    else 
        CHECK_CUDA(cudaMalloc((void**) &dBuffer1, bufferSize1))
    
    // inspect the matrices A and B to understand the memory requirement for the next step
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1))

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(cusparseSpGEMM_compute(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL))
    if(stream)
        CHECK_CUDA(cudaMallocAsync((void**) &dBuffer2, bufferSize2, *stream))
    else
        CHECK_CUDA(cudaMalloc((void**) &dBuffer2, bufferSize2))

    // compute the intermediate product of A * B
    CHECK_CUSPARSE(cusparseSpGEMM_compute(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC,
                                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2))
    if(stream)
        CHECK_CUDA(cudaFreeAsync(dBuffer1, *stream))
    else
        CHECK_CUDA(cudaFree(dBuffer1))

    // get matrix C non-zero entries C_nnz1
    CHECK_CUSPARSE(cusparseSpMatGetSize(*matC, &c->nrow, &c->ncol, &c->nnz))

    // allocate matrix C
    if(stream) {
        CHECK_CUDA(cudaMallocAsync((void**) &c->indices_ptr, c->nnz * sizeof(int), *stream))
        CHECK_CUDA(cudaMallocAsync((void**) &c->data_ptr, c->nnz * sizeof(float), *stream))
    }
    else {
        CHECK_CUDA(cudaMalloc((void**) &c->indices_ptr, c->nnz * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void**) &c->data_ptr, c->nnz * sizeof(float)))
    }

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(*matC, c->indptr_ptr, c->indices_ptr, c->data_ptr))

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(cusparseSpGEMM_copy(ctx->handle, opA, opB, &alpha, ctx->matA, matB, &beta, *matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc))

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
    CHECK_CUSPARSE(cusparseDestroySpMat(matB))
    if(b->buf && !stream) {    // handles if GPU memory backing b was allocated as one chunk or as multiple allocations
        
        // TODO(as) actually we cant do this here if we want graph capture to work....
            
        CHECK_CUDA(cudaFree(b->buf))
        b->buf = NULL;
    }
    else if(!b->buf) {
        if(stream) {
            CHECK_CUDA(cudaFreeAsync(b->data_ptr, *stream))
            CHECK_CUDA(cudaFreeAsync(b->indices_ptr, *stream))
            CHECK_CUDA(cudaFreeAsync(b->indptr_ptr, *stream))
        }
        else {
            CHECK_CUDA(cudaFree(b->data_ptr))
            CHECK_CUDA(cudaFree(b->indices_ptr))
            CHECK_CUDA(cudaFree(b->indptr_ptr))
        }
        b->data_ptr = NULL;
        b->indices_ptr = NULL;
        b->indptr_ptr = NULL;
    }
    if(stream)
        CHECK_CUDA(cudaFreeAsync(dBuffer2, *stream))
    else
        CHECK_CUDA(cudaFree(dBuffer2))
    return EXIT_SUCCESS;
}

int cuda_spgemm_dense_result(struct CUDA_spgemm_context* ctx, struct CSR* a, struct CSR* b, struct CSR* c, ssize_t* dense_bytes, float** d_dense, void** dBuffer, bool alg3) {
    CHECK(a->nnz > 0 && a->nrow > 0 && a->ncol > 0);
    CHECK(b->nnz > 0 && b->nrow > 0 && b->ncol > 0);

    // set up result structs
    cusparseSpMatDescr_t matC;
    c->buf = NULL;
    c->item_sz = a->item_sz;
    c->idx_sz = a->idx_sz;
    c->nnz = 0;
    c->nrow = a->nrow;
    c->ncol = b->ncol;
    c->data_ptr=NULL;
    c->indptr_ptr=NULL;
    c->indices_ptr=NULL;
    
    // Perform spgemm matmul on operands
    if(alg3) {
        CHECK_CUDA(cuda_spgemm_sparse_result_alg3(ctx, a, b, &matC, c, NULL));
    }
    else {
        CHECK_CUDA(cuda_spgemm_sparse_result(ctx, a, b, &matC, c, NULL));
    }

    //-------------------------------------------------------------------------
    // Perform sparse to dense conversion
    cusparseDnMatDescr_t matD;
    size_t bufferSize = 0;
    *dense_bytes = a->nrow * b->ncol * sizeof(float);
    ssize_t ld = b->ncol;       // TODO(as) "leading dimension", a bit sketch but this is how the example does it...
    CHECK_CUDA(cudaMalloc((void**) d_dense, *dense_bytes))
    CHECK_CUSPARSE(cusparseCreateDnMat(&matD, a->nrow, b->ncol, ld, *d_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW))
    
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(ctx->handle, matC, matD, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA(cudaMalloc(dBuffer, bufferSize))

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE(cusparseSparseToDense(ctx->handle, matC, matD, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, *dBuffer))
    
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnMat(matD))
    CHECK_CUSPARSE(cusparseDestroySpMat(matC))

    return EXIT_SUCCESS;
}




int cuda_spgemm_dense_result_file_write(struct CUDA_spgemm_context* ctx, struct CSR* a, struct CSR* b, char* c_fname) {
    struct CSR c;
    void* dBuffer = NULL;
    float* d_dense = NULL;
    ssize_t dense_bytes;
    CHECK_CUDA(cuda_spgemm_dense_result(ctx, a, b, &c, &dense_bytes, &d_dense, &dBuffer, false))

    // copy directly from device into mmap-ed file
    int d_fd = open(c_fname, O_RDWR | O_CREAT | O_TRUNC, 0777);
    CHECK(d_fd != -1)
    cuda_mmap_write(d_fd, dense_bytes, (char*)d_dense);
    CHECK(close(d_fd) != -1)


    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(c.indptr_ptr))
    CHECK_CUDA(cudaFree(c.indices_ptr))
    CHECK_CUDA(cudaFree(c.data_ptr))
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(d_dense))

    return EXIT_SUCCESS;
}

void spgemm_batched_matmul(struct CSRSerialize* a_serial, struct CSRSerialize** b_serial_arr, char** c_fname_arr, int b_arr_len) {
    // Leave it to caller to handle result slicing + processing of separate files for each given B matrix. Avoids 
    // complicating the GPU/disk transfers + needing to implement array slicing here

    int item_sz = sizeof(float);
    int idx_sz = sizeof(int);
    struct CSR a_csr = {.nnz=a_serial->nnz, .nrow=a_serial->nrow, .ncol=a_serial->ncol, .item_sz=item_sz, .idx_sz=idx_sz};
    read_csr_onto_gpu(a_serial->fname, &a_csr);

    struct CUDA_spgemm_context ctx;
    init_spgemm_ctx(&ctx, &a_csr);

    for(int idx = 0; idx < b_arr_len; idx++) {
        struct CSRSerialize* b_serial = b_serial_arr[idx];
        char* c_fname = c_fname_arr[idx];

        // Read inputs from A and B files
        struct CSR b_csr = {.nnz=b_serial->nnz, .nrow=b_serial->nrow, .ncol=b_serial->ncol, .item_sz=item_sz, .idx_sz=idx_sz};
        read_csr_onto_gpu(b_serial->fname, &b_csr);

        // Setup + execute matmul on CUDA device + return required deserialization info to caller
        CHECK(cuda_spgemm_dense_result_file_write(&ctx, &a_csr, &b_csr, c_fname) == EXIT_SUCCESS);
    }
    deinit_spgemm_ctx(&ctx);
    CHECK_CUDA_NORET(cudaFree(a_csr.buf))
}




