#include "cuda_util.h"


void mmap_cuda_transfer(int fd, ssize_t sz, char** dst_buf) {
    // NOTE: this assumes that system has enough memory to load entire file into memory. Can 
    // perform copy with multiple mmap/memcpy calls if want to reduce memory burden
    void* mmap_arr = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK(mmap_arr != MAP_FAILED)
    CHECK_CUDA_NORET(cudaMalloc((void**)dst_buf, sz))
    CHECK_CUDA_NORET(cudaMemcpy((void*) *dst_buf, mmap_arr, sz, cudaMemcpyHostToDevice))
    CHECK(munmap(mmap_arr, sz) != -1)
}


void cuda_mmap_write(int fd, ssize_t sz, char* cuda_buf) {
    CHECK(lseek(fd, sz, SEEK_SET) != -1)       
    char zero = 0;
    CHECK(write(fd, &zero, 1) == 1)     // size file correctly before making a mmap (cannot change size of mmap)

    void* mmap_arr = mmap(NULL, sz, PROT_WRITE, MAP_SHARED, fd, 0);
    CHECK(mmap_arr != MAP_FAILED)
    CHECK_CUDA_NORET(cudaMemcpy(mmap_arr, cuda_buf, sz, cudaMemcpyDeviceToHost))
    CHECK(munmap(mmap_arr, sz) != -1)
}

void read_csr_onto_gpu(char* fname, struct CSR* csr) {
    // Allocate buffer for CSR data
    ssize_t data_bytes = csr->item_sz * csr->nnz;
    ssize_t indices_bytes = csr->idx_sz * csr->nnz;
    ssize_t indptr_bytes = csr->idx_sz * (csr->nrow + 1);

    // Read serialized data from file
    // printf("file: %s\n", fname);
    int fd = open(fname, O_RDONLY);
    CHECK(fd != -1)
    mmap_cuda_transfer(fd, data_bytes + indices_bytes + indptr_bytes, &csr->buf);
    CHECK(close(fd) != -1)

    // Deserialize into struct
    csr->data_ptr = (float*)csr->buf;
    csr->indices_ptr = (int*)(&csr->buf[data_bytes]);
    csr->indptr_ptr = (int*)(&csr->buf[data_bytes + indices_bytes]);
}

void init_spgemm_ctx(struct CUDA_spgemm_context* ctx, struct CSR* a) {
    CHECK_CUSPARSE_NORET(cusparseCreate(&ctx->handle))
    CHECK_CUSPARSE_NORET(cusparseCreateCsr(&ctx->matA, a->nrow, a->ncol, a->nnz, a->indptr_ptr, a->indices_ptr, a->data_ptr,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
}

void deinit_spgemm_ctx(struct CUDA_spgemm_context* ctx) {
    CHECK_CUSPARSE_NORET(cusparseDestroySpMat(ctx->matA))
    CHECK_CUSPARSE_NORET(cusparseDestroy(ctx->handle))
}


void validate_device_ptr(void* ptr) {
    struct cudaPointerAttributes attr;
    CHECK_CUDA_NORET(cudaPointerGetAttributes(&attr, ptr))
    CHECK(attr.type == cudaMemoryTypeDevice)
}





