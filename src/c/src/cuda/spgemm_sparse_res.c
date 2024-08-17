#include "cuda_util.h"
#include "file_wrapper.h"
#include "sync.h"
#include "pinned_circ_buffer.h"
#include <cublas_v2.h>

#include <sys/time.h>
#include <pthread.h>


extern int cuda_spgemm_dense_result(struct CUDA_spgemm_context* ctx, struct CSR* a, struct CSR* b, struct CSR* c, ssize_t* dense_bytes, float** d_dense, void** dBuffer, bool alg3);

struct DataLoaderArgs {
    struct SyncArgs sync;
    struct PinnedMemBuffer* pinned_buf;
    
    struct CSRSerialize** a_serial_arr;
    struct CSRSerialize** b_serial_arr;
    struct CSR* a_csr_arr;
    struct CSR* b_csr_arr;
    struct CSR* c_csr_arr;
    int b_arr_len;
    int64_t max_preload;
    char** fname_arr;
};


void* disk_write_func(void* a) {
    struct DataLoaderArgs* args = (struct DataLoaderArgs*) a;
    for(int64_t idx = 0; idx < args->b_arr_len; idx++) {
        consumer_wait(&args->sync);
        struct CSR* c_csr = &args->c_csr_arr[idx];
        int fd = open(args->fname_arr[idx], O_RDWR | O_CREAT | O_TRUNC, 0777);
        CHECK(fd != -1)
        write_file(fd, c_csr->nnz, (char*)c_csr->buf);
        CHECK(close(fd) != -1)
        free(c_csr->buf);
        consumer_signal(&args->sync);
    }    
    return NULL;
}




void* disk_read_func(void* a) {
    int item_sz = sizeof(float);
    int idx_sz = sizeof(int);
    struct DataLoaderArgs* args = (struct DataLoaderArgs*) a;

    int tid = -1;
    for(int idx = loader_get_next_index(&args->sync); idx < args->b_arr_len; idx = loader_get_next_index(&args->sync)) {
        if(tid < 0)
            tid = idx;
        
        // set up "a" operand for matmul
        struct CSRSerialize* a_serial = args->a_serial_arr[idx];
        CHECK(a_serial->nnz > 0 && a_serial->nrow > 0 && a_serial->ncol > 0);
        struct CSR* a_csr = &args->a_csr_arr[idx];
        a_csr->nnz = a_serial->nnz;
        a_csr->nrow = a_serial->nrow;
        a_csr->ncol = a_serial->ncol;
        a_csr->item_sz = item_sz;
        a_csr->idx_sz = idx_sz;

        // set up "b" operand for matmul
        struct CSRSerialize* b_serial = args->b_serial_arr[idx];
        CHECK(b_serial->nnz > 0 && b_serial->nrow > 0 && b_serial->ncol > 0);
        struct CSR* b_csr = &args->b_csr_arr[idx];
        b_csr->nnz = b_serial->nnz;
        b_csr->nrow = b_serial->nrow;
        b_csr->ncol = b_serial->ncol;
        b_csr->item_sz = item_sz;
        b_csr->idx_sz = idx_sz;

        producer_wait(&args->sync, args->max_preload);

        // single allocation of pinned host memory for "a" and "b" operands. pointer to start of the 
        // allocation is stored on the "b" operand
        ssize_t b_data_bytes = item_sz * b_csr->nnz;
        ssize_t b_indices_bytes = idx_sz * b_csr->nnz;
        ssize_t b_indptr_bytes = idx_sz * (b_csr->nrow + 1);
        ssize_t b_bytes = b_data_bytes + b_indices_bytes + b_indptr_bytes;

        ssize_t a_data_bytes = item_sz * a_csr->nnz;
        ssize_t a_indices_bytes = idx_sz * a_csr->nnz;
        ssize_t a_indptr_bytes = idx_sz * (a_csr->nrow + 1);
        ssize_t a_bytes = a_data_bytes + a_indices_bytes + a_indptr_bytes;

        // pointer to head of allocation is stored on the "b" operand struct
        // CHECK_CUDA_NORET(cudaMallocHost((void**)&b_csr->buf, b_bytes + a_bytes))
        struct PinnedMem* pinned_ptr = get_next_pinned_mem(args->pinned_buf);
        CHECK(pinned_ptr->n_bytes >= a_data_bytes + b_data_bytes);
        b_csr->buf = pinned_ptr->ptr;
        CHECK(b_csr->buf);
        a_csr->buf = &b_csr->buf[b_bytes];
        CHECK(a_csr->buf);

        // read "b" file into in-memory buffer
        int fd = open(b_serial->fname, O_RDONLY);
        CHECK(fd != -1)
        read_file(fd, b_bytes, b_csr->buf);
        CHECK(close(fd) != -1)
        b_csr->data_ptr = NULL;
        b_csr->indices_ptr = NULL;
        b_csr->indptr_ptr = NULL;

        // read "a" file into in-memory buffer
        fd = open(a_serial->fname, O_RDONLY);
        CHECK(fd != -1)
        read_file(fd, a_bytes, a_csr->buf);
        CHECK(close(fd) != -1)
        a_csr->data_ptr = NULL;
        a_csr->indices_ptr = NULL;
        a_csr->indptr_ptr = NULL;

        producer_signal(&args->sync, idx);
    }
    return NULL;
}

void profile_log(int idx, struct timeval* start, struct timeval* stop, struct DataLoaderArgs* args, char* type);


void move_onto_gpu(struct CSR* b_csr, struct CSR* a_csr, int item_sz, int idx_sz, struct PinnedMemBuffer* pinned_buf) {
    CHECK(b_csr->item_sz == a_csr->item_sz && b_csr->item_sz == item_sz);
    CHECK(b_csr->idx_sz == a_csr->idx_sz && b_csr->idx_sz == idx_sz);
    
    // both operands allocated in the same chunk of pinned host memory
    ssize_t b_data_bytes = item_sz * b_csr->nnz;
    ssize_t b_indices_bytes = idx_sz * b_csr->nnz;
    ssize_t b_indptr_bytes = idx_sz * (b_csr->nrow + 1);
    ssize_t b_bytes = b_data_bytes + b_indices_bytes + b_indptr_bytes;

    ssize_t a_data_bytes = item_sz * a_csr->nnz;
    ssize_t a_indices_bytes = idx_sz * a_csr->nnz;
    ssize_t a_indptr_bytes = idx_sz * (a_csr->nrow + 1);
    ssize_t a_bytes = a_data_bytes + a_indices_bytes + a_indptr_bytes;


    struct timeval start, stop;
    gettimeofday(&start, NULL);
    char* dev_cpy = NULL;
    CHECK_CUDA_NORET(cudaMalloc((void**)&dev_cpy, b_bytes+a_bytes))
    gettimeofday(&stop, NULL);
    // profile_log(-1, &start, &stop, NULL, "\tmalloc");
    
    // pinned host memory allocation is stored on the "b" operand
    gettimeofday(&start, NULL);
    CHECK_CUDA_NORET(cudaMemcpy((void*)dev_cpy, b_csr->buf, b_bytes+a_bytes, cudaMemcpyHostToDevice))
    gettimeofday(&stop, NULL);
    // profile_log(-1, &start, &stop, NULL, "\tmemcpy");
    

    gettimeofday(&start, NULL);
    // CHECK_CUDA_NORET(cudaFreeHost(b_csr->buf))
    done_with_pinned_mem_ptr(b_csr->buf, pinned_buf);
    gettimeofday(&stop, NULL);
    // profile_log(-1, &start, &stop, NULL, "\tfree");



    b_csr->buf = dev_cpy;
    b_csr->data_ptr = (float*) b_csr->buf;
    b_csr->indices_ptr = (int*)(&b_csr->buf[b_data_bytes]);
    b_csr->indptr_ptr = (int*)(&b_csr->buf[b_data_bytes + b_indices_bytes]);

    a_csr->buf = &b_csr->buf[b_bytes];
    a_csr->data_ptr = (float*) a_csr->buf;
    a_csr->indices_ptr = (int*)(&a_csr->buf[a_data_bytes]);
    a_csr->indptr_ptr = (int*)(&a_csr->buf[a_data_bytes + a_indices_bytes]);
}




void pinned_mem_buf_alloc(struct PinnedMemBuffer* pinned_buf, struct CSRSerialize** a_serial_arr, struct CSRSerialize** b_serial_arr, int b_arr_len, int item_sz, int idx_sz) {
    // Find largest operand pair, then use that as the alloc size
    ssize_t alloc_sz = 0;
    for(int idx = 0; idx < b_arr_len; idx++) {
        struct CSRSerialize* a_csr = a_serial_arr[idx];
        CHECK(a_csr->nnz > 0 && a_csr->nrow > 0 && a_csr->ncol > 0);
        struct CSRSerialize* b_csr = b_serial_arr[idx];
        CHECK(b_csr->nnz > 0 && b_csr->nrow > 0 && b_csr->ncol > 0);

        ssize_t b_data_bytes = item_sz * b_csr->nnz;
        ssize_t b_indices_bytes = idx_sz * b_csr->nnz;
        ssize_t b_indptr_bytes = idx_sz * (b_csr->nrow + 1);
        ssize_t b_bytes = b_data_bytes + b_indices_bytes + b_indptr_bytes;

        ssize_t a_data_bytes = item_sz * a_csr->nnz;
        ssize_t a_indices_bytes = idx_sz * a_csr->nnz;
        ssize_t a_indptr_bytes = idx_sz * (a_csr->nrow + 1);
        ssize_t a_bytes = a_data_bytes + a_indices_bytes + a_indptr_bytes;

        ssize_t tot = a_bytes + b_bytes;
        if(tot > alloc_sz)
            alloc_sz = tot;
    }
    init_pinned_mem_buf(pinned_buf, alloc_sz);
}




void spgemm_batched_matmul_dop(struct CSRSerialize** a_serial_arr, struct CSRSerialize** b_serial_arr, int b_arr_len, char** c_fname_arr) {
    // Leave it to caller to handle result slicing + processing of separate files for each given B matrix. Avoids 
    // complicating the GPU/disk transfers + needing to implement array slicing here

    int read_max_preload = 4;
    int item_sz = sizeof(float);
    int idx_sz = sizeof(int);

    struct PinnedMemBuffer pinned_buf;
    pinned_buf.n_buf = 2 * read_max_preload;            // TODO(as) bug somewhere in sync code, why is this needed?
    pinned_mem_buf_alloc(&pinned_buf, a_serial_arr, b_serial_arr, b_arr_len, item_sz, idx_sz);

    // set up data loader threads
    struct CSR a_csr_arr[b_arr_len];
    struct CSR b_csr_arr[b_arr_len];
    struct CSR c_csr_arr[b_arr_len];
    pthread_t write_thread, read_thread1, read_thread2;
    struct DataLoaderArgs read_args = {.a_serial_arr=a_serial_arr, .a_csr_arr=a_csr_arr, .b_serial_arr=b_serial_arr, .b_csr_arr=b_csr_arr, .b_arr_len=b_arr_len, .c_csr_arr=NULL, .fname_arr=NULL, .max_preload=read_max_preload, .pinned_buf=&pinned_buf};
    sync_args_init(&read_args.sync);
    CHECK(pthread_create(&read_thread1, NULL, disk_read_func, &read_args) == 0);
    CHECK(pthread_create(&read_thread2, NULL, disk_read_func, &read_args) == 0);
    
    struct DataLoaderArgs write_args = {.a_serial_arr=NULL, .a_csr_arr=NULL, .b_serial_arr=NULL, .b_csr_arr=NULL, .pinned_buf=NULL, .c_csr_arr=c_csr_arr, .b_arr_len=b_arr_len, .max_preload=-1, .fname_arr=c_fname_arr};
    sync_args_init(&write_args.sync);
    CHECK(pthread_create(&write_thread, NULL, disk_write_func, &write_args) == 0);

    struct CUDA_spgemm_context ctx;
    CHECK_CUSPARSE_NORET(cusparseCreate(&ctx.handle))

    for(int idx = 0; idx < b_arr_len; idx++) {
        struct CSRSerialize* a_serial = a_serial_arr[idx];
        struct CSRSerialize* b_serial = b_serial_arr[idx];
        char* c_fname = c_fname_arr[idx];

        // load both operands onto GPU, free memory allocated by loader thread, then signal loader threads
        struct timeval read_start, read_stop;
        gettimeofday(&read_start, NULL);
        consumer_wait(&read_args.sync);
        gettimeofday(&read_stop, NULL);
        // profile_log(idx, &read_start, &read_stop, &read_args, "read");


        struct timeval trans_start, trans_stop;
        gettimeofday(&trans_start, NULL);
        struct CSR a_csr = a_csr_arr[idx];
        struct CSR b_csr = b_csr_arr[idx];
        move_onto_gpu(&b_csr, &a_csr, item_sz, idx_sz, &pinned_buf);

        consumer_signal(&read_args.sync);
        gettimeofday(&trans_stop, NULL);
        // profile_log(idx, &trans_start, &trans_stop, NULL, "transfer");
        
        gettimeofday(&trans_start, NULL);
        CHECK_CUSPARSE_NORET(cusparseCreateCsr(&ctx.matA, a_csr.nrow, a_csr.ncol, a_csr.nnz, a_csr.indptr_ptr, a_csr.indices_ptr, a_csr.data_ptr,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
        gettimeofday(&trans_stop, NULL);
        // profile_log(idx, &trans_start, &trans_stop, NULL, "create csr");

        // Setup + execute matmul on CUDA device + return required deserialization info to caller
        struct timeval gpu_start, gpu_stop;
        gettimeofday(&gpu_start, NULL);
        struct CSR* c_csr = &c_csr_arr[idx];
        ssize_t dense_bytes;
        void* dBuffer = NULL;
        float* d_dense = NULL;
        CHECK(cuda_spgemm_dense_result(&ctx, &a_csr, &b_csr, c_csr, &dense_bytes, &d_dense, &dBuffer, true) == EXIT_SUCCESS);
        CHECK(c_csr->buf == NULL)
        CHECK(b_csr.buf == NULL)
        a_csr.buf = NULL;       // b_csr.buf allocation also contains a_csr.buf. b_csr.buf is deallocated by cuda_spgemm_dense_result, so a_csr.buf is a dangling pointer

        CHECK_CUSPARSE_NORET(cusparseDestroySpMat(ctx.matA));
        CHECK_CUDA_NORET(cudaFree(c_csr->indptr_ptr))
        CHECK_CUDA_NORET(cudaFree(c_csr->indices_ptr))
        CHECK_CUDA_NORET(cudaFree(c_csr->data_ptr))
        CHECK_CUDA_NORET(cudaFree(dBuffer))

        // transpose resulting dense matrix
        // (hacky: sgeam supports transposing of arguments, use it to perform transpose but zero out second arg with beta 
        // coefficient so just transposed first arg is returned)
        float* clone = NULL;        // make copy + operate on that copy due to sgeam in-place requirements
        CHECK_CUDA_NORET(cudaMalloc((void**)&clone, dense_bytes))
        CHECK_CUDA_NORET(cudaMemcpy((void*)clone, d_dense, dense_bytes, cudaMemcpyDeviceToDevice))
        float alpha = 1.0;
        float beta = 0.0;
        cublasHandle_t handle;
        cublasCreate(&handle);
        int m = a_csr.nrow;
        int n = b_csr.ncol;
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, clone, n, &beta, clone, m, d_dense, m);
        cublasDestroy(handle);
        CHECK_CUDA_NORET(cudaFree(clone))
        gettimeofday(&gpu_stop, NULL);
        // profile_log(idx, &gpu_start, &gpu_stop, NULL, "gpu");

        // copy result off of device + hand off to write thread to manage disk operations
        struct timeval alloc_start, alloc_stop;
        gettimeofday(&alloc_start, NULL);
        c_csr->buf = malloc(dense_bytes);
        CHECK(c_csr->buf);
        c_csr->nnz = dense_bytes;
        CHECK_CUDA_NORET(cudaMemcpy((void*)c_csr->buf, d_dense, dense_bytes, cudaMemcpyDeviceToHost))
        CHECK_CUDA_NORET(cudaFree(d_dense))
        producer_signal(&write_args.sync, idx);
        gettimeofday(&alloc_stop, NULL);
        // profile_log(idx, &alloc_start, &alloc_stop, &write_args, "alloc");
    }
    struct timeval join_start, join_stop;
    gettimeofday(&join_start, NULL);
    CHECK_CUSPARSE_NORET(cusparseDestroy(ctx.handle))
    CHECK(pthread_join(read_thread1, NULL) == 0);
    CHECK(pthread_join(read_thread2, NULL) == 0);
    CHECK(pthread_join(write_thread, NULL) == 0);
    gettimeofday(&join_stop, NULL);

    cleanup_pinned_mem_buf(&pinned_buf);

    // profile_log(-1, &join_start, &join_stop, NULL, "worker join");    
}



void profile_log(int idx, struct timeval* start, struct timeval* stop, struct DataLoaderArgs* args, char* type) {
    double resolution = 1000;
    double start_ms = (((double)start->tv_sec)*resolution)+(((double)start->tv_usec)/resolution);
    double stop_ms = (((double)stop->tv_sec)*resolution)+(((double)stop->tv_usec)/resolution);
    double diff_in_sec = (stop_ms - start_ms)/resolution;
    if(args) {
        CHECK(pthread_mutex_lock(&args->sync.mutex) == 0);
        printf("(%d) Main waiting for %s (%d, %d): %f sec\n", idx, type, args->sync.consumer_ptr, args->sync.loader_ptr, diff_in_sec);
        CHECK(pthread_mutex_unlock(&args->sync.mutex) == 0);
    }
    else
        printf("%s: %f sec\n", type, diff_in_sec);
}
