
#include "util.h"

#include <sys/mman.h>
#include <sys/time.h>


struct COO {
    ssize_t nnz;
    ssize_t nrow;
    ssize_t ncol;

    ssize_t item_sz;
    ssize_t idx_sz;

    char* buf;
    float* data_ptr;
    int64_t* row_ptr;
    int64_t* col_ptr;
};

struct CSR {
    ssize_t nnz;
    ssize_t nrow;
    ssize_t ncol;

    ssize_t item_sz;
    ssize_t idx_sz;

    char* buf;
    float* data_ptr;
    int64_t* indices_ptr;
    int64_t* indptr_ptr;
};



void scipy_coo_tocsr(struct COO* coo, struct CSR* csr) {
    // TODO(as) 
    //      - improve mmap handling (perhaps process in chunks + handle files directly?)
    //      - multithreading


    struct timeval start, t1, t2, t3, t4;
    gettimeofday(&start, NULL);

    // compute number of non-zero entries per row of A 
    // (loop partially unrolled to help with FP unit latency)
    memset(csr->indptr_ptr, 0, csr->nrow * csr->idx_sz);
    int64_t term = (int64_t)(csr->nnz / 4);
    term *= 4;
    for (int64_t n = 0; n < term; n=n+4){
        csr->indptr_ptr[coo->row_ptr[n]]++;
        csr->indptr_ptr[coo->row_ptr[n+1]]++;
        csr->indptr_ptr[coo->row_ptr[n+2]]++;
        csr->indptr_ptr[coo->row_ptr[n+3]]++;
    }
    for (int64_t n = term; n < csr->nnz; n++) {
        csr->indptr_ptr[coo->row_ptr[n]]++;
    }
    gettimeofday(&t1, NULL);

    // cumsum the nnz per row to get nnz --> at end, each indptr entry stores the index of the start of its indices/data
    int64_t cumsum = 0;
    for(int64_t i = 0; i < csr->nrow; i++) {
        int64_t temp = csr->indptr_ptr[i];
        csr->indptr_ptr[i] = cumsum;
        cumsum += temp;
    }
    csr->indptr_ptr[csr->nrow] = csr->nnz;
    gettimeofday(&t2, NULL);

    // copy data/col. indices into CSR data structures --> start at the beginning of each rows space then incrementally write
    // elements while incrementing indptr
    // NOTE: could parallelize this so each thread processes a single row (but this requires some pre-computation...)
    for(int64_t n = 0; n < csr->nnz; n++){
        int64_t row  = coo->row_ptr[n];
        int64_t dest = csr->indptr_ptr[row];
        csr->indices_ptr[dest] = coo->col_ptr[n];
        csr->data_ptr[dest] = coo->data_ptr[n];
        csr->indptr_ptr[row]++;
    }
    gettimeofday(&t3, NULL);

    // shift all indptr values over by one index (to account for the row filling loop above this one)
    int64_t last = 0;
    for(int64_t i = 0; i <= csr->nrow; i++){
        int64_t temp = csr->indptr_ptr[i];
        csr->indptr_ptr[i]  = last;
        last = temp;
    }
    gettimeofday(&t4, NULL);


    double start_ms = (((double)start.tv_sec)*1000)+(((double)start.tv_usec)/1000);
    double t1_ms = (((double)t1.tv_sec)*1000)+(((double)t1.tv_usec)/1000);
    double t2_ms = (((double)t2.tv_sec)*1000)+(((double)t2.tv_usec)/1000);
    double t3_ms = (((double)t3.tv_sec)*1000)+(((double)t3.tv_usec)/1000);
    double t4_ms = (((double)t4.tv_sec)*1000)+(((double)t4.tv_usec)/1000);

    double diff_1 = (t1_ms - start_ms) / 1000;
    double diff_2 = (t2_ms - t1_ms) / 1000;
    double diff_3 = (t3_ms - t2_ms) / 1000;
    double diff_4 = (t4_ms - t3_ms) / 1000;
    printf("\tTimings: %f %f %f %f\n", diff_1, diff_2, diff_3, diff_4);
}

void coo2csr_param(int64_t nrow, int64_t ncol, int64_t nnz,
                    int64_t* coo_row, int64_t* coo_col, float* coo_data,
                    int64_t* csr_indices, int64_t* csr_indptr, float* csr_data) {
    struct COO coo = {.nnz=nnz, .nrow=nrow, .ncol=ncol, .buf=NULL, .data_ptr=coo_data, .row_ptr=coo_row, .col_ptr=coo_col};
    struct CSR csr = {.nnz=nnz, .nrow=nrow, .ncol=ncol, .buf=NULL, .data_ptr=csr_data, .indices_ptr=csr_indices, .indptr_ptr=csr_indptr};

    struct timeval start, stop;
    gettimeofday(&start, NULL);
    scipy_coo_tocsr(&coo, &csr);
    gettimeofday(&stop, NULL);
    
    // get duration to millisecond precision
    double start_ms = (((double)start.tv_sec)*1000)+(((double)start.tv_usec)/1000);
    double stop_ms = (((double)stop.tv_sec)*1000)+(((double)stop.tv_usec)/1000);
    double diff_in_sec = (stop_ms - start_ms)/1000;

    printf("\tConverted COO (%ld, %ld) with %ld nnz to CSR in: %f sec\n", coo.nrow, coo.ncol, coo.nnz, diff_in_sec);
}


