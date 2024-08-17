
#include "file_wrapper.h"
#include <math.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <pthread.h>

bool DEBUG = false;

struct CSRSliceSerialize64 {
    char* data_fname;
    char* indices_fname;
    char* indptr_fname;
    ssize_t nnz;
    ssize_t nrow;
    ssize_t ncol;
    ssize_t start;
    ssize_t end;
};

struct SliceBuilder {
    struct CSRSliceSerialize* serialize;
    struct FileWrapper data_file;
    struct FileWrapper indices_file;
    struct FileWrapper indptr_file;
    ssize_t nnz_tracker;
};

void reached_end_of_row(struct SliceBuilder* slices, ssize_t n_slices, ssize_t row, bool row_slice) {
    // reached end of row, update indptr for each slice with its current nnz
    for(ssize_t i = 0; i < n_slices; i++) {
        struct SliceBuilder* slc = &slices[i];
        ssize_t slc_nnz = slc->data_file.ptr;
        if(slc_nnz - slc->nnz_tracker > 0) {        
            // row not empty, track first + last occurrence of this for each slice
            slc->serialize->col_end = row;
            if(row_slice) {
                if(slc->serialize->col_start == -1)
                    slc->serialize->col_start = row;
            }
            // else{
            //     slc->serialize->col_start = 0;
            //     slc->serialize->col_end = 
            // }
        }
        slc->nnz_tracker = slc_nnz;
        // printf("(%ld, %ld): %ld %ld\n", row, i, slc->serialize->col_start, slc->serialize->col_end);
        file_append_single_element(&slc->indptr_file, &slc_nnz);
    }
}

ssize_t _concat_single_file(struct FileWrapper* dst, struct FileWrapper* src, char* buf, ssize_t buf_idx, ssize_t buf_sz) {
    // read src into memory, then write to dst using buf as in-memory cache
    ssize_t read_ptr = 0;
    ssize_t read_bytes = src->el_sz * src->ptr;
    CHECK(lseek(src->fid, 0, SEEK_SET) != -1);     // reset to start of src file

    if(DEBUG)
        printf("%s -> %s\n", src->fname, dst->fname);
    while(read_ptr < read_bytes) {
        // read
        ssize_t buf_remain = buf_sz - buf_idx - 1;              // # of bytes available in the buffer
        ssize_t bytes_in_file = read_bytes - read_ptr;      // # of bytes left in the file
        ssize_t bytes_to_read = buf_remain < bytes_in_file ? buf_remain : bytes_in_file;        // read min of bytes left in file + space left in buffer
        ssize_t bytes_read = read(src->fid, &buf[buf_idx], bytes_to_read);
        if(DEBUG)
            printf("\tread: %ld %ld %ld %ld %ld %ld\n", buf_remain, bytes_in_file, bytes_to_read, bytes_read, read_ptr, read_bytes);
        CHECK(bytes_read > 0)       
        read_ptr += bytes_read;
        buf_idx += bytes_read;
        CHECK(buf_idx > 0 && buf_idx < buf_sz)

        // write in-memory buffer to disk if less than 10% of the buffer is left
        if(buf_idx >= (buf_sz - (int)(buf_sz * 0.1))) {
            ssize_t write_ptr = 0;
            while(write_ptr < buf_idx) {
                if(DEBUG)
                    printf("\twrite: %ld %ld %ld\n", buf_sz, buf_idx, write_ptr);
                ssize_t amt = write(dst->fid, &buf[write_ptr], buf_idx-write_ptr);
                CHECK(amt > 0);
                write_ptr += amt;
                CHECK(write_ptr > 0 && write_ptr <= buf_idx)
            }
            buf_idx = 0;
        }
    }
    return buf_idx;
}

void concatenate_results(struct SliceBuilder* slc, ssize_t buf_sz) {
    char* buf = (char*)malloc(buf_sz);
    CHECK(buf != NULL);
    ssize_t buf_idx = 0;
    buf_idx = _concat_single_file(&slc->data_file, &slc->indices_file, buf, buf_idx, buf_sz);
    CHECK(buf_idx >= 0 && buf_idx < buf_sz)
    buf_idx = _concat_single_file(&slc->data_file, &slc->indptr_file, buf, buf_idx, buf_sz);
    CHECK(buf_idx >= 0 && buf_idx < buf_sz)

    // write entire buffer to disk
    if(DEBUG)
        printf("flush buffer to disk:\n");
    if(buf_idx > 0) {
        ssize_t write_ptr = 0;
        while(write_ptr < buf_idx) {
            if(DEBUG)
                printf("\twrite: %ld %ld %ld\n", buf_sz, buf_idx, write_ptr);
            ssize_t amt = write(slc->data_file.fid, &buf[write_ptr], buf_idx-write_ptr);
            CHECK(amt > 0);
            write_ptr += amt;
            CHECK(write_ptr > 0 && write_ptr <= buf_idx)
        }
    }

}


void flush_slice_builders(struct SliceBuilder* slices, int64_t n_slices, ssize_t concat_buffer_len, bool row_slice);
void csr_col_slice(struct CSRSliceSerialize64* csr_serialize, struct CSRSliceSerialize** results, ssize_t n_slices, ssize_t slice_sz, ssize_t buf_sz, bool row_slice) {
    CHECK(n_slices > 0 && slice_sz > 0 && buf_sz > 0)
    
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    // Initialization
    // struct SliceBuilder slices[n_slices];
    struct SliceBuilder* slices = (struct SliceBuilder*) malloc(sizeof(struct SliceBuilder) * n_slices);
    CHECK(slices);
    for(ssize_t i = 0; i < n_slices; i++) {
        struct SliceBuilder* slc = &slices[i];
        slc->serialize = results[i];
        if(row_slice) {
            slc->serialize->col_start = -1;
            slc->serialize->col_end = -1;
        }
        slc->nnz_tracker = 0;
        init_file_wrapper(slc->serialize->data_fname, &slc->data_file, sizeof(float), buf_sz, -1, -1);
        init_file_wrapper(slc->serialize->indices_fname, &slc->indices_file, sizeof(int), buf_sz, -1, -1);
        init_file_wrapper(slc->serialize->indptr_fname, &slc->indptr_file, sizeof(int), buf_sz, -1, -1);

        // indptr arrays start with zero
        int zero = 0;
        file_append_single_element(&slc->indptr_file, &zero);
    }

    // manage matrix to process (note CSRSerialize to CSRSliceSerialize conversion, actually all on a single file)
    struct SliceBuilder csr = {.serialize=NULL};
    printf("%ld %ld\n", csr_serialize->nnz, csr_serialize->nrow);
    CHECK(csr_serialize->nnz >= 0 && csr_serialize->nrow >= 0);
    init_file_wrapper(csr_serialize->data_fname, &csr.data_file, sizeof(float), buf_sz, -1, -1);
    init_file_wrapper(csr_serialize->indices_fname, &csr.indices_file, sizeof(int64_t), buf_sz, -1, -1);
    init_file_wrapper(csr_serialize->indptr_fname, &csr.indptr_file, sizeof(int64_t), buf_sz, -1, -1);
    file_load_read_buffer(&csr.data_file);
    file_load_read_buffer(&csr.indices_file);
    file_load_read_buffer(&csr.indptr_file);

    ssize_t next_indptr;
    file_read_single_element(&csr.indptr_file, &next_indptr);
    CHECK(next_indptr == 0);        // first indptr element must be zero
    file_read_single_element(&csr.indptr_file, &next_indptr);
    ssize_t row_cnt = 0;
    printf("Processing row:\n%ld / %ld", row_cnt, csr_serialize->nrow);

    // Iterate through all nonzero elements of the CSR matrix, copying them to the appropriate slice
    for(ssize_t idx = 0; idx < csr_serialize->nnz; idx++) {

        // reached start of next row, update indptr for all slices (maybe have multiple consecutive empty rows)
        while(idx == next_indptr) {
            reached_end_of_row(slices, n_slices, row_cnt, row_slice);
            file_read_single_element(&csr.indptr_file, &next_indptr);
            
            row_cnt++;
            printf("\33[2K\r%ld / %ld", row_cnt, csr_serialize->nrow);
            fflush(stdout);
        }

        // Get corresponding column slice
        int64_t el_col = 0;
        file_read_single_element(&csr.indices_file, &el_col);
        int64_t slc_idx = el_col / ((int64_t)slice_sz);
        if(slc_idx >= n_slices) {
            printf("ERROR: invalid column index %ld %ld %ld %ld\n", slc_idx, el_col, slice_sz, n_slices);
            CHECK(slc_idx < n_slices);
        }
        struct SliceBuilder* slc = &slices[slc_idx];

        // Copy data value to slice
        float el_data = 0;
        file_read_single_element(&csr.data_file, &el_data);        
        file_append_single_element(&slc->data_file, &el_data);
        
        // Store column to the slice (offsetting into the column)
        int int32_col = el_col - (int64_t)(slc_idx * slice_sz);
        file_append_single_element(&slc->indices_file, &int32_col);
    
        CHECK(csr.data_file.ptr == csr.indices_file.ptr);
    }

    // may have multiple empty rows at the end, populate indptr
    while(row_cnt < csr_serialize->nrow) {
        reached_end_of_row(slices, n_slices, row_cnt, row_slice);
        row_cnt++;
        printf("\33[2K\r%ld / %ld", row_cnt, csr_serialize->nrow);
        fflush(stdout);
    }
    reached_end_of_row(slices, n_slices, row_cnt, row_slice);      // final indptr entry that marks end of last row
    printf("\n\tconverting slices to unified memory format...\n");
    
    // multithreaded flushing of slices
    ssize_t concat_buffer_len = sizeof(float) * pow(2, 20) * 15;        // large buffer to handle concatenation of files
    flush_slice_builders(slices, n_slices, concat_buffer_len, row_slice);
    free(slices);
    
    // get duration to millisecond precision
    gettimeofday(&stop, NULL);
    double start_ms = (((double)start.tv_sec)*1000)+(((double)start.tv_usec)/1000);
    double stop_ms = (((double)stop.tv_sec)*1000)+(((double)stop.tv_usec)/1000);
    double diff_in_sec = (stop_ms - start_ms)/1000;
    printf("\tGenerated %ld column slices from (%ld, %ld) in: %f sec\n", n_slices, csr_serialize->nrow, csr_serialize->ncol, diff_in_sec);

}



void csr_col_slice_param(float* data, int64_t* indices, int64_t* indptr, ssize_t nnz, ssize_t nrow, ssize_t ncol, struct CSRSliceSerialize** results, ssize_t n_slices, ssize_t slice_sz, ssize_t buf_sz) {
    CHECK(n_slices > 0 && slice_sz > 0 && buf_sz > 0)
    
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    // Initialization
    struct SliceBuilder slices[n_slices];
    for(ssize_t i = 0; i < n_slices; i++) {
        struct SliceBuilder* slc = &slices[i];
        slc->serialize = results[i];
        slc->serialize->col_start = -1;
        slc->serialize->col_end = -1;
        slc->nnz_tracker = 0;
        init_file_wrapper(slc->serialize->data_fname, &slc->data_file, sizeof(float), buf_sz, -1, -1);
        init_file_wrapper(slc->serialize->indices_fname, &slc->indices_file, sizeof(int), buf_sz, -1, -1);
        init_file_wrapper(slc->serialize->indptr_fname, &slc->indptr_file, sizeof(int), buf_sz, -1, -1);

        // indptr arrays start with zero
        int zero = 0;
        file_append_single_element(&slc->indptr_file, &zero);
    }

    // file_read_single_element(&csr.indptr_file, &next_indptr);
    CHECK(indptr[0] == 0);        // first indptr element must be zero
    ssize_t indptr_idx = 1;
    ssize_t row_cnt = 0;
    if(DEBUG)
        printf("Processing row:\n%ld / %ld", row_cnt, nrow);

    // Iterate through all nonzero elements of the CSR matrix, copying them to the appropriate slice
    for(ssize_t idx = 0; idx < nnz; idx++) {

        // reached start of next row, update indptr for all slices (maybe have multiple consecutive empty rows)
        while(idx == indptr[indptr_idx]) {
            reached_end_of_row(slices, n_slices, row_cnt, false);
            indptr_idx++;
            
            row_cnt++;
            if(DEBUG)
                printf("\33[2K\r%ld / %ld", row_cnt, nrow);
            fflush(stdout);
        }

        // Get corresponding column slice
        int64_t slc_idx = indices[idx] / ((int64_t)slice_sz);
        CHECK(slc_idx < n_slices);
        struct SliceBuilder* slc = &slices[slc_idx];

        // Copy data value to slice
        file_append_single_element(&slc->data_file, &data[idx]);
        
        // Store column to the slice (offsetting into the column)
        int int32_col = indices[idx] - (int64_t)(slc_idx * slice_sz);
        file_append_single_element(&slc->indices_file, &int32_col);    
    }

    // may have multiple empty rows at the end, populate indptr
    while(row_cnt < nrow) {
        reached_end_of_row(slices, n_slices, row_cnt, false);
        row_cnt++;
        if(DEBUG)
            printf("\33[2K\r%ld / %ld", row_cnt, nrow);
        fflush(stdout);
    }
    reached_end_of_row(slices, n_slices, row_cnt, false);      // final indptr entry that marks end of last row
    if(DEBUG)
        printf("\n\tconverting slices to unified memory format...\n");


    // multithreaded flushing of slices
    ssize_t concat_buffer_len = sizeof(float) * pow(2, 20) * 10;        // large buffer to handle concatenation of files
    // flush_slice_builders(slices, n_slices, concat_buffer_len, false);



    for(ssize_t i = 0; i < n_slices; i++) {
        // flush/store results
        struct SliceBuilder* slc = &slices[i];
        slc->serialize->nnz = slc->data_file.ptr;     // copy over nnz for the slice
        if(DEBUG)
            printf("slc %ld: %ld\n", i, slc->serialize->nnz);
        file_flush_write_buffer(&slc->data_file);
        CHECK(slc->data_file.buf_idx == 0)
        file_flush_write_buffer(&slc->indices_file);
        CHECK(slc->indices_file.buf_idx == 0)
        file_flush_write_buffer(&slc->indptr_file);
        CHECK(slc->indptr_file.buf_idx == 0)

        // concatenate indices + indptr to end of data file to create the expected "unified" memory format
        concatenate_results(slc, concat_buffer_len);

        // Clean up
        CHECK(close(slc->data_file.fid) != -1)
        free(slc->data_file.buf);
        CHECK(close(slc->indices_file.fid) != -1)
        free(slc->indices_file.buf);
        CHECK(close(slc->indptr_file.fid) != -1)
        free(slc->indptr_file.buf);
    }
    
    // get duration to millisecond precision
    gettimeofday(&stop, NULL);
    double start_ms = (((double)start.tv_sec)*1000)+(((double)start.tv_usec)/1000);
    double stop_ms = (((double)stop.tv_sec)*1000)+(((double)stop.tv_usec)/1000);
    double diff_in_sec = (stop_ms - start_ms)/1000;
    if(DEBUG)
        printf("\tGenerated %ld column slices from (%ld, %ld) in: %f sec\n", n_slices, nrow, ncol, diff_in_sec);

}







struct FlusherArgs {
    struct SliceBuilder* slices;
    ssize_t concat_buffer_len;
    bool row_slice;
    int64_t n_slices;
    int64_t worker_idx;
    int64_t nworker;
};

void* _flush_worker_func(void* a) {
    // evenly (statically) divide work amongst workers
    struct FlusherArgs* args = (struct FlusherArgs*)a;
    int64_t step = args->n_slices / args->nworker;
    if(step == 0) {
        printf("ERROR: number of workers must be less than or equal to number of slices. %ld %ld\n", args->n_slices, args->nworker);
        CHECK(false)
    }
    if(step * args->nworker < args->n_slices) {
        if(DEBUG)
            printf("worker %ld: increasing step %ld by 1 (%ld %ld)\n", args->worker_idx, step, args->n_slices, args->nworker);
        step += 1;
    }
    int64_t start = step * args->worker_idx;
    int64_t end = start + step < args->n_slices ? start + step : args->n_slices;
    if(DEBUG)
        printf("worker (%ld): %ld %ld %ld %ld\n", args->worker_idx, start, end, step, args->n_slices);

    for(int64_t idx = start; idx < end; idx++) {
        struct SliceBuilder* slc = &args->slices[idx];
        slc->serialize->nnz = slc->data_file.ptr;     // copy over nnz for the slice
        if(DEBUG)
            printf("slc %ld: %ld\n", idx, slc->serialize->nnz);
        file_flush_write_buffer(&slc->data_file);
        CHECK(slc->data_file.buf_idx == 0)
        file_flush_write_buffer(&slc->indices_file);
        CHECK(slc->indices_file.buf_idx == 0)
        file_flush_write_buffer(&slc->indptr_file);
        CHECK(slc->indptr_file.buf_idx == 0)

        // adjust indptr to remove first col_start rows. effectively row slicing
        if(args->row_slice) {
            if(slc->serialize->col_start >= 0) {
                CHECK(slc->serialize->col_end > 0)

                // printf("(%ld) final: %ld %ld\n", i, slc->serialize->col_start, slc->serialize->col_end);
                ssize_t indptr_bytes = sizeof(int) * slc->indptr_file.ptr;
                int* indptr = (int*)malloc(indptr_bytes);
                CHECK(indptr)
                CHECK(lseek(slc->indptr_file.fid, 0, SEEK_SET) != -1)
                read_file(slc->indptr_file.fid, indptr_bytes, (char*)indptr);
                CHECK(lseek(slc->indptr_file.fid, 0, SEEK_SET) != -1)
                CHECK(indptr[slc->serialize->col_start] == 0)
                slc->indptr_file.ptr -= slc->serialize->col_start;
                write_file(slc->indptr_file.fid, slc->indptr_file.ptr * sizeof(int), (char*)&indptr[slc->serialize->col_start]);
                free(indptr);
            }
            else {
                printf("\n\nWARNING: col_start never set??? (%ld, %ld, %ld)\n\n", idx, slc->serialize->col_start, slc->serialize->col_end);
                // col_start = 0;
                // CHECK(col_end < 0)
                // col_end = 0;
                CHECK(false);
            }
        }

        // concatenate indices + indptr to end of data file to create the expected "unified" memory format
        concatenate_results(slc, args->concat_buffer_len);

        // Clean up
        CHECK(close(slc->data_file.fid) != -1)
        free(slc->data_file.buf);
        CHECK(close(slc->indices_file.fid) != -1)
        free(slc->indices_file.buf);
        CHECK(close(slc->indptr_file.fid) != -1)
        free(slc->indptr_file.buf);
    }
}

void flush_slice_builders(struct SliceBuilder* slices, int64_t n_slices, ssize_t concat_buffer_len, bool row_slice) {
    int64_t max_worker = 6;
    int64_t n_flusher_thr = n_slices < max_worker ? n_slices : max_worker;      // if fewer slices than workers, only use # slices workers
    pthread_t flusher_thr[n_flusher_thr];
    struct FlusherArgs fa[n_flusher_thr];
    for(int64_t worker_idx = 0; worker_idx < n_flusher_thr; worker_idx++) {
        struct FlusherArgs* f = &fa[worker_idx];
        f->slices = slices;
        f->row_slice = row_slice;
        f->concat_buffer_len = concat_buffer_len;
        f->n_slices = n_slices;
        f->worker_idx = worker_idx;
        f->nworker = n_flusher_thr;
        CHECK(pthread_create(&flusher_thr[worker_idx], NULL, _flush_worker_func, &fa[worker_idx]) == 0);
    }

    for(int64_t worker_idx = 0; worker_idx < n_flusher_thr; worker_idx++) {
        CHECK(pthread_join(flusher_thr[worker_idx], NULL) == 0);
    }
}




