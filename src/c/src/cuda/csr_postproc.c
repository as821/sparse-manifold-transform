#include "cuda_util.h"
#include "file_wrapper.h"
#include "sync.h"

#include <sys/time.h>
#include <pthread.h>






void csr_postproc(char** input_files, char* out_file, int64_t n_inp, int64_t inp_nrow, int64_t* inp_ncol) {
    // TODO(as) iterate through all the input files, collecting entries for one row at a time + writing it to the output file
    // NOTE: assumes entire contents of one row of the output matrix can fit into RAM (not a hard constraint, can be avoided by using a tmp file...)


    // open all input files
    int64_t total_col = 0;
    int fd_buf[n_inp];
    for(int64_t idx = 0; idx < n_inp; idx++) {
        total_col += inp_ncol[idx];
        fd_buf[idx] = open(input_files[idx], O_RDONLY);
        CHECK(fd_buf[idx] != -1);
    }

    float* row_buf = (float*) malloc(sizeof(float)*total_col);
    CHECK(row_buf);
    
    // open output file
    int out_fd = open(out_file, O_RDWR | O_APPEND);
    CHECK(out_fd != -1);

    for(int64_t rdx = 0; rdx < inp_nrow; rdx++) {
        // collect rows from input file into row buffer
        int64_t row_ptr = 0;
        for(int64_t idx = 0; idx < n_inp; idx++) {
            read_file(fd_buf[idx], inp_ncol[idx] * sizeof(float), (char*)&row_buf[row_ptr]);
            row_ptr += inp_ncol[idx];
        }
        CHECK(row_ptr == total_col);

        // dump row to output file
        write_file(out_fd, sizeof(float)*total_col, (char*)row_buf);
    }

    // close output file + all input files
    CHECK(close(out_fd) != -1);
    for(int idx = 0; idx < n_inp; idx++) {
        CHECK(close(fd_buf[idx]) != -1);
    }
}


