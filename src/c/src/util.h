#pragma once


#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <execinfo.h>
#include <sys/types.h>
#include <stdint.h>
#include <stddef.h>

void print_stack_trace();
void read_file(int fd, ssize_t sz, char* buf);
void write_file(int fd, ssize_t sz, char* buf);

#define O_DIRECT_ALIGN  512         // obtained by running "sudo blockdev --getss <drive to use>"
void write_file_aligned(int fd, ssize_t sz, char* buf);
void* direct_io_malloc(int64_t len);
int64_t next_greatest_alignment(int64_t len);

#define CHECK(x)                                                                                    \
{                                                                                                   \
    if(!(x)) {                                                                                      \
        printf("ERROR (line %d, file:%s) (%d): %s\n", __LINE__, __FILE__, errno, strerror(errno));  \
        print_stack_trace();                                                                        \
        exit(EXIT_FAILURE);                                                                         \
    }                                                                                               \
}

// only define structs that are common across CUDA and "normal" C code
struct CSRSliceSerialize {
    char* data_fname;
    char* indices_fname;
    char* indptr_fname;
    ssize_t nnz;
    ssize_t nrow;
    ssize_t ncol;
    ssize_t start;
    ssize_t end;
    ssize_t col_start;
    ssize_t col_end;
};




