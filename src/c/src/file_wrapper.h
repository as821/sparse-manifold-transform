#pragma once

#include "util.h"

struct FileWrapper {
    int fid;
    ssize_t ptr;
    ssize_t el_sz;
    char* fname;
    ssize_t file_offset_bytes;              // only valid when reading
    ssize_t sz;                             // only valid when reading

    char* buf;
    ssize_t buf_idx;
    ssize_t max_buf;
};

void init_file_wrapper(char* fname, struct FileWrapper* file, ssize_t el_sz, ssize_t max_buf, ssize_t file_offset_bytes, ssize_t sz);
void file_flush_write_buffer(struct FileWrapper* file);
void file_load_read_buffer(struct FileWrapper* file);
void file_append_single_element(struct FileWrapper* file, void* element);
void file_read_single_element(struct FileWrapper* file, void* result);
void file_wrapper_reset(struct FileWrapper* file, ssize_t max_buf_sz);

void file_append_buffer(struct FileWrapper* file, void* buf, int64_t n_bytes);
void init_file_wrapper_append(char* fname, struct FileWrapper* file, ssize_t max_buf);
void file_append_from_file(struct FileWrapper* file, int fd, int64_t n_bytes);
void init_file_wrapper_append_fd(int fd, struct FileWrapper* file, ssize_t max_buf);
void* file_wrapper_cleanup(void* a);
void file_append_direct_edit(struct FileWrapper* file, int* arr, int64_t sz, int64_t offset);
