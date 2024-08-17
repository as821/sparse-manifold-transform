
#include "file_wrapper.h"


void init_file_wrapper(char* fname, struct FileWrapper* file, ssize_t el_sz, ssize_t max_buf, ssize_t file_offset_bytes, ssize_t sz) {
    if(access(fname, F_OK) == 0)
        file->fid = open(fname, O_RDWR);
    else
        file->fid = open(fname, O_RDWR | O_CREAT | O_TRUNC, 0777);      // create file if DNE
    CHECK(file->fid != -1)

    if(file_offset_bytes > 0)
        CHECK(lseek(file->fid, file_offset_bytes, SEEK_SET) != -1);

    file->ptr = 0;
    file->el_sz = el_sz;
    file->buf_idx = 0;
    file->fname = fname;
    file->file_offset_bytes = file_offset_bytes;
    file->sz = sz;

    if(max_buf > 0) {
        file->buf = malloc(max_buf * el_sz);
        file->max_buf = max_buf;
        CHECK(file->buf);
    }
    else {
        file->buf = NULL;
        file->max_buf = 0;
    }
}

void init_file_wrapper_append(char* fname, struct FileWrapper* file, ssize_t max_buf) {
    int fd = open(fname, O_RDWR | O_APPEND);
    CHECK(fd != -1)
    init_file_wrapper_append_fd(fd, file, max_buf);
}


void init_file_wrapper_append_fd(int fd, struct FileWrapper* file, ssize_t max_buf) {
    file->fid = fd;
    CHECK(file->fid != -1)

    file->buf_idx = 0;
    file->fname = "(undefined fname)";
    CHECK(max_buf > 0)

    // make sure max_buf support 64 bit integer alignment if needed
    if(max_buf % sizeof(int64_t) != 0) {
        int64_t diff = max_buf % sizeof(int64_t);
        max_buf += (sizeof(int64_t) - diff);        // if not aligned, add until buf size is a multiple of integer size
    }
    CHECK(max_buf % sizeof(int64_t) == 0)

    file->buf = direct_io_malloc(max_buf);
    CHECK(file->buf);
    file->max_buf = max_buf;
    file->el_sz = 1;        // just to make file_flush_write_buffer work properly

    // unused for this kind of buffering. probably clean all this up!
    file->ptr = -1;
    file->sz = -1;
    file->file_offset_bytes = -1;
}




void file_flush_write_buffer(struct FileWrapper* file) {
    if(file->buf_idx > 0) {
        ssize_t bytes = file->el_sz * file->buf_idx;
        write_file(file->fid, bytes, (char*)file->buf);
        file->buf_idx = 0;
    }
}


void file_load_read_buffer(struct FileWrapper* file) {
    // If size is set, check that we will not read past the end of the section. Otherwise, rely on EOF to avoid overruns
    ssize_t n_el_to_read;
    if(file->sz > 0) {
        ssize_t diff = file->sz - file->ptr;
        n_el_to_read = file->max_buf < diff ? file->max_buf : diff;
    }
    else
        n_el_to_read = file->max_buf;

    // load the next max_buf*el_sz bytes from the file
    // NOTE: the file might be smaller than the cache, so allow reads of less than max_buf. however, need to resize buffer to avoid reading past EOF
    ssize_t bytes = n_el_to_read * file->el_sz;
    ssize_t res = read(file->fid, file->buf, bytes);
    CHECK(res > 0 && res <= bytes);
    if(res < bytes) {
        // is possible that this happens midway through the file due to some disk issue + then we use a smaller 
        // read buffer for the rest of the file, but that seems unlikely so this is ok for now...
        CHECK(res % file->el_sz == 0)
        printf("NOTE: downsizing read buffer from %ld to %ld (%ld) for %s\n", bytes, res, res / file->el_sz, file->fname);
        file->max_buf = res;
    }
    file->buf_idx = 0;
}


void file_append_single_element(struct FileWrapper* file, void* element) {
    // append a single element to the given file (apply in-memory buffering where possible)
    CHECK(file->el_sz > 0)      // not opened with "append" setup
    if(file->max_buf > 0) {
        if(file->buf_idx == file->max_buf - 1)
            file_flush_write_buffer(file);
        
        if(file->el_sz == sizeof(float)) 
            ((float*)file->buf)[file->buf_idx] = *((float*)element);
        else if(file->el_sz == sizeof(int)) 
            ((int*)file->buf)[file->buf_idx] = *((int*)element);
        else 
            CHECK(false);
        file->buf_idx++;
    }
    else
        CHECK(write(file->fid, element, file->el_sz) == file->el_sz)
    file->ptr += 1;
}

void file_read_single_element(struct FileWrapper* file, void* result) {
    // read a single element from the given file (+ in-memory caching)
    CHECK(result);
    if(file->max_buf > 0) {
        if(file->buf_idx == file->max_buf) 
            file_load_read_buffer(file);
        
        if(file->el_sz == sizeof(float)) 
            *((float*) result) = ((float*)file->buf)[file->buf_idx];
        else if(file->el_sz == sizeof(int)) 
            *((int*)result) = ((int*)file->buf)[file->buf_idx];
        else if(file->el_sz == sizeof(int64_t)) 
            *((int64_t*)result) = ((int64_t*)file->buf)[file->buf_idx];
        else 
            CHECK(false);
        file->buf_idx++;
    }
    else {
        if(file->sz > 0)
            // check that section will not be overrun by reading this element
            CHECK(file->sz - (file->ptr + 1) >= 0)
        int res = read(file->fid, result, file->el_sz);
        CHECK(res == file->el_sz)
    }
    file->ptr += 1;
}


void file_wrapper_reset(struct FileWrapper* file, ssize_t max_buf_sz) {
    // reset file descriptor to start of file + reset buffer
    CHECK(lseek(file->fid, 0, SEEK_SET) != -1);
    file->ptr = 0;
    file->buf_idx = 0;
    file->file_offset_bytes = 0;
    file->max_buf = max_buf_sz;
}


void file_append_direct_edit(struct FileWrapper* file, int* arr, int64_t sz, int64_t offset) {
    CHECK(file->max_buf > 0 && file->ptr < 0)       // opened in append mode
    
    // assumes 64 bit integer alignment
    CHECK(file->max_buf % sizeof(int64_t) == 0)
    CHECK(file->buf_idx % sizeof(int64_t) == 0)

    int64_t arr_ptr = 0;
    while(arr_ptr < sz) {
        CHECK(file->buf_idx % sizeof(int64_t) == 0)
        int64_t buf_start_idx = file->buf_idx / sizeof(int64_t);

        int64_t buf_remain = (file->max_buf - file->buf_idx) / sizeof(int64_t);
        int64_t arr_remain = sz - arr_ptr;
        int64_t inner_end = buf_remain < arr_remain ? buf_remain : arr_remain;
        for(int64_t idx = 0; idx < inner_end; idx++) {
            ((int64_t*)file->buf)[buf_start_idx + idx] = (int64_t)(arr[idx]) + offset; 
        }

        arr_ptr += inner_end;
        file->buf_idx += inner_end * sizeof(int64_t);
        CHECK(file->buf_idx <= file->max_buf)
        if(file->buf_idx == file->max_buf)
            file_flush_write_buffer(file);
    }
}



void file_append_buffer(struct FileWrapper* file, void* buf, int64_t n_bytes) {
    CHECK(file->max_buf > 0 && file->ptr < 0)       // opened in append mode
    if(file->buf_idx + n_bytes >= file->max_buf)
        file_flush_write_buffer(file);

    if(n_bytes >= file->max_buf) {
        // buffer too small, write straight to filesystem
        printf("Skipping buffer for file (%s). %ld vs max %ld", file->fname, n_bytes, file->max_buf);
        write_file(file->fid, n_bytes, (char*)buf);
    }

    memcpy(&file->buf[file->buf_idx], buf, n_bytes);
    file->buf_idx += n_bytes;
}

void file_append_from_file(struct FileWrapper* file, int fd, int64_t n_bytes) {
    CHECK(file->max_buf > 0 && file->ptr < 0)       // opened in append mode
    if(file->buf_idx + n_bytes >= file->max_buf) {
        file_flush_write_buffer(file);
    }

    if(n_bytes >= file->max_buf) {
        // could implement this by reading in source file + writing to destination file in max_buf chunks
        printf("Error: file_append_from_file failing due to insufficient buffer for src file to dest file transfer (%s). %ld vs max %ld. Increase buffer size!", file->fname, n_bytes, file->max_buf);
        CHECK(false);
    }

    read_file(fd, n_bytes, (char*)(&file->buf[file->buf_idx]));
    file->buf_idx += n_bytes;
}

void* file_wrapper_cleanup(void* a) {
    struct FileWrapper* fw = (struct FileWrapper*)a;
    file_flush_write_buffer(fw);
    // CHECK(fsync(fw->fid) != -1);
    CHECK(close(fw->fid) != -1)
    free(fw->buf);
    return NULL;
}


