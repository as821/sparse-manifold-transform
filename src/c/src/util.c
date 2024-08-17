#include "util.h"



void print_stack_trace() {
    void *array[100];
    size_t size;
    char **strings;
    size_t i;

    size = backtrace(array, 100);
    strings = backtrace_symbols(array, size);

    printf("Stack trace:\n");
    for (i = 0; i < size; i++) {
        printf("\t%s\n", strings[i]);
    }

    free(strings);
}


void read_file(int fd, ssize_t sz, char* buf) {
    ssize_t cnt = 0;
    while(cnt < sz) {
        ssize_t n = read(fd, &buf[cnt], sz-cnt);
        if(n == 0) {
            print_stack_trace();
            printf("ERROR: unexpectedly reach end of file after reading %ld / %ld bytes.\n", cnt, sz);
            exit(EXIT_FAILURE);
        }
        else {
            CHECK(n != -1)
            cnt += n;
        }
    }
}


void write_file(int fd, ssize_t sz, char* buf) {
    ssize_t cnt = 0;
    while(cnt < sz) {
        ssize_t n = write(fd, &buf[cnt], sz-cnt);
        if(n == 0) {
            print_stack_trace();
            printf("ERROR: unexpectedly reach end of file after reading %ld / %ld bytes.\n", cnt, sz);
            exit(EXIT_FAILURE);
        }
        else {
            CHECK(n != -1)
            cnt += n;
        }
    }
}



int64_t next_greatest_alignment(int64_t len) {
    if(len % O_DIRECT_ALIGN != 0)
        len = ((int64_t)(len / O_DIRECT_ALIGN) + 1) * O_DIRECT_ALIGN;
    return len;
}

void write_file_aligned(int fd, ssize_t sz, char* buf) {
    CHECK(sz % O_DIRECT_ALIGN == 0);
    ssize_t cnt = 0;
    while(cnt < sz) {
        CHECK((sz - cnt) % O_DIRECT_ALIGN == 0);     // fail here rather than in syscall, dont try to recover
        ssize_t n = write(fd, &buf[cnt], sz-cnt);
        if(n == 0) {
            printf("ERROR: unexpectedly reach end of file after reading %ld / %ld bytes.\n", cnt, sz);
            exit(EXIT_FAILURE);
        }
        else {
            CHECK(n != -1)
            cnt += n;
        }
    }
}

void* direct_io_malloc(int64_t len) {
    // Allocate memory suitable for using with O_DIRECT I/O (aligned to and an integer mulitple of O_DIRECT_ALIGN)
    // printf("Allocating %ld bytes.", len);
    len = next_greatest_alignment(len);
    // printf("--> promoted to %ld to match %d alignment requirement.\n", len, O_DIRECT_ALIGN);

    void* ptr = aligned_alloc(O_DIRECT_ALIGN, len);
    CHECK(ptr);
    // printf("\t%p\n", ptr);
    return ptr;
}



