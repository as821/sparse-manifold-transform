#pragma once

 #include "cuda_util.h"

#include <pthread.h>

struct PinnedMem {
    pthread_mutex_t mutex;              // protext in_use
    bool in_use;
    void* ptr;
    ssize_t n_bytes;
};


// book-keeping for a circular buffer that DOES NOT implement buffering until a slot in the buffer becomes available
struct PinnedMemBuffer {
    struct PinnedMem* mem;
    pthread_mutex_t mutex;              // protext next_idx
    int next_idx;                       // index of the next buffer section to use
    int n_buf;
};

void init_pinned_mem_buf(struct PinnedMemBuffer* buf, ssize_t alloc_sz);
struct PinnedMem* get_next_pinned_mem(struct PinnedMemBuffer* buf);
void done_with_pinned_mem(struct PinnedMem* mem);
void done_with_pinned_mem_ptr(void* ptr, struct PinnedMemBuffer* buf);
void cleanup_pinned_mem_buf(struct PinnedMemBuffer* buf);



