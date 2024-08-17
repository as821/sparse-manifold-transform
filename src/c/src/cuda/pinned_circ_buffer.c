
#include "pinned_circ_buffer.h"

void init_pinned_mem_buf(struct PinnedMemBuffer* buf, ssize_t alloc_sz) {
    // initialize + allocate buffer memory
    CHECK(pthread_mutex_init(&buf->mutex, NULL) == 0);
    buf->next_idx = 0;
    buf->mem = (struct PinnedMem*)malloc(sizeof(struct PinnedMem) * buf->n_buf);
    CHECK(buf->mem);
    for(int idx = 0; idx < buf->n_buf; idx++) {
        struct PinnedMem* m = &buf->mem[idx];
        CHECK(pthread_mutex_init(&m->mutex, NULL) == 0);
        m->in_use = false;
        m->n_bytes = alloc_sz;
        CHECK_CUDA_NORET(cudaMallocHost((void**)&m->ptr, m->n_bytes))
        CHECK(m->ptr);
    }
}

void cleanup_pinned_mem_buf(struct PinnedMemBuffer* buf) {
    CHECK(pthread_mutex_lock(&buf->mutex) == 0);
    for(int idx = 0; idx < buf->n_buf; idx++) {
        struct PinnedMem* m = &buf->mem[idx];
        CHECK(pthread_mutex_lock(&m->mutex) == 0);        
        CHECK(!m->in_use);
        CHECK_CUDA_NORET(cudaFreeHost(m->ptr))
        CHECK(pthread_mutex_unlock(&m->mutex) == 0);
    }
    free(buf->mem);
    CHECK(pthread_mutex_unlock(&buf->mutex) == 0);
}


struct PinnedMem* get_next_pinned_mem(struct PinnedMemBuffer* buf) {
    // get next buffer and do basic check that its not in use
    CHECK(pthread_mutex_lock(&buf->mutex) == 0);
    int idx = buf->next_idx;
    buf->next_idx++;
    buf->next_idx = buf->next_idx % buf->n_buf;
    CHECK(idx < buf->n_buf);
    struct PinnedMem* m = &buf->mem[idx];
    CHECK(pthread_mutex_lock(&m->mutex) == 0);
    CHECK(pthread_mutex_unlock(&buf->mutex) == 0);

    // printf("getting %d (%p)\n", idx, m->ptr);
    CHECK(!m->in_use);
    m->in_use = true;
    CHECK(pthread_mutex_unlock(&m->mutex) == 0);
    return m;
}


void done_with_pinned_mem(struct PinnedMem* mem) {
    CHECK(pthread_mutex_lock(&mem->mutex) == 0);
    mem->in_use = false;
    // printf("released %p\n", mem->ptr);
    CHECK(pthread_mutex_unlock(&mem->mutex) == 0);
}


void done_with_pinned_mem_ptr(void* ptr, struct PinnedMemBuffer* buf) {
    CHECK(pthread_mutex_lock(&buf->mutex) == 0);
    bool found = false;
    for(int idx = 0; idx < buf->n_buf; idx++) {
        struct PinnedMem* m = &buf->mem[idx];
        CHECK(pthread_mutex_lock(&m->mutex) == 0);
        if(m->ptr == ptr) {
            CHECK(m->in_use);
            m->in_use = false;
            // printf("released %p (ptr)\n", m->ptr);
            CHECK(pthread_mutex_unlock(&m->mutex) == 0);
            found = true;
            break;
        }
        CHECK(pthread_mutex_unlock(&m->mutex) == 0);
    }
    CHECK(found);
    CHECK(pthread_mutex_unlock(&buf->mutex) == 0);
}


