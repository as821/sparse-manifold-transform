
#include "sync.h"


void producer_wait(struct SyncArgs* args, int64_t max_preload) {
    CHECK(max_preload > 0);

    // wait until can process the current index (ensure do not preload too many slices)
    CHECK(pthread_mutex_lock(&args->mutex) == 0);
    while(args->consumer_ptr + max_preload < args->loader_ptr) {
        CHECK(pthread_cond_wait(&args->loader_wait, &args->mutex) == 0);
    }
    CHECK(pthread_mutex_unlock(&args->mutex) == 0);
}

void producer_signal(struct SyncArgs* args, int64_t idx_completed) {
    // indicate that this slice is ready to be consumed
    CHECK(pthread_mutex_lock(&args->mutex) == 0);

    // wait until the slice before this one has been "produced"
    // preserves invariant that a signal is sent only for a contiguous block of slices (will not signal for "n" before "n-1"). allows consumer logic to remain dead simple
    // (obviously inefficient but keeps implementation of multiple reader threads relatively straight-forward)
    CHECK(args->loader_ptr < idx_completed);
    while(args->loader_ptr < (idx_completed-1)) {
        CHECK(pthread_cond_wait(&args->consumer_wait, &args->mutex) == 0);
    }
    CHECK(args->loader_ptr == idx_completed-1);
    args->loader_ptr++;  
    CHECK(pthread_cond_broadcast(&args->consumer_wait) == 0);
    CHECK(pthread_mutex_unlock(&args->mutex) == 0);
}


void consumer_wait(struct SyncArgs* args) {
    // wait until this index has been processed by the dataloader thread
    // NOTE: this assumes a single consumer thread (releases lock without incrementing consumer ptr + broadcast is used, multiple consumers could consume same slice)
    CHECK(pthread_mutex_lock(&args->mutex) == 0);
    while(args->consumer_ptr == args->loader_ptr) {
        CHECK(args->consumer_ptr <= args->loader_ptr);      // consumer should never outpace the loader
        CHECK(pthread_cond_wait(&args->consumer_wait, &args->mutex) == 0);
    }
    CHECK(pthread_mutex_unlock(&args->mutex) == 0);
}

void consumer_signal(struct SyncArgs* args) {
    // signal that a slice has been "consumed" + loader can continue
    CHECK(pthread_mutex_lock(&args->mutex) == 0);
    args->consumer_ptr++;
    CHECK(pthread_cond_signal(&args->loader_wait) == 0);
    CHECK(pthread_mutex_unlock(&args->mutex) == 0);
}


int64_t loader_get_next_index(struct SyncArgs* args) {
    CHECK(pthread_mutex_lock(&args->mutex) == 0);
    int64_t out = args->worker_idx;
    args->worker_idx++;
    CHECK(pthread_mutex_unlock(&args->mutex) == 0);
    return out;
}


void sync_args_init(struct SyncArgs* args) {
    args->loader_ptr = -1;
    args->consumer_ptr = -1;
    args->worker_idx = 0;
    CHECK(pthread_mutex_init(&args->mutex, NULL) == 0);
    CHECK(pthread_cond_init(&args->loader_wait, NULL) == 0);
    CHECK(pthread_cond_init(&args->consumer_wait, NULL) == 0);
}


